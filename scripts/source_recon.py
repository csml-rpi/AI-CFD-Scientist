#!/usr/bin/env python3
"""OpenFOAM source reconnaissance for code-mod tasks.

Two-phase anti-hallucination flow for `code_mod_prepare` → `code_mod_apply_compile`:

  1. Mechanical pre-filter of $FOAM_SRC (directories + .H files, depth-capped,
     noise stripped). Produces a short, topic-agnostic tree listing.

  2. LLM slate-search loop: maintains a constant working set of up to 10 files
     across ≤5 rounds (min 2). Each round the LLM returns a keep/drop/add diff
     over the current slate. Cycle detection bans files that thrash. Stops when
     the model flags confidence (`ready_to_code_mod: true`), the slate is
     unchanged round-to-round, or the hard cap is hit.

After the loop, extracts verified include directories and class signatures from
the final slate and writes `discovered_paths.json` + `discovered_paths.history.json`.

Both files are consumed by `code_mod_prepare.py` / `code_mod_apply_compile.py`
so the code-mod LLM sees only paths that actually exist on disk.

Usage:
    python scripts/source_recon.py \\
        --foam-src /mnt/sda1/openfoam10/src \\
        --topic "Improve droplet evaporation model" \\
        --mode custom_case_library \\
        --output runs/.../discovered_paths.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Mechanical pre-filter
# ---------------------------------------------------------------------------

_SKIP_DIR_NAMES = {
    "lnInclude", "Make", ".git", "doc", "tutorials", "applications",
    "test", "tests", "wmake", "etc", "bin",
}
_SKIP_DIR_SUFFIXES = ("Test", "Tests")
_KEEP_EXTENSIONS = {".H"}
_MAX_DEPTH = 10
_MAX_ENTRIES = 20000
_MAX_H_FILE_BYTES = 60_000  # per-file cap when actually read in-round


def _should_skip_dir(name: str) -> bool:
    if name in _SKIP_DIR_NAMES:
        return True
    if name.startswith("."):
        return True
    for suf in _SKIP_DIR_SUFFIXES:
        if name.endswith(suf):
            return True
    return False


def prefilter_tree(foam_src: Path) -> List[Dict[str, Any]]:
    """Walk FOAM_SRC, return pruned listing of directories + .H files."""
    foam_src = foam_src.resolve()
    entries: List[Dict[str, Any]] = []
    if not foam_src.is_dir():
        return entries

    stack: List[Tuple[Path, int]] = [(foam_src, 0)]
    while stack and len(entries) < _MAX_ENTRIES:
        root, depth = stack.pop()
        try:
            children = sorted(root.iterdir(), key=lambda p: p.name.lower())
        except (PermissionError, OSError):
            continue
        for p in children:
            if len(entries) >= _MAX_ENTRIES:
                break
            try:
                name = p.name
                rel = str(p.resolve().relative_to(foam_src))
            except ValueError:
                continue
            if p.is_dir():
                if _should_skip_dir(name):
                    continue
                entries.append({"kind": "dir", "rel": rel})
                if depth < _MAX_DEPTH:
                    stack.append((p, depth + 1))
            elif p.is_file():
                if p.suffix in _KEEP_EXTENSIONS:
                    try:
                        sz = p.stat().st_size
                    except OSError:
                        sz = 0
                    entries.append({"kind": "h", "rel": rel, "size": sz})
    return entries


def render_tree_listing(entries: List[Dict[str, Any]], max_chars: int = 120_000) -> str:
    """Compact one-entry-per-line rendering for LLM prompts."""
    lines: List[str] = []
    total = 0
    for e in entries:
        if e["kind"] == "dir":
            line = f"d  {e['rel']}/"
        else:
            line = f"h  {e['rel']}  ({e.get('size', 0)}B)"
        if total + len(line) + 1 > max_chars:
            lines.append("... [truncated]")
            break
        lines.append(line)
        total += len(line) + 1
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Slate-search data model
# ---------------------------------------------------------------------------

_WORKING_SET_SIZE = 10
_MAX_ROUNDS = 5
_MIN_ROUNDS = 2
_CYCLE_STRIKE_LIMIT = 2  # added/dropped this many times → banned
_MAX_TOTAL_READ_BYTES = 200_000  # total across the 10-file slate


@dataclass
class SlateRound:
    round_no: int
    slate: List[str]
    diff: Dict[str, Any]
    ready: bool
    reasoning: str


@dataclass
class SlateState:
    foam_src: Path
    slate: List[str] = field(default_factory=list)
    history: List[SlateRound] = field(default_factory=list)
    add_drop_counts: Dict[str, int] = field(default_factory=dict)
    banned: Set[str] = field(default_factory=set)


def _read_h_excerpt(path: Path, cap: int = _MAX_H_FILE_BYTES) -> str:
    try:
        data = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""
    if len(data) > cap:
        return data[:cap] + f"\n... [truncated at {cap} bytes]"
    return data


def _budget_slate_reads(foam_src: Path, slate: List[str], total_cap: int = _MAX_TOTAL_READ_BYTES) -> Dict[str, str]:
    """Read slate files, trimming large ones so total stays under cap."""
    out: Dict[str, str] = {}
    remaining = total_cap
    for rel in slate:
        p = (foam_src / rel).resolve()
        if not p.is_file():
            out[rel] = "[MISSING ON DISK]"
            continue
        cap = min(_MAX_H_FILE_BYTES, max(remaining // max(1, len(slate)), 4_000))
        txt = _read_h_excerpt(p, cap=cap)
        out[rel] = txt
        remaining -= len(txt)
        if remaining <= 0:
            break
    return out


# ---------------------------------------------------------------------------
# Prompting
# ---------------------------------------------------------------------------

_SYS_PROMPT = (
    "You are an OpenFOAM source-code navigator. Your job: find the exact header "
    "files, class definitions, and include directories in the OpenFOAM source "
    "tree that a separate code-generation step will need in order to write a "
    "compiling case-local custom library (wmake libso under case/customModels).\n\n"
    "You work in rounds, maintaining a working slate of UP TO 10 files. Each "
    "round you receive (a) the task, (b) a pre-filtered tree listing of "
    "$FOAM_SRC directories and .H files, (c) the current slate with file "
    "contents, and (d) the previous round's diff. You return a JSON object.\n\n"
    "Hard rules:\n"
    "- Every file you add MUST appear in the provided tree listing. Do NOT "
    "  invent paths. Do NOT use ESI / foam-extend layout assumptions.\n"
    "- Prefer base-class headers (e.g. the abstract base that declares virtual "
    "  methods the custom model must override) over every concrete descendant.\n"
    "- Include one Make/options for the relevant library subtree IF the listing "
    "  shows one; this gives the code-gen step a ground-truth include set.\n"
    "- When dropping a file, cite what in its contents told you it was irrelevant.\n"
    "- Set ready_to_code_mod=true only when the slate contains: the parent "
    "  class header, any run-time-selection registration macro header, and "
    "  enough context to write compatible method signatures.\n\n"
    "Output strict JSON, no prose outside JSON:\n"
    "{\n"
    '  "slate": ["rel/path/to/file1.H", ... up to 10 entries],\n'
    '  "drop_reasons": {"rel/path": "why dropped"},\n'
    '  "add_reasons": {"rel/path": "why added"},\n'
    '  "ready_to_code_mod": false,\n'
    '  "reasoning_brief": "≤400 chars summary of this round\'s decision"\n'
    "}\n"
)


def _build_round_prompt(
    task: Dict[str, Any],
    tree_listing: str,
    state: SlateState,
    slate_contents: Dict[str, str],
    round_no: int,
    rejected_last_round: Optional[List[str]] = None,
) -> str:
    prev_diff: Dict[str, Any] = {}
    if state.history:
        prev_diff = state.history[-1].diff
    slate_block = "\n\n".join(
        f"=== {rel} ===\n{slate_contents.get(rel, '[not read]')}"
        for rel in state.slate
    ) or "(slate is empty — first round, add up to 10 files)"

    banned_block = (", ".join(sorted(state.banned)) or "(none)")

    # If the previous round proposed paths that ALL failed tree validation,
    # feed that back loudly so the LLM doesn't repeat the same hallucinated set.
    rejection_block = ""
    if rejected_last_round:
        rejection_block = (
            "\n⚠️ PREVIOUS ROUND REJECTION: Every path you proposed last round was "
            "NOT FOUND in the tree listing and was dropped. This usually means you are "
            "pattern-matching on a different OpenFOAM layout (ESI / foam-extend / "
            "OpenFOAM-v24xx) instead of the installation actually present here. "
            "DO NOT propose the same paths again. Instead, this round:\n"
            "  1. Scan the tree listing carefully for the TOP-LEVEL directories that "
            "appear — only those exist.\n"
            "  2. Pick files whose relative paths are CHARACTER-IDENTICAL to entries "
            "in the tree listing. Copy-paste them verbatim. Do not guess.\n"
            "  3. If the concept you want (e.g. a base class) isn't at the path you "
            "expected, search the listing for keyword substrings and pick whatever "
            "IS present, even if the name differs from your expectation.\n"
            f"Paths you proposed that DID NOT EXIST:\n{json.dumps(rejected_last_round[:20], indent=2)}\n"
        )

    return (
        f"Task:\n{json.dumps(task, indent=2)[:4000]}\n\n"
        f"Pre-filtered tree listing of $FOAM_SRC ({state.foam_src}):\n"
        f"```\n{tree_listing}\n```\n\n"
        f"Round {round_no}/{_MAX_ROUNDS} (min {_MIN_ROUNDS}).\n"
        f"Current slate ({len(state.slate)}/{_WORKING_SET_SIZE}) with contents:\n"
        f"{slate_block}\n\n"
        f"Previous round diff:\n{json.dumps(prev_diff, indent=2)[:2000]}\n\n"
        f"Banned (cycled too often): {banned_block}\n"
        f"{rejection_block}\n"
        "Return the JSON object per the system-prompt contract."
    )


# ---------------------------------------------------------------------------
# LLM driver
# ---------------------------------------------------------------------------

def _bootstrap_paths(repo_root: Path) -> None:
    foam_src = repo_root / "Foam-Agent" / "src"
    lang_src = repo_root / "src"
    if str(foam_src) not in sys.path:
        sys.path.insert(0, str(foam_src))
    if str(lang_src) not in sys.path:
        sys.path.insert(0, str(lang_src))


def _call_llm(messages: List[Any]) -> str:
    from langchain_core.messages import HumanMessage, SystemMessage  # noqa: F401
    from cfd_langgraph.config import get_settings
    from cfd_langgraph.llm.factory import create_langchain_llm

    settings = get_settings()
    llm = create_langchain_llm(model=settings.model, temperature=0.0, effort="low")
    resp = llm.invoke(messages)
    return str(getattr(resp, "content", "") or "")


def _parse_round_response(raw: str) -> Dict[str, Any]:
    from cfd_langgraph.utils import strip_json_fences  # type: ignore
    txt = strip_json_fences(raw).strip()
    s, e = txt.find("{"), txt.rfind("}")
    if s == -1 or e == -1 or e <= s:
        raise ValueError(f"No JSON object in LLM response: {raw[:200]!r}")
    return json.loads(txt[s : e + 1])


# ---------------------------------------------------------------------------
# Core loop
# ---------------------------------------------------------------------------

def _validate_slate_against_tree(
    proposed: List[str], tree_paths: Set[str], banned: Set[str]
) -> List[str]:
    """Keep only entries that exist in the tree listing and aren't banned."""
    out: List[str] = []
    seen: Set[str] = set()
    for rel in proposed:
        if not isinstance(rel, str) or not rel.strip():
            continue
        rel = rel.strip().lstrip("./").replace("\\", "/")
        if rel in seen or rel in banned:
            continue
        if rel not in tree_paths:
            continue
        out.append(rel)
        seen.add(rel)
        if len(out) >= _WORKING_SET_SIZE:
            break
    return out


def _update_cycle_strikes(prev: List[str], new: List[str], state: SlateState) -> None:
    prev_set, new_set = set(prev), set(new)
    for rel in prev_set - new_set:  # dropped this round
        state.add_drop_counts[rel] = state.add_drop_counts.get(rel, 0) + 1
    for rel in new_set - prev_set:  # re-added this round
        # count only re-adds (strike if it was previously dropped)
        if rel in state.add_drop_counts and state.add_drop_counts[rel] > 0:
            state.add_drop_counts[rel] += 1
            if state.add_drop_counts[rel] >= _CYCLE_STRIKE_LIMIT * 2:
                state.banned.add(rel)


def run_slate_search(
    *,
    foam_src: Path,
    task: Dict[str, Any],
    cache_path: Path,
    history_path: Optional[Path] = None,
    allow_cache_hit: bool = True,
) -> Dict[str, Any]:
    """Main entry. Returns discovered_paths.json contents."""
    foam_src = foam_src.resolve()
    if allow_cache_hit and cache_path.is_file():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if isinstance(cached, dict) and cached.get("selected_files"):
                cached["cache_hit"] = True
                return cached
        except Exception:
            pass

    if not foam_src.is_dir():
        raise FileNotFoundError(f"FOAM_SRC not found: {foam_src}")

    entries = prefilter_tree(foam_src)
    tree_paths = {e["rel"] for e in entries}
    tree_listing = render_tree_listing(entries)

    state = SlateState(foam_src=foam_src)

    from langchain_core.messages import HumanMessage, SystemMessage  # type: ignore

    round_no = 0
    stopped_reason = "max_rounds"
    rejected_last_round: List[str] = []
    while round_no < _MAX_ROUNDS:
        round_no += 1
        slate_contents = _budget_slate_reads(foam_src, state.slate)
        user_prompt = _build_round_prompt(
            task, tree_listing, state, slate_contents, round_no,
            rejected_last_round=rejected_last_round,
        )
        messages = [
            SystemMessage(content=_SYS_PROMPT),
            HumanMessage(content=user_prompt),
        ]
        try:
            raw = _call_llm(messages)
            parsed = _parse_round_response(raw)
        except Exception as exc:
            state.history.append(SlateRound(
                round_no=round_no, slate=list(state.slate),
                diff={"error": str(exc)[:300]}, ready=False, reasoning=""
            ))
            if round_no >= _MIN_ROUNDS:
                stopped_reason = f"llm_error_round_{round_no}"
                break
            continue

        proposed_slate = parsed.get("slate") or []
        if not isinstance(proposed_slate, list):
            proposed_slate = []
        new_slate = _validate_slate_against_tree(proposed_slate, tree_paths, state.banned)

        # Capture rejections so next round's prompt can show the LLM which of its
        # proposed paths failed tree-validation. This breaks the loop where an
        # LLM confidently proposes the same hallucinated ESI/foam-extend paths
        # round after round.
        proposed_norm = [
            (p.strip().lstrip("./").replace("\\", "/") if isinstance(p, str) else "")
            for p in proposed_slate
        ]
        rejected_last_round = [p for p in proposed_norm if p and p not in tree_paths]

        _update_cycle_strikes(state.slate, new_slate, state)

        diff = {
            "kept": sorted(set(state.slate) & set(new_slate)),
            "dropped": sorted(set(state.slate) - set(new_slate)),
            "added": sorted(set(new_slate) - set(state.slate)),
            "drop_reasons": parsed.get("drop_reasons", {}),
            "add_reasons": parsed.get("add_reasons", {}),
            "banned_after_round": sorted(state.banned),
            "rejected_by_tree_validator": rejected_last_round,
        }
        ready = bool(parsed.get("ready_to_code_mod", False))
        reasoning = str(parsed.get("reasoning_brief", ""))[:500]
        state.history.append(SlateRound(
            round_no=round_no, slate=new_slate,
            diff=diff, ready=ready, reasoning=reasoning,
        ))

        prev_slate = state.slate
        state.slate = new_slate

        # Don't stop on no_change if the slate is STILL empty — the LLM's first
        # attempt was invalid and the feedback prompt hasn't had its chance.
        # (ready=True is honored regardless — LLM owns the confidence signal.)
        no_change = (round_no > 1 and set(new_slate) == set(prev_slate) and bool(new_slate))
        if round_no >= _MIN_ROUNDS and (ready or no_change):
            stopped_reason = "ready" if ready else "no_change"
            break

    # Extract deliverables from final slate
    verified_includes, class_sigs = _extract_deliverables(foam_src, state.slate)

    result: Dict[str, Any] = {
        "version": 1,
        "foam_src": str(foam_src),
        "task": task,
        "selected_files": [
            {"rel": rel, "abs": str((foam_src / rel).resolve()), "exists": (foam_src / rel).is_file()}
            for rel in state.slate
        ],
        "verified_include_paths": verified_includes,
        "class_signatures": class_sigs,
        "rounds_run": round_no,
        "stopped_reason": stopped_reason,
        "cache_hit": False,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if history_path is not None:
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history_obj = {
            "foam_src": str(foam_src),
            "task": task,
            "rounds": [
                {
                    "round_no": r.round_no,
                    "slate": r.slate,
                    "diff": r.diff,
                    "ready": r.ready,
                    "reasoning": r.reasoning,
                }
                for r in state.history
            ],
            "banned": sorted(state.banned),
            "stopped_reason": stopped_reason,
        }
        history_path.write_text(json.dumps(history_obj, indent=2), encoding="utf-8")
    return result


# ---------------------------------------------------------------------------
# Deliverable extraction
# ---------------------------------------------------------------------------

_CLASS_RE = re.compile(r"^\s*(?:template\s*<[^>]*>\s*)?class\s+([A-Za-z_]\w*)", re.MULTILINE)
_VIRTUAL_RE = re.compile(r"^\s*virtual\s+[^;]+;", re.MULTILINE)


def _extract_deliverables(
    foam_src: Path, slate: List[str]
) -> Tuple[List[Dict[str, str]], List[Dict[str, Any]]]:
    """From the final slate, derive verified -I paths + class signatures."""
    include_dirs: Dict[str, str] = {}  # abs -> source file that justified it
    class_sigs: List[Dict[str, Any]] = []

    for rel in slate:
        p = (foam_src / rel).resolve()
        if not p.is_file():
            continue
        parent = p.parent
        # Canonical OpenFOAM include convention: add the parent *library* dir's
        # lnInclude if it exists (that's what wmake generates and what other
        # modules include from). Fall back to the parent dir itself.
        lninclude = None
        probe = parent
        for _ in range(4):
            cand = probe / "lnInclude"
            if cand.is_dir():
                lninclude = cand
                break
            probe = probe.parent
            if probe == foam_src.parent or probe == Path("/"):
                break
        chosen = lninclude if lninclude else parent
        include_dirs.setdefault(str(chosen), rel)

        # Class signatures
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for m in _CLASS_RE.finditer(txt):
            class_sigs.append({
                "class": m.group(1),
                "defined_in": rel,
                "line": txt.count("\n", 0, m.start()) + 1,
            })
        virtuals = _VIRTUAL_RE.findall(txt)[:6]
        if virtuals:
            class_sigs.append({
                "source_file": rel,
                "virtual_decls": [v.strip()[:200] for v in virtuals],
            })

    # Convert to -I form relative to $LIB_SRC where possible
    lib_src = foam_src  # $LIB_SRC == $WM_PROJECT_DIR/src
    include_paths: List[Dict[str, str]] = []
    for abs_dir, justified_by in sorted(include_dirs.items()):
        try:
            rel_to_src = str(Path(abs_dir).resolve().relative_to(lib_src))
            make_options_form = f"-I$(LIB_SRC)/{rel_to_src}"
        except ValueError:
            make_options_form = f"-I{abs_dir}"
        include_paths.append({
            "abs": abs_dir,
            "make_options_form": make_options_form,
            "justified_by": justified_by,
        })
    return include_paths, class_sigs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _discover_foam_src(override: str = "") -> Optional[Path]:
    if override:
        p = Path(override).expanduser().resolve()
        return p if p.is_dir() else None
    wm = os.environ.get("WM_PROJECT_DIR", "").strip()
    if wm:
        cand = Path(wm).expanduser().resolve() / "src"
        if cand.is_dir():
            return cand
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--foam-src", default="", help="OpenFOAM src dir (defaults to $WM_PROJECT_DIR/src)")
    parser.add_argument("--topic", required=True)
    parser.add_argument("--mode", default="", help="code-mod mode hint (custom_viscosity|custom_turbulence_model_modification|custom_source|custom_case_library)")
    parser.add_argument("--parent", default="", help="parent class hint, e.g. SpalartAllmaras")
    parser.add_argument("--formula", default="", help="formula / model-change text")
    parser.add_argument("--output", required=True, help="where to write discovered_paths.json")
    parser.add_argument("--history", default="", help="optional path for discovered_paths.history.json")
    parser.add_argument("--force", action="store_true", help="ignore existing cache and re-run")
    args = parser.parse_args()

    foam_src = _discover_foam_src(args.foam_src)
    if foam_src is None:
        print("ERROR: $WM_PROJECT_DIR/src not found; pass --foam-src", file=sys.stderr)
        return 1

    repo_root = Path(__file__).resolve().parent.parent
    _bootstrap_paths(repo_root)

    cache_path = Path(args.output).expanduser().resolve()
    history_path = Path(args.history).expanduser().resolve() if args.history else (
        cache_path.parent / "discovered_paths.history.json"
    )
    task = {
        "topic": args.topic,
        "mode": args.mode,
        "parent_class": args.parent,
        "formula": args.formula[:4000],
    }
    result = run_slate_search(
        foam_src=foam_src,
        task=task,
        cache_path=cache_path,
        history_path=history_path,
        allow_cache_hit=not args.force,
    )
    print(json.dumps({
        "cache_hit": result.get("cache_hit", False),
        "rounds_run": result.get("rounds_run"),
        "stopped_reason": result.get("stopped_reason"),
        "selected_files": [f["rel"] for f in result.get("selected_files", [])],
        "verified_include_paths": [p["make_options_form"] for p in result.get("verified_include_paths", [])],
        "output": str(cache_path),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
