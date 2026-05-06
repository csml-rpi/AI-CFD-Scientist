#!/usr/bin/env python3
"""
OED extensions — Phase 1 (multi-metric), Phase 2 (close+far diversity),
Phase 3 (multi-flow / multi-reference).

All capabilities are opt-in via flags. When all flags are off, the existing
single-metric / single-flow / greedy OED loop runs unchanged.

Designed as a side module so the main open_ended_discovery.py keeps a small
diff. Pure functions; no global state. Imports from open_ended_discovery.py
are deferred to function bodies to avoid circular imports.

Phase 1 — multi-metric:
  propose_metric_set(...)              — LLM enumerates relevant metrics
  discover_existing_comparators(...)   — find compare_*.py producing each metric
  author_comparator(...)               — LLM authors a comparator for missing metrics
  selftest_comparator(...)             — sanity-check authored comparator on ref data
  compute_metric_vector(...)           — run all comparators, emit {metric: value, ...}
  render_metric_vector_for_prompt(...) — format for LLM decision prompt

Phase 2 — diversity:
  load_family_tracker(...)             — read families_explored.json
  update_family_tracker(...)           — record family of accepted candidate
  decide_search_mode(...)              — close|far for next iteration
  render_diversity_constraint(...)     — prompt fragment forcing far-from-baseline

Phase 3 — multi-flow:
  detect_multi_flow_setup(...)         — auto-detect or honor --starter-dirs
  build_per_flow_contracts(...)        — Phase 1 contract per flow
  aggregate_flow_scores(...)           — combine per-flow metric vectors
  render_score_matrix(...)             — format flow x metric matrix for LLM
"""
from __future__ import annotations

import json
import multiprocessing as _mp
import pickle as _pickle
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_METRIC_NUMBER_RE = re.compile(r"(?i)([A-Za-z_][A-Za-z0-9_]*)\s*[:=]\s*(-?\d+\.?\d*(?:[eE][+-]?\d+)?)")


def _read_json(p: Path, default: Any) -> Any:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = "\n".join(s.split("\n")[1:])
        s = s.rsplit("```", 1)[0]
    return s.strip()


def _llm_worker(msgs_pickle: bytes, model: str, temp: float, queue: Any) -> None:
    """Child-process worker for `_llm_invoke`. Defined at module scope so the
    `fork` multiprocessing context can locate it cleanly. The result is
    pushed onto `queue` as a (kind, payload) tuple where kind is 'ok' or 'err'.
    """
    try:
        from langchain_core.messages import HumanMessage, SystemMessage  # type: ignore
        from cfd_langgraph.llm.factory import create_langchain_llm  # type: ignore
        msgs_in = _pickle.loads(msgs_pickle)
        msgs = []
        for role, content in msgs_in:
            cls = SystemMessage if role == "system" else HumanMessage
            msgs.append(cls(content=content))
        llm = create_langchain_llm(model=model, temperature=temp)
        raw = llm.invoke(msgs)
        queue.put(("ok", str(getattr(raw, "content", raw))))
    except Exception as e:  # pragma: no cover — child-process error path
        import traceback
        queue.put(("err", f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=5)}"))


def _llm_invoke(messages: List[Tuple[str, str]], temperature: float = 0.0,
                timeout_s: int = 600) -> str:
    """Single-shot LLM call with hard timeout. messages = [(role, content), ...]
    with role in {'system','user'}. Returns content string.

    Runs the underlying ``llm.invoke`` in a forked child process. On timeout,
    the child is terminated (SIGTERM then SIGKILL), which cascades to any
    SDK subprocess (e.g. the ``claude`` binary spawned by claude-code) so we
    do not leak hung descendants. Raises:
      - ``TimeoutError`` if the call exceeds ``timeout_s`` (default 600s).
      - ``RuntimeError`` if the child terminates without producing output, or
        if the child raised an exception.
    """
    from cfd_langgraph.config import get_settings  # type: ignore
    model = get_settings().model
    ctx = _mp.get_context("fork")
    q = ctx.Queue()
    p = ctx.Process(
        target=_llm_worker,
        args=(_pickle.dumps(messages), model, temperature, q),
    )
    p.start()
    p.join(timeout=timeout_s)
    if p.is_alive():
        # SIGTERM then SIGKILL the python fork-child. The child may have
        # spawned a SDK subprocess (e.g. claude-code's `claude` binary).
        # Killing only the python parent leaves the SDK reparented to init,
        # consuming session resources (we have observed this in the wild).
        # So before we terminate the python child, snapshot its descendants
        # via /proc and reap them via SIGKILL.
        try:
            child_pid = p.pid
            descendants: List[int] = []
            if child_pid is not None:
                # Walk /proc/<pid>/status to find PPID==child_pid (or transitive
                # descendants) — reasonable on Linux. Also try pgrep -P.
                try:
                    import subprocess as _sub
                    out = _sub.run(
                        ["pgrep", "-P", str(child_pid)],
                        capture_output=True, text=True, timeout=2,
                    )
                    descendants = [int(x) for x in out.stdout.split() if x.isdigit()]
                    # Recurse one level: each direct child may have its own
                    # descendants (e.g. SDK -> child workers).
                    for d in list(descendants):
                        out2 = _sub.run(
                            ["pgrep", "-P", str(d)],
                            capture_output=True, text=True, timeout=2,
                        )
                        descendants.extend(int(x) for x in out2.stdout.split() if x.isdigit())
                except Exception:
                    pass
        except Exception:
            descendants = []

        p.terminate()
        p.join(timeout=5)
        if p.is_alive():
            p.kill()
            p.join(timeout=2)

        # Reap any descendants that survived (claude SDK and the like).
        if descendants:
            import os as _os
            import signal as _signal
            for d in descendants:
                try:
                    _os.kill(d, _signal.SIGKILL)
                except Exception:
                    pass

        raise TimeoutError(
            f"LLM call exceeded {timeout_s}s; SDK subprocess killed "
            f"(reaped {len(descendants)} descendant pids)"
        )
    if q.empty():
        raise RuntimeError("LLM child terminated without producing output")
    kind, payload = q.get_nowait()
    if kind == "err":
        raise RuntimeError(f"LLM child raised: {payload}")
    return payload


# ===========================================================================
# PHASE 1 — multi-metric
# ===========================================================================

_DEFAULT_AGGREGATOR = "weighted_sum"

# Maximum tool-loop turns the metric-proposer LLM may take before being forced
# to emit a final spec list. Mirrors `_VERIFIER_MAX_TURNS` semantics.
_PROPOSER_MAX_TURNS = 8

# Regex catching named numeric expectations in a metric description or hint,
# e.g. "DNS x_reattach ~ 4.72h", "expected value = 9.99", "reference ≈ 0.32",
# "Baseline SA error: |7.7529 - 4.7256| ~ 3.027". Captures (label, value).
# Topic-agnostic. Matches `~`, `≈`, OR `=` followed by a number.
_NAMED_EXPECTATION_RE = re.compile(
    r"([A-Za-z][A-Za-z0-9_\-/ ]{0,60}?)\s*[~≈=]\s*(-?\d+(?:\.\d+)?)",
)


def _extract_all_expectations(spec: Dict[str, Any]) -> List[Tuple[str, float]]:
    """Scan a metric spec's description AND computation_hint for any
    `<label> ~|=|≈ <value>` patterns. Returns deduplicated list of (label, value)
    pairs. Topic-agnostic — pulls whatever the proposer wrote.
    """
    out: List[Tuple[str, float]] = []
    seen: set = set()
    for field in ("description", "computation_hint", "ref_column"):
        text = str(spec.get(field, "") or "")
        for m in _NAMED_EXPECTATION_RE.finditer(text):
            label = (m.group(1) or "").strip().rstrip(":,;")
            try:
                val = float(m.group(2))
            except Exception:
                continue
            if not label:
                continue
            key = (label.lower(), round(val, 6))
            if key in seen:
                continue
            seen.add(key)
            out.append((label, val))
    return out


# ---------------------------------------------------------------------------
# Data-source inventory (generic, topic-agnostic).
# Walks the case to enumerate candidate data sources for the metric proposer
# and shows the first lines of each so the LLM can match a metric's intent
# to the file's actual content (per-face boundary fields vs per-timestep
# aggregates), not just the filename.
# ---------------------------------------------------------------------------

_INV_SKIP_EXTS = {".gz", ".swp", ".bak", ".tmp"}
_INV_MAX_BYTES = 50 * 1024 * 1024  # 50 MB
_INV_BIN_SAMPLE = 256


def _looks_binary(p: Path) -> bool:
    """Cheap binary heuristic: NUL byte in first byte, or >5% non-printable
    bytes in the first 256-byte window. Used to skip obviously-binary files
    so the inventory `head` excerpts stay useful."""
    try:
        with p.open("rb") as fh:
            chunk = fh.read(_INV_BIN_SAMPLE)
        if not chunk:
            return False
        if chunk[:1] == b"\x00":
            return True
        # printable = ASCII 9,10,13 or 32..126; everything else is "non-printable"
        nonprint = 0
        for b in chunk:
            if b in (9, 10, 13):
                continue
            if 32 <= b <= 126:
                continue
            nonprint += 1
        return (nonprint / max(1, len(chunk))) > 0.05
    except Exception:
        return True


def _read_head(p: Path, head_lines: int) -> str:
    """Read first `head_lines` lines of `p` as text; latin-1 fallback. Truncate
    each line to 200 chars."""
    raw = b""
    try:
        with p.open("rb") as fh:
            # Read enough bytes to likely cover head_lines lines without
            # slurping huge files.
            raw = fh.read(64 * 1024)
    except Exception:
        return ""
    text = ""
    for enc in ("utf-8", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except Exception:
            continue
    if not text:
        return ""
    out_lines: List[str] = []
    for ln in text.splitlines()[:head_lines]:
        if len(ln) > 200:
            ln = ln[:200] + "..."
        out_lines.append(ln)
    return "\n".join(out_lines)


def _latest_numeric_time_dir(case_dir: Path) -> Optional[Path]:
    """Return the latest numeric time directory of an OpenFOAM case.
    Largest non-zero directory name parseable as float; falls back to '0'
    if only zero exists; None if no numeric time dir present."""
    if not case_dir.is_dir():
        return None
    candidates: List[Tuple[float, Path]] = []
    zero_dir: Optional[Path] = None
    for child in case_dir.iterdir():
        if not child.is_dir():
            continue
        try:
            t = float(child.name)
        except Exception:
            continue
        if t == 0.0:
            zero_dir = child
        else:
            candidates.append((t, child))
    if candidates:
        candidates.sort(key=lambda x: x[0])
        return candidates[-1][1]
    return zero_dir


def _templatize_path_components(rel_parts: Tuple[str, ...]) -> str:
    """Replace any path component that is purely numeric (int or float-like)
    with the literal `<time>` placeholder. Generic — does not depend on
    OpenFOAM's specific numbering."""
    out: List[str] = []
    for part in rel_parts:
        try:
            float(part)
            out.append("<time>")
        except Exception:
            out.append(part)
    return "/".join(out)


def _sample_evenly(items: List[Any], cap: int) -> List[Any]:
    """Sample at most `cap` items evenly across name-sorted order."""
    if len(items) <= cap:
        return items
    step = len(items) / float(cap)
    picked: List[Any] = []
    for i in range(cap):
        idx = int(i * step)
        if idx >= len(items):
            idx = len(items) - 1
        picked.append(items[idx])
    return picked


def _enumerate_data_sources(
    baseline_case_dir: Path,
    reference_search_paths: List[Path],
    head_lines: int = 8,
    max_files_per_class: int = 30,
) -> Dict[str, Any]:
    """Enumerate all candidate data sources for a CFD case, generically.

    Returns:
      {
        "boundary_fields": [{"path": "<time>/<fieldName>", "head": "...", "size": int}, ...],
        "postprocessing":  [{"path": "postProcessing/<fo>/<time>/<file>", "head": "...", "size": int}, ...],
        "reference_files": [{"path": "/abs/path/foo.csv", "head": "...", "size": int}, ...],
        "all_paths":       [<flat list of valid data_source strings>, ...],
      }

    Topic-agnostic. No specific patch / field / file / solver names are
    encoded here.
    """
    inv: Dict[str, Any] = {
        "boundary_fields": [],
        "postprocessing": [],
        "reference_files": [],
        "all_paths": [],
    }

    def _file_ok(p: Path) -> bool:
        try:
            if not p.is_file():
                return False
            if p.suffix.lower() in _INV_SKIP_EXTS:
                return False
            if p.stat().st_size > _INV_MAX_BYTES:
                return False
            # Topic-agnostic skip: never read polyMesh contents from any
            # OpenFOAM case dir. polyMesh holds large geometry/connectivity
            # files that waste tokens during agent exploration.
            if "polyMesh" in p.parts:
                return False
            return True
        except Exception:
            return False

    case_dir = Path(baseline_case_dir) if baseline_case_dir else None

    # --- Boundary fields: files directly under the latest numeric time-dir,
    # excluding the `uniform/` subdir contents. ---
    if case_dir and case_dir.is_dir():
        time_dir = _latest_numeric_time_dir(case_dir)
        if time_dir is not None and time_dir.is_dir():
            cands: List[Path] = []
            for child in sorted(time_dir.iterdir(), key=lambda q: q.name):
                # Skip the uniform/ subdir entirely.
                if child.is_dir() and child.name == "uniform":
                    continue
                # Boundary fields are normally regular files at this depth.
                if child.is_file() and _file_ok(child):
                    cands.append(child)
            cands = _sample_evenly(cands, max_files_per_class)
            for f in cands:
                rel = f.relative_to(case_dir)
                templ = _templatize_path_components(rel.parts)
                if _looks_binary(f):
                    inv["boundary_fields"].append({
                        "path": templ,
                        "head": "<binary, skipped>",
                        "size": int(f.stat().st_size),
                    })
                    continue
                head = _read_head(f, head_lines)
                inv["boundary_fields"].append({
                    "path": templ,
                    "head": head,
                    "size": int(f.stat().st_size),
                })
                if templ not in inv["all_paths"]:
                    inv["all_paths"].append(templ)

    # --- postProcessing: recursive walk of postProcessing/ if it exists. ---
    if case_dir and (case_dir / "postProcessing").is_dir():
        pp_root = case_dir / "postProcessing"
        cands_pp: List[Path] = []
        for f in sorted(pp_root.rglob("*"), key=lambda q: str(q)):
            if not _file_ok(f):
                continue
            if f.suffix.lower() in _INV_SKIP_EXTS:
                continue
            cands_pp.append(f)
        cands_pp = _sample_evenly(cands_pp, max_files_per_class)
        for f in cands_pp:
            rel = f.relative_to(case_dir)
            templ = _templatize_path_components(rel.parts)
            if _looks_binary(f):
                inv["postprocessing"].append({
                    "path": templ,
                    "head": "<binary, skipped>",
                    "size": int(f.stat().st_size),
                })
                continue
            head = _read_head(f, head_lines)
            inv["postprocessing"].append({
                "path": templ,
                "head": head,
                "size": int(f.stat().st_size),
            })
            if templ not in inv["all_paths"]:
                inv["all_paths"].append(templ)

    # --- Reference files: walk each search path (depth <= 3). ---
    seen_refs: set = set()
    for root in reference_search_paths or []:
        try:
            root = Path(root)
        except Exception:
            continue
        if not root or not root.exists():
            continue
        if root.is_file():
            iter_files: List[Path] = [root]
        else:
            iter_files = []
            root_parts = len(root.resolve().parts)
            for f in sorted(root.rglob("*"), key=lambda q: str(q)):
                try:
                    depth = len(f.resolve().parts) - root_parts
                except Exception:
                    continue
                if depth > 3:
                    continue
                if not _file_ok(f):
                    continue
                # Topic-agnostic skip: never list polyMesh contents from
                # starter/reference search paths. polyMesh files are large,
                # binary-leaning, and irrelevant for metric setup — they
                # waste agent turns on geometry exploration.
                if "polyMesh" in f.parts:
                    continue
                iter_files.append(f)
        for f in iter_files:
            abs_p = str(f.resolve())
            if abs_p in seen_refs:
                continue
            seen_refs.add(abs_p)
            if _looks_binary(f):
                inv["reference_files"].append({
                    "path": abs_p,
                    "head": "<binary, skipped>",
                    "size": int(f.stat().st_size),
                })
                continue
            head = _read_head(f, head_lines)
            inv["reference_files"].append({
                "path": abs_p,
                "head": head,
                "size": int(f.stat().st_size),
            })
            if abs_p not in inv["all_paths"]:
                inv["all_paths"].append(abs_p)
    # Cap reference list as well
    if len(inv["reference_files"]) > max_files_per_class:
        inv["reference_files"] = _sample_evenly(inv["reference_files"], max_files_per_class)
        # Rebuild all_paths reference portion to match cap
        kept_refs = {entry["path"] for entry in inv["reference_files"]
                     if entry.get("head") != "<binary, skipped>"}
        inv["all_paths"] = [p for p in inv["all_paths"]
                            if (not p.startswith("/")) or p in kept_refs]

    return inv


def _format_inventory_block(inventory: Dict[str, Any]) -> str:
    """Render the inventory dict as a text block for the metric-proposer
    prompt. Generic — uses placeholder names only."""
    if not inventory:
        return ""
    bf = inventory.get("boundary_fields") or []
    pp = inventory.get("postprocessing") or []
    rf = inventory.get("reference_files") or []
    if not (bf or pp or rf):
        return ""

    def _block(entries: List[Dict[str, Any]]) -> str:
        out: List[str] = []
        for e in entries:
            out.append(f"  {e.get('path','')}")
            head = e.get("head", "")
            if head:
                head_lines_ = head.splitlines() or [""]
                first = head_lines_[0]
                out.append(f"    head: {first}")
                for extra in head_lines_[1:]:
                    out.append(f"          {extra}")
        return "\n".join(out) if out else "  (none)"

    parts: List[str] = []
    parts.append("=== AVAILABLE DATA SOURCES (you MUST pick `data_source` from this list) ===")
    parts.append("")
    parts.append("[Boundary fields - per-face/per-cell spatial data; pair with constant/polyMesh]")
    parts.append(_block(bf) if bf else "  (none)")
    parts.append("")
    parts.append("[postProcessing aggregates - per-timestep summary, NO spatial info]")
    parts.append(_block(pp) if pp else "  (none)")
    parts.append("")
    parts.append("[Reference files]")
    parts.append(_block(rf) if rf else "  (none)")
    parts.append("")
    parts.append("ENFORCEMENT: your `data_source` field must be exactly one of the paths listed")
    parts.append("above. The loop will reject any metric pointing at a path not in this list.")
    return "\n".join(parts)


def propose_metric_set(
    *,
    topic: str,
    starter_understanding: Dict[str, Any],
    reference_files: List[Path],
    sample_postprocessing: str = "",
    extra_context: str = "",
    baseline_case_dir: Optional[Path] = None,
    reference_search_paths: Optional[List[Path]] = None,
    out_dir: Optional[Path] = None,
    baseline_metrics_path: Optional[Path] = None,
    objective_contract: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Have the LLM enumerate metrics relevant to the topic given the reference
    data and post-processing structure available.

    Returns a list of metric specs:
      {"name": "Cf_RMSE", "description": "...", "direction": "min",
       "data_source": "wallShearStress" | "U" | "p" | "...",
       "ref_column": "Cf" | "x_reattach" | "...",
       "computation_hint": "..."}
    Empty list on failure (caller should fall back to single-metric mode).
    """
    ref_samples = []
    for rf in reference_files[:3]:
        try:
            text = rf.read_text(errors="ignore")[:1500]
            ref_samples.append(f"--- {rf.name} ---\n{text}")
        except Exception:
            pass
    ref_block = "\n\n".join(ref_samples) if ref_samples else "(no reference samples)"

    starter_excerpt = json.dumps(
        {k: starter_understanding.get(k, '') for k in ('flow_parameters', 'reference_data', 'formula_or_model_spec')},
        ensure_ascii=False,
    )[:2000]

    # baseline_metrics_block: stringify baseline_metrics.json (if available).
    # Resolution priority: explicit `baseline_metrics_path` kwarg, else
    # `<baseline_case_dir>/../baseline_metrics.json`, else
    # `<baseline_case_dir>/baseline_metrics.json`. Empty on failure.
    baseline_metrics_block = ""
    _bm_candidates: List[Path] = []
    if baseline_metrics_path is not None:
        _bm_candidates.append(Path(baseline_metrics_path))
    if baseline_case_dir is not None:
        _bcd = Path(baseline_case_dir)
        _bm_candidates.append(_bcd.parent / "baseline_metrics.json")
        _bm_candidates.append(_bcd / "baseline_metrics.json")
    for _bmp in _bm_candidates:
        try:
            if _bmp.is_file():
                _bm_obj = json.loads(_bmp.read_text(encoding="utf-8"))
                _bm_str = json.dumps(_bm_obj, indent=2, ensure_ascii=False)
                if len(_bm_str) > 3000:
                    _bm_str = _bm_str[:3000] + "\n... (truncated)"
                baseline_metrics_block = _bm_str
                break
        except Exception:
            continue
    if not baseline_metrics_block:
        baseline_metrics_block = "(no baseline_metrics.json available)"

    # exemplar_block: if any reference / contract file ends with .py and
    # matches `compare_*.py` or `*_compare.py`, embed its full content
    # (truncated). Used by the LLM as the canonical parsing template.
    _exemplar_candidates: List[Path] = []
    for _rf in (reference_files or []):
        _exemplar_candidates.append(Path(_rf))
    if isinstance(objective_contract, dict):
        for _rf in (objective_contract.get("reference_files") or []):
            try:
                _exemplar_candidates.append(Path(_rf))
            except Exception:
                continue
    exemplar_text = ""
    exemplar_path_used = ""
    for _ec in _exemplar_candidates:
        try:
            if _ec.suffix != ".py" or not _ec.is_file():
                continue
            _stem = _ec.stem
            if not (_stem.startswith("compare_") or _stem.endswith("_compare")):
                continue
            _src = _ec.read_text(encoding="utf-8", errors="ignore")
            if len(_src) > 6000:
                _src = _src[:6000] + "\n# ... (truncated)"
            exemplar_text = _src
            exemplar_path_used = str(_ec)
            break
        except Exception:
            continue
    if exemplar_text:
        exemplar_block = (
            f"# source: {exemplar_path_used}\n"
            f"# Treat the parsing logic below as the canonical, domain-correct\n"
            f"# template (file paths, sign-change rules, windowing, columns).\n"
            f"{exemplar_text}"
        )
    else:
        exemplar_block = "(no exemplar comparator available)"

    # Build a generic inventory of candidate data sources (boundary fields,
    # postProcessing aggregates, reference files) with `head` excerpts. The
    # LLM picks `data_source` from this enumerated list; we validate after.
    inventory: Dict[str, Any] = {}
    file_inventory_block = ""
    try:
        if baseline_case_dir is not None:
            inventory = _enumerate_data_sources(
                baseline_case_dir=Path(baseline_case_dir),
                reference_search_paths=[Path(p) for p in (reference_search_paths or [])],
            )
            file_inventory_block = _format_inventory_block(inventory)
    except Exception as exc:
        print(f"[OED-EXT][phase1] inventory build failed (non-fatal): {exc}")
        inventory = {}
        file_inventory_block = ""

    # Load MetricProposer prompts from prompts.yaml when available, else fall
    # back to in-file defaults so this still works when prompts.yaml is older.
    sys_msg = ""
    user_msg = ""
    try:
        from cfd_langgraph.config import get_settings  # type: ignore
        from cfd_langgraph.ideation import load_prompts  # type: ignore
        prompts = load_prompts(get_settings().prompts_path) or {}
        block = prompts.get("MetricProposer", {}) or {}
        sys_msg = block.get("metric_proposer_system_prompt", "") or ""
        user_tmpl = block.get("metric_proposer_user_prompt", "") or ""
        if sys_msg and user_tmpl:
            _fmt_kwargs = dict(
                topic=topic,
                ref_block=ref_block,
                sample_postprocessing=sample_postprocessing[:2000],
                starter_understanding_excerpt=starter_excerpt,
                extra_context=extra_context,
                file_inventory_block=file_inventory_block,
                baseline_metrics_block=baseline_metrics_block,
                exemplar_block=exemplar_block,
            )
            try:
                user_msg = user_tmpl.format(**_fmt_kwargs)
            except KeyError:
                # Older prompts.yaml: drop unknown placeholders progressively
                # and append the missing blocks at the end for partial benefit.
                _trail = ""
                _kw_min = dict(
                    topic=topic,
                    ref_block=ref_block,
                    sample_postprocessing=sample_postprocessing[:2000],
                    starter_understanding_excerpt=starter_excerpt,
                    extra_context=extra_context,
                )
                try:
                    user_msg = user_tmpl.format(**_kw_min, file_inventory_block=file_inventory_block)
                except KeyError:
                    user_msg = user_tmpl.format(**_kw_min)
                    if file_inventory_block:
                        _trail += f"\n\n{file_inventory_block}"
                if baseline_metrics_block:
                    _trail += f"\n\n=== BASELINE METRICS ===\n{baseline_metrics_block}"
                if exemplar_block:
                    _trail += f"\n\n=== EXEMPLAR COMPARATOR ===\n{exemplar_block}"
                user_msg = f"{user_msg}{_trail}"
    except Exception:
        sys_msg = ""
        user_msg = ""
    if not (sys_msg and user_msg):
        sys_msg = (
            "You are a CFD evaluation expert. Given a research topic, reference "
            "datasets, and the postProcessing structure of OpenFOAM cases, "
            "enumerate the QUANTITATIVE METRICS that should be tracked to judge "
            "whether a candidate model improves over baseline.\n\n"
            "RULES:\n"
            "- Propose 2 to 6 metrics covering different aspects (global error, "
            "spatial features, profile shape). Avoid redundancy.\n"
            "- Each metric must be derivable from OpenFOAM output (boundary "
            "fields, sampling lines, postProcessing function-object outputs, "
            "surface fields) plus the reference dataset.\n"
            "- For each metric specify: name, one-line description, direction "
            "('min' for errors, 'max' for correlation/agreement), data_source "
            "(which output it reads), ref_column (which column or feature in "
            "the reference dataset), computation_hint (a short note on how to "
            "compute it).\n"
            "- Return STRICT JSON array only. No prose.\n\n"
            "OpenFOAM data source taxonomy. When you specify `data_source`:\n"
            "- Boundary field (per-face, spatial): `<time>/<fieldName>` — the "
            "volScalarField/volVectorField file inside a time directory. "
            "Required for any metric needing spatial information (profile "
            "errors, RMSE along a coordinate, zero-crossings, local extrema, "
            "sampled lines/planes).\n"
            "- postProcessing aggregate: `postProcessing/<funcObjectName>/<time>/<file>.dat` "
            "— per-timestep summary (min/max/avg/integral) only; no spatial "
            "information. Use only for scalar global quantities.\n"
            "- postProcessing per-cell-set: `postProcessing/<funcObjectName>/<time>/<file>` "
            "(no `.dat`) — boundary-field-like format; treat as boundary field.\n"
            "Pick the right source. Do NOT name a postProcessing aggregate "
            "`.dat` for spatial metrics."
        )
        user_msg = (
            f"TOPIC:\n{topic}\n\n"
            f"REFERENCE DATA SAMPLES:\n{ref_block}\n\n"
            f"POSTPROCESSING TREE (sample):\n{sample_postprocessing[:2000]}\n\n"
            f"STARTER UNDERSTANDING (excerpt):\n{starter_excerpt}\n\n"
            f"=== BASELINE METRICS ===\n{baseline_metrics_block}\n\n"
            f"=== EXEMPLAR COMPARATOR ===\n{exemplar_block}\n\n"
            f"{extra_context}"
        )
        if file_inventory_block:
            user_msg = f"{user_msg}\n\n{file_inventory_block}"
    def _coerce_specs(arr: Any) -> List[Dict[str, Any]]:
        if not isinstance(arr, list):
            return []
        parsed: List[Dict[str, Any]] = []
        for m in arr:
            if not isinstance(m, dict):
                continue
            name = str(m.get("name", "")).strip()
            if not name:
                continue
            pm = str(m.get("preferred_method", "auto")).lower().strip() or "auto"
            if pm not in {"text", "pyvista", "auto"}:
                pm = "auto"
            parsed.append({
                "name": name,
                "description": str(m.get("description", "")).strip(),
                "direction": str(m.get("direction", "min")).lower().strip() or "min",
                "data_source": str(m.get("data_source", "")).strip(),
                "ref_column": str(m.get("ref_column", "")).strip(),
                "computation_hint": str(m.get("computation_hint", "")).strip(),
                "preferred_method": pm,
            })
        return parsed

    def _parse_proposer_response(raw_text: str) -> Tuple[str, Any]:
        """Return (kind, payload). kind in {'tool','final','unknown'}.
        - 'tool'  -> payload = {'code': str}
        - 'final' -> payload = list[metric-spec-dict]
        Accepts either {"tool":"python_script","code":...},
        {"metrics":[...]}, or a bare top-level JSON array (legacy).
        """
        body = _strip_code_fences(raw_text or "")
        # Try object first.
        i_obj = body.find("{")
        i_arr = body.find("[")
        # Prefer whichever appears first.
        first = min([x for x in (i_obj, i_arr) if x >= 0], default=-1)
        if first < 0:
            return "unknown", {"raw": body}
        if first == i_obj:
            depth = 0
            end = -1
            in_str = False
            esc = False
            for j in range(i_obj, len(body)):
                ch = body[j]
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                else:
                    if ch == '"':
                        in_str = True
                    elif ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            end = j + 1
                            break
            if end < 0:
                return "unknown", {"raw": body}
            try:
                obj = json.loads(body[i_obj:end])
            except Exception:
                return "unknown", {"raw": body[i_obj:end]}
            if isinstance(obj, dict) and obj.get("tool") == "python_script" and isinstance(obj.get("code"), str):
                return "tool", {"code": obj["code"]}
            if isinstance(obj, dict) and isinstance(obj.get("metrics"), list):
                return "final", _coerce_specs(obj["metrics"])
            return "unknown", {"raw": obj}
        # Top-level array (legacy single-shot format).
        depth = 0
        end = -1
        for j in range(i_arr, len(body)):
            if body[j] == "[":
                depth += 1
            elif body[j] == "]":
                depth -= 1
                if depth == 0:
                    end = j + 1
                    break
        if end < 0:
            return "unknown", {"raw": body}
        try:
            arr = json.loads(body[i_arr:end])
        except Exception:
            return "unknown", {"raw": body[i_arr:end]}
        return "final", _coerce_specs(arr)

    allowed_paths = list(inventory.get("all_paths") or [])
    enforce = bool(file_inventory_block) and bool(allowed_paths)

    attempts_log: List[Dict[str, Any]] = []
    final_valid: List[Dict[str, Any]] = []
    dropped: List[Dict[str, Any]] = []
    self_test_log: List[Dict[str, Any]] = []
    messages: List[Tuple[str, str]] = [("system", sys_msg), ("user", user_msg)]

    def _run_tool_loop(msgs: List[Tuple[str, str]]) -> Tuple[List[Dict[str, Any]], str, List[Tuple[str, str]]]:
        """Drive the proposer through up to `_PROPOSER_MAX_TURNS` turns.
        Returns (specs, last_raw, updated_messages). `specs` is [] if no
        final answer was produced.
        """
        local_msgs = list(msgs)
        last_raw = ""
        for turn_idx in range(1, _PROPOSER_MAX_TURNS + 1):
            try:
                raw = _llm_invoke(local_msgs, temperature=0.0)
            except TimeoutError:
                return [], last_raw, local_msgs
            last_raw = raw
            kind, payload = _parse_proposer_response(raw)
            if kind == "final":
                local_msgs = local_msgs + [("assistant", raw)]
                return list(payload), raw, local_msgs
            if kind == "tool" and turn_idx < _PROPOSER_MAX_TURNS:
                code = payload.get("code", "")
                stdout, stderr, rc = _run_verifier_script_sandboxed(code)
                feedback = (
                    f"TOOL RESULT (turn {turn_idx}):\n"
                    f"STDOUT:\n{(stdout or '')[:2000]}\n"
                    f"STDERR:\n{(stderr or '')[:1000]}\n"
                    f"RETURNCODE: {rc}\n\n"
                    "Continue: either issue another `python_script` tool call "
                    "or emit the final {\"metrics\": [...]} JSON."
                )
                local_msgs = local_msgs + [("assistant", raw), ("user", feedback)]
                continue
            # unknown, or tool on last turn: nudge for a final answer.
            if turn_idx < _PROPOSER_MAX_TURNS:
                local_msgs = local_msgs + [
                    ("assistant", raw),
                    ("user",
                     "Your last response was not parseable as a tool call or a "
                     "final answer. Emit ONLY a single JSON object: either "
                     "{\"tool\": \"python_script\", \"code\": \"...\"} or "
                     "{\"metrics\": [...]}."),
                ]
                continue
            break
        return [], last_raw, local_msgs

    def _named_expectation(desc: str) -> Optional[Tuple[str, float]]:
        """Return (label, value) for the first named numeric expectation in
        `desc`, or None. Topic-agnostic regex; skips obvious noise (single
        trailing-period sentence ends, etc.)."""
        if not desc:
            return None
        for m in _NAMED_EXPECTATION_RE.finditer(desc):
            label = (m.group(1) or "").strip().rstrip(":,;")
            try:
                val = float(m.group(2))
            except Exception:
                continue
            if not label:
                continue
            return (label, val)
        return None

    def _self_test_metric(spec: Dict[str, Any], expectation: Tuple[str, float],
                          base_msgs: List[Tuple[str, str]]) -> Dict[str, Any]:
        """Ask the LLM to write & run a script implementing `spec`'s
        computation_hint, then compare against the named expectation.
        Returns {status, value, relative_error, attempts}.
        """
        label, expected = expectation
        attempts = 0
        last_value: Optional[float] = None
        last_rel_err: Optional[float] = None
        local_msgs = list(base_msgs)
        for round_idx in range(1, 3):  # up to 2 rounds
            attempts = round_idx
            ask = (
                f"SELF-TEST REQUEST for metric `{spec.get('name','')}`.\n"
                f"Your description references a named expectation: "
                f"`{label}` ~ {expected}.\n"
                f"Write a `python_script` tool call implementing your\n"
                f"`computation_hint` against the reference data and PRINT\n"
                f"a single line of the form:\n"
                f"    SELFTEST_VALUE: <float>\n"
                f"so we can compare against the expectation. Use only\n"
                f"pathlib, json, numpy, pandas, csv, re. No network.\n\n"
                f"COMPUTATION HINT:\n{spec.get('computation_hint','')}\n\n"
                f"DATA SOURCE: {spec.get('data_source','')}\n"
                f"REF COLUMN: {spec.get('ref_column','')}\n"
            )
            local_msgs = local_msgs + [("user", ask)]
            try:
                raw = _llm_invoke(local_msgs, temperature=0.0)
            except Exception:
                return {"status": "skipped", "value": None,
                        "relative_error": None, "attempts": attempts,
                        "reason": "llm_error"}
            kind, payload = _parse_proposer_response(raw)
            if kind != "tool":
                return {"status": "skipped", "value": last_value,
                        "relative_error": last_rel_err, "attempts": attempts,
                        "reason": "no_tool_call"}
            code = payload.get("code", "")
            stdout, stderr, rc = _run_verifier_script_sandboxed(code)
            local_msgs = local_msgs + [("assistant", raw)]
            # Extract SELFTEST_VALUE.
            value: Optional[float] = None
            mtch = re.search(r"SELFTEST_VALUE\s*:\s*(-?\d+(?:\.\d+)?)", stdout or "")
            if mtch:
                try:
                    value = float(mtch.group(1))
                except Exception:
                    value = None
            if value is None:
                # Also try the last numeric token on stdout as a fallback.
                tokens = re.findall(r"-?\d+(?:\.\d+)?", stdout or "")
                if tokens:
                    try:
                        value = float(tokens[-1])
                    except Exception:
                        value = None
            last_value = value
            if value is None or expected == 0:
                rel_err = None
            else:
                rel_err = abs(value - expected) / abs(expected)
            last_rel_err = rel_err
            if rel_err is not None and rel_err <= 0.20:
                return {"status": "pass", "value": value,
                        "relative_error": rel_err, "attempts": attempts,
                        "reason": "within_tolerance"}
            # Round failed: append corrective and ask for a revised spec.
            corrective = (
                f"SELF-TEST FAIL (round {round_idx}). Your computation_hint "
                f"produced {value} but the named expectation is "
                f"{expected} (rel_err={rel_err}).\n"
                f"Either (a) revise your `computation_hint` so it correctly "
                f"reproduces `{label}` ~ {expected}, or (b) update the "
                f"named expectation in your description if it was wrong, "
                f"then re-run the self-test.\n"
                f"On the next turn either issue another `python_script` "
                f"tool call (preferred) or emit a final {{\"metrics\": "
                f"[<revised single metric>]}} JSON object."
            )
            local_msgs = local_msgs + [("user", corrective)]
            # Try once more to get an updated spec / value via _run_tool_loop
            # implicitly handled in the next loop iteration's ask.
        return {"status": "fail", "value": last_value,
                "relative_error": last_rel_err, "attempts": attempts,
                "reason": "exceeded_rounds"}

    try:
        max_attempts = 3 if enforce else 1  # original + up to 2 retries
        last_specs: List[Dict[str, Any]] = []
        for attempt in range(1, max_attempts + 1):
            specs, last_raw, messages = _run_tool_loop(messages)
            last_specs = specs

            if not enforce:
                final_valid = specs
                attempts_log.append({
                    "attempt": attempt,
                    "n_metrics_proposed": len(specs),
                    "n_invalid": 0,
                    "invalid_paths": [],
                    "retry": False,
                })
                break

            valid: List[Dict[str, Any]] = []
            invalid: List[Dict[str, Any]] = []
            invalid_paths: List[str] = []
            for s in specs:
                ds = (s.get("data_source") or "").strip()
                if ds in allowed_paths:
                    valid.append(s)
                else:
                    invalid.append(s)
                    invalid_paths.append(ds)

            attempts_log.append({
                "attempt": attempt,
                "n_metrics_proposed": len(specs),
                "n_invalid": len(invalid),
                "invalid_paths": invalid_paths,
                "retry": (len(invalid) > 0 and attempt < max_attempts),
            })

            if not invalid:
                final_valid = valid
                break

            if attempt >= max_attempts:
                # Final attempt still has invalids: keep valid ones, drop bad ones.
                final_valid = valid
                dropped = invalid
                print(
                    f"[OED-EXT][phase1] metric proposer: dropping "
                    f"{len(invalid)} metric(s) after {attempt} attempts due to "
                    f"out-of-inventory data_source values: {invalid_paths}"
                )
                break

            # Build corrective re-prompt and continue.
            flat_list = "\n".join(f"  - {p}" for p in allowed_paths)
            bad_lines = "\n".join(
                f"  - metric `{s.get('name','')}` -> data_source `{s.get('data_source','')}`"
                for s in invalid
            )
            corrective = (
                "Some of your proposed metrics use a `data_source` that is NOT in "
                "the available inventory.\n\n"
                f"Invalid entries:\n{bad_lines}\n\n"
                "You MUST pick `data_source` from this exact list:\n"
                f"{flat_list}\n\n"
                "Re-emit the COMPLETE final answer as "
                "{\"metrics\": [...]} with corrected `data_source` values."
            )
            messages = messages + [("user", corrective)]

        # Self-test pass over final_valid metrics. Annotate each metric in
        # place with self-test outcome fields. Failures are logged but do
        # not block (downstream verifier remains the safety net).
        for spec in final_valid:
            desc = spec.get("description", "") or ""
            ds = (spec.get("data_source") or "").lower()
            ref_col = (spec.get("ref_column") or "").lower()
            references_ref_data = (
                "reference" in desc.lower()
                or bool(ref_col)
                or any(str(rf).lower() in desc.lower() for rf in (reference_files or []))
            )
            expectation = _named_expectation(desc)
            if expectation is None or not references_ref_data:
                spec.update({
                    "self_test_attempts": 0,
                    "named_expectation": None,
                    "self_test_value": None,
                    "self_test_relative_error": None,
                    "self_test_status": "skipped",
                })
                self_test_log.append({"name": spec.get("name", ""), **{
                    k: spec[k] for k in (
                        "self_test_attempts", "named_expectation",
                        "self_test_value", "self_test_relative_error",
                        "self_test_status",
                    )
                }})
                continue
            label, expected_val = expectation
            outcome = _self_test_metric(spec, expectation, messages)
            spec.update({
                "self_test_attempts": int(outcome.get("attempts", 0)),
                "named_expectation": float(expected_val),
                "self_test_value": (
                    float(outcome["value"]) if outcome.get("value") is not None else None
                ),
                "self_test_relative_error": (
                    float(outcome["relative_error"])
                    if outcome.get("relative_error") is not None else None
                ),
                "self_test_status": str(outcome.get("status", "skipped")),
            })
            self_test_log.append({
                "name": spec.get("name", ""),
                "named_expectation_label": label,
                "named_expectation": float(expected_val),
                "self_test_attempts": spec["self_test_attempts"],
                "self_test_value": spec["self_test_value"],
                "self_test_relative_error": spec["self_test_relative_error"],
                "self_test_status": spec["self_test_status"],
                "reason": outcome.get("reason", ""),
            })

        # Persist validation artifact when out_dir is provided.
        if out_dir is not None:
            try:
                _write_json(
                    Path(out_dir) / "metric_proposer_validation.json",
                    {
                        "attempts": attempts_log,
                        "final_valid_metrics": final_valid,
                        "dropped_metrics": dropped,
                        "self_tests": self_test_log,
                        "exemplar_used": exemplar_path_used,
                    },
                )
            except Exception as exc:
                print(f"[OED-EXT][phase1] failed to write validation artifact: {exc}")

        return final_valid if final_valid is not None else last_specs
    except TimeoutError as exc:
        print(f"[OED-EXT][phase1] propose_metric_set timed out: {exc}; "
              "returning empty spec list (caller falls back).")
        return []
    except Exception as exc:
        print(f"[OED-EXT][phase1] propose_metric_set failed: {exc}")
        return []


def discover_existing_comparators(
    *, search_roots: List[Path], metrics: List[Dict[str, Any]]
) -> Dict[str, str]:
    """
    For each metric, find a starter-local Python script that appears to
    compute that metric. Returns {metric_name: absolute_path}.

    Candidates are the .py files the LLM-content classifier flagged as role
    "comparator" or whose summary mentions the metric (cached at
    open_ended_discovery/comparator_classification.json by an earlier call).
    If no cache exists yet, falls back to a starter-local scan of any .py
    file whose content mentions the metric keywords. We do NOT walk the
    repo or runs/ dirs — that previously leaked cross-starter scripts.
    """
    EXCLUDE_DIR_PARTS = {
        "runs", "__pycache__", "node_modules", ".git", "platforms",
        "0", "system", "constant", "polyMesh",
    }
    found: Dict[str, str] = {}

    # Try the LLM classifier's cache first. The cache lives next to
    # open_ended_discovery/objective_contract.json. We don't know the run
    # dir directly, but search_roots[0] is conventionally the starter dir;
    # the cache file path is derivable from any of the search_roots only if
    # one of them happens to be inside a run_dir, so we just scan search
    # roots' parents for an open_ended_discovery/ dir.
    classifier_cached: List[Dict[str, Any]] = []
    for root in search_roots:
        if not root.is_dir():
            continue
        for cand in (root.parent, root):
            cls_file = cand / "open_ended_discovery" / "comparator_classification.json"
            if cls_file.is_file():
                try:
                    data = json.loads(cls_file.read_text())
                    if isinstance(data, dict) and isinstance(data.get("scripts"), list):
                        classifier_cached = data["scripts"]
                        break
                except Exception:
                    pass
        if classifier_cached:
            break

    candidates: List[Path] = []
    if classifier_cached:
        # Use only entries the classifier marked as comparator-like.
        for s in classifier_cached:
            try:
                role = str(s.get("role") or "").lower()
                p = Path(str(s.get("path") or ""))
                if not p.is_file():
                    continue
                if role in ("comparator", "reader"):  # readers may also be hand-written scorers
                    candidates.append(p.resolve())
            except Exception:
                continue
    if not candidates:
        # Cache absent or empty → starter-local content scan with NO naming
        # convention. We accept any .py that mentions any metric's keyword
        # in its body (filtered later by the per-metric scoring loop below).
        for root in search_roots:
            if not root.is_dir():
                continue
            for p in root.rglob("*.py"):
                if not p.is_file():
                    continue
                parts = set(part.lower() for part in p.parts)
                if parts & EXCLUDE_DIR_PARTS:
                    continue
                try:
                    if p.stat().st_size > 200_000:
                        continue
                except OSError:
                    continue
                candidates.append(p.resolve())
    if not candidates:
        return found

    for m in metrics:
        name = m["name"]
        keywords = [name.lower(), m.get("ref_column", "").lower(), m.get("data_source", "").lower()]
        keywords = [k for k in keywords if k]
        # split common composites: "Cf_RMSE" -> ["cf", "rmse"]
        for k in list(keywords):
            for sub in re.split(r"[_\-\s]+", k):
                if sub and sub not in keywords:
                    keywords.append(sub)
        best: Optional[Path] = None
        best_score = 0
        for c in candidates:
            try:
                text = c.read_text(encoding="utf-8", errors="ignore").lower()
            except Exception:
                continue
            score = 0
            for kw in keywords:
                if not kw or len(kw) < 2:
                    continue
                score += text.count(kw)
                if kw in c.name.lower():
                    score += 5
            if score > best_score:
                best_score = score
                best = c
        if best is not None and best_score >= 2:
            found[name] = str(best)
    return found


def author_comparator(
    *,
    metric: Dict[str, Any],
    reference_file: Path,
    sample_pp_tree: str,
    sample_pp_data: str,
    flow_params: Dict[str, Any],
    out_path: Path,
    exemplar_text: str = "",
    baseline_final_time: Optional[float] = None,
) -> Optional[Path]:
    """
    Have the LLM author a self-contained Python comparator for one metric.

    Contract for the generated script:
      - Reads CASE_DIR (Path), REF_FILE (Path), FLOW_PARAMS (dict) from CLI args:
          --case <dir> --reference <file>
        Optional: --time <t> (latest if omitted)
      - Prints exactly one line of the form:
          METRIC <name>: <numeric value>
        followed by optional diagnostic lines.
      - Exits 0 on success, non-zero on failure.

    Returns the script path on success, None on failure.
    """
    metric_name = metric["name"]
    # Pull comparator-author prompts from prompts.yaml if available; fall back
    # to in-file defaults so this still works when prompts.yaml is older.
    sys_msg = ""
    user_msg = ""
    try:
        from cfd_langgraph.config import get_settings  # type: ignore
        from cfd_langgraph.ideation import load_prompts  # type: ignore
        prompts = load_prompts(get_settings().prompts_path) or {}
        block = prompts.get("ComparatorAuthor", {}) or {}
        sys_msg = block.get("comparator_author_system_prompt", "") or ""
        user_tmpl = block.get("comparator_author_user_prompt", "") or ""
        if sys_msg and user_tmpl:
            user_msg = user_tmpl.format(
                metric_name=metric_name,
                direction=metric.get("direction", "min"),
                description=metric.get("description", ""),
                data_source=metric.get("data_source", ""),
                ref_column=metric.get("ref_column", ""),
                computation_hint=metric.get("computation_hint", ""),
                reference_file=str(reference_file),
                reference_sample=(reference_file.read_text(errors="ignore")[:2000]
                                  if reference_file.is_file() else "(missing)"),
                sample_pp_tree=sample_pp_tree[:2000],
                sample_pp_data=sample_pp_data[:1500],
                flow_params=json.dumps(flow_params, ensure_ascii=False)[:1500],
                baseline_final_time=("(unknown)" if baseline_final_time is None
                                     else f"{baseline_final_time}"),
                exemplar_text=exemplar_text[:4000],
            )
    except Exception:
        sys_msg = ""
        user_msg = ""
    if not (sys_msg and user_msg):
        # Fallback to legacy in-file prompt if prompts.yaml not loadable.
        direction = metric.get("direction", "min")
        sys_msg = (
            "You are a CFD post-processing expert. Write a self-contained Python "
            "script that computes ONE specific metric from an OpenFOAM case "
            "against a reference dataset.\n\n"
            "STRICT RULES:\n"
            "- Imports: argparse, pathlib, numpy, csv, json, re, sys. Nothing else.\n"
            "- CLI: --case <dir>, --reference <file>, optional --time <t>, "
            "optional --baseline-time <float>, optional --out <dir>.\n"
            "- When --baseline-time is given, pin to that exact time directory; "
            "else use the largest numeric time dir within tolerance >= 0.5*case "
            "controlDict endTime; never silent-fall back to time=0.\n"
            "- REFUSE to score time==0: print 'METRIC <name>: nan' and exit 2.\n"
            "- Print EXACTLY ONE line of the form 'METRIC <name>: <value>' and "
            "an additional REQUIRED diagnostic line 'TIME_USED: <float>'.\n"
            "- Wrap every file read in try/except; on failure print "
            "'PARSE_WARNING: <path> <reason>' and continue.\n"
            "- Output ONLY raw Python code. No markdown."
        )
        user_msg = (
            f"METRIC NAME: {metric_name}\n"
            f"DIRECTION: {direction} (lower is better if 'min')\n"
            f"DESCRIPTION: {metric.get('description','')}\n"
            f"DATA SOURCE (postProcessing output to read): {metric.get('data_source','')}\n"
            f"REFERENCE COLUMN/FEATURE: {metric.get('ref_column','')}\n"
            f"COMPUTATION HINT: {metric.get('computation_hint','')}\n\n"
            f"REFERENCE FILE: {reference_file}\n"
            f"REFERENCE SAMPLE:\n{reference_file.read_text(errors='ignore')[:2000] if reference_file.is_file() else '(missing)'}\n\n"
            f"POSTPROCESSING TREE:\n{sample_pp_tree[:2000]}\n\n"
            f"POSTPROCESSING SAMPLE DATA:\n{sample_pp_data[:1500]}\n\n"
            f"FLOW PARAMETERS (use for normalization, never hardcode 1):\n"
            f"{json.dumps(flow_params, ensure_ascii=False)[:1500]}\n\n"
            f"BASELINE FINAL TIME (pin scoring to this; refuse time==0): "
            f"{baseline_final_time if baseline_final_time is not None else '(unknown)'}\n\n"
            f"{exemplar_text[:4000]}"
        )
    try:
        raw = _llm_invoke([("system", sys_msg), ("user", user_msg)], temperature=0.0)
        code = _strip_code_fences(raw)
        if not code or "import" not in code:
            return None
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(code, encoding="utf-8")
        return out_path
    except TimeoutError:
        # Surface to author_and_selftest so the attempt log can record
        # failure_mode='timeout' and proceed with retry logic.
        print(f"[OED-EXT][phase1] author_comparator({metric_name}) timed out")
        raise
    except Exception as exc:
        print(f"[OED-EXT][phase1] author_comparator({metric_name}) failed: {exc}")
        return None


def _comparator_supports_baseline_time(comparator: Path, timeout_s: int = 10) -> bool:
    """Probe `--help` to detect `--baseline-time` support. Comparators shipped
    by older starters may not accept the flag; we then run without it."""
    try:
        res = subprocess.run(
            [sys.executable, str(comparator), "--help"],
            capture_output=True, text=True, timeout=timeout_s,
        )
        blob = (res.stdout or "") + "\n" + (res.stderr or "")
        return "--baseline-time" in blob
    except Exception:
        return False


def _run_comparator_with_optional_baseline_time(
    *,
    comparator: Path,
    case_dir: Path,
    reference_file: Path,
    baseline_time: Optional[float],
    timeout_s: int,
) -> "subprocess.CompletedProcess[str]":
    """Run a comparator, forwarding --baseline-time when supported. On any
    error invoking with the flag, retry without it (back-compat)."""
    base_cmd = [sys.executable, str(comparator),
                "--case", str(case_dir),
                "--reference", str(reference_file)]
    if baseline_time is not None and _comparator_supports_baseline_time(comparator):
        try:
            return subprocess.run(
                base_cmd + ["--baseline-time", str(baseline_time)],
                capture_output=True, text=True, timeout=timeout_s,
            )
        except Exception:
            pass
    return subprocess.run(base_cmd, capture_output=True, text=True, timeout=timeout_s)


_TIME_USED_RE = re.compile(r"^TIME_USED:\s*(-?\d+\.?\d*(?:[eE][+-]?\d+)?)\s*$", re.MULTILINE)


def selftest_comparator(
    *,
    comparator: Path,
    case_dir: Path,
    reference_file: Path,
    metric_name: str,
    timeout_s: int = 60,
    baseline_time: Optional[float] = None,
) -> Tuple[bool, str, Optional[float]]:
    """
    Run the authored comparator on the given case and check it produces
    `METRIC <name>: <number>` on stdout. We don't have an oracle value, but we
    do verify:
      (a) the script runs without crashing,
      (b) it emits a finite numeric value (not nan),
      (c) the metric name in output matches what we asked for.

    Returns (ok, reason, value).
    """
    if not comparator.is_file():
        return False, "comparator missing", None
    try:
        res = _run_comparator_with_optional_baseline_time(
            comparator=comparator, case_dir=case_dir, reference_file=reference_file,
            baseline_time=baseline_time, timeout_s=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return False, "timeout", None
    except Exception as exc:
        return False, f"exec error: {exc}", None
    blob = (res.stdout or "") + "\n" + (res.stderr or "")
    # Time-pinned scoring: parse TIME_USED diagnostic line. Refuse if 0 or below
    # tolerance (must match baseline_time within 1e-6 when caller pinned).
    tu_match = _TIME_USED_RE.search(blob)
    time_used: Optional[float] = None
    if tu_match:
        try:
            time_used = float(tu_match.group(1))
        except Exception:
            time_used = None
    if time_used is not None and time_used == 0.0:
        return False, "selftest TIME_USED=0 (comparator silently fell back to t=0)", None
    if (baseline_time is not None and time_used is not None
            and abs(time_used - float(baseline_time)) > 1e-3 * max(1.0, abs(float(baseline_time)))):
        # Soft tolerance: allow drift within 0.1% of baseline_time. Reject if comparator pinned the wrong dir.
        return False, (
            f"selftest TIME_USED={time_used} mismatches baseline_time={baseline_time}"
        ), None
    m = re.search(rf"METRIC\s+{re.escape(metric_name)}\s*:\s*(-?\d+\.?\d*(?:[eE][+-]?\d+)?|nan)",
                  blob, re.IGNORECASE)
    if not m:
        return False, f"no METRIC {metric_name} line in output", None
    val_s = m.group(1).lower()
    if val_s == "nan":
        return False, "metric value is nan", None
    try:
        val = float(val_s)
    except Exception:
        return False, f"unparseable value: {val_s}", None
    if res.returncode != 0:
        return False, f"non-zero exit ({res.returncode}); stderr: {(res.stderr or '')[:300]}", val
    return True, "ok", val


def _classify_selftest_failure(
    stdout: str,
    stderr: str,
    value: Optional[float],
    baseline_time: Optional[float],
    time_used: Optional[float],
    returncode: Optional[int] = None,
) -> str:
    """Categorise a comparator self-test failure for use in corrective re-prompts.

    Labels: 'parse_error', 'nan_value', 'wrong_time', 'exception', 'unknown'.
    """
    blob = (stdout or "") + "\n" + (stderr or "")
    if "PARSE_WARNING" in blob or "Unrecognised format" in blob:
        return "parse_error"
    # exception: non-zero rc and a Python traceback in stderr.
    if returncode is not None and returncode != 0 and ("Traceback (most recent call last)" in (stderr or "")):
        return "exception"
    # nan-valued result.
    import math
    if value is None:
        return "nan_value"
    try:
        if math.isnan(float(value)):
            return "nan_value"
    except Exception:
        return "nan_value"
    # wrong time: time_used differs significantly from baseline_time.
    if baseline_time is not None and time_used is not None:
        try:
            if time_used == 0.0:
                return "wrong_time"
            if float(time_used) < 0.5 * float(baseline_time):
                return "wrong_time"
            if abs(float(time_used) - float(baseline_time)) > 1e-3 * max(1.0, abs(float(baseline_time))):
                return "wrong_time"
        except Exception:
            pass
    if returncode is not None and returncode != 0:
        return "exception"
    return "unknown"


def _author_comparator_corrective(
    *,
    metric: Dict[str, Any],
    reference_file: Path,
    sample_pp_tree: str,
    sample_pp_data: str,
    flow_params: Dict[str, Any],
    out_path: Path,
    exemplar_text: str,
    baseline_final_time: Optional[float],
    prev_attempt: int,
    prev_source: str,
    selftest_blob: str,
    selftest_value: Optional[float],
    selftest_reason: str,
    failure_mode: str,
    use_pyvista: bool,
) -> Optional[Path]:
    """Re-author a comparator with a corrective prompt that includes the prior
    source, self-test output, and a failure-mode label.

    When use_pyvista=True, the system prompt switches to the PyVista variant.
    """
    metric_name = metric["name"]
    sys_msg = ""
    user_msg = ""
    try:
        from cfd_langgraph.config import get_settings  # type: ignore
        from cfd_langgraph.ideation import load_prompts  # type: ignore
        prompts = load_prompts(get_settings().prompts_path) or {}
        block = prompts.get("ComparatorAuthor", {}) or {}
        if use_pyvista:
            sys_msg = block.get("comparator_author_pyvista_system_prompt", "") or ""
        else:
            sys_msg = block.get("comparator_author_system_prompt", "") or ""
        user_tmpl = block.get("comparator_author_corrective_user_prompt", "") or ""
        if sys_msg and user_tmpl:
            user_msg = user_tmpl.format(
                prev_attempt=prev_attempt,
                failure_mode=failure_mode,
                selftest_value=("nan" if selftest_value is None else f"{selftest_value}"),
                selftest_reason=(selftest_reason or "")[:500],
                prev_source=(prev_source or "")[:6000],
                selftest_blob=(selftest_blob or "")[:2500],
                metric_name=metric_name,
                direction=metric.get("direction", "min"),
                description=metric.get("description", ""),
                data_source=metric.get("data_source", ""),
                ref_column=metric.get("ref_column", ""),
                computation_hint=metric.get("computation_hint", ""),
                reference_file=str(reference_file),
                reference_sample=(reference_file.read_text(errors="ignore")[:2000]
                                  if reference_file.is_file() else "(missing)"),
                sample_pp_tree=sample_pp_tree[:2000],
                sample_pp_data=sample_pp_data[:1500],
                flow_params=json.dumps(flow_params, ensure_ascii=False)[:1500],
                baseline_final_time=("(unknown)" if baseline_final_time is None
                                     else f"{baseline_final_time}"),
                exemplar_text=exemplar_text[:4000],
            )
    except Exception:
        sys_msg = ""
        user_msg = ""
    if not (sys_msg and user_msg):
        # Conservative fallback corrective prompt (still generic).
        sys_msg = (
            "You author a self-contained Python comparator script for an "
            "OpenFOAM metric. The previous attempt failed; produce a "
            "corrected version. Same I/O contract as before: CLI flags "
            "--case --reference --baseline-time --out, stdout lines "
            "'METRIC <name>: <value>' and 'TIME_USED: <float>', refuse "
            "time==0. " + ("Use PyVista (pv.OpenFOAMReader) instead of "
            "text parsing — touch <case>/case.foam if missing." if use_pyvista else
            "If the previous attempt read a postProcessing aggregate `.dat` "
            "for a spatial metric, switch to the boundary-field path under "
            "<time>/<fieldName>.")
        )
        user_msg = (
            f"PREV ATTEMPT #{prev_attempt} FAILED — failure_mode={failure_mode}\n"
            f"selftest_value={selftest_value}\nselftest_reason={selftest_reason}\n\n"
            f"PREVIOUS SOURCE:\n{(prev_source or '')[:6000]}\n\n"
            f"STDOUT/STDERR:\n{(selftest_blob or '')[:2500]}\n\n"
            f"METRIC NAME: {metric_name}\n"
            f"DESCRIPTION: {metric.get('description','')}\n"
            f"DATA SOURCE: {metric.get('data_source','')}\n"
            f"REF COLUMN: {metric.get('ref_column','')}\n"
            f"COMPUTATION HINT: {metric.get('computation_hint','')}\n\n"
            f"REFERENCE FILE: {reference_file}\n"
            f"BASELINE FINAL TIME: {baseline_final_time}\n"
        )
    try:
        raw = _llm_invoke([("system", sys_msg), ("user", user_msg)], temperature=0.0)
        code = _strip_code_fences(raw)
        if not code or "import" not in code:
            return None
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(code, encoding="utf-8")
        return out_path
    except TimeoutError:
        print(f"[OED-EXT][phase1] _author_comparator_corrective({metric_name}) timed out")
        raise
    except Exception as exc:
        print(f"[OED-EXT][phase1] _author_comparator_corrective({metric_name}) failed: {exc}")
        return None


def _selftest_comparator_full(
    *,
    comparator: Path,
    case_dir: Path,
    reference_file: Path,
    metric_name: str,
    timeout_s: int = 60,
    baseline_time: Optional[float] = None,
) -> Dict[str, Any]:
    """Like selftest_comparator but returns full diagnostics: ok, value,
    reason, time_used, stdout, stderr, returncode. Used by author_and_selftest.
    """
    if not comparator.is_file():
        return {"ok": False, "value": None, "reason": "comparator missing",
                "time_used": None, "stdout": "", "stderr": "", "returncode": None}
    try:
        res = _run_comparator_with_optional_baseline_time(
            comparator=comparator, case_dir=case_dir, reference_file=reference_file,
            baseline_time=baseline_time, timeout_s=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "value": None, "reason": "timeout",
                "time_used": None, "stdout": "", "stderr": "", "returncode": None}
    except Exception as exc:
        return {"ok": False, "value": None, "reason": f"exec error: {exc}",
                "time_used": None, "stdout": "", "stderr": "", "returncode": None}
    stdout = res.stdout or ""
    stderr = res.stderr or ""
    blob = stdout + "\n" + stderr
    tu_match = _TIME_USED_RE.search(blob)
    time_used: Optional[float] = None
    if tu_match:
        try:
            time_used = float(tu_match.group(1))
        except Exception:
            time_used = None
    m = re.search(rf"METRIC\s+{re.escape(metric_name)}\s*:\s*(-?\d+\.?\d*(?:[eE][+-]?\d+)?|nan)",
                  blob, re.IGNORECASE)
    value: Optional[float] = None
    reason = "ok"
    ok = True
    if not m:
        ok = False
        reason = f"no METRIC {metric_name} line in output"
    else:
        val_s = m.group(1).lower()
        if val_s == "nan":
            ok = False
            reason = "metric value is nan"
        else:
            try:
                value = float(val_s)
            except Exception:
                ok = False
                reason = f"unparseable value: {val_s}"
    if ok and time_used is not None and time_used == 0.0:
        ok = False
        reason = "selftest TIME_USED=0 (comparator silently fell back to t=0)"
    if ok and baseline_time is not None and time_used is not None:
        try:
            if abs(time_used - float(baseline_time)) > 1e-3 * max(1.0, abs(float(baseline_time))):
                ok = False
                reason = f"selftest TIME_USED={time_used} mismatches baseline_time={baseline_time}"
            elif time_used < 0.5 * float(baseline_time):
                ok = False
                reason = f"selftest TIME_USED={time_used} below 0.5*baseline_time={baseline_time}"
        except Exception:
            pass
    if ok and res.returncode != 0:
        ok = False
        reason = f"non-zero exit ({res.returncode}); stderr: {stderr[:300]}"
    return {"ok": ok, "value": value, "reason": reason, "time_used": time_used,
            "stdout": stdout, "stderr": stderr, "returncode": res.returncode}


# ---------------------------------------------------------------------------
# Verifier LLM (independently re-derives a metric to catch wrong-quantity bugs)
# ---------------------------------------------------------------------------

MAX_TEXT_ATTEMPTS = 5
MAX_PYVISTA_ATTEMPTS = 10
_VERIFIER_MAX_TURNS = 8
_VERIFIER_SCRIPT_TIMEOUT_S = 30


def _run_verifier_script_sandboxed(code: str, timeout_s: int = _VERIFIER_SCRIPT_TIMEOUT_S) -> Tuple[str, str, int]:
    """Run a small verifier-authored Python script in a temp directory with no
    network access (best-effort: rely on script-time-only allowlist). Returns
    (stdout, stderr, returncode). On exception returns ("", str(exc), -1).
    """
    import tempfile, os
    try:
        with tempfile.TemporaryDirectory(prefix="oed_verifier_") as td:
            script_path = Path(td) / "verifier_script.py"
            script_path.write_text(code, encoding="utf-8")
            env = os.environ.copy()
            env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
            try:
                res = subprocess.run(
                    [sys.executable, str(script_path)],
                    capture_output=True, text=True, timeout=timeout_s,
                    cwd=td, env=env,
                )
                return (res.stdout or ""), (res.stderr or ""), int(res.returncode)
            except subprocess.TimeoutExpired as exc:
                return "", f"verifier script timeout after {timeout_s}s: {exc}", -1
    except Exception as exc:
        return "", f"verifier sandbox error: {exc}", -1


def _parse_verifier_response(raw: str) -> Tuple[str, Dict[str, Any]]:
    """Parse a verifier turn output. Returns (kind, payload) where kind is
    'tool', 'verdict', or 'unknown'. Tool payload is {'code': str}. Verdict
    payload is the verdict dict.
    """
    body = _strip_code_fences(raw or "")
    # Find first { ... matching }
    i = body.find("{")
    if i < 0:
        return "unknown", {"raw": body}
    depth = 0
    end = -1
    in_str = False
    esc = False
    for j in range(i, len(body)):
        ch = body[j]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = j + 1
                    break
    if end < 0:
        return "unknown", {"raw": body}
    try:
        obj = json.loads(body[i:end])
    except Exception:
        return "unknown", {"raw": body[i:end]}
    if isinstance(obj, dict) and obj.get("tool") == "python_script" and isinstance(obj.get("code"), str):
        return "tool", {"code": obj["code"]}
    if isinstance(obj, dict) and "verdict" in obj:
        return "verdict", obj
    return "unknown", {"raw": obj}


def _verify_comparator_llm(
    *,
    metric: Dict[str, Any],
    comparator_path: Path,
    comparator_stdout: str,
    comparator_value: Optional[float],
    baseline_case_dir: Path,
    reference_path: Path,
) -> Dict[str, Any]:
    """Independently verify a comparator's output using a verifier LLM with
    up to 3 python_script tool turns. Returns a verdict dict. On verifier
    failure (timeouts, parse errors, no schema), returns a synthetic
    SUSPICIOUS/cannot_verify dict (so the caller can still bind).
    """
    metric_name = metric.get("name", "")
    try:
        comparator_source = comparator_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        comparator_source = ""

    sys_msg = ""
    user_tmpl = ""
    try:
        from cfd_langgraph.config import get_settings  # type: ignore
        from cfd_langgraph.ideation import load_prompts  # type: ignore
        prompts = load_prompts(get_settings().prompts_path) or {}
        block = prompts.get("ComparatorVerifier", {}) or {}
        sys_msg = block.get("comparator_verifier_system_prompt", "") or ""
        user_tmpl = block.get("comparator_verifier_user_prompt", "") or ""
    except Exception:
        sys_msg = ""
        user_tmpl = ""
    if not (sys_msg and user_tmpl):
        # Verifier prompts unavailable — cannot verify; do not block.
        return {
            "verdict": "SUSPICIOUS",
            "comparator_value": comparator_value,
            "independent_estimate": None,
            "discrepancy_class": "cannot_verify",
            "rationale": "verifier prompts not loadable; skipped verification",
            "corrective_hint_for_author": "",
        }

    turn_log: List[Dict[str, str]] = []

    def _render_turn_history() -> str:
        if not turn_log:
            return "(no prior tool turns)"
        chunks = []
        for i, t in enumerate(turn_log, 1):
            chunks.append(
                f"--- turn {i} ---\nMODEL TOOL CALL CODE:\n{t.get('code','')[:2000]}\n"
                f"STDOUT:\n{t.get('stdout','')[:1500]}\n"
                f"STDERR:\n{t.get('stderr','')[:800]}\n"
                f"RETURNCODE: {t.get('returncode','')}"
            )
        return "\n\n".join(chunks)

    # Build a ground-truth block from any named numeric expectations the
    # proposer left in description / computation_hint / ref_column. The
    # verifier MUST treat these as authoritative reference values.
    _gt_pairs = _extract_all_expectations(metric)
    if _gt_pairs:
        _gt_lines = [
            f"  - {lbl}: {val}" for lbl, val in _gt_pairs
        ]
        ground_truth_block = (
            "The metric spec names the following expected numeric values. "
            "Treat these as authoritative ground truth. The comparator's "
            "value MUST match the relevant one (typically the baseline-error "
            "value) within ~20% on the baseline case. If the comparator's "
            "value differs from a relevant expectation by >20%, return "
            "verdict=WRONG with discrepancy_class=off_by_factor or "
            "wrong_window/wrong_sign_change_pair as appropriate.\n"
            + "\n".join(_gt_lines)
        )
    else:
        ground_truth_block = (
            "(no named numeric expectations found in metric spec; verify by "
            "independent re-derivation only)"
        )

    for turn_idx in range(1, _VERIFIER_MAX_TURNS + 1):
        try:
            try:
                user_msg = user_tmpl.format(
                    metric_name=metric_name,
                    metric_description=metric.get("description", ""),
                    data_source=metric.get("data_source", ""),
                    ref_column=metric.get("ref_column", ""),
                    computation_hint=metric.get("computation_hint", ""),
                    comparator_source=(comparator_source or "")[:8000],
                    comparator_stdout=(comparator_stdout or "")[:3000],
                    baseline_case_dir=str(baseline_case_dir),
                    reference_path=str(reference_path),
                    turn_history=_render_turn_history(),
                    ground_truth_block=ground_truth_block,
                )
            except KeyError:
                # Older YAML lacks the {ground_truth_block} placeholder; append it.
                user_msg = user_tmpl.format(
                    metric_name=metric_name,
                    metric_description=metric.get("description", ""),
                    data_source=metric.get("data_source", ""),
                    ref_column=metric.get("ref_column", ""),
                    computation_hint=metric.get("computation_hint", ""),
                    comparator_source=(comparator_source or "")[:8000],
                    comparator_stdout=(comparator_stdout or "")[:3000],
                    baseline_case_dir=str(baseline_case_dir),
                    reference_path=str(reference_path),
                    turn_history=_render_turn_history(),
                ) + "\n\n=== AUTHORITATIVE GROUND TRUTH ===\n" + ground_truth_block
        except Exception as exc:
            return {
                "verdict": "SUSPICIOUS",
                "comparator_value": comparator_value,
                "independent_estimate": None,
                "discrepancy_class": "cannot_verify",
                "rationale": f"verifier prompt format error: {exc}",
                "corrective_hint_for_author": "",
            }

        try:
            raw = _llm_invoke([("system", sys_msg), ("user", user_msg)], temperature=0.0)
        except TimeoutError as exc:
            return {
                "verdict": "SUSPICIOUS",
                "comparator_value": comparator_value,
                "independent_estimate": None,
                "discrepancy_class": "cannot_verify",
                "rationale": f"verifier LLM timeout: {exc}",
                "corrective_hint_for_author": "",
            }
        except Exception as exc:
            return {
                "verdict": "SUSPICIOUS",
                "comparator_value": comparator_value,
                "independent_estimate": None,
                "discrepancy_class": "cannot_verify",
                "rationale": f"verifier LLM error: {exc}",
                "corrective_hint_for_author": "",
            }

        kind, payload = _parse_verifier_response(raw)
        if kind == "verdict":
            verdict = str(payload.get("verdict", "")).upper()
            if verdict not in {"OK", "SUSPICIOUS", "WRONG"}:
                verdict = "SUSPICIOUS"
            payload["verdict"] = verdict
            payload.setdefault("comparator_value", comparator_value)
            payload.setdefault("independent_estimate", None)
            payload.setdefault("discrepancy_class", "cannot_verify" if verdict == "SUSPICIOUS" else "ok")
            payload.setdefault("rationale", "")
            payload.setdefault("corrective_hint_for_author", "")
            return payload
        if kind == "tool" and turn_idx < _VERIFIER_MAX_TURNS:
            code = payload.get("code", "")
            stdout, stderr, rc = _run_verifier_script_sandboxed(code)
            turn_log.append({"code": code, "stdout": stdout, "stderr": stderr,
                             "returncode": str(rc)})
            continue
        # 'unknown' or tool-call on final turn -> force a verdict next turn
        if kind == "tool":
            # Final turn used for tool; we can't re-prompt, so log and force.
            code = payload.get("code", "")
            stdout, stderr, rc = _run_verifier_script_sandboxed(code)
            turn_log.append({"code": code, "stdout": stdout, "stderr": stderr,
                             "returncode": str(rc)})
        # otherwise unknown: fall through and break
        break

    # No verdict after max turns: SUSPICIOUS / cannot_verify.
    return {
        "verdict": "SUSPICIOUS",
        "comparator_value": comparator_value,
        "independent_estimate": None,
        "discrepancy_class": "cannot_verify",
        "rationale": f"verifier did not produce a verdict within {_VERIFIER_MAX_TURNS} turns",
        "corrective_hint_for_author": "",
    }


def author_and_selftest(
    metric: Dict[str, Any],
    *,
    baseline_case_dir: Path,
    out_dir: Path,
    starter_dir: Optional[Path] = None,
    reference_path: Optional[Path] = None,
    baseline_final_time: Optional[float] = None,
    settings: Any = None,  # cfd_langgraph settings (unused; kept for signature compat)
    max_attempts: int = 10,
    sample_pp_tree: str = "",
    sample_pp_data: str = "",
    flow_params: Optional[Dict[str, Any]] = None,
    exemplar_text: str = "",
    pyvista_after_attempt: int = 5,
) -> Dict[str, Any]:
    """Author + self-test a comparator with up to ``max_attempts`` retries.

    Loop logic:
      - Attempt 1: standard ``author_comparator`` (text-parsing, generic
        prompt).
      - Attempt 2..N: re-author with a corrective system message that
        includes the previous source, self-test stdout/stderr, and a
        ``failure_mode`` label.
      - After ``pyvista_after_attempt`` failed text-parsing attempts, switch
        to the PyVista variant.

    Only one comparator file is kept on disk per metric (overwritten each
    attempt). The full attempt history is persisted to
    ``<out_dir>/<metric>_attempt_log.json`` for debugging.

    Returns a dict shape-compatible with the existing ``bound_comparators``
    consumers, plus the new keys ``attempts`` and ``final_method``:
        {path, origin, selftest_ok, selftest_value, selftest_reason,
         attempts, final_method}.
    """
    flow_params = flow_params or {}
    metric_name = metric["name"]
    safe = re.sub(r"[^A-Za-z0-9_]+", "_", metric_name)
    cand_path = out_dir / f"compare_{safe}.py"
    log_path = out_dir / f"{safe}_attempt_log.json"
    out_dir.mkdir(parents=True, exist_ok=True)

    reference_file = reference_path if reference_path is not None else (
        (starter_dir / "reference.csv") if starter_dir else Path("reference.csv")
    )

    # Per-method caps and routing based on preferred_method (default 'auto'
    # for backwards compat with old metric_specs.json without that field).
    preferred_method = str(metric.get("preferred_method", "auto")).lower().strip() or "auto"
    if preferred_method not in {"text", "pyvista", "auto"}:
        preferred_method = "auto"
    max_text_attempts = MAX_TEXT_ATTEMPTS
    max_pyvista_attempts = MAX_PYVISTA_ATTEMPTS
    if preferred_method == "pyvista":
        # Skip text entirely.
        text_budget = 0
        pyvista_budget = max_pyvista_attempts
        current_method = "pyvista"
    else:
        # 'text' or 'auto' -> try text first, fall back.
        text_budget = max_text_attempts
        pyvista_budget = max_pyvista_attempts
        current_method = "text"

    attempt_history: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None  # best (ok or finite) attempt seen
    final_method = current_method

    prev_source = ""
    prev_blob = ""
    prev_value: Optional[float] = None
    prev_reason = ""
    prev_failure_mode = "unknown"

    text_used = 0
    pyvista_used = 0
    attempt = 0
    first_attempt_in_method = True  # first attempt in current method uses non-corrective author

    while True:
        # Decide method/budget. If text exhausted and we were on text, switch to pyvista.
        if current_method == "text" and text_used >= text_budget:
            current_method = "pyvista"
            first_attempt_in_method = True
        if current_method == "pyvista" and pyvista_used >= pyvista_budget:
            break  # all budgets exhausted
        if current_method == "text" and text_budget == 0:
            current_method = "pyvista"
            first_attempt_in_method = True
            if pyvista_used >= pyvista_budget:
                break

        attempt += 1
        use_pyvista = (current_method == "pyvista")
        final_method = current_method

        # Author or re-author.
        try:
            if first_attempt_in_method and not use_pyvista:
                # First text attempt: standard non-corrective authoring.
                authored = author_comparator(
                    metric=metric, reference_file=reference_file,
                    sample_pp_tree=sample_pp_tree, sample_pp_data=sample_pp_data,
                    flow_params=flow_params, out_path=cand_path,
                    exemplar_text=exemplar_text,
                    baseline_final_time=baseline_final_time,
                )
            else:
                # Corrective re-author (also used for first PyVista attempt
                # so the prompt explains the switch from text-parsing).
                authored = _author_comparator_corrective(
                    metric=metric, reference_file=reference_file,
                    sample_pp_tree=sample_pp_tree, sample_pp_data=sample_pp_data,
                    flow_params=flow_params, out_path=cand_path,
                    exemplar_text=exemplar_text,
                    baseline_final_time=baseline_final_time,
                    prev_attempt=attempt - 1,
                    prev_source=prev_source,
                    selftest_blob=prev_blob,
                    selftest_value=prev_value,
                    selftest_reason=prev_reason,
                    failure_mode=prev_failure_mode,
                    use_pyvista=use_pyvista,
                )
        except TimeoutError as exc:
            attempt_history.append({
                "attempt": attempt, "method": current_method,
                "authoring_ok": False, "selftest_ok": False, "selftest_value": None,
                "selftest_reason": f"LLM timeout: {exc}",
                "failure_mode": "timeout",
                "verifier_verdict": None,
            })
            if use_pyvista:
                pyvista_used += 1
            else:
                text_used += 1
            prev_failure_mode = "timeout"
            prev_reason = f"LLM timeout: {exc}"
            prev_blob = ""
            prev_value = None
            prev_source = ""
            first_attempt_in_method = False
            continue

        if authored is None:
            attempt_history.append({
                "attempt": attempt, "method": current_method,
                "authoring_ok": False, "selftest_ok": False, "selftest_value": None,
                "selftest_reason": "authoring failed",
                "failure_mode": "exception",
                "verifier_verdict": None,
            })
            if use_pyvista:
                pyvista_used += 1
            else:
                text_used += 1
            prev_failure_mode = "exception"
            prev_reason = "authoring failed"
            prev_blob = ""
            prev_value = None
            prev_source = ""
            first_attempt_in_method = False
            continue

        try:
            prev_source = cand_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            prev_source = ""

        st = _selftest_comparator_full(
            comparator=cand_path, case_dir=baseline_case_dir,
            reference_file=reference_file, metric_name=metric_name,
            baseline_time=baseline_final_time,
        )
        failure_mode = "ok" if st["ok"] else _classify_selftest_failure(
            stdout=st["stdout"], stderr=st["stderr"], value=st["value"],
            baseline_time=baseline_final_time, time_used=st["time_used"],
            returncode=st["returncode"],
        )

        rec: Dict[str, Any] = {
            "attempt": attempt,
            "method": current_method,
            "authoring_ok": True,
            "selftest_ok": bool(st["ok"]),
            "selftest_value": st["value"],
            "selftest_reason": st["reason"],
            "time_used": st["time_used"],
            "failure_mode": failure_mode,
            "verifier_verdict": None,
            "verifier_hint": None,
        }

        verifier_verdict: Optional[str] = None
        verifier_hint: Optional[str] = None

        if st["ok"]:
            # Verifier LLM step: independently re-derive the metric.
            try:
                vres = _verify_comparator_llm(
                    metric=metric, comparator_path=cand_path,
                    comparator_stdout=(st["stdout"] or "") + "\n" + (st["stderr"] or ""),
                    comparator_value=st["value"],
                    baseline_case_dir=baseline_case_dir,
                    reference_path=reference_file,
                )
            except Exception as exc:  # belt-and-braces; verifier itself shouldn't raise
                vres = {
                    "verdict": "SUSPICIOUS",
                    "comparator_value": st["value"],
                    "independent_estimate": None,
                    "discrepancy_class": "cannot_verify",
                    "rationale": f"verifier raised: {exc}",
                    "corrective_hint_for_author": "",
                }
            verifier_verdict = str(vres.get("verdict", "SUSPICIOUS")).upper()
            verifier_hint = str(vres.get("corrective_hint_for_author", "") or "")
            rec["verifier_verdict"] = verifier_verdict
            rec["verifier_hint"] = verifier_hint
            rec["verifier_rationale"] = vres.get("rationale", "")
            rec["verifier_independent_estimate"] = vres.get("independent_estimate")
            rec["verifier_discrepancy_class"] = vres.get("discrepancy_class", "")

        attempt_history.append(rec)

        # Increment per-method counter (verifier WRONG also counts).
        if use_pyvista:
            pyvista_used += 1
        else:
            text_used += 1

        # Track best attempt.
        if best is None:
            best = rec
        elif st["ok"] and not best.get("selftest_ok"):
            best = rec
        elif (not best.get("selftest_ok")) and st["value"] is not None and best.get("selftest_value") is None:
            best = rec

        if st["ok"]:
            if verifier_verdict == "OK":
                _write_json(log_path, {
                    "metric": metric_name,
                    "preferred_method": preferred_method,
                    "max_text_attempts": max_text_attempts,
                    "max_pyvista_attempts": max_pyvista_attempts,
                    "attempts": attempt_history,
                })
                return {
                    "path": str(cand_path),
                    "origin": "authored",
                    "selftest_ok": True,
                    "selftest_value": st["value"],
                    "selftest_reason": st["reason"],
                    "attempts": attempt,
                    "final_method": current_method,
                    "verifier_verdict": "OK",
                    "preferred_method": preferred_method,
                }
            if verifier_verdict == "SUSPICIOUS":
                # Don't bind on SUSPICIOUS — verifier couldn't validate, which
                # historically masked wrong-physics-window comparators. Re-author
                # with a corrective hint pointing at the most common failure
                # modes so the next attempt is more likely to be checkable.
                failure_mode = "verifier_suspicious"
                rec["failure_mode"] = failure_mode
                prev_failure_mode = failure_mode
                susp_hint = (
                    verifier_hint
                    or "the verifier could not independently confirm your value. "
                       "Make the comparator's selection rule explicit in code "
                       "(window bounds for any sign-change search, first vs last "
                       "selector, exact patch / field / time used) and emit "
                       "PARSE_WARNING lines stating each chosen value, so a "
                       "re-derivation can match yours."
                )
                prev_reason = (
                    f"verifier returned SUSPICIOUS / cannot_verify "
                    f"({vres.get('rationale','')}). Hint: {susp_hint}"
                )
                prev_blob = ((st["stdout"] or "") + "\n" + (st["stderr"] or "") +
                             f"\nVERIFIER_HINT: {susp_hint}\n"
                             f"VERIFIER_RATIONALE: {vres.get('rationale','')}\n"
                             f"VERIFIER_DISCREPANCY: {vres.get('discrepancy_class','')}")
                prev_value = st["value"]
                first_attempt_in_method = False
                continue
            # WRONG: re-author with corrective hint folded into the failure
            # message. Treat as a failed attempt; loop continues.
            failure_mode = "verifier_flagged"
            rec["failure_mode"] = failure_mode
            prev_failure_mode = failure_mode
            prev_reason = (
                f"verifier flagged WRONG: {vres.get('rationale','')}. "
                f"Hint: {verifier_hint}"
            )
            prev_blob = ((st["stdout"] or "") + "\n" + (st["stderr"] or "") +
                         f"\nVERIFIER_HINT: {verifier_hint}\n"
                         f"VERIFIER_RATIONALE: {vres.get('rationale','')}\n"
                         f"VERIFIER_DISCREPANCY: {vres.get('discrepancy_class','')}")
            prev_value = st["value"]
            first_attempt_in_method = False
            continue

        # Not OK -> selftest failure path.
        prev_blob = (st["stdout"] or "") + "\n" + (st["stderr"] or "")
        prev_value = st["value"]
        prev_reason = st["reason"]
        prev_failure_mode = failure_mode
        first_attempt_in_method = False

    # All budgets exhausted; persist log and return the best (or last) record.
    _write_json(log_path, {
        "metric": metric_name,
        "preferred_method": preferred_method,
        "max_text_attempts": max_text_attempts,
        "max_pyvista_attempts": max_pyvista_attempts,
        "attempts": attempt_history,
    })
    last = attempt_history[-1] if attempt_history else {}
    # Pick the best selftest-passing attempt across the whole history (verifier
    # may have kept saying SUSPICIOUS but the comparator itself ran cleanly).
    selftest_passers = [r for r in attempt_history if r.get("selftest_ok")]
    if selftest_passers:
        chosen = selftest_passers[-1]
        return {
            "path": chosen.get("path") or (str(cand_path) if cand_path.is_file() else ""),
            "origin": "authored",
            "selftest_ok": True,
            "selftest_value": chosen.get("selftest_value"),
            "selftest_reason": chosen.get("selftest_reason", "ok"),
            "attempts": len(attempt_history),
            "final_method": chosen.get("method", final_method),
            "verifier_verdict": chosen.get("verifier_verdict") or "SUSPICIOUS",
            "verifier_warning": (
                "verifier never returned OK across all attempts; binding "
                "best-effort selftest-passing comparator. Metric values may be "
                "physically incorrect — review verifier_rationale and consider "
                "re-running with a tighter computation_hint."
            ),
            "preferred_method": preferred_method,
        }
    return {
        "path": str(cand_path) if cand_path.is_file() else "",
        "origin": "authored",
        "selftest_ok": False,
        "selftest_value": (best or last).get("selftest_value"),
        "selftest_reason": (best or last).get("selftest_reason", "all attempts failed"),
        "attempts": len(attempt_history),
        "final_method": (best or last).get("method", final_method),
        "preferred_method": preferred_method,
    }


def resolve_metric_comparators(
    *,
    metrics: List[Dict[str, Any]],
    search_roots: List[Path],
    reference_file: Path,
    flow_params: Dict[str, Any],
    baseline_case_dir: Optional[Path],
    out_dir: Path,
    sample_pp_tree: str = "",
    sample_pp_data: str = "",
    exemplar_text: str = "",
    baseline_final_time: Optional[float] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    For each metric:
      - find an existing comparator if available;
      - else have LLM author one and self-test it against the baseline case.
    Returns {metric_name: {"path": str, "origin": "existing|authored",
                            "selftest_ok": bool, "selftest_value": float|None}}.
    """
    existing = discover_existing_comparators(search_roots=search_roots, metrics=metrics)
    bound: Dict[str, Dict[str, Any]] = {}
    for m in metrics:
        name = m["name"]
        # Try existing comparator first — but ALWAYS self-test before trusting.
        if name in existing and baseline_case_dir is not None:
            ep = existing[name]
            ok, reason, val = selftest_comparator(
                comparator=Path(ep), case_dir=baseline_case_dir,
                reference_file=reference_file, metric_name=name,
                baseline_time=baseline_final_time,
            )
            if ok:
                bound[name] = {"path": ep, "origin": "existing",
                               "selftest_ok": True, "selftest_value": val,
                               "selftest_reason": reason}
                continue
            # Existing comparator did not produce the METRIC line for this
            # metric. Fall through to LLM authoring (it might compute a
            # different metric, or be too case-specific).
            print(f"[OED-EXT][phase1] existing comparator at {ep} failed selftest "
                  f"for metric {name}: {reason}. Authoring a fresh one.")

        # Author + self-test (with retry loop and PyVista fallback).
        if baseline_case_dir is None:
            cand = out_dir / f"compare_{re.sub(r'[^A-Za-z0-9_]+','_', name)}.py"
            authored = author_comparator(
                metric=m, reference_file=reference_file,
                sample_pp_tree=sample_pp_tree, sample_pp_data=sample_pp_data,
                flow_params=flow_params, out_path=cand,
                exemplar_text=exemplar_text,
                baseline_final_time=baseline_final_time,
            )
            if authored is None:
                bound[name] = {"path": "", "origin": "authored",
                               "selftest_ok": False, "selftest_value": None,
                               "selftest_reason": "authoring failed",
                               "attempts": 1, "final_method": "text"}
            else:
                bound[name] = {"path": str(authored), "origin": "authored",
                               "selftest_ok": False, "selftest_value": None,
                               "selftest_reason": "no baseline case for self-test",
                               "attempts": 1, "final_method": "text"}
            continue
        result = author_and_selftest(
            m,
            baseline_case_dir=baseline_case_dir,
            out_dir=out_dir,
            starter_dir=None,
            reference_path=reference_file,
            baseline_final_time=baseline_final_time,
            settings=None,
            max_attempts=10,
            sample_pp_tree=sample_pp_tree,
            sample_pp_data=sample_pp_data,
            flow_params=flow_params,
            exemplar_text=exemplar_text,
            pyvista_after_attempt=5,
        )
        bound[name] = result
        if not result.get("selftest_ok"):
            print(f"[OED-EXT][phase1] all selftest attempts failed for {name} "
                  f"(attempts={result.get('attempts')}, method={result.get('final_method')}): "
                  f"{result.get('selftest_reason')}")
    return bound


def compute_metric_vector(
    *,
    case_dir: Path,
    bound_comparators: Dict[str, Dict[str, Any]],
    reference_file: Path,
    timeout_s: int = 90,
    baseline_final_time: Optional[float] = None,
    metric_specs: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Run all bound comparators against case_dir. Returns:
      {"metrics": {name: value, ...}, "raw_outputs": {name: blob}, "errors": {name: reason}}
    Skips comparators with selftest_ok=False unless their path is set
    (they may still work on a different case).
    """
    metrics: Dict[str, float] = {}
    raw_outputs: Dict[str, str] = {}
    errors: Dict[str, str] = {}
    # Resolve baseline_final_time per metric: prefer explicit kwarg, else look
    # for it on each spec.
    spec_by_name: Dict[str, Dict[str, Any]] = {}
    for s in (metric_specs or []):
        if isinstance(s, dict) and s.get("name"):
            spec_by_name[str(s["name"])] = s
    for name, info in bound_comparators.items():
        path = info.get("path", "")
        if not path:
            errors[name] = "no comparator path"
            continue
        sp = Path(path)
        if not sp.is_file():
            errors[name] = "comparator missing"
            continue
        # Skip comparators that failed self-test — they won't produce a
        # trustworthy METRIC line. The judge already handles missing metrics
        # gracefully (returns INDETERMINATE if vector is too sparse).
        if info.get("selftest_ok") is False:
            errors[name] = f"selftest failed: {info.get('selftest_reason', '')[:120]}"
            continue
        # Per-metric baseline_final_time (kwarg wins; else spec field).
        bt = baseline_final_time
        if bt is None:
            sp_spec = spec_by_name.get(name) or {}
            try:
                if sp_spec.get("baseline_final_time") is not None:
                    bt = float(sp_spec["baseline_final_time"])
            except Exception:
                bt = None
        try:
            res = _run_comparator_with_optional_baseline_time(
                comparator=sp, case_dir=case_dir, reference_file=reference_file,
                baseline_time=bt, timeout_s=timeout_s,
            )
        except Exception as exc:
            errors[name] = f"exec: {exc}"
            continue
        blob = (res.stdout or "") + "\n" + (res.stderr or "")
        raw_outputs[name] = blob[-2000:]
        m = re.search(rf"METRIC\s+{re.escape(name)}\s*:\s*(-?\d+\.?\d*(?:[eE][+-]?\d+)?|nan)",
                      blob, re.IGNORECASE)
        if not m:
            # also accept any RMSE-like value if metric name not found
            m2 = _METRIC_NUMBER_RE.search(blob)
            if m2:
                try:
                    metrics[name] = float(m2.group(2))
                    continue
                except Exception:
                    pass
            errors[name] = "no METRIC line"
            continue
        val_s = m.group(1).lower()
        if val_s == "nan":
            errors[name] = "nan"
            continue
        try:
            metrics[name] = float(val_s)
        except Exception:
            errors[name] = f"unparseable: {val_s}"
    return {"metrics": metrics, "raw_outputs": raw_outputs, "errors": errors}


def render_metric_vector_for_prompt(
    *, metric_vector: Dict[str, Any], baseline_vector: Optional[Dict[str, Any]] = None,
    metric_specs: Optional[List[Dict[str, Any]]] = None
) -> str:
    """Pretty-print metric comparison for inclusion in LLM decision prompt."""
    metrics = metric_vector.get("metrics", {})
    base = (baseline_vector or {}).get("metrics", {})
    dirs = {m["name"]: m.get("direction", "min") for m in (metric_specs or [])}
    if not metrics:
        return "(no metrics computed)"
    lines = ["Metric vector (current candidate vs baseline):"]
    for name, val in metrics.items():
        b = base.get(name)
        d = dirs.get(name, "min")
        delta = ""
        if isinstance(b, (int, float)) and b != 0:
            pct = 100.0 * (val - b) / abs(b)
            improving = (val < b) if d == "min" else (val > b)
            arrow = "↓" if val < b else ("↑" if val > b else "=")
            tag = " (better)" if improving else (" (worse)" if val != b else "")
            delta = f"  Δ={pct:+.2f}% {arrow}{tag}"
        elif isinstance(b, (int, float)):
            delta = f"  Δ_abs={val - b:+.4g}"
        lines.append(f"  {name:24s} = {val:.6g}  baseline={b!r}  dir={d}{delta}")
    errs = metric_vector.get("errors", {})
    if errs:
        lines.append("Metric errors (could not compute):")
        for k, v in errs.items():
            lines.append(f"  {k}: {v}")
    return "\n".join(lines)


def llm_judge_iteration(
    *,
    topic: str,
    model_description: str,
    metric_vector: Dict[str, Any],
    baseline_vector: Dict[str, Any],
    metric_specs: List[Dict[str, Any]],
    history_summary: str = "",
    best_so_far: Optional[Dict[str, Any]] = None,
    extra_context: str = "",
    # Full-context fields — give the judge everything it needs to make a real call.
    formula_or_modification: str = "",
    modification_category: str = "",
    parameters: Optional[Dict[str, Any]] = None,
    propose_rationale: str = "",
    compiled_model_name: str = "",
    flow_parameters: Optional[Dict[str, Any]] = None,
    reference_description: str = "",
    interpreter_summary: str = "",
    run_log_excerpt: str = "",
    iteration: Optional[int] = None,
    budget_total: Optional[int] = None,
    budget_used: Optional[int] = None,
    prior_proceed_summaries: Optional[List[Dict[str, Any]]] = None,
    diversity_mode: str = "off",
    families_seen_list: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    LLM-as-judge: holistic evaluation of one candidate's metric vector against
    baseline. Replaces fixed aggregators (weighted_sum / min_improvement /
    pareto_rank). The LLM sees the full vector + baseline + history + topic
    and decides PROCEED / REVISE / RERUN with reasoning, plus emits a
    synthesized primary_score (lower = better) for ranking purposes.

    Returns:
      {"decision": "PROCEED" | "REVISE" | "RERUN" | "INDETERMINATE",
       "is_improvement_over_baseline": bool,
       "is_best_so_far": bool,
       "primary_score": float,                # lower=better, LLM-synthesized
       "rationale": str,
       "strengths": [str, ...],               # metrics that improved
       "weaknesses": [str, ...],              # metrics that regressed
       "method": "llm_judge"}
    """
    metrics = metric_vector.get("metrics", {}) or {}
    base = baseline_vector.get("metrics", {}) if baseline_vector else {}
    if not metrics:
        return {"decision": "INDETERMINATE", "is_improvement_over_baseline": False,
                "is_best_so_far": False, "primary_score": None,
                "rationale": "no metrics computed", "strengths": [], "weaknesses": [],
                "method": "llm_judge"}

    sys_msg = (
        "You are a CFD evaluator. Given a candidate model's metric vector, the "
        "baseline's metric vector, and the study topic/history, decide whether "
        "the candidate is a real, holistic improvement worth keeping.\n\n"
        "RULES:\n"
        "- A candidate is an improvement only if it materially advances the "
        "study goal across the relevant metrics — gaming one metric while "
        "regressing others is NOT improvement.\n"
        "- 'PROCEED' = candidate is a holistic win or a clearly promising "
        "direction worth pursuing further.\n"
        "- 'REVISE'  = the model class shows promise but the specific variant "
        "needs adjustment (parameter or structural tweak suggested in rationale).\n"
        "- 'RERUN'   = numerical / convergence issue, NOT physics — re-run with "
        "tighter timestep / mesh / scheme.\n"
        "- 'INDETERMINATE' = data insufficient to decide.\n"
        "- Set 'is_improvement_over_baseline' true ONLY if the candidate is "
        "better than baseline considering ALL metrics holistically (not just "
        "majority or weighted sum).\n"
        "- Set 'is_best_so_far' true ONLY if the candidate is also better than "
        "every prior PROCEED in history on the relevant metrics.\n"
        "- 'primary_score': synthesize a single number (lower = better) that "
        "reflects holistic candidate quality. Use baseline's primary metric "
        "magnitude as a rough scale anchor. This score is used for ranking; "
        "be self-consistent across calls.\n"
        "- METRIC SEMANTICS. Each metric has a `direction` field. "
        "`direction=min` means smaller-is-better; `direction=max` means "
        "larger-is-better. When a metric also has a `target_value`, the "
        "metrics table will show `target=X` and `|Δ_target|` (the absolute "
        "distance from the candidate's value to the target). For metrics "
        "with a target, 'better' means moving |Δ_target| toward zero — a "
        "candidate that overshoots the target on the other side is NOT "
        "necessarily improving even if its raw value moved a lot from "
        "baseline. Always reason about |Δ_target| when a target is shown.\n"
        "- Return STRICT JSON only. No prose."
    )

    spec_lines = []
    # Build a target-value lookup keyed by metric name.
    # Topic-agnostic: we accept either `target_value` (preferred) or the
    # legacy `dns_truth_value` from older spec sets.
    target_lookup: Dict[str, Any] = {}
    for m in metric_specs:
        nm = m.get("name", "")
        tv = m.get("target_value")
        if tv is None:
            tv = m.get("dns_truth_value")
        if isinstance(tv, (int, float)):
            target_lookup[nm] = float(tv)

    for m in metric_specs:
        nm = m.get("name", "")
        line = (
            f"  {nm}: dir={m.get('direction','min')}; "
            f"{m.get('description','')}"
        )
        if nm in target_lookup:
            line += f"  [target_value={target_lookup[nm]:.6g}]"
        spec_lines.append(line)
    metrics_table = []
    for name, val in metrics.items():
        b = base.get(name)
        tgt = target_lookup.get(name)
        extras: List[str] = []
        if isinstance(b, (int, float)) and b != 0:
            pct = 100.0 * (val - b) / abs(b)
            extras.append(f"Δ={pct:+.2f}% vs baseline {b:.6g}")
        if isinstance(tgt, (int, float)):
            extras.append(f"target={tgt:.6g}  |Δ_target|={abs(val - tgt):.6g}")
        line = f"  {name} = {val:.6g}"
        if extras:
            line += "   " + "   ".join(extras)
        metrics_table.append(line)

    best_block = ""
    if best_so_far:
        bm = best_so_far.get("metrics", {}) or {}
        best_block = "BEST PROCEED SO FAR:\n" + "\n".join(
            f"  {k} = {v:.6g}" for k, v in bm.items()
        ) + "\n"

    # Build a rich context block. Every field is included only if non-empty so
    # the prompt stays clean for short-context cases.
    context_lines: List[str] = []
    context_lines.append(f"TOPIC:\n{topic}")
    if iteration is not None:
        budget_str = ""
        if budget_total is not None and budget_used is not None:
            budget_str = f" (budget {budget_used}/{budget_total} used)"
        context_lines.append(f"ITERATION: {iteration}{budget_str}")
    if reference_description:
        context_lines.append(f"REFERENCE DATA DESCRIPTION:\n{reference_description[:1500]}")
    if flow_parameters:
        context_lines.append(
            "FLOW PARAMETERS (authoritative; use for interpretation):\n  "
            + json.dumps(flow_parameters, ensure_ascii=False)[:1200]
        )
    context_lines.append(f"CANDIDATE MODEL DESCRIPTION:\n{model_description}")
    if compiled_model_name:
        context_lines.append(f"COMPILED CLASS / LIBRARY: {compiled_model_name}")
    if modification_category:
        context_lines.append(f"MODIFICATION CATEGORY: {modification_category}")
    if formula_or_modification:
        context_lines.append(
            f"MODIFICATION FORMULA / SPEC:\n{str(formula_or_modification)[:2000]}"
        )
    if parameters:
        context_lines.append(
            f"PARAMETER OVERRIDES:\n  {json.dumps(parameters, ensure_ascii=False)[:1200]}"
        )
    if propose_rationale:
        context_lines.append(
            f"WHY THE PROPOSER LLM CHOSE THIS CANDIDATE:\n{propose_rationale[:1500]}"
        )
    context_lines.append("METRIC SPECS:\n" + "\n".join(spec_lines))
    context_lines.append("CANDIDATE METRIC VECTOR:\n" + "\n".join(metrics_table))
    context_lines.append(
        "BASELINE METRIC VECTOR:\n"
        + "\n".join(f"  {k} = {v:.6g}" for k, v in base.items())
    )
    if best_block:
        context_lines.append(best_block.strip())
    if prior_proceed_summaries:
        compact = []
        for p in prior_proceed_summaries[-5:]:
            compact.append(
                f"  iter={p.get('iteration')} model='{str(p.get('model_description',''))[:80]}' "
                f"score={p.get('primary_score','?')} metrics={p.get('metrics',{})}"
            )
        context_lines.append("PRIOR PROCEED CANDIDATES (most recent up to 5):\n" + "\n".join(compact))
    if interpreter_summary:
        context_lines.append(f"INTERPRETER SUMMARY (physics plausibility):\n{interpreter_summary[:1500]}")
    if run_log_excerpt:
        context_lines.append(f"RUN LOG TAIL:\n{run_log_excerpt[-1500:]}")
    if diversity_mode and diversity_mode != "off":
        ds = ", ".join(families_seen_list or []) or "(none yet)"
        context_lines.append(
            f"SEARCH POLICY: diversity_mode={diversity_mode}; "
            f"FAMILIES ALREADY EXPLORED: {ds}"
        )
    context_lines.append(f"RECENT HISTORY (compact):\n{history_summary[:3000]}")
    if extra_context:
        context_lines.append(extra_context)

    user_msg = (
        "\n\n".join(context_lines)
        + "\n\nReturn STRICT JSON: "
        "{\"decision\": \"PROCEED|REVISE|RERUN|INDETERMINATE\", "
        "\"is_improvement_over_baseline\": bool, "
        "\"is_best_so_far\": bool, "
        "\"primary_score\": <float>, "
        "\"rationale\": \"<short, cites specific metrics + the modification>\", "
        "\"strengths\": [\"<metric>\", ...], "
        "\"weaknesses\": [\"<metric>\", ...]}"
    )
    try:
        raw = _llm_invoke([("system", sys_msg), ("user", user_msg)], temperature=0.0)
        body = _strip_code_fences(raw)
        s, e = body.find("{"), body.rfind("}")
        if s < 0 or e < 0:
            raise ValueError("no JSON object in judge response")
        obj = json.loads(body[s:e + 1])
        out = {
            "decision": str(obj.get("decision", "INDETERMINATE")).upper().strip(),
            "is_improvement_over_baseline": bool(obj.get("is_improvement_over_baseline", False)),
            "is_best_so_far": bool(obj.get("is_best_so_far", False)),
            "primary_score": float(obj["primary_score"]) if obj.get("primary_score") is not None else None,
            "rationale": str(obj.get("rationale", ""))[:1200],
            "strengths": [str(s) for s in (obj.get("strengths") or []) if s][:8],
            "weaknesses": [str(w) for w in (obj.get("weaknesses") or []) if w][:8],
            "method": "llm_judge",
        }
        if out["decision"] not in ("PROCEED", "REVISE", "RERUN", "INDETERMINATE"):
            out["decision"] = "INDETERMINATE"
        return out
    except TimeoutError as exc:
        print(f"[OED-EXT][judge] LLM judge timed out: {exc}; returning INDETERMINATE.")
        return {"decision": "INDETERMINATE", "is_improvement_over_baseline": False,
                "is_best_so_far": False, "primary_score": None,
                "rationale": f"judge timed out: {exc}", "strengths": [], "weaknesses": [],
                "method": "llm_judge", "failure_mode": "timeout"}
    except Exception as exc:
        print(f"[OED-EXT][judge] LLM judge failed: {exc}; returning INDETERMINATE.")
        return {"decision": "INDETERMINATE", "is_improvement_over_baseline": False,
                "is_best_so_far": False, "primary_score": None,
                "rationale": f"judge error: {exc}", "strengths": [], "weaknesses": [],
                "method": "llm_judge"}


def aggregate_metric_vector(
    *,
    metric_vector: Dict[str, Any],
    metric_specs: List[Dict[str, Any]],
    baseline_vector: Optional[Dict[str, Any]] = None,
    weights: Optional[Dict[str, float]] = None,
    aggregator: str = _DEFAULT_AGGREGATOR,
) -> Dict[str, Any]:
    """
    Reduce a metric vector to a single primary score. Supports:
      'weighted_sum'    — Σ w_i * normalized_error_i (lower=better)
      'min_improvement' — minimum (over metrics) of relative improvement vs baseline
                          (so a model is gated by its WORST metric)
      'pareto_rank'     — caller decides; we still return a scalar = sum of
                          metric ranks (lower=better)

    Normalization: each metric's value is divided by its baseline magnitude
    (absolute). 'max' direction metrics are negated so lower-better holds.
    """
    metrics = metric_vector.get("metrics", {}) or {}
    if not metrics:
        return {"primary": None, "aggregator": aggregator, "components": {}}

    base = (baseline_vector or {}).get("metrics", {}) or {}
    dirs = {m["name"]: m.get("direction", "min") for m in (metric_specs or [])}
    w = weights or {m["name"]: 1.0 for m in (metric_specs or [])}

    components: Dict[str, float] = {}
    for name, val in metrics.items():
        d = dirs.get(name, "min")
        signed = val if d == "min" else -val
        b = base.get(name)
        if isinstance(b, (int, float)) and b != 0:
            components[name] = signed / abs(b)
        else:
            components[name] = signed

    if aggregator == "min_improvement":
        # for each metric, compute relative improvement vs baseline (positive=better)
        improvements = []
        for name, val in metrics.items():
            b = base.get(name)
            d = dirs.get(name, "min")
            if not isinstance(b, (int, float)) or b == 0:
                continue
            if d == "min":
                imp = (b - val) / abs(b)
            else:
                imp = (val - b) / abs(b)
            improvements.append(imp)
        primary = -min(improvements) if improvements else 0.0  # negate so lower=better
    elif aggregator == "pareto_rank":
        primary = sum(components.values())
    else:  # weighted_sum
        total = 0.0
        for name, c in components.items():
            total += w.get(name, 1.0) * c
        primary = total

    return {"primary": primary, "aggregator": aggregator, "components": components,
            "direction": "min"}  # always emit lower=better


# ===========================================================================
# PHASE 2 — diversity (close + far)
# ===========================================================================

_FAMILIES_PATH_NAME = "families_explored.json"


def load_family_tracker(disc_dir: Path) -> Dict[str, Any]:
    p = disc_dir / _FAMILIES_PATH_NAME
    obj = _read_json(p, {})
    if not isinstance(obj, dict):
        obj = {}
    obj.setdefault("families", [])  # list of {family, model_class, equation, iter, decision}
    obj.setdefault("close_streak", 0)
    obj.setdefault("far_attempts", 0)
    return obj


def update_family_tracker(
    disc_dir: Path,
    *,
    iteration: int,
    family: str,
    model_class: str,
    equation_touched: str,
    decision: str,
    score: Optional[float] = None,
) -> None:
    state = load_family_tracker(disc_dir)
    state["families"].append({
        "iter": iteration, "family": family, "model_class": model_class,
        "equation": equation_touched, "decision": decision, "score": score,
    })
    if decision == "PROCEED":
        state["close_streak"] = state.get("close_streak", 0) + 1
    else:
        state["close_streak"] = 0
    _write_json(disc_dir / _FAMILIES_PATH_NAME, state)


def families_seen(disc_dir: Path) -> List[str]:
    state = load_family_tracker(disc_dir)
    seen = set()
    for f in state.get("families", []):
        fam = f.get("family", "").strip()
        if fam:
            seen.add(fam)
    return sorted(seen)


def decide_search_mode(
    *,
    iteration: int,
    total_budget: int,
    history: List[Dict[str, Any]],
    far_ratio: float = 0.3,
    far_after_n_no_improvement: int = 3,
    diversity_mode: str = "off",
) -> str:
    """
    Decide if iteration `iteration` should be 'close' or 'far'.

    diversity_mode:
      off       — always close (current behavior)
      hybrid    — every 1/far_ratio iters is forced 'far'; also far if
                  N consecutive iters had no improvement
      aggressive — alternate close/far
    """
    if diversity_mode == "off":
        return "close"
    if diversity_mode == "aggressive":
        return "far" if iteration % 2 == 0 else "close"

    # hybrid
    period = max(2, int(round(1.0 / max(0.05, far_ratio))))
    if iteration > 0 and iteration % period == 0:
        return "far"
    # check no-improvement streak in recent history
    recent = [h for h in history[-far_after_n_no_improvement:]
              if isinstance(h, dict) and h.get("action_type") in ("code_mod", "experiment")]
    if len(recent) >= far_after_n_no_improvement:
        if all(h.get("status") != "PROCEED" for h in recent):
            return "far"
    return "close"


def render_diversity_constraint(
    *, mode: str, families_seen_list: List[str],
    current_best_family: Optional[str] = None,
) -> str:
    """Prompt fragment to inject into the propose/decide prompt."""
    if mode == "close":
        target = f" (current best family: {current_best_family})" if current_best_family else ""
        return (
            f"\nSEARCH MODE: CLOSE-REFINEMENT{target}.\n"
            "Propose a small, parameter-level or structural-tweak modification of "
            "the current best direction. Explore nearby variants of the same "
            "model family. Goal: incremental improvement.\n"
        )
    seen = ", ".join(families_seen_list) if families_seen_list else "(none yet)"
    return (
        f"\nSEARCH MODE: FAR-FROM-BASELINE.\n"
        f"Families ALREADY explored (do NOT repeat these): {seen}\n"
        "Propose a candidate from a DIFFERENT model family or that touches a "
        "different equation than any explored above. Acceptable directions "
        "include: a different turbulence model class entirely (k-ω SST, "
        "transition models, RSM components), a different equation in the same "
        "model (production source vs destruction vs diffusion), a fundamentally "
        "different functional form (anisotropic stress limiter, non-equilibrium "
        "correction, Reynolds-stress augmentation). Justify the structural "
        "novelty in your rationale. Goal: escape local optima.\n"
    )


def classify_family(model_description: str, model_class: str = "") -> Tuple[str, str]:
    """
    Heuristic classifier mapping a model description to (family, equation_touched).
    Generic — uses keyword matching. Returns ('SA-RC', 'production'), etc.
    """
    text = f"{model_class} {model_description}".lower()
    family = "unknown"
    eq = "unknown"
    if "spalart" in text or "salmaras" in text or "sa-" in text or "sa_" in text:
        family = "SA"
        if "rotation" in text or "rc" in text:
            family = "SA-RC"
        elif "neq" in text or "non-equilibrium" in text or "non_equilibrium" in text:
            family = "SA-NEQ"
        elif "apg" in text:
            family = "SA-APG"
        elif "cb1" in text or "production" in text:
            family = "SA-Production"
    elif "k-omega" in text or "komega" in text or "k_omega" in text or "sst" in text:
        family = "k-omega-SST"
    elif "k-epsilon" in text or "k_epsilon" in text or "kepsilon" in text:
        family = "k-epsilon"
    elif "rsm" in text or "reynolds-stress" in text or "reynolds_stress" in text:
        family = "RSM"
    elif "smagorinsky" in text or "wale" in text or "dynamic-eddy" in text:
        family = "LES-SGS"
    if "production" in text or "stilde" in text or "p_eff" in text:
        eq = "production"
    elif "destruction" in text or "fw" in text:
        eq = "destruction"
    elif "diffusion" in text:
        eq = "diffusion"
    elif "source" in text or "fvoption" in text:
        eq = "source"
    elif "limiter" in text or "realizability" in text:
        eq = "limiter"
    return family, eq


# ===========================================================================
# PHASE 3 — multi-flow / multi-reference
# ===========================================================================

def detect_multi_flow_setup(
    *, starter_dir: Optional[Path], starter_dirs: Optional[List[Path]] = None
) -> Dict[str, Path]:
    """
    Resolve flow_id → flow_dir mapping.

    Priority:
      1. If starter_dirs is non-empty, use each as a separate flow.
         flow_id = directory name.
      2. Else if starter_dir contains multiple subdirs each having a
         reference_data/ subdir, treat each as a flow.
      3. Else single-flow mode: {default: starter_dir}.
    """
    flows: Dict[str, Path] = {}
    if starter_dirs:
        for sd in starter_dirs:
            if sd.is_dir():
                flows[sd.name] = sd.resolve()
        if flows:
            return flows
    if starter_dir and starter_dir.is_dir():
        sub_with_ref = []
        for child in starter_dir.iterdir():
            if child.is_dir() and (child / "reference_data").is_dir():
                sub_with_ref.append(child)
        if len(sub_with_ref) >= 2:
            for child in sub_with_ref:
                flows[child.name] = child.resolve()
            return flows
        flows["default"] = starter_dir.resolve()
        return flows
    return flows


def aggregate_flow_scores(
    *,
    per_flow_metrics: Dict[str, Dict[str, Any]],
    per_flow_baseline: Dict[str, Dict[str, Any]],
    metric_specs_per_flow: Dict[str, List[Dict[str, Any]]],
    flow_weights: Optional[Dict[str, float]] = None,
    aggregator: str = "weighted_sum",
) -> Dict[str, Any]:
    """
    Combine per-flow metric vectors into a unified score. Each flow's vector
    is first reduced to a per-flow primary via aggregate_metric_vector, then
    flows are combined.

    Returns:
      {"primary": float|None, "per_flow_primary": {flow: float},
       "aggregator": str, "direction": "min", "details": {...}}
    """
    per_flow_primary: Dict[str, float] = {}
    details: Dict[str, Any] = {}
    for flow_id, mv in per_flow_metrics.items():
        specs = metric_specs_per_flow.get(flow_id, [])
        baseline = per_flow_baseline.get(flow_id, {})
        red = aggregate_metric_vector(
            metric_vector=mv, metric_specs=specs, baseline_vector=baseline,
            aggregator=aggregator,
        )
        if red.get("primary") is not None:
            per_flow_primary[flow_id] = float(red["primary"])
        details[flow_id] = red

    if not per_flow_primary:
        return {"primary": None, "per_flow_primary": {}, "aggregator": aggregator,
                "direction": "min", "details": details}

    if aggregator == "min_improvement":
        # pessimistic: use the worst flow's reduction (already lower-better)
        primary = max(per_flow_primary.values())
    else:
        w = flow_weights or {f: 1.0 for f in per_flow_primary}
        s = sum(w.get(f, 1.0) for f in per_flow_primary)
        if s <= 0:
            s = 1.0
        primary = sum(w.get(f, 1.0) * v for f, v in per_flow_primary.items()) / s
    return {"primary": primary, "per_flow_primary": per_flow_primary,
            "aggregator": aggregator, "direction": "min", "details": details}


def render_score_matrix(
    *,
    per_flow_metrics: Dict[str, Dict[str, Any]],
    per_flow_baseline: Optional[Dict[str, Dict[str, Any]]] = None,
    metric_specs_per_flow: Optional[Dict[str, List[Dict[str, Any]]]] = None,
) -> str:
    """
    Format the per-flow x per-metric matrix for inclusion in the LLM prompt.
    """
    if not per_flow_metrics:
        return "(no per-flow metrics)"
    per_flow_baseline = per_flow_baseline or {}
    metric_specs_per_flow = metric_specs_per_flow or {}
    lines = ["Score matrix (flow × metric):"]
    for flow_id, mv in per_flow_metrics.items():
        lines.append(f"  [{flow_id}]")
        block = render_metric_vector_for_prompt(
            metric_vector=mv,
            baseline_vector=per_flow_baseline.get(flow_id, {}),
            metric_specs=metric_specs_per_flow.get(flow_id, []),
        )
        for ln in block.splitlines():
            lines.append(f"    {ln}")
    return "\n".join(lines)


# ===========================================================================
# Top-level helpers used by open_ended_discovery.py
# ===========================================================================

def is_phase1_enabled(args_or_env: Any) -> bool:
    """Single source of truth for whether multi-metric mode is on."""
    if isinstance(args_or_env, dict):
        return bool(args_or_env.get("multi_metric"))
    return bool(getattr(args_or_env, "multi_metric", False))


def is_phase2_enabled(args_or_env: Any) -> bool:
    if isinstance(args_or_env, dict):
        return str(args_or_env.get("diversity_mode", "off")) != "off"
    return str(getattr(args_or_env, "diversity_mode", "off")) != "off"


def is_phase3_enabled(args_or_env: Any) -> bool:
    if isinstance(args_or_env, dict):
        return bool(args_or_env.get("multi_flow"))
    return bool(getattr(args_or_env, "multi_flow", False))


def write_metric_artifact(
    *, disc_dir: Path, iteration: int, case_dir: Path,
    metric_vector: Dict[str, Any], aggregated: Dict[str, Any],
    flow_id: Optional[str] = None,
) -> None:
    """Persist per-iteration metric record so the history captures the vector."""
    rec = {
        "iter": iteration,
        "case_dir": str(case_dir),
        "flow_id": flow_id,
        "metrics": metric_vector.get("metrics", {}),
        "errors": metric_vector.get("errors", {}),
        "primary": aggregated.get("primary"),
        "aggregator": aggregated.get("aggregator"),
        "direction": aggregated.get("direction", "min"),
        "components": aggregated.get("components", {}),
    }
    out = disc_dir / "metric_artifacts" / f"iter_{iteration:03d}{('_'+flow_id) if flow_id else ''}.json"
    _write_json(out, rec)


__all__ = [
    # Phase 1
    "propose_metric_set", "discover_existing_comparators", "author_comparator",
    "selftest_comparator", "author_and_selftest", "resolve_metric_comparators",
    "compute_metric_vector",
    "render_metric_vector_for_prompt", "aggregate_metric_vector",
    # Phase 2
    "load_family_tracker", "update_family_tracker", "families_seen",
    "decide_search_mode", "render_diversity_constraint", "classify_family",
    # Phase 3
    "detect_multi_flow_setup", "aggregate_flow_scores", "render_score_matrix",
    # Helpers
    "is_phase1_enabled", "is_phase2_enabled", "is_phase3_enabled",
    "write_metric_artifact",
]
