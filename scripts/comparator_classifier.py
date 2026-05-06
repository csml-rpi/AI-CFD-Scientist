#!/usr/bin/env python3
"""LLM-based classifier for starter scripts.

Replaces the brittle `rglob("compare*.py")` naming-convention used elsewhere in
the pipeline. Reads each .py file's CONTENT and asks the LLM to classify its
role (comparator, reader, plot helper, utility, other). Caches the verdict to
a JSON file so subsequent callers don't re-spend tokens.

Anti-contamination guarantee: only paths that resolve INSIDE the starter dir
are ever returned, regardless of what the LLM hallucinates.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


# Directory components excluded from the candidate scan. OpenFOAM time + system
# dirs would otherwise pollute candidates; runs/, __pycache__ etc. are pure
# noise.
_EXCLUDE_PATH_PARTS = {
    "runs", "__pycache__", "node_modules", ".git", "platforms",
    "0", "system", "constant", "polyMesh", "_diag", ".venv",
    "build", "dist", ".pytest_cache",
}
_MAX_CANDIDATES = 24
_MAX_HEAD_LINES = 60
_MAX_TAIL_LINES = 20
_MAX_FILE_BYTES = 200_000   # over-large files are usually compiled-output / data
_MIN_FILE_BYTES = 100


def _walk_candidates(starter_dir: Path) -> List[Path]:
    out: List[Path] = []
    sd = starter_dir.resolve()
    for p in sd.rglob("*.py"):
        if not p.is_file():
            continue
        parts_lower = {part.lower() for part in p.parts}
        if parts_lower & _EXCLUDE_PATH_PARTS:
            continue
        try:
            sz = p.stat().st_size
        except OSError:
            continue
        if sz < _MIN_FILE_BYTES or sz > _MAX_FILE_BYTES:
            continue
        out.append(p.resolve())
        if len(out) >= _MAX_CANDIDATES:
            break
    return out


def _format_excerpt(p: Path) -> str:
    try:
        text = p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""
    lines = text.splitlines()
    if len(lines) <= _MAX_HEAD_LINES + _MAX_TAIL_LINES:
        return text
    head = "\n".join(lines[:_MAX_HEAD_LINES])
    tail = "\n".join(lines[-_MAX_TAIL_LINES:])
    return head + "\n# ... (omitted) ...\n" + tail


def _bootstrap_repo() -> None:
    """Make sure cfd_langgraph is importable from this script."""
    here = Path(__file__).resolve()
    repo_root = here.parent.parent
    candidate = repo_root / "src"
    if str(candidate) not in sys.path and candidate.is_dir():
        sys.path.insert(0, str(candidate))
    scripts = repo_root / "scripts"
    if str(scripts) not in sys.path and scripts.is_dir():
        sys.path.insert(0, str(scripts))


def classify_starter_scripts(
    *,
    starter_dir: Optional[Path],
    topic: str,
    quantities: Optional[List[str]] = None,
    cache_path: Optional[Path] = None,
    force_refresh: bool = False,
) -> Dict[str, Any]:
    """Classify .py files in starter_dir by content using an LLM.

    Args:
        starter_dir: starter directory to scan. Only files under this path may
            be returned (enforced server-side regardless of what the LLM picks).
        topic: study topic, used to bias the LLM toward selecting the
            comparator that fits THIS run, not just any comparator-shaped file.
        quantities: optional list of QoI keywords (Cf, K, drag, etc.).
        cache_path: optional Path to read/write a cache JSON. If set and the
            file already exists with status="ok", it is returned without
            re-calling the LLM (unless force_refresh=True).
        force_refresh: ignore any existing cache and re-run.

    Returns dict:
        {
          "comparator_path": Optional[str absolute path],
          "scripts": [{"path", "role", "summary"}],
          "reasoning": str,
          "topic_used": str,
          "status": "ok" | "no_starter" | "no_candidates" | "llm_unavailable",
          "cached": bool,
        }
    """
    # Cache hit fast-path
    if cache_path and Path(cache_path).is_file() and not force_refresh:
        try:
            data = json.loads(Path(cache_path).read_text())
            if isinstance(data, dict) and data.get("status") == "ok":
                data["cached"] = True
                return data
        except Exception:
            pass

    if not starter_dir or not Path(starter_dir).is_dir():
        result = {
            "comparator_path": None, "scripts": [], "topic_used": topic,
            "status": "no_starter", "cached": False,
        }
        return result

    sd = Path(starter_dir).resolve()
    candidates = _walk_candidates(sd)
    if not candidates:
        result = {
            "comparator_path": None, "scripts": [], "topic_used": topic,
            "status": "no_candidates", "cached": False,
        }
        if cache_path:
            try:
                Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
                Path(cache_path).write_text(json.dumps(result, indent=2))
            except Exception:
                pass
        return result

    # Build prompt
    sys_prompt = (
        "You classify Python scripts found in a CFD research starter directory.\n\n"
        "For each script, decide its ROLE:\n"
        "- comparator: computes a numeric metric COMPARING simulation output to reference\n"
        "  data, with CLI flags like --case and/or --reference, and prints something like\n"
        "  \"METRIC <name>: <value>\" on stdout. This is the kind of script the open-ended\n"
        "  discovery loop would invoke to score a candidate model against reference data.\n"
        "- reader: parses simulation output (OpenFOAM fields, postProcessing files, .dat,\n"
        "  .csv) into structured form. Does NOT compute a score itself.\n"
        "- plot_helper: creates figures (matplotlib/pyvista/pandas plots). Does NOT score.\n"
        "- utility: file/data conversion, format helpers, build glue.\n"
        "- other: cannot classify, or none of the above.\n\n"
        "Pick the ONE script most likely to be THE comparator the orchestrator should\n"
        "bind for scoring candidate models against reference data for THIS topic. If no\n"
        "candidate is clearly a comparator (or all are unrelated to the topic), set\n"
        "comparator_path to null and explain in reasoning. Do NOT pick based on filename\n"
        "alone — the user may have named the script anything; classify by content.\n\n"
        "Return STRICT JSON only (no prose, no markdown fences). Schema:\n"
        "{\n"
        "  \"comparator_path\": \"<absolute path or null>\",\n"
        "  \"scripts\": [{\"path\": \"<abs>\", \"role\": \"comparator|reader|plot_helper|utility|other\", \"summary\": \"<1 sentence>\"}],\n"
        "  \"reasoning\": \"<1-3 sentences explaining the pick (or why nothing was picked)>\"\n"
        "}\n"
    )

    user_lines = [f"TOPIC: {topic}"]
    if quantities:
        user_lines.append(f"QUANTITIES OF INTEREST: {', '.join(str(q) for q in quantities)}")
    user_lines.append(f"\nCANDIDATE SCRIPTS ({len(candidates)} total — only paths under starter_dir are valid):\n")
    for c in candidates:
        try:
            rel = str(c.relative_to(sd))
        except ValueError:
            rel = c.name
        user_lines.append(f"--- {rel} (abs: {c}) ---")
        user_lines.append(_format_excerpt(c))
        user_lines.append("")
    user_msg = "\n".join(user_lines)

    # Call LLM
    raw_response = ""
    try:
        _bootstrap_repo()
        from langchain_core.messages import HumanMessage, SystemMessage  # type: ignore
        from cfd_langgraph.config import get_settings  # type: ignore
        from cfd_langgraph.llm.factory import create_langchain_llm  # type: ignore
        try:
            from cfd_langgraph.utils import strip_json_fences  # type: ignore
        except Exception:
            def strip_json_fences(s: str) -> str:
                return re.sub(r"^```(?:json)?\s*|\s*```$", "", s.strip(), flags=re.MULTILINE | re.DOTALL)

        llm = create_langchain_llm(model=get_settings().model, temperature=0.0)
        resp = llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=user_msg)])
        raw_response = resp.content if hasattr(resp, "content") else str(resp)
        cleaned = strip_json_fences(raw_response)
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not m:
            raise ValueError(f"no JSON object in LLM response: {raw_response[:200]!r}")
        parsed = json.loads(m.group(0))

        comparator_raw = parsed.get("comparator_path")
        comparator_resolved: Optional[str] = None
        if comparator_raw and isinstance(comparator_raw, str):
            try:
                cand = Path(comparator_raw).resolve()
                # Anti-contamination: must be a real file under starter_dir.
                if cand.is_file() and str(cand).startswith(str(sd)):
                    comparator_resolved = str(cand)
            except Exception:
                comparator_resolved = None

        # Sanitize scripts list — only keep entries that point inside starter_dir.
        scripts_clean: List[Dict[str, str]] = []
        for s in (parsed.get("scripts") or []):
            if not isinstance(s, dict):
                continue
            sp = s.get("path") or ""
            try:
                spr = Path(sp).resolve()
            except Exception:
                continue
            if not spr.is_file() or not str(spr).startswith(str(sd)):
                continue
            scripts_clean.append({
                "path": str(spr),
                "role": str(s.get("role") or "other"),
                "summary": str(s.get("summary") or "")[:300],
            })

        result = {
            "comparator_path": comparator_resolved,
            "scripts": scripts_clean,
            "reasoning": str(parsed.get("reasoning") or "")[:1000],
            "topic_used": topic,
            "status": "ok",
            "cached": False,
        }
    except Exception as exc:
        result = {
            "comparator_path": None,
            "scripts": [{"path": str(p), "role": "other", "summary": "(LLM unavailable)"} for p in candidates],
            "reasoning": "",
            "topic_used": topic,
            "status": "llm_unavailable",
            "error": repr(exc)[:400],
            "raw_excerpt": raw_response[:800],
            "cached": False,
        }

    # Persist cache
    if cache_path:
        try:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            Path(cache_path).write_text(json.dumps(result, indent=2))
        except Exception:
            pass
    return result


def find_comparator_for_starter(
    *,
    starter_dir: Optional[Path],
    topic: str,
    cache_path: Optional[Path] = None,
    quantities: Optional[List[str]] = None,
) -> Optional[Path]:
    """Convenience wrapper — returns just the comparator Path, or None.

    Honors the cache; safe to call multiple times.
    """
    info = classify_starter_scripts(
        starter_dir=starter_dir,
        topic=topic,
        quantities=quantities,
        cache_path=cache_path,
    )
    cp = info.get("comparator_path")
    if cp and Path(cp).is_file():
        return Path(cp)
    return None
