from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _run_rag_queries(requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Answer several index questions in one subprocess.

    A subprocess, not an in-process call, so the ~2 GB embedding model is
    released the moment retrieval is done rather than being held for the whole
    multi-hour study. One subprocess for all three questions, not three: the
    index load dominates the cost, and it used to be paid once per question.

    Returns one payload per request, in order; a failed batch degrades every
    entry rather than raising, because the caller already has a documented
    fallback for "retrieval unavailable".
    """
    failed = lambda detail: [{"ok": False, "stderr": detail} for _ in requests]
    try:
        proc = subprocess.run(
            [sys.executable, "scripts/rag_query.py", "--batch", json.dumps(requests), "--output", "-"],
            cwd=str(_REPO_ROOT), capture_output=True, text=True, timeout=600,
        )
    except Exception as exc:
        return failed(str(exc))
    if proc.returncode != 0:
        return failed((proc.stderr or "")[-800:])
    try:
        items = json.loads(proc.stdout).get("batch", [])
    except Exception:
        return failed("unparseable rag_query.py output")
    if len(items) != len(requests):
        return failed(f"expected {len(requests)} results, got {len(items)}")
    for item in items:
        item["ok"] = True
    return items


def _results_to_text(results: List[Any]) -> str:
    if not results:
        return ""
    per_item_budget = max(500, 8000 // len(results))
    parts = []
    for i, r in enumerate(results, 1):
        text = json.dumps(r, ensure_ascii=False, default=str)[:per_item_budget]
        parts.append(f"<similar_case_{i}>\n{text}\n</similar_case_{i}>")
    return "\n".join(parts)


def _fallback_tutorial_listing(case_solver: str) -> Dict[str, Any]:
    """FoamAgent's own documented degraded path when the FAISS indices
    (see cfd_langgraph/foam_native/faiss_index.py) aren't present: read one real
    tutorial from $WM_PROJECT_DIR/tutorials/<solver>/ directly. Lower
    quality, but the writer stages still work — see
    cfd-skills/cfd-foamagent/SKILL.md Stage 2."""
    empty = {"ok": False, "tutorial_reference": "", "dir_structure": "", "allrun_reference": "", "fallback": True}
    wm_project_dir = os.environ.get("WM_PROJECT_DIR", "")
    if not wm_project_dir:
        return empty
    solver_dir = Path(wm_project_dir) / "tutorials" / case_solver
    if not solver_dir.is_dir():
        return empty
    case_dirs = sorted(p for p in solver_dir.rglob("*") if p.is_dir() and (p / "system").is_dir())[:1]
    if not case_dirs:
        return empty
    example = case_dirs[0]
    structure = "\n".join(str(p.relative_to(example)) for p in sorted(example.rglob("*")) if p.is_file())
    allrun_path = example / "Allrun"
    allrun_text = allrun_path.read_text(encoding="utf-8", errors="ignore") if allrun_path.is_file() else ""
    return {
        "ok": True,
        "tutorial_reference": f'<similar_case_1 path="{example}">\n{structure}\n</similar_case_1>',
        "dir_structure": structure,
        "allrun_reference": allrun_text,
        "fallback": True,
    }


def retrieve_references(
    user_requirement: str, case_solver: str, case_domain: str = "", case_category: str = "", top: int = 3
) -> Dict[str, Any]:
    """FoamAgent Stage 2 (RAG retrieval). Queries the three FAISS indices via
    scripts/rag_query.py; on any failure (indices not built, embedding model
    missing — rc=2/3, matching the documented failure modes), falls back to
    reading a real tutorial from $WM_PROJECT_DIR/tutorials/ directly rather
    than blocking the whole case on the prebuilt indices being present."""
    tutorials, structure, allrun = _run_rag_queries(
        [
            {"db": "openfoam_tutorials_details", "query": f"{case_solver} {user_requirement}", "top": top},
            {"db": "openfoam_tutorials_structure", "query": f"{case_solver} {case_domain} {case_category}", "top": top},
            {"db": "openfoam_allrun_scripts", "query": f"{case_solver} {user_requirement}", "top": top},
        ]
    )

    if not (tutorials.get("ok") and structure.get("ok") and allrun.get("ok")):
        return _fallback_tutorial_listing(case_solver)

    return {
        "ok": True,
        "tutorial_reference": _results_to_text(tutorials.get("results", [])),
        "dir_structure": _results_to_text(structure.get("results", [])),
        "allrun_reference": _results_to_text(allrun.get("results", [])),
        "fallback": False,
    }
