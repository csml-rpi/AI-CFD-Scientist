"""Assemble run-directory context for unified paper planning."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def _tree_lines(root: Path, prefix: str = "", max_depth: int = 4, depth: int = 0) -> List[str]:
    lines: List[str] = []
    if depth > max_depth:
        return lines
    try:
        entries = sorted([p for p in root.iterdir() if not p.name.startswith(".")], key=lambda p: p.name)
    except OSError:
        return lines
    for i, p in enumerate(entries[:80]):
        lines.append(f"{prefix}{p.name}/" if p.is_dir() else f"{prefix}{p.name}")
        if p.is_dir() and depth < max_depth and p.name not in {"0", "postProcessing", "processor0"}:
            lines.extend(_tree_lines(p, prefix + "  ", max_depth, depth + 1))
    if len(entries) > 80:
        lines.append(f"{prefix}... ({len(entries) - 80} more)")
    return lines


def summarize_run_directory(run_dir: Path, max_depth: int = 3) -> str:
    run_dir = run_dir.expanduser().resolve()
    lines = [f"RUN_DIR={run_dir}", ""]
    lines.extend(_tree_lines(run_dir, max_depth=max_depth))
    return "\n".join(lines[:2000])


def load_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def build_planner_payload(
    *,
    run_dir: Path,
    topic: str,
    manifest: Dict[str, Any],
    requirements: List[Dict[str, Any]],
    analysis: Dict[str, Any],
    mesh_bundle: Dict[str, Any],
    code_mod_note: str,
    starter_understanding_note: str = "",
) -> str:
    """Single text block for the planner LLM."""
    cases = manifest.get("cases") if isinstance(manifest.get("cases"), list) else []
    case_summaries: List[str] = []
    for c in cases[:40]:
        if not isinstance(c, dict):
            continue
        cid = c.get("case_id", "")
        st = c.get("status", "")
        p = c.get("case_path", "")
        case_summaries.append(f"- {cid}: status={st} path={p}")
    req_brief: List[str] = []
    for r in requirements[:40]:
        if not isinstance(r, dict):
            continue
        cid = r.get("case_id", "")
        desc = (r.get("description") or "")[:400]
        txt = (r.get("user_requirement_text") or "")[:600]
        req_brief.append(f"=== {cid} ===\ndescription: {desc}\nrequirement_excerpt: {txt}\n")
    mesh_note = ""
    if mesh_bundle:
        mesh_note = json.dumps(
            {
                "selected_stable_name": mesh_bundle.get("selected_stable_name"),
                "metrics_levels": len(mesh_bundle.get("metrics_by_mesh_level") or []),
            },
            indent=2,
        )
    analysis_excerpt = json.dumps(analysis, indent=2)[:25_000]
    parts = [
        f"TOPIC:\n{topic}\n",
        "\nRUN DIRECTORY LAYOUT (partial):\n" + summarize_run_directory(run_dir),
        "\nMANIFEST CASES:\n" + "\n".join(case_summaries),
        "\nREQUIREMENTS (per case):\n" + "\n".join(req_brief),
        "\nANALYSIS JSON (truncated):\n" + analysis_excerpt,
        "\nMESH INDEPENDENCE BUNDLE (summary):\n" + (mesh_note or "(none)"),
        "\nCODE MOD / CUSTOM MODEL NOTE:\n" + (code_mod_note or "(none)"),
    ]
    if starter_understanding_note:
        parts.insert(1, f"\nSTARTER FOLDER CONTEXT (authoritative — flow params, formula, reference data):\n{starter_understanding_note}\n")
    return "\n".join(parts)
