from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def _try_read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def summarize_refchecker_report(report: Dict[str, Any]) -> str:
    """
    Convert refchecker JSON into a compact, LLM-friendly string.
    """
    if not report:
        return ""

    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    if not summary:
        # Some refchecker versions might not nest under "summary".
        return json.dumps(report, indent=2)[:8000]

    parts: list[str] = []
    # Rollup counters (based on refchecker README sample output).
    for k in [
        "total_errors_found",
        "total_warnings_found",
        "total_unverified_refs",
        "flagged_records",
        "flagged_papers",
        "total_references_processed",
    ]:
        if k in summary:
            parts.append(f"{k}={summary.get(k)}")

    # Top records with issues (if present).
    records = report.get("records")
    if isinstance(records, list) and records:
        parts.append("Top flagged records:")
        for r in records[:5]:
            if not isinstance(r, dict):
                continue
            err_type = r.get("error_type") or r.get("error_type_counts") or ""
            title = r.get("ref_title") or r.get("source_title") or ""
            if isinstance(err_type, dict):
                err_type = ", ".join(f"{kk}={vv}" for kk, vv in list(err_type.items())[:3])
            if title:
                parts.append(f"- {err_type}: {title}")
            else:
                parts.append(f"- {err_type}")

    return "\n".join(parts).strip()


def run_refchecker_on_tex(
    tex_path: Path,
    out_dir: Path,
    attempt: int,
    verbose: bool = False,
) -> Tuple[bool, str]:
    """
    Run `academic-refchecker` against a generated LaTeX file.

    Returns:
      (ok, reference_report_summary)
    """
    # Allow opting out globally.
    if os.environ.get("CFD_REFCHECKER_ENABLE", "1") != "1":
        return True, ""

    academic = shutil.which("academic-refchecker")
    if not academic:
        # Optional dependency: skip gracefully if not installed.
        if verbose:
            print("[Writer] refchecker skipped (academic-refchecker not found).")
        return False, ""

    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"refcheck_report_attempt_{attempt}.json"

    cmd = [
        "academic-refchecker",
        "--paper",
        str(tex_path),
        "--report-file",
        str(report_path),
        "--report-format",
        "json",
    ]

    llm_provider = os.environ.get("REFCHECKER_LLM_PROVIDER")
    if llm_provider:
        cmd += ["--llm-provider", llm_provider]

    llm_model = os.environ.get("REFCHECKER_LLM_MODEL")
    if llm_model:
        cmd += ["--llm-model", llm_model]

    if verbose:
        cmd.append("--debug")

    if verbose:
        print(f"[Writer] Running refchecker (attempt {attempt})...")

    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if proc.returncode != 0:
        if verbose:
            print(f"[Writer] refchecker failed (rc={proc.returncode}).")
            print(proc.stderr[:4000])
        return False, (proc.stderr or proc.stdout or "").strip()[:8000]

    report = _try_read_json(report_path)
    summary = summarize_refchecker_report(report)
    return True, summary

