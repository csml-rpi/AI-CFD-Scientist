#!/usr/bin/env python3
"""
Post-paper Semantic Scholar reference verification + optional LLM cleanup.

Runs after paper write/review: checks each \\bibitem / .bib entry via DOI or title
(S2 API, same spirit as s2-title-to-bibtex). Unverified keys are removed from the
manuscript and bibliography using an LLM pass.

Environment: S2_API_KEY optional but recommended (same as scripts/lit.py).
Skip cleanup: omit --apply-cleanup or set CFD_REFERENCE_VERIFY_CLEANUP=0.
Skip entire stage: CFD_SKIP_REFERENCE_VERIFY=1 (orchestrator: --skip-reference-verify).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def bootstrap_paths() -> Path:
    root = Path(__file__).resolve().parent.parent
    foam_src = root / "Foam-Agent" / "src"
    lang_src = root / "src"
    if str(foam_src) not in sys.path:
        sys.path.insert(0, str(foam_src))
    if str(lang_src) not in sys.path:
        sys.path.insert(0, str(lang_src))
    return root


def main() -> int:
    bootstrap_paths()
    parser = argparse.ArgumentParser(description="Verify references via Semantic Scholar; optionally clean LaTeX.")
    parser.add_argument("--paper-dir", required=True, type=str)
    parser.add_argument("--literature", default="", type=str, help="lit.json from literature stage")
    parser.add_argument("--output", required=True, type=str, help="reference_verify_report.json")
    parser.add_argument(
        "--apply-cleanup",
        action="store_true",
        help="Run LLM to strip bad keys from TeX/.bib (orchestrator enables this by default)",
    )
    parser.add_argument("--no-apply-cleanup", action="store_true", help="Force no LLM edits")
    parser.add_argument("--recompile", action="store_true", help="Run pdflatex on main.tex after cleanup")
    parser.add_argument("--model", default="", type=str)
    args = parser.parse_args()

    if os.environ.get("CFD_SKIP_REFERENCE_VERIFY", "").strip() in ("1", "true", "yes"):
        print("[REF-VERIFY] Skipped (CFD_SKIP_REFERENCE_VERIFY).")
        return 0

    paper_dir = Path(args.paper_dir).resolve()
    if not paper_dir.is_dir():
        print(f"[REF-VERIFY] paper-dir not found: {paper_dir}", file=sys.stderr)
        return 1
    main_tex_path = paper_dir / "main.tex"
    if not main_tex_path.is_file():
        print(f"[REF-VERIFY] main.tex missing in {paper_dir}", file=sys.stderr)
        return 1

    lit_path = Path(args.literature).resolve() if args.literature.strip() else None

    from cfd_langgraph.reference_cleanup_llm import (
        cleanup_hallucinated_references,
        load_model_from_settings,
        sync_body_tex,
    )
    from cfd_langgraph.reference_verify_s2 import build_verification_report, merge_missing_cites_as_hallucinations
    from timeline_logger import append_timeline_event, resolve_timeline_path

    timeline_path = resolve_timeline_path(os.environ.get("CFD_ORCH_TIMELINE_PATH", ""))

    report = build_verification_report(paper_dir, lit_path)
    report = merge_missing_cites_as_hallucinations(report)
    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[REF-VERIFY] Wrote report: {out_path}")

    append_timeline_event(
        timeline_path,
        {
            "stage": "reference_verify",
            "event": "report_written",
            "report_path": str(out_path),
            "unverified": report["counts"].get("unverified", 0),
            "cleanup_keys": report.get("cleanup_keys", []),
        },
    )

    cleanup_keys = report.get("cleanup_keys") or []
    env_cleanup = os.environ.get("CFD_REFERENCE_VERIFY_CLEANUP", "1").strip().lower() not in ("0", "false", "no")
    do_cleanup = bool(args.apply_cleanup) and env_cleanup and (not args.no_apply_cleanup)

    if not cleanup_keys:
        print("[REF-VERIFY] No hallucination candidates; cleanup not needed.")
        append_timeline_event(
            timeline_path,
            {"stage": "reference_verify", "event": "cleanup_skipped", "reason": "no_bad_keys"},
        )
        return 0

    if not do_cleanup:
        print(f"[REF-VERIFY] Found {len(cleanup_keys)} key(s) to fix; cleanup disabled. Re-run with --apply-cleanup.")
        append_timeline_event(
            timeline_path,
            {
                "stage": "reference_verify",
                "event": "cleanup_skipped",
                "reason": "flag_or_env",
                "keys": cleanup_keys,
            },
        )
        return 0

    model = args.model.strip() or load_model_from_settings()
    main_tex = main_tex_path.read_text(encoding="utf-8", errors="ignore")
    bib_path = paper_dir / "references.bib"
    bib_text = bib_path.read_text(encoding="utf-8", errors="ignore") if bib_path.is_file() else ""

    backup_dir = paper_dir / "_reference_verify_backup"
    backup_dir.mkdir(parents=True, exist_ok=True)
    (backup_dir / "main.tex.bak").write_text(main_tex, encoding="utf-8")
    if bib_text:
        (backup_dir / "references.bib.bak").write_text(bib_text, encoding="utf-8")

    hallucinated = list(report.get("hallucination_candidates", []))
    print(f"[REF-VERIFY] Running LLM cleanup for {len(hallucinated)} candidate(s)...")
    new_tex, new_bib = cleanup_hallucinated_references(
        main_tex=main_tex,
        references_bib=bib_text,
        hallucinated=hallucinated,
        model=model,
    )
    main_tex_path.write_text(new_tex, encoding="utf-8")
    sync_body_tex(paper_dir, new_tex)
    bib_path.write_text(new_bib, encoding="utf-8")

    append_timeline_event(
        timeline_path,
        {
            "stage": "reference_verify",
            "event": "cleanup_applied",
            "keys_removed": cleanup_keys,
            "backup_dir": str(backup_dir),
        },
    )
    print(f"[REF-VERIFY] Updated {main_tex_path} and {bib_path} (backup in {backup_dir})")

    if args.recompile:
        from cfd_langgraph.paper_utils import compile_tex_to_pdf

        ok, pdf_path, err = compile_tex_to_pdf(main_tex_path, work_dir=paper_dir)
        if ok:
            print(f"[REF-VERIFY] Recompiled PDF -> {pdf_path}")
            append_timeline_event(
                timeline_path,
                {"stage": "reference_verify", "event": "recompiled", "pdf": str(pdf_path)},
            )
        else:
            print(f"[REF-VERIFY] Recompile failed:\n{err[:4000]}", file=sys.stderr)
            append_timeline_event(
                timeline_path,
                {"stage": "reference_verify", "event": "recompile_failed", "error_tail": (err or "")[:2000]},
            )
            # Cleanup already applied; do not fail the orchestrator on pdflatex alone.
            return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
