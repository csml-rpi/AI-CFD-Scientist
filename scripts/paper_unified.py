#!/usr/bin/env python3
"""Unified analysis-aware paper pipeline: plan, PyVista paper figures, write+review loop."""
from __future__ import annotations

import argparse
import json
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
    parser = argparse.ArgumentParser(description="Unified paper: plan + paper PyVista figs + write/review.")
    parser.add_argument("--repo-root", required=True, type=str)
    parser.add_argument("--run-dir", required=True, type=str)
    parser.add_argument("--topic", required=True, type=str)
    parser.add_argument("--paper-dir", required=True, type=str)
    parser.add_argument("--analysis", required=True, type=str)
    parser.add_argument("--manifest", required=True, type=str)
    parser.add_argument("--requirements", required=True, type=str)
    parser.add_argument("--literature", required=True, type=str)
    parser.add_argument("--review-output", required=True, type=str)
    parser.add_argument("--mesh-independence", default="", type=str)
    parser.add_argument("--template", default="neurips", choices=["neurips", "icml", "iclr", "nature"])
    parser.add_argument("--max-review-loops", type=int, default=10)
    parser.add_argument(
        "--max-viz-inner-attempts",
        type=int,
        default=10,
        help="Max batch-script regen cycles per outer paper step (per-image VLM QA).",
    )
    parser.add_argument("--score-threshold", type=float, default=0.82)
    parser.add_argument("--model", default="", type=str)
    parser.add_argument("--starter-understanding", default="", type=str,
                        help="Path to starter_understanding.json for flow params, formula, reference data context")
    args = parser.parse_args()

    from cfd_langgraph.config import get_settings
    from cfd_langgraph.paper_unified.pipeline import run_unified_paper_pipeline

    repo_root = Path(args.repo_root).resolve()
    run_dir = Path(args.run_dir).resolve()
    settings = get_settings()
    model = args.model.strip() or settings.model
    mip = Path(args.mesh_independence).resolve() if args.mesh_independence.strip() else None
    sup = Path(args.starter_understanding).resolve() if args.starter_understanding.strip() else None

    out = run_unified_paper_pipeline(
        repo_root=repo_root,
        run_dir=run_dir,
        topic=args.topic,
        paper_dir=Path(args.paper_dir).resolve(),
        analysis_path=Path(args.analysis).resolve(),
        manifest_path=Path(args.manifest).resolve(),
        requirements_path=Path(args.requirements).resolve(),
        lit_path=Path(args.literature).resolve(),
        review_path=Path(args.review_output).resolve(),
        mesh_independence_path=mip if mip and mip.is_file() else None,
        starter_understanding_path=sup if sup and sup.is_file() else None,
        paper_template=args.template,
        model=model,
        max_review_loops=max(1, int(args.max_review_loops)),
        max_viz_inner_attempts=max(1, int(args.max_viz_inner_attempts)),
        score_threshold=float(args.score_threshold),
        verbose=True,
    )
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
