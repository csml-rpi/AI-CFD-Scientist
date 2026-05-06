#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


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
    parser = argparse.ArgumentParser(description="Generate visualization figures.")
    parser.add_argument("--case", type=str, default="")
    parser.add_argument("--cases", nargs="*", default=[])
    parser.add_argument("--mode", choices=["interpret", "full"], required=True)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument(
        "--viz-plan-json",
        default="",
        type=str,
        help="Optional analysis plan JSON with case_viz_spec_default and case_viz_overrides.",
    )
    args = parser.parse_args()

    from cfd_langgraph.config import get_settings
    from cfd_langgraph.viz_creator import viz_creator

    model = get_settings().model
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    figures: List[Dict[str, Any]] = []
    viz_plan: Dict[str, Any] = {}
    if args.viz_plan_json:
        p = Path(args.viz_plan_json).resolve()
        if p.exists():
            try:
                viz_plan = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                viz_plan = {}

    if args.mode == "interpret":
        if not args.case:
            print("--case is required for interpret mode", file=sys.stderr)
            return 1
        case = Path(args.case).resolve()
        req = "Generate residual trend plot and velocity magnitude contour to assess convergence."
        result = viz_creator(
            model=model,
            foam_output_dir=case,
            viz_dir=out_dir,
            what_to_visualize=req,
            user_requirement=req,
            strict_quality=False,
        )
        for p in result.get("images", []):
            figures.append({"case": str(case), "path": p})
    else:
        if not args.cases:
            print("--cases is required for full mode", file=sys.stderr)
            return 1
        default_req = "Create publication-quality contour, streamline, and profile figures for this CFD case."
        if isinstance(viz_plan, dict):
            v = str(viz_plan.get("case_viz_spec_default", "")).strip()
            if v:
                default_req = v
        overrides = viz_plan.get("case_viz_overrides", {}) if isinstance(viz_plan, dict) else {}
        if not isinstance(overrides, dict):
            overrides = {}
        for c in args.cases:
            case = Path(c).resolve()
            case_out = out_dir / case.name
            case_out.mkdir(parents=True, exist_ok=True)
            req = str(overrides.get(case.name, default_req))
            result = viz_creator(
                model=model,
                foam_output_dir=case,
                viz_dir=case_out,
                what_to_visualize=req,
                user_requirement=req,
            )
            for p in result.get("images", []):
                figures.append({"case": str(case), "path": p})

    (out_dir / "figures.json").write_text(json.dumps(figures, indent=2), encoding="utf-8")
    for f in figures:
        print(f["path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
