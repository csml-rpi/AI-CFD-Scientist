#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List
from timeline_logger import append_timeline_event, resolve_timeline_path


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
    parser = argparse.ArgumentParser(description="Interpret case results using interpreter agent.")
    parser.add_argument("--case", required=True, type=str)
    parser.add_argument("--figs", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--requirement", default="", type=str,
                        help="Full experiment requirement text; injected into interpreter for correctness checking.")
    parser.add_argument("--timeline", default="", type=str)
    args = parser.parse_args()
    timeline_path = resolve_timeline_path(args.timeline)

    from cfd_langgraph.agents.interpreter_agent import ResultsInterpreterAgent
    from cfd_langgraph.config import get_settings
    from cfd_langgraph.prompts.loader import PromptLoader

    case_dir = Path(args.case).resolve()
    figs_dir = Path(args.figs).resolve()
    if not case_dir.exists():
        print(f"Case path not found: {case_dir}", file=sys.stderr)
        return 1
    fig_paths: List[str] = [str(p) for p in figs_dir.glob("*.png")]

    settings = get_settings()
    prompts = PromptLoader(settings.prompts_path)
    agent = ResultsInterpreterAgent(model=settings.model, prompt_loader=prompts)

    experiment_requirement = args.requirement.strip() if args.requirement else ""
    description = experiment_requirement if experiment_requirement else f"Interpretation for case {case_dir.name}"

    interp = agent.interpret(
        idea_json={"description": description},
        experiment_spec={"simulation_id": case_dir.name, "case_data": {"name": case_dir.name}},
        experiment_results={"output_dir": str(case_dir), "figures": fig_paths, "returncode": 0},
    )
    rerun = bool(interp.get("rerun_required", False))
    if rerun:
        status = "RERUN"
    elif not bool(interp.get("requirement_met", True)):
        status = "REVISE"
    else:
        status = "PROCEED"

    # Confidence score: a numeric summary of how strongly the interpreter is
    # recommending the label. Lets downstream stages (analysis, paper) apply
    # their own thresholds — e.g. "include all cases with confidence >= 0.5"
    # — rather than being limited to the three-valued PROCEED/RERUN/REVISE
    # categorical. Generic, mode-agnostic.
    #
    # Derivation (LLM-free heuristic):
    #   - If the interpreter agent already returned a 'confidence' field, use it.
    #   - Else map the boolean signals to a {0.0, 0.5, 1.0} scale.
    raw_conf = interp.get("confidence")
    try:
        confidence = float(raw_conf) if raw_conf is not None else None
    except Exception:
        confidence = None
    if confidence is None:
        if status == "PROCEED":
            confidence = 0.8 if interp.get("requirement_met", False) else 0.5
        elif status == "RERUN":
            confidence = 0.15
        else:  # REVISE
            confidence = 0.35
    confidence = max(0.0, min(1.0, float(confidence)))

    decision: Dict[str, Any] = {
        "status": status,
        "confidence": confidence,
        "reason": interp.get("summary", interp.get("reasons", "")),
        "suggested_changes": interp.get("issues", []),
        "raw": interp,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(decision, indent=2), encoding="utf-8")
    append_timeline_event(
        timeline_path,
        {
            "stage": "interpret",
            "case_id": case_dir.name,
            "status": decision["status"],
            "reason": decision.get("reason", ""),
            "suggested_changes": decision.get("suggested_changes", []),
            "output_path": str(out_path),
        },
    )
    print(f"{decision['status']}: {decision['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
