from __future__ import annotations

import argparse
import json
from pathlib import Path

from cfd_langgraph.config import get_settings
from cfd_langgraph.ideation import run_ideation
from cfd_langgraph.llm.token_stats import get_token_stats, estimate_sonnet_46_cost_usd
from cfd_langgraph.prompts.loader import PromptLoader
from cfd_langgraph.workflow.graph import CFDWorkflow


def main():
    parser = argparse.ArgumentParser(
        description="CFD Scientist workflow (LangGraph-orchestrated)"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ideate = sub.add_parser("ideate", help="Run literature-aware ideation only")
    p_ideate.add_argument(
        "--topic",
        default="Parametric CFD study in a 2D domain.",
        help="Research topic for ideation",
    )
    p_ideate.add_argument(
        "--out", default="-", help="Output JSON file path or '-' for stdout"
    )

    p_topic = sub.add_parser("run-topic", help="Run full end-to-end flow for a topic")
    p_topic.add_argument("--topic", required=True, help="Research topic")
    p_topic.add_argument(
        "--out-dir", required=True, help="Output directory for artifacts"
    )
    p_topic.add_argument(
        "--execute", action="store_true", help="Actually execute Foam-Agent runs"
    )
    p_topic.add_argument(
        "--allow-non-executed-artifacts",
        action="store_true",
        help="Allow analysis/paper generation when --execute is not set",
    )
    p_topic.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print progress from each agent (default: True)",
    )
    p_topic.add_argument(
        "--no-verbose",
        action="store_true",
        dest="no_verbose",
        help="Disable verbose agent output",
    )

    p_restart = sub.add_parser(
        "restart-topic",
        help="Restart pipeline AFTER Foam-Agent runs (reuse existing foam_output and re-run interpreter+analysis+writer).",
    )
    p_restart.add_argument(
        "--out-dir", required=True, help="Existing output directory for the previous run"
    )
    p_restart.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print progress from each agent (default: True)",
    )
    p_restart.add_argument(
        "--no-verbose",
        action="store_true",
        dest="no_verbose",
        help="Disable verbose agent output",
    )

    p_resume = sub.add_parser(
        "resume-topic",
        help="Resume pipeline after partial Foam-Agent runs (finish remaining runs, then interpreter+analysis+writer).",
    )
    p_resume.add_argument(
        "--out-dir", required=True, help="Existing output directory for the previous run"
    )
    p_resume.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print progress from each agent (default: True)",
    )
    p_resume.add_argument(
        "--no-verbose",
        action="store_true",
        dest="no_verbose",
        help="Disable verbose agent output",
    )

    args = parser.parse_args()
    settings = get_settings()
    print(
        f"[CFD-WORKFLOW] Using LLM provider/model: provider={settings.llm_provider} model={settings.model}"
    )

    if args.cmd == "ideate":
        verbose = not getattr(args, "no_verbose", False)
        result = run_ideation(settings=settings, research_topic=args.topic, verbose=verbose)
        text = json.dumps(result, indent=2)
        if args.out == "-":
            print(text)
        else:
            with open(args.out, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"Saved: {args.out}")
        return

    if args.cmd == "run-topic":
        verbose = getattr(args, "no_verbose", False) is False and getattr(args, "verbose", True)
        wf = CFDWorkflow(
            settings=settings, prompt_loader=PromptLoader(settings.prompts_path)
        )
        result = wf.run_topic(
            topic=args.topic,
            out_dir=Path(args.out_dir).expanduser().resolve(),
            execute=args.execute,
            allow_non_executed_artifacts=args.allow_non_executed_artifacts,
            verbose=verbose,
        )
        stats = get_token_stats()
        cost = estimate_sonnet_46_cost_usd(
            prompt_tokens=stats.prompt_tokens,
            completion_tokens=stats.completion_tokens,
        )
        print(
            json.dumps(
                {
                    "ok": True,
                    "out_dir": args.out_dir,
                    "analysis": result.get("analysis"),
                    "paper": result.get("paper"),
                    "llm_usage": {
                        "model": settings.model,
                        "prompt_tokens": stats.prompt_tokens,
                        "completion_tokens": stats.completion_tokens,
                        "pricing": {
                            "prompt_usd_per_1M_tokens": 3.0,
                            "completion_usd_per_1M_tokens": 15.0,
                        },
                        "cost_estimate_usd": cost,
                    },
                },
                indent=2,
            )
        )
        return

    if args.cmd == "restart-topic":
        verbose = getattr(args, "no_verbose", False) is False and getattr(
            args, "verbose", True
        )
        wf = CFDWorkflow(
            settings=settings, prompt_loader=PromptLoader(settings.prompts_path)
        )
        result = wf.restart_from_foam(out_dir=Path(args.out_dir).expanduser().resolve(), verbose=verbose)
        print(
            json.dumps(
                {
                    "ok": True,
                    "out_dir": args.out_dir,
                    "analysis": result.get("analysis"),
                    "paper": result.get("paper"),
                },
                indent=2,
            )
        )
        return

    if args.cmd == "resume-topic":
        verbose = getattr(args, "no_verbose", False) is False and getattr(
            args, "verbose", True
        )
        wf = CFDWorkflow(
            settings=settings, prompt_loader=PromptLoader(settings.prompts_path)
        )
        result = wf.resume_after_runs(out_dir=Path(args.out_dir).expanduser().resolve(), verbose=verbose)
        print(
            json.dumps(
                {
                    "ok": True,
                    "out_dir": args.out_dir,
                    "analysis": result.get("analysis"),
                    "paper": result.get("paper"),
                },
                indent=2,
            )
        )
        return


if __name__ == "__main__":
    main()
