from __future__ import annotations

import argparse
import json
from pathlib import Path

from cfd_langgraph.config import get_settings
from cfd_langgraph.ideation import run_ideation
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
        default="Study the effect of fuel velocity and inlet box sizes in 2D small pool fire.",
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

    args = parser.parse_args()
    settings = get_settings()

    if args.cmd == "ideate":
        result = run_ideation(settings=settings, research_topic=args.topic)
        text = json.dumps(result, indent=2)
        if args.out == "-":
            print(text)
        else:
            with open(args.out, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"Saved: {args.out}")
        return

    if args.cmd == "run-topic":
        wf = CFDWorkflow(
            settings=settings, prompt_loader=PromptLoader(settings.prompts_path)
        )
        result = wf.run_topic(
            topic=args.topic,
            out_dir=Path(args.out_dir),
            execute=args.execute,
            allow_non_executed_artifacts=args.allow_non_executed_artifacts,
        )
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
