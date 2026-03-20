from __future__ import annotations

import argparse
import json
from pathlib import Path

from cfd_langgraph.config import get_settings
from cfd_langgraph.prompts.loader import PromptLoader
from cfd_langgraph.agents.writer_agent import WriterAgent


def _read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Writer-only: generate paper from existing analysis artifacts.")
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Run output directory (e.g. runs/backward_step_trial2).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print writer progress.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    settings = get_settings()
    prompt_loader = PromptLoader(settings.prompts_path)

    writer = WriterAgent(model=settings.model, prompt_loader=prompt_loader)

    # Topic + idea come from ideation_output.json
    ideation_path = out_dir / "ideation_output.json"
    ideation_payload = _read_json(ideation_path)
    ideation_bundle = ideation_payload.get("ideation", {}) if isinstance(ideation_payload, dict) else {}
    topic = str(ideation_payload.get("topic") or "").strip()
    idea = ideation_bundle.get("idea", {}) if isinstance(ideation_bundle, dict) else {}

    if not topic:
        # Fallback: try to infer from analysis_report header
        analysis_path = out_dir / "analysis_report.md"
        if analysis_path.is_file():
            topic = analysis_path.read_text(encoding="utf-8").splitlines()[0].strip("# ").strip()

    analysis_path = out_dir / "analysis_report.md"
    if not analysis_path.is_file():
        raise FileNotFoundError(f"Missing analysis report: {analysis_path}")
    analysis_text = analysis_path.read_text(encoding="utf-8")

    if not idea:
        # If idea is missing, we still can proceed; the writer prompt will be more generic.
        idea = {}

    section_context = json.dumps(idea, indent=2) + "\n\n" + analysis_text

    # Writer-only mode: we don't have a visualization_bundle here (unless you add it).
    # The writer may still produce a paper draft, but it will be more limited.
    visualization_bundle: list[dict] = []
    ideation_literature_bundle = None  # let writer do its own literature citations

    writer.write_paper_with_literature_and_review(
        topic=topic,
        section_context=section_context,
        out_dir=out_dir,
        work_dir=out_dir,
        ideation_literature_bundle=ideation_literature_bundle,
        visualization_bundle=visualization_bundle,
        verbose=bool(args.verbose),
    )


if __name__ == "__main__":
    main()

