from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from cfd_langgraph.config import get_settings
from cfd_langgraph.prompts.loader import PromptLoader
from cfd_langgraph.agents.analysis_agent import AnalysisAgent
from cfd_langgraph.agents.writer_agent import WriterAgent


def _read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_topic_and_idea(out_dir: Path) -> Tuple[str, dict]:
    ideation_path = out_dir / "ideation_output.json"
    payload = _read_json(ideation_path)
    topic = str(payload.get("topic") or "").strip()
    ideation_bundle = payload.get("ideation", {}) if isinstance(payload, dict) else {}
    idea = ideation_bundle.get("idea", {}) if isinstance(ideation_bundle, dict) else {}

    if not topic:
        analysis_path = out_dir / "analysis_report.md"
        if analysis_path.is_file():
            # First header line fallback
            first = analysis_path.read_text(encoding="utf-8").splitlines()[0].strip()
            topic = first.strip("# ").strip()
    return topic, (idea if isinstance(idea, dict) else {})


def _load_sim_meta(out_dir: Path) -> Dict[str, Dict[str, Any]]:
    """
    Return mapping: simulation_id -> {case_name, description}
    (best-effort; falls back to empty).
    """
    meta: Dict[str, Dict[str, Any]] = {}
    pipeline = _read_json(out_dir / "pipeline_log.json")
    sims = pipeline.get("simulations", []) if isinstance(pipeline, dict) else []
    if not isinstance(sims, list):
        return meta
    for entry in sims:
        if not isinstance(entry, dict):
            continue
        sim = entry.get("simulation", {})
        if not isinstance(sim, dict):
            continue
        sid = str(sim.get("simulation_id") or "").strip()
        if not sid:
            continue
        meta[sid] = {
            "case_name": str(sim.get("case_name") or sid),
            "description": str(sim.get("description") or ""),
            "experiment_idea": sim.get("case_data", None),
        }
    return meta


def _collect_experiments(out_dir: Path) -> List[Dict[str, Any]]:
    sim_meta = _load_sim_meta(out_dir)
    experiments: List[Dict[str, Any]] = []
    for p in sorted(out_dir.iterdir()):
        if not p.is_dir() or not p.name.startswith("exp_"):
            continue
        sim_id = p.name
        req_path = p / "user_requirement.txt"
        foam_output_dir = p / "foam_output"
        experiments.append(
            {
                "simulation_id": sim_id,
                "case_name": sim_meta.get(sim_id, {}).get("case_name", sim_id),
                "description": sim_meta.get(sim_id, {}).get("description", ""),
                "experiment_idea": sim_meta.get(sim_id, {}).get("experiment_idea", None),
                "user_requirement": req_path.read_text(encoding="utf-8") if req_path.is_file() else "",
                "sim_dir": p,
                "foam_output_dir": foam_output_dir,
            }
        )
    return experiments


def _build_visualization_bundle(out_dir: Path, experiments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Build a writer-compatible visualization bundle from existing PNG artifacts.
    - Per-experiment: exp_*/analysis_viz/*.png
    - Cross-experiment: cross_experiment_analysis/*.png (attached to a synthetic entry)
    """
    bundle: List[Dict[str, Any]] = []
    for ex in experiments:
        sim_id = str(ex.get("simulation_id") or "").strip()
        sim_dir = ex.get("sim_dir")
        if not sim_id or not sim_dir:
            continue
        sim_dir = Path(sim_dir)
        viz_dir = sim_dir / "analysis_viz"
        images = sorted(str(p) for p in viz_dir.glob("*.png") if p.is_file())
        if not images:
            # fallback: any png under sim_dir
            images = sorted(str(p) for p in sim_dir.rglob("*.png") if p.is_file())[:50]
        bundle.append(
            {
                "simulation_id": sim_id,
                "case_name": ex.get("case_name", sim_id),
                "visualization": {"ok": bool(images), "images": images},
            }
        )

    cross_dir = out_dir / "cross_experiment_analysis"
    cross_images = sorted(str(p) for p in cross_dir.glob("*.png") if p.is_file())
    if cross_images:
        bundle.append(
            {
                "simulation_id": "cross_experiment",
                "case_name": "cross_experiment_analysis",
                "visualization": {"ok": True, "images": cross_images},
            }
        )
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resume from existing run: cross-experiment processing + writer (no viz regeneration)."
    )
    parser.add_argument("--out-dir", required=True, help="Run output directory (e.g. runs/plume_trial1).")
    parser.add_argument("--verbose", action="store_true", default=False, help="Print detailed logs.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    if not out_dir.is_dir():
        raise FileNotFoundError(f"out_dir not found: {out_dir}")

    analysis_path = out_dir / "analysis_report.md"
    if not analysis_path.is_file():
        raise FileNotFoundError(f"Missing analysis report: {analysis_path}")
    analysis_text = analysis_path.read_text(encoding="utf-8")

    topic, idea = _load_topic_and_idea(out_dir)
    experiments = _collect_experiments(out_dir)
    if not experiments:
        raise FileNotFoundError(f"No exp_* folders found under: {out_dir}")

    settings = get_settings()
    prompt_loader = PromptLoader(settings.prompts_path)

    # 1) Cross-experiment quantitative processing (may generate a script)
    analysis_agent = AnalysisAgent(model=settings.model)
    data_proc = analysis_agent.run_cross_experiment_data_processing(
        topic=topic,
        experiments=experiments,
        out_dir=out_dir,
        verbose=True,  # force visibility; this step is cheap compared to rerendering figures
    )
    report_text = ""
    report_path = Path(str(data_proc.get("report_path") or ""))
    if report_path.is_file():
        try:
            report_text = report_path.read_text(encoding="utf-8")
        except Exception:
            report_text = ""
    interp_text = str(data_proc.get("interpretation_text") or "")
    if not interp_text:
        interp_path = Path(str(data_proc.get("interpretation_path") or ""))
        if interp_path.is_file():
            try:
                interp_text = interp_path.read_text(encoding="utf-8")
            except Exception:
                interp_text = ""

    # 2) Writer (reuses existing analysis report; does NOT regenerate viz)
    writer = WriterAgent(model=settings.model, prompt_loader=prompt_loader)
    exp_ideas_for_writer: List[Dict[str, Any]] = []
    for ex in experiments:
        exp_ideas_for_writer.append(
            {
                "simulation_id": ex.get("simulation_id"),
                "case_name": ex.get("case_name"),
                "experiment_idea": ex.get("experiment_idea"),
            }
        )
    section_context = (
        json.dumps(idea, indent=2, ensure_ascii=False)
        + "\n\nPER-EXPERIMENT IDEAS/CONTEXT:\n"
        + json.dumps(exp_ideas_for_writer, indent=2, ensure_ascii=False)
        + "\n\n"
        + analysis_text
    )
    if report_text.strip():
        section_context += "\n\n---\n\n" + report_text
    if interp_text.strip():
        section_context += "\n\n---\n\n# Cross-Experiment Interpretation (LLM)\n\n" + interp_text

    visualization_bundle = _build_visualization_bundle(out_dir, experiments)
    ideation_literature_bundle = None

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

