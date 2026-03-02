from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from cfd_langchain.agents.analysis_agent import AnalysisAgent
from cfd_langchain.agents.hypothesis_agent import HypothesisAgent
from cfd_langchain.agents.interpreter_agent import ResultsInterpreterAgent
from cfd_langchain.agents.writer_agent import WriterAgent
from cfd_langchain.config import Settings
from cfd_langchain.foam.runner import FoamAgentRunner
from cfd_langchain.ideation import run_ideation
from cfd_langchain.prompts.loader import PromptLoader


@dataclass
class CFDWorkflow:
    settings: Settings
    prompt_loader: PromptLoader

    def __post_init__(self):
        self.hypothesis = HypothesisAgent(self.settings.model, self.prompt_loader)
        self.interpreter = ResultsInterpreterAgent(
            self.settings.model, self.prompt_loader
        )
        self.analysis = AnalysisAgent(self.settings.model)
        self.writer = WriterAgent(self.settings.model, self.prompt_loader)
        self.foam = FoamAgentRunner(
            self.settings.foam_agent_main, self.settings.openfoam_path
        )

    @staticmethod
    def _expand_simulations(
        idea: Dict[str, Any], max_total: int = 50
    ) -> List[Dict[str, Any]]:
        sims: List[Dict[str, Any]] = []
        cases = idea.get("cases", []) if isinstance(idea, dict) else []

        sim_id = 0
        for c in cases:
            case_name = c.get("name", f"case_{sim_id}")
            fuel_list = (
                c.get("fuel_speed_list")
                if isinstance(c.get("fuel_speed_list"), list)
                else [c.get("fuel speed")]
            )
            box_list = (
                c.get("box_size_list")
                if isinstance(c.get("box_size_list"), list)
                else [c.get("box size")]
            )

            for fs in fuel_list:
                for bs in box_list:
                    sim_id += 1
                    sims.append(
                        {
                            "simulation_id": f"sim_{sim_id:03d}",
                            "case_name": case_name,
                            "parameter_value": {"fuel_speed": fs, "box_size": bs},
                            "description": f"Sweep fuel_speed={fs}, box_size={bs}",
                            "visualization": idea.get("post", {}).get(
                                "visualization", ""
                            )
                            if isinstance(idea.get("post"), dict)
                            else "",
                            "case_data": c,
                        }
                    )
                    if len(sims) >= max_total:
                        return sims

        return sims[:max_total]

    @staticmethod
    def _locate_foam_dataset(foam_output_dir: Path) -> Path | None:
        if not foam_output_dir.exists():
            return None

        candidates: List[Path] = []
        for ext in (".vtm", ".vtu", ".vtk"):
            candidates.extend(foam_output_dir.rglob(f"*{ext}"))

        if not candidates:
            return None

        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates[0]

    @staticmethod
    def _write_requirement(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    @staticmethod
    def _normalize_feedback_lines(interp: Dict[str, Any]) -> List[str]:
        lines: List[str] = []
        for key in (
            "rerun_reason",
            "requirement_update",
            "requirement_updates",
            "recommended_requirement_update",
            "suggested_requirement_update",
            "next_requirement",
        ):
            val = interp.get(key)
            if isinstance(val, str) and val.strip():
                lines.append(val.strip())
            elif isinstance(val, list):
                lines.extend(str(x).strip() for x in val if str(x).strip())

        health = interp.get("health")
        if isinstance(health, dict) and health.get("has_error_signals"):
            lines.append(
                "Previous run showed hard error signals. Tighten requirement to avoid invalid setup."
            )

        return lines

    def _revise_requirement_from_feedback(
        self, req: str, interp: Dict[str, Any]
    ) -> Dict[str, Any]:
        feedback = self._normalize_feedback_lines(interp)
        if not feedback:
            verdict = self.hypothesis.llm_validate_requirement(req)
            return {
                "requirement": req,
                "valid": bool(verdict.get("valid", False)),
                "feedback": [],
            }

        revised = self.hypothesis.repair_requirement(
            req,
            issues=[f"Interpreter feedback: {x}" for x in feedback],
            guidance=[
                "Update the requirement to address interpreter-detected execution issues.",
                "Keep the requirement executable by Foam-Agent and do not include visualization instructions.",
            ],
        )
        revised = self.hypothesis._strip_visualization_mentions(revised)

        verdict = self.hypothesis.llm_validate_requirement(revised)
        if verdict.get("valid", False):
            return {"requirement": revised, "valid": True, "feedback": feedback}

        repaired = self.hypothesis.repair_requirement(
            revised,
            issues=verdict.get("issues", []),
            guidance=verdict.get("repair_guidance", []),
        )
        repaired = self.hypothesis._strip_visualization_mentions(repaired)
        repaired_verdict = self.hypothesis.llm_validate_requirement(repaired)
        return {
            "requirement": repaired,
            "valid": bool(repaired_verdict.get("valid", False)),
            "feedback": feedback,
            "validator": repaired_verdict,
        }

    def run_topic(
        self,
        topic: str,
        out_dir: Path,
        execute: bool = False,
        allow_non_executed_artifacts: bool = False,
    ) -> Dict[str, Any]:
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1) User topic -> 2) literature-aware ideation
        ideation_result = run_ideation(self.settings, topic)
        idea = ideation_result.get("idea", {})

        # Expand into experiment list and cap at configured max
        simulations = self._expand_simulations(
            idea,
            max_total=min(
                self.settings.ideation_max_experiments,
                self.settings.workflow_max_experiments_total,
            ),
        )

        pipeline_log: Dict[str, Any] = {
            "topic": topic,
            "ideation": ideation_result,
            "simulations": [],
            "analysis": None,
            "paper": None,
        }

        for sim in simulations:
            sim_dir = out_dir / sim["simulation_id"]
            sim_dir.mkdir(parents=True, exist_ok=True)

            # 3) Hypothesis conversion + logical validator + auto-repair loop
            hyp = self.hypothesis.generate_validated_requirement(
                idea=idea, simulation=sim, max_retries=3
            )
            req_text = hyp["requirement"]
            req_path = sim_dir / "user_requirement.txt"

            run_history = []
            run_result = {}
            interp = {}
            requirement_updates: List[Dict[str, Any]] = []
            req_valid = bool(hyp.get("valid", False))

            # 4) Foam-Agent run; 5) interpreter + rerun loop
            for attempt in range(self.settings.workflow_max_reruns_per_experiment + 1):
                if not req_text or not req_text.strip():
                    run_result = {
                        "planned": False,
                        "skipped": True,
                        "reason": "Empty user requirement; Foam run not attempted.",
                    }
                    run_history.append(run_result)
                    interp = {
                        "rerun_required": False,
                        "rerun_reason": "Requirement is empty; cannot execute.",
                    }
                    break

                if not req_valid:
                    run_result = {
                        "planned": False,
                        "skipped": True,
                        "reason": "Hypothesis requirement failed validation gate; Foam run not attempted.",
                    }
                    run_history.append(run_result)
                    interp = {
                        "rerun_required": False,
                        "rerun_reason": "Requirement invalid; fix hypothesis before execution.",
                    }
                    break

                self._write_requirement(req_path, req_text)
                run_result = self.foam.run(
                    user_requirement_path=req_path,
                    output_dir=sim_dir / "foam_output",
                    project_root=Path.cwd(),
                    execute=execute,
                )

                if execute:
                    foam_dataset = self._locate_foam_dataset(sim_dir / "foam_output")
                    if foam_dataset is not None:
                        viz_request = (
                            "Create a compact set of diagnostic figures sufficient to "
                            "detect obviously broken or empty CFD runs (e.g., zero fields, "
                            "exploding values, or missing flow features). Use robust defaults."
                        )
                        viz_dir = sim_dir / "viz_interpreter"
                        viz_summary = self.analysis.generate_plots_from_foam_data(
                            foam_data_path=foam_dataset,
                            request_text=viz_request,
                            out_dir=viz_dir,
                        )
                        run_result["viz"] = viz_summary

                run_history.append(run_result)

                interp = self.interpreter.interpret(
                    idea_json=idea,
                    experiment_spec=sim,
                    experiment_results=run_result,
                )

                rerun = bool(interp.get("rerun_required", False))
                if (not execute) or (not rerun):
                    break

                revision = self._revise_requirement_from_feedback(req_text, interp)
                next_req = revision["requirement"]
                req_valid = bool(revision.get("valid", False))
                requirement_updates.append(
                    {
                        "attempt": attempt,
                        "feedback": revision.get("feedback", []),
                        "valid": req_valid,
                        "requirement": next_req,
                    }
                )
                req_text = next_req

            pipeline_log["simulations"].append(
                {
                    "simulation": sim,
                    "hypothesis": hyp,
                    "run_history": run_history,
                    "interpreter": interp,
                    "requirement_updates": requirement_updates,
                }
            )

        allow_final_artifacts = execute or allow_non_executed_artifacts
        if allow_final_artifacts:
            # 6) Analysis agent after all experiments, with richer visualization
            viz_bundle = []
            for entry in pipeline_log["simulations"]:
                sim_meta = entry.get("simulation", {})
                sim_id = sim_meta.get("simulation_id")
                if not sim_id:
                    continue
                sim_dir = out_dir / sim_id
                foam_dataset = CFDWorkflow._locate_foam_dataset(
                    sim_dir / "foam_output"
                )
                if foam_dataset is None:
                    continue

                viz_dir = sim_dir / "viz_analysis"
                viz_request = (
                    "Create publication-quality visualizations (contours, slices, lines, "
                    "streamlines, or other helpful plots) that best support analyzing this "
                    "simulation in the context of the overall study."
                )
                viz_summary = self.analysis.generate_plots_from_foam_data(
                    foam_data_path=foam_dataset,
                    request_text=viz_request,
                    out_dir=viz_dir,
                )
                entry["analysis_visualization"] = viz_summary
                viz_bundle.append(
                    {
                        "simulation_id": sim_id,
                        "case_name": sim_meta.get("case_name"),
                        "description": sim_meta.get("description"),
                        "visualization": viz_summary,
                    }
                )

            summary_text = json.dumps(pipeline_log["simulations"], indent=2)
            analysis_text = self.analysis.analyze_text_bundle(
                batch_name="cfd_topic_batch",
                bundle_text=summary_text,
                extra_context=f"Topic: {topic}",
            )
            analysis_path = out_dir / "analysis_report.md"
            self.analysis.save_analysis(analysis_path, analysis_text)
            pipeline_log["analysis"] = str(analysis_path)

            # 7) Writer agent
            paper_context = (
                f"TOPIC:\n{topic}\n\n"
                f"IDEA:\n{json.dumps(idea, indent=2)}\n\n"
                f"ANALYSIS:\n{analysis_text}\n\n"
                "Write paper draft using available artifacts."
            )
            paper_text = self.writer.write_paper_with_literature(
                topic=topic,
                section_context=paper_context,
                ideation_literature_bundle=ideation_result.get("literature_used", []),
                visualization_bundle=viz_bundle,
            )
            paper_path = out_dir / "paper_draft.tex"
            paper_path.write_text(paper_text, encoding="utf-8")
            pipeline_log["paper"] = str(paper_path)
        else:
            pipeline_log["analysis"] = None
            pipeline_log["paper"] = None
            pipeline_log["artifact_gate"] = {
                "execute": execute,
                "allow_non_executed_artifacts": allow_non_executed_artifacts,
                "skipped": ["analysis", "paper"],
            }

        # Save pipeline summary
        (out_dir / "pipeline_log.json").write_text(
            json.dumps(pipeline_log, indent=2), encoding="utf-8"
        )
        return pipeline_log
