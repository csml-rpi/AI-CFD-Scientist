from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict, cast

from langgraph.graph import END, START, StateGraph

from cfd_langgraph.agents.analysis_agent import AnalysisAgent
from cfd_langgraph.agents.hypothesis_agent import HypothesisAgent
from cfd_langgraph.agents.interpreter_agent import ResultsInterpreterAgent
from cfd_langgraph.agents.writer_agent import WriterAgent
from cfd_langgraph.config import Settings
from cfd_langgraph.foam.runner import FoamAgentRunner
from cfd_langgraph.ideation import run_ideation
from cfd_langgraph.prompts.loader import PromptLoader


class WorkflowState(TypedDict, total=False):
    topic: str
    out_dir: Path
    execute: bool
    allow_non_executed_artifacts: bool

    ideation_result: Dict[str, Any]
    idea: Dict[str, Any]
    simulations: List[Dict[str, Any]]
    sim_index: int

    current_sim: Dict[str, Any]
    sim_dir: Path
    hyp: Dict[str, Any]
    req_text: str
    req_valid: bool
    attempt: int
    run_history: List[Dict[str, Any]]
    run_result: Dict[str, Any]
    interp: Dict[str, Any]
    requirement_updates: List[Dict[str, Any]]
    skip_run: bool

    pipeline_log: Dict[str, Any]


@dataclass
class CFDWorkflow:
    settings: Settings
    prompt_loader: PromptLoader
    _app: Any = field(default=None, init=False, repr=False)

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

    def _build_langgraph_app(self):
        def ideate(state: WorkflowState) -> WorkflowState:
            out_dir = state["out_dir"]
            out_dir.mkdir(parents=True, exist_ok=True)
            ideation_result = run_ideation(
                settings=self.settings, research_topic=state["topic"]
            )
            idea = cast(Dict[str, Any], ideation_result.get("idea", {}) or {})
            return {
                "ideation_result": ideation_result,
                "idea": idea,
            }

        def expand_and_init(state: WorkflowState) -> WorkflowState:
            idea = cast(Dict[str, Any], state.get("idea", {}) or {})
            simulations = self._expand_simulations(
                idea,
                max_total=min(
                    self.settings.ideation_max_experiments,
                    self.settings.workflow_max_experiments_total,
                ),
            )
            pipeline_log: Dict[str, Any] = {
                "topic": state["topic"],
                "ideation": state.get("ideation_result", {}),
                "simulations": [],
                "analysis": None,
                "paper": None,
            }
            return {
                "simulations": simulations,
                "sim_index": 0,
                "pipeline_log": pipeline_log,
            }

        def prepare_next_sim(state: WorkflowState) -> WorkflowState:
            idx = int(state.get("sim_index", 0) or 0)
            sims = state.get("simulations", []) or []
            if idx >= len(sims):
                return {}

            sim = sims[idx]
            sim_id = sim.get("simulation_id", f"sim_{idx:03d}")
            sim_dir = state["out_dir"] / str(sim_id)
            sim_dir.mkdir(parents=True, exist_ok=True)
            return {
                "current_sim": sim,
                "sim_dir": sim_dir,
                "hyp": {},
                "req_text": "",
                "req_valid": False,
                "attempt": 0,
                "run_history": [],
                "run_result": {},
                "interp": {},
                "requirement_updates": [],
                "skip_run": False,
            }

        def route_after_prepare(state: WorkflowState) -> str:
            idx = int(state.get("sim_index", 0) or 0)
            sims = state.get("simulations", []) or []
            return "generate_requirement" if idx < len(sims) else "final_artifacts_gate"

        def generate_requirement(state: WorkflowState) -> WorkflowState:
            hyp = self.hypothesis.generate_validated_requirement(
                idea=cast(Dict[str, Any], state.get("idea", {}) or {}),
                simulation=cast(Dict[str, Any], state.get("current_sim", {}) or {}),
                max_retries=3,
            )
            req_text = str(hyp.get("requirement", "") or "")
            req_valid = bool(hyp.get("valid", False))
            return {
                "hyp": hyp,
                "req_text": req_text,
                "req_valid": req_valid,
                "attempt": 0,
            }

        def precheck(state: WorkflowState) -> WorkflowState:
            req_text = str(state.get("req_text", "") or "")
            req_valid = bool(state.get("req_valid", False))
            run_history = list(state.get("run_history", []) or [])

            if not req_text.strip():
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
                return {
                    "skip_run": True,
                    "run_result": run_result,
                    "run_history": run_history,
                    "interp": interp,
                }

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
                return {
                    "skip_run": True,
                    "run_result": run_result,
                    "run_history": run_history,
                    "interp": interp,
                }

            return {"skip_run": False}

        def route_after_precheck(state: WorkflowState) -> str:
            return "append_simulation_log" if state.get("skip_run") else "foam_run"

        def foam_run(state: WorkflowState) -> WorkflowState:
            sim_dir = state["sim_dir"]
            req_path = sim_dir / "user_requirement.txt"
            self._write_requirement(req_path, str(state.get("req_text", "") or ""))

            run_result = self.foam.run(
                user_requirement_path=req_path,
                output_dir=sim_dir / "foam_output",
                project_root=Path.cwd(),
                execute=bool(state.get("execute", False)),
            )

            run_history = list(state.get("run_history", []) or [])
            run_history.append(run_result)
            return {"run_result": run_result, "run_history": run_history}

        def interpret(state: WorkflowState) -> WorkflowState:
            idea = cast(Dict[str, Any], state.get("idea", {}) or {})
            sim = cast(Dict[str, Any], state.get("current_sim", {}) or {})
            run_result = cast(Dict[str, Any], state.get("run_result", {}) or {})
            if bool(state.get("execute", False)):
                sim_dir = state["sim_dir"]
                foam_dataset = self._locate_foam_dataset(sim_dir / "foam_output")
                if foam_dataset is not None:
                    viz_request = (
                        "Create a compact set of diagnostic figures sufficient to "
                        "detect obviously broken or empty CFD runs (e.g., zero fields, "
                        "exploding values, missing boundary conditions, or instability). "
                        "Choose appropriate plot types (contours/slices/glyphs/streamlines/lines/etc.) based on available arrays. "
                        "Use robust defaults and save images."
                    )
                    viz_dir = sim_dir / "viz_interpreter"
                    viz_summary = self.analysis.generate_plots_from_foam_data(
                        foam_data_path=foam_dataset,
                        request_text=viz_request,
                        out_dir=viz_dir,
                    )
                    run_result = {**run_result, "viz": viz_summary}
                else:
                    run_result = {
                        **run_result,
                        "viz": {
                            "ok": False,
                            "error": "No readable VTK dataset found for interpreter diagnostics.",
                        },
                    }
            interp = self.interpreter.interpret(
                idea_json=idea,
                experiment_spec=sim,
                experiment_results=run_result,
            )
            return {"interp": interp, "run_result": run_result}

        def route_after_interpret(state: WorkflowState) -> str:
            if not bool(state.get("execute", False)):
                return "append_simulation_log"

            rerun = bool((state.get("interp", {}) or {}).get("rerun_required", False))
            attempt = int(state.get("attempt", 0) or 0)
            max_reruns = int(self.settings.workflow_max_reruns_per_experiment)
            if rerun and attempt < max_reruns:
                return "revise_requirement"
            return "append_simulation_log"

        def revise_requirement(state: WorkflowState) -> WorkflowState:
            req_text = str(state.get("req_text", "") or "")
            interp = cast(Dict[str, Any], state.get("interp", {}) or {})
            revision = self._revise_requirement_from_feedback(req_text, interp)
            next_req = str(revision.get("requirement", "") or "")
            req_valid = bool(revision.get("valid", False))

            requirement_updates = list(state.get("requirement_updates", []) or [])
            attempt = int(state.get("attempt", 0) or 0)
            requirement_updates.append(
                {
                    "attempt": attempt,
                    "feedback": revision.get("feedback", []),
                    "valid": req_valid,
                    "requirement": next_req,
                }
            )

            return {
                "req_text": next_req,
                "req_valid": req_valid,
                "requirement_updates": requirement_updates,
                "attempt": attempt + 1,
            }

        def append_simulation_log(state: WorkflowState) -> WorkflowState:
            pipeline_log = cast(Dict[str, Any], state.get("pipeline_log", {}) or {})
            sims_list = cast(List[Dict[str, Any]], pipeline_log.get("simulations", []) or [])
            sims_list.append(
                {
                    "simulation": state.get("current_sim", {}),
                    "hypothesis": state.get("hyp", {}),
                    "run_history": state.get("run_history", []),
                    "interpreter": state.get("interp", {}),
                    "requirement_updates": state.get("requirement_updates", []),
                }
            )
            pipeline_log["simulations"] = sims_list
            return {"pipeline_log": pipeline_log, "sim_index": int(state.get("sim_index", 0) or 0) + 1}

        def final_artifacts_gate(state: WorkflowState) -> WorkflowState:
            # no-op; routing decides next node
            return {}

        def route_after_final_gate(state: WorkflowState) -> str:
            allow_final = bool(state.get("execute", False)) or bool(
                state.get("allow_non_executed_artifacts", False)
            )
            return "analysis_and_writer" if allow_final else "save_pipeline_log"

        def analysis_and_writer(state: WorkflowState) -> WorkflowState:
            out_dir = state["out_dir"]
            topic = state["topic"]
            pipeline_log = cast(Dict[str, Any], state.get("pipeline_log", {}) or {})

            viz_bundle: List[Dict[str, Any]] = []
            for entry in pipeline_log.get("simulations", []) or []:
                sim_meta = entry.get("simulation", {}) if isinstance(entry, dict) else {}
                sim_id = sim_meta.get("simulation_id")
                if not sim_id:
                    continue
                sim_dir = out_dir / str(sim_id)
                foam_dataset = CFDWorkflow._locate_foam_dataset(sim_dir / "foam_output")
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

            summary_text = json.dumps(pipeline_log.get("simulations", []), indent=2)
            analysis_text = self.analysis.analyze_text_bundle(
                batch_name="cfd_topic_batch",
                bundle_text=summary_text,
                extra_context=f"Topic: {topic}",
            )
            analysis_path = out_dir / "analysis_report.md"
            self.analysis.save_analysis(analysis_path, analysis_text)
            pipeline_log["analysis"] = str(analysis_path)

            paper_context = (
                f"TOPIC:\n{topic}\n\n"
                f"IDEA:\n{json.dumps(state.get('idea', {}), indent=2)}\n\n"
                f"ANALYSIS:\n{analysis_text}\n\n"
                "Write paper draft using available artifacts."
            )
            paper_text = self.writer.write_paper_with_literature(
                topic=topic,
                section_context=paper_context,
                ideation_literature_bundle=(state.get("ideation_result", {}) or {}).get(
                    "literature_used", []
                ),
                visualization_bundle=viz_bundle,
            )
            paper_path = out_dir / "paper_draft.tex"
            paper_path.write_text(paper_text, encoding="utf-8")
            pipeline_log["paper"] = str(paper_path)

            return {"pipeline_log": pipeline_log}

        def save_pipeline_log(state: WorkflowState) -> WorkflowState:
            out_dir = state["out_dir"]
            pipeline_log = cast(Dict[str, Any], state.get("pipeline_log", {}) or {})

            allow_final = bool(state.get("execute", False)) or bool(
                state.get("allow_non_executed_artifacts", False)
            )
            if not allow_final:
                pipeline_log["analysis"] = None
                pipeline_log["paper"] = None
                pipeline_log["artifact_gate"] = {
                    "execute": bool(state.get("execute", False)),
                    "allow_non_executed_artifacts": bool(
                        state.get("allow_non_executed_artifacts", False)
                    ),
                    "skipped": ["analysis", "paper"],
                }

            (out_dir / "pipeline_log.json").write_text(
                json.dumps(pipeline_log, indent=2), encoding="utf-8"
            )
            return {"pipeline_log": pipeline_log}

        g = StateGraph(WorkflowState)
        g.add_node("ideate", ideate)
        g.add_node("expand_and_init", expand_and_init)
        g.add_node("prepare_next_sim", prepare_next_sim)
        g.add_node("generate_requirement", generate_requirement)
        g.add_node("precheck", precheck)
        g.add_node("foam_run", foam_run)
        g.add_node("interpret", interpret)
        g.add_node("revise_requirement", revise_requirement)
        g.add_node("append_simulation_log", append_simulation_log)
        g.add_node("final_artifacts_gate", final_artifacts_gate)
        g.add_node("analysis_and_writer", analysis_and_writer)
        g.add_node("save_pipeline_log", save_pipeline_log)

        g.add_edge(START, "ideate")
        g.add_edge("ideate", "expand_and_init")
        g.add_edge("expand_and_init", "prepare_next_sim")
        g.add_conditional_edges("prepare_next_sim", route_after_prepare)

        g.add_edge("generate_requirement", "precheck")
        g.add_conditional_edges("precheck", route_after_precheck)
        g.add_edge("foam_run", "interpret")
        g.add_conditional_edges("interpret", route_after_interpret)
        g.add_edge("revise_requirement", "precheck")
        g.add_edge("append_simulation_log", "prepare_next_sim")

        g.add_conditional_edges("final_artifacts_gate", route_after_final_gate)
        g.add_edge("analysis_and_writer", "save_pipeline_log")
        g.add_edge("save_pipeline_log", END)

        return g.compile()

    def _get_app(self):
        if self._app is None:
            self._app = self._build_langgraph_app()
        return self._app

    def run_topic(
        self,
        topic: str,
        out_dir: Path,
        execute: bool = False,
        allow_non_executed_artifacts: bool = False,
    ) -> Dict[str, Any]:
        initial_state: WorkflowState = {
            "topic": topic,
            "out_dir": out_dir,
            "execute": execute,
            "allow_non_executed_artifacts": allow_non_executed_artifacts,
        }
        final_state = cast(WorkflowState, self._get_app().invoke(initial_state))
        return cast(Dict[str, Any], final_state.get("pipeline_log", {}) or {})
