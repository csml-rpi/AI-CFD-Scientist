from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict, cast

from langgraph.graph import END, START, StateGraph

from cfd_langgraph.agents.analysis_agent import AnalysisAgent
from cfd_langgraph.agents.hypothesis_agent import HypothesisAgent
from cfd_langgraph.agents.interpreter_agent import ResultsInterpreterAgent
from cfd_langgraph.agents.rerun_analysis_agent import RerunAnalysisAgent
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
    verbose: bool

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

    # Rerun loop (batch mode, after interpreter_batch)
    rerun_queue: List[str]
    rerun_idx: int
    rerun_round: int


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
        self.rerun_analysis = RerunAnalysisAgent(self.settings.model, self.prompt_loader)
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

        # Preferred canonical schema: experiments[]
        experiments = idea.get("experiments", []) if isinstance(idea, dict) else []
        if isinstance(experiments, list) and experiments:
            for i, e in enumerate(experiments[:max_total], 1):
                if not isinstance(e, dict):
                    continue
                sims.append(
                    {
                        "simulation_id": e.get("experiment_id", f"sim_{i:03d}"),
                        "case_name": e.get("name", f"experiment_{i:03d}"),
                        "parameter_value": e.get("parameters", {}),
                        "description": e.get("notes", e.get("name", "")),
                        "visualization": "",
                        "case_data": e,
                    }
                )
            return sims[:max_total]

        # Generic fallback path: cases[] without explicit experiments[]
        cases = idea.get("cases", []) if isinstance(idea, dict) else []
        sim_id = 0
        for c in cases:
            if not isinstance(c, dict):
                continue
            sim_id += 1
            sims.append(
                {
                    "simulation_id": f"sim_{sim_id:03d}",
                    "case_name": c.get("name", f"case_{sim_id}"),
                    "parameter_value": c.get("parameters", {}),
                    "description": c.get("description", ""),
                    "visualization": "",
                    "case_data": c,
                }
            )
            if len(sims) >= max_total:
                return sims

        return sims[:max_total]

    @staticmethod
    def _locate_foam_dataset(foam_output_dir: Path) -> Path | None:
        """
        Use OpenFOAM-native loading path for PyVista:
        - Prefer an existing .foam file
        - If none exists, create one in the case folder and return it
        """
        if not foam_output_dir.exists():
            return None

        foam_files = sorted(foam_output_dir.rglob("*.foam"), key=lambda p: p.stat().st_mtime, reverse=True)
        if foam_files:
            return foam_files[0]

        # Create a marker .foam file so PyVista can load the OpenFOAM case directly.
        marker = foam_output_dir / "case.foam"
        try:
            marker.touch(exist_ok=True)
            return marker
        except Exception:
            return None

    @staticmethod
    def _write_requirement(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    @staticmethod
    def _write_json(path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @staticmethod
    def _read_json(path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

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


    @staticmethod
    def _viz_is_acceptable(interp: Dict[str, Any], run_result: Dict[str, Any]) -> bool:
        """Best-effort viz quality gate.

        Accepts viz if generation succeeded and interpreter does not explicitly request viz redo.
        """
        viz = run_result.get("viz") if isinstance(run_result, dict) else None
        viz_ok = isinstance(viz, dict) and bool(viz.get("ok", False))

        # Require at least one successfully generated diagnostic image.
        min_ok_images = 1
        ok_image_count = 0

        if isinstance(viz, dict):
            results = viz.get("results", [])
            if isinstance(results, list):
                for r in results:
                    if not isinstance(r, dict) or not bool(r.get("ok", False)):
                        continue
                    out = r.get("output")
                    if isinstance(out, str) and out.strip() and Path(out).exists():
                        ok_image_count += 1

        viz_ok = viz_ok and (ok_image_count >= min_ok_images)

        if not isinstance(interp, dict):
            return bool(viz_ok)

        explicit_bad = any(
            interp.get(k) is False
            for k in ("viz_ok", "viz_quality_ok", "visualization_ok", "viz_acceptable")
        )
        explicit_redo = any(
            bool(interp.get(k, False))
            for k in ("redo_viz", "viz_rerun_required", "regenerate_visualization")
        )

        return bool(viz_ok) and (not explicit_bad) and (not explicit_redo)

    @staticmethod
    def _viz_retry_reason(interp: Dict[str, Any], run_result: Dict[str, Any]) -> str:
        if isinstance(interp, dict):
            for k in (
                "viz_rerun_reason",
                "viz_reason",
                "visualization_issue",
                "visualization_feedback",
                "rerun_reason",
            ):
                v = interp.get(k)
                if isinstance(v, str) and v.strip():
                    return v.strip()

        viz = run_result.get("viz") if isinstance(run_result, dict) else None
        if isinstance(viz, dict):
            err = viz.get("error")
            if isinstance(err, str) and err.strip():
                return err.strip()
        return "Visualization diagnostics were not acceptable; regenerating plots."

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
        def log_stage(msg: str) -> None:
            print(f"[CFD-WORKFLOW] {msg}", flush=True)

        def ideate(state: WorkflowState) -> WorkflowState:
            log_stage("START ideation")
            out_dir = state["out_dir"]
            out_dir.mkdir(parents=True, exist_ok=True)
            ideation_result = run_ideation(
                settings=self.settings,
                research_topic=state["topic"],
                verbose=bool(state.get("verbose", True)),
            )
            idea = cast(Dict[str, Any], ideation_result.get("idea", {}) or {})
            self._write_json(
                out_dir / "ideation_output.json",
                {
                    "topic": state["topic"],
                    "ideation": ideation_result,
                    "experiments": idea.get("experiments", []) if isinstance(idea, dict) else [],
                },
            )
            log_stage("END ideation")
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

            # Batch hypothesis generation for ALL experiments before any Foam-Agent execution.
            log_stage(f"START hypothesis-batch :: total={len(simulations)}")
            hyp_items: List[Dict[str, Any]] = []
            updated_sims: List[Dict[str, Any]] = []
            for sim in simulations:
                sim_id = str(sim.get("simulation_id", ""))
                log_stage(f"START hypothesis :: {sim_id}")
                hyp = self.hypothesis.generate_validated_requirement(
                    idea=idea,
                    simulation=sim,
                    run_topic=str(state.get("topic", "") or ""),
                    max_retries=3,
                )
                req_text = str(hyp.get("requirement", "") or "")
                req_valid = bool(hyp.get("valid", False))
                log_stage(f"END hypothesis :: {sim_id} (valid={req_valid})")

                sim_enriched = dict(sim)
                sim_enriched["_hyp"] = hyp
                sim_enriched["_req_text"] = req_text
                sim_enriched["_req_valid"] = req_valid
                updated_sims.append(sim_enriched)

                hyp_items.append(
                    {
                        "simulation_id": sim_id,
                        "case_name": sim.get("case_name"),
                        "experiment": sim,
                        "prompt_for_foamagent": req_text,
                        "valid": req_valid,
                        "history": hyp.get("history", []),
                    }
                )

            self._write_json(
                state["out_dir"] / "hypothesis_output.json",
                {
                    "topic": state["topic"],
                    "count": len(hyp_items),
                    "items": hyp_items,
                },
            )
            log_stage("END hypothesis-batch")

            pipeline_log: Dict[str, Any] = {
                "topic": state["topic"],
                "ideation": state.get("ideation_result", {}),
                "simulations": [],
                "analysis": None,
                "paper": None,
            }
            return {
                "simulations": updated_sims,
                "sim_index": 0,
                "pipeline_log": pipeline_log,
            }

        def prepare_next_sim(state: WorkflowState) -> WorkflowState:
            idx = int(state.get("sim_index", 0) or 0)
            sims = state.get("simulations", []) or []
            if idx >= len(sims):
                return {}

            sim = sims[idx]
            log_stage(f"START case {idx + 1}/{len(sims)} :: {sim.get('simulation_id', 'unknown')}")
            sim_id = sim.get("simulation_id", f"sim_{idx:03d}")
            sim_dir = state["out_dir"] / str(sim_id)
            sim_dir.mkdir(parents=True, exist_ok=True)
            return {
                "current_sim": sim,
                "sim_dir": sim_dir,
                "hyp": cast(Dict[str, Any], sim.get("_hyp", {}) or {}),
                "req_text": str(sim.get("_req_text", "") or ""),
                "req_valid": bool(sim.get("_req_valid", False)),
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
            return "precheck" if idx < len(sims) else "final_artifacts_gate"

        def generate_requirement(state: WorkflowState) -> WorkflowState:
            current_sim = cast(Dict[str, Any], state.get("current_sim", {}) or {})
            log_stage(f"START hypothesis :: {current_sim.get('simulation_id', 'unknown')}")
            hyp = self.hypothesis.generate_validated_requirement(
                idea=cast(Dict[str, Any], state.get("idea", {}) or {}),
                simulation=cast(Dict[str, Any], state.get("current_sim", {}) or {}),
                run_topic=str(state.get("topic", "") or ""),
                max_retries=3,
                verbose=bool(state.get("verbose", True)),
            )
            req_text = str(hyp.get("requirement", "") or "")
            req_valid = bool(hyp.get("valid", False))

            out_dir = state["out_dir"]
            current_sim = cast(Dict[str, Any], state.get("current_sim", {}) or {})
            sim_id = str(current_sim.get("simulation_id", ""))
            hypothesis_json_path = out_dir / "hypothesis_output.json"
            hypothesis_payload = self._read_json(hypothesis_json_path)
            items = hypothesis_payload.get("items", []) if isinstance(hypothesis_payload.get("items", []), list) else []
            item = {
                "simulation_id": sim_id,
                "case_name": current_sim.get("case_name"),
                "experiment": current_sim,
                "prompt_for_foamagent": req_text,
                "valid": req_valid,
                "history": hyp.get("history", []),
            }
            replaced = False
            for i, existing in enumerate(items):
                if isinstance(existing, dict) and str(existing.get("simulation_id", "")) == sim_id:
                    items[i] = item
                    replaced = True
                    break
            if not replaced:
                items.append(item)
            self._write_json(
                hypothesis_json_path,
                {
                    "topic": state["topic"],
                    "count": len(items),
                    "items": items,
                },
            )
            log_stage(f"END hypothesis :: {sim_id} (valid={req_valid})")

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
            sim = cast(Dict[str, Any], state.get("current_sim", {}) or {})
            sim_id = sim.get("simulation_id", "unknown")
            log_stage(f"START foamagent :: {sim_id}")
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
            rc = run_result.get("returncode") if isinstance(run_result, dict) else None
            log_stage(f"END foamagent :: {sim_id} (returncode={rc})")
            return {"run_result": run_result, "run_history": run_history}

        def interpret(state: WorkflowState) -> WorkflowState:
            idea = cast(Dict[str, Any], state.get("idea", {}) or {})
            sim = cast(Dict[str, Any], state.get("current_sim", {}) or {})
            sim_id = sim.get("simulation_id", "unknown")
            log_stage(f"START interpreter :: {sim_id}")

            run_result = cast(Dict[str, Any], state.get("run_result", {}) or {})
            interp: Dict[str, Any] = {}

            # Interpreter owns viz: it generates PyVista script, runs it, retries viz up to 25x,
            # then interprets with user_req + images and sets rerun_required.
            interp = self.interpreter.interpret(
                idea_json=idea,
                experiment_spec=sim,
                experiment_results=run_result,
                verbose=bool(state.get("verbose", True)),
            )
            run_result["viz_attempts"] = interp.get("viz_attempts", []) if isinstance(interp, dict) else []
            run_result["viz_final_ok"] = bool(interp.get("viz_ok", False)) if isinstance(interp, dict) else False
            run_result["viz"] = {
                "ok": run_result["viz_final_ok"],
                "from_interpreter": True,
            }

            log_stage(
                f"END interpreter :: {sim_id} (rerun_required={bool(interp.get('rerun_required', False)) if isinstance(interp, dict) else False})"
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
                    "attempt": int(state.get("attempt", 0) or 0),
                }
            )
            pipeline_log["simulations"] = sims_list

            out_dir = state["out_dir"]

            foam_cases: List[Dict[str, Any]] = []
            for entry in sims_list:
                sim = entry.get("simulation", {}) if isinstance(entry, dict) else {}
                run_history = entry.get("run_history", []) if isinstance(entry, dict) else []
                latest_run = run_history[-1] if isinstance(run_history, list) and run_history else {}
                status = "unknown"
                if isinstance(latest_run, dict):
                    if latest_run.get("skipped"):
                        status = "skipped"
                    elif latest_run.get("planned"):
                        status = "planned"
                    elif latest_run.get("returncode") == 0:
                        status = "success"
                    elif latest_run.get("returncode") is not None:
                        status = "failed"
                foam_cases.append(
                    {
                        "simulation_id": sim.get("simulation_id"),
                        "case_name": sim.get("case_name"),
                        "status": status,
                        "latest_run": latest_run,
                    }
                )

            self._write_json(
                out_dir / "foamagent_output.json",
                {
                    "total_cases": len(foam_cases),
                    "success_count": sum(1 for c in foam_cases if c.get("status") == "success"),
                    "failed_count": sum(1 for c in foam_cases if c.get("status") == "failed"),
                    "skipped_count": sum(1 for c in foam_cases if c.get("status") == "skipped"),
                    "planned_count": sum(1 for c in foam_cases if c.get("status") == "planned"),
                    "cases": foam_cases,
                },
            )

            sim = cast(Dict[str, Any], state.get("current_sim", {}) or {})
            log_stage(f"END simulation :: {sim.get('simulation_id', 'unknown')}")

            return {"pipeline_log": pipeline_log, "sim_index": int(state.get("sim_index", 0) or 0) + 1}

        def final_artifacts_gate(state: WorkflowState) -> WorkflowState:
            # no-op; routing decides next node
            return {}

        def interpret_batch(state: WorkflowState) -> WorkflowState:
            if not bool(state.get("execute", False)):
                return {}

            log_stage("START interpreter-batch")
            out_dir = state["out_dir"]
            idea = cast(Dict[str, Any], state.get("idea", {}) or {})
            pipeline_log = cast(Dict[str, Any], state.get("pipeline_log", {}) or {})
            sims_list = cast(List[Dict[str, Any]], pipeline_log.get("simulations", []) or [])

            interp_cases: List[Dict[str, Any]] = []
            for entry in sims_list:
                sim = entry.get("simulation", {}) if isinstance(entry, dict) else {}
                sim_id = sim.get("simulation_id", "unknown")
                log_stage(f"START interpreter :: {sim_id}")

                run_history = entry.get("run_history", []) if isinstance(entry, dict) else []
                run_result = run_history[-1] if isinstance(run_history, list) and run_history else {}
                run_result = cast(Dict[str, Any], run_result if isinstance(run_result, dict) else {})

                sim_dir = out_dir / str(sim_id)
                if not run_result.get("output_dir") and (sim_dir / "foam_output").exists():
                    run_result["output_dir"] = str(sim_dir / "foam_output")

                interp = self.interpreter.interpret(
                    idea_json=idea,
                    experiment_spec=cast(Dict[str, Any], sim or {}),
                    experiment_results=run_result,
                )
                run_result["viz_attempts"] = interp.get("viz_attempts", []) if isinstance(interp, dict) else []
                run_result["viz_final_ok"] = bool(interp.get("viz_ok", False)) if isinstance(interp, dict) else False
                run_result["viz"] = {
                    "ok": run_result["viz_final_ok"],
                    "from_interpreter": True,
                }

                entry["interpreter"] = interp
                if isinstance(run_history, list) and run_history:
                    run_history[-1] = run_result
                    entry["run_history"] = run_history

                rerun_required = bool(interp.get("rerun_required", False)) if isinstance(interp, dict) else False
                interp_cases.append(
                    {
                        "simulation_id": sim.get("simulation_id"),
                        "case_name": sim.get("case_name"),
                        "rerun_required": rerun_required,
                        "rerun_reason": interp.get("rerun_reason") if isinstance(interp, dict) else None,
                        "status": ("rerun" if rerun_required else "ok"),
                        "attempts_used": int(entry.get("attempt", 0) if isinstance(entry, dict) else 0),
                        "final_returncode": run_result.get("returncode"),
                        "viz_final_ok": run_result.get("viz_final_ok"),
                        "interpreter": interp,
                    }
                )
                log_stage(f"END interpreter :: {sim_id} (rerun_required={rerun_required})")

            pipeline_log["simulations"] = sims_list
            self._write_json(
                out_dir / "interpreter_output.json",
                {
                    "total_cases": len(interp_cases),
                    "rerun_required_count": sum(1 for c in interp_cases if c.get("rerun_required") is True),
                    "ok_count": sum(1 for c in interp_cases if c.get("status") == "ok"),
                    "cases": interp_cases,
                },
            )
            log_stage("END interpreter-batch")
            return {"pipeline_log": pipeline_log}

        def rerun_round_start(state: WorkflowState) -> WorkflowState:
            """Start a rerun round by building a queue of cases to rerun."""
            pipeline_log = cast(Dict[str, Any], state.get("pipeline_log", {}) or {})
            sims_list = cast(List[Dict[str, Any]], pipeline_log.get("simulations", []) or [])
            max_reruns = int(self.settings.workflow_max_reruns_per_experiment)
            queue: List[str] = []
            for entry in sims_list:
                if not isinstance(entry, dict):
                    continue
                sim = entry.get("simulation", {}) if isinstance(entry.get("simulation", {}), dict) else {}
                sim_id = str(sim.get("simulation_id", "") or "")
                if not sim_id:
                    continue
                interp = entry.get("interpreter", {}) if isinstance(entry.get("interpreter", {}), dict) else {}
                rerun_required = bool(interp.get("rerun_required", False)) if isinstance(interp, dict) else False
                attempt_used = int(entry.get("attempt", 0) or 0)
                if rerun_required and attempt_used < max_reruns:
                    queue.append(sim_id)

            round_idx = int(state.get("rerun_round", 0) or 0) + 1
            log_stage(f"RERUN-ROUND-START round={round_idx} queue={queue}")
            return {"rerun_queue": queue, "rerun_idx": 0, "rerun_round": round_idx}

        def route_after_rerun_round_start(state: WorkflowState) -> str:
            queue = state.get("rerun_queue", []) or []
            idx = int(state.get("rerun_idx", 0) or 0)
            return "rerun_run_one" if idx < len(queue) else "analysis_and_writer"

        def _get_entry_by_sim_id(
            sims_list: List[Dict[str, Any]], sim_id: str
        ) -> Optional[Dict[str, Any]]:
            for entry in sims_list:
                if not isinstance(entry, dict):
                    continue
                sim = entry.get("simulation", {}) if isinstance(entry.get("simulation", {}), dict) else {}
                if str(sim.get("simulation_id", "")) == str(sim_id):
                    return entry
            return None

        def rerun_run_one(state: WorkflowState) -> WorkflowState:
            """
            For one queued case: revise requirement from interpreter report, run Foam-Agent, re-interpret.
            """
            pipeline_log = cast(Dict[str, Any], state.get("pipeline_log", {}) or {})
            sims_list = cast(List[Dict[str, Any]], pipeline_log.get("simulations", []) or [])
            queue = list(state.get("rerun_queue", []) or [])
            idx = int(state.get("rerun_idx", 0) or 0)
            if idx >= len(queue):
                return {"pipeline_log": pipeline_log, "rerun_idx": idx}

            sim_id = str(queue[idx])
            entry = _get_entry_by_sim_id(sims_list, sim_id)
            if entry is None:
                log_stage(f"RERUN-SKIP missing entry for sim_id={sim_id}")
                return {"pipeline_log": pipeline_log, "rerun_idx": idx + 1}

            sim = entry.get("simulation", {}) if isinstance(entry.get("simulation", {}), dict) else {}
            interp = entry.get("interpreter", {}) if isinstance(entry.get("interpreter", {}), dict) else {}

            # Determine last requirement text (prefer saved file, fallback to hypothesis).
            out_dir = state["out_dir"]
            sim_dir = out_dir / str(sim_id)
            req_path = sim_dir / "user_requirement.txt"
            if req_path.exists():
                try:
                    current_req = req_path.read_text(encoding="utf-8")
                except Exception:
                    current_req = ""
            else:
                hyp = entry.get("hypothesis", {}) if isinstance(entry.get("hypothesis", {}), dict) else {}
                current_req = str(hyp.get("requirement", "") or hyp.get("prompt_for_foamagent", "") or "")

            log_stage(f"START rerun_analysis :: {sim_id}")
            revision = self.rerun_analysis.revise_requirement(
                current_req,
                cast(Dict[str, Any], interp or {}),
                verbose=bool(state.get("verbose", True)),
            )
            next_req = str(revision.get("requirement", "") or "")
            req_valid = bool(revision.get("valid", False))
            log_stage(f"END rerun_analysis :: {sim_id} (valid={req_valid})")

            requirement_updates = entry.get("requirement_updates", []) if isinstance(entry.get("requirement_updates", []), list) else []
            attempt_used = int(entry.get("attempt", 0) or 0)
            requirement_updates.append(
                {
                    "attempt": attempt_used,
                    "feedback": revision.get("feedback", []),
                    "valid": req_valid,
                    "requirement": next_req,
                }
            )
            entry["requirement_updates"] = requirement_updates
            entry["attempt"] = attempt_used + 1

            # If invalid, don't run Foam-Agent; keep rerun_required True but record reason.
            if not req_valid or not next_req.strip():
                entry["interpreter"] = {
                    **(cast(Dict[str, Any], interp or {})),
                    "rerun_required": True,
                    "rerun_reason": "rerun_analysis produced invalid/empty requirement; skipping rerun.",
                }
                pipeline_log["simulations"] = sims_list
                return {"pipeline_log": pipeline_log, "rerun_idx": idx + 1}

            # Run Foam-Agent for this case
            log_stage(f"START foamagent (rerun) :: {sim_id}")
            self._write_requirement(req_path, next_req)
            run_result = self.foam.run(
                user_requirement_path=req_path,
                output_dir=sim_dir / "foam_output",
                project_root=Path.cwd(),
                execute=True,
            )
            rc = run_result.get("returncode") if isinstance(run_result, dict) else None
            log_stage(f"END foamagent (rerun) :: {sim_id} (returncode={rc})")

            run_history = entry.get("run_history", []) if isinstance(entry.get("run_history", []), list) else []
            run_history.append(run_result)
            entry["run_history"] = run_history

            # Re-interpret only this case
            idea = cast(Dict[str, Any], state.get("idea", {}) or {})
            run_result = cast(Dict[str, Any], run_result if isinstance(run_result, dict) else {})
            if not run_result.get("output_dir") and (sim_dir / "foam_output").exists():
                run_result["output_dir"] = str(sim_dir / "foam_output")

            log_stage(f"START interpreter (rerun) :: {sim_id}")
            interp2 = self.interpreter.interpret(
                idea_json=idea,
                experiment_spec=cast(Dict[str, Any], sim or {}),
                experiment_results=run_result,
                verbose=bool(state.get("verbose", True)),
            )
            log_stage(
                f"END interpreter (rerun) :: {sim_id} (rerun_required={bool(interp2.get('rerun_required', False)) if isinstance(interp2, dict) else False})"
            )
            entry["interpreter"] = interp2

            pipeline_log["simulations"] = sims_list
            return {"pipeline_log": pipeline_log, "rerun_idx": idx + 1}

        def route_after_rerun_run_one(state: WorkflowState) -> str:
            queue = state.get("rerun_queue", []) or []
            idx = int(state.get("rerun_idx", 0) or 0)
            return "rerun_run_one" if idx < len(queue) else "rerun_finalize_round"

        def rerun_finalize_round(state: WorkflowState) -> WorkflowState:
            """
            Write an updated interpreter_output.json snapshot after rerun round.
            """
            pipeline_log = cast(Dict[str, Any], state.get("pipeline_log", {}) or {})
            sims_list = cast(List[Dict[str, Any]], pipeline_log.get("simulations", []) or [])
            out_dir = state["out_dir"]

            interp_cases: List[Dict[str, Any]] = []
            for entry in sims_list:
                sim = entry.get("simulation", {}) if isinstance(entry, dict) else {}
                interp = entry.get("interpreter", {}) if isinstance(entry, dict) else {}
                sim_id = sim.get("simulation_id", "unknown")
                rerun_required = bool(interp.get("rerun_required", False)) if isinstance(interp, dict) else False
                interp_cases.append(
                    {
                        "simulation_id": sim.get("simulation_id"),
                        "case_name": sim.get("case_name"),
                        "rerun_required": rerun_required,
                        "rerun_reason": interp.get("rerun_reason") if isinstance(interp, dict) else None,
                        "status": ("rerun" if rerun_required else "ok"),
                        "attempts_used": int(entry.get("attempt", 0) if isinstance(entry, dict) else 0),
                        "interpreter": interp,
                    }
                )
                log_stage(f"RERUN-STATUS :: {sim_id} rerun_required={rerun_required}")

            self._write_json(
                out_dir / "interpreter_output.json",
                {
                    "total_cases": len(interp_cases),
                    "rerun_required_count": sum(1 for c in interp_cases if c.get("rerun_required") is True),
                    "ok_count": sum(1 for c in interp_cases if c.get("status") == "ok"),
                    "cases": interp_cases,
                    "rerun_round": int(state.get("rerun_round", 0) or 0),
                },
            )
            pipeline_log["simulations"] = sims_list
            return {"pipeline_log": pipeline_log}

        def route_after_rerun_finalize_round(state: WorkflowState) -> str:
            """
            Continue rerun rounds until no case needs rerun (or attempts exhausted).
            """
            pipeline_log = cast(Dict[str, Any], state.get("pipeline_log", {}) or {})
            sims_list = cast(List[Dict[str, Any]], pipeline_log.get("simulations", []) or [])
            max_reruns = int(self.settings.workflow_max_reruns_per_experiment)
            for entry in sims_list:
                if not isinstance(entry, dict):
                    continue
                interp = entry.get("interpreter", {}) if isinstance(entry.get("interpreter", {}), dict) else {}
                if not isinstance(interp, dict):
                    continue
                if bool(interp.get("rerun_required", False)) and int(entry.get("attempt", 0) or 0) < max_reruns:
                    return "rerun_round_start"
            return "analysis_and_writer"

        def route_after_interpret_batch(state: WorkflowState) -> str:
            """
            If any case needs rerun, enter rerun loop; otherwise proceed to analysis.
            """
            if not bool(state.get("execute", False)):
                return "analysis_and_writer"
            pipeline_log = cast(Dict[str, Any], state.get("pipeline_log", {}) or {})
            sims_list = cast(List[Dict[str, Any]], pipeline_log.get("simulations", []) or [])
            max_reruns = int(self.settings.workflow_max_reruns_per_experiment)
            for entry in sims_list:
                if not isinstance(entry, dict):
                    continue
                interp = entry.get("interpreter", {}) if isinstance(entry.get("interpreter", {}), dict) else {}
                if not isinstance(interp, dict):
                    continue
                if bool(interp.get("rerun_required", False)) and int(entry.get("attempt", 0) or 0) < max_reruns:
                    return "rerun_round_start"
            return "analysis_and_writer"

        def route_after_final_gate(state: WorkflowState) -> str:
            if bool(state.get("execute", False)):
                return "interpret_batch"
            allow_final = bool(state.get("allow_non_executed_artifacts", False))
            return "analysis_and_writer" if allow_final else "save_pipeline_log"

        def analysis_and_writer(state: WorkflowState) -> WorkflowState:
            log_stage("START analysis")
            out_dir = state["out_dir"]
            topic = state["topic"]
            pipeline_log = cast(Dict[str, Any], state.get("pipeline_log", {}) or {})

            # Build experiments list for full pipeline: decide viz -> create with interpreter ref -> analyze with images
            experiments: List[Dict[str, Any]] = []
            for entry in pipeline_log.get("simulations", []) or []:
                sim_meta = entry.get("simulation", {}) if isinstance(entry, dict) else {}
                sim_id = sim_meta.get("simulation_id")
                if not sim_id:
                    continue
                sim_dir = out_dir / str(sim_id)
                foam_output_dir = sim_dir / "foam_output"
                if CFDWorkflow._locate_foam_dataset(foam_output_dir) is None:
                    continue
                hyp = entry.get("hypothesis", {}) if isinstance(entry, dict) else {}
                user_req = (hyp.get("requirement", "") or "") if isinstance(hyp, dict) else ""
                if not user_req and (sim_dir / "user_requirement.txt").exists():
                    user_req = (sim_dir / "user_requirement.txt").read_text(encoding="utf-8")
                experiments.append({
                    "simulation_id": sim_id,
                    "case_name": sim_meta.get("case_name", sim_id),
                    "description": sim_meta.get("description", ""),
                    "user_requirement": user_req,
                    "sim_dir": sim_dir,
                    "foam_output_dir": foam_output_dir,
                })

            if experiments:
                verbose = state.get("verbose", True)
                result = self.analysis.run_full_analysis_pipeline(experiments, topic, verbose=verbose)
                analysis_text = result["analysis_text"]
                viz_bundle = result["visualizations"]
                viz_by_id = {v["simulation_id"]: v.get("visualization") for v in viz_bundle}
                for entry in pipeline_log.get("simulations", []) or []:
                    sim_meta = entry.get("simulation", {}) if isinstance(entry, dict) else {}
                    sid = sim_meta.get("simulation_id")
                    if sid and sid in viz_by_id:
                        entry["analysis_visualization"] = viz_by_id[sid]
            else:
                analysis_text = ""
                viz_bundle = []

            analysis_path = out_dir / "analysis_report.md"
            self.analysis.save_analysis(analysis_path, analysis_text, topic=topic)
            pipeline_log["analysis"] = str(analysis_path)
            log_stage("END analysis")

            log_stage("START writer")
            paper_context = (
                f"TOPIC:\n{topic}\n\n"
                f"IDEA:\n{json.dumps(state.get('idea', {}), indent=2)}\n\n"
                f"ANALYSIS:\n{analysis_text}\n\n"
                "Write paper draft using available artifacts."
            )
            work_dir = out_dir.parent.parent if out_dir.parent.name == "runs" else out_dir
            paper_text, pdf_path, review_info = self.writer.write_paper_with_literature_and_review(
                topic=topic,
                section_context=paper_context,
                out_dir=out_dir,
                work_dir=work_dir,
                ideation_literature_bundle=(state.get("ideation_result", {}) or {}).get(
                    "literature_used", []
                ),
                visualization_bundle=viz_bundle,
                max_review_tries=10,
                verbose=bool(state.get("verbose", True)),
            )
            paper_path = out_dir / "paper_draft.tex"
            paper_path.write_text(paper_text, encoding="utf-8")
            pipeline_log["paper"] = str(paper_path)
            pipeline_log["paper_pdf"] = str(pdf_path) if pdf_path else None
            pipeline_log["review_tries"] = review_info.get("tries", 0)

            self._write_json(
                out_dir / "analysis_output.json",
                {
                    "topic": topic,
                    "analysis_report": str(analysis_path),
                    "paper_draft": str(paper_path),
                    "paper_pdf": str(pdf_path) if pdf_path else None,
                    "analysis_text": analysis_text,
                    "visualizations": viz_bundle,
                },
            )
            log_stage("END writer")

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
                self._write_json(
                    out_dir / "analysis_output.json",
                    {
                        "topic": str(state.get("topic", "")),
                        "analysis_report": None,
                        "paper_draft": None,
                        "status": "skipped",
                        "reason": "analysis_and_writer stage skipped by artifact gate",
                        "artifact_gate": pipeline_log.get("artifact_gate", {}),
                    },
                )

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
        g.add_node("interpret_batch", interpret_batch)
        g.add_node("rerun_round_start", rerun_round_start)
        g.add_node("rerun_run_one", rerun_run_one)
        g.add_node("rerun_finalize_round", rerun_finalize_round)
        g.add_node("analysis_and_writer", analysis_and_writer)
        g.add_node("save_pipeline_log", save_pipeline_log)

        g.add_edge(START, "ideate")
        g.add_edge("ideate", "expand_and_init")
        g.add_edge("expand_and_init", "prepare_next_sim")
        g.add_conditional_edges("prepare_next_sim", route_after_prepare)

        g.add_edge("generate_requirement", "precheck")
        g.add_conditional_edges("precheck", route_after_precheck)
        g.add_edge("foam_run", "append_simulation_log")
        g.add_edge("append_simulation_log", "prepare_next_sim")

        g.add_conditional_edges("final_artifacts_gate", route_after_final_gate)
        g.add_conditional_edges("interpret_batch", route_after_interpret_batch)
        g.add_conditional_edges("rerun_round_start", route_after_rerun_round_start)
        g.add_conditional_edges("rerun_run_one", route_after_rerun_run_one)
        g.add_conditional_edges("rerun_finalize_round", route_after_rerun_finalize_round)
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
        verbose: bool = True,
    ) -> Dict[str, Any]:
        initial_state: WorkflowState = {
            "topic": topic,
            "out_dir": out_dir,
            "execute": execute,
            "allow_non_executed_artifacts": allow_non_executed_artifacts,
            "verbose": verbose,
        }
        final_state = cast(WorkflowState, self._get_app().invoke(initial_state))
        return cast(Dict[str, Any], final_state.get("pipeline_log", {}) or {})
