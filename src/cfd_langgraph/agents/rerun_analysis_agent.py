from __future__ import annotations

import json
from typing import Any, Dict, List

from cfd_langgraph.agents.hypothesis_agent import HypothesisAgent
from cfd_langgraph.prompts.loader import PromptLoader


class RerunAnalysisAgent:
    """
    Reads the interpreter report for a case that needs re-run and revises the
    Foam-Agent user requirement accordingly (no viz instructions).
    """

    def __init__(self, model: str, prompt_loader: PromptLoader):
        self.hypothesis = HypothesisAgent(model=model, prompt_loader=prompt_loader)

    @staticmethod
    def _format_experiment_idea(experiment_idea: Dict[str, Any] | None) -> str:
        """
        Produce a short, stable text snippet for prompt guidance.

        experiment_idea is expected to come from ideation_output.json / simulation.case_data,
        and may include experiment_id/name/notes/parameters/controls.
        """
        if not isinstance(experiment_idea, dict) or not experiment_idea:
            return ""

        exp_id = experiment_idea.get("experiment_id")
        name = experiment_idea.get("name")
        notes = experiment_idea.get("notes") or experiment_idea.get("description")
        topology = experiment_idea.get("topology")

        params = experiment_idea.get("parameters")
        params_txt = ""
        if isinstance(params, dict) and params:
            # Only include simple scalar-ish values to keep prompts small.
            scalar_items: List[tuple[str, Any]] = []
            for k, v in params.items():
                if isinstance(v, (str, int, float, bool)):
                    scalar_items.append((str(k), v))
            scalar_items.sort(key=lambda x: x[0])
            scalar_items = scalar_items[:8]
            if scalar_items:
                params_txt = json.dumps({k: v for k, v in scalar_items}, ensure_ascii=False, sort_keys=True)

        controls = experiment_idea.get("controls")
        controls_txt = ""
        if isinstance(controls, dict) and controls:
            controls_txt = json.dumps(controls, ensure_ascii=False, sort_keys=True)
            # Hard truncate to avoid very large controls blocks.
            if len(controls_txt) > 600:
                controls_txt = controls_txt[:600] + "..."

        lines: List[str] = []
        if exp_id:
            lines.append(f"experiment_id: {exp_id}")
        if name:
            lines.append(f"name: {name}")
        if topology:
            lines.append(f"topology: {topology}")
        if isinstance(notes, str) and notes.strip():
            lines.append(f"notes: {notes.strip()}")
        if params_txt:
            lines.append(f"key_parameters: {params_txt}")
        if controls_txt:
            lines.append(f"controls: {controls_txt}")

        return "\n".join(lines).strip()

    @staticmethod
    def _extract_feedback(interp: Dict[str, Any]) -> List[str]:
        lines: List[str] = []
        for key in (
            "rerun_reason",
            "reasons",
            "issues",
            "summary",
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
            elif isinstance(val, dict):
                # sometimes issues come as {"issues":[...]}
                if key == "issues":
                    xs = val.get("issues")
                    if isinstance(xs, list):
                        lines.extend(str(x).strip() for x in xs if str(x).strip())
        return [x for x in lines if x]

    def revise_requirement(
        self,
        current_requirement: str,
        interpreter_report: Dict[str, Any],
        experiment_idea: Dict[str, Any] | None = None,
        reference_summary: str | None = None,
        reference_diff_report: str | None = None,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        if verbose:
            print("[RerunAnalysis] Revising requirement from interpreter feedback...", flush=True)
        feedback = self._extract_feedback(interpreter_report or {})
        if not feedback:
            verdict = self.hypothesis.llm_validate_requirement(current_requirement, verbose=verbose)
            return {
                "requirement": current_requirement,
                "valid": bool(verdict.get("valid", False)),
                "feedback": [],
                "validator": verdict,
            }

        guidance_lines: List[str] = [
            "Update the requirement to address interpreter-detected execution or physics issues.",
            "Keep the requirement executable by Foam-Agent (solver, time controls, BCs, mesh).",
            "Do not include visualization instructions.",
        ]
        experiment_idea_txt = self._format_experiment_idea(experiment_idea)
        if experiment_idea_txt:
            guidance_lines.append(
                "Experiment concept you must preserve while repairing the requirement:\n"
                f"{experiment_idea_txt}"
            )
        if reference_summary:
            guidance_lines.append(
                "You also have a summary of a closely related WORKING case. "
                "Where helpful, borrow only the minimal necessary details from that reference "
                "(e.g., mesh layout, boundary-condition pattern, turbulence model settings, "
                "or solver/relaxation parameters) to fix the failing case, while preserving "
                "the intended sweep dimension (what the study topic says should vary). "
                "The reference summary is:\n"
                f"{reference_summary}\n"
                "Never copy or mention any contents from constant/polyMesh or any time directories "
                "other than time 0; those are generated outputs, not inputs."
            )

        if reference_diff_report:
            guidance_lines.append(
                "Diff report between a closely related WORKING case and the failing case. "
                "Use this diff report as the primary source of what to change in the failing-case "
                "Foam-Agent requirement. Do not summarize the diff report; only incorporate the concrete "
                "file-level changes into the revised requirement:\n"
                f"{reference_diff_report}"
            )

        revised = self.hypothesis.repair_requirement(
            current_requirement,
            issues=[f"Interpreter feedback: {x}" for x in feedback],
            guidance=guidance_lines,
        )
        revised = self.hypothesis._strip_visualization_mentions(revised)

        verdict = self.hypothesis.llm_validate_requirement(revised)
        if verdict.get("valid", False):
            return {"requirement": revised, "valid": True, "feedback": feedback, "validator": verdict}

        repaired = self.hypothesis.repair_requirement(
            revised,
            issues=verdict.get("issues", []),
            guidance=verdict.get("repair_guidance", []),
        )
        repaired = self.hypothesis._strip_visualization_mentions(repaired)
        repaired_verdict = self.hypothesis.llm_validate_requirement(repaired, verbose=verbose)
        if verbose:
            print("[RerunAnalysis] Done: valid=%s" % repaired_verdict.get("valid", False), flush=True)
        return {
            "requirement": repaired,
            "valid": bool(repaired_verdict.get("valid", False)),
            "feedback": feedback,
            "validator": repaired_verdict,
        }

