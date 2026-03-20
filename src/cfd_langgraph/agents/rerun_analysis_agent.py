from __future__ import annotations

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
        reference_summary: str | None = None,
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

