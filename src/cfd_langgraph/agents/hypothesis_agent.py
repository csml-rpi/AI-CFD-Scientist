from __future__ import annotations

import json
from typing import Any, Dict, List
from langchain_core.prompts import ChatPromptTemplate

from cfd_langgraph.llm.factory import create_langchain_llm
from cfd_langgraph.prompts.loader import PromptLoader
from cfd_langgraph.utils import strip_json_fences


class HypothesisAgent:
    def __init__(self, model: str, prompt_loader: PromptLoader):
        self.model = model
        self.prompts = prompt_loader.section("HypothesisAgent")
        self.llm = create_langchain_llm(model=model, temperature=0.3)
        # Keep validator non-deterministic (LLM semantic QA), but lower-temp than generator.
        self.validator_llm = create_langchain_llm(model=model, temperature=0.2)

    def generate_user_requirement(
        self,
        idea: Dict[str, Any],
        simulation: Dict[str, Any],
        run_topic: str = "",
        verbose: bool = False,
    ) -> str:
        if verbose:
            print("[Hypothesis] Generating requirement for %s..." % simulation.get("simulation_id", "?"), flush=True)
        sys_t = self.prompts.get("hypothesis_system_prompt", "")
        usr_t = self.prompts.get("hypothesis_user_prompt", "")
        if not sys_t or not usr_t:
            raise ValueError("Missing HypothesisAgent prompts")

        case_data = simulation.get("case_data", {})
        param_val = simulation.get("parameter_value", {})
        if not isinstance(param_val, dict):
            param_val = {}
        # Generic parameter representation: pass full param dict so prompts work for any case type.
        param_str = json.dumps(param_val, indent=2) if param_val else ""
        topic_val = run_topic or ""
        payload = {
            "study_id": idea.get("study_id", ""),
            "description": idea.get("description", ""),
            "case_name": simulation.get("case_name", ""),
            "simulation_id": simulation.get("simulation_id", ""),
            "parameter_values": param_str,
            "simulation_description": simulation.get("description", ""),
            "run_topic": topic_val,
            "topic": topic_val,
            "experiment_concept": {
                "case_data": case_data,
                "solver": idea.get("solver", "icoFoam"),
                "target_CFL": idea.get("target_CFL", 0.5),
                "post": idea.get("post", {}),
            },
        }

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", sys_t),
                ("human", usr_t),
            ]
        )
        chain = prompt | self.llm
        out = chain.invoke(payload).content.strip()
        if verbose:
            print("[Hypothesis] Requirement generated (%d chars)" % len(out), flush=True)
        return out

    def llm_validate_requirement(self, req: str, verbose: bool = False) -> Dict[str, Any]:
        """
        LLM semantic validator for Foam-Agent prompt quality and consistency.
        Non-deterministic by design (requested).
        """
        system = (
            "You are a strict CFD QA checker for Foam-Agent prompts. "
            "Decide whether this requirement is logically consistent and executable by Foam-Agent. "
            "Validate things like solver presence, time-control consistency, boundary-condition completeness, "
            "mesh details, physics coherence, units. "
            "Explicitly sanity-check time controls: endTime > 0, deltaT > 0, deltaT <= endTime, "
            "writeInterval is consistent with deltaT/endTime, and any 'run from t0 to tf' matches the chosen controls. "
            "Also ensure the prompt includes enough detail for Foam-Agent to pick a tutorial/solver and write all required files "
            "(e.g., solver, viscosity/nu or transport properties, initial/boundary conditions, and mesh description). "
            "Also ensure NO visualization instructions are present "
            "(no 'Visualize', no plotting requests, no figure-generation instructions)."
        )
        user = (
            "Requirement:\n{req}\n\n"
            "Return ONLY valid JSON with schema:\n"
            "{{\n"
            '  "valid": true/false,\n'
            '  "issues": ["..."],\n'
            '  "repair_guidance": ["..."]\n'
            "}}"
        )
        prompt = ChatPromptTemplate.from_messages([("system", system), ("human", user)])
        chain = prompt | self.validator_llm
        raw = chain.invoke({"req": req}).content

        try:
            parsed = json.loads(strip_json_fences(raw))
            if not isinstance(parsed, dict):
                raise ValueError("not dict")
            parsed.setdefault("valid", False)
            parsed.setdefault("issues", [])
            parsed.setdefault("repair_guidance", [])
            if verbose:
                print("[Hypothesis] Validation: valid=%s" % parsed.get("valid", False), flush=True)
            return parsed
        except Exception as e:
            if verbose:
                print("[Hypothesis] Validation parse error: %s" % e, flush=True)
            return {
                "valid": False,
                "issues": ["Validator response was not parseable JSON."],
                "repair_guidance": [
                    "Return one complete executable requirement paragraph with solver, time controls, and boundary conditions. Do not include visualization instructions."
                ],
                "raw": raw,
            }

    def repair_requirement(
        self, req: str, issues: List[str], guidance: List[str]
    ) -> str:
        system = (
            "You repair CFD prompts for Foam-Agent. "
            "The repaired requirement must remain a single, executable paragraph that is logically coherent: "
            "include solver, geometry/domain, mesh, boundary conditions, and time controls consistent with the study and topic constraints, in SI units, "
            "and do NOT add any visualization or plotting instructions. "
            "Output exactly one corrected plain-English requirement paragraph."
        )
        user = (
            "Requirement:\n{req}\n\n"
            "Issues detected:\n{issues}\n\n"
            "Repair guidance:\n{guidance}\n\n"
            "Return only the corrected requirement paragraph."
        )
        prompt = ChatPromptTemplate.from_messages([("system", system), ("human", user)])
        chain = prompt | self.validator_llm
        return chain.invoke(
            {
                "req": req,
                "issues": "\n".join(f"- {i}" for i in issues),
                "guidance": "\n".join(f"- {g}" for g in guidance),
            }
        ).content.strip()

    @staticmethod
    def _strip_visualization_mentions(req: str) -> str:
        banned_tokens = [
            "visualize",
            "visualization",
            "plot",
            "contour",
            "streamline",
            "stream line",
            "save as png",
            "screenshot",
            "render",
            "paraview",
            "para view",
            "pyvista",
            "vtk",
            "glyph",
            "quiver",
            "vector",
            "post-process",
            "postprocess",
            "post processing",
            "figure",
            "image",
            "png",
            "jpg",
            "jpeg",
        ]
        sentences = [s.strip() for s in req.replace("\n", " ").split(".")]
        kept = []
        for s in sentences:
            low = s.lower()
            if any(tok in low for tok in banned_tokens):
                continue
            if s:
                kept.append(s)
        out = ". ".join(kept).strip()
        return (out + ".") if out and not out.endswith(".") else out

    def generate_validated_requirement(
        self,
        idea: Dict[str, Any],
        simulation: Dict[str, Any],
        run_topic: str = "",
        max_retries: int = 3,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        req = self._strip_visualization_mentions(
            self.generate_user_requirement(idea, simulation, run_topic=run_topic, verbose=verbose)
        )
        history: List[Dict[str, Any]] = []

        for attempt in range(max(1, max_retries + 1)):
            if verbose and attempt > 0:
                print("[Hypothesis] Retry %d/%d (validation failed)" % (attempt, max_retries), flush=True)
            verdict = self.llm_validate_requirement(req, verbose=verbose)
            history.append({"requirement": req, "verdict": verdict})
            if verdict.get("valid", False):
                return {
                    "requirement": self._strip_visualization_mentions(req),
                    "valid": True,
                    "history": history,
                }
            if verbose:
                print("[Hypothesis] Repairing requirement...", flush=True)
            req = self.repair_requirement(
                req,
                issues=verdict.get("issues", []),
                guidance=verdict.get("repair_guidance", []),
            )
            req = self._strip_visualization_mentions(req)

        if verbose:
            print("[Hypothesis] Final validation...", flush=True)
        final_verdict = self.llm_validate_requirement(req, verbose=verbose)
        return {
            "requirement": self._strip_visualization_mentions(req),
            "valid": bool(final_verdict.get("valid", False)),
            "history": history,
            "final_verdict": final_verdict,
        }
