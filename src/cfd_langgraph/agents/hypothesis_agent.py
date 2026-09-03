from __future__ import annotations

import json
import time
from typing import Any, Dict, List
from langchain_core.prompts import ChatPromptTemplate

from cfd_langgraph.llm.factory import create_langchain_llm
from cfd_langgraph.prompts.loader import PromptLoader
from cfd_langgraph.utils import extract_json_object, strip_json_fences


# How many times to re-ask the validator when its reply will not parse.
_VALIDATOR_PARSE_ATTEMPTS = 3


class HypothesisAgent:
    def __init__(self, model: str, prompt_loader: PromptLoader):
        self.model = model
        self.prompts = prompt_loader.section("HypothesisAgent")
        self.llm = create_langchain_llm(model=model, temperature=0.0)
        # Temperature 0.0, not 0.2. This is a QA checker, and sampling bought
        # nothing but disagreement with itself: measured on the approved
        # hypothesis of runs/ph_codex_20260902_1806, the validator read one
        # requirement and returned 13 specific issues, then read the IDENTICAL
        # text and returned valid. Across three rounds its issue count went
        # 10 -> 4 -> 13 while the text only grew by 161 characters between the
        # last two. A verdict that unstable makes the repair loop argue with
        # noise at ~100s a call.
        self.validator_llm = create_langchain_llm(model=model, temperature=0.0)

    def generate_user_requirement(
        self,
        idea: Dict[str, Any],
        simulation: Dict[str, Any],
        run_topic: str = "",
        case_context: str = "",
        all_experiment_ideas: str = "",
        current_experiment_idea: str = "",
        previous_requirement: str = "",
        verbose: bool = False,
    ) -> str:
        if verbose:
            print("[Hypothesis] Generating requirement for %s..." % simulation.get("simulation_id", "?"), flush=True)
        sys_t = self.prompts.get("hypothesis_system_prompt", "")
        usr_t = self.prompts.get("hypothesis_user_prompt", "")
        if case_context:
            # Blind to the real case, this stage invented a whole study:
            # OpenFOAM v2312 (the machine runs OpenFOAM 10), a custom solver
            # cloned from pimpleFoam (the case is simpleFoam), and reference
            # paths that do not exist. 13 of 17 requirements failed validation
            # for exactly that, so requirements.json was never published and
            # the manager re-ran the stage forever.
            sys_t = sys_t + """ + repr(CTX) + """ + case_context
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
            "all_experiment_ideas": all_experiment_ideas or "",
            "current_experiment_idea": current_experiment_idea or "",
            "previous_requirement": previous_requirement or "",
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

    def llm_validate_requirement(
        self, req: str, verbose: bool = False, case_context: str = ""
    ) -> Dict[str, Any]:
        """
        LLM semantic validator for Foam-Agent prompt quality and consistency.
        Non-deterministic by design (requested).
        """
        system = (
            (
                # The validator has to judge against the same reality the
                # generator writes for, or it rejects correct requirements and
                # accepts ones that name a case that does not exist.
                """ + repr(CTX) + """ + case_context + "\n\n" if case_context else ""
            )
            + "You are a strict CFD QA checker for Foam-Agent prompts. "
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

        # An unreadable reply is not a verdict, and it used to be recorded as
        # one: any parse failure returned valid=False with the invented issue
        # "Validator response was not parseable JSON" plus generic repair
        # guidance. The repair agent then rewrote a requirement nobody had
        # found a defect in, at ~200s a round, and the rewrite grew the text so
        # every later validate ran slower. Measured live on
        # runs/ph_codex_20260902_1806: "Validation parse error: Expecting value:
        # line 1 column 1 (char 0)" -- an EMPTY body -- charged to the
        # requirement as a failure. `raw` was captured and never printed, so
        # nothing in the log distinguished a real rejection from an unread one.
        #
        # Retry the call instead. Empty and truncated bodies from the provider
        # were transient every time we measured them, and _retry_empty_turn
        # cannot catch these: it fires only when a turn has neither text nor
        # tool calls, while an empty *string* is a perfectly well-formed reply.
        last_err = None
        raw = ""
        for attempt in range(1, _VALIDATOR_PARSE_ATTEMPTS + 1):
            raw = chain.invoke({"req": req}).content
            try:
                parsed = json.loads(extract_json_object(raw))
                if not isinstance(parsed, dict):
                    raise ValueError("not a JSON object")
                parsed.setdefault("valid", False)
                parsed.setdefault("issues", [])
                parsed.setdefault("repair_guidance", [])
                if verbose:
                    print("[Hypothesis] Validation: valid=%s" % parsed.get("valid", False), flush=True)
                return parsed
            except Exception as e:
                last_err = e
                # Always surfaced, verbose or not: this is the difference
                # between "your requirement is wrong" and "we could not read
                # the answer", and it was invisible.
                print(
                    "[Hypothesis] validator reply unparseable on attempt "
                    "%d/%d (%s); raw=%r"
                    % (attempt, _VALIDATOR_PARSE_ATTEMPTS, e, (raw or "")[:200]),
                    flush=True,
                )
                if attempt < _VALIDATOR_PARSE_ATTEMPTS:
                    time.sleep(2 ** attempt)

        # Still unreadable. Report it as what it is rather than as a defect in
        # the requirement, and give the repair step nothing to act on -- there
        # is no finding to repair.
        print(
            "[Hypothesis] validator gave no readable verdict after %d attempts; "
            "treating as unvalidated, not as a defect" % _VALIDATOR_PARSE_ATTEMPTS,
            flush=True,
        )
        return {
            "valid": False,
            "parse_failed": True,
            "issues": [],
            "repair_guidance": [],
            "raw": raw,
            "parse_error": str(last_err),
        }

    def repair_requirement(
        self,
        req: str,
        issues: List[str],
        guidance: List[str],
        run_topic: str = "",
        all_experiment_ideas: str = "",
        current_experiment_idea: str = "",
        previous_requirement: str = "",
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
            "Run topic:\n{run_topic}\n\n"
            "All experiment ideas/context summary:\n{all_experiment_ideas}\n\n"
            "Current experiment idea:\n{current_experiment_idea}\n\n"
            "Previous experiment requirement (for consistency):\n{previous_requirement}\n\n"
            "Issues detected:\n{issues}\n\n"
            "Repair guidance:\n{guidance}\n\n"
            "Return only the corrected requirement paragraph."
        )
        prompt = ChatPromptTemplate.from_messages([("system", system), ("human", user)])
        chain = prompt | self.validator_llm
        return chain.invoke(
            {
                "req": req,
                "run_topic": run_topic or "",
                "all_experiment_ideas": all_experiment_ideas or "",
                "current_experiment_idea": current_experiment_idea or "",
                "previous_requirement": previous_requirement or "",
                "issues": "\n".join(f"- {i}" for i in issues),
                "guidance": "\n".join(f"- {g}" for g in guidance),
            }
        ).content.strip()

    def _strip_visualization_mentions(self, req: str) -> str:
        """Remove visualization instructions without touching anything else.

        This used to split on every "." and rejoin with ". ", which corrupted
        every decimal and filename in the requirement — `h=1.0` became
        `h=1. 0`, `Cf.csv` became `Cf. csv`, 30 times in a single generated
        requirement — and flattened newlines. It was damaging far more than
        the plotting sentences it existed to delete.

        Deciding which sentences are visualization instructions is a reading
        task, so it is given to the model. If that call fails the text is
        returned UNCHANGED: the generation prompt and the validator already
        both forbid visualization, so this is a third line of defence, and a
        third line of defence must never be the thing that breaks the output.
        """
        text = str(req or "")
        if not text.strip():
            return text
        try:
            cleaned = self.llm.invoke(
                "Remove any sentence that instructs visualization, plotting, contouring, "
                "streamlines, figure or image generation, or post-processing output.\n"
                "Change NOTHING else: keep every number, filename, path, newline and "
                "sentence that remains exactly as written. Do not summarise, reword, or "
                "reformat. If no such sentence is present, return the text verbatim.\n"
                "Return only the resulting text.\n\n"
                f"TEXT:\n{text}"
            )
            out = getattr(cleaned, "content", cleaned)
            out = out if isinstance(out, str) else str(out)
            # A model that returns something drastically shorter has summarised
            # rather than filtered; keep the original over a lossy rewrite.
            if out.strip() and len(out) >= 0.5 * len(text):
                return out.strip()
        except Exception:
            pass
        return text

    def generate_validated_requirement(
        self,
        idea: Dict[str, Any],
        simulation: Dict[str, Any],
        run_topic: str = "",
        all_experiment_ideas: str = "",
        current_experiment_idea: str = "",
        previous_requirement: str = "",
        max_retries: int = 3,
        verbose: bool = False,
        case_context: str = "",
        validator_context: str = "",
    ) -> Dict[str, Any]:
        req = self._strip_visualization_mentions(
            self.generate_user_requirement(
                idea,
                simulation,
                run_topic=run_topic,
                all_experiment_ideas=all_experiment_ideas,
                current_experiment_idea=current_experiment_idea,
                previous_requirement=previous_requirement,
                verbose=verbose,
                case_context=case_context,
            )
        )
        history: List[Dict[str, Any]] = []

        for attempt in range(max(1, max_retries + 1)):
            if verbose and attempt > 0:
                print("[Hypothesis] Retry %d/%d (validation failed)" % (attempt, max_retries), flush=True)
            verdict = self.llm_validate_requirement(
                req, verbose=verbose, case_context=validator_context or case_context
            )
            if not verdict.get("valid", False):
                # Confirm before rejecting. This validator is non-deterministic
                # by design, and it was measured contradicting itself: a
                # requirement marked valid during a run came back invalid on an
                # immediate re-check of the identical text. With 17 requirements
                # and every one required to pass, a single unstable verdict
                # blocks the entire study and the stage silently re-runs for
                # ever. A genuinely bad requirement fails both times, so the
                # gate still catches what it is for.
                second = self.llm_validate_requirement(
                    req, verbose=verbose, case_context=validator_context or case_context
                )
                if second.get("valid", False):
                    verdict = second
                    if verbose:
                        print("[Hypothesis] first verdict not reproduced; accepting", flush=True)
            history.append({"requirement": req, "verdict": verdict})
            if verdict.get("valid", False):
                return {
                    "requirement": self._strip_visualization_mentions(req),
                    "valid": True,
                    "history": history,
                }
            if verdict.get("parse_failed"):
                # No readable verdict means no finding, and repairing against
                # no finding is a ~200s rewrite of text nobody objected to --
                # which also grows it and slows every later validate. Go round
                # again and try to get a verdict instead.
                if verbose:
                    print("[Hypothesis] no verdict to act on; re-validating without repair", flush=True)
                continue
            if verbose:
                print("[Hypothesis] Repairing requirement...", flush=True)
            req = self.repair_requirement(
                req,
                issues=verdict.get("issues", []),
                guidance=verdict.get("repair_guidance", []),
                run_topic=run_topic,
                all_experiment_ideas=all_experiment_ideas,
                current_experiment_idea=current_experiment_idea,
                previous_requirement=previous_requirement,
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
