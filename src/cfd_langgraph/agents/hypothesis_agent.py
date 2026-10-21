from __future__ import annotations

import json
import time
from typing import Any, Dict, List
from langchain_core.prompts import ChatPromptTemplate

from cfd_langgraph.llm.factory import create_langchain_llm
from cfd_langgraph.prompts.loader import PromptLoader
from cfd_langgraph.utils import extract_json_object, strip_json_fences
from cfd_langgraph.llm.reply import reply_text


# How many times to re-ask the validator when its reply will not parse.
_VALIDATOR_PARSE_ATTEMPTS = 3



def _numbers_in(text: str) -> List[float]:
    """Every number written in ``text`` (0.10, 3.5, 1.5e-5, ...)."""
    keep = set("0123456789.-+eE")
    words = "".join(c if c in keep else " " for c in str(text)).split()
    found: List[float] = []
    for w in words:
        w = w.strip(".+")
        try:
            found.append(float(w))
        except ValueError:
            continue
    return found


def _missing_parameter_values(requirement: str, parameters: Any, prefix: str = "") -> List[str]:
    """The experiment's numeric parameter values its requirement never states.
    Generated from the starter's case, every breakwater requirement of a study
    said the starter's wave (H 0.10 m, T 3.0 s) instead of its own (0.15-0.30 m),
    the validator passed them, and four cases ran the same wave
    (shelf_demo_olaflow_20261011_1242). 0 and 1 are too common to tell."""
    missing: List[str] = []
    if isinstance(parameters, dict):
        for k, v in parameters.items():
            missing += _missing_parameter_values(requirement, v, f"{prefix}{k}")
    elif isinstance(parameters, (list, tuple)):
        for i, v in enumerate(parameters):
            missing += _missing_parameter_values(requirement, v, f"{prefix}[{i}]")
    elif isinstance(parameters, (int, float)) and not isinstance(parameters, bool):
        value = float(parameters)
        if value not in (0.0, 1.0) and not any(
                abs(n - value) <= 1e-9 + 1e-6 * abs(value) for n in _numbers_in(requirement)):
            missing.append(f"{prefix} = {parameters:g}")
    return missing




def _named_paths(text: str) -> List[str]:
    """Absolute paths a requirement names. Echoing the topic's own environment
    lines, regenerated DAFoam requirements still told each case where to work
    and what to read under /mnt (shelf_demo_dafoam_20261011_1437)."""
    found: List[str] = []
    for word in str(text).split():
        w = word.strip("`'\".,;:()[]{}<>")
        if w.startswith("/") and w.count("/") >= 2 and w not in found:
            found.append(w)
    return found


def _physical_parameters(parameters: Any) -> Any:
    """The parameters a requirement must state: numbers and short settings,
    without folders, paths or long notes (a requirement names no folder)."""
    if isinstance(parameters, dict):
        kept = {k: _physical_parameters(v) for k, v in parameters.items()}
        return {k: v for k, v in kept.items() if v not in (None, {}, [])}
    if isinstance(parameters, (list, tuple)):
        kept = [_physical_parameters(v) for v in parameters]
        return [v for v in kept if v not in (None, {}, [])]
    if isinstance(parameters, (int, float)):
        return parameters
    if isinstance(parameters, str) and "/" not in parameters and len(parameters) <= 80:
        return parameters
    return None


# Heading before the starter's case context in the writer's and checker's
# prompts. Both used to carry the literal text ' + repr(CTX) + ' there.
_CASE_CONTEXT_HEADING = "\n\nTHE CASE THIS STUDY RUNS (already fixed):\n"

# Stems of the words a visualization instruction uses; a requirement with none
# of them has nothing for _strip_visualization_mentions to remove.
_VISUAL_WORDS = ("visuali", "plot", "contour", "streamline", "figure", "image", "picture",
                 "render", "paraview", "animation", "snapshot", "chart", "graph")


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
            sys_t = sys_t + _CASE_CONTEXT_HEADING + case_context
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
        out = reply_text(chain.invoke(payload)).strip()
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
                _CASE_CONTEXT_HEADING + case_context + "\n\n" if case_context else ""
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
            "(no 'Visualize', no plotting requests, no figure-generation instructions). "
            + (
                # The writer is told to name no folders, paths or versions (the
                # framework copies the starter into each case's own folder), and
                # the checker used to fail requirements for leaving exactly those
                # out: four of a DAFoam study's ten went through every repair
                # round for "paths, mesh files and DAFoam revision not
                # identified" (shelf_demo_dafoam_20261019_1057).
                "The framework copies the case described above (its mesh, initial fields, "
                "dictionaries and scripts) into each case's own folder and runs it on the "
                "installed solver build. A requirement therefore names no folders, paths, file "
                "names to locate, checksums or software versions, and is not invalid for leaving "
                "them out or for referring to that case's mesh and files as given; judge what it "
                "does state against the case above."
                if case_context else ""
            )
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
            raw = reply_text(chain.invoke({"req": req}))
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
        return reply_text(chain.invoke(
            {
                "req": req,
                "run_topic": run_topic or "",
                "all_experiment_ideas": all_experiment_ideas or "",
                "current_experiment_idea": current_experiment_idea or "",
                "previous_requirement": previous_requirement or "",
                "issues": "\n".join(f"- {i}" for i in issues),
                "guidance": "\n".join(f"- {g}" for g in guidance),
            }
        )).strip()

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
        # Nothing to remove without a word for one. Asked anyway, the model
        # was given the whole requirement to send back verbatim; on a DAFoam
        # study it often sent back nothing, and each such reply was retried
        # for minutes (shelf_demo_dafoam_20261019_1057).
        lowered = text.lower()
        if not any(word in lowered for word in _VISUAL_WORDS):
            return text
        # The model names the sentences; the code removes them. A whole text
        # sent back verbatim is a long answer to get wrong, and an empty or
        # unusable reply here simply changes nothing.
        from cfd_langgraph.llm.factory import quick_empty_replies

        try:
            # An empty reply only means no change here, so it is not retried
            # for minutes.
            with quick_empty_replies():
                reply = reply_text(self.llm.invoke(
                    "List every sentence of the TEXT below that instructs visualization, plotting, "
                    "contouring, streamlines, or figure or image generation. Copy each one exactly "
                    "as it appears, one per line, and nothing else. If there is none, reply NONE.\n\n"
                    f"TEXT:\n{text}"
                ))
        except Exception:
            return text
        out = text
        for line in str(reply or "").splitlines():
            sentence = line.strip().strip("-*\u2022 ").strip()
            if len(sentence) > 15 and sentence in out:
                out = out.replace(sentence, "")
        if out == text:
            return text
        out = "\n".join(" ".join(part.split()) for part in out.splitlines()).strip()
        # A model that names most of the text has misread the task; keep the
        # original over a gutted requirement.
        return out if len(out) >= 0.5 * len(text) else text

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
            unstated = _missing_parameter_values(req, simulation.get("parameter_value") or {})
            paths = _named_paths(req)
            if paths:
                unstated = unstated or ["(no folder or path)"]
                verdict = {
                    "valid": False,
                    "issues": [f"The requirement names folders or paths ({', '.join(paths[:6])}); the "
                               "framework runs this case in its own folder, so a requirement names none."],
                    "repair_guidance": ["Remove every folder and path, and every instruction to create, "
                                        "copy, move, preserve or verify folders or files, or about what "
                                        "may be read or installed. Keep the physics, settings and outputs."],
                }
                if verbose:
                    print(f"[Hypothesis] requirement names paths {paths[:3]}", flush=True)
            elif unstated:
                verdict = {
                    "valid": False,
                    "issues": [f"This case's parameter {u} is not in the requirement; it must be "
                               "used as given, not the starter's or another case's value."
                               for u in unstated],
                    "repair_guidance": ["State each of these values of this case explicitly: "
                                        + "; ".join(unstated)],
                }
                if verbose:
                    print(f"[Hypothesis] requirement leaves out this case's {unstated}", flush=True)
            else:
                # A value can appear in the text by coincidence (another
                # quantity of 0.25), so the checker also compares by name.
                params = _physical_parameters(simulation.get("parameter_value") or {}) or {}
                params_note = (
                    "\n\nTHIS CASE'S PARAMETERS (the requirement must use exactly these values "
                    "for these quantities, not the starter's or another case's; if it states a "
                    "different value for any of them, it is invalid):\n"
                    # Braces escaped: the checker's prompt is a template.
                    + json.dumps(params, indent=1)[:4000].replace("{", "{{").replace("}", "}}")
                ) if params else ""
                verdict = self.llm_validate_requirement(
                    req, verbose=verbose,
                    case_context=(validator_context or case_context or "") + params_note
                )
            if not verdict.get("valid", False) and not unstated:
                # Confirm before rejecting. This validator is non-deterministic
                # by design, and it was measured contradicting itself: a
                # requirement marked valid during a run came back invalid on an
                # immediate re-check of the identical text. With 17 requirements
                # and every one required to pass, a single unstable verdict
                # blocks the entire study and the stage silently re-runs for
                # ever. A genuinely bad requirement fails both times, so the
                # gate still catches what it is for.
                second = self.llm_validate_requirement(
                    req, verbose=verbose,
                    case_context=(validator_context or case_context or "") + params_note
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
