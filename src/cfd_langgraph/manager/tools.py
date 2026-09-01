from __future__ import annotations

import json
import math
import statistics
import os
import re
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from cfd_langgraph.config import Settings, get_settings
from cfd_langgraph.hypothesis_pipeline import run_propose_critique_rank
from cfd_langgraph.knowledge_bundle import KnowledgeBundle
from cfd_langgraph.llm.factory import create_langchain_llm
from cfd_langgraph.scheduling import CaseCoordinator
from cfd_langgraph import foam_native
from cfd_langgraph.foam_native.openfoam_env import resolve_openfoam_env

_REPO_ROOT = Path(__file__).resolve().parents[3]

# scripts/ isn't a proper package under src/cfd_langgraph — bootstrap it onto
# sys.path the same way open_ended_discovery.py/oed_extensions.py already
# bootstrap each other, so the OED candidate-runner tools below can import
# their pure-Python pieces (SearchArchive, compute_metric_vector) directly
# instead of needing a subprocess for every candidate's scoring step.
_SCRIPTS_DIR = _REPO_ROOT / "scripts"

# Files the workflow itself owns: written only by the tool that produces them,
# never by a model with a write tool. These are the records every downstream
# gate trusts as ground truth (did this case run, what did it score, was it
# approved, did the audit pass), so a model able to write one can manufacture
# any conclusion it likes. Enforced by BOTH _writable_path (the general write
# tools) and _safe_case_path (foam_write_case_file), since several of these
# live inside a case directory.
_PROTECTED_ARTIFACT_NAMES = frozenset({
    "analysis.json",
    "audit_passed.json",
    "baseline_score.json",
    "bound_comparators.json",
    "bridge.json",
    "candidate_record.json",
    "checkpoints.sqlite",
    "decision.json",
    "history.json",
    "hypotheses_approved.json",
    "hypotheses_ranked.json",
    "lit.json",
    "manifest.json",
    "mesh_independence_context.json",
    "metric_specs.json",
    "objective_contract.json",
    "oed_artifact.json",
    "paper_plan.json",
    "proposals.json",
    "requirements.json",
    "requirements_draft.json",
    "review.json",
    "run_result.json",
    "search_config.json",
    "selected_mesh_spec.json",
    "state.json",
})


def _is_protected_artifact(path: Path, rel_parts: tuple = ()) -> bool:
    """Whether ``path`` is a workflow-owned artifact no model may write.

    Matched case-insensitively (a case-insensitive filesystem would otherwise
    let ``Candidate_Record.json`` overwrite the real file), and extended to
    SQLite's sidecars: writing ``checkpoints.sqlite-wal`` or ``-shm`` corrupts
    the checkpoint database just as surely as writing the database itself,
    and durable resume is a stated guarantee of this CLI.
    """
    name = path.name.lower()
    if name in _PROTECTED_ARTIFACT_NAMES:
        return True
    for suffix in ("-wal", "-shm", "-journal"):
        if name.endswith(suffix) and name[: -len(suffix)] in _PROTECTED_ARTIFACT_NAMES:
            return True
    lowered_parts = {part.lower() for part in rel_parts}
    return "checkpoints" in lowered_parts or "state" in lowered_parts
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from oed_search_archive import (  # noqa: E402
    STRATEGIES,
    STRATEGY_GUIDANCE,
    SearchArchive,
    normalize_strategy,
)

try:
    import oed_extensions as _oedx  # noqa: E402
except Exception:
    _oedx = None  # type: ignore


class _OEDCandidateSpec(BaseModel):
    """One proposed candidate from ``oed_propose_candidates`` — matches what
    ``build_oed_candidate_runner_subagent``'s ``task`` description should
    hand each concurrent candidate-runner call."""

    variant_name: str = Field(description="Short filesystem-safe slug, e.g. 'sa_rc_v2'.")
    action_type: Literal["code_mod", "experiment"] = Field(description="'code_mod' (build/modify a model, whatever method you use to determine it) or 'experiment' (reuse an already-compiled model with new coefficients, no rebuild).")
    strategy: str = Field(
        default="",
        description=(
            "HOW you intend to determine this modification, in your own words — not what "
            "you are modifying. Examples of distinct answers: reason the form out from "
            "physics and pick coefficients by hand; fit coefficients by optimising the "
            "scored objective through the solver (a-posteriori); fit a form offline to "
            "stored high-fidelity data before it meets the solver; sweep coefficients of "
            "a model already compiled. You are free to combine or stage these. The "
            "archive scores each strategy separately, so this is a real choice with "
            "measured consequences, not a label."
        ),
    )
    plan: str = Field(
        default="",
        description=(
            "The steps a candidate agent should carry out, if the strategy needs more "
            "than 'implement the hypothesis'. Name the data files to read, the fit or "
            "optimiser to run, and what the fitted result becomes. Leave empty when the "
            "hypothesis alone is the whole instruction."
        ),
    )
    hypothesis: str = Field(description="The concrete, technical modification — specific equations/terms/coefficients, implementable as-is.")
    target_family: str = Field(default="", description="The model family this targets, for the manager's own bookkeeping.")
    model_name_to_reuse: str = Field(default="", description="For action_type='experiment' only: the compiled model name to reuse.")
    base_case_dir: str = Field(default="", description="For action_type='experiment' only: case directory containing the compiled model to reuse.")
    parameters: Dict[str, float] = Field(default_factory=dict, description="For action_type='experiment' only: coefficient overrides.")


class _OEDCandidateBatch(BaseModel):
    candidates: List[_OEDCandidateSpec]


class _OEDDuplicateVerdict(BaseModel):
    """Which already-evaluated candidate a proposal repeats, if any."""

    duplicate_of: str = Field(
        default="",
        description="variant_name of the already-evaluated candidate this proposal repeats, "
        "or empty string if it is a genuinely new experiment.",
    )
    reason: str = Field(default="", description="One sentence: what makes it the same, or different.")


def _llm_duplicate_of(
    candidate: Dict[str, Any], history: List[Dict[str, Any]], llm: Any
) -> Optional[str]:
    """Ask the model whether a proposal repeats work already paid for.

    Text normalisation cannot answer this. The same experiment was described
    seven ways in one real run — a full paragraph down to
    "SA-Cross-Diffusion with c_cd=1.0" — and any string comparison calls those
    different. Judging sameness of a described experiment is a reading task,
    so it is given to the reader.

    Only reached for candidates whose identity is NOT already exact: a
    coefficient experiment carries its coefficients, and comparing those needs
    no model call and cannot be wrong. Returns the matching variant_name, or
    None. Returns None on any failure — a missed duplicate costs one
    evaluation, whereas a wrongly-dropped candidate loses an idea outright, so
    this fails toward spending.
    """
    prior = [
        {
            "variant_name": str(h.get("variant_name") or ""),
            "family": str(h.get("family") or ""),
            "action_type": str(h.get("action_type") or ""),
            "description": str(h.get("model_description") or "")[:400],
        }
        for h in history
        if isinstance(h, dict) and h.get("action_type") in {"code_mod", "experiment"}
    ][-40:]
    if not prior:
        return None
    try:
        verdict = llm.with_structured_output(_OEDDuplicateVerdict).invoke(
            "You are preventing wasted compute in a scientific search. Each evaluation below "
            "already ran and cost real budget.\n\n"
            "Is the PROPOSED experiment the same experiment as any of them — the same change to "
            "the same model, with the same values? Re-wording, a different variant name, or a "
            "different level of detail do NOT make it a new experiment. A different coefficient "
            "value, a different term, or a different model family DO make it new.\n\n"
            f"ALREADY EVALUATED:\n{json.dumps(prior, indent=1)}\n\n"
            f"PROPOSED:\n{json.dumps({k: candidate.get(k) for k in ('target_family', 'action_type', 'hypothesis', 'parameters')}, indent=1, default=str)}\n\n"
            "Return duplicate_of as the exact variant_name it repeats, or an empty string."
        )
        match = str(getattr(verdict, "duplicate_of", "") or "").strip()
        known = {entry["variant_name"] for entry in prior}
        return match if match in known else None
    except Exception as exc:
        print(f"[oed] duplicate check unavailable ({type(exc).__name__}); allowing candidate", flush=True)
        return None


# How many baseline-beating candidates get promoted into cases/ for
# interpretation and the paper. Enough for a comparison table with a clear
# best result and its near neighbours; not so many that a saturated search
# copies dozens of case trees and interprets each one.
_OED_MAX_PROMOTED = 8

# Fewest papers that counts as literature grounding. Below this, ideation has
# nothing to react to and proposes whatever it would have proposed anyway.
_LIT_MIN_PAPERS = 10

# Wall-clock fence for one evaluation-case solver run (seconds). These are
# plain re-runs of a declared case with a compiled model already in hand — no
# agent, no compile — so a case that has not reached End in an hour is stuck
# rather than slow.
_OED_EVAL_CASE_TIMEOUT_S = 3600

# How many timed-out evaluation cases before a candidate is abandoned. Two, not
# one: a single slow case can be the heaviest in the set rather than a verdict
# on the model, but two in a row is the model.
_OED_EVAL_TIMEOUT_ABANDON = 2
# How many repair attempts a candidate that scored null is worth before the
# search accepts the null and moves on. Two, because the first attempt acts on
# the diagnosis and the second acts on what the first one learned; a third is
# budget a different mechanism could spend, and by then the diagnosis was
# probably wrong about the cause.
_OED_REPAIR_ATTEMPTS = 2

# Turn cap handed to every build agent. Named here rather than left to
# code_mod_agentic.py's own default so the diagnosis can tell the model
# whether an agent stopped because it ran out of turns or ran out of clock --
# two different problems with two different answers, indistinguishable if
# this side does not know what the cap was.
_OED_MAX_TURNS = 120

# Extensions of the wall clock, per candidate. Same reasoning as the repair
# budget: the first extension acts on a diagnosis, a second acts on what the
# first one learned, and a third is budget that a different mechanism could
# spend on a candidate that is not already twice behind schedule.
_OED_EXTENSION_ATTEMPTS = 2

# Hard ceiling on any single build, however much time a diagnosis asks for.
# This is not a judgement about the science -- the model decides how much
# longer its work needs, and that estimate is what gets granted. It is a bound
# on the machine: a candidate holds a CaseCoordinator slot for as long as it
# runs, so an estimate that came back wrong by an order of magnitude would
# block every other candidate behind it for as long as it liked. Six hours
# matches the ceiling already used on this file's other long-running path.
_OED_MAX_EXTENDED_S = 21600


def _candidate_strategy(candidate_path: Path) -> str:
    """Which search strategy produced this candidate, as recorded when it ran.

    Read from the candidate's own record rather than inferred from its name or
    its hypothesis text -- the record is what the archive niches on, so a
    fence built from it is fenced against the same population the archive
    compares against.
    """
    record = _read_json(candidate_path / "candidate_record.json") or {}
    strategy = str(record.get("strategy") or "").strip()
    if strategy:
        return strategy
    invocation = _read_json(candidate_path / "candidate_invocation.json") or {}
    return str(invocation.get("strategy") or "").strip()


def _oed_candidate_timeout(disc_dir: Path, strategy: str = "") -> Optional[int]:
    """Wall-clock fence for one agentic candidate, measured from this study.

    A fixed number cannot work: candidate cost is set by the case, the mesh,
    and the solver, none of which are known here. So the fence is an outlier
    bound over the durations candidates in *this* study have actually taken.

    Durations are log-normal, so the standard Tukey fence is taken in log
    space. On run oed_20260822_1626_codex_high (39 candidates, median 434s)
    the linear fence lands at 991s and would have killed a legitimate 1024s
    candidate; the log fence lands at 1323s, which spares it and still stops
    the 8468s runaway that ran 35 solver simulations doing its own private
    coefficient sweep while three finished candidates sat blocked behind it.

    Floored at the slowest candidate that has already succeeded, because a
    duration already observed to be productive is by definition not an
    outlier -- without it the fence sits at 402s after four completions and
    kills healthy work. Returns None until quartiles are meaningful, leaving
    the existing subprocess timeout as the only bound, exactly as before.

    Fenced PER STRATEGY once that strategy has enough successes of its own.
    Pooling every strategy into one fence sounds neutral and is not: the pool
    is dominated by whichever strategy finishes fastest, so the fence
    converges on that strategy's pace and guillotines every slower one. On run
    closure_20260826_codex the five successes setting the fence were four
    `analytic` candidates plus one quick fit, putting it at 2249s -- while
    `solver_fit`, which must run the solver inside an optimiser loop, took a
    median of 111 turns and finished 0 times out of 4. The fence killed them,
    which kept them out of the pool, which kept the fence at analytic pace.
    Fencing a strategy against its own kind breaks that loop: a slow strategy
    is judged an outlier only against other attempts at the same kind of work.

    Until a strategy has four successes of its own there is nothing to measure
    it against, so it falls back to the pooled fence and relies on the
    extension path (`oed_extend_candidate`) to earn its first ones.
    """
    def _fence_over(values: List[float]) -> Optional[int]:
        if len(values) < 4:
            return None
        logs = sorted(math.log(d) for d in values)
        q1, _median, q3 = statistics.quantiles(logs, n=4)
        bound = math.exp(q3 + 1.5 * (q3 - q1))
        inliers = [d for d in values if d <= bound]
        return int(max(bound, max(inliers))) if inliers else int(bound)

    durations: List[float] = []
    by_strategy: Dict[str, List[float]] = {}
    try:
        candidate_dirs = sorted(disc_dir.glob("cand_*"))
    except OSError:
        return None
    for candidate in candidate_dirs:
        result = _read_json(candidate / "agentic_result.json") or {}
        if result.get("status") != "OK":
            continue
        value = result.get("duration_s")
        if isinstance(value, (int, float)) and value > 0:
            durations.append(float(value))
            by_strategy.setdefault(_candidate_strategy(candidate), []).append(float(value))
    wanted = str(strategy or "").strip()
    if wanted:
        own = _fence_over(by_strategy.get(wanted, []))
        if own is not None:
            return own
    if len(durations) < 4:
        return None
    logs = sorted(math.log(d) for d in durations)
    q1, _median, q3 = statistics.quantiles(logs, n=4)
    fence = math.exp(q3 + 1.5 * (q3 - q1))
    # Floor on the slowest NON-outlier success. A runaway still reports
    # status OK when it finally returns, so taking a plain max() over every
    # duration lets the one candidate this fence exists to stop raise the
    # fence to its own runtime and exempt itself.
    inliers = [d for d in durations if d <= fence]
    return int(max(fence, max(inliers))) if inliers else int(fence)


def _render_archive_summary(archive: Any, disc_dir: Path) -> str:
    """The archive summary WITH the baseline it should be read against.

    Both call sites used to invoke render_summary() with no arguments, so the
    "Δ vs baseline" column never rendered and the per-case difficulty block had
    nothing to compare against -- the proposer saw absolute scores with no way
    to tell a win from a loss without doing the arithmetic itself. The baseline
    is sitting in baseline_score.json next to the archive; this passes it in.
    """
    baseline = _read_json(disc_dir / "baseline_score.json") or {}
    value = baseline.get("value")
    per_case = baseline.get("per_case")
    try:
        return archive.render_summary(
            baseline_score=float(value) if isinstance(value, (int, float)) else None,
            baseline_direction=str(baseline.get("direction") or "min"),
            baseline_per_case=per_case if isinstance(per_case, dict) else None,
        )
    except TypeError:
        # An older archive without the per-case parameter must not break the
        # loop -- the summary is guidance, not a gate.
        return archive.render_summary()


def _approved_hypothesis_directions(out_dir: Path) -> List[Dict[str, str]]:
    """The concrete modifications the hypothesis stage already produced.

    That stage fetches literature, generates ideas against it, critiques each
    one for physical plausibility and implementability, and puts the survivors
    in front of a human for approval. What comes out is a list of named,
    implementable experiments — exactly what the search is otherwise asked to
    invent unaided.

    Nothing consumed them. `oed_propose_candidates` was told only to "target a
    NEW model family not yet in the archive", so it worked from unaided recall
    and ran dry at 64 of 100 budget units while nine vetted directions sat
    unused on disk — including the APG production-damping idea whose family
    turned out to be the best result of the run.
    """
    approved = _read_json(out_dir / "hypotheses_approved.json") or {}
    directions: List[Dict[str, str]] = []
    for hypothesis in approved.get("approved_hypotheses", []) or []:
        if not isinstance(hypothesis, dict):
            continue
        idea = hypothesis.get("idea") or {}
        for experiment in idea.get("experiments", []) or []:
            if not isinstance(experiment, dict):
                continue
            name = str(experiment.get("name") or experiment.get("experiment_id") or "").strip()
            detail = str(experiment.get("notes") or experiment.get("description") or "").strip()
            if not (name or detail):
                continue
            directions.append({"name": name, "detail": detail[:400]})
    return directions


def _oed_candidate_fingerprint(
    family: str, action_type: str, hypothesis: str, parameters: Optional[Dict[str, Any]] = None,
    strategy: str = "",
) -> str:
    """Identity of a candidate by what it actually *does*, not what it is called.

    Variant names are deduped separately so two candidates never share a
    directory, and a collision there is resolved by appending a suffix. That
    is right for directories and wrong as a guard against repeated work: the
    same experiment proposed under six different names produced six distinct
    slugs, six paid evaluations and six byte-identical scores. Measured on a
    real run, two experiments were each re-run six times — roughly 12 of 29
    budget units spent re-deriving numbers already in history.json.

    Keyed on the family, the action, the normalised hypothesis text and any
    coefficient overrides, so a genuine variation (a different coefficient
    value, the same idea in another family) still reads as new.
    """
    # Strategy is part of the identity. The archive niches on (mechanism,
    # strategy) precisely so the same mechanism can be tried a different way —
    # reasoned out by hand, then fitted through the solver. Without it here,
    # the second attempt fingerprints identically to the first and is thrown
    # away as a duplicate, which makes the whole second dimension unreachable.
    family_key = str(family or "").strip().lower()
    action_key = str(action_type or "").strip().lower()
    strategy_key = str(strategy or "").strip().lower()
    params = parameters if isinstance(parameters, dict) else {}
    param_key = ";".join(f"{k}={float(params[k]):g}" for k in sorted(params))

    if action_key == "experiment" and param_key:
        # A coefficient experiment IS its coefficients. Ignoring the prose here
        # is the whole point: seven candidates that all set c_cd = 1.0 were
        # described seven different ways, from a full paragraph down to
        # "SA-Cross-Diffusion with c_cd=1.0", and every one was paid for.
        # Exact, free, and cannot be wrong — so no model call for this case.
        return "|".join([family_key, action_key, strategy_key, param_key])

    # Anything else has no exact identity to compare, and normalising the prose
    # was never going to give it one. `_llm_duplicate_of` reads the two
    # descriptions and decides; this key only prevents a byte-identical repeat
    # inside a single batch.
    return "|".join([family_key, action_key, strategy_key,
                      " ".join(str(hypothesis or "").lower().split())])


def _run_succeeded(result: Dict[str, Any]) -> bool:
    """Accept both run-result schemas used in this repository."""
    if result.get("success") is True:
        return True
    return str(result.get("status", "") or "").strip().lower() in {"success", "ok"}


def _safe_variant_slug(raw: str, fallback: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(raw or "").strip()).strip("_-")
    return (slug or fallback)[:80]


def _model_coeffs_dict_name(momentum_transport_text: str) -> str:
    """``<ModelName>Coeffs`` for the RAS/LES model a case selects, or ""."""
    match = re.search(r"\bmodel\s+([A-Za-z_][A-Za-z0-9_]*)\s*;", momentum_transport_text)
    if not match:
        match = re.search(r"\bRASModel\s+([A-Za-z_][A-Za-z0-9_]*)\s*;", momentum_transport_text)
    return f"{match.group(1)}Coeffs" if match else ""


def _set_model_coefficients(case_dir: Path, parameters: Dict[str, float]) -> Dict[str, str]:
    """Write turbulence-model coefficients into ``<Model>Coeffs``.

    A compiled model exposes its coefficients through ``coeffDict()``, i.e.
    ``<ModelName>Coeffs`` inside constant/momentumTransport, and OpenFOAM
    materialises that sub-dictionary at RUNTIME via lookupOrAddToDict — it is
    never written back to the case files. So a plain text search over
    constant/ finds nothing to patch, and every coefficient-only experiment
    was rejected as "coefficient names were not found", making the cost-1
    action the search relies on for exploiting a family impossible to use.

    Creates the sub-dictionary when absent, updates entries in place when
    present. Returns {coefficient: relative path written}.
    """
    path = case_dir / "constant" / "momentumTransport"
    if not path.is_file():
        return {}
    text = path.read_text(encoding="utf-8", errors="ignore")
    dict_name = _model_coeffs_dict_name(text)
    if not dict_name:
        return {}

    written: Dict[str, str] = {}
    block = re.search(rf"(^|\n)([ \t]*){re.escape(dict_name)}\s*\n?\s*\{{", text)
    if block:
        start = text.index("{", block.start())
        depth, end = 0, None
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end is None:
            return {}
        body = text[start + 1 : end]
        for key, value in parameters.items():
            entry = rf"(^|\n)(\s*){re.escape(key)}\s+[^;]+;"
            body, count = re.subn(entry, rf"\1\2{key} {value};", body)
            if not count:
                body = body.rstrip() + f"\n        {key} {value};\n    "
            written[key] = "constant/momentumTransport"
        text = text[: start + 1] + body + text[end:]
    else:
        # No coeffs dictionary yet: add one inside the RAS/LES block, which
        # is where coeffDict() resolves from.
        parent = re.search(r"(^|\n)(\s*)(RAS|LES)\s*\n?\s*\{", text)
        if not parent:
            return {}
        start = text.index("{", parent.start())
        depth, end = 0, None
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end is None:
            return {}
        lines = "\n".join(f"        {k} {v};" for k, v in parameters.items())
        insert = f"\n    {dict_name}\n    {{\n{lines}\n    }}\n"
        text = text[:end] + insert + text[end:]
        for key in parameters:
            written[key] = "constant/momentumTransport"

    path.write_text(text, encoding="utf-8")
    return written


def _runtime_coefficients(case_dir: Path) -> List[str]:
    """Coefficient names a compiled model exposes through its coeffDict.

    A model written with ``lookupOrAddToDict("cr1", ...)`` can be re-run with a
    different cr1 for one budget unit and no recompile
    (oed_run_experiment_candidate), instead of two units and a full rebuild.
    The proposer cannot know which names are tunable unless it is told, and on
    a real run it spent cost-2 code_mod rebuilds on nothing but C_cr = 1.2 and
    C_cr = 3.6 against a parent that exposed Ccr all along.

    Read from the parent's own sources, so this reflects what was actually
    compiled rather than what the hypothesis claimed.
    """
    names: List[str] = []
    models_dir = Path(case_dir) / "customModels"
    if not models_dir.is_dir():
        return names
    for source in sorted(models_dir.rglob("*.C")):
        try:
            text = source.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for match in re.finditer(
            r'lookupOrAddToDict\s*\(\s*\n?\s*"([A-Za-z_][A-Za-z0-9_]*)"', text
        ):
            name = match.group(1)
            if name not in names:
                names.append(name)
    return names


def _null_score_evidence(candidate_path: Path, execution_doc: Dict[str, Any],
                         score_error: str) -> str:
    """Everything on disk that bears on why this candidate produced no score.

    Bounded on purpose: a diverged OpenFOAM log is 30 MB and its last 60 lines
    carry the whole story, while the middle would push out the compile error
    that actually explains it.
    """
    lines: List[str] = []
    lines.append(f"CANDIDATE: {candidate_path.name}")
    if score_error:
        lines.append(f"SCORING ERROR: {score_error}")

    lines.append("")
    lines.append("BUILD RESULT (the model's own trial run, on the starter geometry):")
    for key in ("status", "compile_ok", "converged", "error", "compile_error_hint",
                "compiled_model_name", "solver_invocations"):
        if key in execution_doc:
            lines.append(f"  {key}: {str(execution_doc.get(key))[:400]}")

    evaluation = _read_json(candidate_path / "evaluation_run_result.json") or {}
    if evaluation:
        lines.append("")
        lines.append("EVALUATION OVER THE STUDY'S GRADED CASES:")
        lines.append(f"  declared: {evaluation.get('cases_declared')}"
                     f"  succeeded: {evaluation.get('cases_succeeded')}"
                     f"  timed_out: {evaluation.get('timed_out_cases')}"
                     f"  abandoned: {evaluation.get('abandoned')}")
        failures = evaluation.get("failures") or []
        if failures:
            lines.append(f"  {len(failures)} case(s) failed:")
            for entry in failures[:12]:
                lines.append(f"    - {entry.get('case')}: {str(entry.get('error'))[:220]}")
            if len(failures) > 12:
                lines.append(f"    ... and {len(failures) - 12} more")
        else:
            lines.append("  no per-case failures recorded")
    else:
        lines.append("")
        lines.append("EVALUATION: never ran (no evaluation_run_result.json).")

    # The tail of a solver log that actually failed, if one can be found.
    log_source: Optional[Path] = None
    for entry in (evaluation.get("failures") or []):
        directory = Path(str(entry.get("case_dir") or ""))
        if directory.is_dir():
            logs = sorted(directory.glob("log.*"), key=lambda q: q.stat().st_mtime, reverse=True)
            if logs:
                log_source = logs[0]
                break
    if log_source is None:
        case_dir = Path(str(execution_doc.get("case_dir") or ""))
        if case_dir.is_dir():
            logs = sorted(case_dir.glob("log.*"), key=lambda q: q.stat().st_mtime, reverse=True)
            log_source = logs[0] if logs else None
    if log_source is not None:
        try:
            size = log_source.stat().st_size
            with log_source.open("r", encoding="utf-8", errors="ignore") as handle:
                handle.seek(max(0, size - 6000))
                tail = handle.read()
            lines.append("")
            lines.append(f"TAIL OF {log_source.name} ({size} bytes):")
            lines.append(tail[-6000:])
        except OSError:
            pass

    trajectory = candidate_path / "agentic_trajectory.log"
    if trajectory.is_file():
        try:
            size = trajectory.stat().st_size
            with trajectory.open("r", encoding="utf-8", errors="ignore") as handle:
                handle.seek(max(0, size - 3000))
                lines.append("")
                lines.append("TAIL OF THE BUILD AGENT'S TRAJECTORY:")
                lines.append(handle.read()[-3000:])
        except OSError:
            pass
    return "\n".join(lines)


def _diagnose_null_score(candidate_path: Path, execution_doc: Dict[str, Any],
                         score_error: str, settings: Any) -> Dict[str, Any]:
    """Read the wreckage and say what happened, the way a person would.

    A null score used to be a full stop: the candidate was recorded FAILED with
    no reason attached, and the run moved on. That discarded real results --
    cdomega_f1_taper_045_065 solved all 32 graded cases and was the best model
    in run closure_20260826_codex at +4.20%, thrown away because its build's
    trial run had not converged. A person looking at that would have asked why,
    seen the 32 successes, and kept it.

    So the question is put to the model, with the actual evidence: the build
    result, every case's own failure reason, the tail of a log that failed, and
    the build agent's last moves. It answers what went wrong, whether a bounded
    change would fix it, and -- critically -- whether that change would touch
    anything the benchmark grades on, because a "fix" that alters the mesh,
    the physics or the closure under test is not a fix, it is a different
    experiment.

    Never raises: a diagnosis that fails leaves the candidate exactly as it
    was, which is the old behaviour.
    """
    evidence = _null_score_evidence(candidate_path, execution_doc, score_error)
    try:
        from cfd_langgraph.llm.factory import create_langchain_llm

        llm = create_langchain_llm(model=settings.model, temperature=0.0)
        verdict = llm.with_structured_output(_NullScoreDiagnosis).invoke(
            "A candidate model in an automated CFD closure search produced no score.\n"
            "Diagnose it from the evidence below, the way an engineer reading the run "
            "would: what failed, was it the tooling or the model, and is it worth "
            "another attempt.\n\n"
            "Judge the GRADED cases above the build's own trial run. The trial run is "
            "one case on the starter geometry that the build agent used while "
            "developing; the graded cases are what the study is scored on. A model "
            "that solved every graded case has not failed, whatever the trial run did.\n\n"
            "Set alters_graded_setup=true if the repair you propose would change the "
            "mesh, boundary conditions, physics, endTime, or the closure itself. "
            "Relaxing solver tolerances or fixing our own scoring plumbing does not. "
            "Be honest: if the closure diverged, say so and set repairable=false, "
            "because grinding through two more attempts on a broken model wastes "
            "budget that a different mechanism could use.\n\n"
            "Set score_anyway=true when the graded cases are sound enough that a score "
            "computed over them should be trusted despite the trial run. Judge that "
            "from what the cases actually did, not from a count: cases that ran to End "
            "and wrote results are evidence; cases that diverged, timed out, or wrote "
            "nothing are not, and a score built on those is worse than no score. Note "
            "that this study's metric is a mean over the FULL declared set, so a score "
            "will be refused anyway if any declared case is missing — say so in cause "
            "if that is the situation.\n\n"
            + evidence
        )
        result = verdict.model_dump()
        result["ok"] = True
        return result
    except Exception as exc:
        return {
            "ok": False,
            "cause": f"diagnosis unavailable ({type(exc).__name__}: {exc})",
            "category": "unknown",
            "repairable": False,
            "alters_graded_setup": False,
            "repair_steps": [],
            "confidence": 0.0,
        }


def _unclean_finish_evidence(candidate_path: Path, execution_doc: Dict[str, Any],
                             granted_timeout_s: int, max_turns: int) -> str:
    """Everything bearing on why a build agent stopped before finishing.

    Deliberately different from _null_score_evidence, which reads the wreckage
    of a run that produced no number. This reads the wreckage of a run that
    stopped early and may well have produced a number anyway -- a number that
    looks fine and means nothing, because the model it scored was never
    finished. So the evidence here is about COMPLETENESS, not failure: how far
    the agent got, what it left on disk, and above all whether the value it was
    working out ever reached the case that gets graded.

    Nothing here is parsed into a verdict. The model source and the case's
    activation dictionary go in verbatim and the reading of them is the
    diagnosing model's job, because the thing being judged -- "is this
    coefficient the fitted one or the class default?" -- is a question about
    C++ and OpenFOAM semantics, not a pattern a matcher can settle.
    """
    lines: List[str] = []
    lines.append(f"CANDIDATE: {candidate_path.name}")
    lines.append("")
    lines.append("HOW THE BUILD AGENT STOPPED:")
    lines.append(f"  reported status: {execution_doc.get('status')}")
    lines.append(f"  aborted_reason: {execution_doc.get('aborted_reason') or '(none recorded)'}")
    lines.append(f"  error: {str(execution_doc.get('error') or '(none)')[:300]}")
    turns = execution_doc.get("turns_used")
    lines.append(f"  turns used: {turns} of a {max_turns}-turn cap"
                 + ("  <-- AT THE CAP" if isinstance(turns, int) and turns >= max_turns else ""))
    duration = execution_doc.get("duration_s")
    lines.append(f"  wall clock: {duration}s of a {granted_timeout_s}s fence"
                 if granted_timeout_s else f"  wall clock: {duration}s (no fence was set)")
    solves = execution_doc.get("solver_invocations")
    lines.append(f"  solver launches made: {solves}")
    if isinstance(solves, int) and solves > 0 and isinstance(duration, (int, float)) and duration > 0:
        lines.append(f"  => averaged {duration / solves:.0f}s per solver launch, which is the "
                     f"number to reason from when estimating how much longer it needs")
    lines.append(f"  compiled a library: {execution_doc.get('compile_ok')}")
    lines.append(f"  its own trial case converged: {execution_doc.get('converged')}")

    # What the agent left behind. A fit that ran and then died leaves its
    # result on disk -- frozen_alphaK2.json, fit_evidence.json and the like --
    # and that file is the difference between "needs more time" and "needs the
    # number copying into the case".
    lines.append("")
    lines.append("WHAT THE AGENT LEFT IN ITS DIRECTORY:")
    try:
        for entry in sorted(candidate_path.iterdir()):
            kind = "dir " if entry.is_dir() else "file"
            size = entry.stat().st_size if entry.is_file() else 0
            lines.append(f"  {kind} {entry.name}" + (f" ({size} bytes)" if size else ""))
    except OSError:
        lines.append("  (unreadable)")

    # The fit's OWN record of what it did, which is the direct evidence of
    # progress and the thing this diagnosis most often gets wrong without it.
    #
    # Measured on run closure_20260826_codex: bradshaw_b1_solver_fit was
    # diagnosed "abandon" on the stated grounds of "no converged trial,
    # fitted-value artifact, or fresh case-local .so" while eight converged
    # objective evaluations and a selected coefficient sat in
    # objective_ledger.jsonl and fit_status_bounded_035_070.json right beside
    # the trajectory. The diagnosis was reading the agent's last few turns and
    # nothing else, so a fit that was working looked like one that had never
    # started. A solver-in-the-loop fit writes its progress to a ledger by
    # design; that ledger is the answer to "did this make progress", and it
    # belongs in front of whoever is deciding.
    #
    # Ledger lines go in as raw text, oldest and newest, with no interpretation
    # here: whether an objective that moved 0.13% across the whole coefficient
    # range counts as progress is a judgement about the physics, not a
    # threshold this function should be applying.
    ledgers = sorted(
        [q for q in candidate_path.glob("*.jsonl") if "ledger" in q.name.lower()]
        + [q for q in candidate_path.glob("*ledger*.json") if q.is_file()]
    )
    for ledger in ledgers[:3]:
        try:
            rows = ledger.read_text(encoding="utf-8", errors="ignore").strip().splitlines()
        except OSError:
            continue
        if not rows:
            continue
        lines.append("")
        lines.append(f"FIT LEDGER {ledger.name} -- {len(rows)} objective evaluation(s) "
                     f"recorded. This is the fit's own account of its progress:")
        for row in rows[:6]:
            lines.append("  " + row[:600])
        if len(rows) > 12:
            lines.append(f"  ... {len(rows) - 12} more ...")
        for row in rows[max(6, len(rows) - 6):]:
            lines.append("  " + row[:600])

    # Status and self-reported-failure files a fit driver writes when it
    # finishes or gives up. crossgrad_ksource_solver_fit wrote
    # FIT_INCOMPLETE.json saying "incomplete_do_not_score_as_fitted" and
    # nothing was looking at it.
    for pattern in ("fit_status*.json", "FIT_*.json", "frozen_*.json", "fit_*.json"):
        for status in sorted(candidate_path.glob(pattern))[:2]:
            try:
                if status.stat().st_size > 6000:
                    continue
                lines.append("")
                lines.append(f"FIT STATUS {status.name}:")
                lines.append(status.read_text(encoding="utf-8", errors="ignore")[:6000])
            except OSError:
                continue

    # Small JSON artifacts verbatim: these are where a fit records what it
    # converged to, and their contents decide whether the work was done.
    known = {"agentic_result.json", "candidate_record.json", "candidate_invocation.json",
             "evaluation_run_result.json"}
    shown = 0
    for artifact in sorted(candidate_path.glob("*.json")):
        if artifact.name in known or shown >= 4:
            continue
        try:
            if artifact.stat().st_size > 4000:
                continue
            lines.append("")
            lines.append(f"ARTIFACT {artifact.name}:")
            lines.append(artifact.read_text(encoding="utf-8", errors="ignore")[:4000])
            shown += 1
        except OSError:
            continue

    # The compiled model's own source, and the dictionary that activates it in
    # a graded case. Read together these answer the question that matters: is
    # the number the agent worked out actually in the model that will run, or
    # is the model sitting at whatever its class defaults to?
    case_dir = Path(str(execution_doc.get("case_dir") or ""))
    sources = sorted(case_dir.glob("customModels/*/*.C")) if case_dir.is_dir() else []
    sources = [q for q in sources if "lnInclude" not in q.parts]
    if sources:
        try:
            text = sources[0].read_text(encoding="utf-8", errors="ignore")
            lines.append("")
            lines.append(f"COMPILED MODEL SOURCE {sources[0].name} ({len(text)} chars):")
            lines.append(text[:9000] + ("\n... (truncated)" if len(text) > 9000 else ""))
        except OSError:
            pass
    for dict_name in ("momentumTransport", "turbulenceProperties", "transportProperties",
                      "fvModels", "fvOptions"):
        activation = case_dir / "constant" / dict_name
        if activation.is_file():
            try:
                lines.append("")
                lines.append(f"ACTIVATION DICTIONARY constant/{dict_name} IN THE BUILT CASE:")
                lines.append(activation.read_text(encoding="utf-8", errors="ignore")[-2500:])
            except OSError:
                pass

    trajectory = candidate_path / "agentic_trajectory.log"
    if trajectory.is_file():
        try:
            size = trajectory.stat().st_size
            with trajectory.open("r", encoding="utf-8", errors="ignore") as handle:
                handle.seek(max(0, size - 9000))
                lines.append("")
                lines.append(f"TAIL OF THE BUILD AGENT'S TRAJECTORY ({size} bytes total) -- "
                             f"this is what it was doing when it stopped:")
                lines.append(handle.read()[-9000:])
        except OSError:
            pass
    return "\n".join(lines)


def _diagnose_unclean_finish(candidate_path: Path, execution_doc: Dict[str, Any],
                             granted_timeout_s: int, max_turns: int,
                             extensions_used: int, settings: Any) -> Dict[str, Any]:
    """Read an unfinished build and say what should happen to it.

    The gap this closes: diagnosis used to fire only when a candidate produced
    NO score. A build agent that is killed at the wall clock mid-fit produces
    a score -- its library compiled, its cases all run, and every one of them
    runs the closure at whatever the class defaults to, because the coefficient
    the agent was still fitting never reached the case dictionary. Measured on
    run closure_20260826_codex: six candidates came back at 0.1136009392817217
    against a baseline of 0.11360099048446087, bit-identical across five
    different mechanisms, and every one was recorded as a real evaluation of a
    real model. Nothing was null, so nothing was ever diagnosed, and the
    archive learned that fitting does not work from six experiments that never
    ran.

    So this is asked BEFORE the result is accepted, of every agent that did not
    finish cleanly, and it answers a different question from the null-score
    diagnosis: not "why did this fail" but "is what is on disk actually
    finished". The four verdicts it can return map onto the four things a
    person would do -- score it, fix it, give it more time, or drop it.

    Never raises: a diagnosis that fails returns verdict "unknown", which the
    caller treats as "no opinion" and leaves the candidate exactly as it was.
    """
    evidence = _unclean_finish_evidence(candidate_path, execution_doc,
                                        granted_timeout_s, max_turns)
    try:
        from cfd_langgraph.llm.factory import create_langchain_llm

        llm = create_langchain_llm(model=settings.model, temperature=0.0)
        verdict = llm.with_structured_output(_UncleanFinishDiagnosis).invoke(
            "A build agent in an automated CFD closure search stopped before finishing "
            "cleanly. Decide what should happen to its work.\n\n"
            "The question is NOT primarily whether it failed. It is whether what is on "
            "disk is FINISHED. An agent that compiled a parameterised model and was "
            "killed while still fitting its coefficient leaves behind something that "
            "compiles, runs, and scores -- as the unmodified baseline, because the "
            "coefficient it was working out never reached the case. That result looks "
            "healthy and is worthless, and it is the single most important thing for "
            "you to catch.\n\n"
            "So read the model source and the activation dictionary together and ask: "
            "if this case ran right now, would it run the intended model, or would it "
            "run the class defaults? A coefficient the source reads from the case "
            "dictionary, that the case dictionary does not set, is a default. A "
            "coefficient compiled into the source as a literal is not. Set "
            "model_is_complete accordingly, and say in your cause which coefficients "
            "you checked and where you found them.\n\n"
            "Then choose ONE verdict:\n"
            "  complete -- the model is finished and self-contained. The agent ran out "
            "of turns or clock during tidying-up, not during the work. Score it.\n"
            "  repair -- something specific and bounded is broken in our plumbing: a "
            "library that is not being loaded, a coefficient the agent computed and "
            "left in a file but never wrote into the case, a missing dictionary entry. "
            "Name the exact steps.\n"
            "  extend -- the work itself was still genuinely in progress and more time "
            "would plausibly finish it. Only choose this if the trajectory shows real "
            "forward progress, not if the agent was thrashing on the same error.\n"
            "  abandon -- the closure itself is broken, diverged, or ill-posed, or the "
            "agent made no meaningful progress. Say so plainly; a candidate that is "
            "not going to work is better dropped than ground through more attempts.\n\n"
            "If a FIT LEDGER is shown, it is the fit's own record of what it "
            "achieved and it outranks anything the trajectory tail suggests. A ledger "
            "with converged objective evaluations in it means the fit WAS working, "
            "however the agent's last turns looked. Read it for three things:\n"
            "  - how many evaluations completed, and how long each one took;\n"
            "  - whether a coefficient was selected and written down;\n"
            "  - whether the objective actually MOVES across the range explored. An "
            "objective that varies by a fraction of a percent between the extremes "
            "means the coefficient does nothing, and more time will buy more of "
            "nothing -- say so and abandon, however healthy the fit machinery looks.\n"
            "Watch for two specific traps the ledger exposes: an optimiser whose "
            "bounds exclude the best value it actually measured, and an optimiser that "
            "reported success after a handful of evaluations, which is a box-corner, "
            "not a fit.\n\n"
            "For extend you must justify extra_seconds_needed with arithmetic in "
            "estimate_basis, from the evidence: how many solver launches or optimiser "
            "iterations remain, and how long each has actually been taking. An estimate "
            "with no arithmetic behind it will be refused. Be realistic rather than "
            "generous -- the time comes out of the same budget every other candidate "
            "draws on. If the arithmetic says the remaining work needs more than a few "
            "hours, extending is the wrong answer however real the progress: the work "
            "itself is too big for one slot, and the honest verdict is abandon with the "
            "cost stated in cause, so the search can propose a cheaper version of the "
            "same idea.\n\n"
            "A repair may fix our own plumbing. It may NOT change the mesh, the "
            "boundary conditions, the physics, the endTime, or the closure under test: "
            "set alters_graded_setup=true if the repair you describe would touch any of "
            "those, and it will not be applied.\n\n"
            f"This candidate has already been extended {extensions_used} time(s).\n\n"
            + evidence
        )
        result = verdict.model_dump()
        result["ok"] = True
        return result
    except Exception as exc:
        return {
            "ok": False,
            "cause": f"diagnosis unavailable ({type(exc).__name__}: {exc})",
            "verdict": "unknown",
            "model_is_complete": False,
            "stopped_because": "unknown",
            "work_completed": "",
            "extra_seconds_needed": 0,
            "estimate_basis": "",
            "repair_steps": [],
            "alters_graded_setup": False,
            "confidence": 0.0,
        }


def _extract_target_improvement_pct(topic: str) -> float:
    """Best-effort extraction of an explicit improvement target.

    Returns zero when the topic only asks to beat baseline without naming a
    margin.  This is intentionally conservative: an arbitrary percentage is
    never invented.
    """
    text = str(topic or "")
    # A period is allowed through the gap ONLY as a decimal point (followed by
    # a digit), so an intervening baseline value does not look like the end of
    # a sentence. Without this, the natural phrasing "beating the baseline
    # (Cf RMSE 0.004297) by at least 10%" extracted 0% — the decimal point in
    # the quoted baseline broke the match, leaving the search with no success
    # threshold at all.
    gap = r"(?:[^.%\n]|\.(?=\d))"
    # "by 10%", "by at least 10%", "by more than 10%", "by over 10%", "by >=10%".
    qual = r"(?:at\s+least\s+|more\s+than\s+|greater\s+than\s+|over\s+|>=\s*|≥\s*|about\s+|around\s+)?"
    patterns = [
        rf"(?:beat(?:s|ing)?|outperform(?:s|ing)?)(?:{gap}){{0,80}}?\bby\s*{qual}(\d+(?:\.\d+)?)\s*%",
        r"(?:target(?:ing)?|at\s+least|>=|≥|by)\s*(\d+(?:\.\d+)?)\s*%\s*(?:improvement|better|reduction)",
        rf"(?:improv(?:e|ement)|reduc(?:e|tion))(?:{gap}){{0,40}}?{qual}(\d+(?:\.\d+)?)\s*%",
        r"(\d+(?:\.\d+)?)\s*%\s*(?:improvement|better|reduction)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                return max(0.0, float(match.group(1)))
            except Exception:
                pass
    return 0.0


def _llm_target_improvement_pct(topic: str, settings: Any) -> float:
    """The success threshold the study's own prompt asks for.

    Read by the model, not by pattern matching. A regex has to anticipate the
    phrasing, and it silently returns 0 — "no threshold at all" — when it
    guesses wrong: the natural sentence "beating the unmodified baseline
    (Cf RMSE 0.004297) by at least 10%" defeated it, because the decimal point
    in the quoted baseline looked like the end of a sentence. Silence is the
    worst failure mode here, since a 0% threshold makes any candidate that is
    not strictly worse count as a success.

    ``_extract_target_improvement_pct`` remains the fallback for when the model
    is unavailable or answers unusably.
    """
    from cfd_langgraph.llm.factory import create_langchain_llm

    fallback = _extract_target_improvement_pct(topic)
    try:
        llm = create_langchain_llm(model=settings.model, temperature=0.0)
        reply = llm.invoke([
            (
                "system",
                "Read a research topic and report the minimum improvement over "
                "baseline it demands, as a percentage.\n"
                "Reply with STRICT JSON only: {\"target_improvement_pct\": <number or null>}.\n"
                "Use null when the topic asks to beat baseline without naming a "
                "margin. Never invent a margin. A number quoted as the baseline "
                "VALUE is not a target. Percentages of anything other than the "
                "improvement over baseline are not the target either.",
            ),
            ("user", f"TOPIC:\n{topic}"),
        ])
        text = getattr(reply, "content", "")
        if isinstance(text, list):
            text = "".join(b.get("text", "") for b in text if isinstance(b, dict))
        match = re.search(r"\{.*\}", str(text), re.S)
        if match:
            value = json.loads(match.group(0)).get("target_improvement_pct")
            if value is None:
                return fallback
            parsed = float(value)
            if 0.0 <= parsed <= 100.0:
                return parsed
    except Exception as exc:
        print(f"[oed] improvement-target extraction fell back to pattern matching: {exc}")
    return fallback


def _improvement_pct(value: float, baseline: float, direction: str) -> float:
    denom = max(abs(float(baseline)), 1e-12)
    if str(direction).lower() == "max":
        return (float(value) - float(baseline)) / denom * 100.0
    return (float(baseline) - float(value)) / denom * 100.0


def _foamagent_env(openfoam_path: str = "") -> Dict[str, str]:
    """Environment for subprocess calls that need OpenFOAM and a model.

    Two things the default environment does not provide:

    ``FOAMAGENT_MODEL_PROVIDER`` / ``FOAMAGENT_MODEL_VERSION`` are a second
    config surface, separate from ``cfd_langgraph.llm.factory``, that some
    scripts still read as a model fallback (see ``code_mod_agentic.py``).
    They are mirrored straight from ``CFD_SCIENTIST_LLM_PROVIDER`` /
    ``CFD_SCIENTIST_MODEL`` with no remapping: a provider that isn't supported
    downstream should fail loudly with a clear provider error, not get quietly
    rerouted to a different provider and budget the way the old
    ``foam/runner.py`` wrapper's ``gemini -> openai`` hack did.

    It also starts from ``resolve_openfoam_env(openfoam_path)`` rather than a
    bare ``os.environ.copy()``: these subprocesses read ``WM_PROJECT_DIR``
    from their own environment the same way Allrun scripts do, and have the
    identical failure mode (see ``foam_native/openfoam_env.py``) if the
    launching shell never sourced OpenFOAM.
    """
    env = resolve_openfoam_env(openfoam_path)
    provider = (env.get("CFD_SCIENTIST_LLM_PROVIDER") or env.get("CFD_SCIEINTIST_LLM_PROVIDER") or "").strip().lower()
    model = (env.get("CFD_SCIENTIST_MODEL") or env.get("CFD_SCIENITST_MODEL") or "").strip()
    if provider:
        env["FOAMAGENT_MODEL_PROVIDER"] = provider
    if model:
        env["FOAMAGENT_MODEL_VERSION"] = model
    env["PYTHONPATH"] = f"{_REPO_ROOT}:{env.get('PYTHONPATH', '')}"
    return env


class _StudyMetric(BaseModel):
    """One quantity this study is judged on, and how to compute it."""

    name: str = Field(description="Short identifier, e.g. 'Cf' or 'cf_rmse'. Used as the QoI key.")
    description: str = Field(default="", description="What it is, in one line.")
    direction: str = Field(default="min", description="'min' or 'max' — which way is better.")
    computation_hint: str = Field(
        description="Exactly how to compute it from a finished case: which field, which patch, "
        "the formula, and where every constant comes from (name the file and the value). "
        "Anyone following this must get the same number."
    )
    reference_files: List[str] = Field(
        default_factory=list,
        description="Every external data file this metric is computed against, as a repository-"
        "relative path copied verbatim from the inventory you were given. Column names inside a "
        "file are NOT files. Empty if the metric needs no reference data.",
    )


class _StudyMetrics(BaseModel):
    metrics: List[_StudyMetric] = Field(description="Most important first.")
    reason: str = Field(default="", description="One line: why these, from the prompt/starter files.")


def _reference_file_inventory(out_dir: Path) -> List[str]:
    """Every data file in the starter case a metric could legitimately be
    computed against, repository-relative.

    Given to the model up front so it never has to guess a path, and used
    afterwards to check that what it named actually exists.
    """
    understanding = _read_json(out_dir / "starter_understanding.json") or {}
    starter = str(understanding.get("starter_dir") or "").strip()
    roots: List[Path] = []
    if starter:
        candidate = Path(starter)
        if not candidate.is_absolute():
            candidate = _REPO_ROOT / candidate
        candidate = candidate.resolve()
        roots.append(candidate)
        # Reference data commonly sits beside the case rather than inside it
        # (starter_oed_turbulence/reference_data/ vs .../periodic_hill_sa/),
        # so the parent is worth walking too -- but only while it is still
        # inside a starter tree. When starter_dir is already a top-level
        # starter folder its parent IS the repository root, and walking that
        # returns the whole repo (measured: 1881 files, including
        # knowledge_bundle/), burying the handful of real reference files.
        parent = candidate.parent
        if parent != _REPO_ROOT.resolve() and _REPO_ROOT.resolve() in parent.parents:
            roots.append(parent)
    found: List[str] = []
    seen = set()
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".csv", ".dat", ".txt", ".json"}:
                continue
            try:
                rel = str(path.resolve().relative_to(_REPO_ROOT))
            except ValueError:
                rel = str(path.resolve())
            if rel not in seen:
                seen.add(rel)
                found.append(rel)
    return found


def _unresolvable_reference_files(specs: List[Dict[str, Any]]) -> List[str]:
    """Declared reference files that do not exist on disk.

    A metric spec is a contract every later stage executes against; a path in
    it that resolves to nothing cannot be discovered until a solver has
    already run. Measured on run oed_20260823_opus_low: the spec named
    "reference_data/cf_dns_exactmatch" -- a CSV *column* written as if it
    were a file -- so the extractor could not compute the metric at all, and
    the mesh gate failed after 506s of solver time having produced only
    generic field statistics.
    """
    missing: List[str] = []
    for spec in specs:
        for declared in spec.get("reference_files") or []:
            text = str(declared).strip()
            if not text:
                continue
            path = Path(text)
            if not path.is_absolute():
                path = _REPO_ROOT / path
            if not path.is_file():
                missing.append(text)
    return missing


def _latest_solved_time(case_dir: Path) -> Optional[str]:
    """The largest non-zero time directory in a case, or None if it never wrote.

    Time 0 does not count: it is the initial condition, and scoring it reports
    how good the initial guess was.
    """
    best: Optional[float] = None
    try:
        children = list(case_dir.iterdir())
    except OSError:
        return None
    for child in children:
        if not child.is_dir():
            continue
        try:
            value = float(child.name)
        except ValueError:
            continue
        if value > 0 and (best is None or value > best):
            best = value
    if best is None:
        return None
    return f"{best:g}"


class _EvalProc:
    """A CompletedProcess-shaped result that can also report a timeout."""

    def __init__(self, returncode: int, stdout: str, stderr: str, timed_out: bool = False):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.timed_out = timed_out


def _run_one_evaluation_case(replica: Path, application: str, env: Dict[str, str]) -> _EvalProc:
    """One evaluation-case solve, where a timeout is a result rather than a raise.

    A candidate whose closure is pathologically slow — fighting the linear
    solvers rather than merely adding terms — will exceed the fence on every
    case. Measured on run closure_20260826_codex: stock SST solves CBFS's
    30,000 iterations in 1016s, and a candidate was still running at 3600s.
    That is information about the candidate, and it belongs in its record; a
    raised TimeoutExpired instead ended the manager's whole step and threw away
    an hour of compile work.
    """
    try:
        proc = subprocess.run(
            ["bash", "-lc", f'cd "{replica}" && {application}'],
            capture_output=True, text=True, timeout=_OED_EVAL_CASE_TIMEOUT_S, env=env,
        )
        return _EvalProc(proc.returncode, proc.stdout or "", proc.stderr or "")
    except subprocess.TimeoutExpired as exc:
        def _text(stream: Any) -> str:
            if stream is None:
                return ""
            return stream.decode("utf-8", "replace") if isinstance(stream, bytes) else str(stream)

        return _EvalProc(
            124,
            _text(exc.stdout),
            _text(exc.stderr)
            + f"\n[timeout] {application} exceeded {_OED_EVAL_CASE_TIMEOUT_S}s on {replica.name}.",
            timed_out=True,
        )


def _openfoam_failure_reason(proc: Any) -> str:
    """Why an OpenFOAM run failed, in a form the manager can act on.

    OpenFOAM writes FOAM FATAL ERROR to STDOUT, not stderr, and exits non-zero.
    Reporting only the stderr tail therefore hands back an empty string for the
    most common failures — a missing initial field, an unreadable dictionary, a
    model the case cannot provide inputs for. Measured while testing the
    evaluation loop: three cases failed with "cannot find file .../0/nuTilda"
    and the tool reported nothing but "No evaluation case ran successfully",
    which is unactionable.

    Prefers the FATAL block, falls back to the stdout tail, then stderr.
    """
    stdout = getattr(proc, "stdout", "") or ""
    stderr = getattr(proc, "stderr", "") or ""
    marker = stdout.find("FOAM FATAL")
    if marker >= 0:
        block = stdout[marker:marker + 400]
        return " ".join(block.split())
    marker = stderr.find("FOAM FATAL")
    if marker >= 0:
        return " ".join(stderr[marker:marker + 400].split())
    if stderr.strip():
        return " ".join(stderr[-400:].split())
    return " ".join(stdout[-400:].split()) or "solver did not reach End and gave no diagnostic"


def _case_application(case_dir: Path) -> str:
    """The solver an OpenFOAM case asks for, read from its own controlDict.

    Read rather than assumed: the benchmark sets studied here run simpleFoam,
    but a compressible or transient case in the same study needs its own
    application, and silently running the wrong solver produces a converged,
    plausible, meaningless field.
    """
    control = case_dir / "system" / "controlDict"
    try:
        text = control.read_text()
    except OSError:
        return ""
    match = re.search(r"(?m)^\s*application\s+(\w+)\s*;", text)
    return match.group(1) if match else ""


def _evaluation_cases(disc_dir: Path) -> List[Path]:
    """The cases every candidate is scored on, in a fixed order.

    A study whose objective is one flow scores one case, which is what every
    run so far has done. A benchmark-style objective scores a SET of held-out
    cases and reports their mean, and the set has to be declared once at setup
    rather than chosen per candidate — otherwise two candidates can be scored
    on different cases and their scores are not comparable, which is the one
    thing a search archive cannot tolerate.

    Empty list means single-case scoring: the candidate's own case, exactly as
    before. Nothing about existing studies changes.
    """
    config = _read_json(disc_dir / "search_config.json") or {}
    raw = config.get("evaluation_cases") or []
    cases: List[Path] = []
    for entry in raw:
        path = Path(str(entry)).expanduser()
        if not path.is_absolute():
            path = (_REPO_ROOT / path).resolve()
        cases.append(path)
    return cases


def _install_model_into_case(
    source_case: Path, target_case: Path, model_name: str
) -> Optional[str]:
    """Put a compiled custom model into another case so it can be run there.

    Returns an error string, or None on success.

    The candidate agent builds and validates its model on ONE case, which is
    the right feedback loop: fast, and every compile error surfaces against a
    case it already understands. Spreading that model over the remaining
    evaluation cases is mechanical -- copy the built library, point the case
    at it -- and doing it here rather than asking the agent to repeat itself
    seven more times keeps agent turns for the part that needs judgement.
    """
    custom = source_case / "customModels"
    if not custom.is_dir():
        return f"no customModels/ in {source_case}"
    if not target_case.is_dir():
        return f"target case not found: {target_case}"

    target_custom = target_case / "customModels"
    if target_custom.exists():
        shutil.rmtree(target_custom)
    try:
        shutil.copytree(custom, target_custom)
    except OSError as exc:
        return f"could not copy customModels: {exc}"

    # Point the case at the model.
    #
    # Two file names and two key spellings, and they do not pair up the way
    # you would guess. OpenFOAM 8+ renamed constant/turbulenceProperties to
    # constant/momentumTransport AND renamed the key from `RASModel` to
    # `model`; the closure-challenge cases are from an older lineage and use
    # `RASModel` inside turbulenceProperties. A study can legitimately involve
    # either, so both are handled and whichever is present is rewritten.
    #
    # A substitution that matches nothing is a hard error, not a no-op: the
    # case would then run its ORIGINAL closure while the framework recorded
    # the result as the candidate's score — a silently wrong number, which is
    # far worse than a failed install.
    substitutions = 0
    for dict_name in ("momentumTransport", "turbulenceProperties"):
        dict_path = target_case / "constant" / dict_name
        if not dict_path.is_file():
            continue
        try:
            text = dict_path.read_text()
        except OSError as exc:
            return f"could not read {dict_path}: {exc}"
        for key in ("model", "RASModel", "LESModel"):
            text, count = re.subn(
                rf"(?m)^(\s*{key}\s+)\S+;", rf"\g<1>{model_name};", text
            )
            substitutions += count
        try:
            dict_path.write_text(text)
        except OSError as exc:
            return f"could not write {dict_path}: {exc}"
    if substitutions == 0:
        return (
            f"could not point {target_case.name} at {model_name}: no model/RASModel/LESModel "
            "entry found in constant/momentumTransport or constant/turbulenceProperties. "
            "Refusing to leave the case running its original closure."
        )

    # And load the library. By ABSOLUTE path: OpenFOAM dlopens the string as
    # given, so a bare filename is only found if the .so sits on the loader
    # path, which a case-local build never does. Every working candidate this
    # repository has produced writes the full path — e.g.
    #   libs ("/.../cand_x/x/customModels/SANegCb2Diff/libSANegCb2Diff.so");
    # and a bare name fails at startup with a dlopen error.
    control = target_case / "system" / "controlDict"
    if control.is_file():
        try:
            text = control.read_text()
        except OSError as exc:
            return f"could not read controlDict: {exc}"
        so_files = sorted({f.name: f for f in target_custom.rglob("*.so")}.values())
        if not so_files:
            return f"no compiled .so found under {target_custom}"
        names = " ".join(f'"{f.resolve()}"' for f in so_files)

        # Appended as a NEW top-level entry rather than rewriting an existing
        # `libs` line. controlDict carries `libs` inside functionObject blocks
        # too (libfieldFunctionObjects.so and friends); replacing the first
        # match in the file would clobber one of those and still leave the
        # model unloaded.
        text = re.sub(
            r"(?m)^libs\s*\([^)]*\)\s*;\s*$\n?", "", text
        )
        text = text.rstrip() + f"\n\nlibs ( {names} );\n"
        try:
            control.write_text(text)
        except OSError as exc:
            return f"could not write controlDict: {exc}"
    return None


class _NullScoreDiagnosis(BaseModel):
    """Why a candidate produced no score, and whether that is repairable."""

    cause: str = Field(
        description="What actually went wrong, in one or two sentences, citing the "
                    "specific evidence that says so."
    )
    category: str = Field(
        description="One of: harness (our tooling/scoring/plumbing is at fault, the "
                    "physics may be fine); case_setup (a case's numerics or output "
                    "settings, not the closure); model_physics (the closure itself "
                    "diverged or is ill-posed); unknown."
    )
    repairable: bool = Field(
        description="True only if a concrete, bounded change would plausibly turn this "
                    "into a real score. False when the closure itself is the problem."
    )
    alters_graded_setup: bool = Field(
        description="True if the repair would change anything the benchmark grades on: "
                    "the mesh, the boundary conditions, the physics, endTime, or the "
                    "closure being tested. Such a repair invalidates the comparison and "
                    "must not be applied automatically."
    )
    repair_steps: List[str] = Field(
        default_factory=list,
        description="Concrete, ordered steps a shell-and-file agent could carry out. "
                    "Empty when not repairable.",
    )
    score_anyway: bool = Field(
        default=False,
        description="True if the graded evaluation cases are sound enough that a score "
                    "computed over them should be trusted, even though the build's own "
                    "trial run failed. False when the evaluation itself is compromised.",
    )
    confidence: float = Field(
        default=0.0, description="0-1 confidence that the stated cause is the real one."
    )


class _UncleanFinishDiagnosis(BaseModel):
    """What happened to a build agent that stopped early, and what to do next."""

    cause: str = Field(
        description="What actually stopped it and how far it had got, in two or three "
                    "sentences, citing the specific evidence. Say which coefficients "
                    "you checked and whether you found them in the case dictionary or "
                    "only as class defaults."
    )
    stopped_because: str = Field(
        description="One of: timeout (hit the wall clock); turn_cap (used every turn); "
                    "llm_error (the provider failed); crash; unknown."
    )
    work_completed: str = Field(
        default="",
        description="What the agent actually finished before it stopped -- compiled the "
                    "library, ran N of M optimiser iterations, wrote the fitted value "
                    "to a file, and so on. This is what a continuation would build on."
    )
    model_is_complete: bool = Field(
        description="True only if the model that would run right now is the intended "
                    "one. False if any coefficient it depends on is absent from the "
                    "case dictionary and would fall back to a class default, which "
                    "makes the run a disguised baseline rather than an experiment."
    )
    verdict: str = Field(
        description="One of: complete (finished, score it); repair (bounded plumbing "
                    "fix, name the steps); extend (real progress, needs more time); "
                    "abandon (broken or going nowhere)."
    )
    extra_seconds_needed: int = Field(
        default=0,
        description="For verdict=extend only: additional wall-clock seconds to grant. "
                    "Zero otherwise.",
    )
    estimate_basis: str = Field(
        default="",
        description="The arithmetic behind extra_seconds_needed -- how much work "
                    "remains and how long each unit of it has been taking. Required "
                    "for verdict=extend; an estimate without it is refused.",
    )
    repair_steps: List[str] = Field(
        default_factory=list,
        description="For verdict=repair: concrete, ordered steps a shell-and-file agent "
                    "could carry out. Empty otherwise.",
    )
    alters_graded_setup: bool = Field(
        default=False,
        description="True if the repair described would change the mesh, boundary "
                    "conditions, physics, endTime, or the closure under test. Such a "
                    "repair invalidates the comparison and is never applied.",
    )
    confidence: float = Field(
        default=0.0, description="0-1 confidence that the stated cause is the real one."
    )


class _SearchQuery(BaseModel):
    """A literature search query distilled from a study objective."""

    query: str = Field(
        description="The search query: the physics question only, 6-15 words, "
        "the words a researcher would type into a paper search."
    )
    broader_query: str = Field(
        default="",
        description="A shorter, more general fallback for when the first query "
        "returns too little. Name the field and the phenomenon, nothing else.",
    )


def _literature_query(topic: str, llm: Any) -> Tuple[str, str]:
    """Turn a study objective into something a paper search can match.

    A study objective is written for the pipeline: it carries file paths, metric
    names, case counts, thresholds and directory names. A bibliographic search
    matches none of that. Measured on run closure_20260824_codex, the objective
    was passed through verbatim and Semantic Scholar returned TWO papers, one of
    them about aircraft intake S-ducts; ideation then had nothing to ground on,
    proposed over-broad studies, and the critic rejected all six — three times
    over, at roughly 26 minutes a round.

    Distilled by the model rather than by pattern-stripping: which part of an
    objective is the research question and which part is configuration is a
    judgement about meaning, and no amount of removing paths and numbers turns
    "lower the equally weighted mean velocity_mae across 32 cases" into
    "data-driven RANS closure for separated flow".

    Returns (query, broader_query). On any failure the original topic is
    returned unchanged — a bad search is better than no search.
    """
    text = str(topic or "").strip()
    if not text:
        return "", ""
    try:
        decided = llm.with_structured_output(_SearchQuery).invoke(
            "Turn this research objective into a literature search query.\n\n"
            "Keep the physics: the flow, the model, the phenomenon, the method. "
            "Drop everything that is specific to how this particular study is run "
            "— file paths, directory names, metric variable names, case counts, "
            "percentage targets, budget numbers, tool names. A search engine "
            "matches published titles and abstracts, and none of those appear in "
            "one.\n\n"
            "Write what a researcher would type to find prior work on the same "
            "question.\n\n"
            f"OBJECTIVE:\n{text[:4000]}"
        )
        query = str(getattr(decided, "query", "") or "").strip()
        broader = str(getattr(decided, "broader_query", "") or "").strip()
        return (query or text), broader
    except Exception as exc:
        print(f"[lit] could not distil a search query ({type(exc).__name__}: {exc}); "
              "searching on the objective as written", flush=True)
        return text, ""


def _study_resources(out_dir: Path, disc_dir: Optional[Path] = None) -> str:
    """What this study actually has to work with, as prompt text.

    The proposer cannot choose a strategy it does not know is possible. It
    will not propose fitting a correction to high-fidelity data if nothing
    ever tells it that twenty cases of such data are sitting on disk, and it
    will not propose an optimiser-driven fit if it does not know scipy is
    installed. Measured over runs oed_20260822_1626_codex_high and
    oed_20260823_opus_low: 0 of 86 proposed candidates involved fitting of any
    kind, and 0 of 90 candidate trajectories ever imported sklearn, scipy
    optimisers or torch — with all three installed the whole time.

    So this is an inventory, not an instruction. It reports the data, the
    libraries and the measured price of one solver run, and leaves the
    proposer to draw its own conclusion about how to spend the budget.
    """
    lines: List[str] = []

    understanding = _read_json(out_dir / "starter_understanding.json") or {}
    starter = str(understanding.get("starter_dir") or "").strip()
    reference_files = _reference_file_inventory(out_dir)

    lines.append("RESOURCES AVAILABLE TO A CANDIDATE (facts, not instructions):")
    if starter:
        lines.append(f"  starter case (read-only): {starter}")
    if reference_files:
        lines.append(f"  high-fidelity / reference data files ({len(reference_files)}):")
        for path in reference_files[:20]:
            lines.append(f"    {path}")
        if len(reference_files) > 20:
            lines.append(f"    ... and {len(reference_files) - 20} more")
    else:
        lines.append("  high-fidelity / reference data files: none found")

    # Libraries a candidate can actually import inside run_bash. Probed, not
    # assumed: claiming a library that is absent sends a candidate down a path
    # that fails at import time and wastes its whole budget.
    available: List[str] = []
    for module in ("numpy", "scipy", "sklearn", "torch", "pandas", "pyvista"):
        try:
            __import__(module)
            available.append(module)
        except Exception:
            continue
    lines.append(f"  python libraries importable in the candidate sandbox: {', '.join(available) or 'none'}")
    if "scipy" in available:
        lines.append(
            "    (scipy.optimize provides least_squares, minimize and "
            "differential_evolution; a candidate may call them from run_bash)"
        )

    # The measured price of one solver run, from this study's own history.
    # Without it, "run the solver 40 times inside a fit" is a decision made
    # blind.
    disc = disc_dir if disc_dir is not None else (out_dir / "open_ended_discovery")
    durations: List[float] = []
    solver_counts: List[int] = []
    try:
        for candidate in sorted(Path(disc).glob("cand_*")):
            result = _read_json(candidate / "agentic_result.json") or {}
            if result.get("status") != "OK":
                continue
            duration = result.get("duration_s")
            runs = result.get("solver_invocations")
            if isinstance(duration, (int, float)) and duration > 0:
                durations.append(float(duration))
            if isinstance(runs, int) and runs > 0:
                solver_counts.append(runs)
    except OSError:
        pass
    if durations:
        durations.sort()
        median = durations[len(durations) // 2]
        lines.append(
            f"  measured cost: a completed candidate has taken a median of "
            f"{median:.0f}s wall clock in this study ({len(durations)} samples)"
        )
    if solver_counts:
        solver_counts.sort()
        lines.append(
            f"  measured solver launches per candidate: median "
            f"{solver_counts[len(solver_counts) // 2]}, max {solver_counts[-1]}"
        )
    return "\n".join(lines)


def _study_metrics(out_dir: Path, llm: Any) -> List[Dict[str, Any]]:
    """The quantities this study is judged on — decided ONCE, reused everywhere.

    Decided from the user's own prompt (verbatim, in ``user_prompt.txt``) and
    what reading the starter folder established, right after the hypotheses and
    before anything measures anything. If the prompt names a metric that is the
    answer; otherwise the model picks one appropriate to the objective.

    Written to ``study_metrics.json`` and read back on every later call, because
    a metric re-derived per call site is a different metric. Measured, twice, on
    the same case in the same hour: the mesh gate's extractor produced
    Cf = -2.755e-04 on one run and -3.939e-07 on the next — the second having
    quietly used Ub = 1.0 instead of the case's 0.028, a factor of 1276, the
    exact error the study's own scoring contract warns about. Carrying the
    formula and the constants with the metric leaves nothing to re-derive.
    """
    spec_path = out_dir / "study_metrics.json"
    existing = _read_json(spec_path)
    if isinstance(existing, list) and existing:
        return existing

    prompt_path = Path(out_dir) / "user_prompt.txt"
    try:
        prompt_text = prompt_path.read_text(encoding="utf-8").strip()
    except Exception:
        prompt_text = ""
    understanding = _read_json(out_dir / "starter_understanding.json") or {}
    reference = understanding.get("reference_data") or {}
    if not prompt_text and not understanding:
        return []
    inventory = _reference_file_inventory(out_dir)
    inventory_text = (
        "\n".join(f"  {path}" for path in inventory)
        if inventory else "  (none found — this study may need no reference data)"
    )
    base_prompt = (
        "Decide the quantity (or quantities) this CFD study is judged on. This decision is made "
        "once and used for every measurement in the study — mesh independence, candidate scoring, "
        "final comparison — so it must be unambiguous.\n\n"
        "If the objective names a metric, that metric is the answer; do not add conventional "
        "extras because they are usual. If it names none, choose what the objective actually "
        "implies. Prefer one metric unless the objective genuinely needs more.\n\n"
        "For each, `computation_hint` must be complete enough that two people following it "
        "independently get the same number: name the field, the boundary patch if it is a wall "
        "quantity, the formula, and where EVERY constant comes from — the file it is read from "
        "and its value in this case. A hint that leaves a reference velocity or length scale to "
        "be guessed is not acceptable.\n\n"
        "Every reference data file you rely on MUST be listed in `reference_files`, copied "
        "verbatim from the inventory below, and referred to by that same path in the hint. Do "
        "not invent a path, and do not write a column name as if it were a file.\n\n"
        f"REFERENCE DATA FILES THAT EXIST (the only paths you may name):\n{inventory_text}\n\n"
        f"THE STUDY'S OBJECTIVE, VERBATIM:\n{prompt_text}\n\n"
        f"WHAT THE STARTER CASE PROVIDES:\n"
        f"  reference quantities: {reference.get('quantities')}\n"
        f"  how results are compared: {str(reference.get('usage_guidance') or '')[:800]}\n"
        f"  flow parameters: {understanding.get('flow_parameters')}\n"
    )

    # A metric spec naming a file that does not exist is unexecutable, and
    # nothing downstream can tell that apart from a metric that is merely
    # hard to compute — it surfaces as an empty extractor result after a
    # solver has already run. Check it here, where the only cost of being
    # wrong is another model call.
    prompt = base_prompt
    for attempt in range(1, 4):
        try:
            decided = llm.with_structured_output(_StudyMetrics).invoke(prompt)
        except Exception as exc:
            print(f"[study] could not decide the study metric ({type(exc).__name__}: {exc})", flush=True)
            return []
        specs = [m.model_dump() for m in (decided.metrics or []) if str(m.name).strip()]
        if not specs:
            return []
        missing = _unresolvable_reference_files(specs)
        if not missing:
            _write_json(spec_path, specs)
            print(
                f"[study] metric(s) for this study: {[m['name'] for m in specs]} — {decided.reason[:140]}",
                flush=True,
            )
            if inventory:
                print(f"[study] reference data verified on disk: {sorted({f for m in specs for f in (m.get('reference_files') or [])})}", flush=True)
            return specs
        print(
            f"[study] attempt {attempt}: reference file(s) do not exist: {missing} — asking again",
            flush=True,
        )
        prompt = (
            base_prompt
            + "\n\nCORRECTION — your previous answer named reference files that do not exist "
            f"on disk:\n{chr(10).join('  ' + m for m in missing)}\n"
            "Every path in `reference_files` must be copied verbatim from the inventory above. "
            "If one of those names is a column inside a file rather than a file, name the file "
            "that contains the column instead, and say in the hint which column to read.\n"
        )

    print(
        f"[study] refusing to write a metric spec whose reference files do not exist: {missing}. "
        "Every later stage executes this spec; a bad path here fails only after a solver has run.",
        flush=True,
    )
    return []


def _starter_case_context(
    out_dir: Path, openfoam_path: str = "", *, include_scoring: bool = True
) -> str:
    """The already-fixed case setup, as prompt text for ideation and critique.

    Both stages were blind to it. The ideator therefore proposed studies of a
    *different* configuration — on a real run every candidate was a
    setup-sensitivity study, several at Re_H=10595 when the starter case is
    Re_H=5600 — and the reviewer, asked to judge whether an idea specifies
    geometry/mesh/BCs/solver, rejected all six for not restating details the
    case had already decided. Zero hypotheses survived, and the study had
    nothing to approve.
    """
    understanding = _read_json(out_dir / "starter_understanding.json")
    if not isinstance(understanding, dict):
        return ""
    flow = understanding.get("flow_parameters") or {}
    reference = understanding.get("reference_data") or {}
    lines = []

    # Read from disk, not from the model's summary of the starter folder. The
    # summary describes the *case*; the OpenFOAM version and the solver are
    # facts about the installation and the controlDict, and leaving them out
    # is what let requirement generation write "OpenFOAM-v2312" on a machine
    # running OpenFOAM 10, and clone a custom solver from pimpleFoam for a
    # case whose controlDict says simpleFoam.
    version = ""
    openfoam_path = openfoam_path or get_settings().openfoam_path
    bashrc = Path(openfoam_path or "") / "etc" / "bashrc"
    if bashrc.is_file():
        for line in bashrc.read_text(errors="ignore").splitlines():
            stripped = line.strip().removeprefix("export ").strip()
            if stripped.startswith("WM_PROJECT_VERSION="):
                version = stripped.split("=", 1)[1].strip().strip('"\'')
                break
    if version:
        lines.append(
            f"  OpenFOAM version: {version} (OpenFOAM Foundation, installed at "
            f"{openfoam_path}). Use THIS version. Do not name any other."
        )

    starter_dir = str(understanding.get("starter_dir") or "").strip()
    base_case_path = str(understanding.get("base_case_path") or "").strip()
    case_dir = Path(starter_dir) / base_case_path if starter_dir and base_case_path else None
    if case_dir and case_dir.is_dir():
        lines.append(f"  existing case directory: {case_dir}")
        control = case_dir / "system" / "controlDict"
        if control.is_file():
            for line in control.read_text(errors="ignore").splitlines():
                parts = line.strip().rstrip(";").split()
                if len(parts) == 2 and parts[0] == "application":
                    lines.append(
                        f"  solver: {parts[1]} (already configured in this case's "
                        "system/controlDict). Do not clone or compile a different solver."
                    )
                    break
    elif understanding.get("base_case_path"):
        lines.append(f"  base case: {understanding['base_case_path']}")
    for key in ("Re", "nu", "Ub", "dimension"):
        if flow.get(key) is not None:
            lines.append(f"  {key}: {flow[key]}")
    if flow.get("geometry"):
        lines.append(f"  geometry/mesh/BCs/solver: {str(flow['geometry'])[:900]}")
    if include_scoring:
        # How a finished run is *scored* belongs to the search layer
        # (oed_setup_search and its bound comparators), not to a FoamAgent case
        # requirement, whose job is to write and run the case. Handing it to the
        # requirement VALIDATOR turned it into a checklist — every requirement
        # was rejected for "does not require running compare_exactmatch_cf.py",
        # "success criterion missing", "wallShearStress extraction not stated" —
        # none of which a case requirement should carry.
        if reference.get("quantities"):
            lines.append(f"  reference data available: {', '.join(map(str, reference['quantities']))}")
        if reference.get("usage_guidance"):
            lines.append(f"  how results are compared: {str(reference['usage_guidance'])[:400]}")
    spec = str(understanding.get("formula_or_model_spec") or "")
    if spec and "NO EXPLICIT FORMULA" not in spec.upper():
        lines.append(f"  model spec to implement: {spec[:600]}")
    return "\n".join(lines)


def _with_progress(fn):
    """Print a start/finish line around a tool call, in real time.

    ``graph.stream(..., stream_mode="updates")`` only reports a node *after*
    it finishes — silent for the entire duration of anything long (a case run
    can take hours), which is the opposite of useful. This prints the moment
    the call actually starts and the moment it actually ends, from inside the
    real function call, so it's honest progress, not a fake spinner.

    Uses ``functools.wraps`` (which sets ``__wrapped__``) so
    ``inspect.signature``/docstring introspection — what deepagents uses to
    build each tool's schema — still sees through to the original function.
    """
    import functools
    import time as _time

    from cfd_langgraph.cli.activity import BOARD

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        label = fn.__name__
        arg_preview = ", ".join(
            [repr(a)[:60] for a in args] + [f"{k}={v!r:.60}" for k, v in kwargs.items()]
        )
        print(f"\n▶ {label}({arg_preview[:200]}) ...", flush=True)
        # Registering the span here, around the real call, is what lets the
        # CLI's status line stay honest: it animates only while a tool is
        # genuinely in flight, and its elapsed time is this call's own.
        token = BOARD.start(label, arg_preview[:80])
        t0 = _time.monotonic()
        try:
            result = fn(*args, **kwargs)
        except Exception as exc:
            BOARD.finish(token, ok=False)
            print(f"✗ {label} failed after {_time.monotonic() - t0:.1f}s: {exc}", flush=True)
            raise
        BOARD.finish(token, ok=True)
        print(f"✓ {label} done in {_time.monotonic() - t0:.1f}s", flush=True)
        return result

    return wrapper


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str))


def _read_json(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _default_subprocess_env() -> Dict[str, str]:
    """Base environment for every scripts/*.py subprocess call: this
    process's environment, plus ``src/`` on PYTHONPATH.

    ``sys.path.insert(...)`` in ``cfd_cli.py`` only affects *this* Python
    process — it does not propagate to PYTHONPATH for children, so a bare
    ``subprocess.run(..., env=None)`` gives a child process no way to import
    ``cfd_langgraph`` unless the target script bootstraps its own sys.path
    (most of scripts/*.py do; not all — e.g. starter_understand.py doesn't).
    Fixing it here, once, covers every caller regardless of which scripts
    remember to bootstrap themselves.
    """
    env = os.environ.copy()
    src_dir = str(_REPO_ROOT / "src")
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{src_dir}:{existing}" if existing else src_dir
    return env


def _run_script(
    args: List[str], *, cwd: Path = _REPO_ROOT, timeout: int = 1800, env: Optional[Dict[str, str]] = None
) -> subprocess.CompletedProcess:
    """Shell out to one of the existing tested scripts/*.py CLIs.

    Every stage below that already has a working, tested script (literature,
    viz, interpret, analyze, paper_unified) is wired this way rather than
    reimplemented in-process — same reasoning as ``run_audit_and_record``:
    reuse the exact tested code path instead of risking a subtly different
    reimplementation. ``env=None`` (the default) uses ``_default_subprocess_env()``;
    pass an explicit dict (see ``_foamagent_env``) when the subprocess needs
    OpenFOAM sourced or a
    further translated/extended environment.
    """
    try:
        return subprocess.run(
            [sys.executable, *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env if env is not None else _default_subprocess_env(),
        )
    except subprocess.TimeoutExpired as exc:
        # Surface a timeout as a failed CompletedProcess, never as a raised
        # exception. Every caller already branches on ``returncode``; letting
        # this propagate would tear down the whole study turn and drop the
        # session to idle over one slow script.
        def _tail(stream) -> str:
            if stream is None:
                return ""
            if isinstance(stream, bytes):
                return stream.decode("utf-8", errors="replace")
            return str(stream)

        return subprocess.CompletedProcess(
            args=[sys.executable, *args],
            returncode=124,
            stdout=_tail(exc.stdout),
            stderr=(
                _tail(exc.stderr)
                + f"\n[timeout] {Path(args[0]).name} exceeded {timeout}s and was killed."
            ),
        )



def build_manager_tools(settings: Settings, out_dir: Path) -> Dict[str, Any]:
    """Build the manager's tools and the case-runner subagent's tools for one study.

    Split in two because they go to different places in ``deep_agent.py``:
    ``manager_tools`` go straight on the top-level manager; ``case_runner_tools``
    go only on the case-runner subagent, so a case's FoamAgent output stays in
    that subagent's isolated context instead of flooding the manager's.

    Both halves close over the same ``CaseCoordinator``, so however many cases
    the manager fans out to the case-runner at once, the real concurrency cap
    (computed from a calibration run, see ``scheduling/``) is enforced in one
    shared place — not left to the model to self-limit.
    """
    out_dir = Path(out_dir)
    coordinator = CaseCoordinator(
        safety_margin=settings.resource_safety_margin,
        forced_max_concurrency=settings.max_parallel_cases,
    )
    bundle = KnowledgeBundle(settings.knowledge_bundle_dir)
    foam_llm = create_langchain_llm(model=settings.model, temperature=0.15)
    state_lock = threading.Lock()

    def _now() -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def _update_study_state(
        *, topic: str = "", mode: str = "", current_stage: str = "", status: str = "running"
    ) -> Dict[str, Any]:
        """Atomically maintain the artifact state expected by the final audit."""
        with state_lock:
            state_path = out_dir / "state.json"
            state = _read_json(state_path) or {}
            if not isinstance(state, dict):
                state = {}
            state.setdefault("run_id", out_dir.name)
            state.setdefault("started_at", _now())
            if topic:
                state["topic"] = topic
            state.setdefault("topic", "")
            if mode:
                state["mode"] = mode
            state.setdefault("mode", "research")
            if current_stage:
                state["current_stage"] = current_stage
            state.setdefault("current_stage", "routing")
            state["status"] = status
            _write_json(state_path, state)
            return state

    def _write_checkpoint(name: str, payload: Optional[Dict[str, Any]] = None) -> Path:
        """Write a checkpoint only from the tool that completed that stage."""
        path = out_dir / "checkpoints" / f"{name}.json"
        data = {"stage": name.removesuffix("_done"), "ts": _now(), **(payload or {})}
        with state_lock:
            _write_json(path, data)
        return path

    def _ensure_routing(topic: str, mode: str = "research") -> None:
        _update_study_state(topic=topic, mode=mode, current_stage="routing")
        cp = out_dir / "checkpoints" / "routing_done.json"
        if not cp.is_file():
            _write_checkpoint("routing_done", {"mode": mode, "topic": topic})

    def _metric_specs(disc_dir: Path) -> List[Dict[str, Any]]:
        raw = _read_json(disc_dir / "metric_specs.json") or []
        if isinstance(raw, dict):
            raw = raw.get("metrics") or []
        return [m for m in raw if isinstance(m, dict) and m.get("name")]

    def _select_primary_metric(
        metrics: Dict[str, Any], specs: List[Dict[str, Any]]
    ) -> tuple[str, Optional[float], str]:
        by_name = {str(s.get("name")): s for s in specs}
        ordered_names = [str(s.get("name")) for s in specs] + list(metrics.keys())
        for name in ordered_names:
            if name not in metrics:
                continue
            try:
                value = float(metrics[name])
            except Exception:
                continue
            if not math.isfinite(value):
                continue
            direction = str((by_name.get(name) or {}).get("direction", "min")).lower()
            if direction not in {"min", "max"}:
                direction = "min"
            return name, value, direction
        return "", None, "min"

    def _copy_case_inputs(src: Path, dst: Path) -> None:
        """Copy only reusable case inputs, never stale solver outputs/logs."""
        dst.mkdir(parents=True, exist_ok=True)
        for name in ("0", "constant", "system", "customModels"):
            item = src / name
            if item.is_dir():
                shutil.copytree(item, dst / name, dirs_exist_ok=True)
        for name in ("Allrun", "Allclean"):
            item = src / name
            if item.is_file():
                shutil.copy2(item, dst / name)

    def _copy_completed_case(src: Path, dst: Path) -> None:
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)

    def _selected_turbulence_model(case_dir: Path) -> str:
        """The model name the case's dictionaries actually select, or "".

        Both spellings, because OpenFOAM changed them: `RASModel` in the older
        `constant/turbulenceProperties`, `model` in OF8+'s
        `constant/momentumTransport`. A case ported from one to the other can
        carry both files, so the one naming a non-stock model wins.
        """
        for name in ("momentumTransport", "turbulenceProperties"):
            path = case_dir / "constant" / name
            if not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            match = re.search(
                r"(?m)^\s*(?:model|RASModel|LESModel)\s+([A-Za-z_][A-Za-z0-9_]*)\s*;", text
            )
            if match:
                return match.group(1)
        return ""

    def _solver_logs(case_dir: Path) -> List[Path]:
        """The case's solver log(s), newest first.

        `application` in system/controlDict names the log when it is there, but
        it is optional in OpenFOAM and real benchmark cases omit it: not one of
        the 32 closure-challenge controlDicts declares it, because each case is
        launched by name. Requiring it made this function answer "no clean log"
        for every case in that study no matter how cleanly it had run, which
        marks a promoted case failed and a converged candidate unbuilt.

        So fall back to the log.<something> files actually present. Their
        OpenFOAM banner is what identifies them as solver output, checked by
        the caller, so a stray log.foamJob or log.blockMesh cannot pass by
        virtue of its name alone.
        """
        control = case_dir / "system" / "controlDict"
        named: List[Path] = []
        if control.is_file():
            text = control.read_text(encoding="utf-8", errors="ignore")
            match = re.search(r"(?m)^\s*application\s+([^;\s]+)\s*;", text)
            if match:
                candidate = case_dir / f"log.{match.group(1)}"
                if candidate.is_file():
                    named.append(candidate)
        try:
            found = [p for p in case_dir.glob("log.*") if p.is_file() and p not in named]
        except OSError:
            found = []
        found.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return named + found

    def _case_has_clean_solver_log(case_dir: Path) -> bool:
        for log in _solver_logs(case_dir):
            try:
                with log.open("r", encoding="utf-8", errors="ignore") as handle:
                    head = handle.read(8000)
                    if "OpenFOAM" not in head:
                        continue
                    # Seek the tail rather than reading the file: a converged
                    # 30000-step log in this study is 30 MB, and this runs once
                    # per case over 32 cases.
                    size = log.stat().st_size
                    handle.seek(max(0, size - 8000))
                    tail = handle.read()
            except OSError:
                continue
            if (
                "ExecutionTime" in tail
                and re.search(r"(?m)^\s*End\s*$", tail)
                and "FOAM FATAL" not in tail
            ):
                return True
        return False

    def _user_prompt() -> str:
        """The user's own prompt for this study, verbatim, or "".

        Written once by the CLI at run start. Every OED tool takes its topic
        from here rather than from the ``topic`` argument, because the manager
        summarises: a real run passed an 86-character paraphrase where the
        prompt was 775 characters, discarding the scoring contract and the
        "by at least 10%" target. Comparators were then authored with no
        normalisation guidance (cf_rmse came out 2.4x wrong) and the success
        threshold silently became 0%, so any candidate that was not strictly
        worse counted as a win. Instructing the manager to pass it verbatim is
        not enough — this is the objective, so it is read from disk.
        """
        try:
            return (out_dir / "user_prompt.txt").read_text(encoding="utf-8").strip()
        except Exception:
            return ""

    def _effective_topic(topic: str) -> str:
        """The study's objective: the user's prompt when we have it."""
        return _user_prompt() or str(topic or "")

    def _starter_base_case_dir() -> Optional[Path]:
        """The starter's base case, if this study was given one.

        Its controlDict carries the function objects that make the case
        measurable (wallShearStress for Cf, yPlus, sampled sets). Generated
        cases inherit that block verbatim rather than relying on the case
        writer to infer specific function-object entries from prose.
        """
        su = _read_json(out_dir / "starter_understanding.json") or {}
        starter_dir = str(su.get("starter_dir") or "").strip()
        base_case = str(su.get("base_case_path") or "").strip()
        if not starter_dir or not base_case:
            return None
        candidate = Path(starter_dir) / base_case
        return candidate if (candidate / "system" / "controlDict").is_file() else None

    def _safe_case_path(case_id: str, folder_name: str = "", file_name: str = "") -> tuple[Path, Optional[str]]:
        case_dir = out_dir / "cases" / _safe_variant_slug(case_id, "case")
        rel = Path(folder_name) / file_name if file_name else Path(folder_name)
        if rel.is_absolute() or ".." in rel.parts:
            return case_dir, "Case-relative folder/file path may not be absolute or contain '..'."
        target = (case_dir / rel).resolve()
        try:
            target.relative_to(case_dir.resolve())
        except ValueError:
            return case_dir, "Resolved case path escapes the case directory."
        # Same protected-artifact rule the general write tools enforce. Without
        # this, foam_write_case_file was a way around it: run_result.json and
        # decision.json live inside a case directory and are exactly what
        # interpret_case and analyze_all_cases treat as ground truth, so a
        # subagent could write its own successful run record for a case that
        # never ran.
        if _is_protected_artifact(target):
            return case_dir, (
                f"Protected workflow artifact may only be written by its owning tool: {target.name}"
            )
        return target, None

    def _canonical_requirement(case_id: str, requirement_text: str) -> tuple[str, Optional[str]]:
        requirements = _read_json(out_dir / "requirements.json") or []
        if not isinstance(requirements, list):
            return "", "requirements.json is missing or invalid."
        matches = [
            item for item in requirements
            if isinstance(item, dict) and str(item.get("case_id", "")) == case_id
        ]
        if len(matches) != 1:
            return "", f"case_id {case_id!r} is not uniquely defined in requirements.json."
        canonical = str(matches[0].get("user_requirement_text", "") or "").strip()
        if not canonical or canonical != str(requirement_text or "").strip():
            return "", "requirement_text does not match the approved requirements.json entry."
        return canonical, None

    def _writable_path(path: str) -> tuple[Optional[Path], Optional[str]]:
        p = Path(path).expanduser()
        if not p.is_absolute():
            p = (_REPO_ROOT / p).resolve()
        else:
            p = p.resolve()
        try:
            rel = p.relative_to(out_dir.resolve())
        except ValueError:
            return None, f"Writes are restricted to this study output directory: {out_dir.resolve()}"
        if _is_protected_artifact(p, rel.parts):
            return None, f"Protected workflow artifact may only be written by its owning tool: {p}"
        return p, None

    # -----------------------------------------------------------------
    # General filesystem/shell access — real disk, real subprocess, no
    # deepagents virtual filesystem involved. deepagents does ship built-in
    # ls/read_file/glob/grep/execute tools, but only against whatever
    # `backend=` is configured (we don't pass one, so they'd hit an in-memory
    # StateBackend, not the real disk), and `execute` needs a
    # SandboxBackendProtocol backend we don't have. These are plain Python
    # instead, exactly like every other tool in this file, so "the starter
    # folder is at <arbitrary real path>" actually works — including paths
    # outside this repo.
    # -----------------------------------------------------------------

    def list_directory(path: str, max_entries: int = 300) -> dict:
        """List one directory's immediate contents (name, is_dir, size_bytes).
        ``path`` can be any real path — relative to the repo root, or absolute."""
        p = Path(path).expanduser()
        if not p.is_absolute():
            p = (_REPO_ROOT / p).resolve()
        if not p.is_dir():
            return {"error": f"Not a directory: {p}"}
        entries = []
        for child in sorted(p.iterdir())[:max_entries]:
            try:
                entries.append({"name": child.name, "is_dir": child.is_dir(), "size_bytes": child.stat().st_size if child.is_file() else None})
            except OSError:
                continue
        return {"path": str(p), "entries": entries, "truncated": len(list(p.iterdir())) > max_entries}

    def directory_tree(path: str = ".", max_depth: int = 3, max_entries: int = 500) -> dict:
        """Recursive tree view of a folder — the same kind of picture you'd
        get from `tree`, so you can see structure at a glance instead of
        listing one level at a time. Any real path, any depth you ask for."""
        p = Path(path).expanduser()
        if not p.is_absolute():
            p = (_REPO_ROOT / p).resolve()
        if not p.is_dir():
            return {"error": f"Not a directory: {p}"}

        lines = [str(p)]
        count = 0

        def walk(d: Path, prefix: str, depth: int) -> None:
            nonlocal count
            if depth > max_depth or count >= max_entries:
                return
            try:
                entries = sorted(d.iterdir(), key=lambda x: (not x.is_dir(), x.name))
            except OSError:
                return
            for i, entry in enumerate(entries):
                if count >= max_entries:
                    lines.append(f"{prefix}... (truncated at {max_entries} entries)")
                    return
                last = i == len(entries) - 1
                lines.append(f"{prefix}{'└── ' if last else '├── '}{entry.name}{'/' if entry.is_dir() else ''}")
                count += 1
                if entry.is_dir():
                    walk(entry, prefix + ("    " if last else "│   "), depth + 1)

        walk(p, "", 1)
        return {"path": str(p), "tree": "\n".join(lines), "truncated": count >= max_entries}

    def make_directory(path: str) -> dict:
        """Create a directory (and missing parents) inside this study's output directory."""
        p, error = _writable_path(path)
        if error or p is None:
            return {"error": error}
        p.mkdir(parents=True, exist_ok=True)
        return {"path": str(p)}

    def find_files(root: str, pattern: str = "*", max_results: int = 200) -> dict:
        """Recursively glob for files under ``root`` matching ``pattern``
        (e.g. "*.csv", "**/system/controlDict").

        ``root`` must resolve inside the repository, and the walk stops at
        ``max_results`` rather than materialising the whole tree first —
        an unbounded rglob over a home directory has no timeout to save it.
        """
        p = Path(root).expanduser()
        if not p.is_absolute():
            p = (_REPO_ROOT / p).resolve()
        p = p.resolve()
        if not p.is_dir():
            return {"error": f"Not a directory: {p}"}
        if p != _REPO_ROOT and _REPO_ROOT not in p.parents:
            return {
                "error": (
                    f"Refusing to search outside the repository: {p}. "
                    f"Pass a root under {_REPO_ROOT}."
                ),
                "repo_root": str(_REPO_ROOT),
            }
        matches = []
        truncated = False
        try:
            for m in p.rglob(pattern):
                try:
                    if not m.is_file():
                        continue
                except OSError:
                    continue
                matches.append(str(m))
                if len(matches) >= max_results:
                    truncated = True
                    break
        except Exception as exc:
            return {"root": str(p), "pattern": pattern, "matches": sorted(matches),
                    "error": f"{type(exc).__name__}: {exc}"}
        return {"root": str(p), "pattern": pattern, "matches": sorted(matches),
                "truncated": truncated}

    def read_text_file(path: str, max_chars: int = 20000) -> dict:
        """Read a text file (source, config, reference data, notes). Any real path."""
        p = Path(path).expanduser()
        if not p.is_absolute():
            p = (_REPO_ROOT / p).resolve()
        if not p.is_file():
            return {"error": f"Not a file: {p}"}
        text = p.read_text(encoding="utf-8", errors="ignore")
        truncated = len(text) > max_chars
        return {"path": str(p), "content": text[:max_chars], "truncated": truncated, "total_chars": len(text)}

    def write_text_file(path: str, content: str) -> dict:
        """Write a non-protected text file inside this study's output directory.
        Authoritative workflow artifacts must be written by their owning tools."""
        p, error = _writable_path(path)
        if error or p is None:
            return {"error": error}
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return {"path": str(p), "bytes_written": len(content.encode("utf-8"))}

    def edit_text_file(path: str, old_string: str, new_string: str, replace_all: bool = False) -> dict:
        """Exact replacement in a non-protected file inside this study's output directory. Fails if
        ``old_string`` isn't found, or (unless replace_all) isn't unique —
        same contract as Claude Code's own Edit tool, so callers can't
        silently clobber the wrong occurrence."""
        p, error = _writable_path(path)
        if error or p is None:
            return {"error": error}
        if not p.is_file():
            return {"error": f"Not a file: {p}"}
        text = p.read_text(encoding="utf-8")
        count = text.count(old_string)
        if count == 0:
            return {"error": "old_string not found in file"}
        if count > 1 and not replace_all:
            return {"error": f"old_string is not unique ({count} occurrences); pass replace_all=true or a more specific old_string"}
        new_text = text.replace(old_string, new_string) if replace_all else text.replace(old_string, new_string, 1)
        p.write_text(new_text, encoding="utf-8")
        return {"path": str(p), "replacements": count if replace_all else 1}

    def grep_files(pattern: str, root: str = ".", glob: str = "*", max_results: int = 200) -> dict:
        """Search file contents for a regex pattern under ``root``, filtered
        to files matching ``glob`` (e.g. "*.py", "*.log", "log.*").

        ``root`` must resolve inside the repository. A search that scans a
        whole home directory is never what this tool is for, and one that
        times out must come back as a tool *result* the model can react to
        — never as a raised exception, which would tear down the whole
        study turn over a failed grep.
        """
        p = Path(root).expanduser()
        if not p.is_absolute():
            p = (_REPO_ROOT / p).resolve()
        p = p.resolve()
        if not p.is_dir():
            return {"error": f"Not a directory: {p}"}
        if p != _REPO_ROOT and _REPO_ROOT not in p.parents:
            return {
                "error": (
                    f"Refusing to search outside the repository: {p}. "
                    f"Pass a root under {_REPO_ROOT} — the run directory, "
                    "the starter case, or scripts/."
                ),
                "repo_root": str(_REPO_ROOT),
            }
        try:
            proc = subprocess.run(
                ["grep", "-rn", "--include", glob, "-E", pattern, str(p)],
                capture_output=True, text=True, timeout=120,
            )
        except subprocess.TimeoutExpired:
            return {
                "root": str(p),
                "pattern": pattern,
                "matches": [],
                "match_count": 0,
                "error": (
                    f"grep timed out after 120s under {p} with glob {glob!r}. "
                    "Narrow the search: use a deeper root and a specific glob "
                    "(e.g. '*.json', '*.py') instead of '*'."
                ),
            }
        except Exception as exc:
            return {"root": str(p), "pattern": pattern, "matches": [], "match_count": 0,
                    "error": f"{type(exc).__name__}: {exc}"}
        lines = (proc.stdout or "").splitlines()[:max_results]
        return {"root": str(p), "pattern": pattern, "matches": lines, "match_count": len(lines)}

    def read_starter_folder(starter_dir: str, topic: str = "") -> dict:
        """Characterize a starter case folder — geometry/BC/solver setup,
        available reference data (DNS/experimental) and how to compare
        against it, flow parameters to treat as authoritative. Writes
        starter_understanding.json, which generate_case_requirements and
        write_paper both pick up automatically if present. Use this the
        moment a user points at a starter/base-case path, before anything
        else tries to build requirements against it.
        """
        p = Path(starter_dir).expanduser()
        if not p.is_absolute():
            p = (_REPO_ROOT / p).resolve()
        if not p.is_dir():
            return {"error": f"Not a directory: {p}"}
        out_path = out_dir / "starter_understanding.json"
        proc = _run_script(
            ["scripts/starter_understand.py", "--starter-dir", str(p), "--topic", topic, "--output", str(out_path)],
            timeout=900,
        )
        result = _read_json(out_path) or {}
        return {
            "ok": proc.returncode == 0,
            "status": result.get("status"),
            "path": str(out_path),
            "flow_parameters": (result.get("flow_parameters") or {}) if isinstance(result, dict) else {},
            "stderr_tail": proc.stderr[-1000:] if proc.returncode != 0 else "",
        }

    def fetch_literature(topic: str, limit: int = 20) -> dict:
        """Search Semantic Scholar for papers on ``topic`` and write lit.json —
        the standard artifact ``write_paper`` reads for the paper's literature
        section. Call this once, near the start of a study."""
        topic = str(topic or "").strip()
        if not topic:
            return {"ok": False, "error": "Literature topic is empty."}
        # A floor, not just a ceiling. Two papers is not a literature review,
        # and ideation grounded on two papers invents its own direction —
        # measured on run closure_20260824_codex, where a two-paper lit.json
        # (one of them about aircraft intake S-ducts) preceded three straight
        # rounds of every hypothesis being rejected as under-specified.
        requested = max(1, min(int(limit), int(settings.ideation_max_papers)))
        limit = max(requested, _LIT_MIN_PAPERS) if requested < _LIT_MIN_PAPERS else requested
        _ensure_routing(topic, "research")
        lit_path = out_dir / "lit.json"

        # Search on the research question, record the objective. The topic is
        # what propose_and_rank_hypotheses matches against, so distilling the
        # QUERY leaves that guard working exactly as before.
        query, broader = _literature_query(topic, foam_llm)
        attempts: List[Dict[str, Any]] = []
        proc = None
        paper_count = 0
        papers = None
        for label, search in (("distilled", query), ("broader", broader)):
            if not search:
                continue
            if attempts and search == attempts[-1]["query"]:
                continue
            proc = _run_script(
                ["scripts/lit.py", "--topic", search, "--limit", str(limit), "--output", str(lit_path)]
            )
            found = _read_json(lit_path) if (proc.returncode == 0 and lit_path.exists()) else None
            count = len(found) if isinstance(found, list) else 0
            attempts.append({"stage": label, "query": search, "papers": count})
            print(f"[lit] {label} query {search!r} -> {count} paper(s)", flush=True)
            if count > paper_count:
                paper_count, papers = count, found
            if count >= _LIT_MIN_PAPERS:
                break

        ok = bool(proc is not None and proc.returncode == 0 and paper_count > 0)
        if ok:
            _write_checkpoint(
                "literature_done",
                {"path": str(lit_path), "paper_count": paper_count, "source": "semantic_scholar",
                 "topic": topic, "search_attempts": attempts},
            )
            _update_study_state(topic=topic, current_stage="literature")
        return {
            "ok": ok,
            "paper_count": paper_count,
            "path": str(lit_path),
            "search_attempts": attempts,
            "thin_literature": (
                f"Only {paper_count} paper(s) found. Ideation grounded on this little "
                "tends to invent its own direction and fail critique; consider a "
                "different phrasing of the research question before proceeding."
                if 0 < paper_count < _LIT_MIN_PAPERS else ""
            ),
            "stderr_tail": (
                proc.stderr[-1000:] if proc.returncode != 0
                else "Literature search returned no papers; hypothesis generation is blocked."
                if paper_count == 0 else ""
            ),
        }

    def propose_and_rank_hypotheses(topic: str, num_candidates: int = 6) -> dict:
        """Propose candidate hypotheses for ``topic``, critique and rank them,
        and write the ranked list to disk.

        Safe to call any time — it only produces a list for a human to
        review. Nothing downstream (case specs, experiments) happens from
        this call alone; that requires ``advance_with_approved_hypotheses``.
        """
        _ensure_routing(topic, "research")
        # Hypotheses already approved for this study: refuse to re-propose.
        # Without this, ANY later failure (a mesh gate that won't converge, a
        # comparator that won't bind) lets the manager "recover" by rewinding
        # to the first stage — re-running the whole propose/critique/rank
        # cycle, re-firing the human approval gate, and discarding an hour of
        # downstream work. A downstream failure is never evidence that the
        # approved hypotheses were wrong; it has to be fixed where it happened.
        already_approved = _read_json(out_dir / "hypotheses_approved.json") or {}
        approved_list = already_approved.get("approved_hypotheses") or []
        if approved_list:
            return {
                "already_approved": True,
                "approved_count": len(approved_list),
                "approved_candidate_ids": [
                    str(h.get("candidate_id", "")) for h in approved_list if isinstance(h, dict)
                ],
                "path": str(out_dir / "hypotheses_approved.json"),
                "note": (
                    "This study already has approved hypotheses, so they were NOT regenerated. "
                    "Re-proposing would restart the study and throw away the requirements, mesh "
                    "gate and search state built on them. If a later stage is failing, fix that "
                    "stage — do not return here."
                ),
            }
        literature = _read_json(out_dir / "lit.json")
        if not isinstance(literature, list) or not literature:
            return {
                "error": "Blocked: lit.json is missing or empty; hypotheses cannot be literature-grounded.",
                "path": str(out_dir / "lit.json"),
            }
        literature_checkpoint = _read_json(out_dir / "checkpoints" / "literature_done.json") or {}
        if str(literature_checkpoint.get("topic", "")).strip() != str(topic or "").strip():
            return {
                "error": "Hypothesis topic does not match the topic used to retrieve lit.json; refetch literature first.",
                "literature_topic": literature_checkpoint.get("topic", ""),
                "hypothesis_topic": topic,
            }
        result = run_propose_critique_rank(
            settings,
            topic,
            num_candidates=max(1, min(num_candidates, settings.hypothesis_num_candidates)),
            literature_records=literature,
            require_literature=True,
            case_context=_starter_case_context(out_dir, settings.openfoam_path),
        )
        _write_json(out_dir / "hypotheses_ranked.json", result)
        _update_study_state(topic=topic, current_stage="hypothesis")
        top = [
            {
                "candidate_id": c["candidate_id"],
                "objective": (c.get("idea", {}) or {}).get("objective", ""),
                "rank_rationale": c.get("rank_rationale", ""),
            }
            for c in result["ranked_hypotheses"][:5]
        ]
        return {
            "num_proposed": result["num_proposed"],
            "num_passed_critique": result["num_passed_critique"],
            "top_ranked": top,
            "path": str(out_dir / "hypotheses_ranked.json"),
        }

    def advance_with_approved_hypotheses(approved_candidate_ids: List[str], notes: str = "") -> dict:
        """Lock in which ranked hypotheses actually become experiments.

        This is the tool the CLI's approval gate reviews before it runs.
        Propose the top-ranked candidate_ids here, but expect a human to
        edit this list (or reject it outright) before it executes — that is
        the interrupt point. Nothing in the requirements/experiments stage
        will run without a ``hypotheses_approved.json`` on disk (enforced in
        ``run_case`` below, not just by instruction).
        """
        ranked = _read_json(out_dir / "hypotheses_ranked.json") or {}
        by_id = {c["candidate_id"]: c for c in ranked.get("ranked_hypotheses", [])}
        unknown = [cid for cid in approved_candidate_ids if cid not in by_id]
        approved = [by_id[cid] for cid in approved_candidate_ids if cid in by_id]
        if unknown or not approved:
            return {
                "error": "Approval rejected: choose at least one valid ranked candidate ID.",
                "unknown_candidate_ids": unknown,
                "valid_candidate_ids": list(by_id),
            }
        record = {
            "approved_candidate_ids": approved_candidate_ids,
            "approved_hypotheses": approved,
            "notes": notes,
        }
        _write_json(out_dir / "hypotheses_approved.json", record)
        ranked["human_review"] = {
            "status": "approved",
            "decision": approved_candidate_ids,
            "notes": notes,
        }
        _write_json(out_dir / "hypotheses_ranked.json", ranked)
        _write_checkpoint(
            "hypothesis_done",
            {"approved_candidate_ids": approved_candidate_ids, "approved_count": len(approved)},
        )
        _update_study_state(current_stage="hypothesis")
        return {"approved_count": len(approved), "path": str(out_dir / "hypotheses_approved.json")}

    def generate_case_requirements() -> dict:
        """Turn every approved hypothesis's experiments into validated,
        executable FoamAgent requirements and write requirements.json (same
        schema ``scripts/requirements.py`` produces: case_id,
        user_requirement_text, experiment_id, description, study_id).

        Uses ``HypothesisAgent.generate_validated_requirement`` — the existing,
        tested LLM validate/repair loop — one experiment at a time, rather
        than the plain-template synthesis ``scripts/requirements.py`` falls
        back to when no requirement text is already present.
        """
        approved = _read_json(out_dir / "hypotheses_approved.json")
        if not approved:
            return {"error": "Blocked: hypotheses_approved.json is missing — approve hypotheses first."}
        # Same rewind hazard as propose_and_rank_hypotheses, and worse here:
        # regenerating requirements changes user_requirement_text, which
        # run_mesh_gate matches byte for byte, so a re-derived set silently
        # invalidates an already-converged mesh gate.
        existing_reqs = _read_json(out_dir / "requirements.json")
        if isinstance(existing_reqs, list) and existing_reqs:
            return {
                "already_generated": True,
                "num_requirements": len(existing_reqs),
                "path": str(out_dir / "requirements.json"),
                "case_ids": [
                    str(r.get("case_id", "")) for r in existing_reqs if isinstance(r, dict)
                ],
                "note": (
                    "requirements.json already exists and was NOT regenerated. Read it and "
                    "use one of these entries verbatim. Regenerating would reword "
                    "user_requirement_text and invalidate any mesh gate already run against it."
                ),
            }
        ranked = _read_json(out_dir / "hypotheses_ranked.json") or {}
        topic = ranked.get("research_topic", "")

        from cfd_langgraph.agents.hypothesis_agent import HypothesisAgent
        from cfd_langgraph.prompts.loader import PromptLoader

        overlay_path = settings.knowledge_bundle_dir / "active_prompts.yaml"
        prompt_loader = PromptLoader(settings.prompts_path, overlay_path=overlay_path)
        agent = HypothesisAgent(settings.model, prompt_loader)

        requested_total = sum(
            len(((h.get("idea", {}) or {}).get("experiments", []) or []))
            for h in approved.get("approved_hypotheses", [])
            if isinstance(h, dict)
        )
        if requested_total > settings.workflow_max_experiments_total:
            return {
                "error": "Approved hypotheses exceed the configured total experiment limit.",
                "requested": requested_total,
                "limit": settings.workflow_max_experiments_total,
            }

        # One (index, idea, study_id, simulation) per experiment, built up front
        # so the generation below is a flat, independent work list.
        jobs: List[Dict[str, Any]] = []
        for h in approved.get("approved_hypotheses", []):
            idea = h.get("idea", {}) or {}
            study_id = idea.get("study_id") or h.get("candidate_id", "")
            for exp in idea.get("experiments", []) or []:
                index = len(jobs) + 1
                jobs.append(
                    {
                        "index": index,
                        "idea": idea,
                        "study_id": study_id,
                        "experiment": exp,
                        "simulation": {
                            "simulation_id": exp.get("experiment_id", f"exp_{index:03d}"),
                            "case_name": exp.get("name", ""),
                            "case_data": exp,
                            "parameter_value": exp.get("parameters", {}),
                            "description": exp.get("notes", ""),
                        },
                    }
                )

        def _one_requirement(job: Dict[str, Any]) -> Dict[str, Any]:
            experiment = job["experiment"]
            simulation = job["simulation"]
            started = time.monotonic()
            result = agent.generate_validated_requirement(
                idea=job["idea"], simulation=simulation, run_topic=topic,
                case_context=case_context,
                validator_context=validator_context,
            )
            print(
                f"  [requirements] {job['index']}/{len(jobs)} {simulation['simulation_id']} "
                f"({time.monotonic() - started:.0f}s)"
                + ("" if result.get("valid", False) else "  INVALID"),
                flush=True,
            )
            return {
                "case_id": f"case_{job['index']:03d}",
                "user_requirement_text": result["requirement"],
                "experiment_id": simulation["simulation_id"],
                "description": experiment.get("notes", "") or experiment.get("name", ""),
                "study_id": job["study_id"],
                "requirement_valid": result.get("valid", False),
            }

        # These calls are independent — each turns one experiment into one
        # requirement and reads nothing the others write. Serially they were a
        # silent ~50-minute block for 17 experiments, indistinguishable from a
        # hang, and one dropped connection lost the whole batch. Measured at
        # both low and high reasoning effort: the cost is the number of calls,
        # not the depth of any one of them.
        case_context = _starter_case_context(out_dir, settings.openfoam_path)
        # The validator judges whether the case is writable and runnable, so it
        # sees the physical setup only — not how the result will be scored.
        validator_context = _starter_case_context(
            out_dir, settings.openfoam_path, include_scoring=False
        )
        if not case_context:
            print("  [requirements] WARNING: no starter_understanding.json — "
                  "requirements will be written blind", flush=True)
        print(f"  [requirements] generating {len(jobs)} requirements concurrently...", flush=True)
        started_all = time.monotonic()
        out: List[Dict[str, Any]] = []
        if jobs:
            with ThreadPoolExecutor(max_workers=min(8, len(jobs))) as pool:
                # Ordered by case_id afterwards, not by completion: case_001..N
                # must stay stable, because run_mesh_gate matches
                # user_requirement_text byte for byte.
                out = sorted(pool.map(_one_requirement, jobs), key=lambda r: r["case_id"])
        print(
            f"  [requirements] {len(out)} generated in {time.monotonic() - started_all:.0f}s",
            flush=True,
        )
        # Publish what passed; exclude and report what did not.
        #
        # This used to be all-or-nothing, which made 16-of-17 indistinguishable
        # from 0-of-17: nothing was written, and the manager regenerated all 17
        # again — ten minutes a cycle, indefinitely. That outcome survives any
        # improvement to the generator, because these are N independent
        # non-deterministic verdicts in series and one will eventually drop.
        #
        # Excluding a requirement is safe in a way that publishing a bad one is
        # not: a case that is never written cannot run wrong. Nothing
        # downstream needs the full set — the mesh gate uses one representative
        # requirement, and an open-ended-discovery run replaces per-case
        # launches with the search loop entirely. Publishing zero, by contrast,
        # stops the study dead.
        valid = [r for r in out if r.get("requirement_valid")]
        invalid = [r for r in out if not r.get("requirement_valid")]
        draft_path = out_dir / "requirements_draft.json"
        if invalid:
            _write_json(draft_path, out)
        if not valid:
            return {
                "error": "Every generated requirement failed validation; nothing runnable was published.",
                "invalid_case_ids": [r["case_id"] for r in invalid],
                "draft_path": str(draft_path),
            }
        req_path = out_dir / "requirements.json"
        _write_json(req_path, valid)
        _write_checkpoint(
            "requirements_done",
            {"path": str(req_path), "num_requirements": len(valid), "num_excluded": len(invalid)},
        )
        _update_study_state(current_stage="requirements")
        result: Dict[str, Any] = {"num_requirements": len(valid), "path": str(req_path)}
        if invalid:
            print(
                f"  [requirements] published {len(valid)}/{len(out)}; excluded "
                f"{', '.join(r['case_id'] for r in invalid)}",
                flush=True,
            )
            result["excluded_case_ids"] = [r["case_id"] for r in invalid]
            result["excluded_reason"] = (
                "These failed validation twice and were left out of requirements.json. "
                "The published requirements are complete and runnable; do NOT regenerate "
                "requirements to recover them — regenerating rewords every "
                "user_requirement_text and invalidates any mesh gate already run."
            )
            result["draft_path"] = str(draft_path)
        return result

    def run_mesh_gate(
        physics_group: str,
        requirement_text: str,
        topic: str = "",
        metrics: Optional[List[str]] = None,
        max_refine_levels: int = 5,
    ) -> dict:
        """Sequential baseline -> refine -> analyze -> decide mesh-independence
        loop, ported from ``orchestrator_run.py``'s ``_run_mesh_gate_group_impl``
        (same convergence logic, via ``mesh_gate_groups.llm_mesh_gate_pair_convergence``
        / ``heuristic_mesh_gate_pair_fallback``): run a baseline case, then refine
        it in a chain (parent -> refined -> refined_2 -> ...), analyzing each
        parent/child pair with ``analyze.py --qoi-source llm_pyvista`` and asking
        an LLM whether the physically-trustworthy QoIs changed by more than ~5%.
        Stops at the first converged parent, or after ``max_refine_levels``.

        There is no formal Richardson-extrapolation/GCI number computed here —
        checked the real orchestrator implementation and it doesn't compute
        one either; "GCI" in this codebase means this iterative LLM-judged
        pairwise refinement, not a literal GCI formula.

        Not ported from the original: the LLM-driven multi-level experiment
        *planner* and metric *selector* (``_mesh_gate_plan_experiments`` /
        ``_llm_decide_analysis_metrics``) — this version takes a fixed metric
        list decided by reading the study (see _llm_mesh_gate_metrics) and always
        walks the baseline -> refined chain,
        rather than letting an LLM propose the level plan up front.
        """
        from cfd_langgraph.llm.factory import create_langchain_llm
        from cfd_langgraph.mesh_gate_groups import (
            heuristic_mesh_gate_pair_fallback,
            llm_mesh_gate_pair_convergence,
            merge_mesh_gate_metrics,
        )

        physics_group = _safe_variant_slug(physics_group, "default")
        # Already converged for this group: return the existing selection
        # instead of running the whole baseline+refinement chain again. A
        # mesh gate is hours of OpenFOAM, and without this the tool had no
        # memory at all — a manager that re-called it (after a rejected
        # requirement text, a resume, or simply losing track) would silently
        # redo, or spin retrying, work that was already finished and
        # checkpointed.
        existing_spec = _read_json(out_dir / "mesh_gate" / physics_group / "selected_mesh_spec.json") or {}
        if existing_spec.get("converged") and Path(str(existing_spec.get("selected_level", ""))).is_dir():
            return {
                **existing_spec,
                "physics_group": physics_group,
                "already_converged": True,
                "note": (
                    "This physics group's mesh gate already converged; returning the existing "
                    "selection rather than re-running it. Proceed to the next step."
                ),
            }

        requirements = _read_json(out_dir / "requirements.json") or []
        approved = [item for item in requirements if isinstance(item, dict)] if isinstance(requirements, list) else []
        approved_texts = {str(item.get("user_requirement_text", "") or "").strip() for item in approved}
        if not str(requirement_text or "").strip() or str(requirement_text).strip() not in approved_texts:
            # Say exactly how to satisfy this, or the model retries the same
            # paraphrase indefinitely — the text must match an approved entry
            # byte for byte, which is not guessable from a generic refusal.
            return {
                "error": (
                    "Mesh-gate requirement must be copied verbatim from an approved "
                    "requirements.json entry's user_requirement_text — not paraphrased, "
                    "summarised, or truncated. Read the file and pass one entry's text exactly."
                ),
                "physics_group": physics_group,
                "requirements_path": str(out_dir / "requirements.json"),
                "available_case_ids": [str(item.get("case_id", "")) for item in approved],
            }
        llm = create_langchain_llm(model=settings.model, temperature=0.0)
        # Caller-supplied metrics win; otherwise read the study rather than
        # falling back to a fixed list that has nothing to do with it.
        mesh_metrics = [str(m).strip() for m in (metrics or []) if str(m).strip()]
        metric_specs = _study_metrics(out_dir, foam_llm)
        if not mesh_metrics:
            mesh_metrics = [str(m.get("name", "")).strip() for m in metric_specs if m.get("name")]
        if not mesh_metrics:
            return {
                "error": (
                    "Cannot run a mesh gate without knowing which quantities to judge it on. "
                    "Pass metrics=[...] naming the quantities this study is scored on."
                ),
                "physics_group": physics_group,
            }
        mesh_dir = out_dir / "mesh_gate" / physics_group
        skip_keys = {"mesh_n_cells", "mesh_n_points", "pyvista_time_used", "Umag_mean", "Umag_max"}

        def _foam_marker(case_dir: Path) -> None:
            marker = case_dir / f"{case_dir.name}.foam"
            if not marker.exists():
                marker.parent.mkdir(parents=True, exist_ok=True)
                marker.touch()

        def _pair_qois(case_a: Path, case_b: Path, label: str) -> tuple[dict, dict, dict]:
            _foam_marker(case_a)
            _foam_marker(case_b)
            pair_out = mesh_dir / f"mesh_analysis_{label}.json"
            spec_path = out_dir / "study_metrics.json"
            proc = _run_script(
                [
                    "scripts/analyze.py",
                    "--cases", str(case_a), str(case_b),
                    "--metrics", ",".join(mesh_metrics),
                    "--output", str(pair_out),
                    "--qoi-source", "llm_pyvista",
                    *(["--metric-spec", str(spec_path)] if spec_path.is_file() else []),
                ],
                timeout=3600,
            )
            # Surface what the extractor said. This was discarded, so three
            # separate failures in a row — a doubled script path, a missing
            # metric, an argparse typo that stopped analyze.py starting at all
            # — each presented identically as "the extractor returned nothing",
            # with the real reason sitting unread in a captured pipe.
            if proc.returncode != 0:
                print(
                    f"[mesh-gate] analyze.py exited {proc.returncode} for {label}:\n"
                    f"{(proc.stderr or '')[-1500:]}",
                    flush=True,
                )
            else:
                for line in (proc.stderr or "").splitlines():
                    if "batch failed" in line or "Traceback" in line or "Error" in line:
                        print(f"[mesh-gate] analyze.py: {line[:300]}", flush=True)
            data = _read_json(pair_out) or {}
            m = data.get("metrics", []) if isinstance(data, dict) else []
            q_a = m[0].get("qoi", {}) if len(m) > 0 and isinstance(m[0], dict) else {}
            q_b = m[1].get("qoi", {}) if len(m) > 1 and isinstance(m[1], dict) else {}
            common = [
                k for k in q_a
                if k in q_b and k not in skip_keys
                and isinstance(q_a.get(k), (int, float)) and isinstance(q_b.get(k), (int, float))
                and math.isfinite(float(q_a[k])) and math.isfinite(float(q_b[k]))
            ]
            # A QoI the extractor could not compute comes back as 0.0 rather
            # than null (measured: y_plus, despite the case's own yPlus.dat
            # holding real data). Zero on BOTH meshes then scores as a
            # flawless "0.00% change" and votes for convergence while
            # carrying no information at all — the mesh-gate equivalent of
            # declaring a winner by comparing a number to itself.
            uninformative = [
                k for k in common
                if abs(float(q_a[k])) < 1e-12 and abs(float(q_b[k])) < 1e-12
            ]
            common = [k for k in common if k not in uninformative]
            if uninformative:
                print(
                    f"[mesh-gate] ignoring QoIs that are zero on both meshes "
                    f"(no mesh-sensitivity information): {sorted(uninformative)}",
                    flush=True,
                )
            # A requested metric that never arrived is a hard stop, not a
            # footnote. The study names the quantity it is judged on; declaring
            # mesh independence in some *other* quantity produces a mesh that is
            # converged in something nobody asked about, and every candidate in
            # the search is then scored on it. Measured: a study specifying
            # "use only Cf" had Cf come back null and the gate judged on
            # centreline velocity instead, announcing it in one scrolling log
            # line.
            missing = [m for m in mesh_metrics if m not in common]
            if missing:
                available = sorted(set(q_a) & set(q_b))
                raise RuntimeError(
                    "mesh gate cannot judge convergence: the QoI extractor returned no usable "
                    f"value for the requested metric(s) {sorted(missing)}. Refusing to substitute "
                    "a different quantity — a mesh converged in the wrong metric is worse than no "
                    f"mesh gate. Extractor produced: {available[:25]}"
                    + (" ..." if len(available) > 25 else "")
                )
            pct = {k: abs(float(q_b[k]) - float(q_a[k])) / max(abs(float(q_a[k])), 1e-12) * 100.0 for k in common}
            return pct, q_a, q_b

        baseline_dir = mesh_dir / "baseline"
        baseline_result = coordinator.run_case(
            f"meshgate::{physics_group}",
            lambda: foam_native.run_foam_case(
                foam_llm, baseline_dir, requirement_text, openfoam_path=settings.openfoam_path,
                functions_seed_case_dir=_starter_base_case_dir(),
                # The gate's baseline level is the study's validated case,
                # run as-is. Nothing is authored, so nothing can be
                # mis-authored; the refinement chain then varies only the
                # mesh from here.
                base_case_seed_dir=_starter_base_case_dir(),
                seed_only=_starter_base_case_dir() is not None,
            ),
        )

        if not _run_succeeded(baseline_result):
            spec = {
                "physics_group": physics_group,
                "baseline_dir": str(baseline_dir),
                "baseline_success": False,
                "selected_level": "",
                "converged": False,
                "levels": [str(baseline_dir)],
                "refine_failed_at": "baseline",
                "metrics_used": mesh_metrics,
                "error": "Baseline mesh-gate case failed; no mesh was selected.",
            }
            _write_json(mesh_dir / "selected_mesh_spec.json", spec)
            return spec

        levels = [baseline_dir]
        refine_failed: Optional[str] = None
        converged = False
        selected = baseline_dir
        max_metric_attempts = 3
        case_solver = baseline_result.get("case_solver", "")

        for level in range(1, max(1, max_refine_levels) + 1):
            parent = levels[-1]
            ref_name = "refined" if level == 1 else f"refined_{level}"
            ref_dir = mesh_dir / ref_name
            refine_instruction = (
                f"Refine this blockMeshDict by roughly 10% in near-wall regions and 5% away from "
                f"the wall relative to its current resolution, keeping domain size, topology, and "
                f"patch names identical — this is a resolution change only."
            )
            # Copies parent's fields/BCs/transport/turbulence files unchanged and
            # edits only system/blockMeshDict — the base_case_dir mesh-copy-and-edit
            # capability scripts/foam_run.py --base-case-dir/--mesh-gate-role=refined
            # used to provide (that path is currently broken — a version mismatch
            # against the vendored Foam-Agent's LLMService — so this replaces it
            # rather than depending on it).
            ref_result = coordinator.run_case(
                f"meshgate::{physics_group}",
                lambda: foam_native.refine_mesh_from_parent(
                    foam_llm, ref_dir, parent, refine_instruction,
                    case_solver=case_solver, openfoam_path=settings.openfoam_path,
                ),
            )
            if ref_result.get("status") != "success":
                refine_failed = ref_name
                break
            levels.append(ref_dir)

            decision: Optional[dict] = None
            pct_changes: dict = {}
            for attempt in range(max_metric_attempts):
                pct_changes, q_a, q_b = _pair_qois(parent, ref_dir, f"{parent.name}_vs_{ref_name}")
                decision = llm_mesh_gate_pair_convergence(
                    llm,
                    parent_label=parent.name,
                    child_label=ref_name,
                    q_a=q_a,
                    q_b=q_b,
                    pct_changes=pct_changes,
                    metrics_requested=mesh_metrics,
                    topic_excerpt=topic or requirement_text,
                    requirement_excerpt=requirement_text,
                    metric_attempt_index=attempt,
                    max_metric_attempts=max_metric_attempts,
                )
                retry_metrics = decision.get("recommended_metrics_for_retry") or []
                if (
                    str(decision.get("qoi_reliability", "")).lower() == "unreliable"
                    and retry_metrics
                    and attempt < max_metric_attempts - 1
                ):
                    merged = merge_mesh_gate_metrics(mesh_metrics, retry_metrics, 10)
                    if merged != mesh_metrics:
                        mesh_metrics = merged
                        continue
                break
            if decision is None:
                decision = heuristic_mesh_gate_pair_fallback(q_a, q_b, pct_changes)

            # No informative QoI survived, so there is no evidence either way.
            # "Nothing changed" and "nothing was measured" are indistinguishable
            # from an empty comparison, and only one of them means converged.
            if not pct_changes:
                print(
                    f"[mesh-gate] {parent.name} vs {ref_name}: no comparable QoI was "
                    "extracted, so mesh independence cannot be established; refusing "
                    "to declare convergence on an empty comparison.",
                    flush=True,
                )
                decision = {
                    **decision,
                    "converged": False,
                    "reason": "No comparable QoI was extracted for this pair.",
                }

            if decision.get("converged"):
                selected = parent
                converged = True
                break

        if not converged:
            selected = levels[-1]

        spec = {
            "physics_group": physics_group,
            "baseline_dir": str(baseline_dir),
            "baseline_success": baseline_result.get("status") == "success",
            "selected_level": str(selected),
            "converged": converged,
            "levels": [str(p) for p in levels],
            "refine_failed_at": refine_failed,
            "metrics_used": mesh_metrics,
            "requirement_suffix": (
                "Use mesh-gate selected setup from the first stabilized mesh level for this physics "
                "group; keep the same topology and numerics as in the mesh study."
            ),
        }
        _write_json(mesh_dir / "selected_mesh_spec.json", spec)
        if converged:
            # Keep an aggregate top-level artifact for the audit/paper tools,
            # while retaining the per-physics-group source of truth.
            aggregate_path = out_dir / "selected_mesh_spec.json"
            aggregate = _read_json(aggregate_path) or {"groups": {}}
            if not isinstance(aggregate, dict):
                aggregate = {"groups": {}}
            groups = aggregate.setdefault("groups", {})
            if not isinstance(groups, dict):
                groups = {}
                aggregate["groups"] = groups
            groups[physics_group] = spec
            _write_json(aggregate_path, aggregate)
            mesh_context_path = out_dir / "mesh_independence_context.json"
            mesh_context = _read_json(mesh_context_path) or {"groups": {}}
            if not isinstance(mesh_context, dict):
                mesh_context = {"groups": {}}
            context_groups = mesh_context.setdefault("groups", {})
            if not isinstance(context_groups, dict):
                context_groups = {}
                mesh_context["groups"] = context_groups
            pair_analyses = []
            for analysis_file in sorted(mesh_dir.glob("mesh_analysis_*.json")):
                data = _read_json(analysis_file)
                if data:
                    pair_analyses.append({"path": str(analysis_file), "analysis": data})
            context_groups[physics_group] = {
                "selected_stable_name": selected.name,
                "selected_level": str(selected),
                "levels": [str(p) for p in levels],
                "metrics_used": mesh_metrics,
                "metrics_by_mesh_level": pair_analyses,
                "converged": True,
            }
            mesh_context["selected_stable_name"] = selected.name
            mesh_context["metrics_by_mesh_level"] = [
                item for group in context_groups.values() if isinstance(group, dict)
                for item in (group.get("metrics_by_mesh_level") or [])
            ]
            mesh_context.setdefault("mesh_figure_paths", [])
            _write_json(mesh_context_path, mesh_context)
            _write_checkpoint(
                "baseline_setup_done",
                {"physics_group": physics_group, "baseline_case_dir": str(baseline_dir)},
            )
            _write_checkpoint(
                "metric_setup_done",
                {"physics_group": physics_group, "metrics": mesh_metrics, "source": "mesh_gate_qoi_selection"},
            )
            _write_checkpoint(
                "mesh_gate_done",
                {"physics_group": physics_group, "selected_level": str(selected), "converged": True},
            )
            _update_study_state(current_stage="mesh_gate")
        return spec

    def interpret_case(case_id: str) -> dict:
        """Generate diagnostic figures and an interpreter decision for one
        finished case (PROCEED / REVISE / RERUN), via viz.py + interpret.py."""
        case_id = _safe_variant_slug(case_id, "case")
        case_dir = out_dir / "cases" / case_id
        run_result = _read_json(case_dir / "run_result.json") or {}
        if not _run_succeeded(run_result):
            return {"case_id": case_id, "error": "Case has no successful run_result.json; interpretation blocked."}
        figs_dir = case_dir / "figs"
        viz_proc = _run_script(
            ["scripts/viz.py", "--case", str(case_dir), "--mode", "interpret", "--output", str(figs_dir)],
            timeout=1200,
        )
        decision_path = case_dir / "decision.json"
        interp_proc = _run_script(
            ["scripts/interpret.py", "--case", str(case_dir), "--figs", str(figs_dir), "--output", str(decision_path)],
            timeout=1200,
        )
        decision = _read_json(decision_path) or {}
        if interp_proc.returncode == 0 and decision.get("status") in {"PROCEED", "REVISE", "RERUN"}:
            bridge_path = out_dir / "bridge.json"
            bridge = _read_json(bridge_path) or {}
            if isinstance(bridge, dict) and bridge.get("provenance") == "cfd-open-discovery:bridge:v1":
                decisions = bridge.setdefault("decisions", {})
                if isinstance(decisions, dict):
                    decisions[case_id] = decision
                    _write_json(bridge_path, bridge)
        return {
            "case_id": case_id,
            "viz_ok": viz_proc.returncode == 0,
            "interpret_ok": interp_proc.returncode == 0,
            "status": decision.get("status"),
            "reason": decision.get("reason"),
            "path": str(decision_path),
        }

    def analyze_all_cases(case_ids: List[str], metrics: str = "Cd,Cl,y_plus") -> dict:
        """Cross-case QoI comparison and discussion, across every finished case."""
        clean_ids = list(dict.fromkeys(_safe_variant_slug(cid, "case") for cid in case_ids))
        if not clean_ids:
            return {"error": "No case IDs supplied for analysis."}
        missing = []
        for cid in clean_ids:
            case_dir = out_dir / "cases" / cid
            rr = _read_json(case_dir / "run_result.json") or {}
            dj = _read_json(case_dir / "decision.json") or {}
            if not _run_succeeded(rr) or dj.get("status") != "PROCEED":
                missing.append(cid)
        if missing:
            return {
                "error": "Analysis accepts only successfully run cases whose interpreter decision is PROCEED.",
                "incomplete_or_rejected_case_ids": missing,
            }
        case_dirs = [str(out_dir / "cases" / cid) for cid in clean_ids]
        _write_checkpoint("experiments_done", {"case_ids": clean_ids, "num_cases": len(clean_ids)})
        analysis_path = out_dir / "analysis.json"
        ranked = _read_json(out_dir / "hypotheses_ranked.json") or {}
        topic = ranked.get("research_topic", "")
        proc = _run_script(
            [
                "scripts/analyze.py",
                "--cases", *case_dirs,
                "--metrics", metrics,
                "--output", str(analysis_path),
                "--topic", topic,
            ],
            timeout=3600,
        )
        ok = proc.returncode == 0 and analysis_path.is_file()
        if ok:
            _write_checkpoint("analysis_done", {"path": str(analysis_path), "case_ids": clean_ids})
            _update_study_state(current_stage="analysis")
        return {"ok": ok, "path": str(analysis_path), "stderr_tail": proc.stderr[-1500:]}

    def write_paper(template: str = "neurips") -> dict:
        """Draft the LaTeX paper, generate figures, and run the reviewer loop.

        Requires lit.json (fetch_literature), analysis.json
        (analyze_all_cases), and requirements.json (generate_case_requirements)
        to already exist.
        """
        ranked = _read_json(out_dir / "hypotheses_ranked.json") or {}
        topic = ranked.get("research_topic", "")

        # manifest.json schema per src/cfd_langgraph/paper_unified/pipeline.py
        # (_case_path_map / _success_case_ids): {"cases": [{case_id, case_path,
        # status}]} — built here from every case's run_result.json + decision.json
        # rather than a hand-waved stub, so the paper pipeline can actually find
        # and filter cases.
        cases_entries: List[Dict[str, Any]] = []
        cases_root = out_dir / "cases"
        if cases_root.is_dir():
            for case_dir in sorted(cases_root.glob("case_*")):
                run_result = _read_json(case_dir / "run_result.json") or {}
                decision = _read_json(case_dir / "decision.json") or {}
                status = "success" if _run_succeeded(run_result) and decision.get("status") == "PROCEED" else "failed"
                cases_entries.append(
                    {"case_id": case_dir.name, "case_path": str(case_dir.resolve()), "status": status}
                )
        required_inputs = [out_dir / "lit.json", out_dir / "requirements.json", out_dir / "analysis.json"]
        missing_inputs = [str(path) for path in required_inputs if not path.is_file()]
        if missing_inputs:
            return {"ok": False, "error": "Paper inputs are incomplete.", "missing_inputs": missing_inputs}
        if not any(entry["status"] == "success" for entry in cases_entries):
            return {"ok": False, "error": "No interpreter-approved successful case is available for the paper."}
        state = _read_json(out_dir / "state.json") or {}
        if state.get("mode") == "open_discovery":
            accepted_winners = []
            for entry in cases_entries:
                case_dir = Path(entry["case_path"])
                rr = _read_json(case_dir / "run_result.json") or {}
                if entry["status"] == "success" and rr.get("oed_status") == "PROCEED":
                    accepted_winners.append(entry["case_id"])
            if not accepted_winners:
                return {
                    "ok": False,
                    "error": "Open-discovery paper blocked: no baseline-beating candidate also passed physical interpretation.",
                }
        manifest_path = out_dir / "manifest.json"
        _write_json(manifest_path, {"study_id": out_dir.name, "topic": topic, "cases": cases_entries})
        paper_dir = out_dir / "paper"
        review_path = out_dir / "review.json"
        proc = _run_script(
            [
                "scripts/paper_unified.py",
                "--repo-root", str(_REPO_ROOT),
                "--run-dir", str(out_dir),
                "--topic", topic,
                "--paper-dir", str(paper_dir),
                "--analysis", str(out_dir / "analysis.json"),
                "--manifest", str(manifest_path),
                "--requirements", str(out_dir / "requirements.json"),
                "--literature", str(out_dir / "lit.json"),
                "--review-output", str(review_path),
                "--template", template,
            ] + (
                ["--mesh-independence", str(out_dir / "mesh_independence_context.json")]
                if (out_dir / "mesh_independence_context.json").is_file() else []
            ),
            timeout=7200,
        )
        plan_path = out_dir / "paper_unified_plan.json"
        cross_dir = out_dir / "cross_experiment_analysis"
        cross_ok = (
            cross_dir.is_dir()
            and (cross_dir / "aggregate.csv").is_file()
            and (cross_dir / "cross_experiment_interpretation.md").is_file()
            and bool(list(cross_dir.glob("*.png")))
        )
        ok = (
            proc.returncode == 0
            and (paper_dir / "main.pdf").is_file()
            and review_path.is_file()
            and plan_path.is_file()
            and cross_ok
        )
        if ok:
            _write_checkpoint("paper_experiment_plan_done", {"path": str(plan_path)})
            _write_checkpoint("cross_experiment_analysis_done", {"path": str(cross_dir)})
            _write_checkpoint("paper_review_done", {"paper": str(paper_dir / "main.pdf"), "review": str(review_path)})
            _update_study_state(current_stage="paper_review")
        return {
            "ok": ok,
            "paper_dir": str(paper_dir),
            "review_path": str(review_path),
            "stderr_tail": proc.stderr[-1500:],
        }

    # -----------------------------------------------------------------
    # Open-ended discovery: manager-driven search loop.
    #
    # Split across manager-scoped tools (oed_setup_search/oed_propose_
    # candidates/oed_record_candidate_results, below) and candidate-runner-
    # scoped tools (oed_run_code_mod_candidate/oed_run_experiment_candidate/
    # oed_score_candidate, further below — go on the oed-candidate-runner
    # subagent, not the manager) precisely so each candidate's execution is
    # a real `task` call the manager can fan out concurrently, get
    # interrupted between, and have cached — none of which applied when
    # this whole search ran as one blocking subprocess. See
    # scripts/oed_search_archive.py for the SearchArchive class every one of
    # these reads/updates, and scripts/open_ended_discovery.py --setup-only
    # for the one-time setup this reuses unchanged.
    # -----------------------------------------------------------------

    def _oed_disc_dir() -> Path:
        return out_dir / "open_ended_discovery"

    def oed_setup_search(
        topic: str, baseline_case_dir: str = "", total_budget: int = 10,
        starter_dir: str = "", evaluation_cases: Optional[List[str]] = None,
        prescribed_mesh_reason: str = "",
    ) -> dict:
        """One-time per-study setup for open-ended discovery — call this once,
        before oed_propose_candidates, for a "find a novel model/modification
        that beats baseline by X%" topic. Resolves reference data and
        authors (or reuses) the scored comparator scripts every candidate
        will be judged against, via ``open_ended_discovery.py --setup-only``
        — the same tested setup path the standalone script itself uses,
        stopped before its own iteration loop since you drive iteration
        yourself via oed_propose_candidates / task / oed_record_candidate_results.

        ``baseline_case_dir`` is required at runtime and should be run_mesh_gate's
        *selected_level* case, so baseline and candidates use the locked mesh.

        ``evaluation_cases`` turns this into a MULTI-CASE study: every
        candidate is run on each of these case directories and scored on the
        mean, instead of being scored on the single case it was built
        against. Use it when the objective is generalisation across a set of
        flows — a benchmark's held-out test cases, say — rather than
        performance on one. Declared once here, never per candidate, because
        two candidates scored on different case sets produce numbers that
        cannot be compared and the archive has no way to notice.

        Leave it empty for an ordinary single-case study; nothing changes.

        ``prescribed_mesh_reason`` is for studies where the mesh is FIXED BY THE
        TASK and a mesh-independence study is not merely unnecessary but
        forbidden — a benchmark that supplies the mesh and grades on it, so a
        refined mesh makes the result incomparable. Give the reason and the
        baseline may be a case supplied in the starter folder instead of a
        mesh-gate selected_level.

        This is not a way to skip mesh convergence. The baseline must be an
        UNMODIFIED case inside the starter directory — supplied data this study
        did not create and therefore cannot have refined. A case the study built
        is still refused, which is what the mesh gate exists to prevent.

        If read_starter_folder has already run (starter_understanding.json
        present), the starter dir is picked up automatically — pass
        starter_dir explicitly only if it hasn't, or to override it.
        """
        _ensure_routing(topic, "open_discovery")
        _update_study_state(topic=topic, mode="open_discovery", current_stage="open_ended_discovery")
        # Everything downstream — comparator authoring, the success threshold,
        # the locked topic every later call is checked against — keys off the
        # user's prompt, not the manager's summary of it.
        topic = _effective_topic(topic)
        # A brand-new family requires a code-modification evaluation (cost 2),
        # so a smaller budget cannot execute even the first legal candidate.
        total_budget = max(2, int(total_budget))
        baseline_path = Path(baseline_case_dir).expanduser().resolve() if baseline_case_dir else None
        if baseline_path is None or not baseline_path.is_dir():
            return {
                "ok": False,
                "error": "A valid mesh-gate selected_level case is required before OED setup.",
                "baseline_case_dir": baseline_case_dir,
            }
        allowed_selected_levels: set[Path] = set()
        aggregate = _read_json(out_dir / "selected_mesh_spec.json") or {}
        aggregate_groups = aggregate.get("groups", {}) if isinstance(aggregate, dict) else {}
        if isinstance(aggregate_groups, dict):
            for spec in aggregate_groups.values():
                if isinstance(spec, dict) and spec.get("converged") and spec.get("selected_level"):
                    allowed_selected_levels.add(Path(str(spec["selected_level"])).expanduser().resolve())
        for spec_path in (out_dir / "mesh_gate").glob("*/selected_mesh_spec.json"):
            spec = _read_json(spec_path) or {}
            if isinstance(spec, dict) and spec.get("converged") and spec.get("selected_level"):
                allowed_selected_levels.add(Path(str(spec["selected_level"])).expanduser().resolve())
        prescribed_reason = str(prescribed_mesh_reason or "").strip()
        mesh_provenance = "mesh_gate"
        if baseline_path not in allowed_selected_levels:
            # The gate exists so candidates are never scored on a mesh whose
            # convergence nobody established. A task that PRESCRIBES the mesh
            # satisfies that concern differently: the mesh is not this study's
            # to justify, and refining it invalidates the comparison the task
            # is asking for.
            #
            # Allowed only for a case inside the starter folder, which is
            # supplied input this study did not produce — so the escape hatch
            # cannot be used to bless a mesh the study refined itself.
            starter_root = starter_dir or (_read_json(out_dir / "starter_understanding.json") or {}).get("starter_dir", "")
            starter_resolved = None
            if starter_root:
                candidate_root = Path(str(starter_root)).expanduser()
                if not candidate_root.is_absolute():
                    candidate_root = (_REPO_ROOT / candidate_root)
                starter_resolved = candidate_root.resolve()
            inside_starter = bool(
                starter_resolved
                and (baseline_path == starter_resolved or starter_resolved in baseline_path.parents)
            )
            if not prescribed_reason:
                return {
                    "ok": False,
                    "error": (
                        "baseline_case_dir is not a converged selected_level produced by this "
                        "study's mesh gate. If the task PRESCRIBES the mesh and forbids changing "
                        "it, pass prescribed_mesh_reason explaining that, and give a baseline "
                        "case from the starter folder."
                    ),
                    "baseline_case_dir": str(baseline_path),
                    "allowed_selected_levels": sorted(str(p) for p in allowed_selected_levels),
                }
            if not inside_starter:
                return {
                    "ok": False,
                    "error": (
                        "prescribed_mesh_reason was given, but baseline_case_dir is not inside "
                        "the starter folder. A prescribed mesh must be supplied input this study "
                        "did not create; a case built during the study still needs the mesh gate."
                    ),
                    "baseline_case_dir": str(baseline_path),
                    "starter_dir": str(starter_resolved or ""),
                }
            mesh_provenance = "prescribed"
            print(
                f"[oed] mesh gate bypassed: the task prescribes the mesh. "
                f"baseline={baseline_path} reason={prescribed_reason[:160]}",
                flush=True,
            )
        # Resolved before anything is measured: the baseline has to be taken
        # over these same cases, and an invalid case is cheapest to reject now.
        declared_cases_resolved: List[Path] = []
        for entry in (evaluation_cases or []):
            path = Path(str(entry)).expanduser()
            if not path.is_absolute():
                path = (_REPO_ROOT / path).resolve()
            path = path.resolve()
            if not (path / "system" / "controlDict").is_file():
                # Refused at setup, where it costs nothing, rather than at the
                # first candidate that tries to run it — by then a model has
                # been compiled and the budget is already spent.
                return {
                    "ok": False,
                    "error": (
                        f"evaluation case is not an OpenFOAM case (no system/controlDict): {path}"
                    ),
                }
            declared_cases_resolved.append(path)

        su_path = out_dir / "starter_understanding.json"
        su = _read_json(su_path) or {}
        resolved_starter_dir = starter_dir or su.get("starter_dir", "")

        args = [
            "scripts/open_ended_discovery.py",
            "--run-dir", str(out_dir),
            "--topic", topic,
            "--budget", str(total_budget),
            "--setup-only",
        ]
        if su_path.is_file():
            args += ["--starter-understanding", str(su_path)]
        if resolved_starter_dir:
            args += ["--starter-dir", resolved_starter_dir]
        # Without this, the metric proposer has no real postProcessing sample
        # and may silently author zero usable comparators.
        args += ["--base-case-dir", str(baseline_path)]
        lit_path = out_dir / "lit.json"
        if lit_path.is_file():
            args += ["--literature", str(lit_path)]

        # 2h, not 30min. Setup authors one comparator script per proposed
        # metric and self-tests each against the baseline case; measured at
        # ~3-6 minutes per metric, so a four-metric objective runs past a
        # half-hour cap and gets killed with the comparators half-written —
        # after which nothing downstream can score anything. This is a
        # once-per-study call, so a generous cap costs nothing when setup is
        # quick.
        proc = _run_script(args, timeout=7200, env=_foamagent_env(settings.openfoam_path))
        disc_dir = _oed_disc_dir()
        result = {
            "ok": False,
            "disc_dir": str(disc_dir),
            "bound_comparators_path": str(disc_dir / "bound_comparators.json"),
            "objective_contract_path": str(disc_dir / "objective_contract.json"),
            "baseline_score": None,
            "baseline_direction": "min",
            "stderr_tail": proc.stderr[-2000:] if proc.returncode != 0 else "",
        }
        if proc.returncode != 0:
            return result

        # No comparators means nothing in this study can ever be scored, so
        # the search would stall a step later with a much less obvious
        # message. Say so here, where the cause is still visible.
        if not (_read_json(disc_dir / "bound_comparators.json") or {}):
            validation = _read_json(disc_dir / "metric_proposer_validation.json") or {}
            result["error"] = (
                "Setup produced no scored comparators, so no candidate could be evaluated. "
                "The metric proposer returned no usable metric for this objective. Re-run "
                "oed_setup_search; if it fails again, the objective or the reference data "
                "needs a look — do not start the search without comparators."
            )
            result["metric_proposer_attempts"] = validation.get("attempts")
            return result

        # Baseline score: computed here (not by the --setup-only call above,
        # which only knows a --baseline-metrics *path* it was never given)
        # directly against baseline_case_dir's own postProcessing, using the
        # comparators just authored/discovered — the same compute_metric_vector
        # every per-candidate score goes through, so the numbers are
        # comparable on the same footing.
        if _oedx is not None:
            try:
                bound = _read_json(disc_dir / "bound_comparators.json") or {}
                contract = _read_json(disc_dir / "objective_contract.json") or {}
                specs = _metric_specs(disc_dir)
                ref_files = [Path(p) for p in (contract.get("reference_files") or []) if Path(p).is_file()]
                if bound and ref_files:
                    # The baseline must be measured over the SAME cases a
                    # candidate is scored on. A 32-case candidate mean compared
                    # against a one-case baseline is not a comparison at all —
                    # measured on run closure_20260825_codex, which declared 32
                    # evaluation cases and then took its baseline from CBFS
                    # alone (0.00783, roughly an order of magnitude below the
                    # 32-case mean). Every candidate would have looked
                    # catastrophic for arithmetic reasons. The run detected it
                    # and refused to proceed, which was correct.
                    scored_baseline_cases = declared_cases_resolved or [baseline_path]
                    per_case_baseline: Dict[str, Any] = {}
                    primary_name = ""
                    direction = "min"
                    values: List[float] = []
                    mv = {}
                    for index, case in enumerate(scored_baseline_cases):
                        mv_one = _oedx.compute_metric_vector(
                            case_dir=case,
                            bound_comparators=bound,
                            reference_file=ref_files[0],
                            metric_specs=specs,
                        )
                        if index == 0:
                            mv = mv_one
                        one_metrics = mv_one.get("metrics") or {}
                        name, value, one_direction = _select_primary_metric(one_metrics, specs)
                        if index == 0:
                            primary_name, direction = name, one_direction
                        try:
                            parsed = float(value)
                        except (TypeError, ValueError):
                            parsed = float("nan")
                        per_case_baseline[case.name] = parsed if math.isfinite(parsed) else None
                        if math.isfinite(parsed):
                            values.append(parsed)

                    if len(scored_baseline_cases) > 1:
                        unscored = [n for n, v in per_case_baseline.items() if v is None]
                        if unscored:
                            # Refused rather than averaged over what worked: the
                            # baseline is the number every later result is judged
                            # against, and one computed over a different subset
                            # than the candidates silently biases the whole study.
                            result["baseline_score"] = None
                            result["baseline_score_error"] = (
                                f"baseline could not be measured on {len(unscored)} of "
                                f"{len(scored_baseline_cases)} declared evaluation cases "
                                f"({', '.join(sorted(unscored)[:6])}); refusing to build a "
                                "baseline over a different case set than candidates are scored on."
                            )
                        else:
                            result["baseline_score"] = sum(values) / len(values)
                            result["baseline_per_case"] = per_case_baseline
                            result["baseline_cases_scored"] = len(values)
                    else:
                        result["baseline_score"] = values[0] if values else None
                    result["baseline_direction"] = direction
                    result["primary_metric"] = primary_name
                    result["baseline_metric_vector"] = mv
            except Exception as exc:
                result["baseline_score_error"] = f"{exc}"

        bound = _read_json(disc_dir / "bound_comparators.json") or {}
        contract = _read_json(disc_dir / "objective_contract.json") or {}
        if not bound or not contract or result["baseline_score"] is None:
            result["error"] = (
                "OED setup did not produce a verified comparator and baseline score; "
                "candidate execution is blocked."
            )
            return result

        target_pct = _llm_target_improvement_pct(_effective_topic(topic), settings)
        baseline_doc = {
            # What the baseline was measured over. A reader of the run — and
            # oed_score_candidate's own guard — needs to know whether this
            # number is one case or the mean of many.
            "evaluation_cases_scored": result.get("baseline_cases_scored"),
            "per_case": result.get("baseline_per_case"),
            "metric": result.get("primary_metric", ""),
            "value": result["baseline_score"],
            "direction": result["baseline_direction"],
            "verified": True,
            "case_dir": str(baseline_path),
        }
        _write_json(disc_dir / "baseline_score.json", baseline_doc)
        config = {
            "topic": topic,
            "total_budget": total_budget,
            "baseline_case_dir": str(baseline_path),
            "baseline_metric": baseline_doc["metric"],
            "baseline_direction": baseline_doc["direction"],
            "target_improvement_pct": target_pct,
            "saturation_window": max(3, total_budget // 4),
            # How this study justified its baseline mesh. Recorded so a reader
            # of the run can tell a gate-converged mesh from a prescribed one
            # without reconstructing the argument.
            "mesh_provenance": mesh_provenance,
            "prescribed_mesh_reason": prescribed_reason if mesh_provenance == "prescribed" else "",
        }
        if declared_cases_resolved:
            config["evaluation_cases"] = [str(p) for p in declared_cases_resolved]
        _write_json(disc_dir / "search_config.json", config)
        result.update({"ok": True, "target_improvement_pct": target_pct, "search_config": str(disc_dir / "search_config.json")})
        _write_checkpoint(
            "baseline_setup_done",
            {"baseline_case_dir": str(baseline_path), "baseline_score": baseline_doc},
        )
        _write_checkpoint(
            "metric_setup_done",
            {"metric_specs": str(disc_dir / "metric_specs.json"), "bound_comparators": str(disc_dir / "bound_comparators.json")},
        )
        return result

    def oed_propose_candidates(
        topic: str, num_candidates: int = 3, force_new_families: bool = False
    ) -> dict:
        """Propose ``num_candidates`` concrete model-modification candidates
        to try next, each conditioned on a niche picked by the search
        archive's explore/exploit selection (scripts/oed_search_archive.py)
        — favoring a family that's scoring well but under-visited, with "try
        a brand-new family" always competing on equal footing. Requires
        oed_setup_search to have already run.

        Set ``force_new_families=True`` when a previous call came back with no
        candidates: every niche is then required to be a family the archive has
        never seen, which is the way out of a round where the proposer keeps
        re-suggesting things already tried. An empty batch means the proposal
        space collapsed onto known ideas, NOT that the search is finished —
        budget_remaining is what says whether the search is finished.

        Launch every returned candidate as a `task` call to the
        oed-candidate-runner subagent — a SINGLE message with one task call
        per candidate, concurrently, the same pattern as launching cases —
        then call oed_record_candidate_results with the candidate_dir each
        one reports back.
        """
        disc_dir = _oed_disc_dir()
        config = _read_json(disc_dir / "search_config.json") or {}
        if not config or not (_read_json(disc_dir / "baseline_score.json") or {}).get("verified"):
            return {"error": "OED search is not initialized with a verified baseline."}
        topic = _effective_topic(topic)
        if str(topic or "").strip() != str(config.get("topic", "") or "").strip():
            # Hand back the locked topic. The match is byte-exact on purpose —
            # the search objective must not drift mid-run — but a refusal that
            # does not say what to match is unguessable, and the caller can
            # only burn retries paraphrasing at it.
            return {
                "error": (
                    "OED proposal topic does not match the topic this search was locked to "
                    "at oed_setup_search. Pass the locked_topic below verbatim."
                ),
                "locked_topic": str(config.get("topic", "") or ""),
                "received_topic": str(topic or ""),
            }
        history = _read_json(disc_dir / "history.json") or []
        if not isinstance(history, list):
            history = []
        total_budget = int(config.get("total_budget", 0) or 0)
        budget_used = sum(
            int(h.get("cost", 0) or 0)
            for h in history if isinstance(h, dict) and h.get("action_type") in {"code_mod", "experiment"}
        )
        budget_remaining = max(0, total_budget - budget_used)
        if budget_remaining <= 0:
            return {"candidates": [], "budget_used": budget_used, "budget_remaining": 0, "budget_exhausted": True}
        archive = SearchArchive()
        if history:
            archive.replay(history, baseline_direction=str(config.get("baseline_direction", "min")))

        picks: List[Dict[str, Any]] = []
        for _ in range(min(max(1, num_candidates), budget_remaining)):
            if budget_remaining == 1 and archive.niches:
                reusable = [
                    (key, niche)
                    for key, niche in archive.niches.items()
                    if isinstance(niche.get("elite_history_entry"), dict)
                    and Path(str(niche["elite_history_entry"].get("case_dir", ""))).is_dir()
                ]
                if reusable:
                    key, niche = min(
                        reusable,
                        key=lambda item: (
                            item[1].get("elite_norm_score") is None,
                            item[1].get("elite_norm_score") or 0.0,
                        ),
                    )
                    sel = {
                        "family": key[0],
                        "strategy": key[1],
                        "is_new": False,
                        "elite": niche["elite_history_entry"],
                        "experiment_only": True,
                    }
                else:
                    break
            elif force_new_families:
                sel = archive.select_niche(
                    budget_remaining=budget_remaining,
                    budget_total=total_budget,
                    force_new_family=True,
                )
            else:
                # Allocation, not just niche choice: deepen an existing lineage,
                # widen into a family already known, or open a new mechanism --
                # decided by Thompson sampling over what each has actually
                # returned (Xin26, Mis25). select_niche remains the cold-start
                # and forced-exploration path, and select_action falls back to
                # it for `widen`, so the older contract is unchanged.
                #
                # This exists because the previous loop could only ask a
                # breadth question. On run closure_20260826_codex that produced
                # 33 mechanism families over 55 evaluations -- 1.1 evaluations
                # per cell -- while the 5 refinements that did happen beat
                # their parent 5 times out of 5 and beat baseline 100% of the
                # time against 72% for fresh starts, on 7% of the budget.
                sel = archive.select_action(
                    budget_remaining=budget_remaining,
                    budget_total=total_budget,
                )
            if sel.get("budget_exhausted"):
                break
            picks.append(sel)
            # Provisional visit bump, discarded after this call — real
            # visits get recorded later via
            # oed_record_candidate_results -> archive.replay(history). This
            # nudges later picks in the batch toward a different niche as a
            # leading one's exploration bonus shrinks, but a niche that has
            # decisively pulled ahead can still legitimately win every slot
            # in the batch — that's correct exploitation once the evidence
            # supports it, not a diversification bug.
            fam = sel.get("family")
            if fam:
                key = (fam, sel.get("strategy") or "analytic")
                archive.niches.setdefault(key, archive._new_niche())
                archive.niches[key]["visits"] += 1
            # A "propose a new family" pick carries no family name yet, so it
            # cannot bump a niche. Deliberately no synthetic placeholder
            # niche here: the new-family option's exploration bonus is
            # sqrt(log(total_visits + 1) / 1), which *grows* with total
            # visits, so faking a visit would make the next pick even more
            # likely to ask for another new family. Batch-level
            # differentiation is enforced instead where it can actually be
            # checked — on the classified family of what the proposer
            # returns (batch_families below), and by naming the already-taken
            # families in the prompt.

        baseline = _read_json(disc_dir / "baseline_score.json") or {}
        known_families = sorted(archive.families())
        directions = _approved_hypothesis_directions(out_dir)
        # Concrete already-tried hypotheses, with their outcome. The archive
        # summary only carries each family's *best* result, so without this the
        # proposer has no way to know that "c_cd = 0.8" was already measured —
        # and it re-proposed exactly that, repeatedly.
        tried_lines: List[str] = []
        for entry in history[-40:]:
            if not isinstance(entry, dict) or entry.get("action_type") not in {"code_mod", "experiment"}:
                continue
            description = str(entry.get("model_description") or "").strip()
            if not description:
                continue
            score = entry.get("score") if isinstance(entry.get("score"), dict) else {}
            value = score.get("value")
            strategy_used = str(entry.get("strategy") or "").strip()
            tried_lines.append(
                f"  - [{entry.get('family', '?')}"
                + (f" via {strategy_used}" if strategy_used else "")
                + f"] {description[:180]}"
                + (f"  -> {value}" if value is not None else "  -> failed")
            )
        niche_lines = []
        for i, sel in enumerate(picks, 1):
            if sel.get("is_new"):
                niche_lines.append(
                    f"{i}. Target a NEW model family not yet in the archive. This must be "
                    "action_type=code_mod, and it must be a different family from every "
                    "other candidate in this same batch — not a second variant of one "
                    "idea. Modifying a different equation of the same model counts as a "
                    "different family (production vs destruction vs diffusion vs "
                    "rotation-curvature); three coefficient variants of one term do not. "
                    # The mechanism is open here, so the strategy is too — and
                    # left unsaid it defaults to analytic every time. State the
                    # choice explicitly and make the model justify it from the
                    # resources rather than from habit.
                    "`strategy` is YOUR choice for this one and it is not "
                    "automatically 'analytic': pick whichever of "
                    + ", ".join(STRATEGIES)
                    + " the available data and libraries actually support for this "
                    "mechanism, and say in `plan` why that way of determining the "
                    "model suits it."
                )
            elif sel.get("is_new_strategy"):
                strategy = str(sel.get("strategy") or "")
                niche_lines.append(
                    f"{i}. Family '{sel.get('family')}' has already scored, but NEVER via "
                    f"strategy '{strategy}' — that cell of the archive is empty and this "
                    "candidate is to fill it. Keep the mechanism; change how the numbers "
                    f"in it are determined. Set strategy='{strategy}', meaning: "
                    + STRATEGY_GUIDANCE.get(strategy, "")
                    + ". Put the concrete steps in `plan` — which files are read, which "
                    "optimiser or fit runs, and how its result becomes the compiled model. "
                    "A plan that merely compiles fixed coefficients is NOT this strategy; "
                    "it will be reclassified and this cell will still be empty. "
                    "Use action_type=code_mod."
                )
            else:
                elite = sel.get("elite") or {}
                formula = str(elite.get("formula") or elite.get("model_description") or "")[:500]
                elite_strategy = str(sel.get("strategy") or "").strip()
                # When the allocator chose to deepen, say so and show the
                # chain's trajectory. "Refine this" and "refine this, it has
                # improved twice in a row and is still moving" are different
                # instructions, and the second is the one that produced the
                # only depth-2 lineage in the last study.
                trace = sel.get("score_trace") or []
                if sel.get("action") == "deepen" and len(trace) > 1:
                    steps = " -> ".join(f"{v:.6g}" for v in trace)
                    niche_lines.append(
                        f"{i}. DEEPEN an existing lineage (depth {sel.get('depth')}, "
                        f"rooted at iteration {sel.get('lineage_id')}). Its scores so far: "
                        f"{steps}. This chain is being refined because it is still "
                        f"improving; continue in the same direction rather than starting "
                        f"over, and change ONE thing so the next step is attributable."
                    )
                niche_lines.append(
                    f"{i}. Build on family '{sel.get('family')}'"
                    + (
                        f" (its best result came via strategy '{elite_strategy}'; you may "
                        "keep that strategy or try this mechanism a different way — the "
                        "archive scores each combination separately)"
                        if elite_strategy else ""
                    )
                    + f" (best result so far, "
                    f"from iteration {elite.get('iteration', '?')}): {formula}. "
                    + (
                        f"That model exposes these coefficients at runtime: "
                        f"{', '.join(_runtime_coefficients(Path(str(elite.get('case_dir', '')))))}. "
                        "Changing any of them ALONE needs no recompile — use "
                        "action_type=experiment with parameters and base_case_dir, for half "
                        "the budget. "
                        if _runtime_coefficients(Path(str(elite.get("case_dir", "")))) else ""
                    )
                    + "Use action_type=experiment only for coefficient-only changes and include its base_case_dir; "
                    + ("You have only one budget unit left, so use action_type=experiment. " if sel.get("experiment_only") else "")
                    + "Otherwise use code_mod."
                )

        prompt = (
            f"Research topic: {topic}\n\n"
            f"Budget remaining: {budget_remaining}/{total_budget} units (code_mod costs 2, experiment costs 1).\n"
            + _study_resources(out_dir, disc_dir) + "\n\n"
            + (
                "CHOOSING HOW, NOT ONLY WHAT. Each candidate carries a `strategy`: how you "
                "will DETERMINE the modification, as distinct from which physics you are "
                "modifying. Reasoning a form out from physics is one way. Fitting "
                "coefficients by optimising the scored objective through the solver is "
                "another. Fitting a form to stored high-fidelity data before it ever meets "
                "the solver is another. Sweeping an already-compiled model's coefficients "
                "is another. They can be staged or combined.\n"
                "The archive below keeps a separate elite for every (mechanism, strategy) "
                "pair and reports what each strategy has actually returned, so this is a "
                "decision you can make from evidence rather than taste. Nothing here "
                "prefers one; a strategy earns budget by scoring.\n"
                "If a strategy needs steps beyond 'implement this hypothesis' — reading "
                "data files, running an optimiser, feeding a fitted result back into the "
                "model — put those steps in `plan`. The candidate agent has a shell, can "
                "read any file listed above, and can import the libraries listed above.\n\n"
            )
            + f"{archive.render_summary(baseline_score=baseline.get('value'), baseline_direction=baseline.get('direction', 'min'))}\n\n"
            + (
                f"Families already evaluated in this study (do NOT propose these as new): "
                f"{', '.join(known_families)}\n\n" if known_families else "\n"
            )
            + (
                "Already evaluated in this study — do NOT propose any of these again, "
                "they are spent budget with known results:\n"
                + "\n".join(tried_lines)
                + "\n\n"
                if tried_lines else ""
            )
            + (
                "APPROVED DIRECTIONS — these came from this study's own literature-grounded "
                "hypothesis stage: generated against the retrieved papers, critiqued for "
                "physical plausibility and implementability, and approved by a human. Prefer "
                "an untried one of these over inventing a family from scratch; they are "
                "already vetted for this exact case. Skip any whose idea appears in the "
                "already-evaluated list above.\n"
                + "\n".join(
                    f"  - {d['name']}: {d['detail']}" if d["detail"] else f"  - {d['name']}"
                    for d in directions
                )
                + "\n\n"
                if directions else ""
            )
            + f"Propose exactly {len(picks)} concrete candidate modifications, one per "
            "instruction below — each must be specific enough to implement directly "
            "(named equations/terms/coefficients, not a vague direction):\n"
            + "\n".join(niche_lines)
        )
        try:
            batch = foam_llm.with_structured_output(_OEDCandidateBatch).invoke(prompt)
            raw_candidates = [c.model_dump() for c in batch.candidates]
        except Exception as exc:
            # A malformed structured-output response is a transient proposer
            # failure, not a reason to hand the manager a raw traceback with
            # no idea what to do next. No budget was spent, so retrying is
            # always safe.
            return {
                "error": (
                    f"Candidate proposer failed ({type(exc).__name__}: {exc}). No budget was "
                    "consumed — call this tool again, optionally with a smaller num_candidates."
                ),
                "candidates": [],
            }
        if len(raw_candidates) < len(picks):
            return {
                "error": f"Candidate proposer returned {len(raw_candidates)} candidates for {len(picks)} requested niches.",
                "candidates": [],
            }

        candidates: List[Dict[str, Any]] = []
        # Seeded from every variant name this study has already spent budget
        # on, not just this batch: a repeated name would write into the same
        # cand_<name> directory, overwrite the earlier candidate_record.json,
        # and then be skipped as already-recorded — paying for a candidate
        # whose result is discarded and corrupting the earlier one's case_dir.
        seen_variants: set[str] = {
            str(h.get("variant_name") or "") for h in history if isinstance(h, dict)
        }
        seen_variants.discard("")
        # What has already been *evaluated*, by content. Budget is the scarce
        # resource here, and paying twice for the same hypothesis buys nothing.
        seen_fingerprints: set[str] = {
            _oed_candidate_fingerprint(
                str(h.get("family") or ""),
                str(h.get("action_type") or ""),
                str(h.get("model_description") or ""),
                h.get("parameters") if isinstance(h.get("parameters"), dict) else None,
                strategy=str(h.get("strategy") or ""),
            )
            for h in history
            if isinstance(h, dict) and h.get("action_type") in {"code_mod", "experiment"}
        }
        duplicates_skipped: List[str] = []
        # Families claimed earlier in THIS batch. archive.niches only knows
        # about families that have already been *evaluated*, so without this
        # every pick in an opening round (empty archive -> every pick is
        # "propose a new family") accepts whatever the proposer returns, and
        # three variants of one family sail through as three new families.
        batch_families: set[str] = set()
        committed_cost = 0
        for i, (candidate, selection) in enumerate(zip(raw_candidates, picks), 1):
            action = candidate.get("action_type")
            if selection.get("is_new"):
                action = "code_mod"
            elif selection.get("experiment_only"):
                action = "experiment"
            cost = 2 if action == "code_mod" else 1
            if committed_cost + cost > budget_remaining:
                continue
            candidate["action_type"] = action
            candidate["variant_name"] = _safe_variant_slug(candidate.get("variant_name", ""), f"candidate_{budget_used + i:03d}")
            base_slug = candidate["variant_name"]
            suffix = 2
            while candidate["variant_name"] in seen_variants:
                candidate["variant_name"] = f"{base_slug}_{suffix}"
                suffix += 1
            seen_variants.add(candidate["variant_name"])

            # The strategy the proposer declared, binned to the archive's
            # coarse vocabulary. Free text would give every candidate its own
            # niche and destroy the second dimension's usefulness; see
            # oed_search_archive.STRATEGIES.
            # use_llm: the plan is prose, and whether it fits or merely reads
            # data is a judgement the keyword table gets wrong -- see
            # normalize_strategy. This is a live proposal, so the model call is
            # paid once here and the answer is stored; replay never re-asks.
            candidate["strategy"] = normalize_strategy(
                candidate.get("strategy") or candidate.get("plan") or candidate.get("hypothesis", ""),
                plan=candidate.get("plan") or "",
                hypothesis=candidate.get("hypothesis") or "",
                use_llm=True,
            )

            elite = selection.get("elite") or {}
            if selection.get("family"):
                candidate["target_family"] = selection["family"]
            else:
                # Niche identity is deterministic and comes from the same
                # classifier used during archive replay. Do not let the LLM
                # invent a label that makes an old family look new.
                # use_llm=True: this is the ONE place a family is decided for
                # a brand-new proposal, and it is stamped into history.json so
                # replay never has to re-derive it. The keyword table cannot
                # name a mechanism nobody added to it, and an over-broad label
                # merges distinct approaches into one archive niche.
                classified = SearchArchive.classify(
                    candidate.get("hypothesis", ""),
                    candidate.get("model_name_to_reuse", ""),
                    use_llm=True,
                )
                # Only an in-batch collision is a reason to drop a proposal.
                # Dropping because the family already exists in the ARCHIVE
                # deadlocks the search: once every family has been visited,
                # select_niche keeps returning "propose a new family" (all
                # elites are None, so every q ties and the zero-visit new
                # option wins), every proposal classifies into some existing
                # family, and every one is discarded — the tool returns an
                # empty list forever while budget_used never advances.
                # Observed: six families visited, then nothing but empty
                # batches until the run gave up.
                if selection.get("is_new") and classified in batch_families:
                    continue
                candidate["target_family"] = classified
            fingerprint = _oed_candidate_fingerprint(
                str(candidate.get("target_family") or ""),
                str(action or ""),
                str(candidate.get("hypothesis") or ""),
                candidate.get("parameters") if isinstance(candidate.get("parameters"), dict) else None,
                strategy=str(candidate.get("strategy") or ""),
            )
            repeats = None
            if fingerprint in seen_fingerprints:
                repeats = "an identical proposal"
            elif not (action == "experiment" and candidate.get("parameters")):
                # Exact-keyed coefficient experiments are already settled above;
                # everything else needs a reading of what the proposal actually
                # does, which is a judgement, not a string comparison.
                repeats = _llm_duplicate_of(candidate, history, foam_llm)
            if repeats:
                duplicates_skipped.append(
                    f"{str(candidate.get('hypothesis') or '')[:100]} (repeats {repeats})"
                )
                seen_variants.discard(candidate["variant_name"])
                continue
            seen_fingerprints.add(fingerprint)
            batch_families.add(str(candidate["target_family"]))
            if action == "experiment":
                base_case = (
                    candidate.get("base_case_dir")
                    or elite.get("case_dir")
                    or elite.get("compiled_case_dir")
                )
                parameters = candidate.get("parameters") or {}
                parameters_valid = bool(parameters) and all(
                    re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", str(key))
                    and isinstance(value, (int, float))
                    and math.isfinite(float(value))
                    for key, value in parameters.items()
                )
                if not base_case or not Path(base_case).is_dir() or not parameters_valid:
                    # An experiment without a compiled parent or actual
                    # finite coefficient changes would only rerun stale or
                    # malformed baseline data.
                    continue
                candidate["base_case_dir"] = str(Path(base_case).resolve())
                candidate["parameters"] = {str(k): float(v) for k, v in parameters.items()}
            # Lineage: which archive entry this candidate was built from.
            # Recorded here, where it is actually known, rather than asked of
            # the subagent later — a candidate's parent is a fact about the
            # selection, not something to be re-typed through two prompts.
            candidate["parent_iteration"] = (elite or {}).get("iteration")
            # From this candidate's own `selection` -- the same pick that set
            # parent_iteration just above -- so it stays correct however many
            # candidates the loop skips. Recorded so record_candidate_results
            # can tell the allocator whether this arm paid off.
            candidate["search_action"] = selection.get("action")
            candidate["lineage_id"] = selection.get("lineage_id")
            committed_cost += cost
            candidates.append(candidate)

        # Persisted so oed_score_candidate can stamp family/lineage onto
        # candidate_record.json from the tool that decided them, instead of
        # trusting whatever the candidate-runner echoes back.
        proposals = _read_json(disc_dir / "proposals.json") or {}
        for candidate in candidates:
            proposals[candidate["variant_name"]] = {
                "target_family": candidate.get("target_family"),
                "parent_iteration": candidate.get("parent_iteration"),
                "action_type": candidate.get("action_type"),
                # Carried through to candidate_record.json and thus history.json.
                # Without them a coefficient experiment's only durable identity
                # was its prose description, and the proposer worded the same
                # experiment seven different ways — "Increase c_cd from 0.5 to
                # 1.0 ..." and "SA-Cross-Diffusion with c_cd=1.0" are the same
                # run and no text comparison will ever say so.
                "parameters": candidate.get("parameters") or {},
                "model_name_to_reuse": candidate.get("model_name_to_reuse", ""),
                "strategy": candidate.get("strategy", ""),
                "plan": candidate.get("plan", ""),
                # Which allocation arm produced this candidate, so its outcome
                # can be fed back to that arm. Without it the widen/new_family
                # arms never leave their priors and the refine-versus-explore
                # split stays fixed -- the schedule Xin26 shows is suboptimal.
                "search_action": candidate.get("search_action"),
                "lineage_id": candidate.get("lineage_id"),
            }
        _write_json(disc_dir / "proposals.json", proposals)

        result = {
            "candidates": candidates,
            "archive_summary": _render_archive_summary(archive, disc_dir),
            "budget_used": budget_used,
            "budget_remaining": budget_remaining,
            "proposed_cost": committed_cost,
        }
        if duplicates_skipped:
            # Surfaced, not swallowed: a batch that shrinks silently looks like
            # the proposer under-delivering, and the fix (tell it what has
            # already been tried) is invisible unless the skips are reported.
            result["duplicates_skipped"] = duplicates_skipped
        if not candidates:
            # Every pick can legitimately be filtered out (duplicate family,
            # unaffordable cost, an experiment with no compiled parent). Say
            # so explicitly with a way forward — an empty list and no error
            # reads as "nothing to do", and the manager can loop on it forever
            # without budget_used ever advancing.
            affordable = budget_remaining >= 2
            guidance = (
                "Call this tool again with force_new_families=True — that requires every "
                "niche to be a family the archive has never seen, which is what breaks the "
                "loop. Do NOT treat this as the search being finished: "
                f"budget_remaining={budget_remaining} of {total_budget}."
                if affordable else
                f"budget_remaining={budget_remaining} is below the 2 units a code_mod costs, "
                "so the search really is finished."
            )
            result["error"] = (
                "No candidate survived validation this round (duplicate families, "
                "unaffordable cost, or an experiment with no compiled parent). No budget "
                "was consumed. " + guidance
            )
            result["force_new_families_available"] = affordable
        return result

    def oed_record_candidate_results(candidate_dirs: List[str]) -> dict:
        """Record a finished batch of candidates into the search history and
        archive. Pass the exact candidate_dir paths you used when launching
        each oed-candidate-runner task — each one's oed_score_candidate call
        already wrote a candidate_record.json there; this reads those back
        (not whatever the subagent said in prose) so nothing gets lost in
        transcription.

        Returns budget_used, proceed_count, is_saturated, and the updated
        archive summary — use these to decide whether to call
        oed_propose_candidates again or move on to interpret_case /
        analyze_all_cases / write_paper for whichever candidates PROCEEDed.
        """
        disc_dir = _oed_disc_dir()
        disc_dir.mkdir(parents=True, exist_ok=True)
        history_path = disc_dir / "history.json"
        history = _read_json(history_path) or []
        if not isinstance(history, list):
            history = []
        next_iter = max([int(h.get("iteration", 0)) for h in history], default=0) + 1
        already_recorded = {
            str(h.get("candidate_dir", "")) for h in history if isinstance(h, dict) and h.get("candidate_dir")
        }

        # Every finished candidate on disk, not only the ones the manager
        # remembered to pass.
        #
        # A candidate's result becomes durable the moment oed_score_candidate
        # writes candidate_record.json, but it only enters history.json when
        # the manager later calls this with that directory. Anything that ends
        # the process in between -- a crash, a kill, an interrupt, a manager
        # that simply concludes the batch produced nothing -- strands it.
        # Measured on run closure_20260826_codex: sst_crossdiff_scale_065 sat
        # unrecorded with a complete record and the best score in the study
        # (0.11028, +2.92% over baseline) while the manager reported that the
        # batch "produced no recorded candidate directories" and moved on.
        #
        # Sweeping costs a directory listing. Records already in history are
        # skipped by candidate_dir below, so re-recording is impossible.
        #
        # The file's existence is NOT proof that scoring finished -- since the
        # repair path was added, oed_apply_repair and oed_note_repair_attempt
        # both create it early to bank an attempt before the work runs. What
        # keeps an unscored candidate out of history is the case_dir check
        # below: a repair-only record has no case_dir, so it fails to resolve
        # inside the candidate and is reported missing rather than recorded.
        swept: List[str] = []
        requested = {str(Path(c).expanduser().resolve()) for c in candidate_dirs}
        for record_path in sorted(disc_dir.glob("cand_*/candidate_record.json")):
            found = str(record_path.parent.resolve())
            if found not in requested and found not in already_recorded:
                swept.append(found)
        candidate_dirs = list(candidate_dirs) + swept

        missing: List[str] = []
        for cdir in candidate_dirs:
            candidate_path = Path(cdir).expanduser().resolve()
            if candidate_path.parent != disc_dir.resolve() or not candidate_path.name.startswith("cand_"):
                missing.append(str(candidate_path))
                continue
            resolved_cdir = str(candidate_path)
            if resolved_cdir in already_recorded:
                continue
            record_path = candidate_path / "candidate_record.json"
            record = _read_json(record_path)
            if not record:
                missing.append(resolved_cdir)
                continue
            try:
                Path(str(record.get("case_dir", ""))).resolve().relative_to(candidate_path)
            except (ValueError, OSError):
                missing.append(resolved_cdir)
                continue
            record["iteration"] = next_iter
            record["candidate_dir"] = resolved_cdir
            next_iter += 1
            history.append(record)
            already_recorded.add(resolved_cdir)
        _write_json(history_path, history)

        config = _read_json(disc_dir / "search_config.json") or {}
        archive = SearchArchive()
        archive.replay(history, baseline_direction=str(config.get("baseline_direction", "min")))
        real_evals = [h for h in history if h.get("action_type") in ("code_mod", "experiment")]
        budget_used = sum(int(h.get("cost", 0) or 0) for h in real_evals)
        proceed_count = sum(1 for h in real_evals if h.get("status") == "PROCEED")
        total_budget = int(config.get("total_budget", 0) or 0)
        window = int(config.get("saturation_window", max(3, len(real_evals) // 3 or 1)))
        saturated = archive.is_saturated(window=window)
        budget_exhausted = bool(total_budget and budget_used >= total_budget)
        # Candidates that measurably beat the baseline. This — not an
        # interpreter verdict — is what makes a candidate worth promoting.
        #
        # `PROCEED` is set by `interpret_case`, and `interpret_case` only
        # resolves cases already promoted into `cases/`. Gating promotion on
        # `proceed_count > 0` therefore closed a loop that could never open:
        # interpretation needs promotion, promotion needs interpretation.
        # Measured on a real study — 49 evaluations, 23 of them better than
        # baseline, best +4.90% — nothing was ever promoted, `search_complete`
        # stayed false at 64/100 budget despite the archive reporting
        # saturation, and the manager could not reach interpretation, analysis
        # or the paper. It tried to register the cases by hand, was correctly
        # refused by the protected-artifact guard, and stopped.
        #
        # Promoting on measured improvement lets `interpret_case` do its actual
        # job — judging whether a numerically better case is *physically*
        # acceptable — instead of being a precondition for itself. A candidate
        # that is worse than baseline is still never promoted, and a study
        # where nothing beat baseline still ends with nothing to write up.
        baseline_value = (_read_json(disc_dir / "baseline_score.json") or {}).get("value")
        direction = str(config.get("baseline_direction", "min")).strip().lower()

        def _beats_baseline(entry: Dict[str, Any]) -> bool:
            score = entry.get("score") if isinstance(entry.get("score"), dict) else None
            value = (score or {}).get("value")
            if value is None or baseline_value is None:
                return False
            try:
                return float(value) > float(baseline_value) if direction == "max" else float(value) < float(baseline_value)
            except (TypeError, ValueError):
                return False

        improving = [h for h in real_evals if _beats_baseline(h)]
        search_complete = budget_exhausted or (saturated and (proceed_count > 0 or bool(improving)))
        has_winner = proceed_count > 0 or bool(improving)

        promoted_case_ids: List[str] = []
        if search_complete and has_winner:
            cases_root = out_dir / "cases"
            cases_root.mkdir(parents=True, exist_ok=True)
            # Promotion is re-entered on every call once the search is
            # complete (a retry after a Ctrl-C pause, a resume, or simply the
            # manager calling this tool twice). _copy_completed_case is
            # rmtree-then-copytree, so re-promoting an already-interpreted
            # case silently destroys the decision.json and figures
            # interpret_case wrote into it, and resets bridge.json's decisions
            # to empty. Copy only what is not already promoted and intact.
            already_promoted = {
                str(cid)
                for cid in ((_read_json(out_dir / "bridge.json") or {}).get("case_ids") or [])
                if (cases_root / str(cid) / "run_result.json").is_file()
            }
            # Preserve the approved requirements this study ran on. They are
            # the authority _canonical_requirement and run_mesh_gate validate
            # against, so replacing them with OED stubs made every later
            # run_case_native fail with "requirement_text does not match the
            # approved requirements.json entry" for a genuinely approved
            # requirement. Drop only this tool's own previously-written stubs,
            # so repeated calls don't accumulate duplicates.
            existing_requirements = _read_json(out_dir / "requirements.json") or []
            requirements: List[Dict[str, Any]] = [
                item for item in existing_requirements
                if isinstance(item, dict) and item.get("study_id") != "open_discovery"
            ]
            existing_manifest = _read_json(out_dir / "manifest.json") or {}
            manifest_entries: List[Dict[str, Any]] = [
                item for item in (existing_manifest.get("cases") or [])
                if isinstance(item, dict) and not str(item.get("case_id", "")).startswith("case_oed_")
            ]

            baseline = _read_json(disc_dir / "baseline_score.json") or {}
            baseline_src = Path(str(config.get("baseline_case_dir", "")))
            if baseline_src.is_dir():
                baseline_id = "case_oed_baseline"
                baseline_dst = cases_root / baseline_id
                if baseline_id not in already_promoted:
                    _copy_completed_case(baseline_src, baseline_dst)
                baseline_ok = _case_has_clean_solver_log(baseline_dst)
                _write_json(
                    baseline_dst / "run_result.json",
                    {
                        "status": "success" if baseline_ok else "failed",
                        "success": baseline_ok,
                        "case_id": baseline_id,
                        "case_dir": str(baseline_dst),
                        "source_case_dir": str(baseline_src),
                        "role": "oed_baseline",
                        "error_logs": [] if baseline_ok else ["No clean solver log found in promoted baseline."],
                    },
                )
                promoted_case_ids.append(baseline_id)
                requirements.append({"case_id": baseline_id, "user_requirement_text": f"OED baseline for: {config.get('topic', '')}", "study_id": "open_discovery"})
                manifest_entries.append({"case_id": baseline_id, "case_path": str(baseline_dst), "status": "success" if baseline_ok else "failed"})

            # Promote the baseline-beating candidates, best first, rather than
            # every valid evaluation. A saturated search runs dozens of
            # candidates, most of them worse than baseline; copying all of them
            # means dozens of full OpenFOAM case trees on disk and an
            # interpret_case call each, to build a comparison table whose extra
            # rows are all "worse than the control". The ones that beat the
            # baseline are the result; the rest are already recorded in
            # history.json for the search narrative.
            def _score_of(entry: Dict[str, Any]) -> float:
                value = (entry.get("score") or {}).get("value")
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    return float("inf")
                return -value if direction == "max" else value

            promotable = sorted(improving, key=_score_of)[:_OED_MAX_PROMOTED]
            if len(improving) > len(promotable):
                print(
                    f"[oed] promoting the {len(promotable)} best of {len(improving)} "
                    "baseline-beating candidates; the rest remain in history.json",
                    flush=True,
                )

            for idx, entry in enumerate(promotable, 1):
                if not entry.get("valid_case"):
                    continue
                src = Path(str(entry.get("case_dir", "")))
                if not src.is_dir():
                    continue
                case_id = f"case_oed_{idx:03d}"
                dst = cases_root / case_id
                if case_id not in already_promoted:
                    _copy_completed_case(src, dst)
                execution_ok = bool(entry.get("execution_ok")) and _case_has_clean_solver_log(dst)
                _write_json(
                    dst / "run_result.json",
                    {
                        "status": "success" if execution_ok else "failed",
                        "success": execution_ok,
                        "case_id": case_id,
                        "case_dir": str(dst),
                        "source_case_dir": str(src),
                        "oed_status": entry.get("status"),
                        "score": entry.get("score"),
                        "error_logs": [] if execution_ok else ["Candidate execution did not finish with a clean solver log."],
                    },
                )
                promoted_case_ids.append(case_id)
                requirements.append({
                    "case_id": case_id,
                    "user_requirement_text": str(entry.get("model_description", "")),
                    "study_id": "open_discovery",
                    "oed_status": entry.get("status"),
                })
                manifest_entries.append({"case_id": case_id, "case_path": str(dst), "status": "success" if execution_ok else "failed"})

            _write_json(out_dir / "requirements.json", requirements)
            _write_json(out_dir / "manifest.json", {"study_id": out_dir.name, "topic": config.get("topic", ""), "cases": manifest_entries})
            stop_reason = "budget_exhausted" if budget_exhausted else "archive_saturated_with_winner"
            bridge = {
                "provenance": "cfd-open-discovery:bridge:v1",
                "case_ids": promoted_case_ids,
                # interpret_case writes its verdicts back into this dict, so
                # resetting it to {} on a repeat call would throw away every
                # interpretation already made.
                "decisions": (_read_json(out_dir / "bridge.json") or {}).get("decisions") or {},
                "history_path": str(history_path),
                "stop_reason": stop_reason,
            }
            _write_json(out_dir / "bridge.json", bridge)
            artifact = {
                "budget_total": total_budget,
                "budget_used": budget_used,
                "stop_reason": stop_reason,
                "winner_iterations": [h.get("iteration") for h in real_evals if h.get("status") == "PROCEED"],
                "promoted_case_ids": promoted_case_ids,
            }
            _write_json(out_dir / "oed_artifact.json", artifact)
            _write_checkpoint(
                "open_ended_discovery_done",
                {**artifact, "bridge_signature": "cfd-open-discovery:bridge:v1"},
            )
            _update_study_state(mode="open_discovery", current_stage="open_ended_discovery")

        return {
            "budget_used": budget_used,
            "budget_total": total_budget,
            "proceed_count": proceed_count,
            "has_winner": has_winner,
            "budget_exhausted": budget_exhausted,
            "is_saturated": saturated,
            "search_complete": search_complete,
            "archive_summary": _render_archive_summary(archive, disc_dir),
            "missing_candidate_records": missing,
            # Finished candidates found on disk that were not passed in. Named
            # explicitly so a recovered result is visible rather than silently
            # appearing in the archive summary.
            "recovered_unrecorded_candidates": swept,
            "history_path": str(history_path),
            "case_ids_to_interpret": promoted_case_ids,
        }


    # -----------------------------------------------------------------
    # FoamAgent, exposed as individual modules — no scripts/foam_run.py,
    # no Foam-Agent services.* import required for the core loop.
    # Each function below is one stage from cfd-skills/cfd-foamagent/SKILL.md,
    # ported into src/cfd_langgraph/foam_native/ (see that package's
    # module-level docstrings for exactly which prompt each one runs
    # verbatim). Exposed individually so the manager or case-runner can call
    # a single stage on its own — inspect a parsed requirement, regenerate
    # just one file, re-run just the reviewer — rather than only the
    # composed run_case_native below.
    # -----------------------------------------------------------------

    def foam_parse_requirement(requirement_text: str) -> dict:
        """FoamAgent Stage 1: natural-language requirement -> {case_name,
        case_domain, case_category, case_solver}."""
        return foam_native.parser.parse_requirement(foam_llm, requirement_text)

    def foam_retrieve_references(requirement_text: str, case_solver: str, case_domain: str = "", case_category: str = "") -> dict:
        """FoamAgent Stage 2: RAG retrieval over OpenFOAM tutorials (with the
        documented $WM_PROJECT_DIR/tutorials/ fallback if FAISS isn't built)."""
        return foam_native.rag.retrieve_references(requirement_text, case_solver, case_domain, case_category)

    def foam_decompose_subtasks(requirement_text: str, dir_structure: str = "") -> dict:
        """FoamAgent Stage 3: requirement -> list of {file_name, folder_name} subtasks."""
        return {"subtasks": foam_native.decomposer.decompose_subtasks(foam_llm, requirement_text, dir_structure)}

    def foam_write_case_file(
        case_id: str, file_name: str, folder_name: str, requirement_text: str,
        tutorial_reference: str = "", case_solver: str = "simpleFoam",
    ) -> dict:
        """FoamAgent Stage 5: write one case file from scratch, into
        <out_dir>/cases/<case_id>/<folder_name>/<file_name>."""
        target_path, path_error = _safe_case_path(case_id, folder_name, file_name)
        if path_error:
            return {"error": path_error}
        content = foam_native.writer.write_file_initial(
            foam_llm, file_name=file_name, folder_name=folder_name, user_requirement=requirement_text,
            tutorial_reference=tutorial_reference, written_files_ctx="", case_solver=case_solver,
        )
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(content)
        return {"path": str(target_path), "bytes_written": len(content)}

    def foam_generate_allrun(
        case_id: str, case_solver: str, dir_structure: str = "", allrun_reference: str = "", mesh_type: str = "standard_mesh",
    ) -> dict:
        """FoamAgent Stage 6: generate and write the case's Allrun script."""
        case_dir, path_error = _safe_case_path(case_id)
        if path_error:
            return {"error": path_error}
        command_text = foam_native.allrun.generate_allrun_commands(
            foam_llm, dir_structure=dir_structure, case_info={"case_solver": case_solver},
            allrun_reference=allrun_reference, mesh_type=mesh_type,
        )
        allrun_path = case_dir / "Allrun"
        allrun_path.parent.mkdir(parents=True, exist_ok=True)
        allrun_path.write_text(foam_native.allrun.build_allrun_script(command_text))
        allrun_path.chmod(0o755)
        return {"path": str(allrun_path)}

    def foam_review_errors(case_id: str, requirement_text: str) -> dict:
        """FoamAgent Stage 8a: read this case's logs and produce a reviewer
        analysis (diagnosis + proposed fixes, no files touched)."""
        case_dir, path_error = _safe_case_path(case_id)
        if path_error:
            return {"error": path_error}
        analysis = foam_native.review.review_errors(
            foam_llm, tutorial_reference="", foamfiles_xml="",
            error_logs=foam_native.loop.collect_error_logs(case_dir), user_requirement=requirement_text,
        )
        return {"analysis": analysis}

    def run_case_native(
        case_id: str, requirement_text: str, physics_group: str = "default",
        mesh_type: str = "standard_mesh", max_loop: int = 10,
    ) -> dict:
        """Run one OpenFOAM case entirely through this workflow's own
        FoamAgent port (parse -> RAG -> decompose -> write -> Allrun -> run
        -> review/rewrite loop -> run_result.json) — the preferred path; does
        not need Foam-Agent's Python package vendored, only OpenFOAM itself
        (and optionally its prebuilt FAISS tutorial index, for better RAG).

        ``physics_group`` should be the same string for every case sharing a
        mesh/physics shape — the first call per group calibrates concurrency
        for the whole group (see scheduling/CaseCoordinator).
        """
        if not (out_dir / "hypotheses_approved.json").exists():
            return {"error": "Blocked: hypotheses_approved.json is missing — hypotheses have not been approved yet."}

        case_id = _safe_variant_slug(case_id, "case")
        physics_group = _safe_variant_slug(physics_group, "default")
        canonical_requirement, requirement_error = _canonical_requirement(case_id, requirement_text)
        if requirement_error:
            return {"error": requirement_error, "case_id": case_id}
        mesh_spec = _read_json(out_dir / "mesh_gate" / physics_group / "selected_mesh_spec.json") or {}
        selected_level = Path(str(mesh_spec.get("selected_level", "")))
        if not mesh_spec.get("converged") or not selected_level.is_dir():
            return {
                "error": "Blocked: this physics group has no converged mesh-gate selection.",
                "physics_group": physics_group,
                "mesh_spec": str(out_dir / "mesh_gate" / physics_group / "selected_mesh_spec.json"),
            }
        case_dir = out_dir / "cases" / case_id
        result = coordinator.run_case(
            physics_group,
            lambda: foam_native.run_foam_case(
                foam_llm, case_dir, canonical_requirement, mesh_type=mesh_type, max_loop=max_loop,
                openfoam_path=settings.openfoam_path,
                mesh_seed_case_dir=selected_level,
                functions_seed_case_dir=_starter_base_case_dir(),
                # The gate's selected level is itself a converged, validated
                # case carrying the locked mesh, so it — not prose — is the
                # starting point for every experiment case. mesh_seed_case_dir
                # stays: it re-stamps blockMeshDict after the write stage and
                # keeps the reviewer from editing it.
                base_case_seed_dir=selected_level,
            ),
        )
        result["physics_group"] = physics_group
        result["concurrency_at_group"] = coordinator.concurrency_for(physics_group)
        _write_json(case_dir / "run_result.json", result)
        return result


    def run_audit_and_record() -> dict:
        """Run the stage-gate audit for this study; on pass, record it into
        the knowledge bundle (a lessons entry, plus a new validation-suite
        case future self-evolution proposals must not regress on)."""
        proc = subprocess.run(
            [sys.executable, "scripts/stage_gate_audit.py", "--out-dir", str(out_dir), "--json"],
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
        )
        passed = proc.returncode == 0
        record = bundle.record_study(out_dir) if passed else None
        if passed:
            _update_study_state(current_stage="finish", status="done")
        return {
            "audit_passed": passed,
            "returncode": proc.returncode,
            "stdout_tail": proc.stdout[-2000:],
            "stderr_tail": proc.stderr[-2000:],
            "knowledge_bundle_entry": record,
        }

    # -----------------------------------------------------------------
    # OED candidate-runner tools — go on the oed-candidate-runner subagent,
    # never on the manager directly, so N of these run concurrently as N
    # separate `task` calls instead of serially inside one Python loop.
    # -----------------------------------------------------------------

    def oed_run_code_mod_candidate(
        topic: str, variant_name: str, hypothesis: str, plan: str = "",
        strategy: str = ""
    ) -> dict:
        """One OED candidate: compile and run a NEW custom OpenFOAM model
        class implementing `hypothesis` — the same agentic code-mod runner
        open_ended_discovery.py itself uses (reads real OpenFOAM
        turbulence-model source, writes+compiles the class, runs it). Costs
        2 budget units. Requires oed_setup_search to have already run for
        this study. This only compiles/runs — call oed_score_candidate on
        the returned case_dir next, it does not score on its own.

        ``plan`` is optional and carries the strategy's own steps when the
        candidate needs more than "implement this hypothesis" — which data to
        read, which fit or optimiser to run, what the fitted result becomes.
        Pass through whatever oed_propose_candidates returned for this
        candidate. The agent has a shell and the study's libraries, so a plan
        may legitimately ask it to fit before it writes any model code.

        ``strategy`` is the search strategy this candidate belongs to, passed
        through from oed_propose_candidates. It sets the wall-clock fence
        against other candidates of the SAME strategy rather than against the
        whole pool, so a solver-in-the-loop fit is not held to the pace of an
        analytic one-shot. Pass it whenever you have it.

        If the build agent does not finish cleanly this does NOT quietly hand
        back whatever is on disk. It diagnoses what stopped it and whether the
        model is actually complete, and returns that under
        ``unclean_finish_diagnosis`` for you to act on -- score it, repair it
        with oed_apply_repair, extend it with oed_extend_candidate, or drop
        it. Read the verdict before you do anything else with the result."""
        disc_dir = _oed_disc_dir()
        config = _read_json(disc_dir / "search_config.json") or {}
        topic = _effective_topic(topic)
        if str(topic or "").strip() != str(config.get("topic", "") or "").strip():
            # Same reasoning as oed_propose_candidates: byte-exact on purpose,
            # but a refusal that does not say what to match is unguessable.
            return {
                "ok": False,
                "error": (
                    "Candidate topic does not match the topic this search was locked to "
                    "at oed_setup_search. Pass the locked_topic below verbatim."
                ),
                "locked_topic": str(config.get("topic", "") or ""),
                "received_topic": str(topic or ""),
            }
        starter_case = str(config.get("baseline_case_dir", "")).strip()
        if not starter_case or not Path(starter_case).is_dir():
            return {"ok": False, "error": "OED search has no valid locked-mesh baseline case."}
        if not str(hypothesis or "").strip():
            return {"ok": False, "error": "Candidate hypothesis is empty."}
        variant_name = _safe_variant_slug(variant_name, "candidate")
        candidate_dir = disc_dir / f"cand_{variant_name}"
        output_path = candidate_dir / "agentic_result.json"

        # A finished candidate is never built twice.
        #
        # LangGraph checkpoints the model's tool-call message before the tool
        # results exist, so an interrupt anywhere in a batch replays EVERY
        # call in that batch on resume. Measured on run closure_20260826_codex:
        # `resume` re-issued all four candidates, one of which
        # (sst_a1_limiter_025) had already compiled its library and solved its
        # case to t=30000 in 1525s. Rebuilding it would have thrown that away
        # and paid for it a second time, and the same replay is what stranded
        # sst_crossdiff_scale_065 -- the study's best model -- earlier that day.
        #
        # Keyed on the hypothesis and plan, not just the directory: a variant
        # name can legitimately be re-proposed with different instructions, and
        # returning a stale build for a changed hypothesis would be a far worse
        # failure than rebuilding. Same instructions, finished result, reuse;
        # anything else runs.
        invocation_path = candidate_dir / "candidate_invocation.json"
        invocation = {
            "hypothesis": str(hypothesis or ""),
            "plan": str(plan or ""),
            "variant_name": variant_name,
        }
        # Kept out of the reuse key on purpose: re-labelling a candidate's
        # strategy is not a change to what was asked of the build agent, so it
        # must not invalidate a finished build. Written alongside it instead,
        # where _candidate_strategy can find it for the per-strategy fence.
        strategy_note = {"strategy": str(strategy or "").strip()}
        finished = _read_json(output_path) or {}
        if not finished and not output_path.exists():
            # Killed after the solver finished, before the verdict was written.
            #
            # This is the gap the reuse test above cannot close: on run
            # closure_20260826_codex, sst_a1_limiter_025 compiled its library
            # and solved to t=30000 in 1525s, and the process died before
            # agentic_result.json existed. There is no verdict to trust, but
            # there is unambiguous evidence of one: a compiled library, and a
            # solver log that reached "End" with no FOAM FATAL.
            #
            # Only when the file is ABSENT. A recorded "FAILED" is the agent's
            # own verdict on work it watched, and a case can reach "End" while
            # producing a diverged solution -- sst_nonlinear_rs_weak_nn ran all
            # 32 cases to completion at 5.2e52. Overturning a recorded failure
            # from the outside would launder exactly that into a success.
            rebuilt_case = candidate_dir / variant_name
            libs = sorted(rebuilt_case.glob("customModels/*/lib/*.so"))
            if len(libs) == 1 and _case_has_clean_solver_log(rebuilt_case):
                model_name = _selected_turbulence_model(rebuilt_case) or libs[0].stem.removeprefix("lib")
                finished = {
                    "status": "OK",
                    "case_dir": str(rebuilt_case),
                    "compile_ok": True,
                    "converged": True,
                    "compiled_model_name": model_name,
                    "compiled_model_description": str(hypothesis or ""),
                    "compiled_case_dir": str(rebuilt_case),
                    "compiled_so": str(libs[0]),
                    "reconstructed_from": "compiled library and clean solver log",
                }
                _write_json(output_path, finished)
                print(
                    f"[oed] {variant_name}: no agentic_result.json, but a compiled "
                    f"library and a clean solver log are on disk; recovered instead "
                    f"of rebuilding.",
                    flush=True,
                )
                # A reconstruction is a guess about a process that died before
                # it could say what it had done. The evidence it rests on -- a
                # library that compiled, a solver log that reached End -- is
                # equally consistent with a finished model and with one whose
                # coefficient was still being fitted when the process died,
                # and the second scores as the baseline while looking healthy.
                # The status written above says OK, which would satisfy the
                # reuse test below and skip every check after it, so the
                # completeness question is asked HERE instead.
                try:
                    reconstruction_verdict = _diagnose_unclean_finish(
                        candidate_dir, finished, 0, _OED_MAX_TURNS, 0, settings,
                    )
                except Exception as exc:
                    reconstruction_verdict = {
                        "ok": False,
                        "verdict": "unknown",
                        "cause": f"diagnosis raised {type(exc).__name__}: {exc}",
                    }
                _write_json(candidate_dir / "unclean_finish_diagnosis.json",
                            reconstruction_verdict)
        if finished.get("status") == "OK":
            previous = _read_json(invocation_path) or {}
            if all(previous.get(k) == v for k, v in invocation.items()):
                standing = _read_json(candidate_dir / "unclean_finish_diagnosis.json") or {}
                return {
                    "ok": True,
                    "reused": True,
                    **({"unclean_finish_diagnosis": standing} if standing else {}),
                    "note": (
                        "Already built and run under identical instructions; the "
                        "finished result on disk was returned instead of rebuilding. "
                        "Score and record it exactly as if it had just run."
                    ),
                    "candidate_dir": str(candidate_dir),
                    "case_dir": finished.get("case_dir", ""),
                    "compile_ok": bool(finished.get("compile_ok")),
                    "converged": bool(finished.get("converged")),
                    "compiled_model_name": finished.get("compiled_model_name", ""),
                    "compiled_model_description": finished.get("compiled_model_description", ""),
                    "compiled_case_dir": finished.get("compiled_case_dir", ""),
                    "compiled_so": finished.get("compiled_so", ""),
                    "compile_error_hint": "",
                    "stderr_tail": "",
                }
        # Written BEFORE the run, so a candidate killed mid-build is not
        # mistaken on resume for one that finished under these instructions --
        # the status=="OK" test above is what gates reuse, and this only says
        # which instructions produced whatever is there.
        candidate_dir.mkdir(parents=True, exist_ok=True)
        _write_json(invocation_path, {**invocation, **strategy_note})

        # Fenced against this candidate's OWN strategy where possible. See
        # _oed_candidate_timeout: a pooled fence converges on whichever
        # strategy is fastest and then kills every slower one permanently.
        # Clamped to the same ceiling an extension is. The fence is an outlier
        # bound over observed durations, and on a strategy whose successes are
        # spread wide the log-Tukey bound stops bounding anything: measured on
        # run closure_20260826_codex, `offline_fit` reached 96721s -- 26.9
        # hours -- from four successes of very different cost. Handing that to
        # an agent is worse than useless now that the agent plans against the
        # number it is given: it would budget a 27-hour fit and be killed by
        # the subprocess timeout hours earlier, which is precisely the failure
        # the budget block exists to prevent.
        granted_timeout = min(_oed_candidate_timeout(disc_dir, strategy) or 0,
                              _OED_MAX_EXTENDED_S)

        # Through the coordinator, exactly like run_case_native: the manager
        # is told to launch a whole batch of candidates as concurrent `task`
        # calls, and each of these compiles a library and then runs a full
        # OpenFOAM case. Called directly, four candidates meant four
        # simultaneous wmake builds plus four solvers with nothing throttling
        # them — the one thing CaseCoordinator exists to prevent.
        proc = coordinator.run_case(
            "oed_candidates",
            lambda: _run_script(
                [
                    "scripts/code_mod_agentic.py",
                    "--hypothesis", hypothesis,
                    *(["--plan", plan] if str(plan or "").strip() else []),
                    "--variant-name", variant_name,
                    "--run-dir", str(candidate_dir),
                    "--starter-case", starter_case,
                    "--topic", topic,
                    "--output", str(output_path),
                    # A real wall-clock fence. "0" disables the cap in
                    # code_mod_agentic.py and leaves --max-turns as the only
                    # bound -- but a turn may run a full solver, so turns are
                    # not a proxy for time. Measured on run
                    # oed_20260822_1626_codex_high: candidate
                    # sa_sr_destruction ran 35 simpleFoam solves over 92
                    # minutes doing its own private coefficient sweep, while
                    # three finished candidates sat blocked behind it in the
                    # batch and the archive recorded the whole thing as one
                    # evaluation of cost 2. Typical candidates finish in
                    # 400-1000s; 45 minutes leaves generous headroom for a
                    # genuinely slow compile-and-run while bounding a runaway.
                    "--timeout", str(granted_timeout),
                    "--max-turns", str(_OED_MAX_TURNS),
                ],
                # Must sit ABOVE the fence the agent was told about, or the
                # agent plans against one deadline and dies at another. Headroom
                # is for the tail of a solver launch already in flight.
                timeout=max(10800, granted_timeout + 1800),
                env=_foamagent_env(settings.openfoam_path),
            ),
        )
        result = _read_json(output_path) or {}
        if result.get("status") == "OK":
            # This attempt finished cleanly, so any verdict describing an
            # earlier one no longer applies. Left in place it would keep
            # oed_score_candidate refusing a candidate that is now fine.
            stale_verdict = candidate_dir / "unclean_finish_diagnosis.json"
            if stale_verdict.exists():
                try:
                    stale_verdict.unlink()
                except OSError:
                    pass
        payload = {
            "ok": proc.returncode == 0 and result.get("status") == "OK",
            "candidate_dir": str(candidate_dir),
            "case_dir": result.get("case_dir", ""),
            "compile_ok": bool(result.get("compile_ok")),
            "converged": bool(result.get("converged")),
            "compiled_model_name": result.get("compiled_model_name", ""),
            "compiled_model_description": result.get("compiled_model_description", ""),
            "compiled_case_dir": result.get("compiled_case_dir", ""),
            "compiled_so": result.get("compiled_so", ""),
            "compile_error_hint": result.get("compile_error_hint", ""),
            "stderr_tail": proc.stderr[-1500:] if proc.returncode != 0 else "",
            "granted_timeout_s": granted_timeout,
            "fenced_against_strategy": str(strategy or "") or "(pooled)",
        }
        # An agent that did not finish cleanly gets read before its work is
        # accepted. Without this the caller sees only compile_ok/converged,
        # both of which are true for a half-fitted model that will score as
        # the baseline -- see _diagnose_unclean_finish for the six candidates
        # that did exactly that.
        if result.get("status") != "OK":
            attempts = _read_json(candidate_dir / "candidate_attempts.json") or {}
            try:
                diagnosis = _diagnose_unclean_finish(
                    candidate_dir, result, granted_timeout, _OED_MAX_TURNS,
                    int(attempts.get("extensions_used", 0) or 0), settings,
                )
            except Exception as exc:
                diagnosis = {
                    "ok": False,
                    "verdict": "unknown",
                    "cause": f"diagnosis raised {type(exc).__name__}: {exc}",
                }
            payload["unclean_finish_diagnosis"] = diagnosis
            # Persisted, not just returned. A prompt telling the runner not to
            # score an incomplete model is advice; oed_score_candidate reading
            # this file is the same judgement actually taking effect. Cleared
            # by _rerun_build_agent, because a later attempt supersedes it.
            _write_json(candidate_dir / "unclean_finish_diagnosis.json", diagnosis)
            payload["next_step"] = (
                "This agent did NOT finish cleanly. Read unclean_finish_diagnosis "
                "before anything else. verdict=complete -> score it as normal. "
                "verdict=repair -> call oed_apply_repair with its repair_steps. "
                "verdict=extend -> call oed_extend_candidate with its "
                "extra_seconds_needed. verdict=abandon -> record it null and move on. "
                "Do NOT score a candidate whose model_is_complete is false: it will "
                "return the baseline and be recorded as a real experiment."
            )
        return payload

    def _rerun_build_agent(candidate_dir: str, *, timeout_s: int,
                           prior_attempt: str = "", repair_goal: str = "") -> dict:
        """Run the build agent again over a candidate that already exists.

        Shared by the extension and the repair paths because they differ only
        in what the agent is told: a continuation gets the story of the last
        attempt and is asked to carry on, a repair gets a specific defect and
        is forbidden from doing anything else. Both reuse the candidate's own
        directory, so the compiled library, the fit artifacts and the case are
        all still there to build on.
        """
        disc_dir = _oed_disc_dir()
        candidate_path = Path(candidate_dir).expanduser().resolve()
        if candidate_path.parent != disc_dir.resolve() or not candidate_path.name.startswith("cand_"):
            return {"ok": False, "error": "candidate_dir must be a direct cand_* child of this study's OED directory."}
        invocation = _read_json(candidate_path / "candidate_invocation.json") or {}
        hypothesis = str(invocation.get("hypothesis") or "")
        if not hypothesis.strip():
            return {"ok": False, "error": "No candidate_invocation.json on disk; cannot tell what this candidate was asked to do."}
        config = _read_json(disc_dir / "search_config.json") or {}
        starter_case = str(config.get("baseline_case_dir", "")).strip()
        if not starter_case or not Path(starter_case).is_dir():
            return {"ok": False, "error": "OED search has no valid locked-mesh baseline case."}
        variant_name = str(invocation.get("variant_name") or candidate_path.name.removeprefix("cand_"))
        output_path = candidate_path / "agentic_result.json"
        # The standing verdict described the PREVIOUS attempt. Whatever this
        # run produces gets judged on its own evidence; leaving the old file
        # in place would have oed_score_candidate refuse a candidate that has
        # just been repaired.
        stale = candidate_path / "unclean_finish_diagnosis.json"
        if stale.exists():
            try:
                stale.unlink()
            except OSError:
                pass
        proc = coordinator.run_case(
            "oed_candidates",
            lambda: _run_script(
                [
                    "scripts/code_mod_agentic.py",
                    "--hypothesis", hypothesis,
                    *(["--plan", str(invocation.get("plan") or "")]
                      if str(invocation.get("plan") or "").strip() else []),
                    "--variant-name", variant_name,
                    "--run-dir", str(candidate_path),
                    "--starter-case", starter_case,
                    "--topic", str(config.get("topic", "") or ""),
                    "--output", str(output_path),
                    "--timeout", str(int(timeout_s)),
                    "--max-turns", str(_OED_MAX_TURNS),
                    *(["--prior-attempt", prior_attempt] if prior_attempt.strip() else []),
                    *(["--repair-goal", repair_goal] if repair_goal.strip() else []),
                ],
                # The agent's own wall clock plus headroom for the tail of a
                # solver launch already in flight, never above the ceiling.
                timeout=min(_OED_MAX_EXTENDED_S + 1800, max(10800, int(timeout_s) + 1800)),
                env=_foamagent_env(settings.openfoam_path),
            ),
        )
        result = _read_json(output_path) or {}
        return {
            "ok": proc.returncode == 0 and result.get("status") == "OK",
            "candidate_dir": str(candidate_path),
            "case_dir": result.get("case_dir", ""),
            "status": result.get("status", ""),
            "compile_ok": bool(result.get("compile_ok")),
            "converged": bool(result.get("converged")),
            "compiled_model_name": result.get("compiled_model_name", ""),
            "compiled_model_description": result.get("compiled_model_description", ""),
            "compiled_case_dir": result.get("compiled_case_dir", ""),
            "compiled_so": result.get("compiled_so", ""),
            "aborted_reason": result.get("aborted_reason", ""),
            "turns_used": result.get("turns_used", 0),
            "duration_s": result.get("duration_s", 0),
            "stderr_tail": proc.stderr[-1500:] if proc.returncode != 0 else "",
        }

    def oed_extend_candidate(candidate_dir: str, extra_seconds: int,
                             rationale: str) -> dict:
        """Give a candidate that ran out of time more time, and carry on.

        Call this when oed_run_code_mod_candidate came back with an
        unclean_finish_diagnosis whose verdict is "extend": the agent was doing
        real work and the wall clock stopped it. Pass the diagnosis's own
        extra_seconds_needed and quote its estimate_basis as the rationale.

        This is not a re-run from scratch. The agent is told what the previous
        attempt achieved and is asked to inspect its own directory and continue
        from there, so a library it already compiled and optimiser iterations it
        already paid for are not repeated. Its trajectory log is appended to
        rather than overwritten, so the whole history stays readable.

        Two extensions per candidate. That is deliberate: a candidate needing a
        third has been mis-estimated twice, and the budget is better spent on a
        different mechanism than on one that keeps not finishing.
        """
        disc_dir = _oed_disc_dir()
        candidate_path = Path(candidate_dir).expanduser().resolve()
        if candidate_path.parent != disc_dir.resolve() or not candidate_path.name.startswith("cand_"):
            return {"ok": False, "error": "candidate_dir must be a direct cand_* child of this study's OED directory."}
        attempts_path = candidate_path / "candidate_attempts.json"
        attempts = _read_json(attempts_path) or {}
        used = int(attempts.get("extensions_used", 0) or 0)
        if used >= _OED_EXTENSION_ATTEMPTS:
            return {
                "ok": False,
                "error": (
                    f"This candidate has already been extended {used} time(s), which is "
                    f"the limit of {_OED_EXTENSION_ATTEMPTS}. Score what is on disk if "
                    f"its model is complete, otherwise record it null and move on."
                ),
                "extensions_used": used,
            }
        try:
            extra = int(extra_seconds)
        except (TypeError, ValueError):
            return {"ok": False, "error": "extra_seconds must be an integer number of seconds."}
        if extra <= 0:
            return {"ok": False, "error": "extra_seconds must be positive."}
        if not str(rationale or "").strip():
            return {
                "ok": False,
                "error": (
                    "rationale is required: state the arithmetic behind the estimate "
                    "(how much work remains, how long each unit has been taking). An "
                    "extension without one is a guess, and guesses are what the fence "
                    "exists to stop."
                ),
            }
        previous = _read_json(candidate_path / "agentic_result.json") or {}
        base = int(previous.get("duration_s") or 0)
        granted = base + extra
        clipped = granted > _OED_MAX_EXTENDED_S
        granted = min(granted, _OED_MAX_EXTENDED_S)
        prior_attempt = (
            f"An earlier attempt ran {previous.get('turns_used', '?')} turns over "
            f"{base}s and stopped: {previous.get('aborted_reason') or previous.get('error') or 'reason not recorded'}.\n"
            f"It made {previous.get('solver_invocations', '?')} solver launches, "
            f"compile_ok={previous.get('compile_ok')}, converged={previous.get('converged')}.\n"
            f"You have been granted {granted}s in total for this continuation, on this "
            f"reasoning: {str(rationale).strip()}"
        )
        # Counted BEFORE the run, for the same reason oed_note_repair_attempt
        # is: a continuation that dies mid-way must still consume its attempt,
        # or a crash loop resets the count and the budget is unenforceable.
        attempts["extensions_used"] = used + 1
        attempts.setdefault("extension_log", []).append({
            "attempt": used + 1,
            "extra_seconds": extra,
            "total_granted_s": granted,
            "rationale": str(rationale)[:2000],
        })
        _write_json(attempts_path, attempts)
        outcome = _rerun_build_agent(str(candidate_path), timeout_s=granted,
                                     prior_attempt=prior_attempt)
        outcome["extensions_used"] = used + 1
        outcome["extensions_remaining"] = max(0, _OED_EXTENSION_ATTEMPTS - (used + 1))
        outcome["total_granted_s"] = granted
        if clipped:
            outcome["note"] = (
                f"The request came to {base + extra}s, above the {_OED_MAX_EXTENDED_S}s "
                f"ceiling on any single build, so {granted}s was granted instead. A "
                f"candidate that genuinely needs more than that is too expensive for "
                f"one slot -- narrow the work (fewer fit cases, a coarser optimiser "
                f"budget) rather than asking for the time again."
            )
        return outcome

    def oed_apply_repair(candidate_dir: str, repair_steps: List[str],
                         rationale: str) -> dict:
        """Carry out a diagnosed repair on a candidate, then re-run its case.

        Call this when a diagnosis returned verdict "repair" with concrete
        steps -- a library that is not being loaded, a coefficient computed but
        never written into the case, a missing dictionary entry. This is the
        tool that actually makes the change: the candidate-runner subagent has
        no filesystem write access of its own, by design, so without this a
        diagnosis that correctly identified a one-line fix could only report it
        and stop. That is exactly what happened to
        lrr_pressure_strain_offline_fit on run closure_20260826_codex, whose 32
        cases all failed on a library that never loaded.

        The repair agent works inside the candidate's own directory and is told
        it may fix our plumbing but may not touch the mesh, the boundary
        conditions, the physics, the endTime, or the closure under test -- a
        "repair" that changes any of those is a different experiment. Refuses
        outright if the diagnosis itself said the repair would cross that line.

        Counts against the same two-attempt repair budget as
        oed_note_repair_attempt, and records the attempt before running so a
        crash mid-repair cannot reset the count.
        """
        disc_dir = _oed_disc_dir()
        candidate_path = Path(candidate_dir).expanduser().resolve()
        if candidate_path.parent != disc_dir.resolve() or not candidate_path.name.startswith("cand_"):
            return {"ok": False, "error": "candidate_dir must be a direct cand_* child of this study's OED directory."}
        steps = [str(x).strip() for x in (repair_steps or []) if str(x).strip()]
        if not steps:
            return {"ok": False, "error": "repair_steps is empty; there is nothing to carry out."}
        if not str(rationale or "").strip():
            return {"ok": False, "error": "rationale is required: say what is broken and why these steps fix it."}
        record = _read_json(candidate_path / "candidate_record.json") or {}
        used = int(record.get("repair_attempts", 0) or 0)
        if used >= _OED_REPAIR_ATTEMPTS:
            return {
                "ok": False,
                "error": (
                    f"This candidate has used all {_OED_REPAIR_ATTEMPTS} repair attempts. "
                    f"Record it null and move on."
                ),
                "repair_attempts_used": used,
            }
        goal = (
            f"Why this candidate is broken: {str(rationale).strip()}\n\n"
            "Steps to carry out, in order:\n"
            + "\n".join(f"  {i}. {step}" for i, step in enumerate(steps, 1))
        )
        record["repair_attempts"] = used + 1
        record.setdefault("repair_log", []).append({
            "attempt": used + 1,
            "change": goal[:2000],
            "applied_by": "oed_apply_repair",
        })
        _write_json(candidate_path / "candidate_record.json", record)
        previous = _read_json(candidate_path / "agentic_result.json") or {}
        # A repair is bounded work on something that already exists, so it is
        # fenced at the strategy's own pace rather than given a fresh full
        # build's worth of clock.
        timeout_s = _oed_candidate_timeout(disc_dir, _candidate_strategy(candidate_path)) or \
            max(1800, int(previous.get("duration_s") or 0))
        outcome = _rerun_build_agent(str(candidate_path), timeout_s=timeout_s,
                                     repair_goal=goal)
        outcome["repair_attempts_used"] = used + 1
        outcome["repair_attempts_remaining"] = max(0, _OED_REPAIR_ATTEMPTS - (used + 1))
        outcome["next_step"] = (
            "Re-run oed_run_evaluation_cases then oed_score_candidate. If the repair "
            "agent said it could not make the change without altering the graded setup, "
            "do NOT try another route: record the candidate null and move on."
        )
        return outcome

    def oed_run_evaluation_cases(candidate_dir: str, case_dir: str) -> dict:
        """Run this candidate's compiled model on every evaluation case the
        study declared, so it is scored on the whole set rather than on the
        one case it was built against.

        Call this after oed_run_code_mod_candidate and BEFORE
        oed_score_candidate, but only when the study declares
        ``evaluation_cases`` in search_config.json — for a single-case study
        it is unnecessary and returns immediately saying so.

        The model is copied into a private replica of each evaluation case,
        never into the case itself, so the declared cases stay pristine and
        two candidates can never contaminate each other. Runs are throttled
        through the same coordinator that governs every other solver launch.
        """
        disc_dir = _oed_disc_dir()
        evaluation_cases = _evaluation_cases(disc_dir)
        if not evaluation_cases:
            return {
                "ok": True,
                "multi_case": False,
                "note": (
                    "This study declares no evaluation_cases, so scoring uses the "
                    "candidate's own case. Nothing to do."
                ),
            }

        candidate_path = Path(candidate_dir).expanduser().resolve()
        if candidate_path.parent != disc_dir.resolve() or not candidate_path.name.startswith("cand_"):
            return {"ok": False, "error": "candidate_dir must be a direct cand_* child of this study's OED directory."}
        source_case = Path(case_dir).expanduser().resolve()
        if not source_case.is_dir():
            return {"ok": False, "error": f"case_dir not found: {source_case}"}

        result_doc = _read_json(candidate_path / "agentic_result.json") or {}
        model_name = str(result_doc.get("compiled_model_name") or "").strip()
        if not model_name:
            return {"ok": False, "error": "No compiled_model_name on the candidate's execution result."}

        eval_root = candidate_path / "evaluation_cases"
        eval_root.mkdir(parents=True, exist_ok=True)

        # Replicas mirror the declared cases' own directory structure, not a
        # flat list of basenames. A comparator that scores a SET of cases finds
        # them by walking up from the case it is handed and mapping each
        # reference case across by relative path — the authored 32-case
        # comparator in run closure_20260826_codex looks for `training/` and
        # `validation/` above the case and returns NaN when it cannot find
        # them. Flattening to basenames breaks that mapping and makes every
        # candidate unscoreable, so the shape is preserved.
        common_root = None
        try:
            common_root = Path(os.path.commonpath([str(c) for c in evaluation_cases]))
        except ValueError:
            common_root = None

        prepared: List[Tuple[str, Path]] = []
        failures: List[Dict[str, str]] = []
        for case in evaluation_cases:
            name = case.name
            if common_root is not None:
                try:
                    relative = case.relative_to(common_root)
                except ValueError:
                    relative = Path(case.name)
            else:
                relative = Path(case.name)
            replica = eval_root / relative
            if replica.exists():
                shutil.rmtree(replica)
            replica.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copytree(case, replica, symlinks=True)
            except OSError as exc:
                failures.append({"case": name, "error": f"copy failed: {exc}"})
                continue
            # Start every evaluation case from its own initial condition, not
            # from a converged field the benchmark happened to ship: a
            # candidate scored from a restart of the BASELINE solution is
            # measuring how far its model moves an already-converged baseline,
            # which is not the same quantity as solving the case with it.
            for child in list(replica.iterdir()):
                if child.is_dir() and child.name not in {"0", "constant", "system"}:
                    try:
                        float(child.name)
                    except ValueError:
                        continue
                    if child.name != "0":
                        shutil.rmtree(child, ignore_errors=True)
            error = _install_model_into_case(source_case, replica, model_name)
            if error:
                failures.append({"case": name, "error": error})
                continue
            prepared.append((name, replica))

        if not prepared:
            return {"ok": False, "error": "No evaluation case could be prepared.", "failures": failures}

        # Every case is offered to the coordinator at once, from its own
        # thread, and the coordinator decides how many actually run.
        #
        # This loop used to be a blocking `for` over prepared cases. Sharing
        # one coordinator group across the cases -- an earlier fix -- did
        # nothing on its own, because a blocking loop never offers a second
        # case for the group's limit to apply to. Measured on run
        # closure_20260826_codex: four candidates evaluating simultaneously
        # produced exactly four concurrent simpleFoam processes on a 128-core
        # machine at load 4.2, one per candidate, and a candidate was on track
        # to take ~15 hours to walk its 32 cases one at a time.
        #
        # The number of concurrent runs is NOT chosen here. CaseCoordinator
        # calibrates the group on its first case and derives the limit from
        # min(cores available / cores per case, memory available / memory per
        # case) with a safety margin -- see scheduling/scheduler.py. Threads
        # beyond that limit sit blocked inside run_case, which is why the pool
        # can be as wide as the case list: a blocked thread costs a stack, not
        # a solver.
        ran_by_name: Dict[str, Dict[str, Any]] = {}
        # A one-element list, not an int: the worker below has to mutate this
        # and read it under the lock, and rebinding an int in a closure would
        # make each worker's increment invisible to the others -- the abandon
        # gate would then read a permanent zero and never fire.
        timeouts = [0]
        eval_lock = threading.Lock()

        def _evaluate_one(name: str, replica: Path) -> Dict[str, Any]:
            # A closure slow enough to blow the fence on one case will blow it
            # on all of them. At an hour apiece that is 32 hours to learn what
            # the first two cases already said, and the budget is measured in
            # solver runs. Cases already running are left to finish; ones that
            # have not started are not started.
            with eval_lock:
                if timeouts[0] >= _OED_EVAL_TIMEOUT_ABANDON:
                    already = timeouts[0]
                    return {
                        "case": name,
                        "case_dir": str(replica),
                        "ok": False,
                        "solved_time": None,
                        "error": (
                            f"not attempted: {already} earlier case(s) exceeded the "
                            f"{_OED_EVAL_CASE_TIMEOUT_S}s solver fence, so this model is "
                            "too slow to evaluate on the full set."
                        ),
                    }
            application = _case_application(replica) or "simpleFoam"
            proc = coordinator.run_case(
                # ONE group for every evaluation case in the study, not one
                # per case. The coordinator calibrates each new group
                # exclusively, so a unique group per case would make every case
                # an exclusive calibration run and serialize all 32 no matter
                # how wide the pool above is.
                "oed_evaluation",
                lambda: _run_one_evaluation_case(
                    replica, application, _foamagent_env(settings.openfoam_path)
                ),
            )
            if getattr(proc, "timed_out", False):
                # A timeout is a property of THIS candidate's model, not of the
                # study, so it is recorded against the case and the other cases
                # continue -- raising here tore down the whole manager step and
                # dropped a candidate that had already cost an hour of compile
                # time. Third time this exception class has ended a step:
                # grep_files and _run_script were fixed the same way.
                with eval_lock:
                    timeouts[0] += 1
            reached_end = "End" in (proc.stdout or "")[-2000:]
            # Reaching End is not the same as producing something to score. A
            # case whose writeInterval never fires within endTime runs happily
            # to completion and writes no time directory at all, leaving only
            # the initial condition. Observed while testing this loop: three
            # cases returned ok with nothing but 0/ on disk, which the scorer
            # would then have had to reject one by one — or, worse, scored the
            # initial field as if it were a result.
            solved_time = _latest_solved_time(replica)
            ok = proc.returncode == 0 and reached_end and solved_time is not None
            if proc.returncode == 0 and reached_end and solved_time is None:
                failure = (
                    "solver reached End but wrote no time directory: nothing to score. "
                    "Check that writeInterval fires within endTime for this case."
                )
            else:
                failure = "" if ok else _openfoam_failure_reason(proc)
            return {
                "case": name,
                "case_dir": str(replica),
                "ok": ok,
                "solved_time": solved_time,
                "error": failure,
            }

        with ThreadPoolExecutor(max_workers=min(len(prepared), 64)) as pool:
            futures = {
                pool.submit(_evaluate_one, name, replica): name
                for name, replica in prepared
            }
            for future in futures:
                name = futures[future]
                try:
                    ran_by_name[name] = future.result()
                except Exception as exc:
                    # One case's crash is that case's result, not the end of
                    # the evaluation. Anything raised here would otherwise
                    # propagate out of the `with` and discard every sibling
                    # case that had already finished.
                    ran_by_name[name] = {
                        "case": name,
                        "case_dir": "",
                        "ok": False,
                        "solved_time": None,
                        "error": f"evaluation raised {type(exc).__name__}: {exc}",
                    }

        timed_out_cases = timeouts[0]
        # Declared order, not completion order: the caller reads this list
        # alongside the declared case list.
        ran: List[Dict[str, Any]] = [
            ran_by_name[name] for name, _replica in prepared if name in ran_by_name
        ]

        succeeded = [r for r in ran if r["ok"]]
        abandoned = timed_out_cases >= _OED_EVAL_TIMEOUT_ABANDON

        # These solver launches go through subprocess.run here, not through the
        # candidate agent's run_bash, so its own counter never sees them. Left
        # unrecorded, a candidate that solved 32 evaluation cases is charged only
        # for the handful of runs it made while building the model — which
        # understates a multi-case candidate by an order of magnitude and makes
        # strategies incomparable, the exact failure measured cost exists to fix.
        # Written where oed_score_candidate reads it, so the charge stays with
        # the candidate even if the study is resumed.
        _write_json(
            candidate_path / "evaluation_run_result.json",
            {
                "solver_invocations": len(ran),
                "cases_declared": len(evaluation_cases),
                "cases_succeeded": len(succeeded),
                "case_dirs": [r["case_dir"] for r in succeeded],
                # WHY each case failed, kept rather than discarded. Without
                # this a null score is a dead end: the reason every case gave
                # was computed here, used for one boolean, and dropped, so
                # nothing downstream could tell "the solver diverged on three
                # cases" from "the solver was never launched" from "it ran
                # fine and wrote no output". Diagnosis needs the reason, and a
                # human reading the run does too.
                "failures": [
                    {"case": r["case"], "case_dir": r["case_dir"], "error": r["error"]}
                    for r in ran if not r["ok"]
                ],
                "timed_out_cases": timed_out_cases,
                "abandoned": abandoned,
            },
        )

        return {
            "ok": bool(succeeded) and not abandoned,
            "multi_case": True,
            "abandoned_for_slowness": abandoned,
            "timed_out_cases": timed_out_cases,
            "solver_invocations": len(ran),
            "cases_run": len(succeeded),
            "cases_declared": len(evaluation_cases),
            "results": ran,
            "failures": failures,
            "case_dirs": [r["case_dir"] for r in succeeded],
            "note": (
                "Pass these case_dirs to oed_score_candidate, which will score each "
                "and report their mean as this candidate's score."
                if succeeded else "No evaluation case ran successfully."
            ),
        }

    def oed_run_experiment_candidate(variant_name: str, base_case_dir: str, parameters: Dict[str, float]) -> dict:
        """One OED candidate: reuse an ALREADY-compiled model — base_case_dir
        is a case_dir a prior oed_run_code_mod_candidate compiled into — with
        new coefficients, no recompile. Costs 1 budget unit. Copies
        base_case_dir, patches constant/fvModels from `parameters`, runs the
        case. Call oed_score_candidate on the returned case_dir next."""
        disc_dir = _oed_disc_dir()
        variant_name = _safe_variant_slug(variant_name, "candidate")
        candidate_dir = disc_dir / f"cand_{variant_name}"
        src = Path(base_case_dir).expanduser().resolve()
        if not src.is_dir():
            return {"ok": False, "error": f"base_case_dir not found: {base_case_dir}"}
        history = _read_json(disc_dir / "history.json") or []
        allowed_parents = {
            str(Path(str(entry.get("case_dir", ""))).resolve())
            for entry in history
            if isinstance(entry, dict)
            and entry.get("action_type") in {"code_mod", "experiment"}
            and entry.get("execution_ok")
            and entry.get("valid_case")
            and entry.get("case_dir")
        }
        if str(src) not in allowed_parents:
            return {
                "ok": False,
                "error": "base_case_dir is not a valid scored parent in this search history.",
            }
        if not parameters:
            return {"ok": False, "error": "Coefficient experiment has no parameter overrides."}
        invalid_parameters = [
            str(key)
            for key, value in parameters.items()
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", str(key))
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ]
        if invalid_parameters:
            return {
                "ok": False,
                "error": "Coefficient names must be identifiers and values must be finite numbers.",
                "invalid_parameters": invalid_parameters,
            }
        case_dir = candidate_dir / "case"
        if case_dir.exists():
            shutil.rmtree(case_dir)
        _copy_case_inputs(src, case_dir)

        remaining = {str(k): v for k, v in parameters.items()}
        patched: Dict[str, List[str]] = {}
        constant_dir = case_dir / "constant"
        for file_path in sorted(p for p in constant_dir.rglob("*") if p.is_file()):
            text = file_path.read_text(encoding="utf-8", errors="ignore")
            original = text
            for key, value in list(remaining.items()):
                pattern = rf"(^|\n)(\s*){re.escape(key)}\s+([^;]+);"
                text, count = re.subn(pattern, rf"\1\2{key} {value};", text)
                if count:
                    patched.setdefault(key, []).append(str(file_path.relative_to(case_dir)))
                    remaining.pop(key, None)
            if text != original:
                file_path.write_text(text, encoding="utf-8")
        if remaining:
            # Not present as plain text anywhere in constant/ — so these are
            # model coefficients, which live in the runtime-only <Model>Coeffs
            # sub-dictionary. Write them there rather than refusing.
            coeff_written = _set_model_coefficients(
                case_dir, {k: remaining[k] for k in list(remaining)}
            )
            for key, rel in coeff_written.items():
                patched.setdefault(key, []).append(rel)
                remaining.pop(key, None)
        if remaining:
            return {
                "ok": False,
                "candidate_dir": str(candidate_dir),
                "case_dir": str(case_dir),
                "error": "One or more coefficient names were not found; refusing to run an unchanged clone.",
                "missing_parameters": sorted(remaining),
                "patched": patched,
            }

        output_path = candidate_dir / "run_result.json"
        # Same shared "oed_candidates" group as the code-mod path: a batch
        # mixes both kinds, and they compete for the same cores and memory.
        proc = coordinator.run_case(
            "oed_candidates",
            lambda: _run_script(
                [
                    "scripts/foam_run_simple.py",
                    "--base-case", str(case_dir),
                    "--output-dir", str(case_dir),
                    "--output", str(output_path),
                    "--timeout", "21600",
                ],
                timeout=22000,
                env=_foamagent_env(settings.openfoam_path),
            ),
        )
        result = _read_json(output_path) or {}
        ok = proc.returncode == 0 and str(result.get("status", "")).lower() in ("success", "ok")
        return {
            "ok": ok,
            "candidate_dir": str(candidate_dir),
            "case_dir": str(case_dir),
            "run_status": result.get("status", ""),
            "patched": patched,
            "stderr_tail": ((result.get("stderr_tail") or "") + proc.stderr[-1000:]) if not ok else "",
        }

    def oed_diagnose_candidate(candidate_dir: str) -> dict:
        """Why did this candidate produce no score, and is it worth another try?

        Call this whenever oed_score_candidate returns a null score, BEFORE
        recording the candidate as failed. It reads what is actually on disk —
        the build result, every graded case's own failure reason, the tail of a
        solver log that failed, the build agent's last moves — and reports the
        cause, whether a bounded change would plausibly fix it, and the concrete
        steps.

        `alters_graded_setup` is the field that decides what you may do next. A
        repair that changes the mesh, boundary conditions, physics, endTime or
        the closure under test is not a repair, it is a different experiment,
        and applying it would make this candidate's score incomparable with
        every other one. Never carry out such steps: record the candidate as
        failed and say why.

        When `repairable` is true and `alters_graded_setup` is false, you may
        carry out `repair_steps` yourself with the file and shell tools, then
        re-run oed_run_evaluation_cases and oed_score_candidate. Two attempts,
        no more — after that record the null score and move to a different
        mechanism. Grinding a broken closure costs budget a new one could use.
        """
        disc_dir = _oed_disc_dir()
        candidate_path = Path(candidate_dir).expanduser().resolve()
        if candidate_path.parent != disc_dir.resolve() or not candidate_path.name.startswith("cand_"):
            return {"ok": False, "error": "candidate_dir must be a direct cand_* child of this study's OED directory."}
        if not candidate_path.is_dir():
            return {"ok": False, "error": f"No such candidate directory: {candidate_path}"}

        record = _read_json(candidate_path / "candidate_record.json") or {}
        execution_doc = (
            _read_json(candidate_path / "agentic_result.json")
            or _read_json(candidate_path / "run_result.json")
            or {}
        )
        attempts = int(record.get("repair_attempts", 0) or 0)
        existing = record.get("failure_diagnosis")
        if isinstance(existing, dict) and existing.get("ok"):
            diagnosis = existing
        else:
            try:
                diagnosis = _diagnose_null_score(
                    candidate_path, execution_doc,
                    str(record.get("score_error", "") or ""), settings
                )
            except Exception as exc:
                return {
                    "ok": False,
                    "error": f"diagnosis raised {type(exc).__name__}: {exc}",
                    "next_step": "Record the null score with what you can see and move on.",
                }

        record["failure_diagnosis"] = diagnosis
        record["repair_attempts"] = attempts
        _write_json(candidate_path / "candidate_record.json", record)

        budget_left = max(0, _OED_REPAIR_ATTEMPTS - attempts)
        may_repair = bool(
            diagnosis.get("repairable")
            and not diagnosis.get("alters_graded_setup")
            and budget_left > 0
        )
        if diagnosis.get("alters_graded_setup"):
            advice = (
                "DO NOT repair. The proposed change would alter something the benchmark "
                "grades on, which would make this candidate incomparable with the rest. "
                "Record it as failed with the cause above."
            )
        elif not diagnosis.get("repairable"):
            advice = (
                "Not repairable — record the null score with the cause above and spend "
                "the budget on a different mechanism."
            )
        elif budget_left <= 0:
            advice = (
                f"Repair budget spent ({attempts}/{_OED_REPAIR_ATTEMPTS} attempts). "
                "Record the null score and move on."
            )
        else:
            advice = (
                f"Repairable, and {budget_left} of {_OED_REPAIR_ATTEMPTS} attempts remain. "
                "Carry out repair_steps with the file and shell tools, then call "
                "oed_run_evaluation_cases and oed_score_candidate again. Call this tool "
                "again afterwards if it still scores null."
            )
        return {
            "ok": True,
            "candidate_dir": str(candidate_path),
            "diagnosis": diagnosis,
            "repair_attempts_used": attempts,
            "repair_attempts_remaining": budget_left,
            "may_repair": may_repair,
            "next_step": advice,
        }

    def oed_note_repair_attempt(candidate_dir: str, what_was_changed: str) -> dict:
        """Record that you just carried out a repair on a null-scoring candidate.

        Call this immediately after making the change and BEFORE re-running the
        evaluation, so the attempt is counted even if the re-run dies. Without
        it the two-attempt budget is unenforceable — a crash mid-repair would
        reset the count and the search could grind one broken candidate
        indefinitely.
        """
        disc_dir = _oed_disc_dir()
        candidate_path = Path(candidate_dir).expanduser().resolve()
        if candidate_path.parent != disc_dir.resolve() or not candidate_path.name.startswith("cand_"):
            return {"ok": False, "error": "candidate_dir must be a direct cand_* child of this study's OED directory."}
        record = _read_json(candidate_path / "candidate_record.json") or {}
        attempts = int(record.get("repair_attempts", 0) or 0) + 1
        log = list(record.get("repair_log") or [])
        log.append({"attempt": attempts, "change": str(what_was_changed)[:2000]})
        record["repair_attempts"] = attempts
        record["repair_log"] = log
        _write_json(candidate_path / "candidate_record.json", record)
        return {
            "ok": True,
            "repair_attempts_used": attempts,
            "repair_attempts_remaining": max(0, _OED_REPAIR_ATTEMPTS - attempts),
            "next_step": (
                "Re-run oed_run_evaluation_cases then oed_score_candidate."
                if attempts < _OED_REPAIR_ATTEMPTS
                else "This was the last attempt — if it still scores null, record it and move on."
            ),
        }

    def oed_score_candidate(
        candidate_dir: str, case_dir: str, action_type: str, variant_name: str,
        model_description: str, target_family: str = "",
        case_dirs: Optional[List[str]] = None,
    ) -> dict:
        """Score one finished OED candidate against the study's baseline and
        write candidate_record.json into candidate_dir — the exact file
        oed_record_candidate_results reads back, so nothing about the score
        needs to be retyped or transcribed anywhere. Call this right after
        oed_run_code_mod_candidate or oed_run_experiment_candidate finishes,
        then just report the candidate_dir back in your final message.

        ``case_dirs`` scores a candidate on a SET of cases instead of one:
        pass the list oed_run_evaluation_cases returned, and the candidate's
        score becomes the mean of the per-case scores. Use it whenever the
        study declares evaluation_cases; leave it out for a single-case study
        and behaviour is exactly as before.

        The mean is over cases the study declared, and a candidate that
        failed to produce a score on any one of them is NOT quietly averaged
        over the rest — a mean of six cases is not comparable with a mean of
        eight, and an archive that mixes the two ranks a candidate for having
        crashed on the hard cases.

        Refuses outright when a standing unclean-finish diagnosis says this
        candidate's model is not complete. That is not a rule imposed from
        outside: it is the diagnosing model's own judgement, made from the
        model source and the case dictionary, taking effect instead of being
        offered as advice. Repair or extend the candidate first — either one
        clears the verdict, because a fresh attempt is judged on its own
        evidence."""
        disc_dir = _oed_disc_dir()
        candidate_path_guard = Path(candidate_dir).expanduser().resolve()
        standing = _read_json(candidate_path_guard / "unclean_finish_diagnosis.json") or {}
        if standing.get("ok") and standing.get("model_is_complete") is False:
            # The precise failure this exists for: on run closure_20260826_codex
            # six candidates whose build agent died mid-fit each scored
            # ~0.1136009, bit-identical to the unmodified baseline, because the
            # coefficient being fitted never reached the case dictionary and the
            # closure ran at its class defaults. execution_ok was True and the
            # score was real, so no existing gate looked at them, and the
            # archive recorded six experiments that had never run.
            return {
                "ok": False,
                "error": (
                    "Refusing to score: the diagnosis of this candidate's unclean "
                    "finish found its model incomplete, so it would run at its class "
                    "defaults and score as the unmodified baseline while being "
                    "recorded as a real experiment."
                ),
                "cause": standing.get("cause", ""),
                "verdict": standing.get("verdict", ""),
                "repair_steps": standing.get("repair_steps", []),
                "extra_seconds_needed": standing.get("extra_seconds_needed", 0),
                "next_step": (
                    "Act on the verdict first: oed_apply_repair for 'repair', "
                    "oed_extend_candidate for 'extend'. Either clears this and the "
                    "next attempt is judged on its own evidence. If the verdict is "
                    "'abandon' or the budgets are spent, record the candidate null "
                    "with this cause rather than scoring it."
                ),
            }
        bound = _read_json(disc_dir / "bound_comparators.json") or {}
        contract = _read_json(disc_dir / "objective_contract.json") or {}
        baseline = _read_json(disc_dir / "baseline_score.json") or {}
        config = _read_json(disc_dir / "search_config.json") or {}
        specs = _metric_specs(disc_dir)
        ref_files = [Path(p) for p in (contract.get("reference_files") or []) if Path(p).is_file()]

        if action_type not in {"code_mod", "experiment"}:
            return {"ok": False, "error": f"Unsupported candidate action_type: {action_type!r}"}
        candidate_path = Path(candidate_dir).expanduser().resolve()
        case_path = Path(case_dir).expanduser().resolve()
        if candidate_path.parent != disc_dir.resolve() or not candidate_path.name.startswith("cand_"):
            return {"ok": False, "error": "candidate_dir must be a direct cand_* child of this study's OED directory."}

        execution_doc = _read_json(
            candidate_path / ("agentic_result.json" if action_type == "code_mod" else "run_result.json")
        ) or {}
        recorded_case = Path(str(execution_doc.get("case_dir", ""))).expanduser().resolve()
        try:
            recorded_case.relative_to(candidate_path)
        except ValueError:
            return {"ok": False, "error": "Execution result points outside its candidate directory."}
        if recorded_case != case_path:
            return {
                "ok": False,
                "error": "case_dir does not match the authoritative execution result.",
                "expected_case_dir": str(recorded_case),
            }
        execution_ok = _run_succeeded(execution_doc)
        if action_type == "code_mod":
            execution_ok = execution_ok and bool(execution_doc.get("compile_ok")) and bool(execution_doc.get("converged"))

        # The graded cases outrank the build's own smoke run.
        #
        # `converged` above describes ONE run the code-mod agent made on the
        # starter geometry while developing the model. The evaluation is a
        # different, larger question: did this closure solve the 32 cases the
        # study is actually scored on. Gating the score on the smoke run threw
        # away candidates that answered yes.
        #
        # Measured on run closure_20260826_codex: cdomega_f1_taper_045_065
        # failed its smoke run, solved all 32 declared cases, and was recorded
        # FAILED with a null score. Scored by hand with the study's own bound
        # comparator it is 0.108830 -- a 4.20% improvement on baseline and the
        # best model the search produced. sst_prod_surface_an04_bn04 (0.3099)
        # and sst_nonlinear_rs_weak_nn (5.2e52) were discarded the same way and
        # genuinely were bad, which is the point: let the SCORE say so, on the
        # cases that count, instead of a proxy deciding in advance.
        #
        # Deliberately narrow. It requires every declared case to have run AND
        # succeeded -- a partial set stays a failure, because a mean over a
        # subset is not this study's metric.
        evaluation_rescue = ""
        early_diagnosis: Optional[Dict[str, Any]] = None
        if not execution_ok and action_type == "code_mod":
            # Whether the graded cases outweigh a failed trial run is a
            # judgement, so it is the model's to make, not a rule's.
            #
            # This was `succeeded == declared`, which is the wrong shape twice
            # over: it says "discard" for 31 of 32 without ever asking why one
            # case failed, and "score it" for 32 of 32 even if all 32 quietly
            # diverged. Neither is a count question. The model is given every
            # case's own failure reason and the tail of a log that failed, and
            # answers whether a score over those cases should be trusted.
            #
            # It is only consulted when the deterministic path has already
            # decided there is no score, so the cost is bounded to failures,
            # and it can only ever turn a refusal into an attempt -- the
            # metric's own invariant below still refuses a mean over a partial
            # set no matter what this says.
            try:
                early_diagnosis = _diagnose_null_score(
                    candidate_path, execution_doc, "", settings
                )
            except Exception as exc:
                # Diagnosis is an aid, never a gate. An exception escaping here
                # would tear down the whole scoring step and lose a candidate
                # that had already cost an hour of compute -- the fourth time
                # this class of bug has ended a step in this file, after
                # grep_files, _run_script and the evaluation timeout.
                early_diagnosis = None
                print(f"[oed] diagnosis unavailable ({type(exc).__name__}: {exc}); "
                      "recording the failure as-is.", flush=True)
            if early_diagnosis and early_diagnosis.get("score_anyway"):
                execution_ok = True
                evaluation_rescue = str(early_diagnosis.get("cause") or "")[:600]
                print(
                    f"[oed] {candidate_path.name}: the build's trial run failed, but the "
                    f"graded cases were judged sound — scoring anyway. "
                    f"{str(early_diagnosis.get('cause'))[:160]}",
                    flush=True,
                )

        score_value: Optional[float] = None
        metric_name = ""
        metric_vector: Dict[str, Any] = {}
        score_error = ""
        score_direction = str(baseline.get("direction", "min"))
        # Which cases this candidate's score is computed over. A study that
        # declared evaluation_cases scores the whole declared set and reports
        # their mean; every other study scores the candidate's own case, as
        # before.
        declared_cases = _evaluation_cases(disc_dir)
        scored_paths: List[Path] = [case_path]
        multi_case = False
        if case_dirs:
            resolved = [Path(str(c)).expanduser().resolve() for c in case_dirs]
            missing_dirs = [str(c) for c in resolved if not c.is_dir()]
            if missing_dirs:
                return {"ok": False, "error": "Some case_dirs do not exist.", "missing": missing_dirs}
            outside = [str(c) for c in resolved if candidate_path not in c.parents]
            if outside:
                # Scoring a directory outside the candidate is how one
                # candidate ends up scored on another's fields, or on the
                # pristine declared case rather than its own replica.
                return {
                    "ok": False,
                    "error": "Every case_dir must live inside the candidate directory.",
                    "outside": outside,
                }
            scored_paths = resolved
            multi_case = True
            if declared_cases and len(resolved) != len(declared_cases):
                # A mean over a subset is not comparable with a mean over the
                # full set, and the archive has no way to tell them apart once
                # it is a single number.
                return {
                    "ok": False,
                    "error": (
                        f"This study declares {len(declared_cases)} evaluation cases but "
                        f"{len(resolved)} were scored. A partial mean is not comparable with "
                        "a full one; re-run the missing cases or record this candidate as failed."
                    ),
                    "declared": [c.name for c in declared_cases],
                    "received": [c.name for c in resolved],
                }

        # The baseline and the candidate must cover the same cases. Checked
        # here as well as at setup because a study can be resumed against a
        # baseline_score.json written before evaluation_cases were declared,
        # and comparing a 32-case mean with a one-case baseline produces a
        # confident, entirely fictitious improvement number.
        if multi_case:
            baseline_cases_scored = baseline.get("evaluation_cases_scored")
            if not isinstance(baseline_cases_scored, int) or baseline_cases_scored != len(scored_paths):
                return {
                    "ok": False,
                    "error": (
                        f"This candidate is scored over {len(scored_paths)} cases but the study's "
                        f"baseline covers {baseline_cases_scored if baseline_cases_scored else 1}. "
                        "A mean over one case set cannot be compared with a mean over another — "
                        "re-run oed_setup_search with the same evaluation_cases so the baseline "
                        "is measured over them too."
                    ),
                    "baseline_cases_scored": baseline_cases_scored,
                    "candidate_cases_scored": len(scored_paths),
                }

        per_case: Dict[str, Any] = {}
        if _oedx is not None and execution_ok and bound and ref_files and all(p.is_dir() for p in scored_paths):
            try:
                baseline_metric = str(baseline.get("metric", ""))
                values: List[float] = []
                for path in scored_paths:
                    mv_one = _oedx.compute_metric_vector(
                        case_dir=path,
                        bound_comparators=bound,
                        reference_file=ref_files[0],
                        metric_specs=specs,
                    )
                    if path == scored_paths[0]:
                        mv = mv_one
                    one_metrics = mv_one.get("metrics") or {}
                    raw = one_metrics.get(baseline_metric)
                    try:
                        parsed = float(raw)
                    except (TypeError, ValueError):
                        parsed = float("nan")
                    per_case[path.name] = parsed if math.isfinite(parsed) else None
                    if math.isfinite(parsed):
                        values.append(parsed)

                metric_vector = mv
                metrics = mv.get("metrics") or {}
                if multi_case:
                    # Every declared case must have produced a value. See the
                    # docstring: a mean over the ones that happened to work
                    # rewards crashing on the hard cases.
                    unscored = [name for name, value in per_case.items() if value is None]
                    if unscored:
                        score_error = (
                            f"metric {baseline_metric!r} produced no value on "
                            f"{len(unscored)} of {len(scored_paths)} evaluation cases "
                            f"({', '.join(sorted(unscored))}); refusing to average over a subset."
                        )
                        metrics = {}
                    else:
                        metrics = dict(metrics)
                        metrics[baseline_metric] = sum(values) / len(values)
                # A candidate is only ever compared against the baseline on
                # the SAME metric. Falling back to whatever else the metric
                # vector happens to contain silently compares, say, a
                # velocity L2 norm against a Cf RMSE and reports the
                # magnitude difference as an improvement — a fabricated
                # winner that then gets promoted and written into the paper.
                # A comparator that failed on this one case is real, honest
                # information: report it as unscored and let the manager
                # decide, rather than scoring on a metric the baseline was
                # never measured on.
                if not baseline_metric:
                    score_error = (
                        "baseline_score.json names no metric, so there is nothing to "
                        "compare this candidate against."
                    )
                elif baseline_metric not in metrics:
                    score_error = (
                        f"comparator for the baseline metric {baseline_metric!r} produced no "
                        f"value for this case (available: {sorted(metrics) or 'none'}); refusing "
                        "to score against a different metric."
                    )
                else:
                    metric_name = baseline_metric
                    parsed_value = float(metrics[baseline_metric])
                    if math.isfinite(parsed_value):
                        score_value = parsed_value
                    else:
                        score_error = f"metric {baseline_metric!r} evaluated to a non-finite value."
            except Exception as exc:
                score_error = f"metric evaluation raised {type(exc).__name__}: {exc}"

        baseline_value = baseline.get("value")
        baseline_direction = str(baseline.get("direction", score_direction))
        baseline_verified = bool(baseline.get("verified")) and baseline_value is not None
        valid_case = execution_ok and score_value is not None and baseline_verified
        target_pct = float(config.get("target_improvement_pct", 0.0) or 0.0)
        improvement: Optional[float] = None
        if valid_case and abs(float(baseline_value)) > 1e-12:
            improvement = _improvement_pct(float(score_value), float(baseline_value), baseline_direction)
        if valid_case and target_pct <= 0:
            target_met = (
                float(score_value) > float(baseline_value)
                if baseline_direction == "max"
                else float(score_value) < float(baseline_value)
            )
        else:
            target_met = improvement is not None and improvement >= target_pct
        # A modification that compiled, ran, converged, and produced a score
        # bit-identical to the baseline did not change the solve. Exact
        # equality is the right test and carries no false positives: two
        # genuinely different solutions over ~15k cells do not agree to all
        # 17 significant digits. Left unflagged this is indistinguishable
        # from a real negative result -- in run oed_20260822_1626_codex_high
        # candidate m051_diff_revflow returned exactly 0.0043210247445270812
        # against a baseline of 0.0043210247445270812, was recorded as a
        # legitimate REVISE, and its whole family was abandoned on it.
        no_op = (
            valid_case
            and score_value is not None
            and baseline_value is not None
            and float(score_value) == float(baseline_value)
        )

        if not execution_ok:
            status = "FAILED"
        elif no_op:
            status = "NO_OP"
        elif score_value is None:
            # The case may have run perfectly and still be unscorable (the
            # baseline's comparator failed on it). That is a scoring failure,
            # not an execution failure, and it must never be silently
            # laundered into a comparable number — see the metric guard above.
            status = "INDETERMINATE" if score_error else "FAILED"
        elif not baseline_verified:
            status = "INDETERMINATE"
        else:
            status = "PROCEED" if target_met else "REVISE"

        variant_slug = _safe_variant_slug(variant_name, "candidate")
        # The proposal is the authority on family and lineage: oed_propose_candidates
        # already classified this candidate deterministically and knows which
        # archive elite it was built from. Falling back to the passed-in
        # target_family only when there is no proposal on record.
        proposal = (_read_json(disc_dir / "proposals.json") or {}).get(variant_slug) or {}
        family = (
            proposal.get("target_family")
            or target_family
            or SearchArchive.classify(model_description, variant_name)
        )
        # Same provenance rule as family: the proposal decided it, so history
        # records what was actually chosen rather than re-deriving a label
        # from prose that may describe the physics but not the method.
        strategy_label = normalize_strategy(
            proposal.get("strategy") or proposal.get("plan") or model_description,
            plan=proposal.get("plan") or "",
            hypothesis=model_description,
            use_llm=True,
        )
        # Measured, not assumed. A candidate that ran the solver forty times
        # inside a fit is not the same price as one that solved once, and with
        # strategy a free choice the search cannot compare strategies honestly
        # unless the cost it charges is the cost that was paid. Falls back to
        # the flat action-type price when no count was recorded (an experiment
        # candidate, or a result predating the counter).
        measured_runs = execution_doc.get("solver_invocations")
        measured_runs = int(measured_runs) if isinstance(measured_runs, int) and measured_runs > 0 else 0
        # Plus every evaluation case this candidate was run on. Those solves are
        # as real as the ones the agent made itself, and on a 32-case study they
        # are most of the candidate's true price.
        evaluation_doc = _read_json(candidate_path / "evaluation_run_result.json") or {}
        evaluation_runs = evaluation_doc.get("solver_invocations")
        evaluation_runs = int(evaluation_runs) if isinstance(evaluation_runs, int) and evaluation_runs > 0 else 0
        total_runs = measured_runs + evaluation_runs
        flat_cost = 2 if action_type == "code_mod" else 1
        cost = max(flat_cost, total_runs) if total_runs > 0 else flat_cost

        record = {
            "action_type": action_type,
            "strategy": strategy_label,
            "family": family,
            "parent_iteration": proposal.get("parent_iteration"),
            # Read back by SearchArchive.replay to relearn which allocation arm
            # has been paying off, so a resumed study keeps what it learned
            # instead of restarting from the priors.
            "search_action": proposal.get("search_action"),
            "lineage_id": proposal.get("lineage_id"),
            "parameters": proposal.get("parameters") or {},
            "model_name_to_reuse": proposal.get("model_name_to_reuse", ""),
            "model_description": model_description,
            "variant_name": variant_slug,
            "case_dir": str(case_path),
            "cost": cost,
            "solver_invocations": total_runs or None,
            "solver_invocations_build": measured_runs or None,
            "solver_invocations_evaluation": evaluation_runs or None,
            "status": status,
            "no_op": no_op,
            "valid_case": valid_case,
            "execution_ok": execution_ok,
            "score": (
                {"metric": metric_name, "value": score_value, "direction": baseline_direction}
                if score_value is not None else None
            ),
            "metric_vector": metric_vector,
            # The per-case scores behind a multi-case mean. Without them the
            # record carries one number and no way to see that it came from
            # eight cases, which of them was worst, or whether the mean is
            # hiding a case that barely moved.
            "per_case_scores": per_case if multi_case else None,
            "evaluation_cases_scored": len(scored_paths) if multi_case else None,
            "score_error": score_error,
            "baseline_metric": str(baseline.get("metric", "")),
            "baseline_score": baseline_value,
            "baseline_direction": baseline_direction,
            "baseline_verified": baseline_verified,
            "improvement_pct": improvement,
            "target_improvement_pct": target_pct,
            "target_met": target_met,
            # Non-empty when the build's trial run failed but the declared
            # evaluation cases all succeeded and decided the outcome instead.
            "evaluation_rescue": evaluation_rescue,
        }
        # A null score always carries a reason.
        #
        # It used to be recorded as bare FAILED with score=None, which is a
        # dead end for everything downstream: the archive cannot tell a
        # diverged closure from a scoring-plumbing bug, and neither can a
        # person reading the run. The evidence to tell them apart is on disk
        # either way; this reads it and says so.
        if score_value is None:
            # Reuse the diagnosis already made above when there was one; it saw
            # the same evidence, and a second call would only cost another
            # request to reach the same place.
            if early_diagnosis is not None:
                record["failure_diagnosis"] = early_diagnosis
            else:
                try:
                    record["failure_diagnosis"] = _diagnose_null_score(
                        candidate_path, execution_doc, score_error, settings
                    )
                except Exception as exc:
                    record["failure_diagnosis"] = {
                        "ok": False,
                        "cause": f"diagnosis raised {type(exc).__name__}: {exc}",
                        "category": "unknown", "repairable": False,
                        "alters_graded_setup": False, "repair_steps": [],
                        "confidence": 0.0, "score_anyway": False,
                    }
            diagnosis = record["failure_diagnosis"]
            print(
                f"[oed] {candidate_path.name}: no score. {diagnosis.get('category')} — "
                f"{str(diagnosis.get('cause'))[:200]}"
                + ("  [repairable]" if diagnosis.get("repairable") else "  [not repairable]"),
                flush=True,
            )
        _write_json(candidate_path / "candidate_record.json", record)
        return {"ok": True, "candidate_dir": str(candidate_path), **record}

    return {
        "manager_tools": [_with_progress(f) for f in [
            list_directory,
            directory_tree,
            make_directory,
            find_files,
            read_text_file,
            write_text_file,
            edit_text_file,
            grep_files,
            read_starter_folder,
            fetch_literature,
            propose_and_rank_hypotheses,
            advance_with_approved_hypotheses,
            generate_case_requirements,
            run_mesh_gate,
            interpret_case,
            analyze_all_cases,
            write_paper,
            oed_setup_search,
            oed_propose_candidates,
            oed_record_candidate_results,
            oed_diagnose_candidate,
            oed_note_repair_attempt,
            # On the manager as well as the candidate runner, for the same
            # reason oed_diagnose_candidate is: the runner acts on a verdict
            # while it still has the candidate in hand, but a runner can also
            # report back without having acted -- it ran out of budget, or the
            # diagnosis arrived after it had finished -- and the manager's own
            # loop (step c1) is then the only thing left that can.
            oed_apply_repair,
            oed_extend_candidate,
            run_audit_and_record,
        ]],
        "case_runner_tools": [_with_progress(f) for f in [
            # General file access, e.g. a reference tutorial or a log file
            # beyond what run_case_native's own error handling surfaces, or a
            # manual fix beyond what the reviewer/rewrite loop attempted.
            list_directory,
            directory_tree,
            make_directory,
            read_text_file,
            write_text_file,
            edit_text_file,
            grep_files,
            # Individual FoamAgent stages — call one directly when only a
            # single step is needed (inspect a parse, regenerate one file,
            # re-run the reviewer) instead of the whole composed loop.
            foam_parse_requirement,
            foam_retrieve_references,
            foam_decompose_subtasks,
            foam_write_case_file,
            foam_generate_allrun,
            foam_review_errors,
            # Composed end-to-end runners — native is the default; scripted
            # is the vendored-Foam-Agent fallback (see its own docstring).
            run_case_native,
        ]],
        "oed_candidate_tools": [_with_progress(f) for f in [
            # Same general file access as case_runner_tools, for the same
            # reason (inspect a log, a manual fix beyond what the agentic
            # code-mod runner's own retry loop attempted).
            list_directory,
            directory_tree,
            read_text_file,
            grep_files,
            oed_run_code_mod_candidate,
            oed_run_experiment_candidate,
            oed_run_evaluation_cases,
            oed_score_candidate,
            oed_diagnose_candidate,
            oed_note_repair_attempt,
            # The two that let a diagnosis actually change something. Without
            # them a verdict of "repair" or "extend" is a well-reasoned
            # observation that nothing acts on.
            oed_apply_repair,
            oed_extend_candidate,
        ]],
        "coordinator": coordinator,
    }
