from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from deepagents import SubAgent

from cfd_langgraph.llm.caching import build_caching_middleware

from .control import DENY_BUILTIN_FILESYSTEM_TOOLS, build_interrupt_on


def _build_case_runner_prompt(out_dir: Path) -> str:
    return f"""You run exactly one OpenFOAM case and report back what happened. Nothing else.

You will be given, in your task description:
- a case_id (e.g. "case_003")
- a physics_group (cases that share a mesh/physics shape use the same group string)
- the full requirement text FoamAgent needs to write and run the case

Default path: call `run_case_native` exactly once with those three values. It runs the
whole FoamAgent loop (parse the requirement, retrieve reference tutorials, decompose
into files, write each file, generate and run Allrun, review-and-retry on failure) as
part of this workflow directly — you do not need a separate Foam-Agent installation
for this path.

If you need to inspect or redo only one step (e.g. the case seems mis-parsed, or you
want to regenerate a single file), call the individual stage tools instead:
`foam_parse_requirement`, `foam_retrieve_references`, `foam_decompose_subtasks`,
`foam_write_case_file`, `foam_generate_allrun`, `foam_review_errors`.

If you write or compile anything yourself beyond what run_case_native produces — a
custom-model source file, a scratch test of something before committing to it — it goes
inside this specific case's own directory, {out_dir}/cases/<case_id>/, e.g. a compiled
custom-model library belongs in {out_dir}/cases/<case_id>/customModels/. Never at the
repo root, never in /tmp — both are shared across every study and case that has ever run,
and writing there either pollutes them permanently or risks a collision with another
case running concurrently right now.

Then report back, in your final message, a short structured summary: case_id, status
(success/failed), loop_count, and — if it failed — the last few lines of error_logs. Do
not attempt to fix a failing case beyond the reviewer loop already built into
run_case_native, do not call the composed runner more than once per case, and do not
include the full stdout/stderr in your final report — the manager only needs the
outcome, not the transcript."""


def build_case_runner_subagent(tools: List[Any], model: Any, out_dir: Path) -> SubAgent:
    """The subagent every experiment case runs through.

    Isolated context by design: FoamAgent's planner/writer/reviewer loop for
    one case can be noisy (multi-hour logs, retry chatter). The manager only
    ever sees this subagent's short final report, not that transcript — the
    same context-isolation deepagents gives every subagent. Concurrency
    itself is not decided here: the manager fans out one `task` call per
    case, as many as it wants, and the real hardware-safe cap is enforced
    inside the shared coordinator `run_case_native` goes through (see
    manager/tools.py CaseCoordinator).

    Prompt caching matters more here than anywhere else in this harness: one
    case can mean a dozen-plus sequential model calls (write each subtask
    file, then a review/rewrite round per retry), all sharing the same large
    system prompt + tool-definitions prefix.
    """
    return SubAgent(
        name="case-runner",
        description=(
            "Runs exactly one OpenFOAM case (plan, write, run, review-and-retry) using "
            "this workflow's own FoamAgent port and reports back pass/fail. Independent "
            "cases can be launched concurrently, one task call per case, in a single message."
        ),
        system_prompt=_build_case_runner_prompt(out_dir),
        tools=tools,
        model=model,
        middleware=build_caching_middleware(model),
        # Same Ctrl-C-driven pause coverage as the manager (see control.py) —
        # matters here even more, since one case can be a dozen-plus
        # sequential tool calls (write each file, then review/rewrite rounds).
        interrupt_on=build_interrupt_on(tools),
        # Same reason as the manager: block deepagents' built-in filesystem
        # tools so a case can't silently "read" or "grep" an empty virtual
        # filesystem instead of the real case directory.
        permissions=DENY_BUILTIN_FILESYSTEM_TOOLS,
    )


def _build_oed_candidate_runner_prompt(out_dir: Path) -> str:
    return f"""You run exactly one open-ended-discovery candidate — one proposed model
modification or coefficient experiment — and report back what happened. Nothing else.

You will be given, in your task description, a candidate spec with:
- variant_name (a short slug)
- action_type: "code_mod" (a new/modified model class) or "experiment" (reuse an
  already-compiled model with new coefficients)
- hypothesis (the concrete modification to implement, for code_mod)
- for experiment: model_name_to_reuse / base_case_dir and parameters (coefficient overrides)
- target_family (which model family this was chosen to explore — carry it through)
- plan (optional): the steps this candidate's strategy needs beyond "implement the
  hypothesis" — which data to read, which fit or optimiser to run, what the fitted
  result becomes. Pass it straight through to oed_run_code_mod_candidate; the runner
  gives it to the agent that builds the model.
- strategy: which search strategy this candidate belongs to (analytic / sweep /
  solver_fit / offline_fit). Pass it through too — it sets the build agent's time
  fence against other candidates of the same strategy instead of against the whole
  pool, which is the difference between a solver-in-the-loop fit getting the time it
  needs and being held to the pace of a one-shot analytic model.

Sequence, always in this order:
1. If action_type == "code_mod": call `oed_run_code_mod_candidate(topic, variant_name,
   hypothesis, plan, strategy)`. If action_type == "experiment": call
   `oed_run_experiment_candidate(variant_name, base_case_dir, parameters)`.
1b. If step 1 came back with an `unclean_finish_diagnosis`, the build agent stopped
   early and you must act on the verdict BEFORE going near step 2:
     - complete → the model is finished; carry on to step 2 as normal.
     - repair → call `oed_apply_repair(candidate_dir, repair_steps, rationale)` with the
       diagnosis's own steps, then carry on to step 2.
     - extend → call `oed_extend_candidate(candidate_dir, extra_seconds, rationale)` with
       its extra_seconds_needed and estimate_basis, then re-read the result: it may come
       back clean, or with a fresh diagnosis to act on again.
     - abandon → do not score it. Report the cause and stop.
   Whatever you do, if `model_is_complete` is false, do NOT proceed to scoring. A model
   whose fitted coefficient never reached the case dictionary runs at its class defaults,
   solves all 32 cases perfectly happily, and scores as the unmodified baseline. That
   result looks healthy and is worthless, and recording it teaches the search that the
   whole strategy is a dead end. Six candidates on a real run did exactly this.
2. Call `oed_run_evaluation_cases(candidate_dir, case_dir)`. In a single-case study this
   returns immediately saying there is nothing to do, and you go straight to step 3. In a
   multi-case study it runs this candidate's compiled model on every evaluation case the
   study declared and returns their case_dirs — the candidate is scored on all of them,
   not just the one it was built against.
3. Call `oed_score_candidate(candidate_dir, case_dir, action_type, variant_name,
   model_description, target_family)` using the candidate_dir/case_dir the first call
   returned, and model_description = the hypothesis text (or a short description of the
   parameters, for an experiment). If step 2 returned case_dirs, pass them as the
   `case_dirs` argument as well — the score then becomes their mean. This writes
   candidate_record.json into candidate_dir — the manager reads that file directly, not
   your final message, so nothing about the score itself needs to be exact in your report.
4. If — and only if — oed_score_candidate came back with a NULL score, call
   `oed_diagnose_candidate(candidate_dir)`. It reads what is actually on disk (the build
   result, every graded case's own failure reason, the tail of a solver log that failed)
   and reports the cause, whether a bounded change would plausibly fix it, and whether
   that change would touch anything the benchmark grades on.
5. Report back, in your final message: variant_name, candidate_dir, whether step 1
   succeeded or failed and why in one or two sentences, and — when you ran step 4 — the
   diagnosis verbatim: cause, category, repairable, alters_graded_setup, and the repair
   steps. If you extended or repaired the candidate at step 1b, say so: the verdict, what
   you granted or changed, and whether it then finished. Do not include full
   stdout/stderr.

A null score is not automatically the end of a candidate. It can mean the closure
diverged, but it can equally mean our own scoring plumbing failed on a model that solved
every graded case — measured on a real run, the best model in the study (+4.20%) was
recorded FAILED with no score because a single trial run on the starter geometry had not
converged, while all 32 graded cases had solved. That is why step 4 exists: the reason
matters, and it is on disk.

You have no write, edit or shell tool of your own, and that is deliberate — a graded
setup that any agent can quietly edit is not a benchmark. Repairs go through
`oed_apply_repair`, which runs a build agent scoped to this candidate's directory under
a hard rule: it may fix our own plumbing (a library that is not being loaded, a
coefficient computed but never written into the case, a missing dictionary entry) and it
may not touch the mesh, the boundary conditions, the physics, the endTime, or the closure
under test. If a diagnosis says the only available fix would cross that line, it is not a
fix — record the candidate null and move on rather than looking for a way round.

Budgets: two repairs and two extensions per candidate, counted before the work runs so a
crash cannot reset them. When they are gone, score what is on disk if the model is
complete, and record null if it is not.

Never paper over a null score from the candidate's own output, and never describe a
candidate as successful when its score is null.

Everything you produce lives inside this specific candidate's own directory (the
candidate_dir the run tool returns) — the run tools already write there. {out_dir} is
shared across every study, and other candidates are running concurrently right now."""


def build_oed_candidate_runner_subagent(tools: List[Any], model: Any, out_dir: Path) -> SubAgent:
    """The subagent every open-ended-discovery candidate runs through.

    Mirrors build_case_runner_subagent exactly, for the same reason: the
    manager fans out one `task` call per candidate (as many concurrently as
    it wants — CaseCoordinator enforces the real hardware-safe cap
    underneath, same shared coordinator run_case_native already goes
    through), each with its own isolated context and its own cached model,
    instead of the old single-subprocess search loop where none of that
    applied. See manager/tools.py's oed_propose_candidates /
    oed_run_code_mod_candidate / oed_run_experiment_candidate /
    oed_score_candidate / oed_record_candidate_results, and
    scripts/oed_search_archive.py for the archive this whole loop serves.
    """
    return SubAgent(
        name="oed-candidate-runner",
        description=(
            "Runs exactly one open-ended-discovery candidate (compile/run a proposed model "
            "modification, or re-run an existing model with new coefficients, then score "
            "it) and reports back. Independent candidates can be launched concurrently, "
            "one task call per candidate, in a single message."
        ),
        system_prompt=_build_oed_candidate_runner_prompt(out_dir),
        tools=tools,
        model=model,
        middleware=build_caching_middleware(model),
        interrupt_on=build_interrupt_on(tools),
        permissions=DENY_BUILTIN_FILESYSTEM_TOOLS,
    )
