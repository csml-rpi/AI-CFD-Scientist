from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from deepagents import SubAgent

# A subagent that loops runs until the graph's step limit stops it, and
# deepagents sets that to 9,999. Measured on the ALCF nemotron-3-ultra cavity
# runs: a case-runner called run_case_native 352 times against a precondition
# only the manager could clear ("no converged mesh-gate selection"), and would
# have kept going for hours. One case legitimately needs tens of steps, so a
# few hundred is generous; past that the subagent is told to report back and
# the manager -- the only agent that can fix an upstream block -- gets its turn.
_SUBAGENT_STEP_LIMIT = int(os.environ.get("CFD_SCIENTIST_SUBAGENT_STEPS") or 300)


def _stopped_message(name: str) -> Any:
    """What the manager is told when a subagent is stopped at its step limit."""
    from langchain_core.messages import AIMessage

    return AIMessage(content=(
        f"The {name} subagent was stopped after {_SUBAGENT_STEP_LIMIT} steps without "
        "finishing: it kept repeating the same actions. Nothing it did is lost. Read "
        "its case/candidate directory to see how far it got, fix whatever blocked it "
        "(for example run the mesh gate if a case run was refused for a missing "
        "converged mesh), and only then launch it again."
    ))


# Refusals in a row after which a subagent is stopped. Each refusal already
# tells the model to do something else; one that keeps making refused calls
# is not going to finish, and every further turn costs a full model call.
# Measured on qwen_retest_20261005/cavity: case-runners made 2,436 refused
# run_case_native calls and 4,881 refused re-reads over twelve hours.
_REFUSAL_STOP = int(os.environ.get("CFD_SCIENTIST_SUBAGENT_REFUSALS") or 20)
from cfd_langgraph.llm.caching import stuck  # noqa: E402


def _refused_message(name: str) -> Any:
    from langchain_core.messages import AIMessage

    return AIMessage(content=(
        f"The {name} subagent was stopped after {_REFUSAL_STOP} or more refused calls among its recent ones: "
        "it kept making calls the tools would not run. Nothing it did is lost. Read its "
        "case/candidate directory and the refusals' reasons, fix what blocked it, and only "
        "then launch it again."
    ))


# The limit is set per call because a recursion_limit in the caller's config wins over
# a bound one, and the task tool passes the manager's 9,999: case-runners told to stop
# at 300 ran ~10,000 steps (qwen_retest_20261005/cavity). Raised, GraphRecursionError
# failed the manager's whole step and left the study idle (nemotron cavity runs).
def _capped(spec: SubAgent, precheck: Optional[Callable[[], Optional[str]]] = None) -> Any:
    """``spec`` compiled with a step limit applied on every call, a stop after a
    run of refused calls, and an optional ``precheck`` that may refuse to start
    it; unchanged if deepagents' compile entrypoint is not where it used to be.
    Hitting a limit hands the manager a plain message instead of an exception."""
    try:
        from deepagents.middleware.subagents import create_sub_agent
        from langgraph.errors import GraphRecursionError
        from langchain_core.messages import AIMessage
        from langchain_core.runnables import RunnableLambda

        compiled = create_sub_agent(spec)
        name = spec["name"]

        def _config(config: Any) -> Dict[str, Any]:
            return {**(config or {}), "recursion_limit": _SUBAGENT_STEP_LIMIT}

        def _refused_start() -> Any:
            reason = precheck() if precheck is not None else None
            if not reason:
                return None
            print(f"  ⛔ {name} not started: {reason[:200]}", flush=True)
            return {"messages": [AIMessage(content=reason)]}

        def _run(state: Any, config: Any = None) -> Any:
            refused = _refused_start()
            if refused is not None:
                return refused
            last = None
            try:
                for last in compiled.stream(state, _config(config), stream_mode="values"):
                    if stuck((last or {}).get("messages", []), _REFUSAL_STOP):
                        print(f"  ⛔ {name} stopped: {_REFUSAL_STOP}+ of its recent calls were refused", flush=True)
                        return {"messages": [_refused_message(name)]}
                return last
            except GraphRecursionError:
                print(f"  ⛔ {name} stopped at its {_SUBAGENT_STEP_LIMIT}-step limit", flush=True)
                return {"messages": [_stopped_message(name)]}

        async def _arun(state: Any, config: Any = None) -> Any:
            refused = _refused_start()
            if refused is not None:
                return refused
            last = None
            try:
                async for last in compiled.astream(state, _config(config), stream_mode="values"):
                    if stuck((last or {}).get("messages", []), _REFUSAL_STOP):
                        print(f"  ⛔ {name} stopped: {_REFUSAL_STOP}+ of its recent calls were refused", flush=True)
                        return {"messages": [_refused_message(name)]}
                return last
            except GraphRecursionError:
                print(f"  ⛔ {name} stopped at its {_SUBAGENT_STEP_LIMIT}-step limit", flush=True)
                return {"messages": [_stopped_message(name)]}

        runnable = RunnableLambda(_run, afunc=_arun)
    except Exception as exc:  # pragma: no cover - depends on deepagents internals
        print(f"[subagents] step limit not applied to {spec['name']}: {exc}", flush=True)
        return spec
    return {"name": spec["name"], "description": spec["description"], "runnable": runnable}


def _no_converged_mesh_gate(out_dir: Path) -> Optional[str]:
    """Why a case-runner cannot do anything yet, or None. run_case_native
    refuses every case of a physics group without a converged mesh gate, and
    only the manager can run the gate."""
    for spec_path in (Path(out_dir) / "mesh_gate").glob("*/selected_mesh_spec.json"):
        try:
            if json.loads(spec_path.read_text()).get("converged"):
                return None
        except (OSError, ValueError):
            continue
    return ("Not started: no physics group in this study has a converged mesh gate yet, so "
            "run_case_native would refuse every case. Only you, the manager, can run "
            "run_mesh_gate. Get a converged gate for the case's physics group first; launching "
            "case-runners before that cannot succeed.")

from cfd_langgraph.llm.caching import (
    build_caching_middleware,
    build_context_middleware,
)

from .control import (
    DENY_BUILTIN_FILESYSTEM_TOOLS,
    build_hide_builtin_filesystem_tools_middleware,
    build_interrupt_on,
)


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

If run_case_native's result has `start_protocol`, the case was started from the study's
settled flow on purpose: its 0/ fields hold that flow and its endTime is the study's run
length, whatever the requirement text says about initial fields or iteration counts. That
is not a setup error -- do not change those files, and do not re-run to "fix" them.

If run_case_native's result lists `requirement_checker_issues`, the approved requirement
failed the requirement checker. Check the case against those issues and your task
description. If one made this case come out wrong (e.g. the requirement lists several
values and the case used the wrong one), call run_case_native once more with the same
requirement_text and `clarification` set to the correct case-specific values.

Then report back, in your final message, a short structured summary: case_id, status
(success/failed), loop_count, and — if it failed — the last few lines of error_logs. Do
not attempt to fix a failing case beyond the reviewer loop already built into
run_case_native, do not call the composed runner more than once per case (apart from
that one clarified re-run), and do not include the full stdout/stderr in your final
report — the manager only needs the outcome, not the transcript."""


def build_case_runner_subagent(tools: List[Any], model: Any, out_dir: Path) -> Any:
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
    return _capped(SubAgent(
        name="case-runner",
        description=(
            "Runs exactly one OpenFOAM case (plan, write, run, review-and-retry) using "
            "this workflow's own FoamAgent port and reports back pass/fail. Independent "
            "cases can be launched concurrently, one task call per case, in a single message."
        ),
        system_prompt=_build_case_runner_prompt(out_dir),
        tools=tools,
        model=model,
        middleware=build_hide_builtin_filesystem_tools_middleware()
        + build_caching_middleware(model) + build_context_middleware(model, tools),
        # Same Ctrl-C-driven pause coverage as the manager (see control.py) —
        # matters here even more, since one case can be a dozen-plus
        # sequential tool calls (write each file, then review/rewrite rounds).
        interrupt_on=build_interrupt_on(tools),
        # Same reason as the manager: block deepagents' built-in filesystem
        # tools so a case can't silently "read" or "grep" an empty virtual
        # filesystem instead of the real case directory.
        permissions=DENY_BUILTIN_FILESYSTEM_TOOLS,
    ), precheck=lambda: _no_converged_mesh_gate(out_dir))


def _build_oed_candidate_runner_prompt(out_dir: Path) -> str:
    return f"""You run exactly one open-ended-discovery candidate and report back what happened.
Nothing else.

A candidate is one new model. In a solver study that is a modified model class compiled and
run on the study's case; in a fitted-model study it is a model trained on the study's data
that writes prediction files. The tools below handle both kinds; read what they return.

Your task description gives you either a candidate spec to build, or an existing candidate
to score. A candidate spec has:
- variant_name (a short slug)
- action_type: "code_mod" (build a new model) or "experiment" (re-run an already-compiled
  solver model with new coefficients; solver studies only)
- hypothesis (what to build, for code_mod)
- for experiment: model_name_to_reuse / base_case_dir and parameters (coefficient overrides)
- target_family (which model family this was chosen to explore — carry it through)
- plan (optional): the steps this candidate's strategy needs beyond "implement the
  hypothesis" — which data to read, which fit or optimiser to run, what the fitted
  result becomes. Pass it straight through to oed_run_code_mod_candidate.
- strategy: which search strategy this candidate belongs to. Pass it through too — it sets
  the build agent's time fence against other candidates of the same strategy instead of
  against the whole pool.

Sequence, always in this order:
1. If action_type == "code_mod": call `oed_run_code_mod_candidate(topic, variant_name,
   hypothesis, plan, strategy)`. If action_type == "experiment": call
   `oed_run_experiment_candidate(variant_name, base_case_dir, parameters)`.
   If your task is to score an EXISTING candidate, call `oed_candidate_status(candidate_dir)`
   instead and follow its next_step: a build that finished goes straight to step 2 with the
   candidate_dir and case_dir it reports. Asking oed_run_code_mod_candidate for an existing
   variant_name, with an empty hypothesis, never rebuilds it — it returns the finished
   result, or the diagnosis of an unfinished one.
1b. If step 1 came back with `finished_cleanly: false`, it carries an
   `unclean_finish_diagnosis`: the build agent stopped before it finished, and you must act
   on the verdict BEFORE going near step 2:
     - complete → the model is finished; carry on to step 2 as normal.
     - repair → call `oed_apply_repair(candidate_dir, repair_steps, rationale)` with the
       diagnosis's own steps, then re-read the result.
     - extend → call `oed_extend_candidate(candidate_dir, extra_seconds, rationale)` with
       its extra_seconds_needed and estimate_basis, then re-read the result: it may come
       back clean, or with a fresh diagnosis to act on again.
     - abandon → do not score it. Report the cause and stop.
   Whatever you do, if `model_is_complete` is false, do NOT proceed to scoring. An
   unfinished model can still produce a healthy-looking score — a solver model whose fitted
   coefficient never reached the case runs at its class defaults and scores as the baseline,
   and a fitted model stopped part-way leaves predictions from an early trial or for only
   some seeds — and recording that teaches the search that a strategy failed when it was
   never finished.
   When step 1 came back with `finished_cleanly: true`, go on to step 2. Its other fields
   describe what was built; fields that do not apply to this kind of study are absent.
2. Call `oed_run_evaluation_cases(candidate_dir, case_dir)`. When the study declares no
   evaluation cases this returns immediately saying there is nothing to do, and you go
   straight to step 3. Otherwise it runs this candidate's model on every evaluation case the
   study declared and returns their case_dirs — the candidate is scored on all of them.
3. Call `oed_score_candidate(candidate_dir, case_dir, action_type, variant_name,
   model_description, target_family)` using the candidate_dir/case_dir from step 1, and
   model_description = the hypothesis text (or a short description of the parameters, for
   an experiment). If step 2 returned case_dirs, pass them as the `case_dirs` argument as
   well — the score then becomes their mean. This writes candidate_record.json into
   candidate_dir — the manager reads that file directly, not your final message, so nothing
   about the score itself needs to be exact in your report.
4. If — and only if — oed_score_candidate came back with a NULL score, call
   `oed_diagnose_candidate(candidate_dir)`. It reads what is actually on disk and reports
   the cause, whether a bounded change would plausibly fix it, and whether that change
   would touch anything the benchmark grades on.
5. Report back, in your final message: variant_name, candidate_dir, whether step 1
   succeeded or failed and why in one or two sentences, and — when you ran step 4 — the
   diagnosis verbatim: cause, category, repairable, alters_graded_setup, and the repair
   steps. If you extended or repaired the candidate at step 1b, say so: the verdict, what
   you granted or changed, and whether it then finished. Do not include full
   stdout/stderr.

A null score is not automatically the end of a candidate. It can mean the model is broken,
but it can equally mean our own scoring plumbing failed on a model that was fine — measured
on a real run, the best model in the study (+4.20%) was recorded FAILED with no score because
a single trial run had not converged, while all 32 graded cases had solved. That is why step
4 exists: the reason matters, and it is on disk.

You have no write, edit or shell tool of your own, and that is deliberate — a graded setup
that any agent can quietly edit is not a benchmark. Repairs go through `oed_apply_repair`,
which runs a build agent scoped to this candidate's directory under a hard rule: it may fix
our own plumbing, and it may not change what the benchmark grades — the mesh, boundary
conditions, physics or endTime of a solver study, the data split, metric or scorer of a
fitted-model study, or the model under test. If a diagnosis says the only available fix
would cross that line, it is not a fix — record the candidate null and move on rather than
looking for a way round.

Budgets: two repairs and two extensions per candidate, counted before the work runs so a
crash cannot reset them. When they are gone, score what is on disk if the model is
complete, and record null if it is not.

Never paper over a null score from the candidate's own output, and never describe a
candidate as successful when its score is null.

Everything you produce lives inside this specific candidate's own directory (the
candidate_dir the run tool returns) — the run tools already write there. {out_dir} is
shared across every study, and other candidates are running concurrently right now."""


def build_oed_candidate_runner_subagent(tools: List[Any], model: Any, out_dir: Path) -> Any:
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
    return _capped(SubAgent(
        name="oed-candidate-runner",
        description=(
            "Runs exactly one open-ended-discovery candidate (build a proposed model, or "
            "re-run an existing solver model with new coefficients, then score it), or "
            "scores an existing candidate it is given, and reports back. Independent "
            "candidates can be launched concurrently, one task call per candidate, in a "
            "single message."
        ),
        system_prompt=_build_oed_candidate_runner_prompt(out_dir),
        tools=tools,
        model=model,
        middleware=build_hide_builtin_filesystem_tools_middleware()
        + build_caching_middleware(model) + build_context_middleware(model, tools),
        interrupt_on=build_interrupt_on(tools),
        permissions=DENY_BUILTIN_FILESYSTEM_TOOLS,
    ))


# deepagents adds a "general-purpose" subagent with the manager's own tools
# unless one is supplied, and that default carries none of this harness's
# controls: on qwen_retest_20261005/cavity_r11 one ran to the 9,999-step limit
# with 375 refused calls in a row. Supplying it here keeps the capability with
# the same step cap, refusal stop and loop control as the other subagents.
def build_general_purpose_subagent(tools: List[Any], model: Any) -> Any:
    """deepagents' general-purpose subagent, with this harness's controls."""
    from deepagents.middleware.subagents import GENERAL_PURPOSE_SUBAGENT

    return _capped(SubAgent(
        name=GENERAL_PURPOSE_SUBAGENT["name"],
        description=GENERAL_PURPOSE_SUBAGENT["description"],
        system_prompt=GENERAL_PURPOSE_SUBAGENT["system_prompt"],
        tools=tools,
        model=model,
        middleware=build_hide_builtin_filesystem_tools_middleware()
        + build_caching_middleware(model) + build_context_middleware(model, tools),
        interrupt_on=build_interrupt_on(tools),
        permissions=DENY_BUILTIN_FILESYSTEM_TOOLS,
    ))
