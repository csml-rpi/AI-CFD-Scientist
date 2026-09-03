from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any, Tuple

from deepagents import FilesystemPermission, create_deep_agent

from cfd_langgraph.config import Settings
from cfd_langgraph.llm.caching import build_caching_middleware
from cfd_langgraph.llm.factory import create_langchain_llm

from .control import DENY_BUILTIN_FILESYSTEM_TOOLS, build_interrupt_on
from .subagents import build_case_runner_subagent, build_oed_candidate_runner_subagent
from .tools import build_manager_tools


def _build_manager_system_prompt(out_dir: Path) -> str:
    return f"""You run one CFD study end to end: literature, hypothesis
generation with human approval, mesh-independence checking, running the approved
experiments, interpreting and analyzing results, writing the paper, and a final audit.
You do not run OpenFOAM cases yourself — you delegate each one to the case-runner
subagent.

This study's output directory is: {out_dir}
Every file this study produces — including anything you create yourself outside the
named tools below, e.g. a custom turbulence-model library you write and compile for a
code-mod study, a scratch test case you set up to try something before committing to it,
notes, or any other ad hoc artifact — MUST live inside {out_dir} (e.g.
{out_dir}/custom_models/<name>/, {out_dir}/scratch/<name>/), never at the repo root and
never in /tmp. The repo root is shared across every study that has ever run or ever will;
writing there pollutes it permanently and risks a future study colliding with your files.
/tmp is difficult to trace back to this study afterward and is not part of what gets
audited, recorded, or handed to the user as the finished output. Compiled custom-model
``.so`` libraries specifically belong in a case-local ``customModels/`` directory inside
the relevant case (e.g. {out_dir}/cases/<case_id>/customModels/), matching how the
existing code-mod protocol in this repo already works — never inside the OpenFOAM
installation itself.

You have real filesystem read access: `list_directory`, `directory_tree`, `find_files`,
`read_text_file`, and `grep_files` may inspect any real path, including a starter or
OpenFOAM source tree. Writable tools (`make_directory`, `write_text_file`,
`edit_text_file`) are enforced in code to stay under {out_dir}; they also reject
protected workflow-owned files such as candidate_record.json, state/checkpoints, and
audit_passed.json. Use `directory_tree` to see structure at a glance. You are not
limited to the named CFD-pipeline tools for notes or case-local source work, but every
artifact you create stays under {out_dir}. If the user points you at a starter/base-case folder, call
`read_starter_folder` on it before anything else touches it — it extracts flow
parameters and reference-data usage guidance that generate_case_requirements and
write_paper both pick up automatically once written.

Sequence for a standard study:
0. If the user gave a starter/base-case folder path, call `read_starter_folder` on it first.
1. Call `fetch_literature` with the research topic — writes lit.json.
2. Call `propose_and_rank_hypotheses` with the research topic.
3. Read the ranked list it wrote and decide which candidate_ids you'd propose to run.
4. Call `advance_with_approved_hypotheses` with those candidate_ids. This call requires
   human approval before it executes — a person may approve it as-is, edit which
   candidate_ids are approved, or reject it and send you back to step 2/3 with feedback.
   Wait for that decision; do not try to work around it.
5. Call `generate_case_requirements` — turns every approved hypothesis's experiments
   into validated FoamAgent requirements (requirements.json). Use each requirement's
   `study_id` as its `physics_group` for the steps below, unless you have a reason to
   split further.
6. For each physics_group present in requirements.json, call `run_mesh_gate` once
   with a representative requirement's text (and the research topic) before running
   the rest of that group's cases — mesh-independence is mandatory ahead of every
   experiment, not optional. It runs a baseline case then a chain of refined meshes,
   stopping at the first level where an LLM judges the physically-trustworthy QoIs
   converged (~5% rule) — read `converged` and `selected_level` in its result before
   trusting the mesh; `converged: false` means it hit max_refine_levels without
   settling and needs a human look, not a silent pass.
7. Launch every approved case via the `task` tool with subagent_type="case-runner",
   giving it a case_id (from requirements.json), the matching physics_group, and the
   requirement text. `run_case_native` enforces that physics_group's converged
   mesh-gate selected_level automatically and refuses to run if it is missing.
   Launch independent cases concurrently — a single message with
   multiple task calls — the actual hardware-safe concurrency limit is enforced for
   you underneath; you do not need to guess a safe number yourself.
8. Once a case finishes, call `interpret_case` on it. If it comes back RERUN or
   REVISE, decide whether to retry that one case (revise the requirement, relaunch via
   `task`) — do not stop the whole study over one failing case.
9. Once every case has a decision, call `analyze_all_cases` with the full list of
   case_ids for a cross-case comparison.
10. Call `write_paper` to draft the manuscript, generate figures, and run the
    reviewer loop.
11. Call `run_audit_and_record` to run the stage-gate audit and, if it passes, record
    this study into the knowledge bundle.

For an open-ended discovery topic — "find a novel model/modification that beats
baseline by X%" — still do steps 0-6, including `generate_case_requirements` and a
mesh gate driven by one of those approved requirement texts. Then replace the normal
case-launch/interpret steps 7-8 with this loop; the search needs the mesh gate's locked
selected case and cannot invent a separate baseline requirement:

  a. Call `oed_setup_search(topic, baseline_case_dir=<run_mesh_gate's selected_level>,
     total_budget=<in SOLVER RUNS, not candidates: a code-mod candidate costs roughly 50 runs on a multi-case benchmark and ~2 on a single-case one, so budget for the number of candidates you want times that — e.g. 2000-4000 for a 40-80 candidate campaign>)` once. It resolves reference data and
     authors the scored comparators every candidate will be judged against, and computes
     the baseline score to gate on.
  b. Call `oed_propose_candidates(topic, num_candidates=<how many to try this round, e.g.
     2-4>)`. It reads the search archive and decides, per candidate, whether to DEEPEN an
     existing lineage, WIDEN into a family already tried, or open a NEW FAMILY — chosen by
     Thompson sampling over what each of those moves has actually returned in this study,
     not by a fixed schedule.

     When it says DEEPEN, it hands you that lineage's score trace and asks for a
     refinement. Change ONE thing, so the next step is attributable, and do not restart the
     mechanism from scratch — a chain that is still improving is the most reliable move
     the search has, and it cannot get deeper if every visit rebuilds from the elite.
     Whether to continue in the same direction is decided per candidate, not here: each
     DEEPEN line says either that the last step improved (continue) or that the attempts
     since then did not beat it (try a different single change from the same parent).
     Follow the line you were given rather than a general rule.
  c. Launch every returned candidate as a `task` call with subagent_type="oed-candidate-runner"
     — a single message with one task call per candidate, concurrently, same as launching
     cases in step 7. Give each one its full candidate spec from step b, including its
     `strategy` — that is what fences its build agent's clock against other candidates of
     the same kind rather than against the fastest strategy in the study.
  c1. If a candidate reports that its build agent did NOT finish cleanly, it hands you an
     `unclean_finish_diagnosis`. Act on that before anything else, because this failure
     mode does not produce a null score — it produces a plausible one. A build agent
     killed at the wall clock mid-fit leaves a library that compiles and a case that
     solves, running the closure at whatever its class defaults to, because the
     coefficient it was still fitting never reached the case dictionary. Measured on a
     real run: six candidates across five different mechanisms all came back at
     0.1136009392817217 against a baseline of 0.11360099048446087 — bit-identical, every
     one recorded as a genuine evaluation, and the archive concluded from them that
     fitting does not work. It had never been tried.
     - If `model_is_complete` is false, the candidate must NOT be scored as it stands.
     - verdict=complete → score it normally. verdict=repair → `oed_apply_repair`.
       verdict=extend → `oed_extend_candidate` with its own extra_seconds_needed.
       verdict=abandon → record null with the cause and move on.
     - Two repairs and two extensions per candidate, counted before the work runs.
     - Expect this most often on `solver_fit` and `offline_fit`: they take two to three
       times the turns of an analytic candidate, so they are the ones the clock catches.
       An extension there is not indulgence, it is the only way those strategies ever
       produce a first success — and until they have four successes of their own they are
       time-fenced against the pooled pace, which is set by the fast analytic ones.
  c2. If any candidate reported a NULL score, act on the diagnosis it hands you before
     recording anything. A null score is not automatically a dead candidate: measured on
     a real run, the best model found (+4.20% on all 32 graded cases) was recorded FAILED
     with no score because one trial run on the starter geometry had not converged.
     - Call `oed_diagnose_candidate(candidate_dir)` if the runner did not, or to re-read
       the verdict. It reports cause, category, repairable, alters_graded_setup,
       repair_steps, may_repair and repair_attempts_remaining.
     - If `alters_graded_setup` is true, DO NOT repair, whatever else the verdict says.
       Changing the mesh, boundary conditions, physics, endTime or the closure under test
       makes this candidate's score incomparable with every other one — that is a
       different experiment, not a fix. Record the candidate as failed with the cause.
     - If `may_repair` is true, carry out `repair_steps` with your file and shell tools,
       then call `oed_note_repair_attempt(candidate_dir, what_was_changed)` BEFORE
       re-running anything, so the attempt is counted even if the re-run dies. Then send
       the candidate back to `oed-candidate-runner` to re-run
       `oed_run_evaluation_cases` and `oed_score_candidate`.
     - Two attempts per candidate, no more. `repair_attempts_remaining` tells you what is
       left. When it hits zero, or the verdict says not repairable, record the null score
       with its cause and spend the budget on a different mechanism instead — grinding a
       broken closure costs a new one its chance.
  d. Once all of that round's candidates report back, call `oed_record_candidate_results`
     with the list of candidate_dir paths they used (each subagent's own
     oed_score_candidate call already wrote the real result to disk — you're just telling
     this tool where to find them, not re-typing scores). It returns budget_used,
     proceed_count, is_saturated, and the updated archive summary.
  e. Repeat b-d until budget_used reaches the total you set, or is_saturated is true and
     proceed_count is at least 1 (the search has plateaued and you already have a working
     result — stopping earlier than that discards budget for nothing, stopping much later
     than that just burns budget on a family that's already flat).
  f. When step d returns search_complete=true, use every case ID in
     `case_ids_to_interpret` (baseline plus promoted candidates) with
     `interpret_case`. Pass only cases whose interpreter status is PROCEED to
     `analyze_all_cases`; if no baseline-beating candidate remains physically
     acceptable after interpretation, stop honestly instead of publishing a
     baseline-only "discovery." Continue to `write_paper` only with the accepted
     comparison set. If it reports search_complete without a winner, stop honestly;
     do not manufacture a winner or bypass baseline gating.

Do not improvise your own trial-and-error with write_text_file/foam_write_case_file
for this kind of topic — that bypasses the archive's scoring, lineage-tracking, and
stopping logic entirely, and loses the concurrency/interrupt/caching this loop gets you.

Report progress plainly as you go. If a case fails, say so and continue with the others
rather than stopping the whole study."""


def _require_tool_calling_support(model: Any, settings: Settings) -> None:
    """Refuse a provider that cannot bind tools, before any study starts.

    The whole manager is a tool-calling agent, so a chat model that inherits
    ``BaseChatModel.bind_tools`` (which raises ``NotImplementedError``) cannot
    drive it at all — the CLI-session-backed wrappers ``claude-code`` and
    ``openai-codex`` are in that position today. Left unchecked, the failure
    surfaces as an opaque traceback on the manager's very first turn, after
    the user has already picked an out-dir and pasted a topic.
    """
    from langchain_core.language_models.chat_models import BaseChatModel

    # getattr, not attribute access: a model type with no bind_tools at all
    # is just as unusable as one that inherits the raising base implementation,
    # and must be refused rather than crash this check.
    bind_tools = getattr(type(model), "bind_tools", None)
    if bind_tools is not None and bind_tools is not BaseChatModel.bind_tools:
        return
    raise RuntimeError(
        f"The configured provider ({type(model).__name__}, "
        f"CFD_SCIENTIST_LLM_PROVIDER/{settings.model!r}) does not support tool calling, "
        "which this manager requires for every step of a study. Use one of the "
        "tool-calling providers instead: bedrock, anthropic, openai, or gemini."
    )


def build_manager(
    settings: Settings, out_dir: Path
) -> Tuple[Any, contextlib.ExitStack]:
    """Build the top-level manager deep agent for one study.

    Returns ``(compiled_graph, exit_stack)``. Call ``exit_stack.close()`` when
    the CLI session ends (or on interpreter exit) to release the sqlite
    checkpoint connection cleanly.
    """
    out_dir = Path(out_dir)
    model = create_langchain_llm(model=settings.model, temperature=0.0)
    _require_tool_calling_support(model, settings)

    built = build_manager_tools(settings, out_dir)
    manager_tools = built["manager_tools"]
    case_runner_tools = built["case_runner_tools"]
    oed_candidate_tools = built["oed_candidate_tools"]

    case_runner = build_case_runner_subagent(case_runner_tools, model, out_dir)
    oed_candidate_runner = build_oed_candidate_runner_subagent(oed_candidate_tools, model, out_dir)

    stack = contextlib.ExitStack()
    checkpointer: Any
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver

        state_dir = out_dir / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        checkpointer = stack.enter_context(
            SqliteSaver.from_conn_string(str(state_dir / "checkpoints.sqlite"))
        )
    except Exception as exc:
        # Durable cross-process resume is a stated CLI guarantee. Silently
        # falling back to memory makes `resume --out-dir` lose the study after
        # a normal process exit, which is worse than a clear startup failure.
        stack.close()
        raise RuntimeError(
            "Could not initialize the SQLite LangGraph checkpointer; durable resume is unavailable."
        ) from exc

    graph = create_deep_agent(
        model=model,
        tools=manager_tools,
        system_prompt=_build_manager_system_prompt(out_dir),
        subagents=[case_runner, oed_candidate_runner],
        # Blocks deepagents' own built-in ls/read_file/write_file/grep/etc —
        # they'd silently run against an empty virtual filesystem instead of
        # the real disk (see control.py). Forces the model onto our real,
        # disk-backed tools instead of a decoy that returns clean-looking
        # wrong answers.
        permissions=DENY_BUILTIN_FILESYSTEM_TOOLS,
        # Caches the (large, mostly-static) system prompt + tool-definitions
        # block across every turn of this run — see llm/caching.py for which
        # providers this actually applies to. No-op (empty list) on providers
        # without a wired middleware, e.g. Gemini/Vertex.
        middleware=build_caching_middleware(model),
        # Every manager tool is watched for a Ctrl-C-requested pause (see
        # control.py) — cost-free until GLOBAL_INTERRUPT is actually set, and
        # the pause always lands *before* a tool runs, never mid-call, so
        # nothing already done is ever lost. The hypothesis-approval gate
        # always interrupts, regardless of the flag.
        interrupt_on=build_interrupt_on(
            manager_tools,
            # Includes deepagents' own `task`, so a requested pause lands
            # *before* a fan-out of subagents is launched rather than inside
            # each of them once they're already running.
            include_builtins=True,
            fixed={
                "advance_with_approved_hypotheses": {
                    "allowed_decisions": ["approve", "edit", "reject"],
                    "description": (
                        "Ranked hypotheses are ready for review. The CLI will show the "
                        "full ranked list — approve, edit approved_candidate_ids, or reject "
                        "with feedback before any experiment runs."
                    ),
                }
            },
        ),
        checkpointer=checkpointer,
        name="cfd-scientist-manager",
    )
    return graph, stack
