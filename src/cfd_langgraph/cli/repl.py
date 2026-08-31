from __future__ import annotations

import argparse
import json
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from langgraph.types import Command
from pydantic import BaseModel, Field

from cfd_langgraph.config import Settings, get_settings
from cfd_langgraph.llm.factory import create_langchain_llm
from cfd_langgraph.manager import build_manager
from cfd_langgraph.cli import ui
from cfd_langgraph.cli.activity import BOARD, STEERING_QUEUE
from cfd_langgraph.manager.control import GLOBAL_INTERRUPT

BANNER = "cfd-scientist  (Ctrl-C pauses before the next tool call, no work lost — press it again to force-quit instead of waiting)"

CONSOLE_BANNER = (
    "cfd-scientist — type at the bar below at any time, even mid-run.\n"
    "esc or ctrl-c pauses at the next tool-call boundary (nothing in flight is lost). /help for more."
)

_HYPOTHESIS_GATE_TOOL = "advance_with_approved_hypotheses"


def _install_sigint_handler() -> None:
    """First Ctrl-C: request a pause at the next tool-call boundary (see
    manager/control.py) and keep running — whatever's currently executing
    finishes normally, nothing is lost. Second Ctrl-C (any time after,
    whether the pause has landed yet or not): force-quit immediately."""

    def _handler(signum: int, frame: Any) -> None:
        # Set the flag before printing anything. Signal handlers re-enter on
        # the next SIGINT wherever the interpreter happens to be — including
        # partway through these prints — so a check-then-print-then-set order
        # let a fast double Ctrl-C take the "pause requested" branch twice and
        # never reach the force-quit.
        already_requested = GLOBAL_INTERRUPT.is_set()
        GLOBAL_INTERRUPT.request()
        if already_requested:
            print("\n\n-- force quit (second Ctrl-C) --")
            print("Whatever tool call is currently running is NOT stopped by this — if it")
            print("spawned a real subprocess (e.g. an OpenFOAM Allrun), that process keeps")
            print("running under its own PID; check `ps aux | grep Allrun` if unsure.")
            print("Nothing already completed is lost. Resume from the last completed step:")
            print("  python scripts/cfd_cli.py resume --out-dir <out-dir>")
            sys.exit(130)
        print("\n\n⏸  pause requested — finishing the current step, then stopping for input.")
        print("    (press Ctrl-C again to force-quit immediately instead of waiting)")

    signal.signal(signal.SIGINT, _handler)


# How a paused study collects one line from the user. The live console
# (cli/ui.py) supplies one backed by its own input bar; without a TTY the plain
# input() version below is used, so piped stdin and CI behave exactly as before.
# Returning None means "no more input" (EOF, or the console shutting down) and
# every caller treats it as `quit`: leave the interrupt unresolved so a later
# `resume` asks the same question again.
Asker = Callable[[str], Optional[str]]


def _plain_ask(prompt: str) -> Optional[str]:
    try:
        return input(prompt)
    except EOFError:
        return None


def _thread_id_for(out_dir: Path) -> str:
    # One thread per study directory: resuming just means pointing at the
    # same out-dir again, no extra ID to remember or lose.
    return str(out_dir.resolve())


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _print_update_chunk(chunk: Dict[str, Any]) -> None:
    for node_name, update in chunk.items():
        if node_name == "__interrupt__":
            continue
        messages = (update or {}).get("messages") if isinstance(update, dict) else None
        if messages:
            last = messages[-1]
            content = getattr(last, "content", None)
            if isinstance(content, list):
                content = " ".join(
                    b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"
                )
            if content:
                print(f"[{node_name}] {str(content)[:400]}", flush=True)
                continue
        print(f"[{node_name}] updated", flush=True)


def _pending_interrupts(graph: Any, config: Dict[str, Any]) -> List[Any]:
    """Every interrupt currently pending on this checkpoint, not just the first.

    A fan-out is the normal case here, not an edge case: the manager is told
    to launch independent cases and OED candidates as several concurrent
    ``task`` calls in one message, and a Ctrl-C during that fan-out makes
    *each* subagent raise its own interrupt. Resuming with a bare
    ``Command(resume=...)`` while more than one is pending raises
    "When there are multiple pending interrupts, you must specify the
    interrupt id when resuming" — which, uncaught, killed the CLI and left
    the study permanently unresumable, since `resume --out-dir` re-entered
    the same checkpoint and hit the identical error.
    """
    snapshot = graph.get_state(config)
    pending: List[Any] = []
    for task in getattr(snapshot, "tasks", []) or []:
        for interrupt_obj in getattr(task, "interrupts", None) or []:
            pending.append(interrupt_obj)
    return pending


def _describe_hypotheses(out_dir: Path) -> None:
    ranked = _read_json(out_dir / "hypotheses_ranked.json")
    if not ranked:
        print("  (hypotheses_ranked.json not found)")
        return
    print(f"  proposed={ranked.get('num_proposed')}  passed_critique={ranked.get('num_passed_critique')}")
    for c in ranked.get("ranked_hypotheses", []):
        idea = c.get("idea", {}) or {}
        print(f"  [{c.get('rank')}] {c.get('candidate_id')} — {idea.get('objective', '')[:120]}")
        rationale = c.get("rank_rationale")
        if rationale:
            print(f"       rationale: {rationale[:160]}")


def _handle_interrupt(
    graph: Any, config: Dict[str, Any], out_dir: Path, pending: List[Any], ask: Asker
) -> None:
    """Show every pending request and collect one decision covering them all.

    Two kinds of pause land here, told apart by which tool was intercepted:
    the fixed hypothesis-approval gate (always fires), or a Ctrl-C-requested
    pause on whatever tool call was about to run next (see control.py). In
    both cases the tool calls in ``action_requests`` have NOT executed yet —
    that's what makes this safe: there's nothing partial to lose, because
    nothing has started.

    ``pending`` may hold several interrupts at once (one per concurrently
    launched subagent). The same verb applies to all of them: they are the
    one fan-out the user just asked to pause, and answering them
    individually would mean N prompts for a decision the user makes once.
    """
    per_interrupt_actions = [(i, (i.value or {}).get("action_requests", []) or []) for i in pending]
    all_actions = [a for _i, actions in per_interrupt_actions for a in actions]
    is_hypothesis_gate = any(a.get("name") == _HYPOTHESIS_GATE_TOOL for a in all_actions)

    print("\n-- paused --")
    if len(pending) > 1:
        print(f"({len(pending)} concurrent calls are paused — your answer applies to all of them)")
    for i, action in enumerate(all_actions):
        print(f"[{i}] {action['name']}({json.dumps(action.get('args', {}))})")
        if action.get("description"):
            print(f"    {action['description']}")

    # Anything the user typed into the chat bar while this step was running.
    # Delivered here rather than sooner because this is the first moment the
    # graph is stopped with nothing in flight (see manager/control.py).
    queued = list(STEERING_QUEUE)

    if is_hypothesis_gate:
        print("\nranked hypotheses:")
        _describe_hypotheses(out_dir)
        prompt = "\n> inspect / approve / edit <ids,comma,separated> / reject [reason] / quit\n> "
    else:
        print("\n(paused — the call above hasn't run yet, nothing is lost)")
        if queued:
            print("\nyour queued message(s):")
            for message in queued:
                print(f"  ✎ {message}")
            prompt = (
                "\n> send (deliver the message above instead of running this call) / "
                "continue / step / inspect / quit\n> "
            )
        else:
            prompt = "\n> continue (run it, keep going freely) / step (run it, pause again before the next one) / inspect / quit\n> "

    while True:
        answer = ask(prompt)
        if answer is None:
            # EOF on the plain path, or the live console shutting down. Treat
            # it as `quit`: leave the interrupt unresolved in the checkpoint
            # so `resume` asks the same question again, rather than crashing
            # out of a paused study.
            print(f"\nPaused (end of input). Resume later with the same --out-dir {out_dir}")
            sys.exit(0)
        cmd = answer.strip()
        if not cmd:
            continue
        verb, _, rest = cmd.partition(" ")
        verb = verb.lower()

        if verb == "inspect":
            if is_hypothesis_gate:
                _describe_hypotheses(out_dir)
            else:
                print(f"out-dir: {out_dir}")
                for p in sorted(out_dir.rglob("*.json"))[:50]:
                    print(f"  {p.relative_to(out_dir)}")
            continue
        if verb == "quit":
            # Deliberately doesn't resolve the pending interrupts — they're
            # left exactly as-is in the checkpoint. `resume` later finds the
            # same decision point and asks the same question again.
            print(f"Paused. Resume later with the same --out-dir {out_dir}")
            sys.exit(0)

        def decisions_for(actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            if verb == "send":
                # A rejection is the only channel LangGraph's HITL resume has
                # for handing text back to the model mid-run, and it is the
                # right one here: the call is cancelled *before* running, the
                # model reads the message as the reason, and re-plans. Framed
                # explicitly as the user speaking so it is not mistaken for a
                # tool error.
                note = "\n".join(queued)
                return [
                    {
                        "type": "reject",
                        "message": (
                            "The user interrupted to say this — read it and adjust the plan "
                            f"before retrying anything:\n{note}"
                        ),
                    }
                    for _ in actions
                ]
            if verb == "edit":
                ids = [x.strip() for x in rest.split(",") if x.strip()]
                out = []
                for action in actions:
                    edited_args = dict(action.get("args", {}))
                    edited_args["approved_candidate_ids"] = ids
                    out.append({"type": "edit", "edited_action": {"name": action["name"], "args": edited_args}})
                return out
            if verb == "reject":
                return [{"type": "reject", "message": rest or "Rejected by reviewer."} for _ in actions]
            return [{"type": "approve"} for _ in actions]

        if verb in ("approve", "edit", "reject") and is_hypothesis_gate:
            # The hypothesis gate always interrupts, flag or no flag. If the
            # flag happens to be set (a Ctrl-C landed while this prompt was
            # up), answering the gate is not a request for single-step mode —
            # clear it, and say so, rather than silently pausing before every
            # subsequent tool call with no explanation.
            if GLOBAL_INTERRUPT.is_set():
                GLOBAL_INTERRUPT.clear()
                print("(a pending Ctrl-C pause request was cleared; press Ctrl-C again to pause)")
            break
        if verb == "send" and queued and not is_hypothesis_gate:
            STEERING_QUEUE.clear()
            GLOBAL_INTERRUPT.clear()
            break
        if verb in ("continue", "step") and not is_hypothesis_gate:
            if queued:
                # Keeping them queued would silently re-prompt at every later
                # pause for a message the user chose not to send. Dropping
                # them here, loudly, is the honest option.
                STEERING_QUEUE.clear()
                print("(your queued message was discarded — retype it to send)")
            if verb == "continue":
                GLOBAL_INTERRUPT.clear()
            # "step" leaves GLOBAL_INTERRUPT set on purpose — the next tool
            # call pauses here again, same as a debugger's step-over.
            break
        print("Unrecognized command for this pause.")

    # One resume value per pending interrupt, keyed by interrupt id. The
    # unkeyed {"decisions": [...]} form is only legal when exactly one
    # interrupt is pending; keying by id is accepted in both cases, so this
    # single path covers a lone hypothesis gate and an N-way fan-out alike.
    resume_cmd = Command(
        resume={
            interrupt_obj.id: {"decisions": decisions_for(actions)}
            for interrupt_obj, actions in per_interrupt_actions
        }
    )
    # Streamed, not invoked: graph.invoke() would drive the run to completion
    # (or the next pause) with its node updates discarded, so everything after
    # the first pause went silent for the rest of the session.
    for chunk in graph.stream(resume_cmd, config=config, stream_mode="updates"):
        _print_update_chunk(chunk)


def _stream_until_interrupt_or_done(
    graph: Any, payload: Any, config: Dict[str, Any], out_dir: Path, ask: Asker
) -> None:
    pending = _pending_interrupts(graph, config)
    if not pending:
        for chunk in graph.stream(payload, config=config, stream_mode="updates"):
            _print_update_chunk(chunk)
        pending = _pending_interrupts(graph, config)

    # _handle_interrupt streams the resumed run itself, so by the time it
    # returns the graph has already run on to completion or to the next set
    # of pauses — just re-read the checkpoint and loop.
    while pending:
        _handle_interrupt(graph, config, out_dir, pending, ask)
        pending = _pending_interrupts(graph, config)

    print("\nDone (or waiting on the next stage). Check the out-dir for artifacts.")


def _drive_study(
    graph: Any,
    payload: Any,
    config: Dict[str, Any],
    out_dir: Path,
    ask: Asker,
    next_message: Callable[[], Optional[str]],
) -> None:
    """Run the study, then keep the session open for follow-up turns.

    The study reaching the end of its plan is not the end of the
    conversation: the user still wants to ask about a result, correct
    something, or push the search further. Each follow-up is a new turn on the
    same thread ID, so the whole checkpointed history is still there.
    """
    while True:
        try:
            _stream_until_interrupt_or_done(graph, payload, config, out_dir, ask)
        except SystemExit:
            raise
        except Exception as exc:  # noqa: BLE001
            # A study is hours of work behind a checkpoint. Ending the session
            # on one exception throws away the live graph and makes the user
            # restart the process; measured on a real run, a single dropped
            # HTTPS stream after 587s of a `generate_case_requirements` turn
            # killed everything. The checkpoint survives either way, so stay
            # up and let the user decide: type to retry, or /quit.
            print(f"\n✗ that step failed: {type(exc).__name__}: {exc}", flush=True)
            print(
                "  Nothing completed is lost. Type an instruction to continue "
                "(e.g. 'retry that step'), or /quit to stop.",
                flush=True,
            )
        BOARD.note("idle")
        text = next_message()
        if not text:
            return
        BOARD.note("")
        payload = {"messages": [{"role": "user", "content": text}]}


class _PromptHints(BaseModel):
    output_folder: Optional[str] = Field(
        default=None,
        description="The output/working directory path the user wants this run's artifacts written "
        "to, if they mentioned one anywhere in the text (any phrasing — 'output folder', 'out-dir', "
        "'save results to', 'put it in', etc). Null if nothing like that is mentioned.",
    )
    starter_folder: Optional[str] = Field(
        default=None,
        description="A starter/base-case folder path the user pointed at, if any. Null if none mentioned.",
    )


def _llm_extract_hints(topic: str, settings: Settings) -> _PromptHints:
    """Ask the model to find an output-folder path (and, incidentally, a
    starter-folder path) mentioned anywhere in the pasted prompt — instead of
    pattern-matching one fixed phrasing, which breaks the moment the wording
    or position varies (a plain regex requiring "Output folder:" at the
    start of a line, for instance, silently never matches once the prompt is
    pasted as one continuous paragraph instead of separate lines).

    out_dir has to be resolved before the graph is built (it's closed over by
    every tool's file paths and by the checkpoint path/thread ID) — the
    manager agent itself has no tool that could redirect it at runtime — so
    this one small extra model call happens up front, before anything else.
    starter_folder doesn't strictly need this (the manager finds it fine on
    its own via read_starter_folder), but surfacing it here too lets the CLI
    confirm what it found before the run starts, rather than staying silent.
    """
    try:
        llm = create_langchain_llm(model=settings.model, temperature=0.0)
        return llm.with_structured_output(_PromptHints).invoke(
            "Find any output/working-directory path and any starter/base-case folder path "
            f"mentioned in this text. Return null for whichever isn't mentioned.\n\nText:\n{topic}"
        )
    except Exception as exc:
        print(f"(could not extract folder hints via the model: {exc})")
        return _PromptHints()


def _stdin_has_buffered_input(timeout: float = 0.25) -> bool:
    """True if unread text is already sitting in stdin.

    This is what tells a PASTE apart from a person pressing Enter. A terminal
    delivers a paste as one burst into the pty, so the instant we finish a
    line the remainder is already readable; someone typing has nothing
    pending. The timeout covers a large paste that the terminal splits across
    writes -- it is a ceiling, not a delay: the common case returns the
    moment bytes are there.
    """
    try:
        import select

        if not sys.stdin.isatty():
            return False
        ready, _, _ = select.select([sys.stdin], [], [], timeout)
        return bool(ready)
    except Exception:
        return False


def _drain_buffered_stdin_lines() -> List[str]:
    """Consume and return whatever is still buffered in stdin, if anything.

    A safety net, not the main path. Nothing that arrived as part of the
    user's paste may ever be left behind for the NEXT input() prompt to
    swallow: that is exactly how a truncated topic once turned into a stream
    of bogus commands at the hypothesis gate.
    """
    extra: List[str] = []
    while _stdin_has_buffered_input():
        try:
            extra.append(input("> "))
        except EOFError:
            break
    return extra


def _read_topic_with_prompt_toolkit() -> Optional[str]:
    """Read the topic through prompt_toolkit's own bracketed-paste handling.

    Preferred over ``_read_multiline_prompt`` whenever available, because it
    removes the guesswork entirely: the terminal hands a paste to
    prompt_toolkit as one bracketed block, inserted into the buffer verbatim —
    newlines and blank lines included — and never mistaken for the Enter that
    submits. Returns None if prompt_toolkit cannot run here, and the
    select()-based reader below takes over.

    Framed with the same chrome as the running session (``cli/ui.py``) rather
    than a bare caret, so the very first thing on screen is the input box the
    rest of the session uses, not a different-looking prompt.
    """
    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.styles import Style

        kb = KeyBindings()

        @kb.add("enter")
        def _(event) -> None:
            event.current_buffer.validate_and_handle()

        @kb.add("escape", "enter")
        def _(event) -> None:
            event.current_buffer.insert_text("\n")

        session = PromptSession(
            multiline=True,
            key_bindings=kb,
            style=Style.from_dict(ui.BOX_STYLE),
            bottom_toolbar=lambda: ui.box_bottom(
                "paste multi-line freely", "⏎ send · alt+⏎ newline"
            ),
        )
        text = session.prompt(lambda: ui.box_top("<ok>what should this study investigate?</ok>"))
        return (text or "").strip()
    except (EOFError, KeyboardInterrupt):
        return ""
    except Exception as exc:
        # Falling back is fine; falling back *silently* is not — the legacy
        # reader has known paste quirks, so it should be visible when it is
        # the one running.
        print(f"(framed prompt unavailable: {type(exc).__name__}: {exc}; using the basic reader)")
        return None


def _read_multiline_prompt() -> str:
    """Read a topic that may span many lines, blank lines and all.

    Pasting a multi-paragraph prompt works: a blank line ends the prompt only
    when it came from a keypress, i.e. when no more of the paste is waiting.
    The earlier version broke on the first blank line unconditionally, which
    silently truncated pasted topics at the first paragraph.
    """
    print("\nPaste or type your topic/prompt below, one or many lines.")
    print("Paste freely -- blank lines inside a paste are kept.")
    print("Press Enter on an empty line when you're done:\n")
    lines: List[str] = []
    while True:
        try:
            line = input("> ")
        except EOFError:
            break
        if line == "":
            if not lines:
                continue
            if _stdin_has_buffered_input():
                # Blank line from within a paste: keep it, the paste goes on.
                lines.append(line)
                continue
            # Looks finished. Anything still trickling in belongs to the
            # paste, so absorb it rather than leaving it for the next prompt.
            trailing = _drain_buffered_stdin_lines()
            if not trailing:
                break
            lines.append(line)
            lines.extend(trailing)
            continue
        lines.append(line)
    lines.extend(_drain_buffered_stdin_lines())
    return "\n".join(lines).strip()


def cmd_run(args: argparse.Namespace) -> None:
    topic = args.topic
    if not topic and getattr(args, "topic_file", None):
        topic = Path(args.topic_file).read_text(encoding="utf-8").strip()
    if not topic:
        while not topic:
            topic = _read_topic_with_prompt_toolkit()
            if topic is None:
                topic = _read_multiline_prompt()
            if not topic:
                print("(empty — try again)")

    settings = get_settings()
    if args.max_parallel_cases is not None:
        settings = settings.model_copy(update={"max_parallel_cases": args.max_parallel_cases})
    if args.num_candidates is not None:
        settings = settings.model_copy(update={"hypothesis_num_candidates": args.num_candidates})

    hints = _PromptHints() if args.out_dir else _llm_extract_hints(topic, settings)

    if args.out_dir:
        out_dir = Path(args.out_dir)
    elif hints.output_folder:
        out_dir = Path(hints.output_folder)
        print(f"(model found an output folder in the prompt: {out_dir})")
    else:
        out_dir = Path("runs") / f"study_{datetime.now():%Y%m%d_%H%M%S}"
        print(f"(no --out-dir given and none found in the prompt, using {out_dir})")
    out_dir.mkdir(parents=True, exist_ok=True)
    # The user's prompt, verbatim and immutable, as the study's authoritative
    # objective. Tools must not depend on the manager passing it through: it
    # summarises. Measured on a real run, an 86-character paraphrase reached
    # oed_setup_search in place of a 775-character prompt, dropping the
    # scoring contract and the "beat it by 10%" target — so comparators were
    # authored blind and the success threshold silently became 0%.
    prompt_path = out_dir / "user_prompt.txt"
    if not prompt_path.is_file():
        prompt_path.write_text(topic, encoding="utf-8")

    if hints.starter_folder:
        print(f"(model also found a starter folder in the prompt: {hints.starter_folder} — it'll read this itself)")

    _start_session(settings, out_dir, {"messages": [{"role": "user", "content": topic}]})


def _start_session(settings: Settings, out_dir: Path, payload: Any) -> None:
    """Build the graph and run it, under the live console when one is possible.

    The console is a presentation layer only: it supplies an ``ask``/
    ``next_message`` pair and nothing else, so the plain fallback below runs
    the identical study code. That keeps piped stdin, nohup, and CI on exactly
    the behaviour they had before the console existed, instead of on a second
    lightly-tested path.
    """
    graph, stack = build_manager(settings, out_dir)
    config = {"configurable": {"thread_id": _thread_id_for(out_dir)}}
    try:
        if ui.available():
            console = ui.Console(out_dir)
            console.run(
                lambda: _drive_study(
                    graph, payload, config, out_dir, console.ask, console.next_message
                ),
                banner=CONSOLE_BANNER,
            )
        else:
            print(BANNER)
            _drive_study(
                graph, payload, config, out_dir, _plain_ask, lambda: _plain_ask("\n› ")
            )
    finally:
        stack.close()


def cmd_resume(args: argparse.Namespace) -> None:
    _start_session(get_settings(), Path(args.out_dir), None)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cfd-scientist-cli")
    sub = p.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="start a new study")
    run_p.add_argument("--topic", default=None, help="If omitted, you'll be prompted to paste it in.")
    run_p.add_argument(
        "--topic-file",
        default=None,
        help="Read the topic verbatim from a file. Safest for multi-paragraph "
             "prompts, since nothing is retyped or re-buffered.",
    )
    run_p.add_argument("--out-dir", default=None, help="If omitted, defaults to runs/study_<timestamp>.")
    run_p.add_argument("--num-candidates", type=int, default=None)
    run_p.add_argument("--max-parallel-cases", type=int, default=None)
    run_p.set_defaults(func=cmd_run)

    resume_p = sub.add_parser("resume", help="resume a paused study")
    resume_p.add_argument("--out-dir", required=True)
    resume_p.set_defaults(func=cmd_resume)

    return p


def main(argv: Optional[List[str]] = None) -> None:
    _install_sigint_handler()
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
