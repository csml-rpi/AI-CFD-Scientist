"""A live console for a study: scrolling output, a status line, a chat bar.

Three things the plain ``print``/``input`` REPL cannot do, all of which matter
when a single step legitimately runs for hours:

* **Show that something is happening.** ``_with_progress`` prints when a tool
  starts and when it ends; between those two lines the terminal is silent.
  The status line here reads ``cli/activity.py`` and animates only while a
  tool is genuinely in flight, with that call's own elapsed time.
* **Stay typable.** ``input()`` owns the terminal, so the prompt and the
  agent's output fight over the same cursor. prompt_toolkit's
  ``patch_stdout`` keeps one input line pinned at the bottom and scrolls all
  output above it, so the user can type at any moment — including in the
  middle of a three-hour case.
* **Take a steering message mid-run.** Typing while the study is busy queues
  the text and requests a pause; it is delivered at the next tool-call
  boundary (see ``manager/control.py`` for why that boundary, and not sooner,
  is the only lossless place to stop).

Every one of those is optional. ``available()`` is false without a TTY or
without prompt_toolkit installed, and ``cli/repl.py`` then runs exactly the
plain path it always did — so piped stdin, CI, and nohup keep working
unchanged.
"""

from __future__ import annotations

import asyncio
import contextlib
import html
import re
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Callable, List, Optional

from cfd_langgraph.cli.activity import (
    BOARD,
    STEERING_QUEUE,
    format_duration,
    spinner_frame,
)
from cfd_langgraph.manager.control import GLOBAL_INTERRUPT

_HELP = """\
  <any text>   send it to the study. If a step is running, it pauses at the
               next tool-call boundary and delivers your message there —
               nothing in flight is thrown away.
  /status      what is running right now, and for how long
  /dir         this study's output directory
  /help        this list
  /quit        stop the session (the study resumes from the last completed step)
               (/exit and /q do the same)
  esc, ctrl-c  request a pause at the next tool-call boundary
               (press again to force-quit)
  alt+enter    newline without sending"""


def _visible(text: str) -> str:
    """Text with markup tags stripped, for width arithmetic."""
    return re.sub(r"<[^>]+>", "", text)


def available() -> bool:
    """True when a live console can actually be driven here."""
    try:
        import prompt_toolkit  # noqa: F401
    except Exception:
        return False
    return bool(sys.stdin.isatty() and sys.stdout.isatty())



# ---------------------------------------------------------------------------
# shared chrome
# ---------------------------------------------------------------------------

BOX_STYLE = {
    # noreverse: prompt_toolkit reverse-videos the bottom toolbar by default,
    # which would paint the box's bottom border as a solid bar and break the
    # frame the top border started.
    "bottom-toolbar": "noreverse bg:default",
    "bottom-toolbar.text": "noreverse bg:default",
    "box": "#5f5f5f",
    "dim": "#808080",
    "run": "#5fafff bold",
    "ok": "#5faf5f bold",
    "wait": "#ffaf5f bold",
    "caret": "#5fafff bold",
}


def box_width() -> int:
    """Box width, taken from prompt_toolkit's own output.

    Not from ``shutil.get_terminal_size``/``COLUMNS``: prompt_toolkit renders
    against the size it queried from the terminal, and when the two disagree
    the border is drawn wider than the render area, wraps, and tears the frame
    in half. Asking the renderer is also what makes a window resize come out
    right. One column spare, because a border drawn to the exact width wraps
    for the same reason.
    """
    columns = 0
    try:
        from prompt_toolkit.application.current import get_app

        columns = get_app().output.get_size().columns
    except Exception:
        columns = 0
    if columns <= 0:
        columns = shutil.get_terminal_size((100, 24)).columns
    return max(40, min(columns - 1, 160))


def box_top(status: str):
    """Status line + top border + the left edge carrying the caret."""
    from prompt_toolkit.formatted_text import HTML

    width = box_width()
    return HTML(f" {status}\n<box>╭{'─' * (width - 2)}╮</box>\n<box>│</box> <caret>›</caret> ")


def box_bottom(left: str, right: str):
    """Bottom border + a two-column footer line.

    PromptSession has no framed-input widget, so the frame is drawn in the
    pieces it does support: top border and left edge in `message`, bottom
    border here. The input line is deliberately left open on the right —
    `rprompt` anchors to the first line of a multi-line message, which is the
    status line, so using it put the right edge in the wrong place entirely.
    """
    from prompt_toolkit.formatted_text import HTML

    width = box_width()
    gap = max(1, width - 4 - len(_visible(left)) - len(_visible(right)))
    return HTML(f"<box>╰{'─' * (width - 2)}╯</box>\n  <dim>{left}</dim>{' ' * gap}<dim>{right}</dim>")


class _Question:
    """One request from the study thread for a line of input."""

    def __init__(self, prompt: str, kind: str) -> None:
        self.prompt = prompt
        self.kind = kind  # "decision" (a paused tool call) | "idle" (study at rest)
        self.event = threading.Event()
        self.answer: Optional[str] = None


class Console:
    def __init__(self, out_dir: Path) -> None:
        self.out_dir = Path(out_dir)
        self._lock = threading.Lock()
        self._question: Optional[_Question] = None
        self._finished = threading.Event()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._app_ref = None
        self._interrupt_pressed = False
        self._phase_key = ""
        self._phase_started = time.monotonic()

    # -- called from the study thread -----------------------------------

    def ask(self, prompt: str, kind: str = "decision") -> Optional[str]:
        """Block the study thread until the user submits a line.

        Returns ``None`` if the console is shutting down — the same contract
        as EOF on the plain path, so callers leave a pending interrupt
        unresolved and ``resume`` asks the same question again.
        """
        question = _Question(prompt, kind)
        with self._lock:
            if self._finished.is_set():
                return None
            self._question = question
        # The caller's prompt text carries the actual menu of choices
        # ("send / continue / step / inspect / quit"). The input bar only has
        # room for a short "decision ›" marker, so the menu is printed into
        # the scroll area — without this the user is asked to decide with no
        # indication of what the options are.
        if prompt.strip():
            print(prompt.rstrip("> \n"), flush=True)
        self._invalidate()
        question.event.wait()
        with self._lock:
            self._question = None
        return question.answer

    def next_message(self) -> Optional[str]:
        """A new user turn once the study is idle. None means "session over"."""
        text = self.ask("", kind="idle")
        return (text or "").strip() or None

    # -- status line ----------------------------------------------------

    def _phase(self, key: str) -> float:
        """Seconds in the current display state, restarting when it changes.

        Without this a long model call between tool calls has no clock of its
        own and the status sits perfectly still — indistinguishable from a
        hang. At high reasoning effort one turn can take a minute with no tool
        in flight at all.
        """
        if key != self._phase_key:
            self._phase_key = key
            self._phase_started = time.monotonic()
        return time.monotonic() - self._phase_started

    def _status_line(self) -> str:
        """The activity line above the box — the thing that says "still alive"."""
        spans, _done, _failed, _note, _total = BOARD.snapshot()
        with self._lock:
            question = self._question

        if question is not None and question.kind == "decision":
            return f"<wait>◆ waiting for your answer</wait> · {format_duration(self._phase('ask'))}"
        if question is not None:
            return "<ok>✓ idle</ok> <dim>· type a message to continue the study</dim>"
        if spans:
            head = spans[0]
            self._phase(f"tool:{head.token}")
            lead = f"{spinner_frame()} {head.label} · {format_duration(head.elapsed)}"
            # Budget what is left of the box width for the argument preview and
            # cut it there. An untruncated preview ran past the frame and made
            # the box look broken — a case dir alone can be 80 characters.
            room = box_width() - len(lead) - 4
            detail = (head.detail or "")[:room] if room > 8 else ""
            return (
                f"<run>{spinner_frame()} {html.escape(head.label)}</run>"
                f" · {format_duration(head.elapsed)}"
                + (f" <dim>{html.escape(detail)}</dim>" if detail else "")
            )
        # Between tool calls the model is thinking. Animated on purpose: a
        # still line here is what made a working run look frozen.
        return f"<run>{spinner_frame()} thinking</run> · {format_duration(self._phase('think'))}"

    def _message(self):
        return box_top(self._status_line())

    def _toolbar(self):
        spans, done, failed, note, total = BOARD.snapshot()
        tally = f"{done} done"
        if failed:
            tally += f" · {failed} failed"
        tally += f" · {format_duration(total)} elapsed"
        if note:
            tally += f" · {html.escape(note)}"
        if len(spans) > 1:
            tally += f" · {len(spans)} running"
        hint = (
            "pause requested — finishing this step"
            if GLOBAL_INTERRUPT.is_set()
            else "esc pause · ⏎ send · /help"
        )
        return box_bottom(tally, hint)

    # -- input routing --------------------------------------------------

    def _handle_line(self, text: str) -> None:
        text = text.strip()
        if not text:
            return

        # Slash commands are checked first, and unconditionally. They are the
        # only way to quit, and the study being blocked on a question is
        # exactly when someone reaches for /quit or /status — routing those
        # into the answer instead sent "/quit" to the model as a chat message
        # and left the session with no way out.
        if text.startswith("/"):
            self._local_command(text)
            return

        with self._lock:
            question = self._question
        # Otherwise a pending question owns the line: a paused tool call is a
        # decision the graph is actively blocked on, so routing that line
        # anywhere else would hang the study waiting for an answer the user
        # believes they already gave.
        if question is not None:
            question.answer = text
            question.event.set()
            return

        STEERING_QUEUE.append(text)
        GLOBAL_INTERRUPT.request()
        print(
            f"\n✎ queued: {text}\n"
            "  (pausing at the next tool-call boundary to deliver it — "
            "nothing already running is discarded)",
            flush=True,
        )

    def _local_command(self, text: str) -> None:
        verb = text.split()[0].lower()
        if verb == "/help":
            print("\n" + _HELP, flush=True)
        elif verb == "/dir":
            print(f"\nout-dir: {self.out_dir}", flush=True)
        elif verb == "/status":
            spans, done, failed, note, total = BOARD.snapshot()
            print(f"\nsession {format_duration(total)} · {done} done · {failed} failed", flush=True)
            if note:
                print(f"  {note}", flush=True)
            if not spans:
                print("  nothing running right now", flush=True)
            for span in spans:
                print(f"  {span.label} · {format_duration(span.elapsed)} · {span.detail}", flush=True)
        elif verb in ("/quit", "/exit", "/q"):
            self._quit()
        else:
            print(f"\nUnknown command {verb}. /help for the list.", flush=True)

    def _quit(self) -> None:
        print(f"\nPaused. Resume later with the same --out-dir {self.out_dir}", flush=True)
        self.shutdown()

    def _interrupt(self) -> None:
        if GLOBAL_INTERRUPT.is_set() and self._interrupt_pressed:
            print(
                "\n-- force quit --\n"
                "Whatever tool call is currently running is NOT stopped by this. If it\n"
                "spawned a real subprocess (e.g. an OpenFOAM Allrun), that process keeps\n"
                "running under its own PID; check `ps aux | grep Allrun` if unsure.\n"
                "Nothing already completed is lost. Resume from the last completed step:\n"
                f"  python scripts/cfd_cli.py resume --out-dir {self.out_dir}",
                flush=True,
            )
            self.shutdown()
            return
        self._interrupt_pressed = True
        GLOBAL_INTERRUPT.request()
        print(
            "\n⏸  pause requested — finishing the current step, then stopping for input.\n"
            "    (press again to force-quit immediately instead of waiting)",
            flush=True,
        )

    # -- lifecycle ------------------------------------------------------

    def _invalidate(self) -> None:
        loop, app = self._loop, self._app_ref
        if loop is None or app is None:
            return
        with contextlib.suppress(RuntimeError):
            loop.call_soon_threadsafe(app.invalidate)

    def shutdown(self) -> None:
        """Stop the console and release anyone blocked in ``ask``.

        Releasing outstanding questions is not optional: the study thread is
        sitting in ``question.event.wait()``, and leaving it there on exit
        deadlocks the process instead of ending the session.
        """
        self._finished.set()
        with self._lock:
            question = self._question
        if question is not None and not question.event.is_set():
            question.answer = None
            question.event.set()
        self._invalidate()

    def run(self, drive: Callable[[], None], banner: str = "") -> None:
        """Run ``drive`` (the study) on a worker thread under a live console."""
        from prompt_toolkit import PromptSession
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.patch_stdout import patch_stdout
        from prompt_toolkit.styles import Style

        kb = KeyBindings()

        @kb.add("c-c")
        @kb.add("escape", eager=True)
        def _(event) -> None:
            self._interrupt()

        @kb.add("enter")
        def _(event) -> None:
            # Enter sends. Newlines still arrive intact from a bracketed
            # paste, which prompt_toolkit inserts as buffer text rather than
            # as Enter keypresses — so a pasted multi-paragraph prompt lands
            # whole instead of being submitted at its first blank line.
            event.current_buffer.validate_and_handle()

        @kb.add("escape", "enter")
        def _(event) -> None:
            event.current_buffer.insert_text("\n")

        @kb.add("c-d")
        def _(event) -> None:
            if not event.current_buffer.text:
                self._quit()

        # noreverse: prompt_toolkit reverse-videos the bottom toolbar by
        # default, which would paint the box's bottom border as a solid bar
        # and break the frame the top border started.
        style = Style.from_dict(BOX_STYLE)

        def _worker() -> None:
            try:
                drive()
            except SystemExit:
                pass
            except BaseException as exc:  # surfaced, then the session ends cleanly
                print(f"\n✗ session ended with an error: {exc!r}", flush=True)
            finally:
                self._finished.set()
                self._invalidate()

        async def _ui() -> None:
            self._loop = asyncio.get_running_loop()
            finished = asyncio.Event()

            def _watch() -> None:
                self._finished.wait()
                with contextlib.suppress(RuntimeError):
                    self._loop.call_soon_threadsafe(finished.set)

            threading.Thread(target=_watch, daemon=True).start()

            session = PromptSession(
                multiline=True,
                key_bindings=kb,
                style=style,
                bottom_toolbar=self._toolbar,
            )
            self._app_ref = session.app

            worker = threading.Thread(target=_worker, name="cfd-study", daemon=True)
            worker.start()

            while not self._finished.is_set():
                prompt_task = asyncio.ensure_future(
                    session.prompt_async(self._message, refresh_interval=0.12)
                )
                done_task = asyncio.ensure_future(finished.wait())
                done, _pending = await asyncio.wait(
                    {prompt_task, done_task}, return_when=asyncio.FIRST_COMPLETED
                )
                if prompt_task in done:
                    done_task.cancel()
                    try:
                        text = prompt_task.result()
                    except (EOFError, KeyboardInterrupt):
                        self._quit()
                        break
                    self._handle_line(text)
                else:
                    prompt_task.cancel()
                    with contextlib.suppress(BaseException):
                        await prompt_task
                    break

        if banner:
            print(banner, flush=True)
        with patch_stdout(raw=True):
            asyncio.run(_ui())
        self.shutdown()


def drain_steering_messages() -> List[str]:
    out = list(STEERING_QUEUE)
    STEERING_QUEUE.clear()
    return out
