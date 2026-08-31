#!/usr/bin/env python3
"""Real-pty tests for the CLI's live console (``src/cfd_langgraph/cli/ui.py``).

Driven through an actual pseudo-terminal rather than a mocked stdin, because
every behaviour under test is a terminal behaviour: whether the status bar
draws at all, whether a bracketed paste survives intact, whether Esc reaches
the app instead of the shell. A test that stubbed ``input()`` would pass while
all of that was broken — which is how the pre-console REPL shipped a paste
handler that silently truncated multi-paragraph prompts.

The child process runs the real ``ui.Console`` against a fake study, so the
console, the activity board and the interrupt flag are the genuine articles;
only the CFD work underneath is stubbed.

Run: python scripts/test_cli_console.py
"""

from __future__ import annotations

import os
import re
import select
import subprocess
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

ANSI = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]|\x1b\][^\x07]*\x07|\x1b[=>]|\r")


# --------------------------------------------------------------------------
# child: the thing being tested, run inside the pty
# --------------------------------------------------------------------------


def _topic_child() -> None:
    """The very first prompt of a session, before any study exists."""
    from cfd_langgraph.cli.repl import _read_topic_with_prompt_toolkit

    print(f"TOPIC_RECEIVED={_read_topic_with_prompt_toolkit()!r}")


def _child() -> None:
    import time as _time

    from cfd_langgraph.cli import ui
    from cfd_langgraph.cli.activity import BOARD, STEERING_QUEUE
    from cfd_langgraph.manager.control import GLOBAL_INTERRUPT

    console = ui.Console("/tmp/fake_out_dir")

    def drive() -> None:
        token = BOARD.start("run_mesh_gate", "physics_group='sa_hill'")
        print("STUDY: mesh gate started")
        for _ in range(300):  # stand-in for a long tool call
            if STEERING_QUEUE:
                break
            _time.sleep(0.1)
        BOARD.finish(token)
        print(f"STUDY: queued={list(STEERING_QUEUE)!r} interrupt={GLOBAL_INTERRUPT.is_set()}")
        answer = console.ask("\n> send / continue / quit\n> ")
        print(f"STUDY: decision={answer!r}")
        STEERING_QUEUE.clear()
        print("STUDY: now idle")
        print(f"STUDY: followup={console.next_message()!r}")
        print(f"STUDY: followup2={console.next_message()!r}")
        print("STUDY: done")

    console.run(drive, banner="BANNER-OK")
    print("CHILD-EXIT")


# --------------------------------------------------------------------------
# parent: drive the pty
# --------------------------------------------------------------------------


class Pty:
    def __init__(self, mode: str = "--child") -> None:
        import pty

        self.mode = mode
        self.buf = b""
        self.pid, self.fd = pty.fork()
        if self.pid == 0:
            os.environ.update(TERM="xterm-256color", COLUMNS="120", LINES="40")
            os.execv(sys.executable, [sys.executable, os.path.abspath(__file__), self.mode])
        # A pty starts 0x0 unless told otherwise, and prompt_toolkit renders to
        # the size it queries — not to $COLUMNS. Without this the child draws
        # its chrome for one width and renders into another.
        import fcntl
        import struct
        import termios

        fcntl.ioctl(self.fd, termios.TIOCSWINSZ, struct.pack("HHHH", 40, 120, 0, 0))

    def pump(self, seconds: float) -> None:
        """Read for a while, answering any cursor-position request (ESC[6n).

        A real terminal replies to CPR. Without a reply prompt_toolkit falls
        back to a reduced renderer that draws no bottom toolbar at all, so an
        unanswered CPR would make these tests blind to the status bar.
        """
        end = time.time() + seconds
        while time.time() < end:
            ready, _, _ = select.select([self.fd], [], [], 0.05)
            if not ready:
                continue
            try:
                chunk = os.read(self.fd, 65536)
            except OSError:
                break
            if not chunk:
                break
            self.buf += chunk
            if b"\x1b[6n" in chunk:
                os.write(self.fd, b"\x1b[24;1R")

    def text(self) -> str:
        """Everything seen so far, control sequences stripped.

        Matching must happen on stripped text: prompt_toolkit interleaves SGR
        codes *inside* the toolbar (the tool name is bold), so a raw search for
        "run_mesh_gate ·" misses even when the bar is plainly on screen.
        """
        return ANSI.sub("", self.buf.decode(errors="replace"))

    def wait_for(self, marker: str, timeout: float = 20.0) -> bool:
        end = time.time() + timeout
        while time.time() < end:
            if marker in self.text():
                return True
            self.pump(0.2)
        return False

    def send(self, data: str) -> None:
        os.write(self.fd, data.encode())

    def close(self) -> None:
        try:
            os.close(self.fd)
        except OSError:
            pass
        try:
            os.waitpid(self.pid, 0)
        except Exception:
            pass


RESULTS: list[tuple[str, bool]] = []


def check(name: str, cond: object, detail: str = "") -> None:
    RESULTS.append((name, bool(cond)))
    print(("PASS  " if cond else "FAIL  ") + name + (f" — {detail}" if not cond and detail else ""))


def test_topic_prompt_is_framed() -> None:
    """The first thing a session shows must be the same input box as the rest.

    It was a bare `>` for two rounds: the framed chrome only existed inside the
    running session, so the prompt you actually start at looked like a
    different program.
    """
    term = Pty("--topic-child")
    try:
        # Generous: this child imports the whole manager stack before it draws.
        if not term.wait_for("╭", timeout=60.0):
            check("the topic prompt is drawn inside a box", False, detail=term.text()[-500:])
            return
        term.pump(1.0)
        screen = term.text()
        check("the topic prompt is drawn inside a box", "╭" in screen and "╰" in screen)
        check("the caret sits inside the box", "│ ›" in screen)
        check("it says what it wants", "investigate" in screen)
        check("and how to send", "⏎ send" in screen)

        term.send("\x1b[200~line one\n\nline three\x1b[201~")
        term.pump(1.0)
        check("a multi-line paste does not submit at its first newline",
              "TOPIC_RECEIVED" not in term.text())
        term.send("\r")
        term.wait_for("TOPIC_RECEIVED", timeout=15.0)
        received = term.text().split("TOPIC_RECEIVED=")[-1]
        check("the pasted topic arrives whole", "line one" in received and "line three" in received)
        check("blank lines inside the paste survive", "\\n\\n" in received, detail=received[:120])
    finally:
        term.close()


def main() -> int:
    term = Pty()
    try:
        if not term.wait_for("run_mesh_gate ·"):
            print("FAIL  status bar never appeared\n" + term.text()[-4000:])
            return 1
        term.pump(1.5)
        busy = term.text()

        # typing while the study is busy must queue, not send
        term.send("please switch to the coarse mesh\r")
        assert term.wait_for("✎ queued: please switch"), "steering message never queued"
        queued = term.text()

        # the decision the study then raises
        assert term.wait_for("> send / continue / quit"), "decision prompt never shown"
        term.pump(0.5)
        term.send("send\r")
        assert term.wait_for("STUDY: decision="), "decision never delivered"

        # a slash command while idle runs locally instead of being sent as chat
        assert term.wait_for("STUDY: now idle"), "study never went idle"
        term.pump(0.5)
        term.send("/status\r")
        assert term.wait_for("nothing running right now"), "/status did not run locally"

        # a multi-line bracketed paste must not submit at its first newline
        term.send("\x1b[200~line one\nline two\n\nline four\x1b[201~")
        term.pump(1.0)
        mid = term.text()
        term.send("\r")
        assert term.wait_for("STUDY: followup="), "pasted follow-up never delivered"

        term.send("\x1b")
        assert term.wait_for("pause requested"), "esc did not request a pause"
        term.send("/quit\r")
        term.wait_for("CHILD-EXIT", timeout=10.0)
        term.pump(1.0)
        out = term.text()
    finally:
        term.close()

    check("banner shown", "BANNER-OK" in out)
    check("status bar names the running tool", "run_mesh_gate" in busy)
    check("status bar shows elapsed time", re.search(r"run_mesh_gate.{0,40}\d+s", busy, re.S))
    check("spinner animates", any(f in busy for f in "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"))
    check("status bar shows the interrupt hint", "esc pause" in busy)
    check("hint flips once a pause is requested", "pause requested — finishing" in out)
    check("typing while busy queues, does not send", "queued: please switch to the coarse mesh" in queued)
    check("queueing requests a pause", "queued=['please switch to the coarse mesh'] interrupt=True" in out)
    check("decision line routed to ask()", "STUDY: decision='send'" in out)
    check("the ask() menu is printed, not swallowed", "> send / continue / quit" in out)
    check("/status handled locally while idle", "nothing running right now" in out)
    check("/status not sent to the study as chat", "followup='/status'" not in out)
    check("multi-line paste not submitted early", "STUDY: followup=" not in mid)
    check("multi-line paste kept whole", "followup='line one\\nline two\\n\\nline four'" in out)
    check("esc requests a pause", "pause requested" in out)
    check("/quit ends the session", "CHILD-EXIT" in out)
    check("no deadlock: ask() released on shutdown", "STUDY: followup2=None" in out)

    test_topic_prompt_is_framed()

    failures = [name for name, ok in RESULTS if not ok]
    if failures:
        print("\n" + f"{len(failures)} FAILURE(S): " + ", ".join(failures))
        print("\n----- console output -----\n" + out[-7000:])
        return 1
    print("\nALL PASS")
    return 0


if __name__ == "__main__":
    if "--topic-child" in sys.argv:
        _topic_child()
    elif "--child" in sys.argv:
        _child()
    else:
        sys.exit(main())
