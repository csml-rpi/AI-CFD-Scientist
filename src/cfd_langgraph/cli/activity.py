"""A process-wide, thread-safe record of what the study is doing right now.

The CLI's status line needs to answer "is anything happening, and what?" at
any instant. ``graph.stream(..., stream_mode="updates")`` cannot answer that:
it reports a node only once the node has *finished*, so a three-hour
``run_case_native`` produces no signal at all until it is over. The honest
source of that information is the tool wrapper itself
(``manager/tools.py::_with_progress``), which brackets the real function
call — so it reports here on entry and on exit, and the UI just reads.

Concurrency is the normal case, not an edge case: the manager launches cases
and OED candidates as several concurrent ``task`` calls, so several tools are
genuinely in flight at once and each runs on its own thread. Hence a dict of
live spans keyed by a unique token rather than a single "current activity".
"""

from __future__ import annotations

import itertools
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class Span:
    """One tool call that has started and not yet finished."""

    token: int
    label: str
    detail: str = ""
    started_at: float = field(default_factory=time.monotonic)

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started_at


class ActivityBoard:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._spans: Dict[int, Span] = {}
        self._ids = itertools.count(1)
        self._completed = 0
        self._failed = 0
        self._note: str = ""
        self._started_at = time.monotonic()

    # -- writing (called from worker threads) ---------------------------

    def start(self, label: str, detail: str = "") -> int:
        with self._lock:
            token = next(self._ids)
            self._spans[token] = Span(token=token, label=label, detail=detail)
            return token

    def finish(self, token: int, ok: bool = True) -> None:
        with self._lock:
            self._spans.pop(token, None)
            if ok:
                self._completed += 1
            else:
                self._failed += 1

    def note(self, text: str) -> None:
        """A short free-text line for the status bar (stage name, phase, ...)."""
        with self._lock:
            self._note = text

    # -- reading (called from the UI thread) ----------------------------

    def snapshot(self) -> Tuple[List[Span], int, int, str, float]:
        with self._lock:
            spans = sorted(self._spans.values(), key=lambda s: s.started_at)
            return spans, self._completed, self._failed, self._note, time.monotonic() - self._started_at

    def is_busy(self) -> bool:
        with self._lock:
            return bool(self._spans)

    def reset_clock(self) -> None:
        with self._lock:
            self._started_at = time.monotonic()


BOARD = ActivityBoard()

# Messages the user typed while the study was busy, waiting to be delivered at
# the next tool-call boundary. Lives here rather than in ``cli/ui.py`` so the
# interrupt handler in ``cli/repl.py`` can read it without importing
# prompt_toolkit, which is optional (see ``ui.available()``).
STEERING_QUEUE: List[str] = []


def format_duration(seconds: float) -> str:
    """Compact and stable in width: 42s, 7m12s, 3h04m."""
    seconds = max(0.0, seconds)
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m{int(seconds % 60):02d}s"
    return f"{int(seconds // 3600)}h{int((seconds % 3600) // 60):02d}m"


# Braille dots, the same cycle Claude Code's spinner uses. Purely cosmetic —
# every real progress number on the status line comes from ActivityBoard, so
# a spinning frame never implies work that is not actually happening: when
# nothing is in flight the UI shows an idle glyph instead of animating.
SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


def spinner_frame(now: Optional[float] = None, period: float = 0.09) -> str:
    now = time.monotonic() if now is None else now
    return SPINNER_FRAMES[int(now / period) % len(SPINNER_FRAMES)]
