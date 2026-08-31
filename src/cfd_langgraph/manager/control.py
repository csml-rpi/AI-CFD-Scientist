from __future__ import annotations

import threading
from typing import Any, Callable, Dict, Iterable, List, Optional

from deepagents import FilesystemPermission

# deepagents ships its own built-in ls/read_file/write_file/edit_file/glob/grep
# tools, wired to whatever `backend=` is configured — we never pass one, so
# they'd silently operate against an empty in-memory StateBackend, not the
# real disk. That's worse than them being absent: a call like `grep` against
# nothing returns a clean, successful-looking "no matches" instead of an
# error, which reads as a real (if unhelpful) answer rather than the decoy it
# is. deny-all on every path pushes every model onto our real, disk-backed
# equivalents (list_directory, read_text_file, grep_files, ...) instead,
# turning a silent wrong answer into an explicit, honest permission error.
DENY_BUILTIN_FILESYSTEM_TOOLS: List[FilesystemPermission] = [
    FilesystemPermission(operations=["read", "write"], paths=["/**"], mode="deny")
]


class InterruptFlag:
    """A process-wide flag requesting a pause at the next tool-call boundary.

    Deliberately never aborts a call already in flight. LangGraph checkpoints
    at completed steps, not mid-tool-call — so the only way to guarantee zero
    information loss is to never interrupt *during* a tool call, only ever
    *before* the next one starts. Setting this flag mid-way through a
    multi-hour ``run_case_native`` call does not stop that call; it lets it
    finish normally, then pauses before whatever would run next. That's the
    deliberate tradeoff: not instant, but nothing partial ever gets thrown
    away. See ``manager/deep_agent.py`` for how every tool gets wired to
    check this via ``InterruptOnConfig.when``, and ``cli/repl.py`` for how
    Ctrl-C sets it.
    """

    def __init__(self) -> None:
        self._event = threading.Event()

    def request(self) -> None:
        self._event.set()

    def clear(self) -> None:
        self._event.clear()

    def is_set(self) -> bool:
        return self._event.is_set()


# One per process. The CLI's SIGINT handler sets it; every tool's
# interrupt_on `when` predicate reads it; the CLI's interrupt handler clears
# it once the user decides what to do next (or leaves it set for
# single-step mode — see cli/repl.py `step`).
GLOBAL_INTERRUPT = InterruptFlag()


def pause_requested(_request: object = None) -> bool:
    """The `when` predicate every tool's interrupt_on entry uses."""
    return GLOBAL_INTERRUPT.is_set()


# deepagents' own built-in tools, which never appear in the repo's tool lists
# and so were invisible to build_interrupt_on. `task` is the important one: it
# is how the manager launches every case and every OED candidate, so a Ctrl-C
# arriving just before a fan-out used to sail straight past the one boundary
# where pausing is cheapest — the launches went ahead, and the pause instead
# landed inside each spawned subagent, several hours later and N at a time.
_BUILTIN_INTERRUPTIBLE_TOOLS = ("task",)


def build_interrupt_on(
    tool_fns: Iterable[Callable[..., Any]],
    fixed: Optional[Dict[str, Dict[str, Any]]] = None,
    include_builtins: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """Build an ``interrupt_on`` dict covering every tool in ``tool_fns``.

    ``fixed`` entries (e.g. the hypothesis-approval gate) always interrupt —
    passed through unchanged. Every other tool gets a ``when``-gated entry:
    normally a no-op (the predicate returns False, so the call proceeds
    untouched), and only actually pauses the graph when
    ``GLOBAL_INTERRUPT.is_set()`` — i.e. only after the user has pressed
    Ctrl-C in the CLI. This is what makes "any tool call, on demand" possible
    without listing tool names up front for what to watch — every tool is
    watched, all the time, at negligible cost until the flag is actually set.
    """
    fixed = fixed or {}
    interrupt_on: Dict[str, Dict[str, Any]] = dict(fixed)
    names = [getattr(fn, "__name__", None) for fn in tool_fns]
    if include_builtins:
        names.extend(_BUILTIN_INTERRUPTIBLE_TOOLS)
    for name in names:
        if not name or name in interrupt_on:
            continue
        interrupt_on[name] = {
            "allowed_decisions": ["approve", "reject"],
            "description": "Paused before this tool call — Ctrl-C was pressed in the CLI.",
            "when": pause_requested,
        }
    return interrupt_on
