#!/usr/bin/env python3
"""Removing visualization instructions from a generated requirement, and how
an empty reply to a plain call is retried.

On shelf_demo_dafoam_20261019_1057 the step sent the model the whole
requirement to return verbatim; the model often returned nothing, and each
empty reply was retried five times with waits up to 120 s, for hours in all,
although an empty reply here only means "keep the text". Now the model names
the sentences to remove, the step is skipped when no visual word appears, and
the step's own call gets one quick retry (every other call keeps the full ladder:
given to all calls without tools, it failed a DAFoam case whose setup got two
empty replies in a row).

Run: python scripts/test_requirement_strip.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from langchain_core.messages import AIMessage  # noqa: E402

from cfd_langgraph.agents.hypothesis_agent import HypothesisAgent  # noqa: E402
import cfd_langgraph.llm.factory as factory  # noqa: E402

FAILURES: list[str] = []


def check(name: str, got, want=True) -> None:
    ok = got == want
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + ("" if ok else f" — got {got!r}, want {want!r}"))
    if not ok:
        FAILURES.append(name)


class FakeLLM:
    def __init__(self, reply: str):
        self.reply, self.calls = reply, 0

    def invoke(self, prompt):
        self.calls += 1
        return AIMessage(content=self.reply)


def agent_with(reply: str) -> tuple[HypothesisAgent, FakeLLM]:
    agent = HypothesisAgent.__new__(HypothesisAgent)
    agent.llm = FakeLLM(reply)
    return agent, agent.llm


REQ = ("Run DASimpleFoam on the 343-cell channel with inlet 10 m/s. Converge to 1e-12. "
       "Plot the velocity contours on the mid-plane and save them as PNG. Write CD to results.json.")
PLAIN = "Run DASimpleFoam on the 343-cell channel with inlet 10 m/s. Write CD to results.json."

agent, llm = agent_with("NONE")
check("a requirement with no visual word is kept without a model call",
      (agent._strip_visualization_mentions(PLAIN), llm.calls), (PLAIN, 0))
agent, llm = agent_with("Plot the velocity contours on the mid-plane and save them as PNG.")
out = agent._strip_visualization_mentions(REQ)
check("a named plotting sentence is removed", "Plot the velocity" in out, False)
check("and the rest is kept as written", ("inlet 10 m/s" in out, "1e-12" in out, "results.json" in out),
      (True, True, True))
agent, _ = agent_with("")
check("an empty reply keeps the text", agent._strip_visualization_mentions(REQ), REQ)
agent, _ = agent_with("NONE")
check("NONE keeps the text exactly", agent._strip_visualization_mentions(REQ), REQ)
agent, _ = agent_with("\n".join(s.strip() + "." for s in REQ.split(".") if s.strip()))
check("a reply naming most of the text keeps the original", agent._strip_visualization_mentions(REQ), REQ)

print("== empty replies to plain calls")
calls = {"n": 0}


def empty():
    calls["n"] += 1
    return "", []


import time as _t  # noqa: E402
real_sleep = _t.sleep
_t.sleep = lambda s: None
try:
    try:
        with factory.quick_empty_replies():
            factory._retry_empty_turn(empty, "test")
    except RuntimeError:
        pass
    check("a call inside quick_empty_replies is tried twice", calls["n"], factory._QUICK_EMPTY_ATTEMPTS)
    calls["n"] = 0
    try:
        factory._retry_empty_turn(empty, "test")
    except RuntimeError:
        pass
    check("every other call, tools or not, keeps the full ladder", calls["n"], factory._EMPTY_TURN_ATTEMPTS)
    print("== a retry after an empty reply is not the identical request")
    from types import SimpleNamespace
    from langchain_core.messages import HumanMessage as _H

    seen: list = []

    class _FakeCodex:
        def invoke(self, messages, items=None, tools=None, tool_choice=None):
            seen.append(messages)
            return SimpleNamespace(content="" if len(seen) == 1 else "the answer", tool_calls=[],
                                   usage=None, response_id=None)

    model = factory.CodexOAuthChatModel(codex_wrapper=_FakeCodex(), model_name="m")
    reply = model.invoke([_H(content="question")])
    check("the second attempt answers", reply.content, "the answer")
    check("the first attempt is sent as it is", len(seen[0]), 1)
    check("the retry carries one more line saying the reply was empty",
          (len(seen[1]), factory._EMPTY_RETRY_NOTE in str(seen[1][-1])), (2, True))
finally:
    _t.sleep = real_sleep

print()
print("FAILURES:", FAILURES if FAILURES else "none")
sys.exit(1 if FAILURES else 0)
