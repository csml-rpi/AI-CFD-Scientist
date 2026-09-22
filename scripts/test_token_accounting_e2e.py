#!/usr/bin/env python3
"""End-to-end token accounting: every billed response is logged exactly once, exactly.

Every call path CFD Scientist uses goes through the real factory and the real LangChain
machinery; only the provider's HTTP reply is replaced, by a streamed reply shaped like the
real Responses API (text / function-call events, then a `response.completed` event that
carries the response id and its usage). Each fake reply bills a distinct, known number of
tokens, so the ground truth is exact and no quota is spent.

The invariant checked at the end, over every case together:
    each billed response id appears in exactly one logged row, and the logged
    input / cached / output / reasoning tokens equal what was billed, to the token.

Cases: plain call, tool call, structured output (both the model method and the
utils.structured_output wrapper the pipeline uses), an empty turn that is retried, a reply
with no usage block, stream(), ainvoke(), batch(), an image (vision) message, two agents
logging concurrently from threads, and one turn of the real manager graph.
"""
from __future__ import annotations

import asyncio
import itertools
import json
import os
import sys
import tempfile
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.environ.update(CFD_SCIENTIST_LLM_PROVIDER="openai-codex", CFD_SCIENTIST_MODEL="gpt-5.6-sol",
                  CFD_SCIENTIST_EFFORT="none", CFD_SCIENTIST_STAGE="test")

from pydantic import BaseModel  # noqa: E402

from cfd_langgraph.llm import codex_oauth  # noqa: E402

fails: list[str] = []


def check(label, got, want):
    ok = got == want
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + ("" if ok else f"\n        got={got}\n        want={want}"))
    if not ok:
        fails.append(label)


# ---------------------------------------------------------------- the fake provider
BILLED: list[dict] = []           # ground truth: every response the provider "billed"
_SCRIPT: list[dict] = []          # what the next replies should contain
_LOCK = threading.Lock()
_IDS = itertools.count(1)


class _FakeStream:
    ok = True
    status_code = 200

    def __init__(self, events):
        self._lines = [f"data: {json.dumps(e)}" for e in events] + ["data: [DONE]"]
        self.text = ""

    def iter_lines(self, decode_unicode=True):
        yield from self._lines

    def json(self):
        return {}

    def close(self):
        pass


def _fake_post(url, headers=None, json=None, timeout=None, stream=False):  # noqa: A002
    with _LOCK:
        spec = _SCRIPT.pop(0) if _SCRIPT else {"text": "ok"}
        n = next(_IDS)
    rid = f"resp_test_{n:04d}"
    # Distinct, recognisable numbers per response so a wrong sum cannot hide.
    usage = {"input_tokens": 1000 * n + 7, "output_tokens": 10 * n + 3,
             "input_tokens_details": {"cached_tokens": 900 * n},
             "output_tokens_details": {"reasoning_tokens": n % 3},
             "total_tokens": 1000 * n + 7 + 10 * n + 3}
    events, output = [], []
    if spec.get("text"):
        events.append({"type": "response.output_text.delta", "delta": spec["text"]})
        output.append({"type": "message", "content": [{"type": "output_text", "text": spec["text"]}]})
    if spec.get("tool"):
        item = {"type": "function_call", "name": spec["tool"], "arguments": json_dumps(spec.get("args", {})),
                "call_id": f"call_{n}"}
        events.append({"type": "response.output_item.done", "item": item})
        output.append(item)
    body = {"id": rid, "output": output}
    if not spec.get("no_usage"):
        body["usage"] = usage
    events.append({"type": "response.completed", "response": body})
    with _LOCK:
        BILLED.append({"id": rid, "usage": None if spec.get("no_usage") else usage})
    return _FakeStream(events)


def json_dumps(o):
    return json.dumps(o)


codex_oauth.requests.post = _fake_post


def script(*specs):
    with _LOCK:
        _SCRIPT.extend(specs)


# ---------------------------------------------------------------- setup
from cfd_langgraph.llm import token_usage_logger as tul  # noqa: E402
from cfd_langgraph.llm.factory import create_langchain_llm  # noqa: E402
from cfd_langgraph.utils import structured_output  # noqa: E402

LOG = Path(tempfile.mkdtemp())
os.environ["CFD_TOKEN_LOG_DIR"] = str(LOG)


def rows():
    p = LOG / "llm_token_usage_calls.jsonl"
    return [json.loads(l) for l in p.open()] if p.exists() else []


def new_rows(before):
    return rows()[before:]


class Answer(BaseModel):
    answer: int


llm = create_langchain_llm("gpt-5.6-sol", temperature=0.0, agent="tester")

from langchain_core.tools import tool  # noqa: E402


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


print("== single-call paths: one row each, attributed to the caller")
cases = [
    ("plain invoke",               lambda: llm.invoke("hi"),                                    [{"text": "ok"}]),
    ("tool call (bind_tools)",     lambda: llm.bind_tools([add]).invoke("add 2 and 3"),         [{"tool": "add", "args": {"a": 2, "b": 3}}]),
    ("structured output (model)",  lambda: llm.with_structured_output(Answer).invoke("2+3?"),   [{"text": '{"answer": 5}'}]),
    ("structured output (utils)",  lambda: structured_output(llm, Answer).invoke("2+3?"),       [{"text": '{"answer": 5}'}]),
    ("stream()",                   lambda: list(llm.stream("hi")),                              [{"text": "ok"}]),
    ("ainvoke()",                  lambda: asyncio.run(llm.ainvoke("hi")),                      [{"text": "ok"}]),
    ("vision message",             lambda: llm.invoke([{"role": "user", "content": [
                                        {"type": "text", "text": "what is this"},
                                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}}]}]),
                                                                                                [{"text": "a picture"}]),
]
for label, fn, spec in cases:
    before = len(rows()); billed_before = len(BILLED)
    script(*spec)
    fn()
    got = new_rows(before)
    billed = BILLED[billed_before:]
    check(f"{label}: one row per billed response", len(got), len(billed))
    if got:
        check(f"{label}: agent label", got[0].get("agent"), "tester")

print("== an empty turn is retried: one logical call, every attempt's tokens")
before = len(rows()); billed_before = len(BILLED)
script({"text": ""}, {"text": "ok"})
llm.invoke("hi")
got, billed = new_rows(before), BILLED[billed_before:]
check("two responses billed", len(billed), 2)
check("one row logged for the call", len(got), 1)
if got:
    check("row input = sum of both attempts", got[0]["input_tokens"], sum(b["usage"]["input_tokens"] for b in billed))
    check("row records 2 attempts", got[0].get("attempts"), 2)
    check("row carries both response ids", sorted(got[0].get("response_ids") or []), sorted(b["id"] for b in billed))

print("== a reply without usage falls back to an estimate, labelled as one")
before = len(rows())
script({"text": "ok", "no_usage": True})
llm.invoke("hi")
got = new_rows(before)
check("labelled estimate", got[0].get("token_source") if got else None, "estimate")

print("== batch of three: three rows")
before = len(rows()); billed_before = len(BILLED)
script({"text": "a"}, {"text": "b"}, {"text": "c"})
llm.batch(["1", "2", "3"])
check("rows == billed", len(new_rows(before)), len(BILLED) - billed_before)

print("== two agents calling concurrently from threads: no cross-attribution")
a_llm = create_langchain_llm("gpt-5.6-sol", temperature=0.0, agent="case-runner")
b_llm = create_langchain_llm("gpt-5.6-sol", temperature=0.0, agent="oed-candidate-runner")
before = len(rows())
script(*[{"text": "ok"}] * 40)
ts = [threading.Thread(target=lambda m=m: [m.invoke("x") for _ in range(20)]) for m in (a_llm, b_llm)]
[t.start() for t in ts]; [t.join() for t in ts]
got = new_rows(before)
check("case-runner rows", sum(r["agent"] == "case-runner" for r in got), 20)
check("oed-candidate-runner rows", sum(r["agent"] == "oed-candidate-runner" for r in got), 20)

print("== one turn of the real manager graph")
try:
    from cfd_langgraph.config import get_settings
    from cfd_langgraph.manager import build_manager
    with tempfile.TemporaryDirectory() as out:
        before = len(rows()); billed_before = len(BILLED)
        script({"text": "Nothing to do; stopping."})
        graph, stack = build_manager(get_settings(), out)
        graph.invoke({"messages": [{"role": "user", "content": "Say hello and stop."}]},
                     config={"configurable": {"thread_id": "tok-e2e"}, "recursion_limit": 6})
        stack.close()
        got, billed = new_rows(before), BILLED[billed_before:]
        check("graph: rows == billed responses", len(got), len(billed))
        check("graph: made at least one model call", len(billed) >= 1, True)
        check("graph: attributed to the manager agent", {r["agent"] for r in got} <= {"manager"}, True)
except Exception as exc:  # pragma: no cover - reported, not hidden
    print(f"  FAIL  manager graph turn raised {type(exc).__name__}: {exc}")
    fails.append("manager graph turn")

print("== the invariant, over every case above")
logged = rows()
ids_logged = [i for r in logged for i in (r.get("response_ids") or [])]
billed_with_usage = [b for b in BILLED if b["usage"]]
check("every billed response is logged", sorted(set(ids_logged)), sorted(b["id"] for b in BILLED))
check("no response is logged twice", len(ids_logged), len(set(ids_logged)))
real_rows = [r for r in logged if r.get("token_source") in ("provider_usage", "provider_usage_partial")]
for key, src in (("input_tokens", "input_tokens"), ("output_tokens", "output_tokens"),
                 ("cached_input_tokens", ("input_tokens_details", "cached_tokens")),
                 ("reasoning_output_tokens", ("output_tokens_details", "reasoning_tokens"))):
    def billed_val(u):
        return u[src] if isinstance(src, str) else u[src[0]][src[1]]
    check(f"logged {key} == billed", sum(r[key] for r in real_rows), sum(billed_val(b["usage"]) for b in billed_with_usage))
t = json.loads((LOG / "llm_token_usage.json").read_text())["totals"]
check("totals.calls == rows", t["calls"], len(logged))
check("totals.input == sum of rows", t["input_tokens"], sum(r["input_tokens"] for r in logged))

print()
print(f"billed responses: {len(BILLED)}   logged rows: {len(logged)}")
print("FAILURES:", fails if fails else "none")
sys.exit(1 if fails else 0)
