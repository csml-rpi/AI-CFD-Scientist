#!/usr/bin/env python3
"""Tool calling on the subscription-backed providers (`claude-code`, `openai-codex`).

Offline. The message streams and HTTP responses are simulated, so this runs
without a subscription and without spending quota; the live round-trips were
verified separately against both backends.

What matters here is that `bind_tools` exists at all — `create_deep_agent` is a
tool-calling agent end to end, and `deep_agent.py:_require_tool_calling_support`
refuses a model that inherits the raising `BaseChatModel.bind_tools`. Both
wrappers used to be in exactly that position, so neither could run a study.

Run: python scripts/test_subscription_providers.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from langchain_core.language_models.chat_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage  # noqa: E402

from cfd_langgraph.llm import tool_bridge  # noqa: E402

FAILURES: List[str] = []


def check(name: str, cond: object, detail: str = "") -> None:
    if cond:
        print(f"[PASS] {name}")
    else:
        FAILURES.append(name)
        print(f"[FAIL] {name}" + (f" — {detail}" if detail else ""))


CONVERSATION = [
    SystemMessage(content="You run one CFD study."),
    HumanMessage(content="Run the mesh gate."),
    AIMessage(
        content="",
        tool_calls=[{"name": "run_mesh_gate", "args": {"group": "hill"}, "id": "call_1", "type": "tool_call"}],
    ),
    ToolMessage(content="converged", tool_call_id="call_1", name="run_mesh_gate"),
]


# ---------------------------------------------------------------------------
# shared bridge
# ---------------------------------------------------------------------------


def test_responses_input_conversion() -> None:
    items = tool_bridge.messages_to_responses_input(CONVERSATION)
    kinds = [it.get("type") or it.get("role") for it in items]
    check("system and user messages become role items", kinds[:2] == ["system", "user"], detail=str(kinds))
    check("a prior tool call becomes a function_call item", "function_call" in kinds, detail=str(kinds))
    check("a tool result becomes a function_call_output item", "function_call_output" in kinds, detail=str(kinds))

    call = next(it for it in items if it.get("type") == "function_call")
    output = next(it for it in items if it.get("type") == "function_call_output")
    check("the call carries the tool name", call["name"] == "run_mesh_gate")
    check("arguments are JSON-encoded", json.loads(call["arguments"]) == {"group": "hill"})
    check(
        "call_id links the call to its result",
        call["call_id"] == output["call_id"] == "call_1",
        detail=f"{call['call_id']} vs {output['call_id']}",
    )


def test_rendered_conversation_keeps_tool_history() -> None:
    """Dropping tool history makes the model repeat calls it already made."""
    system, rest = tool_bridge.split_system(CONVERSATION)
    check("the system message is split out for the SDK option", system == "You run one CFD study.")
    check("it is not left in the rendered messages", all(not isinstance(m, SystemMessage) for m in rest))

    text = tool_bridge.render_conversation(rest)
    check("the rendered transcript names the tool that was called", "run_mesh_gate" in text)
    check("it carries the arguments", '"group": "hill"' in text or '"group":"hill"' in text)
    check("it carries the result", "converged" in text)


def test_mcp_name_round_trip() -> None:
    """LangGraph matches tool calls by name, so the SDK's namespacing has to
    be undone or every call it returns is unroutable."""
    mangled = tool_bridge.mcp_tool_name("run_mesh_gate")
    check("SDK tool names are namespaced", mangled == "mcp__cfd__run_mesh_gate")
    check("and unmangled on the way back", tool_bridge.strip_mcp_prefix(mangled) == "run_mesh_gate")
    check("a name without the prefix is left alone", tool_bridge.strip_mcp_prefix("plain") == "plain")


def test_async_bridge_works_inside_a_running_loop() -> None:
    """LangGraph calls models from both plain threads and async contexts."""

    async def coro() -> str:
        await asyncio.sleep(0)
        return "done"

    check("works with no running loop", tool_bridge.run_coroutine_blocking(coro) == "done")

    async def outer() -> str:
        # asyncio.run() and new_event_loop().run_until_complete() both raise
        # here; the dedicated-thread bridge must not.
        return tool_bridge.run_coroutine_blocking(coro)

    check("works from inside a running loop", asyncio.run(outer()) == "done")

    async def boom() -> None:
        raise ValueError("propagate me")

    try:
        tool_bridge.run_coroutine_blocking(boom)
    except ValueError as exc:
        check("exceptions propagate to the caller", str(exc) == "propagate me")
    else:
        check("exceptions propagate to the caller", False, detail="no exception raised")


# ---------------------------------------------------------------------------
# claude-code
# ---------------------------------------------------------------------------


@dataclass
class _FakeThinking:
    thinking: str = "..."


@dataclass
class _FakeText:
    text: str


@dataclass
class _FakeToolUse:
    name: str
    input: dict
    id: str


@dataclass
class _FakeAssistant:
    content: list


@dataclass
class _FakeResult:
    subtype: str = "success"


def _fake_sdk_scripts(scripts: List[List[Any]]):
    """A fake query that plays a different message list on each call."""

    class _Q:
        def __init__(self) -> None:
            self.calls = 0
            self.prompts: List[str] = []

        def __call__(self, *, prompt: Any, options: Any = None):
            script = scripts[min(self.calls, len(scripts) - 1)]
            self.calls += 1

            async def gen():
                async for item in prompt:
                    self.prompts.append(item["message"]["content"])
                    break
                for m in script:
                    yield m

            return gen()

    return _Q()


def _fake_sdk_stream(messages: List[Any]):
    async def query(*, prompt: Any, options: Any = None):
        async def gen():
            for m in messages:
                yield m

        return gen()

    async def call(*, prompt: Any, options: Any = None):
        pass

    class _Q:
        def __call__(self, *, prompt: Any, options: Any = None):
            async def gen():
                for m in messages:
                    yield m

            return gen()

    return _Q()


def _patched_sdk(monkey_messages: List[Any]):
    """Swap the SDK symbols `invoke_with_tools` imports for fakes."""
    import claude_agent_sdk as sdk

    saved = {
        "AssistantMessage": sdk.AssistantMessage,
        "TextBlock": sdk.TextBlock,
        "ToolUseBlock": sdk.ToolUseBlock,
        "ResultMessage": sdk.ResultMessage,
    }
    _ = monkey_messages
    sdk.AssistantMessage = _FakeAssistant
    sdk.TextBlock = _FakeText
    sdk.ToolUseBlock = _FakeToolUse
    return saved


def test_claude_thinking_block_does_not_swallow_the_turn() -> None:
    """Regression: extended thinking arrives as its own AssistantMessage
    holding a single ThinkingBlock. Stopping at "the first AssistantMessage"
    returned empty content and no tool calls — a clean-looking wrong answer."""
    from cfd_langgraph.llm.factory import _ClaudeCodeAgentWrapper

    import claude_agent_sdk as sdk

    saved = _patched_sdk([])
    try:
        wrapper = _ClaudeCodeAgentWrapper.__new__(_ClaudeCodeAgentWrapper)
        wrapper._model = ""
        wrapper._effort = None
        wrapper._ClaudeAgentOptions = lambda **kw: kw
        wrapper._query = _fake_sdk_stream(
            [
                _FakeAssistant(content=[_FakeThinking()]),  # thinking-only turn
                _FakeAssistant(
                    content=[_FakeToolUse(name="mcp__cfd__run_mesh_gate", input={"group": "hill"}, id="toolu_1")]
                ),
            ]
        )
        specs = [
            {
                "type": "function",
                "function": {
                    "name": "run_mesh_gate",
                    "description": "run it",
                    "parameters": {"type": "object", "properties": {"group": {"type": "string"}}},
                },
            }
        ]
        text, calls = wrapper.invoke_with_tools([HumanMessage(content="go")], specs)
        check("a thinking-only turn is skipped, not treated as the answer", len(calls) == 1, detail=str(calls))
        if calls:
            check("the tool name is unmangled", calls[0]["name"] == "run_mesh_gate", detail=calls[0]["name"])
            check("the arguments survive", calls[0]["args"] == {"group": "hill"})
            check("the SDK's tool-use id is preserved", calls[0]["id"] == "toolu_1")
        check("no text is invented", text == "")
    finally:
        for name, value in saved.items():
            setattr(sdk, name, value)


def test_claude_text_preamble_does_not_swallow_the_tool_call() -> None:
    """Regression from a real run that "finished" instantly having done nothing.

    Claude routinely narrates before acting, and the SDK delivers that as its
    own AssistantMessage: message 1 is `TextBlock("I'll read the starter
    folder.")`, message 2 is the `ToolUseBlock`. A stop rule that fired on
    "this turn produced text" returned an AIMessage with no tool calls, which
    LangGraph reads as "the agent is done" — so the study ended after one
    model turn, and it looked like clean completion rather than failure.
    """
    from cfd_langgraph.llm.factory import _ClaudeCodeAgentWrapper

    import claude_agent_sdk as sdk

    saved = _patched_sdk([])
    try:
        wrapper = _ClaudeCodeAgentWrapper.__new__(_ClaudeCodeAgentWrapper)
        wrapper._model = ""
        wrapper._effort = None
        wrapper._ClaudeAgentOptions = lambda **kw: kw
        wrapper._query = _fake_sdk_stream(
            [
                _FakeAssistant(content=[_FakeThinking()]),
                _FakeAssistant(content=[_FakeText(text="I'll read the starter folder.")]),
                _FakeAssistant(
                    content=[_FakeToolUse(name="mcp__cfd__read_starter_folder", input={"path": "p"}, id="toolu_9")]
                ),
            ]
        )
        specs = [{"type": "function", "function": {"name": "read_starter_folder", "description": "", "parameters": {}}}]
        text, calls = wrapper.invoke_with_tools([HumanMessage(content="go")], specs)
        check("a spoken preamble does not end the turn", len(calls) == 1, detail=str(calls))
        if calls:
            check("the tool call after the preamble survives", calls[0]["name"] == "read_starter_folder")
        check("the preamble text is kept alongside it", "starter folder" in text, detail=repr(text))
    finally:
        for name, value in saved.items():
            setattr(sdk, name, value)


def test_claude_text_only_turn_still_returns() -> None:
    """The other half: a turn that genuinely has no tool call must end, not
    hang waiting for one that never comes."""
    from cfd_langgraph.llm.factory import _ClaudeCodeAgentWrapper

    import claude_agent_sdk as sdk

    saved = _patched_sdk([])
    try:
        sdk.ResultMessage = _FakeResult
        wrapper = _ClaudeCodeAgentWrapper.__new__(_ClaudeCodeAgentWrapper)
        wrapper._model = ""
        wrapper._effort = None
        wrapper._ClaudeAgentOptions = lambda **kw: kw
        wrapper._query = _fake_sdk_stream(
            [_FakeAssistant(content=[_FakeText(text="The mesh gate converged.")]), _FakeResult()]
        )
        specs = [{"type": "function", "function": {"name": "a", "description": "", "parameters": {}}}]
        text, calls = wrapper.invoke_with_tools([HumanMessage(content="go")], specs)
        check("a text-only turn returns its text", "converged" in text, detail=repr(text))
        check("with no tool calls", calls == [])
    finally:
        for name, value in saved.items():
            setattr(sdk, name, value)


def test_claude_stops_at_the_first_tool_call() -> None:
    """The SDK would otherwise run its own agent loop and execute the tool
    itself, stepping past where LangGraph checkpoints and interrupts."""
    from cfd_langgraph.llm.factory import _ClaudeCodeAgentWrapper

    import claude_agent_sdk as sdk

    saved = _patched_sdk([])
    try:
        wrapper = _ClaudeCodeAgentWrapper.__new__(_ClaudeCodeAgentWrapper)
        wrapper._model = ""
        wrapper._effort = None
        wrapper._ClaudeAgentOptions = lambda **kw: kw
        wrapper._query = _fake_sdk_stream(
            [
                _FakeAssistant(content=[_FakeToolUse(name="mcp__cfd__a", input={}, id="t1")]),
                _FakeAssistant(content=[_FakeToolUse(name="mcp__cfd__b", input={}, id="t2")]),
            ]
        )
        specs = [{"type": "function", "function": {"name": "a", "description": "", "parameters": {}}}]
        _text, calls = wrapper.invoke_with_tools([HumanMessage(content="go")], specs)
        check("only the first turn's calls are returned", [c["id"] for c in calls] == ["t1"], detail=str(calls))
    finally:
        for name, value in saved.items():
            setattr(sdk, name, value)


# ---------------------------------------------------------------------------
# openai-codex
# ---------------------------------------------------------------------------


def test_codex_payload_carries_tools_and_allows_parallel_calls() -> None:
    """The manager is told to launch independent work as several tool calls in
    one message; parallel_tool_calls off would serialise every fan-out."""
    from cfd_langgraph.llm.codex_oauth import CodexResponsesWrapper

    wrapper = CodexResponsesWrapper(
        token="t", model="m", base_url="https://chatgpt.com/backend-api/codex", instructions="I"
    )
    tools = tool_bridge.responses_tool_specs(
        [{"type": "function", "function": {"name": "run_mesh_gate", "description": "d", "parameters": {}}}]
    )
    items = tool_bridge.messages_to_responses_input(CONVERSATION)
    payload = wrapper._build_payload(None, items=items, tools=tools)
    check("the tools array is sent", payload["tools"][0]["name"] == "run_mesh_gate")
    check("Responses tool specs are flat, not nested", "function" not in payload["tools"][0])
    check("parallel tool calls are allowed when tools are bound", payload["parallel_tool_calls"] is True)
    check("the prebuilt items are used verbatim", payload["input"] is items)

    plain = wrapper._build_payload([{"role": "user", "content": "hi"}])
    check("parallel tool calls stay off with no tools", plain["parallel_tool_calls"] is False)


def test_codex_decodes_function_calls() -> None:
    from cfd_langgraph.llm.codex_oauth import CodexResponsesWrapper

    decoded = CodexResponsesWrapper._decode_function_call(
        {"type": "function_call", "name": "run_mesh_gate", "arguments": '{"group":"hill"}', "call_id": "call_9"}
    )
    check("name, args and call_id are decoded", decoded == {"name": "run_mesh_gate", "args": {"group": "hill"}, "id": "call_9"}, detail=str(decoded))

    broken = CodexResponsesWrapper._decode_function_call(
        {"type": "function_call", "name": "x", "arguments": "{not json", "call_id": "c"}
    )
    check("malformed arguments keep the call rather than dropping it", broken["name"] == "x")
    check("and surface the raw string", broken["args"].get("__raw_arguments__") == "{not json")


def test_both_models_declare_bind_tools() -> None:
    """The check `deep_agent.py` runs before a study starts."""
    from cfd_langgraph.llm.factory import ClaudeCodeChatModel, CodexOAuthChatModel

    for cls in (ClaudeCodeChatModel, CodexOAuthChatModel):
        own = getattr(cls, "bind_tools", None) is not BaseChatModel.bind_tools
        check(f"{cls.__name__} overrides bind_tools", own)


def test_a_written_out_tool_call_is_caught_and_retried() -> None:
    """Regression from a real run on claude-opus-5.

    The transcript was rendered as a speaker log (`ASSISTANT called tool
    foo({...})`), and the model began writing that line as its answer instead
    of calling anything — then drifted into emitting raw `<parameter name=...>`
    markup and narrating a whole round of results that had never been produced.

    Passing that through hands LangGraph an AIMessage with no tool calls, which
    ends the agent's turn: prose claiming work was done, with nothing run.
    """
    from cfd_langgraph.llm.factory import _ClaudeCodeAgentWrapper

    import claude_agent_sdk as sdk

    saved = _patched_sdk([])
    try:
        sdk.ResultMessage = _FakeResult
        wrapper = _ClaudeCodeAgentWrapper.__new__(_ClaudeCodeAgentWrapper)
        wrapper._model = ""
        wrapper._effort = None
        wrapper._ClaudeAgentOptions = lambda **kw: kw
        fake = _fake_sdk_scripts(
            [
                [
                    _FakeAssistant(
                        content=[
                            _FakeText(
                                text='ASSISTANT called tool fetch_literature({"topic": "x"}) [id=toolu_1]'
                            )
                        ]
                    ),
                    _FakeResult(),
                ],
                [_FakeAssistant(content=[_FakeToolUse(name="mcp__cfd__fetch_literature", input={"topic": "x"}, id="t1")])],
            ]
        )
        wrapper._query = fake
        specs = [{"type": "function", "function": {"name": "fetch_literature", "description": "", "parameters": {}}}]
        _text, calls = wrapper.invoke_with_tools([HumanMessage(content="go")], specs)
        check("a transcribed tool call triggers a retry", fake.calls == 2, detail=f"{fake.calls} calls")
        check("the retry recovers a real tool call", [c["name"] for c in calls] == ["fetch_literature"], detail=str(calls))
        check(
            "the retry tells the model what went wrong",
            len(fake.prompts) > 1 and "<correction>" in fake.prompts[1],
        )
    finally:
        for name, value in saved.items():
            setattr(sdk, name, value)


def test_transcript_is_framed_as_data_not_a_log() -> None:
    """The format itself is the fix — a speaker log invites continuation."""
    text = tool_bridge.render_conversation(CONVERSATION[1:])
    check("tool results are tagged as results", "<tool_result " in text)
    check("prior calls are tagged as already made", "<tool_call_you_already_made " in text)
    check(
        "nothing reads as a line the model should write next",
        "ASSISTANT called tool" not in text and "ASSISTANT:" not in text,
        detail=text[:200],
    )
    check("it says not to write tool calls as text", "Do NOT write tool calls as text" in text)
    check("it says not to invent results", "Do NOT invent tool results" in text)
    for marker in ("ASSISTANT called tool", "<parameter name=", "<invoke name="):
        check(
            f"a written-out call containing {marker!r} is detected",
            tool_bridge.looks_like_a_written_tool_call(f"blah {marker} blah"),
        )
    check(
        "ordinary prose is not flagged",
        not tool_bridge.looks_like_a_written_tool_call("I will call the mesh gate next."),
    )


def test_invoke_accepts_every_shape_langchain_passes() -> None:
    """Regression: a bare string reached the model as one message per character.

    `llm.with_structured_output(Schema).invoke("some prompt")` is the most
    common call shape and six call sites in this repo use it — OED candidate
    proposal, and every stage of native case writing. `str` is a `Sequence`, so
    iterating it produced hundreds of one-letter turns. Nothing raised: the
    model answered, coherently, that it had been given no input, and the
    duplicate check silently approved every repeat.
    """
    from langchain_core.prompt_values import StringPromptValue

    from cfd_langgraph.llm.factory import _lc_messages_to_dicts

    check(
        "a bare string is one user message, not one per character",
        _lc_messages_to_dicts("hello world") == [{"role": "user", "content": "hello world"}],
        detail=str(_lc_messages_to_dicts("hello world"))[:120],
    )
    check(
        "a lone message is wrapped, not iterated",
        _lc_messages_to_dicts(HumanMessage(content="hi")) == [{"role": "user", "content": "hi"}],
    )
    check(
        "a PromptValue from a `prompt | llm` chain is expanded",
        _lc_messages_to_dicts(StringPromptValue(text="pv")) == [{"role": "user", "content": "pv"}],
    )
    check(
        "a message list still round-trips with roles intact",
        _lc_messages_to_dicts([SystemMessage(content="s"), HumanMessage(content="h")])
        == [{"role": "system", "content": "s"}, {"role": "user", "content": "h"}],
    )
    check(
        "an empty string does not become an empty conversation",
        _lc_messages_to_dicts("") == [{"role": "user", "content": ""}],
    )


def main() -> int:
    tests = (
        test_responses_input_conversion,
        test_rendered_conversation_keeps_tool_history,
        test_mcp_name_round_trip,
        test_async_bridge_works_inside_a_running_loop,
        test_claude_thinking_block_does_not_swallow_the_turn,
        test_claude_text_preamble_does_not_swallow_the_tool_call,
        test_claude_text_only_turn_still_returns,
        test_claude_stops_at_the_first_tool_call,
        test_a_written_out_tool_call_is_caught_and_retried,
        test_transcript_is_framed_as_data_not_a_log,
        test_codex_payload_carries_tools_and_allows_parallel_calls,
        test_codex_decodes_function_calls,
        test_both_models_declare_bind_tools,
        test_invoke_accepts_every_shape_langchain_passes,
    )
    for test in tests:
        try:
            test()
        except Exception as exc:  # noqa: BLE001 — a raising test is a failing test
            check(test.__name__, False, detail=f"{type(exc).__name__}: {exc}")
    if FAILURES:
        print(f"\n{len(FAILURES)} FAILURE(S): " + ", ".join(FAILURES))
        return 1
    print("\nALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
