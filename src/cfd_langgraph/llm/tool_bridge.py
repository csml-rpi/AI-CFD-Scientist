"""Tool-calling support for the two subscription-backed providers.

``create_deep_agent`` is a tool-calling agent from top to bottom, so a chat
model that cannot return tool calls cannot drive a study at all — which is why
``deep_agent.py:_require_tool_calling_support`` refuses one up front. Both
subscription wrappers (``claude-code``, ``openai-codex``) were in exactly that
position: they inherited ``BaseChatModel.bind_tools``, which raises.

Both backends do support real, native tool calling; it just wasn't wired:

* The ChatGPT/Codex Responses backend accepts a ``tools`` array and streams
  back ``function_call`` items carrying a ``call_id`` — verified live.
* The Claude Agent SDK exposes tools through an in-process MCP server and
  yields ``ToolUseBlock``s on the assistant turn — also verified live.

That matters more than it might look. The alternative — asking the model to
emit tool calls as JSON in its text and parsing them out — is the failure mode
already costing this project real money on Gemini
(``MALFORMED_FUNCTION_CALL``, see ``code_mod_agentic.py``'s retry ladder).
Going native on both providers avoids importing that problem into a third one.

The two directions are asymmetric on purpose:

* **Out of the model** (the model's tool call) is native on both. This is the
  half that has to be parsed, so it is the half that must not be guessed at.
* **Into the model** (prior tool results) is native for Codex, which has
  first-class ``function_call``/``function_call_output`` input items, and a
  rendered transcript for Claude, whose SDK takes a prompt rather than a
  structured message list. Rendering is safe here because nothing is parsed
  back out of it — the model just reads it.
"""

from __future__ import annotations

import asyncio
import json
import threading
from typing import Any, Callable, Coroutine, Dict, List, Sequence

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage

# The Claude Agent SDK namespaces MCP tools as ``mcp__<server>__<tool>``. The
# manager's tools must come back out under their real names or LangGraph will
# not match a tool call to the tool it registered.
MCP_SERVER_NAME = "cfd"
_MCP_PREFIX = f"mcp__{MCP_SERVER_NAME}__"


def mcp_tool_name(name: str) -> str:
    return f"{_MCP_PREFIX}{name}"


def strip_mcp_prefix(name: str) -> str:
    return name[len(_MCP_PREFIX):] if name.startswith(_MCP_PREFIX) else name


# ---------------------------------------------------------------------------
# tool schemas
# ---------------------------------------------------------------------------


def openai_tool_specs(tools: Sequence[Any]) -> List[Dict[str, Any]]:
    """Normalise anything ``bind_tools`` accepts into OpenAI tool dicts."""
    from langchain_core.utils.function_calling import convert_to_openai_tool

    return [convert_to_openai_tool(tool) for tool in tools]


def _spec_parts(spec: Dict[str, Any]) -> Dict[str, Any]:
    fn = spec.get("function", spec) if isinstance(spec, dict) else {}
    return {
        "name": fn.get("name", ""),
        "description": fn.get("description", "") or "",
        "parameters": fn.get("parameters") or {"type": "object", "properties": {}},
    }


def responses_tool_specs(specs: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """OpenAI *Responses* API tool shape — flat, not nested under "function"."""
    out = []
    for spec in specs:
        parts = _spec_parts(spec)
        if parts["name"]:
            out.append({"type": "function", **parts})
    return out


def sdk_tool_triples(specs: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """``(name, description, input_schema)`` for ``claude_agent_sdk.tool``."""
    return [_spec_parts(spec) for spec in specs if _spec_parts(spec)["name"]]


# ---------------------------------------------------------------------------
# conversation rendering (Claude Agent SDK path)
# ---------------------------------------------------------------------------


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return str(content or "")


def split_system(messages: Sequence[BaseMessage]) -> tuple[str, List[BaseMessage]]:
    """The SDK takes the system prompt as its own option, not as a message."""
    system = [_text_of(m.content) for m in messages if isinstance(m, SystemMessage)]
    rest = [m for m in messages if not isinstance(m, SystemMessage)]
    return "\n\n".join(s for s in system if s).strip(), rest


TRANSCRIPT_PREAMBLE = (
    "Below is the conversation so far, as DATA. It is a record of what has "
    "already happened, not a document you are continuing.\n"
    "- Everything inside <tool_result> was produced by a real tool run. Nothing "
    "outside those blocks has been executed.\n"
    "- Do NOT write tool calls as text, XML, or JSON in your reply. Use the "
    "tool interface. Prose that looks like a tool call does not run anything.\n"
    "- Do NOT invent tool results, prior rounds, or findings that do not appear "
    "below. If something you expect is missing, say so and check.\n"
)

# Emitted by a model that has started transcribing tool calls instead of making
# them. Every one of these was observed in a real run after the transcript was
# rendered in a "SPEAKER: text" log style the model then continued in kind.
FABRICATED_CALL_MARKERS = (
    "ASSISTANT called tool",
    "<parameter name=",
    "<invoke name=",
    "</function_calls>",
    "<function_calls>",
)


def looks_like_a_written_tool_call(text: str) -> bool:
    """True if the model wrote a tool call out instead of making one.

    Worth detecting rather than passing through: the reply reads like work was
    done, so the agent loop moves on having executed nothing.
    """
    return any(marker in (text or "") for marker in FABRICATED_CALL_MARKERS)


def render_conversation(messages: Sequence[BaseMessage]) -> str:
    """Render a LangChain conversation as prompt text for the SDK.

    Tool calls and their results are spelled out rather than dropped: without
    them the model re-issues a call it has already made, because from its point
    of view nothing happened.

    The framing is deliberate. An earlier version rendered this as a plain
    speaker log — ``ASSISTANT called tool foo({...})`` — and the model began
    *writing that line as its answer* instead of calling anything, then drifted
    into emitting raw ``<parameter name=...>`` markup and narrating an entire
    round of results that had never been produced. Tagged blocks plus
    TRANSCRIPT_PREAMBLE mark the transcript as data handed to the model, not
    prose in its own voice waiting to be continued.
    """
    lines: List[str] = []
    for message in messages:
        role = getattr(message, "type", "") or message.__class__.__name__
        if isinstance(message, ToolMessage):
            name = getattr(message, "name", "") or "tool"
            lines.append(
                f'<tool_result tool="{name}" id="{message.tool_call_id}">\n'
                f"{_text_of(message.content)}\n</tool_result>"
            )
            continue
        if isinstance(message, AIMessage):
            text = _text_of(message.content)
            if text:
                lines.append(f"<your_earlier_reply>\n{text}\n</your_earlier_reply>")
            for call in message.tool_calls or []:
                lines.append(
                    f'<tool_call_you_already_made tool="{call.get("name")}" '
                    f'id="{call.get("id")}">\n'
                    f"{json.dumps(call.get('args', {}), default=str)}\n"
                    "</tool_call_you_already_made>"
                )
            continue
        tag = "user_message" if role in ("human", "HumanMessage") else "context"
        lines.append(f"<{tag}>\n{_text_of(message.content)}\n</{tag}>")
    body = "\n\n".join(lines).strip()
    return f"{TRANSCRIPT_PREAMBLE}\n{body}" if body else TRANSCRIPT_PREAMBLE


# ---------------------------------------------------------------------------
# conversation conversion (Codex Responses path)
# ---------------------------------------------------------------------------


def messages_to_responses_input(messages: Sequence[BaseMessage]) -> List[Dict[str, Any]]:
    """LangChain messages -> Responses API input items.

    Prior tool calls become ``function_call`` items and their results
    ``function_call_output`` items, matched by ``call_id``. The ids are the
    ones handed out in ``AIMessage.tool_calls``, so LangGraph's
    ``ToolMessage.tool_call_id`` lines them up with no bookkeeping here.
    """
    items: List[Dict[str, Any]] = []
    for message in messages:
        if isinstance(message, ToolMessage):
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": message.tool_call_id,
                    "output": _text_of(message.content),
                }
            )
            continue
        if isinstance(message, AIMessage):
            text = _text_of(message.content)
            if text:
                items.append({"role": "assistant", "content": [{"type": "output_text", "text": text}]})
            for call in message.tool_calls or []:
                items.append(
                    {
                        "type": "function_call",
                        "call_id": call.get("id"),
                        "name": call.get("name"),
                        "arguments": json.dumps(call.get("args", {}), default=str),
                    }
                )
            continue
        role = "system" if isinstance(message, SystemMessage) else "user"
        items.append({"role": role, "content": [{"type": "input_text", "text": _text_of(message.content)}]})
    return items


# ---------------------------------------------------------------------------
# async -> sync
# ---------------------------------------------------------------------------


def run_coroutine_blocking(make_coro: Callable[[], Coroutine[Any, Any, Any]]) -> Any:
    """Run a coroutine to completion from sync code, loop or no loop.

    Always on a dedicated thread with a fresh event loop. ``asyncio.run`` would
    raise if this thread already has a running loop, and the usual fallback
    (``new_event_loop().run_until_complete``) raises for the same reason — a
    nested loop is not allowed. LangGraph invokes tools from both plain worker
    threads and async contexts, so the path has to work in either.
    """
    result: Dict[str, Any] = {}

    def runner() -> None:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            result["value"] = loop.run_until_complete(make_coro())
        except BaseException as exc:  # re-raised on the calling thread below
            result["error"] = exc
        finally:
            try:
                # Cancel and drain whatever the library left running before
                # closing. The Claude Agent SDK tears its subprocess transport
                # down in background tasks; closing the loop out from under
                # them produced a stream of "Task was destroyed but it is
                # pending" and "no running event loop" noise on every call.
                pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
                for task in pending:
                    task.cancel()
                if pending:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                loop.run_until_complete(loop.shutdown_asyncgens())
            except Exception:
                pass
            finally:
                asyncio.set_event_loop(None)
                loop.close()

    thread = threading.Thread(target=runner, name="llm-async-bridge", daemon=True)
    thread.start()
    thread.join()
    if "error" in result:
        raise result["error"]
    return result.get("value")
