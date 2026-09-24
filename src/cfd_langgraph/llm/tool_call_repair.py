"""Recover a tool call a model wrote as prose instead of emitting properly.

Some models narrate the call they mean to make -- ``[interpret_case(case_id=
"case_001")]`` as message text -- rather than returning it in the tool-call
field. LangGraph reads "no tool calls" as the agent choosing to stop, so the
turn ends and the study parks. Measured on ph_llama_20260910e
(llama-4-maverick): five such turns in one run, including
``[oed_prepare_baseline(...)]``, ``[task(subagent_type="case-runner", ...)]``
and ``[analyze_all_cases(case_ids="['case_001']")]``. Every one was a correct
decision expressed in the wrong shape, and every one ended the study.

Nudging the model to "call the tool properly" wastes the decision it already
made. This asks the model to re-express its own text as a real call instead.

The gate before spending a repair call is an exact substring test against the
names of the tools actually bound -- not a guess at bracket syntax, which
would only ever match the one format we happened to see. Whether the text is
really a call, and what the arguments are, is decided by the model.
"""

from __future__ import annotations

import json
from typing import Any, List, Optional, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


_REPAIR_INSTRUCTION = (
    "Your previous turn described a tool call in the message text instead of "
    "issuing it. Below is that text. If it is asking for a tool to be called, "
    "reply with STRICT JSON only:\n"
    '{"is_tool_call": true, "name": "<tool name>", "args": {<arguments>}}\n'
    "If it is not asking for a tool to be called, reply exactly:\n"
    '{"is_tool_call": false}\n'
    "Use only a tool from this list, with its real argument names: "
)


def looks_like_narrated_call(text: str, tool_names: Sequence[str]) -> Optional[str]:
    """The bound tool named in ``text``, or None.

    Exact substring match against the live tool list, so a model that invents
    a plausible-looking name is not chased, and no assumption is made about
    how the call was formatted.
    """
    if not text:
        return None
    for name in tool_names:
        if name and name in text:
            return name
    return None


def repair_tool_call(
    llm: Any,
    text: str,
    tool_names: Sequence[str],
) -> Optional[List[dict]]:
    """Ask the model to re-express its own narrated call as a real one.

    Returns tool_calls in LangChain's shape, or None when the model says the
    text was not a call, when it names a tool that is not bound, or when
    anything at all goes wrong -- a failed repair must leave the turn exactly
    as it was, never invent a call the model did not ask for.
    """
    try:
        from cfd_langgraph.utils import extract_json_object

        prompt = _REPAIR_INSTRUCTION + ", ".join(tool_names) + "\n\nTEXT:\n" + text[:4000]
        reply = llm.invoke([HumanMessage(content=prompt)])
        raw = getattr(reply, "content", "")
        if isinstance(raw, list):
            raw = "".join(b.get("text", "") for b in raw if isinstance(b, dict))
        parsed = json.loads(extract_json_object(str(raw)))
        if not parsed.get("is_tool_call"):
            return None
        name = str(parsed.get("name") or "").strip()
        if name not in set(tool_names):
            return None
        args = parsed.get("args")
        if not isinstance(args, dict):
            args = {}
        return [{"name": name, "args": args, "id": f"repaired-{name}", "type": "tool_call"}]
    except Exception:
        return None


def repair_if_narrated(
    llm: Any,
    message: BaseMessage,
    tool_names: Sequence[str],
) -> BaseMessage:
    """Return ``message`` with a recovered tool call attached, if there was one."""
    if getattr(message, "tool_calls", None):
        return message
    text = getattr(message, "content", "")
    if isinstance(text, list):
        text = "".join(b.get("text", "") for b in text if isinstance(b, dict))
    if not looks_like_narrated_call(str(text), tool_names):
        return message
    calls = repair_tool_call(llm, str(text), tool_names)
    if not calls:
        return message
    # The narration is not kept as content. It was the model's attempt at the
    # call, now expressed properly, and a turn holding both text and a tool call
    # is refused by some providers once it is sent back as history: Llama 4
    # Maverick on Bedrock answered "Conversation blocks and tool use blocks
    # cannot be provided in the same turn" on the very next request
    # (experiments_for_paper/llama4_maverick/palmo_r2). It stays readable in
    # response_metadata, which is never sent to a provider. The original id is
    # kept so a later update to this turn replaces it instead of appending a
    # second copy.
    return AIMessage(
        content="",
        tool_calls=calls,
        id=getattr(message, "id", None),
        # The turn was still generated and billed. On Gemini, GLM and Bedrock
        # this is the only place its token counts live, so dropping it logged a
        # repaired turn as zero tokens.
        usage_metadata=getattr(message, "usage_metadata", None),
        additional_kwargs=dict(getattr(message, "additional_kwargs", {}) or {}),
        response_metadata={
            **dict(getattr(message, "response_metadata", {}) or {}),
            "repaired_from_narration": str(text)[:2000],
        },
    )
