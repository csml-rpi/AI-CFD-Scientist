"""The text of a model reply, whatever shape its content arrives in."""

from __future__ import annotations

from typing import Any


def reply_text(reply: Any) -> str:
    """The answer text of a model reply: its text blocks joined, nothing else.

    Deliberately not ``str(reply.content)``. Providers return content as a list
    of typed blocks -- Bedrock sends gpt-oss's reasoning and its answer as two
    blocks, Gemini and Claude thinking models add thought blocks -- and ``str``
    of that list is its Python repr. On malmo_gptoss120b_bedrock_20260913 the
    starter-folder reader handed json.loads "[{'type': 'reasoning_content', ..."
    ten times running, while the answer block beside it held valid JSON.

    Takes a message, a bare content value, or whatever else a call returned: a
    string passes through unchanged and None gives "". Reasoning, thinking,
    tool-use and image blocks carry no answer text and are left out.
    """
    content = getattr(reply, "content", reply)
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if block.get("type", "text") == "text" and isinstance(block.get("text"), str):
                    parts.append(block["text"])
            elif getattr(block, "type", "text") == "text" and isinstance(getattr(block, "text", None), str):
                parts.append(block.text)
        return "".join(parts)
    return str(content)
