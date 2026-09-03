"""Shared utilities for cfd_langgraph."""

from __future__ import annotations

import re

_JSON_FENCE_RE = re.compile(
    r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL
)
_LATEX_FENCE_RE = re.compile(
    r"```(?:latex|tex)?\s*\n?(.*?)\n?\s*```", re.DOTALL
)


def strip_json_fences(text: str) -> str:
    """Remove markdown ```json ... ``` fences, returning the inner content.

    If no fences are found the original text is returned unchanged.
    When multiple fenced blocks exist, the first one wins.
    """
    text = text.strip()
    m = _JSON_FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    return text


def extract_json_object(text: str) -> str:
    """The JSON object inside an LLM reply, even when prose surrounds it.

    ``strip_json_fences`` removes ```json fences and nothing else, so a reply
    that opens with a sentence fails ``json.loads`` at character 0 with
    "Expecting value: line 1 column 1". That is not a malformed answer -- the
    JSON is intact, just not first. Measured live on the requirement validator:

        I'll inspect the existing case configuration and file layout to verify
        the requirement against the actual OpenFOAM 10 setup.{"valid": false,
        "issues": [...]}

    Every caller treated that as "no verdict" and paid for it: the validator
    scored the requirement invalid and triggered a ~200s repair round against
    an invented issue, and ideation scored the idea maximally similar and threw
    the generation away.

    Scans for the first balanced object rather than pattern-matching one, so a
    brace inside a string value cannot end it early. Returns the fenced/stripped
    text unchanged when there is no balanced object to find, leaving the
    caller's own error path to handle a genuinely unreadable reply.
    """
    text = strip_json_fences(text)
    start = text.find("{")
    if start == -1:
        return text
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return text


def strip_latex_fences(text: str) -> str:
    """Remove markdown ```latex ... ``` or ```tex ... ``` fences, returning the inner content."""
    text = text.strip()
    m = _LATEX_FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    return text
