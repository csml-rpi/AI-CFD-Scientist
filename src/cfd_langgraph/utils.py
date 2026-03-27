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


def strip_latex_fences(text: str) -> str:
    """Remove markdown ```latex ... ``` or ```tex ... ``` fences, returning the inner content."""
    text = text.strip()
    m = _LATEX_FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    return text
