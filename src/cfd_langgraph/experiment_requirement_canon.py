"""Canonicalize experiment requirement text so Foam-Agent and verify see one spec.

Extracts the JSON object after ``Target parameters:`` and strips embedded
``case_snapshot`` blobs that commonly contradict those numbers. No numerical
solver checks — prompt hygiene only.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional, Tuple


def extract_target_parameters_dict(user_requirement: str) -> Optional[Dict[str, Any]]:
    """Return dict parsed from ``Target parameters: { ... }`` or None."""
    text = user_requirement or ""
    label = "Target parameters:"
    idx = text.find(label)
    if idx < 0:
        return None
    i = text.find("{", idx)
    if i < 0:
        return None
    depth = 0
    j = i
    while j < len(text):
        c = text[j]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                blob = text[i : j + 1]
                try:
                    return json.loads(blob)
                except json.JSONDecodeError:
                    return None
        j += 1
    return None


def _consume_json_string(text: str, open_quote_idx: int) -> int:
    """``open_quote_idx`` points at opening ``"``. Return index just past closing ``"``."""
    i = open_quote_idx + 1
    n = len(text)
    while i < n:
        c = text[i]
        if c == "\\":
            i += 2
            continue
        if c == '"':
            return i + 1
        i += 1
    return n


def _json_object_end(text: str, brace_start: int) -> int:
    """
    ``brace_start`` points at ``{`` starting a JSON object. Return index just past
    the matching ``}``, respecting strings (so braces inside FoamFile headers do not
    confuse depth).
    """
    if brace_start >= len(text) or text[brace_start] != "{":
        return brace_start
    depth = 0
    k = brace_start
    n = len(text)
    while k < n:
        c = text[k]
        if c == '"':
            k = _consume_json_string(text, k)
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return k + 1
        k += 1
    return n


def strip_case_snapshot_from_requirement(user_requirement: str) -> str:
    """Remove ``case_snapshot`` objects from embedded code-mod JSON text."""
    text = user_requirement or ""
    patterns = ('"case_snapshot"', "'case_snapshot'")
    out = text
    for _ in range(32):
        i = -1
        for p in patterns:
            pos = out.find(p)
            if pos >= 0 and (i < 0 or pos < i):
                i = pos
        if i < 0:
            break
        colon = out.find(":", i)
        if colon < 0:
            break
        j = colon + 1
        while j < len(out) and out[j] in " \t\n\r":
            j += 1
        if j >= len(out) or out[j] != "{":
            break
        end_obj = _json_object_end(out, j)
        start_del = i
        while start_del > 0 and out[start_del - 1] in " \t":
            start_del -= 1
        if start_del > 0 and out[start_del - 1] == ",":
            start_del -= 1
            while start_del > 0 and out[start_del - 1] in " \t\n\r":
                start_del -= 1
        out = out[:start_del] + out[end_obj:]
        while ",," in out:
            out = out.replace(",,", ",")
    return out


def build_canonical_body(
    *,
    stripped_user_text: str,
    target_parameters: Optional[Dict[str, Any]],
) -> str:
    """Prepend authoritative target JSON; keep stripped narrative/context after."""
    stripped = (stripped_user_text or "").strip()
    if not target_parameters:
        return stripped
    head = (
        "AUTHORITATIVE_TARGET_PARAMETERS — single source of truth for every numeric "
        "and discrete experiment control (viscosity coefficients, exponents, "
        "velocity / forcing targets, mesh cell counts when listed here, etc.). "
        "Implement the following JSON exactly in 0/, constant/, and system/. "
        "If any later paragraph, code-mod excerpt, or legacy snapshot disagrees with "
        "this block, ignore the conflicting material and follow this block only.\n\n"
        f"{json.dumps(target_parameters, indent=2, sort_keys=True)}\n\n"
        "--- Supporting context (study topic, model class, code-mod paths, libraries). "
        "Do not copy stale dictionary values from removed snapshots.\n---\n\n"
    )
    return head + stripped


def prepare_experiment_requirement_strings(
    *,
    user_requirement: str,
    seed_prefix: str,
    mesh_note: str,
) -> Tuple[str, str, Optional[Dict[str, Any]]]:
    """
    Returns:
      foam_requirement — seed + canonical body + mesh note (Foam-Agent input)
      verify_requirement — canonical body + mesh note (param verify; no seed path)
      target_parameters — parsed dict or None
    """
    raw = user_requirement or ""
    target = extract_target_parameters_dict(raw)
    stripped = strip_case_snapshot_from_requirement(raw)
    body = build_canonical_body(stripped_user_text=stripped, target_parameters=target)
    mesh_suffix = f"\n\n{mesh_note}" if (mesh_note or "").strip() else ""
    foam_req = (seed_prefix or "") + body + mesh_suffix
    verify_req = body + mesh_suffix
    return foam_req, verify_req, target
