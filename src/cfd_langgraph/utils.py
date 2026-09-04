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


# --- Tolerant structured output ------------------------------------------
#
# A schema mismatch is not the same as a wrong answer, and the two were being
# treated identically. Measured on ph_glm_20260902_2340: the candidate
# proposer returned three complete, well-formed candidates -- names, families,
# mechanisms, all correct -- under the key "proposals" where the schema said
# "candidates", and the whole batch was discarded. The same run lost its study
# metrics to "quantities" instead of "metrics", and its first metric-proposer
# attempt to a bare list where an object was expected. Three model calls'
# worth of correct work, thrown away over the spelling of a wrapper key.
#
# Whether this happens is a property of the model, not of this code: GLM and
# Gemini reach Vertex through the identical client, and Gemini holds the field
# names while GLM does not. What is this code's business is refusing to throw
# away an answer it can plainly read.
#
# Every rule below is structural -- shapes and counts, never key names or
# their meanings -- so nothing is guessed about what a field is for. When the
# shape is ambiguous the payload is returned untouched and validation fails as
# it did before.

from typing import Any, Dict, List, Optional, Type, Union, get_args, get_origin  # noqa: E402

from pydantic import BaseModel, model_validator  # noqa: E402

_TOLERANT_CACHE: "Dict[type, type]" = {}


def _list_valued_fields(schema: "Type[BaseModel]") -> "List[str]":
    """Field names on `schema` whose annotation is a list (or Optional list)."""
    names: List[str] = []
    for name, field in schema.model_fields.items():
        annotation = field.annotation
        if annotation is list or get_origin(annotation) is list:
            names.append(name)
            continue
        if get_origin(annotation) is Union:
            if any(get_origin(arg) is list or arg is list for arg in get_args(annotation)):
                names.append(name)
    return names


def _coerce_payload(schema: "Type[BaseModel]", data: Any) -> Any:
    """Reshape a payload that is right in content but wrong in wrapping.

    Three shapes are recovered, each requiring the target to be unambiguous:

    1. A bare list, where the schema wants exactly one list field.
       ``[{...}, {...}]`` -> ``{"metrics": [{...}, {...}]}``
    2. A list under the wrong key, where exactly one list field is missing and
       exactly one unrecognised key holds a list.
       ``{"proposals": [...]}`` -> ``{"candidates": [...]}``
    3. The whole object nested one level under a single wrapper key, where the
       inner object carries fields the schema recognises.
       ``{"result": {"metrics": [...]}}`` -> ``{"metrics": [...]}``

    Anything else is returned unchanged.
    """
    list_fields = _list_valued_fields(schema)

    # 1. bare list
    if isinstance(data, list):
        if len(list_fields) == 1:
            return {list_fields[0]: data}
        return data

    if not isinstance(data, dict):
        return data

    known = set(schema.model_fields)

    # 3. single wrapper key around a recognisable object. Checked before the
    #    rename so that {"result": {"candidates": [...]}} unwraps rather than
    #    being read as a stray list under a wrong name.
    if len(data) == 1:
        (only_value,) = data.values()
        if isinstance(only_value, dict) and (set(only_value) & known):
            return _coerce_payload(schema, only_value)

    # 2. right list, wrong key
    missing_lists = [name for name in list_fields if name not in data]
    strays = [k for k, v in data.items() if k not in known and isinstance(v, list)]
    if len(missing_lists) == 1 and len(strays) == 1:
        reshaped = dict(data)
        reshaped[missing_lists[0]] = reshaped.pop(strays[0])
        return reshaped

    return data


def tolerant_schema(schema: "Type[BaseModel]") -> "Type[BaseModel]":
    """`schema` with a pre-validator that recovers near-miss payloads.

    The JSON Schema is unchanged, so a provider that enforces the schema
    server-side is asked for exactly what it was asked for before; this only
    changes what happens to a reply that arrives shaped slightly differently.
    Subclasses are cached, so repeated calls return the same class and
    isinstance checks stay stable.
    """
    cached = _TOLERANT_CACHE.get(schema)
    if cached is not None:
        return cached

    class _Tolerant(schema):  # type: ignore[misc,valid-type]
        @model_validator(mode="before")
        @classmethod
        def _reshape_near_miss(cls, data: Any) -> Any:
            try:
                return _coerce_payload(schema, data)
            except Exception:
                # Never let salvage turn a recoverable reply into a crash.
                return data

    _Tolerant.__name__ = schema.__name__
    _Tolerant.__qualname__ = schema.__qualname__
    _TOLERANT_CACHE[schema] = _Tolerant
    return _Tolerant


class _RepairingStructured:
    """``with_structured_output`` that asks the model to fix its own reply.

    The shape rules above recover a wrapper that is merely named wrongly. They
    deliberately stop at the edge of guessing: when GLM returned its candidate
    batch under "proposals", the list was moved to "candidates" and then failed
    anyway, because inside it each candidate called its identifier "name" where
    the schema says "variant_name". Renaming that by position or type would be
    inventing an answer, and a scored search is the last place to do that.

    So the repair goes back to the model, which is the thing that knows what it
    meant. It is shown its own reply, the schema, and the exact validation
    error, and asked to re-emit the same content correctly -- content it
    already produced, so nothing new is being reasoned about and nothing is
    substituted on its behalf. One attempt: a model that cannot restate its own
    answer to a schema it was just given is failing at something a further
    round will not fix.
    """

    def __init__(self, llm: Any, schema: "Type[BaseModel]") -> None:
        self._llm = llm
        self._schema = schema
        self._bound = llm.with_structured_output(tolerant_schema(schema))

    def get_num_tokens(self, text: str) -> int:
        return self._llm.get_num_tokens(text)

    def invoke(self, prompt: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return self._bound.invoke(prompt, *args, **kwargs)
        except Exception as first_error:
            try:
                return self._repair(prompt, first_error)
            except Exception:
                # Report the original failure: it describes what the model
                # actually got wrong, where the repair's failure describes only
                # that the rescue did not work.
                raise first_error

    def _repair(self, prompt: Any, error: Exception) -> Any:
        import json as _json

        instruction = (
            "Your previous reply had the right content but did not match the "
            "required schema, so it was rejected.\n\n"
            "REQUIRED JSON SCHEMA:\n"
            + _json.dumps(self._schema.model_json_schema(), indent=2)
            + "\n\nVALIDATION ERROR:\n"
            + str(error)[:2000]
            + "\n\nRe-emit the SAME content as JSON matching that schema exactly. "
            "Use the schema's own field names. Do not add, drop, or change any "
            "of the substance -- only the shape. Return JSON only, no prose and "
            "no markdown fences."
        )
        if isinstance(prompt, str):
            messages: Any = prompt + "\n\n" + instruction
        else:
            messages = list(prompt) + [{"role": "user", "content": instruction}]
        raw = getattr(self._llm.invoke(messages), "content", "")
        if isinstance(raw, list):  # some clients return content blocks
            raw = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in raw
            )
        return tolerant_schema(self._schema).model_validate_json(
            extract_json_object(str(raw))
        )


def structured_output(llm: Any, schema: "Type[BaseModel]") -> Any:
    """``llm.with_structured_output(schema)`` that does not throw away a reply
    it can still use -- reshaping a near-miss, and otherwise asking the model
    to restate its own answer against the schema."""
    return _RepairingStructured(llm, schema)
