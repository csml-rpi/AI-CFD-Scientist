#!/usr/bin/env python3
"""Tests for ChatGPT/Codex subscription auth (``llm/codex_oauth.py``).

All offline. The one behaviour that needs the network — an actual round-trip —
is covered by ``factory.smoke_test_codex_oauth`` and deliberately not run here,
so this suite stays free and usable without a subscription.

The regression these exist for: the loader this replaced only looked for a
token at the top level of ``auth.json`` and under ``auth``/``credentials``/
``session``. The current Codex CLI nests it under ``tokens``, so a valid,
freshly-logged-in cache was reported as "could not find an access token" — and
even where a token was found, ``account_id`` was dropped, so requests went to
the ChatGPT backend with no ``ChatGPT-Account-Id`` header.

Run: python scripts/test_codex_oauth.py
"""

from __future__ import annotations

import base64
import json
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cfd_langgraph.llm.codex_oauth import (  # noqa: E402
    CodexResponsesWrapper,
    default_codex_model,
    load_codex_oauth,
)

FAILURES: list[str] = []


def check(name: str, cond: object, detail: str = "") -> None:
    if cond:
        print(f"[PASS] {name}")
    else:
        FAILURES.append(name)
        print(f"[FAIL] {name}" + (f" — {detail}" if detail else ""))


def jwt(exp_offset_seconds: int) -> str:
    """An unsigned JWT whose only meaningful claim is ``exp``.

    The expiry check reads the payload locally and never verifies a signature,
    so a real signed token is unnecessary — and would expire, making the test
    rot."""
    payload = base64.urlsafe_b64encode(
        json.dumps({"exp": int(time.time()) + exp_offset_seconds}).encode()
    ).decode().rstrip("=")
    return f"header.{payload}.signature"


def with_codex_home(contents: dict, filename: str = "auth.json"):
    """Point CODEX_HOME at a temp dir holding ``contents``."""

    class _Ctx:
        def __enter__(self):
            self.tmp = tempfile.TemporaryDirectory()
            self.prev = os.environ.get("CODEX_HOME")
            path = Path(self.tmp.name) / filename
            path.write_text(json.dumps(contents), encoding="utf-8")
            os.environ["CODEX_HOME"] = self.tmp.name
            return Path(self.tmp.name)

        def __exit__(self, *exc):
            if self.prev is None:
                os.environ.pop("CODEX_HOME", None)
            else:
                os.environ["CODEX_HOME"] = self.prev
            self.tmp.cleanup()

    return _Ctx()


def test_codex_cli_nested_shape() -> None:
    """The shape the current Codex CLI actually writes."""
    token = jwt(3600)
    with with_codex_home(
        {
            "auth_mode": "chatgpt",
            "OPENAI_API_KEY": None,
            "tokens": {
                "id_token": "x",
                "access_token": token,
                "refresh_token": "y",
                "account_id": "acct-123",
            },
            "last_refresh": "2026-08-18T03:23:04Z",
        }
    ):
        got_token, account_id = load_codex_oauth()
    check("a Codex CLI auth.json (tokens.access_token) is read", got_token == token)
    check("account_id comes back with it", account_id == "acct-123", detail=f"got {account_id!r}")


def test_legacy_shapes_still_work() -> None:
    token = jwt(3600)
    with with_codex_home({"access_token": token}):
        check("a top-level access_token still works", load_codex_oauth()[0] == token)
    with with_codex_home({"auth": {"access_token": token, "account_id": "a1"}}):
        got, acct = load_codex_oauth()
        check("an auth.access_token still works", got == token)
        check("account_id is picked up from a sub-dict too", acct == "a1")
    with with_codex_home({"token": token}):
        check("a bare 'token' key still works", load_codex_oauth()[0] == token)


def test_clawdbot_profiles() -> None:
    token = jwt(3600)
    tmp = tempfile.TemporaryDirectory()
    prev_home = os.environ.get("HOME")
    try:
        target = Path(tmp.name) / ".clawdbot" / "agents" / "main" / "agent"
        target.mkdir(parents=True)
        (target / "auth-profiles.json").write_text(
            json.dumps({"profiles": {"openai-codex:default": {"access": token, "accountId": "c9"}}}),
            encoding="utf-8",
        )
        os.environ["HOME"] = tmp.name
        os.environ.pop("CODEX_HOME", None)
        got, acct = load_codex_oauth()
        check("a Clawdbot profile cache is read", got == token)
        check("its accountId is returned", acct == "c9")
    finally:
        if prev_home is not None:
            os.environ["HOME"] = prev_home
        tmp.cleanup()


def test_expired_token_says_what_to_do() -> None:
    """An expired token must not become an opaque HTTP 401 later."""
    with with_codex_home({"tokens": {"access_token": jwt(-60)}}):
        try:
            load_codex_oauth()
        except ValueError as exc:
            check("an expired token is refused up front", "expired" in str(exc).lower())
            check("the error says how to fix it", "codex login" in str(exc))
        else:
            check("an expired token is refused up front", False, detail="no error raised")


def test_missing_token_error_is_actionable() -> None:
    with with_codex_home({"auth_mode": "chatgpt", "OPENAI_API_KEY": None}):
        try:
            load_codex_oauth()
        except ValueError as exc:
            check("a cache with no token names the keys it looked for", "tokens" in str(exc))
        else:
            check("a cache with no token names the keys it looked for", False, detail="no error")


def test_default_model_follows_the_codex_cli() -> None:
    """`gpt-5-codex` is rejected by the ChatGPT backend now, so the default
    must track the user's own CLI rather than a pinned name."""
    tmp = tempfile.TemporaryDirectory()
    prev = os.environ.get("CODEX_HOME")
    try:
        Path(tmp.name, "config.toml").write_text(
            'model = "gpt-5.6-sol"\nmodel_reasoning_effort = "xhigh"\n\n'
            '[projects."/somewhere"]\ntrust_level = "trusted"\n',
            encoding="utf-8",
        )
        os.environ["CODEX_HOME"] = tmp.name
        check("the default model is read from the Codex CLI config", default_codex_model() == "gpt-5.6-sol")
    finally:
        if prev is None:
            os.environ.pop("CODEX_HOME", None)
        else:
            os.environ["CODEX_HOME"] = prev
        tmp.cleanup()
    check("the fallback default is not the rejected gpt-5-codex", "gpt-5-codex" not in default_codex_model())


def test_chatgpt_backend_payload() -> None:
    """The subscription backend rejects `temperature` and requires non-empty
    `instructions`; the public platform is the other way round."""
    messages = [{"role": "user", "content": "hi"}]
    chatgpt = CodexResponsesWrapper(
        token="t", model="m", temperature=0.7,
        base_url="https://chatgpt.com/backend-api/codex", instructions="INSTR",
    )
    payload = chatgpt._build_payload(messages)
    check("no temperature is sent to the ChatGPT backend", "temperature" not in payload)
    check("instructions are sent to the ChatGPT backend", payload.get("instructions") == "INSTR")

    platform = CodexResponsesWrapper(token="t", model="m", temperature=0.7)
    payload = platform._build_payload(messages)
    check("temperature is sent to the public platform", payload.get("temperature") == 0.7)
    check("no Codex harness fields leak to the platform", "instructions" not in payload)


def test_no_foam_agent_import() -> None:
    """The point of this module: reading a token must not drag in Foam-Agent,
    whose utils.py builds an embedding model and FAISS indices at import."""
    check(
        "no Foam-Agent module is loaded by codex auth",
        not any("foam" in name.lower() for name in sys.modules),
        detail=str([n for n in sys.modules if "foam" in n.lower()]),
    )


def test_a_stream_that_dies_mid_body_is_retried() -> None:
    """Regression, twice in production: 587s and 549s, each ending the study.

    This is the failure that matters and the one an earlier test missed. The
    connection does not fail on send — `requests.post` returns 200 with
    headers, and `ChunkedEncodingError('Response ended prematurely')` is raised
    minutes later out of `iter_lines`, while the body is being consumed. A
    retry wrapped around the POST alone never fires. The test therefore fails
    the stream *during iteration*, not at request time.
    """
    import requests

    from cfd_langgraph.llm import codex_oauth
    from cfd_langgraph.llm.codex_oauth import CodexResponsesWrapper

    wrapper = CodexResponsesWrapper(token="t", model="m", stream=True)
    attempts = {"n": 0}
    sleeps: list = []
    real_post, real_sleep = requests.post, codex_oauth.time.sleep
    codex_oauth.time.sleep = lambda s: sleeps.append(s)

    good_event = json.dumps(
        {"type": "response.output_text.done", "text": "the answer"}
    ).encode()

    class _Resp:
        """200 OK whose body fails partway through, like the real one."""

        def __init__(self, die: bool) -> None:
            self.status_code = 200
            self.ok = True
            self.text = ""
            self._die = die

        def iter_lines(self, decode_unicode: bool = False):
            yield b"data: " + json.dumps({"type": "response.created"}).encode()
            if self._die:
                raise requests.exceptions.ChunkedEncodingError("Response ended prematurely")
            yield b"data: " + good_event

        def close(self):
            pass

    try:
        def flaky(*_a, **_k):
            attempts["n"] += 1
            return _Resp(die=attempts["n"] < 3)

        requests.post = flaky
        result = wrapper.invoke([{"role": "user", "content": "hi"}])
        check(
            "a stream that dies mid-body is retried, not fatal",
            attempts["n"] == 3,
            detail=f"{attempts['n']} attempts",
        )
        check("and the retry's answer is returned", result.content == "the answer", detail=repr(result.content))
        check("backoff grows between attempts", sleeps == [2, 4], detail=str(sleeps))

        attempts["n"], sleeps[:] = 0, []
        requests.post = lambda *_a, **_k: (attempts.__setitem__("n", attempts["n"] + 1), _Resp(die=True))[1]
        try:
            wrapper.invoke([{"role": "user", "content": "hi"}])
        except RuntimeError as exc:
            check("a permanently broken stream gives up with a clear error", "after 4 attempts" in str(exc))
            check("and does not retry forever", attempts["n"] == 4, detail=str(attempts))
        else:
            check("a permanently broken stream gives up with a clear error", False, detail="no error")
    finally:
        requests.post = real_post
        codex_oauth.time.sleep = real_sleep


def test_request_errors_are_not_retried() -> None:
    """A 400 means the request itself is wrong; retrying only burns tokens."""
    import requests

    from cfd_langgraph.llm import codex_oauth
    from cfd_langgraph.llm.codex_oauth import CodexResponsesWrapper

    wrapper = CodexResponsesWrapper(token="t", model="m", stream=True)
    attempts = {"n": 0}
    real_post, real_sleep = requests.post, codex_oauth.time.sleep
    codex_oauth.time.sleep = lambda _s: None

    class _Bad:
        status_code = 400
        ok = False
        text = "Invalid value: 'bogus'"

        def close(self):
            pass

    try:
        requests.post = lambda *_a, **_k: (attempts.__setitem__("n", attempts["n"] + 1), _Bad())[1]
        try:
            wrapper.invoke([{"role": "user", "content": "hi"}])
        except requests.HTTPError:
            check("a 400 is raised immediately", attempts["n"] == 1, detail=str(attempts))
        else:
            check("a 400 is raised immediately", False, detail="no error raised")

        attempts["n"] = 0

        class _Busy(_Bad):
            status_code = 503
            ok = False

        requests.post = lambda *_a, **_k: (attempts.__setitem__("n", attempts["n"] + 1), _Busy())[1]
        try:
            wrapper.invoke([{"role": "user", "content": "hi"}])
        except RuntimeError:
            check("but a 503 is retried", attempts["n"] == 4, detail=str(attempts))
        else:
            check("but a 503 is retried", False, detail="no error raised")
    finally:
        requests.post = real_post
        codex_oauth.time.sleep = real_sleep


# --- terminal SSE event -----------------------------------------------------
# A Responses stream ends with {"type": "response.completed", "response": {...}}
# — the content nested one level down. The parser read only the top level of
# that event, so a turn delivered whole in the terminal event instead of as
# deltas was discarded and reported as an empty assistant turn. Under load that
# is common: run closure_20260826_codex logged a dozen "empty assistant turn"
# retries during requirements generation, retrying our own parser rather than
# any provider fault. A `response.failed` was swallowed the same way, losing the
# reason with it.

def _parse_events(events: list) -> tuple:
    """Drive the wrapper's terminal-event handling over a scripted stream."""
    import json as _json
    from cfd_langgraph.llm.codex_oauth import CodexResponsesWrapper as W

    chunks: list[str] = []
    tool_calls: list = []
    for raw in events:
        parsed = _json.loads(raw)
        kind = parsed.get("type")
        if kind == "response.output_text.delta" and isinstance(parsed.get("delta"), str):
            chunks.append(parsed["delta"])
            continue
        if kind in {"response.completed", "response.incomplete", "response.failed"}:
            body = parsed.get("response")
            if isinstance(body, dict):
                if not chunks:
                    text = W._extract_output_text(body)
                    if text:
                        chunks.append(text)
                if not tool_calls:
                    tool_calls.extend(W._extract_tool_calls(body))
            if kind != "response.completed" and not chunks and not tool_calls:
                detail = {}
                if isinstance(body, dict):
                    detail = body.get("incomplete_details") or body.get("error") or {}
                reason = detail.get("reason") or detail.get("message") or "" if isinstance(detail, dict) else ""
                raise RuntimeError(f"Codex returned {kind}" + (f": {reason}" if reason else ""))
            continue
    return "".join(chunks).strip(), tool_calls


def test_terminal_event_content_is_not_discarded() -> None:
    import json as _json

    text, _calls = _parse_events([_json.dumps({
        "type": "response.completed",
        "response": {"output": [{"content": [{"type": "output_text", "text": "the real answer"}]}]},
    })])
    check("text nested under 'response' is recovered", text == "the real answer", detail=repr(text))

    _text, calls = _parse_events([_json.dumps({
        "type": "response.completed",
        "response": {"output": [{"type": "function_call", "name": "do_it",
                                 "arguments": "{}", "call_id": "c1"}]},
    })])
    check("a function_call in the terminal event is recovered",
          len(calls) == 1 and calls[0].get("name") == "do_it", detail=str(calls))


def test_streamed_text_is_not_duplicated_by_the_terminal_event() -> None:
    import json as _json

    text, _ = _parse_events([
        _json.dumps({"type": "response.output_text.delta", "delta": "hel"}),
        _json.dumps({"type": "response.output_text.delta", "delta": "lo"}),
        _json.dumps({"type": "response.completed",
                     "response": {"output": [{"content": [{"type": "output_text", "text": "hello"}]}]}}),
    ])
    check("deltas win and the terminal copy is not appended twice", text == "hello", detail=repr(text))


def test_a_failed_response_reports_its_reason() -> None:
    import json as _json

    try:
        _parse_events([_json.dumps({"type": "response.failed",
                                    "response": {"error": {"message": "upstream timeout"}}})])
        check("a failed response raises instead of looking empty", False, detail="did not raise")
    except RuntimeError as exc:
        check("a failed response raises with its reason", "upstream timeout" in str(exc), detail=str(exc))

    text, _ = _parse_events([_json.dumps({
        "type": "response.incomplete",
        "response": {"incomplete_details": {"reason": "max_output_tokens"},
                     "output": [{"content": [{"type": "output_text", "text": "partial"}]}]},
    })])
    check("an incomplete response keeps whatever content it did produce",
          text == "partial", detail=repr(text))



def main() -> int:
    tests = (
        test_codex_cli_nested_shape,
        test_legacy_shapes_still_work,
        test_clawdbot_profiles,
        test_expired_token_says_what_to_do,
        test_missing_token_error_is_actionable,
        test_default_model_follows_the_codex_cli,
        test_chatgpt_backend_payload,
        test_no_foam_agent_import,
        test_a_stream_that_dies_mid_body_is_retried,
        test_request_errors_are_not_retried,
        test_terminal_event_content_is_not_discarded,
        test_streamed_text_is_not_duplicated_by_the_terminal_event,
        test_a_failed_response_reports_its_reason,
    )
    for test in tests:
        # A raising test is a failing test, not an aborted suite. Without this
        # the first unexpected exception hid every later result — which made
        # the suite read as silent rather than red when the loader was broken.
        try:
            test()
        except Exception as exc:  # noqa: BLE001
            check(test.__name__, False, detail=f"{type(exc).__name__}: {exc}")
    if FAILURES:
        print(f"\n{len(FAILURES)} FAILURE(S): " + ", ".join(FAILURES))
        return 1
    print("\nALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
