"""ChatGPT/Codex subscription access, without the vendored Foam-Agent package.

Everything here used to be reached by importing ``Foam-Agent/src/utils.py`` at
runtime. That import is far more expensive than what it was being used for: the
module builds an embedding model and loads FAISS indices at import time, so
merely reading a JSON token file pulled in Qwen3-Embedding and a vector store.
The case-running path was already rewritten natively (``foam_native/``); this
removes the last runtime dependency on the vendored package from the CLI.

Two pieces, both self-contained:

* ``load_codex_oauth`` — finds the Codex CLI's OAuth cache and returns
  ``(access_token, account_id)``.
* ``CodexResponsesWrapper`` — a minimal client for the OpenAI Responses API,
  pointed at either the public endpoint or the ChatGPT/Codex subscription
  backend.

Note these files hold access and refresh tokens. Treat them like passwords.
"""

from __future__ import annotations

import base64
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Type

import requests
from pydantic import BaseModel

# Raised when the connection gives up, not when the request was wrong. A
# high-effort turn streams for many minutes, which is ample time for anything
# between here and the backend to drop it.
_TRANSIENT_ERRORS = (
    requests.exceptions.ChunkedEncodingError,
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
)

_INSTRUCTIONS_PATH = Path(__file__).with_name("codex_instructions_default.txt")

_INSTRUCTIONS_FALLBACK = (
    "You are Codex, based on GPT-5. You are running as a coding agent in the "
    "Codex CLI on a user's computer."
)


def default_instructions() -> str:
    """The Codex harness preamble the ChatGPT backend expects.

    The subscription endpoint rejects a request whose ``instructions`` field is
    empty, so this is not decoration — it has to be sent.
    """
    try:
        return _INSTRUCTIONS_PATH.read_text(encoding="utf-8")
    except Exception:
        return _INSTRUCTIONS_FALLBACK


# ---------------------------------------------------------------------------
# token cache
# ---------------------------------------------------------------------------


def _token_expiry(token: str) -> Optional[datetime]:
    """The ``exp`` claim of a JWT access token, or None if it isn't one.

    Read locally, purely to turn an expired token into an actionable message.
    A JWT's payload is base64url, not encrypted, so this needs no secret and
    makes no network call.
    """
    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        claims = json.loads(base64.urlsafe_b64decode(payload))
        exp = claims.get("exp")
        return datetime.fromtimestamp(int(exp), timezone.utc) if exp else None
    except Exception:
        return None


def _check_not_expired(token: str, source: Path) -> None:
    expiry = _token_expiry(token)
    if expiry is None or expiry > datetime.now(timezone.utc):
        return
    raise ValueError(
        f"The Codex OAuth token in {source} expired at {expiry:%Y-%m-%d %H:%M UTC}. "
        "Run `codex login` to refresh it."
    )


def _from_auth_json(path: Path) -> Tuple[str, Optional[str]]:
    """Read the Codex CLI's ``auth.json``.

    Deliberately permissive about shape — different Codex versions have stored
    the token differently. The current CLI nests it:

        {"tokens": {"access_token": "...", "account_id": "..."}}

    which an earlier version of this loader did not look at, so a perfectly
    valid, freshly-logged-in cache was reported as "could not find an access
    token". ``account_id`` matters too: the ChatGPT backend wants it in the
    ``ChatGPT-Account-Id`` header, and returning None for it made the auth.json
    path silently send requests without one.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Unexpected JSON in {path}")

    account_id: Optional[str] = None
    candidates: List[str] = []

    def consider(container: Any) -> None:
        nonlocal account_id
        if not isinstance(container, dict):
            return
        for key in ("access_token", "token"):
            value = container.get(key)
            if isinstance(value, str) and value.strip():
                candidates.append(value.strip())
        for key in ("account_id", "accountId", "chatgpt_account_id"):
            value = container.get(key)
            if account_id is None and isinstance(value, str) and value.strip():
                account_id = value.strip()

    consider(data)
    for key in ("tokens", "auth", "credentials", "session"):
        consider(data.get(key))

    if not candidates:
        raise ValueError(
            f"Could not find an access token in {path}. Looked for access_token/token "
            "at the top level and under tokens/auth/credentials/session. "
            "Run `codex login` if you have not signed in with ChatGPT."
        )

    token = candidates[0]
    _check_not_expired(token, path)
    return token, account_id


def _from_clawdbot_profiles(path: Path) -> Tuple[str, Optional[str]]:
    """Clawdbot's cache: ``{"profiles": {"openai-codex:default": {...}}}``."""
    data = json.loads(path.read_text(encoding="utf-8"))
    profiles = data.get("profiles") if isinstance(data, dict) else None
    if not isinstance(profiles, dict):
        raise ValueError(f"Missing 'profiles' in {path}")

    ordered = [profiles.get(k) for k in ("openai-codex:default", "openai-codex")]
    ordered += [v for k, v in profiles.items() if k not in ("openai-codex:default", "openai-codex")]
    for profile in ordered:
        if not isinstance(profile, dict):
            continue
        token = profile.get("access")
        if isinstance(token, str) and token.strip():
            account_id = profile.get("accountId")
            _check_not_expired(token.strip(), path)
            return token.strip(), account_id if isinstance(account_id, str) else None

    raise ValueError(f"No profile in {path} carries an 'access' token.")


def codex_auth_candidates() -> List[Path]:
    paths: List[Path] = []
    codex_home = os.getenv("CODEX_HOME")
    if codex_home:
        paths.append(Path(codex_home) / "auth.json")
    paths.append(Path.home() / ".codex" / "auth.json")
    paths.append(Path.home() / ".clawdbot" / "agents" / "main" / "agent" / "auth-profiles.json")
    return paths


def load_codex_oauth() -> Tuple[str, Optional[str]]:
    """Return ``(access_token, account_id)`` from the first cache that has one."""
    candidates = codex_auth_candidates()
    for path in candidates:
        if not path.exists():
            continue
        if path.name == "auth.json":
            return _from_auth_json(path)
        return _from_clawdbot_profiles(path)

    raise FileNotFoundError(
        "Could not find a Codex/ChatGPT OAuth cache. Looked for: "
        + ", ".join(str(p) for p in candidates)
        + ". Run `codex login` (with file-based credential storage) first."
    )


# ---------------------------------------------------------------------------
# Responses API client
# ---------------------------------------------------------------------------


class CodexResponsesWrapper:
    """Minimal client for an OpenAI Responses-compatible endpoint.

    Exposes just what the LangChain bridge in ``factory.py`` needs:
    ``invoke(messages) -> object with .content``, ``get_num_tokens(text)`` and
    ``with_structured_output(schema)``.

    Two wire endpoints are supported: the public platform
    (``https://api.openai.com/v1``) and the ChatGPT/Codex subscription backend
    (``https://chatgpt.com/backend-api/codex``), which needs the extra harness
    fields set in ``_build_payload`` and rejects ``temperature``.
    """

    class _Resp:
        def __init__(self, content: str, tool_calls: Optional[List[Dict[str, Any]]] = None) -> None:
            self.content = content
            # Native Responses `function_call` items, already decoded. Empty for
            # a plain text answer, so callers can treat this uniformly.
            self.tool_calls: List[Dict[str, Any]] = tool_calls or []

    def __init__(
        self,
        token: str,
        model: str,
        temperature: float = 0.0,
        *,
        base_url: str = "https://api.openai.com/v1",
        account_id: Optional[str] = None,
        instructions: Optional[str] = None,
        stream: bool = False,
        timeout: int = 180,
        effort: Optional[str] = None,
    ) -> None:
        self._token = token
        # mtime of the credential file the token came from, so a refresh
        # written by the Codex CLI is noticed without re-parsing every call.
        self._token_mtime: Optional[float] = None
        self._model = model
        self._temperature = temperature
        self._base_url = base_url.rstrip("/")
        self._account_id = account_id
        self._instructions = instructions
        self._stream = stream
        self._timeout = timeout
        # Reasoning effort, as the Codex CLI's own `model_reasoning_effort`
        # setting. Sent inside `reasoning`, which is the only place the
        # Responses API takes it.
        self._effort = effort
        try:
            import tiktoken

            try:
                self._enc = tiktoken.get_encoding("o200k_base")
            except Exception:
                self._enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self._enc = None

    def get_num_tokens(self, text: str) -> int:
        if self._enc is None:
            return max(1, len((text or "").split()))
        return len(self._enc.encode(text or ""))

    @staticmethod
    def _extract_json_object(text: str) -> str:
        if not text:
            raise ValueError("Empty response; expected JSON")
        s = text.strip()
        if s.startswith("```"):
            s = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", s)
            s = re.sub(r"\n```\s*$", "", s).strip()
        if s.startswith("{") and s.endswith("}"):
            return s
        match = re.search(r"\{[\s\S]*\}", s)
        if not match:
            raise ValueError(f"Could not find a JSON object in response: {s[:200]}")
        return match.group(0)

    def with_structured_output(self, pydantic_obj: Type[BaseModel]) -> Any:
        parent = self

        class _StructuredWrapper:
            def get_num_tokens(self, text: str) -> int:
                return parent.get_num_tokens(text)

            def invoke(self, messages: Any) -> Any:
                hint = (
                    "Return ONLY valid JSON (no markdown) that matches this JSON Schema:\n"
                    + str(pydantic_obj.model_json_schema())
                )
                patched = [{"role": "system", "content": hint}, *list(messages)]
                raw = getattr(parent.invoke(patched), "content", "")
                return pydantic_obj.model_validate_json(parent._extract_json_object(raw))

        return _StructuredWrapper()

    @staticmethod
    def _to_responses_input(messages: Any) -> List[Dict[str, Any]]:
        return [
            {
                "role": m.get("role"),
                "content": [{"type": "input_text", "text": m.get("content", "")}],
            }
            for m in messages
        ]

    @staticmethod
    def _extract_output_text(resp_json: Any) -> str:
        if isinstance(resp_json, dict) and isinstance(resp_json.get("output_text"), str):
            return resp_json["output_text"]
        texts: List[str] = []
        items = resp_json.get("output", []) if isinstance(resp_json, dict) else []
        for item in items:
            for block in item.get("content", []) if isinstance(item, dict) else []:
                if (
                    isinstance(block, dict)
                    and block.get("type") in {"output_text", "text"}
                    and isinstance(block.get("text"), str)
                ):
                    texts.append(block["text"])
        return "\n".join(texts).strip()

    def _build_payload(
        self,
        messages: Any,
        *,
        items: Optional[List[Dict[str, Any]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Any = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self._model,
            # `items` is the tool-calling path: already-built Responses input
            # items, which can include function_call / function_call_output
            # entries that the simple role/content mapping cannot express.
            "input": items if items is not None else self._to_responses_input(messages),
        }
        is_chatgpt = "chatgpt.com" in self._base_url
        if not is_chatgpt:
            payload["temperature"] = self._temperature
            if tools:
                payload["tools"] = tools
                payload["tool_choice"] = tool_choice or "auto"
        else:
            payload.update(
                {
                    "instructions": self._instructions or "",
                    "tools": tools or [],
                    "tool_choice": tool_choice or "auto",
                    # Allowed only when tools are actually bound. The manager
                    # is explicitly told to launch independent cases and OED
                    # candidates as several task calls in one message; forcing
                    # this off would serialise every fan-out and quietly halve
                    # the throughput the concurrency limiter is there to
                    # manage.
                    "parallel_tool_calls": bool(tools),
                    "reasoning": (
                        {"effort": self._effort, "summary": "auto"}
                        if self._effort
                        else {"summary": "auto"}
                    ),
                    "store": False,
                    "stream": bool(self._stream),
                    "include": ["reasoning.encrypted_content"],
                }
            )
        return payload

    @staticmethod
    def _iter_sse_text(resp: requests.Response) -> Iterator[str]:
        for raw in resp.iter_lines(decode_unicode=True):
            if not raw:
                continue
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", errors="ignore")
            line = str(raw).strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            yield data

    # Errors that mean "the network gave up", not "the request was wrong".
    # A high-effort turn can stream for ten minutes, which is ample time for a
    # connection to be dropped by anything between here and the backend.
    _RETRYABLE_STATUS = (429, 500, 502, 503, 504, 522, 524)

    _MAX_ATTEMPTS = 4

    def invoke(
        self,
        messages: Any,
        *,
        items: Optional[List[Dict[str, Any]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Any = None,
    ) -> "CodexResponsesWrapper._Resp":
        """One request, retried whole on a transient failure.

        The retry has to wrap *reading the response*, not just sending it. A
        streamed answer arrives over minutes and the connection dies mid-body:
        `requests.post` returns 200 with headers, and
        `ChunkedEncodingError('Response ended prematurely')` is raised later,
        from `iter_lines`. A retry around the POST alone never fires — measured
        twice on a real run, at 587s and 549s, each time ending the study.

        A half-read stream cannot be resumed, so the whole call is reissued.
        That is safe: these requests set `store: false` and have no server-side
        effect, so a retry costs tokens and nothing else.
        """
        url = f"{self._base_url}/responses"
        headers = {
            "Authorization": f"Bearer {self._reload_token_if_stale()}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream" if self._stream else "application/json",
            "User-Agent": "cfd-scientist",
        }
        if self._account_id:
            headers["ChatGPT-Account-Id"] = self._account_id
        payload = self._build_payload(messages, items=items, tools=tools, tool_choice=tool_choice)

        last_error: Optional[BaseException] = None
        for attempt in range(1, self._MAX_ATTEMPTS + 1):
            try:
                return self._attempt(url, headers, payload)
            except _TRANSIENT_ERRORS as exc:
                last_error = exc
            except requests.HTTPError as exc:
                status = getattr(getattr(exc, "response", None), "status_code", 0)
                if status == 401 and self._refresh_credentials():
                    # The token on disk is newer than the one this process
                    # started with, so the request is worth sending again.
                    #
                    # A Codex OAuth token lives ten days, and a study runs for
                    # hours or days. This process captured the token string
                    # once at construction and put it in every header, so when
                    # the token aged out mid-run every later request 401'd --
                    # and kept 401'ing even after `codex login` wrote a fresh
                    # token to disk, because nothing re-read the file.
                    # Measured on run closure_20260826_codex: the study died
                    # overnight and sat dead for twelve hours.
                    headers["Authorization"] = f"Bearer {self._token}"
                    last_error = exc
                    print("[codex] token had expired; reloaded credentials from disk "
                          "and retrying.", flush=True)
                    continue
                if status not in self._RETRYABLE_STATUS:
                    raise
                last_error = exc
            if attempt < self._MAX_ATTEMPTS:
                delay = 2 ** attempt
                print(
                    f"[codex] {type(last_error).__name__} on attempt {attempt}/{self._MAX_ATTEMPTS} "
                    f"({str(last_error)[:80]}); retrying in {delay}s",
                    flush=True,
                )
                time.sleep(delay)
        raise RuntimeError(
            f"Codex request failed after {self._MAX_ATTEMPTS} attempts: {last_error!r}"
        ) from last_error

    def _reload_token_if_stale(self) -> str:
        """The current token, re-read from disk when the file has changed.

        Cheap: a stat() per request, and the file is only parsed when its
        mtime moves. The Codex CLI rewrites auth.json whenever it refreshes,
        so picking the change up here means a long-running study inherits the
        refresh instead of dying on a token it captured hours ago.
        """
        for path in codex_auth_candidates():
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            if self._token_mtime is not None and mtime <= self._token_mtime:
                return self._token
            try:
                token, account_id = load_codex_oauth()
            except Exception:
                # An unreadable or expired file is not a reason to stop using
                # a token that may still work; the request itself decides.
                self._token_mtime = mtime
                return self._token
            self._token_mtime = mtime
            if token and token != self._token:
                self._token = token
                if account_id:
                    self._account_id = account_id
            return self._token
        return self._token

    def _refresh_credentials(self) -> bool:
        """Force a re-read after a 401. True if the token actually changed."""
        previous = self._token
        self._token_mtime = None
        try:
            self._reload_token_if_stale()
        except Exception:
            return False
        return self._token != previous

    def _attempt(self, url: str, headers: Dict[str, str], payload: Dict[str, Any]) -> "CodexResponsesWrapper._Resp":
        """Send once and consume the whole response. Raises on any failure."""
        resp = requests.post(
            url, headers=headers, json=payload, timeout=self._timeout, stream=bool(self._stream)
        )
        if not resp.ok:
            try:
                detail = resp.text[:2000]
            except Exception:
                detail = ""
            resp.close()
            raise requests.HTTPError(f"HTTP {resp.status_code} for {url}. Body: {detail}", response=resp)

        if not self._stream:
            body = resp.json()
            return self._Resp(self._extract_output_text(body), self._extract_tool_calls(body))

        chunks: List[str] = []
        tool_calls: List[Dict[str, Any]] = []
        try:
            for event in self._iter_sse_text(resp):
                try:
                    parsed = json.loads(event)
                except Exception:
                    continue
                if isinstance(parsed, dict):
                    kind = parsed.get("type")
                    if kind == "response.output_text.delta" and isinstance(parsed.get("delta"), str):
                        chunks.append(parsed["delta"])
                        continue
                    if kind == "response.output_text.done" and isinstance(parsed.get("text"), str):
                        if not chunks:
                            chunks.append(parsed["text"])
                        continue
                    if kind == "response.output_item.done":
                        # The completed item is the only place a function call
                        # arrives whole; the *.delta events carry argument
                        # fragments that would have to be reassembled by hand.
                        item = parsed.get("item")
                        if isinstance(item, dict) and item.get("type") == "function_call":
                            tool_calls.append(self._decode_function_call(item))
                        continue

                    # A terminal event carries the whole response nested under
                    # "response". Not every turn streams deltas — under load the
                    # server can deliver the answer only here — and reading just
                    # the top level of this event finds nothing, so a perfectly
                    # good turn was being discarded as "empty". That is what the
                    # empty-turn retries in run closure_20260826_codex were
                    # actually retrying: our own parser, not a provider fault.
                    if kind in {"response.completed", "response.incomplete", "response.failed"}:
                        body = parsed.get("response")
                        if isinstance(body, dict):
                            if not chunks:
                                text = self._extract_output_text(body)
                                if text:
                                    chunks.append(text)
                            if not tool_calls:
                                tool_calls.extend(self._extract_tool_calls(body))
                        if kind != "response.completed":
                            # Reported rather than swallowed: an incomplete or
                            # failed response is a real outcome with a reason
                            # attached, and returning it as a silent empty turn
                            # throws that reason away.
                            reason = ""
                            if isinstance(body, dict):
                                detail = body.get("incomplete_details") or body.get("error") or {}
                                if isinstance(detail, dict):
                                    reason = str(detail.get("reason") or detail.get("message") or "")
                            if not chunks and not tool_calls:
                                raise RuntimeError(
                                    f"Codex returned {kind}"
                                    + (f": {reason}" if reason else " with no content")
                                )
                        continue
                fallback = self._extract_output_text(parsed)
                if fallback:
                    chunks.append(fallback)
        finally:
            resp.close()
        return self._Resp("".join(chunks).strip(), tool_calls)

    @staticmethod
    def _decode_function_call(item: Dict[str, Any]) -> Dict[str, Any]:
        raw = item.get("arguments") or "{}"
        try:
            args = json.loads(raw)
        except Exception:
            # Keep the call rather than dropping it: a malformed-argument tool
            # call still tells the caller which tool was wanted, and surfacing
            # the raw string produces a usable error instead of a silent no-op.
            args = {"__raw_arguments__": raw}
        return {
            "name": item.get("name", ""),
            "args": args if isinstance(args, dict) else {"__value__": args},
            "id": item.get("call_id") or item.get("id") or "",
        }

    @classmethod
    def _extract_tool_calls(cls, body: Any) -> List[Dict[str, Any]]:
        items = body.get("output", []) if isinstance(body, dict) else []
        return [
            cls._decode_function_call(item)
            for item in items
            if isinstance(item, dict) and item.get("type") == "function_call"
        ]


# ---------------------------------------------------------------------------
# model selection
# ---------------------------------------------------------------------------

# Only a subset of models is served to ChatGPT-account Codex sessions, and the
# set moves: `gpt-5-codex` — the previous default here — is now rejected with
# "not supported when using Codex with a ChatGPT account". Rather than pin
# another name that will age the same way, read whatever the user's own Codex
# CLI is configured to use; this constant is only the last resort.
_FALLBACK_CODEX_MODEL = "gpt-5.6-sol"


def default_codex_model() -> str:
    """The model the local Codex CLI is set to, else a known-served fallback."""
    for base in ([Path(os.environ["CODEX_HOME"])] if os.getenv("CODEX_HOME") else []) + [
        Path.home() / ".codex"
    ]:
        config = base / "config.toml"
        if not config.is_file():
            continue
        try:
            import tomllib

            with config.open("rb") as handle:
                model = tomllib.load(handle).get("model")
            if isinstance(model, str) and model.strip():
                return model.strip()
        except Exception:
            continue
    return _FALLBACK_CODEX_MODEL
