"""Retry one model call through a transient provider failure.

The provider clients carry their own retry ladders, but those sit inside a
single request and give up quickly. A call that still failed used to take its
whole stage down with it: on 2026-09-11 a glm hypothesis step that had already
accepted six ideas lost all of them to one 504, three times in a row, because
the exception unwound the batch and the CLI restarted the step from zero.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Callable, Optional, Sequence, TypeVar

T = TypeVar("T")

# HTTP statuses that mean "try again later", not "this request is wrong".
_TRANSIENT_STATUS = frozenset({408, 429, 500, 502, 503, 504})


def _status_code(exc: BaseException) -> int | None:
    # Each SDK names it differently: openai `status_code`, google-genai `code`,
    # requests/httpx through `response.status_code`.
    for attr in ("status_code", "code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    value = getattr(getattr(exc, "response", None), "status_code", None)
    return value if isinstance(value, int) else None


def _transport_error_types() -> tuple:
    types: list = [TimeoutError, ConnectionError]
    try:
        import httpx

        types.append(httpx.TransportError)
    except ImportError:
        pass
    try:
        import requests

        types += [requests.ConnectionError, requests.Timeout]
    except ImportError:
        pass
    try:
        import openai

        types += [openai.APIConnectionError, openai.APITimeoutError]
    except ImportError:
        pass
    return tuple(types)


def is_transient_error(exc: BaseException) -> bool:
    """True when ``exc``, or anything it wraps, is a timeout, a dropped
    connection, or a retryable HTTP status from the provider."""
    transport = _transport_error_types()
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, transport) or _status_code(current) in _TRANSIENT_STATUS:
            return True
        current = current.__cause__ or current.__context__
    return False


def call_with_retry(fn: Callable[[], T], what: str, delays: Sequence[float] = (20, 60, 120)) -> T:
    """Run ``fn``; on a transient failure wait and run it again.

    Anything that is not transient is raised at once: a malformed request or a
    bug does not get better by waiting.
    """
    for attempt in range(len(delays) + 1):
        try:
            return fn()
        except Exception as exc:
            if (attempt >= len(delays) or not is_transient_error(exc)
                    or provider_quota_reset_s(exc) is not None):
                raise
            delay = delays[attempt]
            print(
                f"[retry] {what} failed ({type(exc).__name__}: {str(exc)[:120]}); "
                f"retrying in {delay:g}s ({attempt + 1}/{len(delays)})",
                flush=True,
            )
            time.sleep(delay)
    raise AssertionError("unreachable")


# Rate limits and provider overload need minutes, not seconds. On 2026-09-12
# every qwen3.8-flash build agent died on OpenRouter's "temporarily
# rate-limited upstream, retry shortly" after a 2/4/8 s ladder -- 14 seconds of
# patience in total -- and the candidates were then judged as failed ideas.
PROVIDER_WAITS_S = (20, 60, 120, 240)
_PROVIDER_STATUS = frozenset({429, 500, 502, 503, 504, 529})


def provider_status(exc: BaseException) -> int | None:
    """The 429/5xx status a provider refused with, anywhere in the exception
    chain, or None. Billing and permission errors (402, 403) are deliberately
    excluded: waiting does not refill an account."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        code = _status_code(current)
        if code in _PROVIDER_STATUS:
            return code
        current = current.__cause__ or current.__context__
    return None


def provider_retry_delay(exc: BaseException, attempt: int) -> float | None:
    """Seconds to wait before provider retry number ``attempt`` (0-based), or
    None when ``exc`` is not a provider refusal or the ladder is spent. A
    Retry-After header, when the provider sends one, can lengthen the wait (to
    at most five minutes) but never shorten it."""
    if provider_status(exc) is None or attempt >= len(PROVIDER_WAITS_S):
        return None
    if provider_quota_reset_s(exc) is not None:
        return None
    wait = float(PROVIDER_WAITS_S[attempt])
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        headers = getattr(getattr(current, "response", None), "headers", None)
        try:
            hint = str((headers or {}).get("retry-after") or "").strip()
        except Exception:
            hint = ""
        try:
            return max(wait, min(float(hint), 300.0)) if hint else wait
        except ValueError:
            pass
        current = current.__cause__ or current.__context__
    return wait


# A provider that says "come back in hours" is not rate limiting, it is out of
# quota, and no retry ladder outlasts that. On 2026-09-13 the Codex account hit
# its usage limit at 18:10 with resets_in_seconds=466583 (5.4 days): each build
# agent spent ~43 minutes of its own clock retrying, and the CLI's step-retry
# ladder then tried again on top.
QUOTA_WAIT_S = 900.0


class ProviderQuotaExhausted(RuntimeError):
    """The provider refused with a wait longer than any retry should cover."""

    status_code = 429

    def __init__(self, message: str, reset_s: float):
        super().__init__(message)
        self.reset_s = float(reset_s)


def _seconds_from_retry_after(value: Any) -> Optional[float]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(text)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return (when - datetime.now(timezone.utc)).total_seconds()


def _seconds_from_body(body: Any) -> Optional[float]:
    """The longest wait an error body states, from the fields providers use."""
    if isinstance(body, (bytes, bytearray)):
        body = body.decode("utf-8", errors="replace")
    if isinstance(body, str):
        try:
            body = json.loads(body)
        except ValueError:
            return None
    found: list = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                name = str(key).lower()
                try:
                    if name in {"resets_in_seconds", "retry_after", "retry_after_seconds"}:
                        found.append(float(value))
                    elif name == "resets_at":
                        found.append(float(value) - time.time())
                except (TypeError, ValueError):
                    pass
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(body)
    return max(found) if found else None


def provider_quota_reset_s(exc: BaseException) -> Optional[float]:
    """Seconds until the provider will take requests again, when it says so and
    the wait is longer than QUOTA_WAIT_S; None otherwise.

    Read from what the provider sent -- a Retry-After header, or the reset
    fields of its error body (Codex's resets_in_seconds / resets_at) -- never
    from the wording of the message."""
    seen: set[int] = set()
    current: Optional[BaseException] = exc
    longest: Optional[float] = None
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        candidates = [getattr(current, "reset_s", None)]
        response = getattr(current, "response", None)
        headers = getattr(response, "headers", None)
        try:
            candidates.append(_seconds_from_retry_after((headers or {}).get("retry-after")))
        except Exception:
            pass
        body = getattr(current, "body", None)
        if body is None and response is not None:
            try:
                body = response.text
            except Exception:
                body = None
        candidates.append(_seconds_from_body(body))
        for value in candidates:
            if isinstance(value, (int, float)) and (longest is None or value > longest):
                longest = float(value)
        current = current.__cause__ or current.__context__
    return longest if longest is not None and longest > QUOTA_WAIT_S else None


def describe_wait(seconds: float) -> str:
    """'about 5 d 9 h (until Sat 19 Sep 04:10)' -- local time."""
    seconds = max(0.0, float(seconds))
    days, rest = divmod(int(seconds), 86400)
    hours, rest = divmod(rest, 3600)
    minutes = rest // 60
    span = (f"{days} d {hours} h" if days else f"{hours} h {minutes} min" if hours
            else f"{minutes} min")
    until = datetime.fromtimestamp(time.time() + seconds).strftime("%a %d %b %H:%M")
    return f"about {span} (until {until})"
