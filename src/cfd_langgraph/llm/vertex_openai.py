"""OpenAI-compatible Vertex AI endpoints (GLM and other MaaS models).

Vertex serves some third-party models through an OpenAI-shaped
``/endpoints/openapi/chat/completions`` route, so ``ChatOpenAI`` drives them
unmodified — no custom transport, no chat-template translation, and native
tool calling including parallel calls.

The only thing that needs care is auth, and it is the same trap the Codex
provider hit: ``ChatOpenAI`` takes the API key as a STRING, read once at
construction. A Vertex bearer token lives 60 minutes, and a CFD study runs for
many hours, so a static token guarantees the run dies mid-flight with a 401.

``ChatGoogleGenerativeAI`` does not have this problem because it holds a
google.auth *credentials object*, which carries a refresh token and mints new
access tokens on demand. This module gives the OpenAI client the same thing:
an httpx auth hook that refreshes the credentials when they expire and stamps
a live bearer on every request. Nothing here tracks expiry itself — google.auth
already does, and reimplementing it is how the Codex path got it wrong.
"""
from __future__ import annotations

import json
import threading
from typing import Any, Optional

# langchain_openai depends on httpx, so importing it here costs nothing and
# lets _ADCBearer subclass httpx.Auth normally. Attaching the base class after
# the fact instead puts httpx.Auth first in the MRO, its no-op auth_flow wins,
# the Authorization header is never stamped, and every request 401s.
import httpx

DEFAULT_LOCATION = "global"


def _raw_predict_url(url: httpx.URL, stream: bool) -> Optional[httpx.URL]:
    """The ``:rawPredict`` address matching a self-hosted endpoint's chat URL, or None.

    Only ``.../endpoints/<id>/chat/completions`` has one. The shared MaaS route,
    ``.../endpoints/openapi/chat/completions``, is served as it is.
    """
    suffix = "/chat/completions"
    path = url.path
    if not path.endswith(suffix):
        return None
    endpoint_path = path[: -len(suffix)]
    parent, _, endpoint_id = endpoint_path.rpartition("/")
    if not parent.endswith("/endpoints") or not endpoint_id or endpoint_id == "openapi":
        return None
    verb = "streamRawPredict" if stream else "rawPredict"
    return url.copy_with(path=f"{endpoint_path}:{verb}")


def _is_stream(request: httpx.Request) -> bool:
    try:
        return json.loads(request.read() or b"{}").get("stream") is True
    except (ValueError, AttributeError):
        return False


def _retarget(request: httpx.Request, url: httpx.URL) -> httpx.Request:
    return httpx.Request(request.method, url, headers=request.headers,
                         content=request.read(), extensions=request.extensions)


class _RawPredictFallback(httpx.BaseTransport):
    """Sends a self-hosted endpoint's chat requests to ``:rawPredict`` when it
    does not serve ``/chat/completions``.

    Vertex answers ``/chat/completions`` on a dedicated endpoint only for some
    deployments. A model uploaded with its own container -- the route left when
    Model Garden's deploy is refused at its quota pre-check -- is reachable only
    through ``:rawPredict``, which passes the body unchanged to the container's
    predict route. With that route set to the server's own OpenAI chat path,
    request and reply are exactly what ChatOpenAI sends and parses. Measured on
    qwen38-27b-fp8-chat (endpoint 191566261639970816): ``/chat/completions``
    returned 404 UNIMPLEMENTED, while ``:rawPredict`` returned a parsed
    get_weather call and, given the tool result, a final answer.

    A request first goes where ChatOpenAI addressed it. Only a 404 there sends
    it to ``:rawPredict``, and once that answers, later requests go there
    directly, so a deployment that serves ``/chat/completions`` is never touched.
    """

    def __init__(self, inner: httpx.BaseTransport) -> None:
        self._inner = inner
        self._raw_predict = False

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        raw_url = _raw_predict_url(request.url, _is_stream(request))
        if raw_url is None:
            return self._inner.handle_request(request)
        if self._raw_predict:
            return self._inner.handle_request(_retarget(request, raw_url))
        response = self._inner.handle_request(request)
        if response.status_code != 404:
            return response
        response.read()
        response.close()
        retry = self._inner.handle_request(_retarget(request, raw_url))
        if retry.status_code != 404:
            self._raw_predict = True
        return retry

    def close(self) -> None:
        self._inner.close()


class _ADCBearer(httpx.Auth):
    """httpx auth hook stamping a live Application Default Credentials token."""

    def __init__(self, scopes: Optional[list] = None) -> None:
        import google.auth

        self._credentials, self.project_id = google.auth.default(
            scopes=scopes or ["https://www.googleapis.com/auth/cloud-platform"]
        )
        import google.auth.transport.requests as _gtr

        self._request = _gtr.Request()
        # Candidates run as concurrent subagents sharing one client, so several
        # threads can reach expiry together. google.auth credentials are not
        # thread-safe to refresh; without this two threads race the token
        # write and one of them signs its request with a torn value.
        self._lock = threading.Lock()

    def _token(self) -> str:
        # `valid` is False both when never fetched and when expired, so this
        # covers the first call and every renewal with one test.
        if not self._credentials.valid:
            with self._lock:
                # Re-check: another thread may have refreshed while we waited,
                # and refreshing again would be a wasted network round trip on
                # every concurrent request at the hour boundary.
                if not self._credentials.valid:
                    self._credentials.refresh(self._request)
        return self._credentials.token

    # httpx calls this per request.
    def auth_flow(self, request):  # type: ignore[no-untyped-def]
        request.headers["Authorization"] = f"Bearer {self._token()}"
        yield request


def create_vertex_openai_chat_model(
    model: str,
    temperature: float = 0.0,
    *,
    project_id: str = "",
    location: str = "",
    callbacks: Optional[list] = None,
    timeout: int = 600,
) -> Any:
    """A ChatOpenAI bound to a Vertex OpenAI-compatible endpoint, with ADC auth."""
    from langchain_openai import ChatOpenAI

    auth = _ADCBearer()
    project = (project_id or auth.project_id or "").strip()
    if not project:
        raise ValueError(
            "No Google Cloud project for the Vertex OpenAI endpoint. Set "
            "GOOGLE_CLOUD_PROJECT, or run `gcloud config set project <id>`."
        )
    region = (location or DEFAULT_LOCATION).strip()
    base_url = (
        f"https://aiplatform.googleapis.com/v1/projects/{project}"
        f"/locations/{region}/endpoints/openapi"
    )
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        base_url=base_url,
        # Never used: the httpx auth hook sets the header. ChatOpenAI still
        # requires the argument, and leaving it unset makes it read
        # OPENAI_API_KEY, which would be the wrong credential entirely.
        api_key="vertex-adc",
        # See create_vertex_endpoint_chat_model: without this the SDK sends
        # timeout=None per request and the httpx value is ignored.
        timeout=timeout,
        http_client=httpx.Client(auth=auth, timeout=timeout),
        callbacks=callbacks or [],
        # See create_vertex_endpoint_chat_model.
        stream_usage=True,
    )

def create_vertex_endpoint_chat_model(
    model: str,
    temperature: float = 0.0,
    *,
    base_url: str = "",
    callbacks: Optional[list] = None,
    timeout: int = 600,
    max_retries: Optional[int] = None,
    extra_body: Optional[dict] = None,
) -> Any:
    """A ChatOpenAI bound to a self-hosted Vertex endpoint, with ADC auth.

    ``extra_body`` is merged into every request, e.g. the chat template's
    thinking switch.

    The MaaS route above is a shared publisher endpoint whose URL can be derived
    from the project alone. A model you deploy yourself cannot: it lives behind a
    per-endpoint dedicated host, so the caller supplies the whole base URL and
    this only attaches the same refreshing credential. Everything in the module
    docstring about the 60-minute token applies here identically -- a
    self-hosted endpoint is reached with the same bearer.
    """
    from langchain_openai import ChatOpenAI

    base = (base_url or "").strip().rstrip("/")
    if not base:
        raise ValueError(
            "A self-hosted Vertex endpoint needs its full base URL, e.g. "
            "https://<endpoint-id>.<region>-<project-number>.prediction.vertexai.goog"
            "/v1/projects/<project>/locations/<region>/endpoints/<endpoint-id>"
        )
    class _RepairingChatOpenAI(ChatOpenAI):
        """ChatOpenAI that recovers a tool call the model wrote as prose.

        Same repair as the Gemini/MaaS path in factory.py, applied here
        because this route hands back a plain ChatOpenAI with no wrapper of
        its own -- and it is the route llama-4-maverick runs on, the model
        measured narrating [oed_prepare_baseline(...)] and
        [interpret_case(...)] as message text and ending the study each time.
        """

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            result = super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
            try:
                from cfd_langgraph.llm.factory import _repair_narrated_tool_calls
                from langchain_core.outputs import ChatResult

                fixed = _repair_narrated_tool_calls(self, list(result.generations), kwargs)
                return ChatResult(generations=fixed, llm_output=result.llm_output)
            except Exception:
                return result

    return _RepairingChatOpenAI(
        model=model,
        temperature=temperature,
        base_url=base,
        api_key="vertex-adc",
        # The openai SDK sends its own per-request timeout, which overrides the
        # httpx client's. Left unset, ChatOpenAI passes None — no timeout at
        # all — so CFD_SCIENTIST_VERTEX_TIMEOUT=60 never applied and a llama
        # call sat 9+ minutes on a silent socket (malmo_llama_20260911).
        timeout=timeout,
        http_client=httpx.Client(auth=_ADCBearer(), timeout=timeout,
                                 transport=_RawPredictFallback(httpx.HTTPTransport())),
        callbacks=callbacks or [],
        # ChatOpenAI asks for usage on a streamed reply only when talking to
        # api.openai.com. Everywhere else a stream ends with no token counts,
        # and every _llm_invoke call streams (for its heartbeat): study_mode,
        # metric_setup, surrogate_setup, run_validity, OED. On the Qwen studies
        # each of those calls was logged as zero tokens.
        stream_usage=True,
        **({"max_retries": max_retries} if max_retries is not None else {}),
        **({"extra_body": extra_body} if extra_body else {}),
    )
