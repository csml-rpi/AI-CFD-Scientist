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

import threading
from typing import Any, Optional

# langchain_openai depends on httpx, so importing it here costs nothing and
# lets _ADCBearer subclass httpx.Auth normally. Attaching the base class after
# the fact instead puts httpx.Auth first in the MRO, its no-op auth_flow wins,
# the Authorization header is never stamped, and every request 401s.
import httpx

DEFAULT_LOCATION = "global"


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
        http_client=httpx.Client(auth=auth, timeout=timeout),
        callbacks=callbacks or [],
    )
