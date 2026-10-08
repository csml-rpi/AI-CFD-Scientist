"""ALCF's inference endpoints (Sophia, Metis, Minerva), as an OpenAI client.

ALCF serves open-weight models from its own clusters behind one
OpenAI-compatible gateway, authenticated with a Globus token rather than an API
key. Docs: https://docs.alcf.anl.gov/services/inference-endpoints

The token is the only real difference from any other OpenAI-compatible
endpoint. It is valid for 48 hours and ``get_access_token()`` refreshes it in
place, so a study that runs for days must ask for it per request rather than
read it once at start-up -- the same reason the Vertex route holds a
credentials object instead of a bearer string (see vertex_openai.py).

First-time setup, once per machine (interactive, needs a browser):

    python scripts/inference_auth_token.py authenticate

Then:

    export CFD_SCIENTIST_LLM_PROVIDER=alcf
    export CFD_SCIENTIST_MODEL=<a model id from /resource_server/list-endpoints>
    export CFD_SCIENTIST_ALCF_CLUSTER=sophia      # or metis, minerva
"""
from __future__ import annotations

import importlib.util
import os
import threading
import time
from pathlib import Path
from typing import Any, Optional

import httpx

GATEWAY = "https://inference-api.alcf.anl.gov/resource_server"

# Each cluster runs its own serving stack behind the same gateway, and they do
# not share a URL shape: Sophia is vLLM, the other two are vendor APIs.
CLUSTER_PATHS = {
    "sophia": "/sophia/vllm/v1",
    "metis": "/metis/api/v1",
    "minerva": "/minerva/api/v1",
}

_AUTH_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "inference_auth_token.py"


def _auth_module() -> Any:
    """ALCF's own token helper, loaded from the copy in scripts/.

    Vendored as a file rather than reimplemented: it owns the client id, the
    scope and the token file's location, and all three are ALCF's to change.
    """
    spec = importlib.util.spec_from_file_location("inference_auth_token", _AUTH_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"ALCF auth helper not found at {_AUTH_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _GlobusBearer(httpx.Auth):
    """Attaches a live ALCF token, refreshed as it ages.

    Cached for ``_TTL_S`` because every call would otherwise re-read the token
    file (and, once it expires, make a refresh round trip) on a request the
    model is already waiting on.
    """

    _TTL_S = 1800

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._token = ""
        self._fetched_at = 0.0

    def _access_token(self) -> str:
        with self._lock:
            if self._token and (time.monotonic() - self._fetched_at) < self._TTL_S:
                return self._token
            try:
                self._token = str(_auth_module().get_access_token() or "").strip()
            except Exception as exc:  # no token, or a refresh that failed
                raise RuntimeError(
                    "No usable ALCF token. Run `python scripts/inference_auth_token.py "
                    "authenticate` once on this machine (re-authentication is required "
                    f"every 30 days). Underlying error: {exc}"
                ) from exc
            self._fetched_at = time.monotonic()
            return self._token

    def auth_flow(self, request):  # type: ignore[no-untyped-def]
        request.headers["Authorization"] = f"Bearer {self._access_token()}"
        yield request


def cluster_base_url(cluster: str = "") -> str:
    name = (cluster or os.environ.get("CFD_SCIENTIST_ALCF_CLUSTER") or "sophia").strip().lower()
    if name not in CLUSTER_PATHS:
        raise ValueError(
            f"Unknown ALCF cluster {name!r}. Pick one of: {', '.join(sorted(CLUSTER_PATHS))}."
        )
    return f"{GATEWAY}{CLUSTER_PATHS[name]}"


def list_endpoints(timeout: int = 60) -> dict:
    """What the gateway is serving, and which models are hot right now."""
    with httpx.Client(auth=_GlobusBearer(), timeout=timeout) as client:
        r = client.get(f"{GATEWAY}/list-endpoints")
        r.raise_for_status()
        return r.json()


def create_alcf_chat_model(
    model: str,
    temperature: float = 0.0,
    *,
    cluster: str = "",
    callbacks: Optional[list] = None,
    timeout: int = 600,
    max_retries: Optional[int] = None,
    extra_body: Optional[dict] = None,
    cls: Any = None,
) -> Any:
    """A ChatOpenAI bound to one ALCF cluster, with a refreshing Globus token.

    ``cls`` lets the factory pass its tool-call-repairing subclass, so this
    route recovers a narrated tool call like every other provider does.
    """
    from langchain_openai import ChatOpenAI

    return (cls or ChatOpenAI)(
        model=model,
        temperature=temperature,
        base_url=cluster_base_url(cluster),
        # Never sent: the httpx auth hook sets the header on every request, so
        # the token is always the current one. ChatOpenAI still demands the
        # argument, and leaving it unset makes it read OPENAI_API_KEY.
        api_key="alcf-globus",
        timeout=timeout,
        http_client=httpx.Client(auth=_GlobusBearer(), timeout=timeout),
        callbacks=callbacks or [],
        # Usage on streamed replies is opt-in for any base URL but OpenAI's.
        stream_usage=True,
        **({"max_retries": max_retries} if max_retries is not None else {}),
        **({"extra_body": extra_body} if extra_body else {}),
    )
