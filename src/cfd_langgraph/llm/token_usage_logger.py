from __future__ import annotations

import contextvars
import json
import os
import sys
import fcntl
from datetime import datetime, timezone
from pathlib import Path
from contextlib import contextmanager
from threading import Lock
from typing import Any, Dict, Iterable


_LOCK = Lock()

# How many recent calls the JSON keeps inline, for reading by eye. The
# complete history is in the .jsonl sidecar next to it.
_CALLS_TAIL = int(os.getenv("CFD_TOKEN_LOG_TAIL") or 500)

_STAGE_KEY = "CFD_SCIENTIST_STAGE"
# The script that a stage label was set for. A script launched by another
# script inherits the parent's environment, label included, so without this a
# nested stage is filed under its parent: code_mod_agentic.py run by
# open_ended_discovery.py was counted as open_ended_discovery, and so were
# interpret.py and viz.py run from there. See _adopt_stage.
_OWNER_KEY = "CFD_SCIENTIST_STAGE_OWNER"

# Which tool, inside a process, made the call. A contextvar rather than an env
# var because the manager runs tools concurrently in a thread pool; LangGraph
# copies the context into each tool thread, so every tool sees only its own.
_CALLER: contextvars.ContextVar[str] = contextvars.ContextVar("cfd_token_caller", default="")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return default


def process_name() -> str:
    """This process's script name, e.g. "code_mod_agentic"; "" for python -c."""
    arg0 = sys.argv[0] if sys.argv else ""
    if not arg0 or arg0 in ("-c", "-m"):
        return ""
    return Path(arg0).stem


def _adopt_stage() -> None:
    """Relabel this process when it is a script launched by another script.

    Runs once, at import. If the stage in the environment was set for a
    different script, this process is its child, so its calls are filed as
    "<parent stage>/<this script>" and anything it launches inherits that
    chain. A process that the label was set for directly, or a forked copy of
    one, keeps the label unchanged.
    """
    stage = (os.environ.get(_STAGE_KEY) or "").strip()
    owner = (os.environ.get(_OWNER_KEY) or "").strip()
    me = process_name()
    if stage and owner and me and owner != me:
        os.environ[_STAGE_KEY] = f"{stage}/{me}"
        os.environ[_OWNER_KEY] = me


_adopt_stage()


def current_stage() -> str:
    """Which pipeline stage the calling code is in.

    Read from the environment rather than a module global because most stages
    are subprocesses -- interpret.py, viz.py, code_mod_agentic.py,
    open_ended_discovery.py -- launched with os.environ.copy(), so an env var
    is the only channel that reaches them without threading an argument
    through every call site. In-process stages set it with ``stage_scope``.
    """
    return (os.getenv(_STAGE_KEY) or "").strip() or "unattributed"


def stage_env(env: Dict[str, str], script: str) -> Dict[str, str]:
    """Label a subprocess environment with the script about to run in it."""
    name = Path(script).stem
    env[_STAGE_KEY] = name
    env[_OWNER_KEY] = name
    return env


@contextmanager
def stage_scope(stage: str):
    """Attribute every LLM call made inside this block to ``stage``.

    Restores the previous value on exit, so nested stages behave sensibly and
    a stage that spawns subprocesses passes its own name down to them.
    """
    previous = {k: os.environ.get(k) for k in (_STAGE_KEY, _OWNER_KEY)}
    os.environ[_STAGE_KEY] = str(stage or "").strip() or "unattributed"
    os.environ[_OWNER_KEY] = process_name()
    try:
        yield
    finally:
        for k, v in previous.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@contextmanager
def caller_scope(name: str):
    """Attribute unlabelled LLM calls made inside this block to ``name``.

    Used around each manager tool, so the model calls a tool makes for itself
    (classifying the study, setting up metrics, checking a run) are filed
    under that tool instead of under the manager's own reasoning.
    """
    token = _CALLER.set(str(name or "").strip())
    try:
        yield
    finally:
        _CALLER.reset(token)


def current_caller() -> str:
    """The manager tool the calling code is running inside, if any."""
    return _CALLER.get()


def _resolve_path() -> Path:
    file_path = (os.getenv("CFD_TOKEN_LOG_PATH") or "").strip()
    if file_path:
        return Path(file_path).expanduser().resolve()
    dir_path = (os.getenv("CFD_TOKEN_LOG_DIR") or "").strip()
    if dir_path:
        return (Path(dir_path).expanduser().resolve() / "llm_token_usage.json")
    return (Path.cwd() / "llm_token_usage.json").resolve()


def _sidecar(path: Path) -> Path:
    return path.with_name(path.stem + "_calls.jsonl")


def _empty() -> Dict[str, Any]:
    return {
        "schema_version": "v1",
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
        "totals": {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "calls": 0,
            "by_model": {},
        },
        "calls": [],
    }


def read_rows(path: Path) -> list:
    rows = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
    except OSError:
        pass
    return rows


def _load(path: Path) -> Dict[str, Any]:
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                data.setdefault("schema_version", "v1")
                data.setdefault("totals", {})
                data.setdefault("calls", [])
                t = data["totals"]
                t.setdefault("input_tokens", 0)
                t.setdefault("output_tokens", 0)
                t.setdefault("total_tokens", 0)
                t.setdefault("calls", 0)
                t.setdefault("by_model", {})
                return data
        except Exception:
            pass
        # Unreadable. Starting from zero here would silently restart the
        # study's totals, so rebuild them from the per-call sidecar, which
        # holds every row, and keep the broken file for inspection.
        rows = read_rows(_sidecar(path))
        try:
            path.rename(path.with_name(f"{path.name}.corrupt-{datetime.now():%Y%m%d_%H%M%S}"))
        except OSError:
            pass
        data = _empty()
        data["totals"] = rebuild_totals(rows)
        data["calls"] = rows[-_CALLS_TAIL:]
        data["rebuilt_from_sidecar_at"] = _now_iso()
        return data
    return _empty()


def _bump(bucket: Dict[str, Any], row: Dict[str, Any]) -> None:
    for key in ("input_tokens", "output_tokens", "total_tokens",
                "cached_input_tokens", "reasoning_output_tokens"):
        bucket[key] = _to_int(bucket.get(key)) + _to_int(row.get(key))
    bucket["calls"] = _to_int(bucket.get("calls")) + 1


def _add_to_totals(totals: Dict[str, Any], row: Dict[str, Any]) -> None:
    """Fold one call into every breakdown. The only place totals are computed,
    so the running totals and a rebuild from the sidecar cannot disagree."""
    provider = row.get("provider") or ""
    mkey = row.get("model") or "unknown"
    stage = row.get("stage") or "unattributed"
    agent = row.get("agent") or "manager"
    _bump(totals, row)
    # How many of the calls carried the provider's own usage, and how many
    # were estimates -- so a total is never mistaken for more than it is.
    src = row.get("token_source") or ""
    src_counts = totals.setdefault("calls_by_token_source", {})
    src_counts[src] = _to_int(src_counts.get(src)) + 1
    model_totals = totals.setdefault("by_model", {}).setdefault(mkey, {"provider": provider})
    model_totals["provider"] = provider or model_totals.get("provider", "")
    _bump(model_totals, row)
    # Per stage, and per model within each stage. The nesting is what makes a
    # mixed-model study legible: "ideation cost X on the frontier model, build
    # agents cost Y on the cheap one" is the question this file exists to
    # answer, and a flat by_model block cannot answer it.
    stage_totals = totals.setdefault("by_stage", {}).setdefault(stage, {"by_model": {}})
    _bump(stage_totals, row)
    sm = stage_totals.setdefault("by_model", {}).setdefault(mkey, {"provider": provider})
    sm["provider"] = provider or sm.get("provider", "")
    _bump(sm, row)
    # Per agent within the stage. Everything running in the manager process
    # is one stage, but several roles make calls there -- the manager graph,
    # the case runner, the OED candidate runner, and each tool's own calls --
    # with very different appetites. Without this split, "the manager cost
    # 64M tokens" cannot be read as either "the manager reasons too much" or
    # "its subagents do", which are opposite conclusions.
    _bump(stage_totals.setdefault("by_agent", {}).setdefault(agent, {}), row)


def rebuild_totals(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Totals recomputed from per-call rows -- what the running totals must equal."""
    totals: Dict[str, Any] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
                              "calls": 0, "by_model": {}}
    for row in rows:
        _add_to_totals(totals, row)
    return totals


def append_usage_call(
    *,
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    token_source: str,
    stage: str | None = None,
    agent: str | None = None,
    cached_input_tokens: int = 0,
    reasoning_output_tokens: int = 0,
    response_ids: list | None = None,
    attempts: int = 1,
) -> None:
    stage = (stage or "").strip() or current_stage()
    # Which agent inside the stage made the call. ``stage`` separates
    # processes (manager vs code_mod_agentic vs analyze); ``agent`` separates
    # the roles that share a process. A model built for a role carries its
    # label on its callback handler (the manager graph, the case runner, the
    # OED candidate runner) -- on the instance rather than on os.environ,
    # because the manager runs tool calls in a thread pool and a
    # process-global label would be read by whichever thread logged next. An
    # unlabelled model called from inside a manager tool is filed under that
    # tool; anything else is the stage's own work, so it takes the stage's
    # name. It used to default to "manager" everywhere, which filed every
    # build agent's and interpreter's calls under an agent that never made them.
    agent = (agent or "").strip() or _CALLER.get() or stage
    in_tok = _to_int(input_tokens)
    out_tok = _to_int(output_tokens)
    row = {
        "ts": _now_iso(),
        "provider": provider or "",
        "model": model or "",
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "total_tokens": in_tok + out_tok,
        # Subsets, not additions: cached tokens are part of input_tokens and
        # reasoning tokens part of output_tokens, as every provider reports
        # them. Kept separately because cached input is billed and served
        # very differently, and it dominates long agent sessions.
        "cached_input_tokens": _to_int(cached_input_tokens),
        "reasoning_output_tokens": _to_int(reasoning_output_tokens),
        "token_source": token_source,
        "response_ids": list(response_ids or []),
        "attempts": _to_int(attempts, 1),
        "stage": stage,
        "agent": agent,
        "script": process_name(),
        "pid": os.getpid(),
        "success": True,
    }
    append_row(row)


def append_row(row: Dict[str, Any]) -> None:
    """Record one call: the sidecar gets the row, the JSON its totals and tail."""
    path = _resolve_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with _LOCK:
        with open(lock_path, "w", encoding="utf-8") as lock_f:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
            try:
                data = _load(path)
                # Full per-call history goes to an append-only sidecar; the
                # JSON keeps only a recent tail. Appending to the in-JSON array
                # meant rewriting the whole file on every LLM call, so the cost
                # of logging grew with the square of the call count: measured
                # at 0.39s per append once the shared log reached 14.5 MB and
                # 59k calls, which over a study's ~26k codex calls is more than
                # an hour spent re-serialising accounting. Totals stay exact
                # either way; only where the individual rows live changes.
                sidecar = _sidecar(path)
                try:
                    if not sidecar.exists() and data["calls"]:
                        # A log from before the sidecar holds its history only
                        # in the JSON array. Copy that in first: once the new
                        # row is in the sidecar, the trim below counts it as
                        # already preserved and drops an old row instead.
                        with open(sidecar, "w", encoding="utf-8") as sf:
                            for old_row in data["calls"]:
                                sf.write(json.dumps(old_row) + "\n")
                    with open(sidecar, "a", encoding="utf-8") as jf:
                        jf.write(json.dumps(row) + "\n")
                except OSError:
                    pass
                data["calls"].append(row)
                if len(data["calls"]) > _CALLS_TAIL:
                    # Migrate before trimming, never truncate blind. Measured
                    # the hard way: a live 59,135-row log was cut to 500 the
                    # moment a subprocess picked up new code, because
                    # subprocesses import current source mid-run.
                    overflow = data["calls"][:-_CALLS_TAIL]
                    try:
                        known = 0
                        if sidecar.exists():
                            with open(sidecar, "r", encoding="utf-8") as sf:
                                known = sum(1 for _ in sf)
                        if known < len(overflow):
                            with open(sidecar, "a", encoding="utf-8") as sf:
                                for old_row in overflow[known:]:
                                    sf.write(json.dumps(old_row) + "\n")
                    except OSError:
                        # Could not preserve them, so do not discard them.
                        overflow = []
                    if overflow:
                        del data["calls"][:-_CALLS_TAIL]
                _add_to_totals(data["totals"], row)
                data["updated_at"] = _now_iso()
                # Written whole to a temporary file and moved into place, so a
                # process killed mid-write (an LLM child cut off on timeout)
                # leaves the previous totals rather than half a file.
                tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
                tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
                os.replace(tmp, path)
            finally:
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)
