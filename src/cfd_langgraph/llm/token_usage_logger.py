from __future__ import annotations

import json
import os
import fcntl
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict


_LOCK = Lock()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return default


def _resolve_path() -> Path:
    file_path = (os.getenv("CFD_TOKEN_LOG_PATH") or "").strip()
    if file_path:
        return Path(file_path).expanduser().resolve()
    dir_path = (os.getenv("CFD_TOKEN_LOG_DIR") or "").strip()
    if dir_path:
        return (Path(dir_path).expanduser().resolve() / "llm_token_usage.json")
    return (Path.cwd() / "llm_token_usage.json").resolve()


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


def append_usage_call(
    *,
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    token_source: str,
) -> None:
    path = _resolve_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        with open(lock_path, "w", encoding="utf-8") as lock_f:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
            try:
                data = _load(path)
                in_tok = _to_int(input_tokens)
                out_tok = _to_int(output_tokens)
                tot_tok = in_tok + out_tok
                row = {
                    "ts": _now_iso(),
                    "provider": provider or "",
                    "model": model or "",
                    "input_tokens": in_tok,
                    "output_tokens": out_tok,
                    "total_tokens": tot_tok,
                    "token_source": token_source,
                    "success": True,
                }
                data["calls"].append(row)
                totals = data["totals"]
                totals["input_tokens"] = _to_int(totals.get("input_tokens")) + in_tok
                totals["output_tokens"] = _to_int(totals.get("output_tokens")) + out_tok
                totals["total_tokens"] = _to_int(totals.get("total_tokens")) + tot_tok
                totals["calls"] = _to_int(totals.get("calls")) + 1
                by_model = totals.setdefault("by_model", {})
                mkey = model or "unknown"
                model_totals = by_model.setdefault(
                    mkey, {"provider": provider or "", "input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "calls": 0}
                )
                model_totals["provider"] = provider or model_totals.get("provider", "")
                model_totals["input_tokens"] = _to_int(model_totals.get("input_tokens")) + in_tok
                model_totals["output_tokens"] = _to_int(model_totals.get("output_tokens")) + out_tok
                model_totals["total_tokens"] = _to_int(model_totals.get("total_tokens")) + tot_tok
                model_totals["calls"] = _to_int(model_totals.get("calls")) + 1
                data["updated_at"] = _now_iso()
                path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            finally:
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)

