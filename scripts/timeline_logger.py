#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_timeline_path(cli_path: Optional[str]) -> Optional[Path]:
    raw = (cli_path or "").strip() or os.environ.get("CFD_ORCH_TIMELINE_PATH", "").strip()
    if not raw:
        return None
    return Path(raw).expanduser().resolve()


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def append_timeline_event(timeline_path: Optional[Path], event: Dict[str, Any]) -> None:
    if timeline_path is None:
        return
    timeline_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _read_json(timeline_path)
    if not isinstance(payload, dict):
        payload = {}
    events = payload.get("events")
    if not isinstance(events, list):
        events = []
    rec = dict(event)
    rec.setdefault("ts", _now_iso())
    events.append(rec)
    payload["events"] = events
    payload["last_updated"] = _now_iso()
    payload.setdefault("schema_version", "v1")
    timeline_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

