#!/usr/bin/env python3
"""When the REPL pushes a stopped study to carry on.

A push meant for a study that stopped during setup also fired on a prescribed
study, which never writes search_config.json: every stop read as "setup
unfinished", and a DAFoam study that had written its paper and reported its
verdict was sent back to work ten times in four days
(shelf_demo_dafoam_20261015_1318).

Run: python scripts/test_repl_continuations.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cfd_langgraph.cli import repl  # noqa: E402

FAILURES: list[str] = []


def check(name: str, got, want) -> None:
    ok = got == want
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + ("" if ok else f" — got {got!r}, want {want!r}"))
    if not ok:
        FAILURES.append(name)


def study(tmp: str, *, prescribed: bool, requirements: bool) -> Path:
    out = Path(tmp)
    (out / "state.json").write_text(json.dumps({"status": "running", "current_stage": "analysis"}))
    (out / "prescribed_study.json").write_text(json.dumps({"prescribed": prescribed}))
    if requirements:
        (out / "requirements.json").write_text("[]")
    return out


with tempfile.TemporaryDirectory() as tmp:
    check("a prescribed study with its cases planned is not pushed as unfinished setup",
          repl._setup_incomplete(study(tmp, prescribed=True, requirements=True)), None)
with tempfile.TemporaryDirectory() as tmp:
    check("a prescribed study that stopped before planning its cases still is",
          repl._setup_incomplete(study(tmp, prescribed=True, requirements=False)), {"stage": "analysis"})
with tempfile.TemporaryDirectory() as tmp:
    check("a search study without search_config.json still is",
          repl._setup_incomplete(study(tmp, prescribed=False, requirements=True)), {"stage": "analysis"})
with tempfile.TemporaryDirectory() as tmp:
    out = study(tmp, prescribed=False, requirements=True)
    (out / "open_ended_discovery").mkdir()
    (out / "open_ended_discovery" / "search_config.json").write_text("{}")
    check("and once its search is set up it is not", repl._setup_incomplete(out), None)

print()
print("FAILURES:", FAILURES if FAILURES else "none")
sys.exit(1 if FAILURES else 0)
