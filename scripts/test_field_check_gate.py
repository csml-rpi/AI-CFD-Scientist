#!/usr/bin/env python3
"""When the discovery loop renders a candidate's field, and when it does not.

A comparator score cannot see the solution it came from: a quantity read on a
boundary leaves the interior unconstrained, so a candidate can match it with a
wrong field. Rendering every candidate is too slow, so the render is gated on
the candidate taking the lead.

Covers the gate, and the incumbent it is compared against when the caller does
not supply one.

Run: python scripts/test_field_check_gate.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from open_ended_discovery import _incumbent_best_from_history, _takes_the_lead  # noqa: E402

FAILURES: list[str] = []


def check(name: str, cond: object, detail: str = "") -> None:
    if cond:
        print(f"[PASS] {name}")
    else:
        FAILURES.append(name)
        print(f"[FAIL] {name}" + (f" — {detail}" if detail else ""))


print("== 1. minimising: a lower score leads")
check("beats the leader", _takes_the_lead(0.0030, 0.0043, "min"))
check("does not beat it", not _takes_the_lead(0.0050, 0.0043, "min"))
check("a tie does not lead", not _takes_the_lead(0.0043, 0.0043, "min"))

print("== 2. maximising")
check("higher leads", _takes_the_lead(0.9, 0.7, "max"))
check("lower does not", not _takes_the_lead(0.5, 0.7, "max"))
check("'maximize' is recognised too", _takes_the_lead(0.9, 0.7, "maximize"))

print("== 3. nothing leading yet means the first candidate IS checked")
check("no incumbent, so it leads", _takes_the_lead(0.0043, None, "min"))
check("an unreadable incumbent also leads", _takes_the_lead(0.0043, "n/a", "min"))

print("== 4. an unusable score is never treated as leading")
for bad in (None, "", "n/a", float("nan")):
    got = _takes_the_lead(bad, 0.0043, "min")
    if bad != bad:  # NaN: comparisons are False, so it cannot lead
        check("a NaN score does not lead", not got)
    else:
        check(f"{bad!r} does not lead", not got)

print("== 5. the incumbent read from history on disk")
tmp = Path(tempfile.mkdtemp())
disc = tmp / "open_ended_discovery"
case = disc / "cand_x"
case.mkdir(parents=True)

check("no history file means nothing leads yet",
      _incumbent_best_from_history(case, "min") is None)

(disc / "history.json").write_text(json.dumps([
    {"iteration": 1, "metric_aggregated": {"primary": 0.0050}},
    {"iteration": 2, "metric_aggregated": {"primary": 0.0041}},
    {"iteration": 3, "metric_aggregated": {"primary": 0.0047}},
]))
check("minimising takes the smallest", _incumbent_best_from_history(case, "min") == 0.0041)
check("maximising takes the largest", _incumbent_best_from_history(case, "max") == 0.0050)

(disc / "history.json").write_text(json.dumps([
    {"iteration": 1, "score": {"value": 0.0062, "metric": "m"}},
    {"iteration": 2, "metric_aggregated": {"primary": None}, "score": {"value": 0.0039}},
]))
check("it falls back to score.value", _incumbent_best_from_history(case, "min") == 0.0039)

(disc / "history.json").write_text(json.dumps([
    {"iteration": 1, "status": "FAILED"},
    {"iteration": 2, "metric_aggregated": {}},
    "not a dict",
]))
check("rows with no usable score are skipped, not crashed",
      _incumbent_best_from_history(case, "min") is None)

print("== 6. the two together: the sequence a study actually sees")
(disc / "history.json").write_text(json.dumps([
    {"iteration": 1, "metric_aggregated": {"primary": 0.0043}},
    {"iteration": 2, "metric_aggregated": {"primary": 0.0041}},
]))
inc = _incumbent_best_from_history(case, "min")
check("a worse candidate is scored but not rendered", not _takes_the_lead(0.0045, inc, "min"))
check("a better candidate is rendered", _takes_the_lead(0.0039, inc, "min"))

import shutil
shutil.rmtree(tmp, ignore_errors=True)
print()
print("FAILURES:", FAILURES if FAILURES else "none")
sys.exit(1 if FAILURES else 0)
