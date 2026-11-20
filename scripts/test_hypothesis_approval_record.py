#!/usr/bin/env python3
"""The hypothesis approval record says who approved.

Auto-approved by the CLI's gate, every approval was still written as
"human_review: approved", and a manager's own amendment to an approved
hypothesis, noted as "human-requested", stood in the record as a human
decision (shelf_demo_dafoam_20261019_1057, 2026-10-20).

Run: python scripts/test_hypothesis_approval_record.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cfd_langgraph.manager.tools as mt  # noqa: E402
from cfd_langgraph.config import get_settings  # noqa: E402

FAILURES: list[str] = []


def check(name: str, got, want=True) -> None:
    ok = got == want
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + ("" if ok else f" — got {got!r}, want {want!r}"))
    if not ok:
        FAILURES.append(name)


def approve(automatic: bool) -> tuple:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        (out / "hypotheses_ranked.json").write_text(json.dumps(
            {"research_topic": "t", "ranked_hypotheses": [{"candidate_id": "cand_01", "idea": {}}]}))
        saved = os.environ.pop("CFD_SCIENTIST_HYPOTHESES_AUTO_APPROVED", None)
        if automatic:
            os.environ["CFD_SCIENTIST_HYPOTHESES_AUTO_APPROVED"] = "1"
        try:
            tools = {fn.__name__: fn for fn in mt.build_manager_tools(get_settings(), out)["manager_tools"]}
            tools["advance_with_approved_hypotheses"](["cand_01"], notes="Human-requested amendment")
        finally:
            os.environ.pop("CFD_SCIENTIST_HYPOTHESES_AUTO_APPROVED", None)
            if saved is not None:
                os.environ["CFD_SCIENTIST_HYPOTHESES_AUTO_APPROVED"] = saved
        return (json.loads((out / "hypotheses_approved.json").read_text()),
                json.loads((out / "hypotheses_ranked.json").read_text())["human_review"])


record, review = approve(automatic=True)
check("an auto-approval is recorded as automatic", "no human reviewed" in record.get("approved_by", ""))
check("whatever its notes claim", record.get("notes"), "Human-requested amendment")
check("and the review status says so", review.get("status"), "auto_approved")
record, review = approve(automatic=False)
check("an approval at the asking gate is recorded as a human's", record.get("approved_by", "").startswith("a human"))
check("with the status approved", review.get("status"), "approved")

print()
print("FAILURES:", FAILURES if FAILURES else "none")
sys.exit(1 if FAILURES else 0)
