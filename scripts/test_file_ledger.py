#!/usr/bin/env python3
"""The study's file ledger (cfd_langgraph.file_ledger): every file is found by
who wrote it, when and what it is about, not by guessing names and places.

1. A run's writes are recorded from the case folder before and after it, at
   any depth and in hidden folders, copies keeping an old mtime too; what the
   harness wrote before the run is not the run's.
2. The review's readers (_run_outputs, _run_logs, _run_reports, the listing)
   take the run's files from the ledger: a report five folders down is shown.
3. Field data is counted, not listed.
4. A folder moved, then renumbered, still leads from its old path.
5. A file recorded as about another goes with it; the ledger is protected.
6. A manager tool's record leaves out folders being run.

Run: python scripts/test_file_ledger.py
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.environ.update(CFD_SCIENTIST_LLM_PROVIDER="openai", CFD_SCIENTIST_MODEL="fake/model",
                  OPENAI_API_KEY="fake", OPENAI_BASE_URL="http://127.0.0.1:9/v1")

from cfd_langgraph import file_ledger as L  # noqa: E402
from cfd_langgraph.foam_native import loop  # noqa: E402

FAILURES = []


def check(label, got, want=True):
    if got == want:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label}\n        got={got!r}\n        want={want!r}")
        FAILURES.append(label)


with tempfile.TemporaryDirectory() as tmp:
    study = Path(tmp) / "study"
    case = study / "cases" / "case_006"
    (case / "system").mkdir(parents=True)
    L.ensure(study)
    old = Path(tmp) / "old_record.json"
    old.write_text(json.dumps({"status": "kept"}))
    os.utime(old, (time.time() - 86400, time.time() - 86400))

    ledger = loop._CaseLedger(case)
    (case / "system" / "controlDict").write_text("application simpleFoam;\n")  # the harness's
    started = time.time()
    ledger.before_run(1)
    deep = case / "campaign_run" / "evaluations" / "inv" / "eval-1" / "results.json"
    deep.parent.mkdir(parents=True)
    deep.write_text(json.dumps({"status": "production_rejected_failed_checks", "failures": ["no record"]}))
    (case / ".deploy" / "logs").mkdir(parents=True)
    (case / ".deploy" / "logs" / "build.log").write_text("make: *** [all] Error 1\n")
    shutil.copy2(old, case / "copied_record.json")  # keeps its old mtime
    (case / "processor0" / "100").mkdir(parents=True)
    for field in ("U", "p", "nut"):
        (case / "processor0" / "100" / field).write_text("internalField uniform 0;\n")
    (case / "weights.bin").write_bytes(b"\x00\x01")
    written = ledger.after_run(1)

    print("1. a run's writes")
    run_files = {str(p) for p in loop._written(case, started)}
    check("a report five folders down is the run's",
          "campaign_run/evaluations/inv/eval-1/results.json" in run_files)
    check("so is a log in a hidden folder", ".deploy/logs/build.log" in run_files)
    check("and a copy that kept its old mtime", "copied_record.json" in run_files)
    check("but not the dictionary the harness wrote before the run", "system/controlDict" not in run_files)
    by = {e["path"]: e for e in L.entries(study)}
    check("the harness's write is recorded as the harness's",
          by["cases/case_006/system/controlDict"]["source"], "harness")
    check("with a hash", len(by["cases/case_006/copied_record.json"].get("sha256", "")), 64)

    print("2. the review's readers")
    shown = "\n".join(loop._run_outputs(case, started))
    check("_run_outputs shows the deep report", "production_rejected_failed_checks" in shown)
    check("_run_logs shows the hidden log", "Error 1" in "\n".join(loop._run_logs(case, started)))
    check("_run_verdicts reads it", loop._run_verdicts(case, started), ["production_rejected_failed_checks"])
    listing = loop._written_listing(case, started)
    check("the listing names a file nothing shows", "weights.bin (2 B)" in listing)

    print("2b. a run's logs outlive the next run")
    (case / "log.simpleFoam").write_text("run 1: FOAM FATAL ERROR here\n")
    ledger.after_run(1)
    ledger.before_run(2)
    (case / "log.simpleFoam").write_text("run 2\n")
    ledger.after_run(2)
    kept = study / ".ledger" / "runs" / "cases" / "case_006"
    check("run 1's log is kept as run 1 left it",
          (kept / "run_1" / "log.simpleFoam").read_text(), "run 1: FOAM FATAL ERROR here\n")
    check("and run 2's as run 2 left it", (kept / "run_2" / "log.simpleFoam").read_text(), "run 2\n")

    print("3. field data")
    check("is counted", written["fields"], {"cases/case_006/processor0": 3})
    check("not listed", any("processor0" in p for p in written["created"]), False)

    print("4. moves")
    attempts = study / "previous_attempts"
    attempts.mkdir()
    case.rename(attempts / "case_006")
    L.moved(study, case, attempts / "case_006", "test")
    (attempts / "case_006").rename(attempts / "case_006.1")
    L.moved(study, attempts / "case_006", attempts / "case_006.1", "test")
    check("an old path leads to the renumbered attempt",
          L.current_path(study, case / "copied_record.json"), attempts / "case_006.1" / "copied_record.json")
    check("a path nothing leads from is None", L.current_path(study, study / "nowhere.json"), None)

    print("5. about, and protection")
    card = study / "network" / "card.json"
    card.parent.mkdir()
    card.write_text("{}")
    (card.parent / "checks.jsonl").write_text("{}\n")
    L.note(study, card.parent / "checks.jsonl", "test", about=[card])
    check("a card's checks are about it", L.about(study, card), [card.parent / "checks.jsonl"])
    from cfd_langgraph.manager import tools  # noqa: E402
    check("the ledger is a protected record",
          tools._is_protected_artifact(study / ".ledger" / "files.jsonl", (".ledger", "files.jsonl")))

    print("6. folders being run")
    other = study / "cases" / "case_004"
    other.mkdir(parents=True)
    L.running(other, True)
    check("are left out of a tool's record", L.running_under(study), ["cases/case_004"])
    L.running(other, False)
    check("until their run ends", L.running_under(study), [])

with tempfile.TemporaryDirectory() as tmp:
    print("8. a run solved in its workers' logs")
    case = Path(tmp) / "case"
    (case / "conditions" / "4.08").mkdir(parents=True)
    (case / "conditions" / "8.69").mkdir(parents=True)
    started = time.time() - 1
    (case / "log.DASimpleFoam").write_text("End\n")
    steps = "".join(f"\nTime = {i}\nCL: 0.5\nExecutionTime = {i} s\n" for i in range(1, 6))
    (case / "conditions" / "4.08" / "solver_output.txt").write_text(steps + "\nFinalising parallel run\n")
    (case / "conditions" / "8.69" / "solver_output.txt").write_text(steps + "\nFinalising parallel run\n")
    (case / "log.decomposePar").write_text("\nTime = 0\nExecutionTime = 0.1 s\nEnd\n")
    check("workers' logs that iterate and end cleanly count as solved", loop._solved_in_written_logs(case, started))
    (case / "conditions" / "8.69" / "solver_output.txt").write_text(
        steps + "\n--> FOAM FATAL ERROR: (openfoam-2506)\n\n    failed lookup of chiSA\n\nFOAM parallel run exiting\n")
    check("one worker's fatal error fails the run", loop._solved_in_written_logs(case, started), False)
    (case / "conditions" / "4.08" / "solver_output.txt").unlink()
    (case / "conditions" / "8.69" / "solver_output.txt").unlink()
    check("a utility's single step is not a solve", loop._solved_in_written_logs(case, started), False)

with tempfile.TemporaryDirectory() as tmp:
    print("7. a case outside a study")
    case = Path(tmp) / "case"
    case.mkdir()
    started = time.time() - 1
    (case / "deep" / "er").mkdir(parents=True)
    (case / "deep" / "er" / "report.json").write_text('{"status": "a_b"}')
    check("its run's files are found from the folder itself",
          [str(p) for p in loop._written(case, started)], ["deep/er/report.json"])

print()
print("FAILURES:", FAILURES if FAILURES else "none")
sys.exit(1 if FAILURES else 0)
