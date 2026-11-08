#!/usr/bin/env python3
"""A candidate's solver cost is its real solver launches, not solver names in
its commands (cylinder_dev_r3 stopped at "502 of 500 runs" with at least 116
solver logs on disk: every `tail log.pimpleFoam` counted, a fit script's own
solves did not).

1. Reading a log, grepping it or echoing a solver's name is no launch.
2. A launch counts, failed ones too, from the shell or from a Python script.

Run: python scripts/test_solver_launch_count.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import code_mod_agentic as m  # noqa: E402

FAILURES = []


def check(label, got, want):
    print(f"  {'PASS' if got == want else 'FAIL'}  {label}" + ("" if got == want else f": got {got}, want {want}"))
    if got != want:
        FAILURES.append(label)


with tempfile.TemporaryDirectory(dir=ROOT / ".cache") as tmp:
    run, starter = Path(tmp) / "run", Path(tmp) / "starter"
    run.mkdir()
    starter.mkdir()
    (run / "log.pimpleFoam").write_text("Time = 1\n")
    sb = m.Sandbox(run_dir=run, starter_case=starter, wm_project_dir=Path("/mnt/sda1/openfoam10"))
    sb.run_bash("tail -1 log.pimpleFoam; grep -c Time log.pimpleFoam; echo pimpleFoam simpleFoam Allrun")
    check("reading and naming a solver is no launch", sb.solver_invocations, 0)
    sb.run_bash("pimpleFoam -help >/dev/null 2>&1")
    check("a launch counts", sb.solver_invocations, 1)
    sb.run_bash("python3 -c \"import subprocess; subprocess.run(['pimpleFoam', '-help'], capture_output=True)\"")
    check("so does one from a Python script", sb.solver_invocations, 2)
    sb.run_bash("pimpleFoam -case nowhere >/dev/null 2>&1; true")
    check("and a failed one", sb.solver_invocations, 3)
    again = m.Sandbox(run_dir=run, starter_case=starter, wm_project_dir=Path("/mnt/sda1/openfoam10"))
    check("a new session of the candidate counts its own launches", again.solver_invocations, 0)

print()
print("FAILURES:", FAILURES if FAILURES else "none")
sys.exit(1 if FAILURES else 0)
