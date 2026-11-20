#!/usr/bin/env python3
"""The example scripts a shelf solver's case writer is shown.

Ranked by word overlap alone, the only field-input example DAFoam's usage
notes point to (tests/runRegTests_DASimpleFoamForward.py) came 7th of the top
4 for a field-inversion requirement, so the writer never saw it. A script the
notes name by path is now always shown. Uses the real shelf; skipped without it.

Run: python scripts/test_shelf_examples.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("CFD_SCIENTIST_SHELF", "/mnt/sda1/cfd_solver_shelf")

from cfd_langgraph.foam_native import rag  # noqa: E402

FAILURES: list[str] = []


def check(name: str, got, want=True) -> None:
    ok = got == want
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + ("" if ok else f" — got {got!r}, want {want!r}"))
    if not ok:
        FAILURES.append(name)


def picked(requirement: str) -> list[str]:
    out = rag.shelf_example_scripts(requirement, "DASimpleFoam")
    return [line.split('"')[1].rsplit("/", 1)[-1] for line in out.splitlines() if line.startswith("<example_script")]


if not Path(os.environ["CFD_SCIENTIST_SHELF"]).is_dir():
    print("[SKIP] no solver shelf on this machine")
    sys.exit(0)

short = picked("Field inversion of betaFINuTilda with a variance objective; IPOPT; gradient check.")
check("the example the notes name is shown for a short requirement", "runRegTests_DASimpleFoamForward.py" in short)
check("and the word-ranked examples still fill the rest", len(short), 4)
other = picked("Shape optimisation of a duct with FFD points to reduce pressure drop")
check("it is shown for any DAFoam requirement", "runRegTests_DASimpleFoamForward.py" in other)
check("a bare 'runScript.py' in the notes pins nothing", "runScript.py" in short + other, False)

print()
print("FAILURES:", FAILURES if FAILURES else "none")
sys.exit(1 if FAILURES else 0)
