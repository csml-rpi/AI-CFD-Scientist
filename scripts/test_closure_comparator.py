#!/usr/bin/env python3
"""The closure-challenge scoring comparator.

`starter_closure_challenge/reference_data/compare_velocity_mae.py` is the
script every candidate in that study is scored with, and the baseline is the
mean of its value over all 32 study cases. Because the framework refuses to
average over a subset — a mean of 31 cases is not comparable with a mean of 32
— a single unparseable case blocks the whole study at setup.

That happened: CBFS stores its high-fidelity field indirectly, and the
comparator returned "no field data", which would have refused the baseline and
stopped the run before a single candidate.

No OpenFOAM run and no LLM call: the shipped converged solutions are scored
directly.

    python3 scripts/test_closure_comparator.py
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
STARTER = REPO / "starter_closure_challenge"
COMPARATOR = STARTER / "reference_data" / "compare_velocity_mae.py"
CASES = STARTER / "cases"

FAILURES: list[str] = []


def check(name: str, cond: object, detail: str = "") -> None:
    if cond:
        print(f"[PASS] {name}")
    else:
        FAILURES.append(name)
        print(f"[FAIL] {name}" + (f" — {detail}" if detail else ""))


def score(case: Path) -> float | None:
    proc = subprocess.run(
        [sys.executable, str(COMPARATOR), "--case", str(case)],
        capture_output=True, text=True, timeout=180,
    )
    match = re.search(r"METRIC velocity_mae: ([0-9.eE+-]+)", proc.stdout)
    return float(match.group(1)) if match else None


def test_every_study_case_is_scoreable() -> None:
    """One unscoreable case blocks the entire study, so all of them must work."""
    cases = sorted(p.parent.parent for p in CASES.rglob("system/controlDict"))
    unscoreable = [c.name for c in cases if score(c) is None]
    check(
        f"all {len(cases)} study cases produce a score",
        not unscoreable,
        detail=f"unscoreable: {unscoreable}",
    )


def test_indirect_fields_are_followed() -> None:
    """CBFS defers its values to an #include'd file and names them with $VAR.

    Regression: the comparator read the field file literally, found no numbers,
    and returned "no field data" — which under the no-subset-mean rule would
    have refused the study's baseline outright.
    """
    cbfs = CASES / "training" / "CBFS"
    if not cbfs.is_dir():
        check("CBFS indirect-field test", True, detail="skipped: case absent")
        return
    truth = cbfs / "0" / "U_LES"
    check(
        "the fixture really is indirect (otherwise this test proves nothing)",
        truth.is_file() and "$" in truth.read_text(errors="replace"),
    )
    value = score(cbfs)
    check("CBFS scores despite the indirection", value is not None, detail="returned None")
    if value is not None:
        check("and the value is physically plausible", 0.0 < value < 1.0, detail=str(value))


def test_scores_are_in_a_sane_range() -> None:
    """A comparator returning nonsense is worse than one that fails: the study
    would optimise against it for hours."""
    cases = sorted(p.parent.parent for p in CASES.rglob("system/controlDict"))
    values = [v for v in (score(c) for c in cases) if v is not None]
    if not values:
        check("baseline range", False, detail="nothing scored")
        return
    mean = sum(values) / len(values)
    check(
        "the k-omega SST baseline mean is in the expected band",
        0.05 < mean < 0.20,
        detail=f"mean={mean:.5f}",
    )
    check("no case scores zero or negative", min(values) > 0, detail=f"min={min(values)}")
    check("no case is wildly off-scale", max(values) < 1.0, detail=f"max={max(values)}")


def main() -> int:
    if not COMPARATOR.is_file():
        print(f"skipped: {COMPARATOR} not present")
        return 0
    for test in (
        test_every_study_case_is_scoreable,
        test_indirect_fields_are_followed,
        test_scores_are_in_a_sane_range,
    ):
        try:
            test()
        except Exception as exc:  # noqa: BLE001 — a raising test is a failing test
            check(test.__name__, False, detail=f"{type(exc).__name__}: {exc}")
    if FAILURES:
        print(f"\n{len(FAILURES)} FAILURE(S): " + ", ".join(FAILURES))
        return 1
    print("\nALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
