"""Frozen validation checks for turbinesFoam (see scripts/shelf_validate.py).

A static 2-D actuator line (one foil element at a set angle of attack in a
uniform stream, the repository's actuatorLine/static tutorial run with
`Allrun 2D 10`). The actuator line samples the flow, finds the angle of attack
it sees, looks its lift and drag up in the foil table, and feeds the force back
into the flow. The reference is the input, not the solver: the set angle
(10 degrees) and the foil table the case supplies (C_l = 0.80, C_d = 0.07 at
10 degrees, system/fvOptions.template).

Tolerances: the detected angle within 0.3 deg -- about the repository's own
test of the same quantity (mean-square error under 0.1 deg^2, tests/test_al.py)
-- and the lift within 4 % (= 0.3 deg of the table's slope, 0.08 per degree);
the drag within 6 %, which covers the Reynolds-number correction the code
applies to the table drag (Re ~1e5 against the table's 9e4).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict


def measure(case: Path, ctx: Dict[str, Any]) -> Dict[str, Any]:
    files = sorted((case / "postProcessing" / "actuatorLines").rglob("*.csv"))
    if not files:
        raise RuntimeError("the actuator line wrote no postProcessing/actuatorLines/*.csv")
    with files[0].open(newline="") as f:
        rows = [r for r in csv.DictReader(f)]
    last = rows[-1]
    return {
        "alpha_deg": float(last["alpha_deg"]),
        "cl": float(last["cl"]),
        "cd": float(last["cd"]),
        "iterations": float(last["time"]),
    }


SOURCE = ("input of the case: set angle 10 deg, foil table C_l(10) = 0.80, C_d(10) = 0.07 "
          "(tutorials/actuatorLine/static/system/fvOptions.template)")

CHECKS = [
    {
        "name": "static 2-D actuator line at 10 degrees",
        "tutorial": "tutorials/actuatorLine/static",
        "translate": False,          # written for openfoam.com, which this build targets
        "run": "bash ./Allrun 2D 10 > log.Allrun 2>&1 && tail -3 log.simpleFoam | grep -q End",
        "timeout": 3600,
        "measure": measure,
        "source": SOURCE,
        "expect": [
            {"quantity": "alpha_deg", "reference": 10.0, "abs_tol": 0.3, "source": "set angle of attack"},
            {"quantity": "cl", "reference": 0.80, "rel_tol": 0.04, "source": SOURCE},
            {"quantity": "cd", "reference": 0.07, "rel_tol": 0.06, "source": SOURCE},
        ],
    },
]
