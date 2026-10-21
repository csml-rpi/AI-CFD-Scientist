"""Frozen validation checks for solids4foam (see scripts/shelf_validate.py).

1. Plate with a circular hole under uniaxial far-field tension (the
   linearElasticity/plateHole tutorial, segregated approach). Kirsch's
   solution is exact for the infinite plate, and the tutorial applies its
   traction on the outer edges, so the quarter plate reproduces it up to the
   discretisation. Two references:
   - by hand: the displacement of the top of the hole (r = a, theta = 90 deg),
     where the Kirsch displacement reduces to u_y = -T a (kappa + 1) / (8 mu),
     plane strain kappa = 3 - 4 nu, mu = E / (2 (1 + nu)); with T = 1 MPa,
     a = 0.5 m, E = 200 GPa, nu = 0.3: u_y = -2.2750e-6 m. Tolerance 1e-7 m,
     the bound the repository's own regression test puts on point
     displacement error (regressionTest.sh, POINT_DISP_TOL).
   - the repository's regression test: the largest cell and point
     displacement errors and the largest xx-stress error against the
     analytical field, as the solver's analytical function object logs them,
     under the same script's bounds (1e-7 m, 1e-7 m, 2e5 Pa).
2. Dam break against a flexible wall (fluidSolidInteraction/flexibleDamBreak),
   the coupled water/air and large-strain solid run the repository's own
   regression test makes: run to 0.3 s and take the largest horizontal
   deflection of the dam's top-left corner. Reference: Meduri et al. (2017),
   0.04907 m at 0.235 s (referenceSolution/refSol.tsv), within the
   regression test's 2.5e-3 m. (Ryzhakov et al. (2010), in the same file,
   peak at 0.04103 m: the published solutions themselves differ by 16 %.)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

E, NU, T, A = 200e9, 0.3, 1e6, 0.5
MU = E / (2 * (1 + NU))
KAPPA = 3 - 4 * NU
HOLE_TOP_UY = -T * A * (KAPPA + 1) / (8 * MU)


def _point_rows(path: Path) -> List[List[float]]:
    rows = []
    for line in path.read_text().splitlines():
        parts = line.replace("(", " ").replace(")", " ").split()
        try:
            rows.append([float(v) for v in parts])
        except ValueError:
            continue
    if not rows:
        raise RuntimeError(f"no numbers in {path}")
    return rows


def _point_file(case: Path, name: str) -> Path:
    files = sorted(case.glob(f"postProcessing/**/solidPointDisplacement_{name}.dat"))
    if not files:
        raise RuntimeError(f"no postProcessing/**/solidPointDisplacement_{name}.dat")
    return files[-1]


def _norm_after(lines: List[str], start: int, component: str = "") -> float:
    """The L-infinity norm printed after a 'Writing <field> field' line: the
    third number on the line below 'Norms:' (below 'Component: <n>' first,
    for a tensor), as regressionTest.sh reads it."""
    i = start + 1
    if component:
        while i < len(lines) and lines[i].split()[:2] != ["Component:", component]:
            i += 1
    while i < len(lines) and not lines[i].strip().startswith("Norms:"):
        i += 1
    if i + 1 >= len(lines):
        raise RuntimeError(f"no norms after line {start + 1}")
    return float(lines[i + 1].split()[2])


def _set_root(case: Path, ctx: Dict[str, Any]) -> None:
    """Point the tutorial's local library at this build's sources, as the
    repository's regression test does for a copied case."""
    options = case / "src" / "Make" / "options"
    lines = options.read_text().splitlines()
    lines = [f"SOLIDS4FOAM_ROOT := {ctx['src']}" if line.startswith("SOLIDS4FOAM_ROOT :=") else line
             for line in lines]
    options.write_text("\n".join(lines) + "\n")


def plate_measure(case: Path, ctx: Dict[str, Any]) -> Dict[str, Any]:
    lines = (case / "log.solids4Foam").read_text(errors="replace").splitlines()
    last = {}
    for i, line in enumerate(lines):
        for field in ("DDifference", "pointDDifference", "cellStressDifference"):
            if line.strip() == f"Writing {field} field":
                last[field] = i
    missing = [f for f in ("DDifference", "pointDDifference", "cellStressDifference") if f not in last]
    if missing:
        raise RuntimeError(f"log.solids4Foam has no 'Writing <field> field' for {missing}")
    rows = _point_rows(_point_file(case, "pointDisp"))
    return {
        "hole_top_uy": rows[-1][2],
        "D_error_linf": _norm_after(lines, last["DDifference"]),
        "pointD_error_linf": _norm_after(lines, last["pointDDifference"]),
        "stress_xx_error_linf": _norm_after(lines, last["cellStressDifference"], "0"),
    }


def dam_prepare(case: Path, ctx: Dict[str, Any]) -> None:
    out = ctx["run"]("foamDictionary -entry endTime -set 0.3 system/controlDict", case, 120)
    if out["rc"] != 0:
        raise RuntimeError(f"could not set endTime: {out['stderr'][-400:]}")


def dam_measure(case: Path, ctx: Dict[str, Any]) -> Dict[str, Any]:
    rows = _point_rows(_point_file(case, "displacement"))
    peak = max(rows, key=lambda r: r[1])
    return {"peak_deflection": peak[1], "peak_time": peak[0], "end_time": rows[-1][0]}


KIRSCH = ("Kirsch's solution by hand: u_y = -T a (kappa+1)/(8 mu) at the top of the hole, "
          "T = 1 MPa, a = 0.5 m, E = 200 GPa, nu = 0.3, plane strain")
REGRESSION = "tutorials/solids/linearElasticity/plateHole/regressionTest.sh bounds"
MEDURI = ("Meduri et al. 2017, peak deflection of the dam's top-left corner, "
          "tutorials/fluidSolidInteraction/flexibleDamBreak/referenceSolution/refSol.tsv")

CHECKS = [
    {
        "name": "plate with a hole, Kirsch solution",
        "tutorial": "tutorials/solids/linearElasticity/plateHole",
        "translate": False,          # written for openfoam.com, which this build targets
        "prepare": _set_root,
        "run": "bash ./Allrun segregated > log.Allrun 2>&1 && tail -5 log.solids4Foam | grep -q End",
        "timeout": 3600,
        "measure": plate_measure,
        "source": KIRSCH,
        "expect": [
            {"quantity": "hole_top_uy", "reference": HOLE_TOP_UY, "abs_tol": 1e-7, "source": KIRSCH},
            {"quantity": "D_error_linf", "reference": 0.0, "abs_tol": 1e-7, "source": REGRESSION},
            {"quantity": "pointD_error_linf", "reference": 0.0, "abs_tol": 1e-7, "source": REGRESSION},
            {"quantity": "stress_xx_error_linf", "reference": 0.0, "abs_tol": 2e5, "source": REGRESSION},
        ],
    },
    {
        "name": "dam break against a flexible wall",
        "tutorial": "tutorials/fluidSolidInteraction/flexibleDamBreak",
        "translate": False,
        "prepare": dam_prepare,
        "run": "bash ./Allrun > log.Allrun 2>&1 && tail -5 log.solids4Foam | grep -q End",
        "timeout": 6 * 3600,
        "measure": dam_measure,
        "source": MEDURI,
        "expect": [
            {"quantity": "peak_deflection", "reference": 0.049073, "abs_tol": 2.5e-3, "source": MEDURI},
            {"quantity": "end_time", "reference": 0.3, "abs_tol": 0.005, "source": "the run reached 0.3 s"},
        ],
    },
]
