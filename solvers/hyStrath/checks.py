"""Frozen validation checks for hyStrath (see scripts/shelf_validate.py).

Both of hyStrath's main solvers on the problem with an exact answer they
share: a closed box of nitrogen, chemistry off, started out of equilibrium --
translation and rotation at 10000 K, vibration at 300 K -- and left to relax.
Energy has nowhere to go, so the temperature every mode settles at is fixed by
energy conservation alone, whatever the relaxation model:

  5/2 T + e_v(T) [+ e_el(T)] = 5/2 T0 + e_v(Tv0) [+ e_el(Tv0)]

with e_v(T) = theta / (exp(theta/T) - 1) the harmonic (and quantum
simple-harmonic) vibrational energy over k, theta = 3371 K for N2 in both
solvers' data, and e_el the electronic energy of the levels hy2Foam's
thermoDEM lists for N2 (the DSMC species data list none). That gives about
7590 K, solved below.

1. dsmcFoam+ (DSMC): the heatBath-5species tutorial made pure N2 without
   reactions, 1e5 simulator particles in its specular-walled cell; the
   translational, rotational and vibrational temperatures averaged over the
   last fifth of the run.
2. hy2Foam (two-temperature CFD): the heatBath tutorial made pure N2 with
   chemistry off; the trans-rotational and vibro-electronic temperatures at
   the end.

The DSMC temperatures are read from the fields dsmcFoam+ writes (every
1e-6 s, each an average over the 100 steps since the last write), not from
the probes function object. Changed on 2026-10-09, with the user's approval,
after the first run: probes load OpenFOAM's libsampling and with it OpenFOAM's
liblagrangian, whose particle classes hyStrath's liblagrangian+ redefines
under the same names; the two free the same globals at exit, so the run that
had already written correct results (7583/7579/7577 K) ended in a double free
and exit code 134. Tolerance and reference are unchanged.

Tolerance: each temperature within 1 % of the equilibrium value. Energy
exchange in both is conservative, so a correct build lands within the
statistical scatter of the DSMC average (well under 0.5 % with 1e5
particles) and the time-integration error of the CFD one; a wrong energy
mode, a non-conservative exchange or a run that has not relaxed misses it.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List

T0, TV0 = 10000.0, 300.0
THETA_N2 = 3371.0
# hy2Foam's thermoDEM, N2 electronicList: (degeneracy, theta_el in K).
N2_ELECTRONIC = [(1, 0.0), (3, 7.223157e4), (6, 8.577863e4), (6, 8.605027e4), (3, 9.535119e4),
                 (1, 9.805636e4), (2, 9.968268e4), (2, 1.048976e5), (5, 1.116490e5), (1, 1.225836e5),
                 (6, 1.248857e5), (6, 1.282476e5), (10, 1.338061e5), (6, 1.404296e5), (6, 1.504959e5)]


def _e_vib(t: float) -> float:
    return THETA_N2 / math.expm1(THETA_N2 / t)


def _e_el(t: float) -> float:
    weights = [(g, th, g * math.exp(-th / t)) for g, th in N2_ELECTRONIC]
    return sum(w * th for _, th, w in weights) / sum(w for _, _, w in weights)


def equilibrium(electronic: bool) -> float:
    """The temperature that holds the box's energy with every mode at it."""
    def energy(t_tr: float, t_ve: float) -> float:
        return 2.5 * t_tr + _e_vib(t_ve) + (_e_el(t_ve) if electronic else 0.0)

    target = energy(T0, TV0)
    lo, hi = 1000.0, T0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if energy(mid, mid) < target else (lo, mid)
    return 0.5 * (lo + hi)


T_EQ_DSMC = equilibrium(electronic=False)
T_EQ_CFD = equilibrium(electronic=True)


def _run(ctx: Dict[str, Any], case: Path, cmd: str) -> None:
    out = ctx["run"](f"hystrath-run bash -c {cmd!r}", case, 600)
    if out["rc"] != 0:
        raise RuntimeError(f"{cmd} failed: {out['stderr'][-500:]}")


def _replace(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    if old not in text:
        raise RuntimeError(f"{path.name} has no {old!r} to change")
    path.write_text(text.replace(old, new))


HEADER = ("FoamFile\n{{\n    version     2.0;\n    format      ascii;\n    class       dictionary;\n"
          "    location    \"system\";\n    object      {name};\n}}\n\n")


def dsmc_prepare(case: Path, ctx: Dict[str, Any]) -> None:
    (case / "system" / "dsmcInitialiseDict").write_text(
        HEADER.format(name="dsmcInitialiseDict")
        + "configurations\n(\n    configuration\n    {\n        type    dsmcMeshFill;\n"
        "        numberDensities\n        {\n            N2    1.0e+22;\n            O2    0;\n"
        "            NO    0;\n            N     0;\n            O     0;\n        };\n"
        f"        translationalTemperature    {T0:g};\n        rotationalTemperature       {T0:g};\n"
        f"        vibrationalTemperature      {TV0:g};\n        electronicTemperature       0;\n"
        "        velocity        (0 0 0);\n    }\n);\n")
    (case / "system" / "chemReactDict").write_text(HEADER.format(name="chemReactDict") + "reactions\n(\n);\n")
    # Fields every 100 steps, the last 21 kept (the final fifth of the run and
    # one more); no probes, which load OpenFOAM's own lagrangian library.
    for entry, value in (("endTime", "1e-04"), ("deltaT", "1e-08"), ("writeInterval", "1e-06"),
                         ("purgeWrite", "21")):
        _run(ctx, case, f"foamDictionary system/controlDict -entry {entry} -set {value}")
    _run(ctx, case, "foamDictionary system/controlDict -entry functions -remove")


def cfd_prepare(case: Path, ctx: Dict[str, Any]) -> None:
    init = case / "0" / "include" / "initialConditions"
    for old, new in (("Ttr       30000;", f"Ttr       {T0:g};"), ("Tve       1000;", f"Tve       {TV0:g};"),
                     ("X_N2      0.5;", "X_N2      1.0;"), ("X_N       0.5;", "X_N       0.0;")):
        _replace(init, old, new)
    _replace(case / "constant" / "chemistryProperties", "chemistry           on;", "chemistry           off;")
    for entry, value in (("endTime", "1e-04"), ("writeInterval", "1e-04")):
        _run(ctx, case, f"foamDictionary system/controlDict -entry {entry} -set {value}")


def _probe(case: Path, field: str) -> List[List[float]]:
    files = sorted(case.glob(f"postProcessing/probes/*/{field}"))
    if not files:
        raise RuntimeError(f"no postProcessing/probes/*/{field}")
    rows = []
    for f in files:
        for line in f.read_text().splitlines():
            if line.strip() and not line.lstrip().startswith("#"):
                try:
                    rows.append([float(v) for v in line.split()[:2]])
                except ValueError:
                    continue
    return sorted(rows)


def _tail_mean(rows: List[List[float]], fraction: float) -> float:
    start = rows[-1][0] - fraction * (rows[-1][0] - rows[0][0])
    tail = [v for t, v in rows if t >= start]
    return sum(tail) / len(tail)


def _uniform_value(path: Path) -> float:
    """The single cell's value of a written one-cell field."""
    for line in path.read_text().splitlines():
        if line.startswith("internalField"):
            text = line.split(None, 1)[1].rstrip(";").strip()
            if text.startswith("uniform"):
                return float(text.split()[1])
            if "(" in text:
                return float(text[text.index("(") + 1:text.index(")")].split()[0])
    raise RuntimeError(f"no single internalField value in {path}")


def dsmc_measure(case: Path, ctx: Dict[str, Any]) -> Dict[str, Any]:
    times = []
    for child in case.iterdir():
        try:
            t = float(child.name)
        except ValueError:
            continue
        if t > 0 and child.is_dir():
            times.append((t, child))
    if not times:
        raise RuntimeError("dsmcFoam+ wrote no time folders")
    times.sort()
    end = times[-1][0]
    tail = [d for t, d in times if t >= 0.8 * end - 1e-12]
    out: Dict[str, Any] = {"end_time": end}
    for mode in ("Ttra", "Trot", "Tvib"):
        values = [_uniform_value(d / f"{mode}_mixture") for d in tail]
        out[mode] = sum(values) / len(values)
        out[f"{mode}_samples"] = len(values)
    return out


def cfd_measure(case: Path, ctx: Dict[str, Any]) -> Dict[str, Any]:
    tt, tv = _probe(case, "Tt"), _probe(case, "Tv")
    return {"Tt": tt[-1][1], "Tv": tv[-1][1], "end_time": tt[-1][0]}


DSMC_SOURCE = (f"energy conservation in a closed box: 5/2 T + e_vib(T) = 5/2 x {T0:g} + e_vib({TV0:g}), "
               f"N2 theta_v = {THETA_N2:g} K -> {T_EQ_DSMC:.1f} K")
CFD_SOURCE = (f"energy conservation in a closed box, with hy2Foam's N2 electronic levels: "
              f"-> {T_EQ_CFD:.1f} K")

CHECKS = [
    {
        "name": "dsmcFoam+ nitrogen heat bath relaxes to the energy-conserving temperature",
        "tutorial": "run/hyStrath/dsmcFoam+/heatBath-5species",
        "translate": False,          # runs on hyStrath's own OpenFOAM v1706
        "prepare": dsmc_prepare,
        "run": ("hystrath-run bash -c 'blockMesh > log.blockMesh 2>&1 && dsmcInitialise+ > log.dsmcInitialise 2>&1 "
                "&& dsmcFoam+ > log.dsmcFoam 2>&1' && tail -5 log.dsmcFoam | grep -q End"),
        "timeout": 4 * 3600,
        "measure": dsmc_measure,
        "source": DSMC_SOURCE,
        "expect": [{"quantity": mode, "reference": T_EQ_DSMC, "rel_tol": 0.01, "source": DSMC_SOURCE}
                   for mode in ("Ttra", "Trot", "Tvib")]
                  + [{"quantity": "end_time", "reference": 1e-4, "rel_tol": 0.001, "source": "the run reached its end"}],
    },
    {
        "name": "hy2Foam nitrogen heat bath relaxes to the energy-conserving temperature",
        "tutorial": "run/hyStrath/hy2Foam/heatBath",
        "translate": False,
        "prepare": cfd_prepare,
        "run": ("hystrath-run bash -c 'blockMesh > log.blockMesh 2>&1 && hy2Foam > log.hy2Foam 2>&1' "
                "&& tail -5 log.hy2Foam | grep -q End"),
        "timeout": 4 * 3600,
        "measure": cfd_measure,
        "source": CFD_SOURCE,
        "expect": [{"quantity": mode, "reference": T_EQ_CFD, "rel_tol": 0.01, "source": CFD_SOURCE}
                   for mode in ("Tt", "Tv")]
                  + [{"quantity": "end_time", "reference": 1e-4, "rel_tol": 0.001, "source": "the run reached its end"}],
    },
]
