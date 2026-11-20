"""Frozen validation checks for olaFlow (see scripts/shelf_validate.py).

A regular stream-function wave generated in the 2-D flume of the
baseWaveFlume tutorial, measured by wave gauges along the flume and compared
with Fenton's Fourier (stream-function) solution for the same wave, which the
repository ships as reference/FourierFenton/Solution.txt -- computed by
Fenton's program, independent of olaFlow:

  depth d = 0.4 m, H/d = 0.375 (H = 0.150 m), period 3.000 s,
  wavelength L/d = 15.2982 (L = 6.119 m), wave speed c/sqrt(g d) = 1.02970
  (c = 2.0398 m/s; linear theory would give 1.92 m/s).

Tolerances: the period is imposed at the paddle, so +-1% only checks the
gauges and the sampling; the wave speed +-3% is half the gap between the
nonlinear and linear speed, so passing it means the wave propagates as the
nonlinear theory says; the height +-10% allows a few percent of numerical
damping at ~15 cells per wave height; the mean water level must stay within
5 mm of still water.

The flume is lengthened from the tutorial's 10 m to 40 m (same cells), so
that no wave reflected by the far end -- which reaches x = 8 m only after
about (80 - 8) / 2.1 = 34 s -- is in the gauges during the 15-30 s window:
the check measures the wave olaFlow generates and propagates, not its
absorbing boundary. Changed on 2026-10-09, with the user's approval, after
the first run in the 10 m flume failed on height (0.168 m): the gauge 2 m
from the absorbing end read 0.19 m and rising, the others 0.152-0.164 m.
The tolerances are unchanged.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List

DEPTH = 0.4
GAUGES_X = (2.0, 4.0, 6.0, 8.0)
WINDOW = (15.0, 30.0)          # five periods after the wave train is established
FLUME_LENGTH = 40.0            # m; the tutorial's is 10


def _set(case: Path, ctx: Dict[str, Any], key: str, value: str, file: str = "system/controlDict") -> None:
    out = ctx["run"](f"foamDictionary -entry {key} -set '{value}' {file}", case, 120)
    if out["rc"] != 0:
        raise RuntimeError(f"could not set {key} in {file}: {out['stderr'][-400:]}")


def prepare(case: Path, ctx: Dict[str, Any]) -> None:
    src: Path = ctx["src"]
    # The stream-function wave the reference solution describes.
    (case / "constant" / "waveDict").write_text(
        (src / "reference" / "waveDict" / "regular" / "streamFunction" / "waveDict").read_text())
    _set(case, ctx, "endTime", "30")
    _set(case, ctx, "writeInterval", "30")
    _set(case, ctx, "writeFormat", "binary")
    locations = " ".join(f"({x} -0.01 0.001)" for x in GAUGES_X)
    control = case / "system" / "controlDict"
    block = ("\nfunctions\n{\n    gauges\n    {\n        type            interfaceHeight;\n"
             "        libs            (\"libfieldFunctionObjects.so\");\n"
             "        writeControl    timeStep;\n        writeInterval   1;\n"
             "        alpha           alpha.water;\n"
             f"        locations       ({locations});\n    }}\n}}\n")
    control.write_text(control.read_text().rstrip() + "\n" + block)
    _set(case, ctx, "numberOfSubdomains", "8", "system/decomposeParDict")
    _set(case, ctx, "method", "scotch", "system/decomposeParDict")
    # The 40 m flume: same height, thickness and cell size, four times as long.
    x1 = FLUME_LENGTH
    _set(case, ctx, "vertices",
         f"((0 -0.02 0) ({x1} -0.02 0) ({x1} -0.02 0.7) (0 -0.02 0.7) "
         f"(0 0 0) ({x1} 0 0) ({x1} 0 0.7) (0 0 0.7))", "system/blockMeshDict")
    _set(case, ctx, "blocks",
         f"(hex (0 1 5 4 3 2 6 7) ({int(round(x1 / 0.02))} 1 70) simpleGrading (1 1 1))",
         "system/blockMeshDict")
    # The water column setFields fills must reach the new end, and the stored
    # initial water field belongs to the old mesh.
    fields = case / "system" / "setFieldsDict"
    text = fields.read_text()
    old_box = "box (-10 -1 -1) (30 1 0.4);"
    if old_box not in text:
        raise RuntimeError(f"setFieldsDict has no '{old_box}' to lengthen")
    fields.write_text(text.replace(old_box, f"box (-10 -1 -1) ({x1 + 10} 1 0.4);"))
    _set(case, ctx, "internalField", "uniform 0", "0.org/alpha.water")


RUN = ("blockMesh > log.blockMesh 2>&1 && rm -rf 0 && cp -r 0.org 0 && setFields > log.setFields 2>&1 "
       "&& decomposePar > log.decomposePar 2>&1 && mpirun -np 8 olaFlow -parallel > log.olaFlow 2>&1 "
       "&& tail -3 log.olaFlow | grep -q End")


def _read_heights(case: Path) -> List[List[float]]:
    files = sorted((case / "postProcessing" / "gauges").rglob("height.dat"))
    if not files:
        raise RuntimeError("the gauges wrote no height.dat")
    rows: List[List[float]] = []
    for f in files:
        for line in f.read_text().splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            try:
                rows.append([float(v) for v in line.split()])
            except ValueError:
                continue
    rows.sort(key=lambda r: r[0])
    return rows


def _resample(t: List[float], y: List[float], dt: float) -> tuple:
    import numpy as np

    grid = np.arange(t[0], t[-1], dt)
    return grid, np.interp(grid, t, y)


def measure(case: Path, ctx: Dict[str, Any]) -> Dict[str, Any]:
    import numpy as np

    rows = _read_heights(case)
    data = np.array([r for r in rows if WINDOW[0] <= r[0] <= WINDOW[1]])
    t = data[:, 0]
    # interfaceHeight writes, per location, the height above the boundary and
    # above the location: columns 1 + 2i are heights above the bottom.
    eta = {x: data[:, 1 + 2 * i] - DEPTH for i, x in enumerate(GAUGES_X)}
    tg, e4 = _resample(t.tolist(), eta[4.0].tolist(), 0.005)
    up = [tg[i] for i in range(1, len(e4)) if e4[i - 1] < 0 <= e4[i]]
    periods = np.diff(up)
    heights = []
    for x in (4.0, 6.0, 8.0):
        _, ex = _resample(t.tolist(), eta[x].tolist(), 0.005)
        for a, b in zip(up[:-1], up[1:]):
            seg = ex[(tg >= a) & (tg < b)]
            if len(seg):
                heights.append(float(seg.max() - seg.min()))
    _, e2 = _resample(t.tolist(), eta[2.0].tolist(), 0.005)
    _, e6 = _resample(t.tolist(), eta[6.0].tolist(), 0.005)
    # Lag of the gauge at x=6 behind the one at x=2, within one period.
    best, lag = -math.inf, float("nan")
    for k in range(1, int(3.0 / 0.005)):
        c = float(np.dot(e2[:-k], e6[k:]) / (len(e2) - k))
        if c > best:
            best, lag = c, k * 0.005
    return {
        "period": float(np.mean(periods)) if len(periods) else float("nan"),
        "wave_height": float(np.mean(heights)) if heights else float("nan"),
        "wave_speed": (GAUGES_X[2] - GAUGES_X[0]) / lag if lag == lag else float("nan"),
        "mean_level": float(np.mean([np.mean(v) for v in eta.values()])),
        "gauge_samples": int(len(t)),
    }


SOURCE = ("Fenton's Fourier/stream-function solution for H/d = 0.375, T sqrt(g/d) = 14.857, "
          "reference/FourierFenton/Solution.txt in the olaFlow repository (d = 0.4 m).")

CHECKS = [
    {
        "name": "stream-function wave in the base flume",
        "tutorial": "tutorials/baseWaveFlume",
        "prepare": prepare,
        "run": RUN,
        "timeout": 3 * 3600,
        "measure": measure,
        "source": SOURCE,
        "expect": [
            {"quantity": "period", "reference": 3.000, "rel_tol": 0.01, "source": "imposed period"},
            {"quantity": "wave_speed", "reference": 2.0398, "rel_tol": 0.03, "source": SOURCE},
            {"quantity": "wave_height", "reference": 0.150, "rel_tol": 0.10, "source": SOURCE},
            {"quantity": "mean_level", "reference": 0.0, "abs_tol": 0.005, "source": "still-water level"},
        ],
    },
]
