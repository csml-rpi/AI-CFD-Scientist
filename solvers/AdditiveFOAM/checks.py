"""Frozen validation checks for AdditiveFOAM (see scripts/shelf_validate.py).

A laser scanning a block, the AMB2018-02-B tutorial's beam (179.2 W,
absorptivity 0.33, 0.8 m/s, super-Gaussian of order 2 and size
85 x 85 x 30 um) and mesh, made into a problem with an exact answer:
properties constant (the tutorial's solid at 300 K), the melting range moved
out of reach so nothing melts and nothing flows, and every wall insulated.
What is left is heat conduction from a moving Gaussian source in a box, which
the method of images solves exactly: each axis convolves the Gaussian with
the heat kernel (variances add: s^2/2 + 2 alpha tau), summed over the
reflections in the insulated walls, integrated over the beam's path. That
solution is computed here, independently of the solver.

Two checks at t = 2 ms (beam at x = 1.6 mm):
- energy: rho cp sum((T - 300) V) = eta P t = 0.11827 J within 2 %. Nothing
  leaves the box, so every joule the beam deposits must be in it.
- temperature rise on the track behind the beam, at cell centres 0.29 and
  0.49 mm behind it 10 um under the surface and 0.29 mm behind it 50 um
  under, within 5 % of the exact rise. The mesh (20 um cells, 2 across the
  source's depth) and second-order discretisation put the expected error near
  1 %; 5 % leaves room for that and fails a wrong source, diffusion or
  normalisation.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

KAPPA = 8.275 + 0.01472 * 300.0        # the tutorial's solid conductivity at 300 K
CP = 579.28
RHO = 7569.92
ALPHA = KAPPA / (RHO * CP)
ETA, POWER, SPEED = 0.33, 179.2, 0.8
DIMS = (85e-6, 85e-6, 30e-6)
S = np.array(DIMS) / math.sqrt(2.0)    # superGaussian, k = 2: exp(-|d/s|^2), s = dims / 2^(1/k)
V0 = math.pi ** 1.5 / 2.0 * float(S.prod())   # its integral over the half space below the surface
BOX = ((-0.5e-3, 2.5e-3), (-0.25e-3, 0.25e-3), (-0.3e-3, 0.0))
T0, T_END = 300.0, 2e-3
POINTS = {
    "rise_0p29mm_behind_10um_deep": (1.31e-3, 0.0, -1e-5),
    "rise_0p49mm_behind_10um_deep": (1.11e-3, 0.0, -1e-5),
    "rise_0p29mm_behind_50um_deep": (1.31e-3, 0.0, -5e-5),
}


def _images(centre, lo: float, hi: float, n: int = 4) -> List[Any]:
    """A source centre and its reflections in insulated walls at lo and hi."""
    length = hi - lo
    return ([centre + 2 * k * length for k in range(-n, n + 1)]
            + [2 * lo - centre + 2 * k * length for k in range(-n, n + 1)])


def exact_temperature(point, t: float = T_END, steps: int = 40000) -> float:
    tp = (np.arange(steps) + 0.5) * t / steps
    tau = t - tp
    total = np.ones_like(tp)
    for axis in range(3):
        var = S[axis] ** 2 / 2.0 + 2.0 * ALPHA * tau
        lo, hi = BOX[axis]
        if axis == 0:
            centres = _images(SPEED * tp, lo, hi)
        elif axis == 1:
            centres = _images(0.0, lo, hi)
        else:
            # The source is centred on the surface wall z = 0: its half below
            # and that half's reflection make one full Gaussian at 0, whose
            # reflections in both walls sit at 2 k depth.
            centres = [2 * k * (hi - lo) for k in range(-4, 5)]
        kernel = sum(np.exp(-(point[axis] - c) ** 2 / (2 * var)) / np.sqrt(2 * math.pi * var)
                     for c in centres)
        total = total * S[axis] * math.sqrt(math.pi) * kernel
    return T0 + ETA * POWER / (RHO * CP * V0) * float(total.sum()) * t / steps


EXACT = {name: exact_temperature(p) for name, p in POINTS.items()}


def prepare(case: Path, ctx: Dict[str, Any]) -> None:
    run = ctx["run"]
    sets = [
        ("constant/transportProperties", phase + "/kappa", f"({KAPPA} 0 0)")
        for phase in ("solid", "liquid", "powder")
    ] + [
        ("constant/transportProperties", phase + "/Cp", f"({CP} 0 0)")
        for phase in ("solid", "liquid", "powder")
    ] + [
        ("0/T", "boundaryField/top/type", "zeroGradient"),
        ("system/controlDict", "endTime", str(T_END)),
        ("system/controlDict", "writeInterval", str(T_END)),
        ("system/controlDict", "writeFormat", "ascii"),
        ("system/controlDict", "writePrecision", "12"),
    ]
    for file, entry, value in sets:
        out = run(f"foamDictionary {file} -entry {entry} -set '{value}'", case, 120)
        if out["rc"] != 0:
            raise RuntimeError(f"could not set {entry} in {file}: {out['stderr'][-400:]}")
    # Solid up to 100000 K: nothing melts, so no latent heat and no flow.
    (case / "constant" / "thermoPath").write_text("(\n100000.0    1.0\n110000.0    0.0\n)\n")


RUN = ("blockMesh > log.blockMesh 2>&1 && decomposePar > log.decomposePar 2>&1 "
       "&& mpirun -np 6 additiveFoam -parallel > log.additiveFoam 2>&1 "
       "&& reconstructPar -latestTime > log.reconstructPar 2>&1 "
       "&& postProcess -func writeCellCentres -latestTime > log.writeCellCentres 2>&1 "
       "&& postProcess -func writeCellVolumes -latestTime > log.writeCellVolumes 2>&1 "
       f"&& test -s {T_END:g}/T")


def _internal_field(path: Path, size: int = 0) -> np.ndarray:
    """The internalField of an ascii field file, scalar or vector."""
    lines = path.read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("internalField"))
    head = lines[start].split()
    if head[1] == "uniform":
        value = " ".join(head[2:]).rstrip(";").strip("()").split()
        return np.tile(np.array([float(v) for v in value]), (size, 1)).squeeze()
    i = start + 1
    while not lines[i].strip().isdigit():
        i += 1
    count = int(lines[i].strip())
    while lines[i].strip() != "(":
        i += 1
    rows = [line.strip().strip("()").split() for line in lines[i + 1:i + 1 + count]]
    return np.array([[float(v) for v in r] for r in rows]).squeeze()


def measure(case: Path, ctx: Dict[str, Any]) -> Dict[str, Any]:
    folder = case / f"{T_END:g}"
    temperature = _internal_field(folder / "T")
    centres = _internal_field(folder / "C", len(temperature))
    volumes = _internal_field(folder / "V", len(temperature))
    values: Dict[str, Any] = {"energy_J": float(RHO * CP * np.sum((temperature - T0) * volumes))}
    for name, point in POINTS.items():
        distance = np.linalg.norm(centres - np.array(point), axis=1)
        cell = int(np.argmin(distance))
        if distance[cell] > 1e-9:
            raise RuntimeError(f"no cell centre at {point} (nearest {distance[cell]:.2e} m away)")
        values[name] = float(temperature[cell] - T0)
    return values


SOURCE = ("exact solution of heat conduction from the moving Gaussian beam in the insulated box "
          "(method of images, computed in checks.py); energy: eta P t")

CHECKS = [
    {
        "name": "moving beam in an insulated block, exact conduction solution",
        "tutorial": "tutorials/AMB2018-02-B",
        "translate": False,          # written for OpenFOAM 10, which this build targets
        "prepare": prepare,
        "run": RUN,
        "timeout": 3 * 3600,
        "measure": measure,
        "source": SOURCE,
        "expect": [
            {"quantity": "energy_J", "reference": ETA * POWER * T_END, "rel_tol": 0.02,
             "source": "energy deposited, eta P t; no heat leaves the insulated box"},
        ] + [
            {"quantity": name, "reference": EXACT[name] - T0, "rel_tol": 0.05, "source": SOURCE}
            for name in POINTS
        ],
    },
]
