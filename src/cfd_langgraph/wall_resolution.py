"""Near-wall spacing, measured per boundary, for a mesh-adequacy judgement.

Two separate questions decide whether a mesh is good enough:

  * has the answer stopped changing under refinement?
  * is the spacing near each boundary right for what is being asked of it?

This module supplies the measurements for the second and leaves the verdict to
the judgement. There is no table of acceptable y+ here: what counts as adequate
depends on the wall treatment, on which boundaries the quantities of interest
depend, and on the flow, which is engineering judgement rather than a threshold.

y+ is computed from the solved fields, not read from a function object, so a
case that was not configured to report it can still be judged:

    u_tau = sqrt((nu + nut) * |U_first_cell| / y)      no-slip wall
    y+    = u_tau * y / nu

y is the distance from each boundary face to its adjacent cell centre. Checked
against OpenFOAM's own yPlus output on several meshes, this agrees to within a
few percent; on a very coarse boundary it can read high, because the velocity
gradient estimate and the face-to-cell adjacency both degrade there.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class WallSpacing:
    """What was measured on one boundary."""

    patch: str
    treatment: str          # the nut boundary condition, verbatim from the case
    yplus_min: float
    yplus_mean: float
    yplus_max: float
    faces: int
    wall_distance_mean: float
    time: float


@dataclass
class Measurements:
    case: str
    nu: Optional[float]
    walls: Dict[str, WallSpacing] = field(default_factory=dict)
    patch_types: Dict[str, str] = field(default_factory=dict)
    note: str = ""

    def summary(self) -> str:
        if not self.walls:
            return self.note or "no boundaries measured"
        return "; ".join(f"{w.patch}: y+ mean {w.yplus_mean:.3g} "
                         f"(min {w.yplus_min:.3g}, max {w.yplus_max:.3g}), {w.treatment}"
                         for w in self.walls.values())


def _read_nu(case: Path) -> Optional[float]:
    """Molecular viscosity, from whichever properties dictionary holds it."""
    for name in ("physicalProperties", "transportProperties", "momentumTransport"):
        f = case / "constant" / name
        if not f.is_file():
            continue
        for line in f.read_text(errors="ignore").splitlines():
            s = line.split("//")[0].strip()
            if s.startswith("nu") and s.endswith(";"):
                for token in reversed(s.rstrip(";").split()):
                    try:
                        return float(token)
                    except ValueError:
                        continue
    return None


def _patch_types(case: Path) -> Dict[str, str]:
    f = case / "constant" / "polyMesh" / "boundary"
    if not f.is_file():
        return {}
    out, name = {}, None
    for line in f.read_text(errors="ignore").splitlines():
        s = line.strip()
        if s and not s.startswith(("{", "}", "//", "(", ")")) and " " not in s and not s.endswith(";"):
            name = s
        elif s.startswith("type") and name:
            out[name] = s.split()[-1].rstrip(";")
            name = None
    return out


def _nut_conditions(case: Path) -> Dict[str, str]:
    """The nut boundary condition per patch, verbatim. Says what the wall
    treatment IS; it does not say what spacing that treatment wants."""
    times = [d for d in case.iterdir() if d.is_dir() and d.name.replace(".", "").isdigit()]
    times.sort(key=lambda d: float(d.name), reverse=True)
    for d in times + [case / "0"]:
        f = d / "nut"
        if not f.is_file():
            continue
        text = f.read_text(errors="ignore")
        i = text.find("boundaryField")
        if i < 0:
            continue
        out, name = {}, None
        for line in text[i:].splitlines():
            s = line.strip()
            if s and not s.startswith(("{", "}", "//")) and " " not in s and not s.endswith(";"):
                name = s
            elif s.startswith("type") and name:
                out[name] = s.split()[-1].rstrip(";")
                name = None
        if out:
            return out
    return {}


def measure(case: Path | str, time: Optional[float] = None) -> Measurements:
    """Measure y+ on every wall boundary, from the solved fields."""
    import numpy as np
    from scipy.spatial import cKDTree

    from cfd_langgraph.foam_load import load_case, load_patches

    case = Path(case)
    nu = _read_nu(case)
    types = _patch_types(case)
    result = Measurements(case=str(case), nu=nu, patch_types=types)
    if nu is None or nu <= 0:
        result.note = "molecular viscosity not found, so y+ cannot be formed"
        return result

    mesh = load_case(case, time=time)
    centres = mesh.cell_centers()
    points = np.asarray(centres.points)
    U = centres.point_data.get("U", centres.cell_data.get("U"))
    if U is None:
        result.note = "no velocity field, so y+ cannot be formed"
        return result
    U = np.asarray(U)
    nut = centres.point_data.get("nut", centres.cell_data.get("nut"))
    nut = np.abs(np.asarray(nut)) if nut is not None else np.zeros(len(points))
    tree = cKDTree(points)

    conditions = _nut_conditions(case)
    for name, patch in load_patches(case, time=time).items():
        if types.get(name) != "wall":
            continue
        faces = np.asarray(patch.cell_centers().points)
        if len(faces) == 0:
            continue
        distance, index = tree.query(faces, k=1)
        distance = np.maximum(distance, 1e-30)
        speed = np.linalg.norm(U[index], axis=1)
        tau = (nu + nut[index]) * speed / distance
        u_tau = np.sqrt(np.maximum(tau, 0.0))
        yplus = u_tau * distance / nu
        result.walls[name] = WallSpacing(
            patch=name,
            treatment=conditions.get(name, "not specified"),
            yplus_min=float(yplus.min()), yplus_mean=float(yplus.mean()),
            yplus_max=float(yplus.max()), faces=int(len(yplus)),
            wall_distance_mean=float(distance.mean()),
            time=float(time) if time is not None else float("nan"),
        )
    if not result.walls:
        result.note = "no wall boundaries found"
    return result


JUDGEMENT_SYSTEM = """You are a senior CFD engineer judging whether a mesh is adequate near its \
boundaries for a specific set of quantities of interest.

Two things are yours to decide, from the flow rather than from a threshold:

WHICH BOUNDARIES MATTER. Spacing does not have to be adequate everywhere, only where the \
quantities of interest depend on it. Patch names do not tell you this. Judge from the geometry, \
the boundary conditions, and how each quantity is computed. Some shapes this takes:
  - A quantity read ON a boundary is only as good as the spacing there: wall shear, skin \
    friction, wall heat flux, wall pressure.
  - A boundary can matter without the quantity being read there, when it sets the flow that \
    sets the quantity -- a wall fixing the blockage, or carrying part of a force balance that \
    holds the flow rate.
  - A boundary can be present and matter little, when it is far from what sets the quantity.
  - A quantity taken along an interior line or plane may depend on no boundary spacing at all, \
    and instead on resolution where its own gradients are.
  - Inlets and outlets matter for quantities defined as differences across them.

WHAT SPACING THAT ROLE NEEDS. The wall treatment in use is given to you verbatim as each \
boundary's turbulent-viscosity condition. Reason from what that treatment assumes about the \
first cell, from the model it belongs to, and from the flow -- a strong pressure gradient or a \
separating boundary layer is not a place where a law of the wall is dependable, whatever the \
spacing. Say what you would require here and why.

You are given, per boundary: its patch type, its turbulent-viscosity condition, and the \
measured y+ (minimum, mean, maximum) with the mean wall distance. The y+ values are computed \
from the solved fields and are accurate to a few percent, except on a very coarse boundary \
where they can read high.

Conclude whether the mesh is adequate for these quantities, and if it is not, which boundary's \
spacing to change and in which direction."""


def judgement_payload(measurements: Measurements, *, geometry: str, metrics: str) -> str:
    lines = [f"GEOMETRY AND FLOW:\n{geometry.strip()[:3000]}\n",
             f"QUANTITIES OF INTEREST AND HOW THEY ARE COMPUTED:\n{metrics.strip()[:4000]}\n",
             f"MOLECULAR VISCOSITY: {measurements.nu}\n",
             "BOUNDARIES:"]
    for name, kind in measurements.patch_types.items():
        w = measurements.walls.get(name)
        if w is None:
            lines.append(f"  {name}: patch type {kind}; no near-wall spacing measured")
            continue
        lines.append(
            f"  {name}: patch type {kind}; turbulent-viscosity condition {w.treatment}; "
            f"y+ min {w.yplus_min:.4g}, mean {w.yplus_mean:.4g}, max {w.yplus_max:.4g} "
            f"over {w.faces} faces; mean wall distance {w.wall_distance_mean:.4g}")
    if measurements.note:
        lines.append(f"\nNOTE: {measurements.note}")
    return "\n".join(lines)


def judge(llm: object, measurements: Measurements, *, geometry: str, metrics: str) -> Dict[str, object]:
    """Ask for the verdict. Returns {} on failure -- no judgement is not approval."""
    from langchain_core.messages import HumanMessage, SystemMessage
    from pydantic import BaseModel, Field

    from cfd_langgraph.utils import structured_output

    class _Boundary(BaseModel):
        patch: str
        quantities_depend_on_it: str = Field(description="directly, indirectly, or little")
        required_spacing: str = Field(description="What you would require here, and why")
        acceptable: bool = Field(description="Is the measured spacing acceptable for that role?")
        reason: str

    class _Verdict(BaseModel):
        boundaries: List[_Boundary]
        adequate: bool = Field(
            description="False only if a boundary the quantities depend on has unacceptable spacing")
        reason: str
        what_to_change: str = Field(default="")

    try:
        out = structured_output(llm, _Verdict).invoke([
            SystemMessage(content=JUDGEMENT_SYSTEM),
            HumanMessage(content=judgement_payload(measurements, geometry=geometry, metrics=metrics)),
        ])
        return {"adequate": bool(out.adequate), "reason": str(out.reason),
                "what_to_change": str(out.what_to_change),
                "boundaries": [b.model_dump() for b in out.boundaries]}
    except Exception:  # noqa: BLE001
        return {}
