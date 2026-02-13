#!/usr/bin/env python3
"""Post-process Foam-Agent OpenFOAM outputs deterministically.

Goal: produce requirement-compliant artifacts even when Foam-Agent's visualization is generic.

Artifacts (if possible):
- visualization.png: velocity magnitude |U| (UMag) at the latest available time on z=0 slice.
- umag_t<...>.png: |U| at requested times (nearest available time).
- uy_centerline_t<...>.csv: Uy along centerline (x=0, z=0) at requested times.

This module is intentionally standalone and uses only runtime dependencies already present
in the OpenFOAM/Foam-Agent environment (pyvista, numpy).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

# Headless-friendly defaults for VTK/PyVista
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

try:
    import pyvista as pv
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "pyvista is required for postprocess.py. Ensure you run this with the same env used by Foam-Agent (e.g., openfoamAgent-v2)."
    ) from e


@dataclass
class PostprocessConfig:
    times: Sequence[float] = (0.05, 0.50, 1.00)
    slice_z: float = 0.0
    centerline_x: float = 0.0
    centerline_z: float = 0.0
    centerline_y0: float = 0.0
    centerline_y1: float = 0.2
    centerline_n: int = 201
    cmap: str = "viridis"
    image_size: Tuple[int, int] = (1024, 768)


_TIME_RE = re.compile(
    r"\bt\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*(?:s|sec|seconds)?\b",
    re.IGNORECASE,
)


def parse_times_from_requirement(user_requirement: str) -> List[float]:
    if not user_requirement:
        return []
    times = []
    for m in _TIME_RE.finditer(user_requirement):
        try:
            times.append(float(m.group(1)))
        except Exception:
            continue
    # common pattern: "at t=0.05 s, 0.50 s, and 1.00 s"; the regex above catches those.
    # de-dupe while preserving order
    out = []
    seen = set()
    for t in times:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out


def _nearest_time(requested: float, available: Sequence[float]) -> Optional[float]:
    if not available:
        return None
    return min(available, key=lambda x: abs(x - requested))


def _compute_umag(mesh) -> Tuple[object, str]:
    """Ensure a scalar field UMag exists on mesh and return (mesh, scalar_name)."""
    # Prefer point-data U if present.
    try:
        if hasattr(mesh, "point_data") and "U" in mesh.point_data:
            U = np.asarray(mesh.point_data["U"])
            mesh.point_data["UMag"] = np.linalg.norm(U, axis=1)
            return mesh, "UMag"
    except Exception:
        pass

    try:
        if hasattr(mesh, "cell_data") and "U" in mesh.cell_data:
            U = np.asarray(mesh.cell_data["U"])
            mesh.cell_data["UMag"] = np.linalg.norm(U, axis=1)
            return mesh, "UMag"
    except Exception:
        pass

    # Fall back to pressure if U is missing; still produce a plot.
    if hasattr(mesh, "point_data") and "p" in getattr(mesh, "point_data", {}):
        return mesh, "p"
    if hasattr(mesh, "cell_data") and "p" in getattr(mesh, "cell_data", {}):
        return mesh, "p"

    return mesh, None  # type: ignore[return-value]


def _read_mesh(foam_path: Path, time_value: Optional[float] = None):
    reader = pv.OpenFOAMReader(str(foam_path))
    try:
        tvals = [float(t) for t in getattr(reader, "time_values", [])]
    except Exception:
        tvals = []

    if time_value is None:
        # latest
        try:
            if tvals:
                reader.set_active_time_value(tvals[-1])
        except Exception:
            pass
    else:
        try:
            reader.set_active_time_value(float(time_value))
        except Exception:
            pass

    data = reader.read()
    mesh = data
    try:
        if hasattr(data, "combine"):
            mesh = data.combine()
    except Exception:
        try:
            mesh = data[0]
        except Exception:
            mesh = data

    return mesh, tvals


def _plot_slice_png(mesh, scalar_name: Optional[str], out_png: Path, *, cfg: PostprocessConfig):
    plotter = pv.Plotter(off_screen=True, window_size=cfg.image_size)
    plotter.set_background("white")

    to_plot = mesh
    # Slice to z=cfg.slice_z if possible
    try:
        to_plot = mesh.slice(normal=(0, 0, 1), origin=(0, 0, cfg.slice_z))
    except Exception:
        to_plot = mesh

    if scalar_name:
        plotter.add_mesh(to_plot, scalars=scalar_name, cmap=cfg.cmap, show_scalar_bar=True)
    else:
        plotter.add_mesh(to_plot, color="lightgray")

    plotter.view_xy()
    plotter.show(auto_close=False)
    plotter.screenshot(str(out_png))
    plotter.close()


def _export_centerline_csv(mesh, out_csv: Path, *, cfg: PostprocessConfig):
    line = pv.Line(
        pointa=(cfg.centerline_x, cfg.centerline_y0, cfg.centerline_z),
        pointb=(cfg.centerline_x, cfg.centerline_y1, cfg.centerline_z),
        resolution=max(2, int(cfg.centerline_n) - 1),
    )
    sampled = line.sample(mesh)

    U = None
    if hasattr(sampled, "point_data") and "U" in sampled.point_data:
        U = np.asarray(sampled.point_data["U"])
    elif hasattr(sampled, "cell_data") and "U" in sampled.cell_data:
        U = np.asarray(sampled.cell_data["U"])

    pts = np.asarray(sampled.points)
    y = pts[:, 1]

    if U is None:
        # still write y column so downstream doesn't crash
        arr = np.column_stack([y, np.full_like(y, np.nan)])
        header = "y,Uy"
    else:
        Uy = U[:, 1]
        arr = np.column_stack([y, Uy])
        header = "y,Uy"

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_csv, arr, delimiter=",", header=header, comments="")


def postprocess_case(output_dir: Path, *, user_requirement: str = "", cfg: Optional[PostprocessConfig] = None) -> dict:
    cfg = cfg or PostprocessConfig()
    output_dir = Path(output_dir).resolve()

    foam_path = output_dir / "output.foam"
    if not foam_path.exists():
        # Some cases use '<case>.foam' naming; pick any .foam
        foams = list(output_dir.glob("*.foam"))
        if foams:
            foam_path = foams[0]

    if not foam_path.exists():
        return {"success": False, "error": f"No .foam file found in {output_dir}"}

    # parse requested times; fall back to cfg.times
    req_times = parse_times_from_requirement(user_requirement)
    if not req_times:
        req_times = list(cfg.times)

    # Make sure PyVista is in off-screen mode
    try:
        pv.OFF_SCREEN = True
    except Exception:
        pass
    try:
        pv.start_xvfb()
    except Exception:
        pass

    # Latest visualization
    mesh_latest, tvals = _read_mesh(foam_path, time_value=None)
    mesh_latest, scalar = _compute_umag(mesh_latest)
    latest_png = output_dir / "visualization.png"
    _plot_slice_png(mesh_latest, scalar, latest_png, cfg=cfg)

    artifacts = {
        "visualization_png": str(latest_png),
        "umag_pngs": [],
        "uy_csvs": [],
        "requested_times": req_times,
        "available_times": tvals,
        "used_times": {},
    }

    # Per-requested-time outputs
    for t in req_times:
        tn = _nearest_time(float(t), tvals) if tvals else None
        if tn is None:
            continue

        mesh_t, _ = _read_mesh(foam_path, time_value=tn)
        mesh_t, scalar_t = _compute_umag(mesh_t)

        # sanitize time for filename
        t_tag = f"{float(t):.2f}".replace(".", "p")
        tn_tag = f"{float(tn):.6g}".replace(".", "p")
        artifacts["used_times"][str(t)] = float(tn)

        umag_png = output_dir / f"umag_req_t{t_tag}_used_t{tn_tag}.png"
        _plot_slice_png(mesh_t, scalar_t, umag_png, cfg=cfg)
        artifacts["umag_pngs"].append(str(umag_png))

        uy_csv = output_dir / f"uy_centerline_req_t{t_tag}_used_t{tn_tag}.csv"
        _export_centerline_csv(mesh_t, uy_csv, cfg=cfg)
        artifacts["uy_csvs"].append(str(uy_csv))

    return {"success": True, "artifacts": artifacts}


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Post-process an OpenFOAM case output directory")
    ap.add_argument("--output-dir", required=True, help="Case output directory (contains output.foam)")
    ap.add_argument("--requirement-path", default=None, help="Optional user_requirement.txt to parse requested times")
    args = ap.parse_args()

    req = ""
    if args.requirement_path:
        try:
            req = Path(args.requirement_path).read_text(encoding="utf-8")
        except Exception:
            req = ""

    res = postprocess_case(Path(args.output_dir), user_requirement=req)
    print(res)


if __name__ == "__main__":
    main()
