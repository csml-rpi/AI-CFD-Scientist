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
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

# Headless-friendly defaults for VTK/PyVista
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

try:
    import pyvista as pv
    _PYVISTA_IMPORT_ERROR = None
except Exception as e:  # pragma: no cover
    pv = None  # type: ignore[assignment]
    _PYVISTA_IMPORT_ERROR = e


@dataclass
class PostprocessConfig:
    times: Sequence[float] = (0.10, 0.50, 1.00, 2.00, 3.00)
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

_FLOAT_RE = re.compile(r"[-+]?(?:\d+\.\d+|\d+)(?:[eE][-+]?\d+)?")


def parse_times_from_requirement(user_requirement: str) -> List[float]:
    """Extract requested visualization times from a natural-language requirement.

    Handles patterns like:
      - "at t=0.10 s, 0.50 s, 1.00 s"
      - "at t=0.10, 0.50, 1.00, 2.00, 3.00 s"

    Strategy:
      1) Find each "t=<number>" anchor.
      2) From the anchor, parse subsequent comma/and-separated numbers in a short window.

    Returns a de-duped list in discovery order.
    """
    if not user_requirement:
        return []

    found: List[float] = []

    for m in _TIME_RE.finditer(user_requirement):
        # Anchor time
        try:
            found.append(float(m.group(1)))
        except Exception:
            continue

        # Look ahead for additional times like ", 0.50 s, 1.00 s".
        tail = user_requirement[m.end() : m.end() + 120]
        # Stop early on obvious hard boundaries to avoid pulling unrelated numbers.
        # NOTE: do NOT split on '.' because that breaks decimal numbers like 0.50.
        for sep in ["\n", ";"]:
            if sep in tail:
                tail = tail.split(sep, 1)[0]
        for fm in _FLOAT_RE.finditer(tail):
            try:
                v = float(fm.group(0))
            except Exception:
                continue
            # Heuristic: plausible visualization times (seconds)
            if 0.0 <= v <= 1.0e3:
                found.append(v)

    # De-dupe preserving order
    out: List[float] = []
    seen = set()
    for t in found:
        if t in seen:
            continue
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


def _apply_bounds_based_geometry(cfg: PostprocessConfig, mesh) -> dict:
    """Update cfg in-place using mesh bounds and return a metadata dict."""
    meta = {}
    try:
        bounds = getattr(mesh, "bounds", None)
    except Exception:
        bounds = None

    if not bounds or len(bounds) != 6:
        return meta

    xmin, xmax, ymin, ymax, zmin, zmax = [float(x) for x in bounds]
    vals = [xmin, xmax, ymin, ymax, zmin, zmax]
    if not all(np.isfinite(v) for v in vals):
        return meta

    x_mid = 0.5 * (xmin + xmax)
    z_mid = 0.5 * (zmin + zmax)

    cfg.slice_z = z_mid
    cfg.centerline_x = x_mid
    cfg.centerline_z = z_mid
    cfg.centerline_y0 = ymin
    cfg.centerline_y1 = ymax

    meta = {
        "bounds": {"xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax, "zmin": zmin, "zmax": zmax},
        "x_mid": x_mid,
        "z_mid": z_mid,
    }
    return meta


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
    if pv is None:
        return {
            "success": False,
            "error": "pyvista is required for deterministic post-processing. Run in the Foam-Agent environment (where pyvista is installed).",
            "import_error": str(_PYVISTA_IMPORT_ERROR),
        }

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

    # Read latest mesh to (a) discover available times and (b) infer geometry from bounds.
    mesh_latest, tvals = _read_mesh(foam_path, time_value=None)
    mesh_latest, scalar = _compute_umag(mesh_latest)

    geom_meta = _apply_bounds_based_geometry(cfg, mesh_latest)

    # Keep a "latest" visualization for human debugging, but downstream evaluation should
    # prefer the time-stamped images.
    latest_png = output_dir / "visualization.png"
    _plot_slice_png(mesh_latest, scalar, latest_png, cfg=cfg)

    def _time_tag(tt: float) -> str:
        return f"{float(tt):.2f}".replace(".", "p")

    artifacts = {
        "requested_times": [float(x) for x in req_times],
        "available_times": [float(x) for x in tvals],
        "used_times": {},
        "time_errors": {},
        "geometry": {
            "slice_z": float(cfg.slice_z),
            "centerline": {
                "x": float(cfg.centerline_x),
                "y0": float(cfg.centerline_y0),
                "y1": float(cfg.centerline_y1),
                "z": float(cfg.centerline_z),
                "n": int(cfg.centerline_n),
            },
            "bounds_meta": geom_meta,
        },
        "files": {
            "latest_png": str(latest_png),
            "umag_pngs": [],
            "uy_csvs": [],
        },
    }

    # Per-requested-time outputs (snap to nearest written time and record what happened).
    # For writeInterval=0.1, a half-interval tolerance is ~0.05.
    tol = 0.051

    for t_req in [float(x) for x in req_times]:
        t_used = _nearest_time(t_req, tvals) if tvals else None
        if t_used is None:
            artifacts["used_times"][str(t_req)] = None
            artifacts["time_errors"][str(t_req)] = None
            continue

        artifacts["used_times"][str(t_req)] = float(t_used)
        artifacts["time_errors"][str(t_req)] = float(abs(float(t_used) - float(t_req)))

        mesh_t, _ = _read_mesh(foam_path, time_value=float(t_used))
        mesh_t, scalar_t = _compute_umag(mesh_t)

        t_tag = _time_tag(t_req)
        umag_png = output_dir / f"umag_t{t_tag}.png"
        _plot_slice_png(mesh_t, scalar_t, umag_png, cfg=cfg)
        artifacts["files"]["umag_pngs"].append({"t": float(t_req), "path": str(umag_png), "used_t": float(t_used)})

        uy_csv = output_dir / f"uy_centerline_t{t_tag}.csv"
        _export_centerline_csv(mesh_t, uy_csv, cfg=cfg)
        artifacts["files"]["uy_csvs"].append({"t": float(t_req), "path": str(uy_csv), "used_t": float(t_used)})

    artifacts_path = output_dir / "artifacts.json"
    try:
        artifacts_path.write_text(json.dumps(artifacts, indent=2), encoding="utf-8")
    except Exception:
        pass

    return {"success": True, "artifacts": artifacts, "artifacts_path": str(artifacts_path)}


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
