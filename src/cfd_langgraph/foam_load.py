"""Open an OpenFOAM case for plotting, through PyVista.

Handles the details that decide whether a figure appears:

  * the reader takes the ``.foam`` marker file, not the case directory;
  * field arrays are off by default, so ``mesh['U']`` raises until enabled;
  * cell data cannot be contoured, so cell-to-point conversion is needed;
  * ``read()`` returns a MultiBlock; the fields are in ``internalMesh``;
  * time 0 is the initial condition, so a case with no later time raises
    rather than returning its own initial guess;
  * rendering is off-screen, as there is no display.

    from foam_load import load_case, new_plotter, wall_shear
    mesh = load_case(case_dir)                 # latest solved time
    walls = wall_shear(case_dir)               # {patch: mesh with wallShearStress}
    p = new_plotter()
    p.add_mesh(mesh.slice(normal="z"), scalars="U")
    p.screenshot("field.png")
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence


class FoamLoadError(RuntimeError):
    """Carries the reason the case could not be opened."""


def _pyvista():
    import pyvista as pv

    pv.OFF_SCREEN = True   # must be set before any Plotter exists
    return pv


def marker_for(case: Path | str) -> Path:
    """The ``.foam`` file the reader needs, created if absent.

    Accepts the case directory or an existing marker.
    """
    case = Path(case).expanduser().resolve()
    if case.is_file() and case.suffix == ".foam":
        return case
    if not case.is_dir():
        raise FoamLoadError(f"not a case directory or .foam marker: {case}")
    marker = case / f"{case.name}.foam"
    if not marker.is_file():
        marker.touch()
    return marker


def solved_times(case: Path | str) -> List[float]:
    """Times the solver wrote. 0 is the initial condition and is excluded."""
    pv = _pyvista()
    marker = marker_for(case)
    reader = pv.OpenFOAMReader(str(marker))
    return [float(t) for t in (reader.time_values or []) if float(t) > 0.0]


def load_case(case: Path | str, time: Optional[float] = None):
    """The internal mesh at one time, fields available as point data.

    ``time`` defaults to the latest solved time.
    """
    pv = _pyvista()
    marker = marker_for(case)
    try:
        reader = pv.OpenFOAMReader(str(marker))
    except Exception as exc:  # noqa: BLE001
        raise FoamLoadError(f"could not open {marker}: {type(exc).__name__}: {exc}") from exc

    times = [float(t) for t in (reader.time_values or [])]
    if time is None:
        positive = [t for t in times if t > 0.0]
        if not positive:
            raise FoamLoadError(
                f"{marker.parent.name} has no solved time (times={times or 'none'}); "
                "the case did not advance past its initial condition"
            )
        time = max(positive)
    elif float(time) not in times:
        raise FoamLoadError(f"time {time} not among {times}")

    reader.set_active_time_value(float(time))
    try:
        reader.enable_all_cell_arrays()
        reader.enable_all_point_arrays()
    except Exception:  # noqa: BLE001 -- older readers expose fewer switches
        pass
    reader.cell_to_point_creation = True

    try:
        block = reader.read()
    except Exception as exc:  # noqa: BLE001
        raise FoamLoadError(f"read failed at t={time}: {type(exc).__name__}: {exc}") from exc

    mesh = block["internalMesh"] if (hasattr(block, "keys") and "internalMesh" in list(block.keys())) else block
    if mesh is None or getattr(mesh, "n_cells", 0) == 0:
        raise FoamLoadError(f"internalMesh came back empty at t={time}")
    return mesh


def sample_points(case: Path | str, points, time: Optional[float] = None) -> dict:
    """Every field's value at each of ``points`` (rows of x, y, z), interpolated
    inside the cell that holds the point -- the value AT the point, unlike
    ``mesh.interpolate``, which is a kernel average over a radius.

    Returns ``{field name: one value or vector per point}`` plus ``"inside"``,
    a boolean per point; a point outside the domain gets NaN. In a 2-D case the
    out-of-plane coordinate is set to the mid-plane, so any value given for it
    is ignored.
    """
    import numpy as np

    pv = _pyvista()
    mesh = load_case(case, time)
    pts = np.array(points, dtype=float).reshape(-1, 3)
    axis = empty_axis(case)
    if axis is not None:
        lo, hi = mesh.bounds[2 * axis], mesh.bounds[2 * axis + 1]
        pts[:, axis] = 0.5 * (lo + hi)
    sampled = pv.PolyData(pts).sample(mesh)
    inside = np.asarray(sampled.point_data.get("vtkValidPointMask", np.ones(len(pts)))).astype(bool)
    out = {"inside": inside}
    for name in mesh.point_data.keys():
        values = np.array(sampled.point_data[name], dtype=float)
        values[~inside] = np.nan
        out[name] = values
    return out


def sample_line(case: Path | str, start, end, n: int = 200, time: Optional[float] = None) -> dict:
    """:func:`sample_points` at ``n`` evenly spaced points from ``start`` to
    ``end``, plus ``"position"`` (the points) and ``"distance"`` (from start)."""
    import numpy as np

    a, b = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    s = np.linspace(0.0, 1.0, max(2, int(n)))
    pts = a + s[:, None] * (b - a)
    out = sample_points(case, pts, time)
    out["position"] = pts
    out["distance"] = s * float(np.linalg.norm(b - a))
    return out


def load_patches(case: Path | str, time: Optional[float] = None) -> dict:
    """Boundary patches by name, for quantities read on a boundary.

    Returns ``{patch_name: mesh}``. Patch arrays are off by default, so the
    reader is told to enable them before reading.
    """
    pv = _pyvista()
    marker = marker_for(case)
    reader = pv.OpenFOAMReader(str(marker))
    times = [float(t) for t in (reader.time_values or [])]
    if time is None:
        positive = [t for t in times if t > 0.0]
        if not positive:
            raise FoamLoadError(f"{marker.parent.name} has no solved time (times={times or 'none'})")
        time = max(positive)
    reader.set_active_time_value(float(time))
    for switch in ("enable_all_patch_arrays", "enable_all_cell_arrays", "enable_all_point_arrays"):
        try:
            getattr(reader, switch)()
        except Exception:  # noqa: BLE001
            pass
    reader.cell_to_point_creation = True
    block = reader.read()

    found: dict = {}

    def walk(node, label=""):
        if hasattr(node, "keys"):
            for key in list(node.keys()):
                walk(node[key], str(key))
        elif node is not None and getattr(node, "n_cells", 0) > 0 and label != "internalMesh":
            found[label] = node

    walk(block)
    return found


def empty_axis(case: Path | str) -> Optional[int]:
    """The out-of-plane axis of a 2-D case, or None if it is really 3-D.

    Taken from the ``empty`` patch rather than from the bounding box: a 2-D
    OpenFOAM case is one cell thick in that direction but the cell can be any
    thickness, so extents do not say which direction is the fake one.
    """
    import numpy as np

    case = Path(case)
    boundary = case / "constant" / "polyMesh" / "boundary"
    if not boundary.is_file():
        return None
    empties, name = [], None
    for line in boundary.read_text(errors="ignore").splitlines():
        t = line.strip()
        if t and not t.startswith(("{", "}", "//", "(", ")")) and " " not in t and not t.endswith(";"):
            name = t
        elif t.startswith("type") and name:
            if t.split()[-1].rstrip(";") == "empty":
                empties.append(name)
            name = None
    if not empties:
        return None
    for patch_name, patch in load_patches(case).items():
        if patch_name not in empties:
            continue
        try:
            normals = np.asarray(patch.extract_surface(algorithm='dataset_surface').face_normals)
        except Exception:  # noqa: BLE001
            continue
        if len(normals) == 0:
            continue
        return int(np.argmax(np.abs(np.mean(np.abs(normals), axis=0))))
    return None


def cut_plane(case: Path | str, time: Optional[float] = None):
    """The mesh reduced to the plane the flow actually lives in.

    For a 2-D case that is a slice through the one cell, so the geometry shows
    as it is rather than edge-on. For a 3-D case it is the mid-plane of the
    thinnest direction. Plot the result directly.
    """
    import numpy as np

    mesh = load_case(case, time=time)
    axis = empty_axis(case)
    if axis is None:
        bounds = np.asarray(mesh.bounds).reshape(3, 2)
        axis = int(np.argmin(bounds[:, 1] - bounds[:, 0]))
    normal = [0.0, 0.0, 0.0]
    normal[axis] = 1.0
    sliced = mesh.slice(normal=normal, origin=mesh.center)
    return sliced if getattr(sliced, "n_points", 0) > 0 else mesh


def new_plotter(window_size: Sequence[int] = (1200, 800), **kwargs):
    """An off-screen Plotter."""
    pv = _pyvista()
    kwargs.setdefault("off_screen", True)
    return pv.Plotter(window_size=list(window_size), **kwargs)


def patch_types(case: Path | str) -> dict:
    """``{patch name: type}`` from constant/polyMesh/boundary."""
    f = Path(case) / "constant" / "polyMesh" / "boundary"
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


def molecular_viscosity(case: Path | str) -> Optional[float]:
    """Kinematic viscosity nu, from whichever properties dictionary holds it."""
    for name in ("physicalProperties", "transportProperties", "momentumTransport"):
        f = Path(case) / "constant" / name
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


def wall_shear(case: Path | str, time: Optional[float] = None, velocity: str = "U") -> dict:
    """Kinematic wall shear stress on every wall patch, computed from the fields.

    Returns ``{patch_name: patch mesh}`` with cell data ``wallShearStress``
    (vector, OpenFOAM's sign convention: it points against the near-wall flow,
    so Cf = -2 * wallShearStress_x / U_ref**2 for flow along +x) and
    ``wallDistance``. Uses the field the case wrote when there is one;
    otherwise (nu + nut) times the tangential velocity of the adjacent cell
    over its wall-normal distance. Pass ``velocity="UMean"`` for the
    time-averaged stress when the case writes UMean.
    """
    import numpy as np

    nu = molecular_viscosity(case)
    types = patch_types(case)
    mesh = load_case(case, time=time)
    centres = np.asarray(mesh.cell_centers().points)
    U = mesh.cell_data.get(velocity)
    if U is None:
        raise FoamLoadError(f"{Path(case).name} has no {velocity} field")
    U = np.asarray(U)
    nut = mesh.cell_data.get("nut")
    nut = np.abs(np.asarray(nut)) if nut is not None else np.zeros(len(centres))
    out: dict = {}
    for name, patch in load_patches(case, time=time).items():
        if types.get(name.split("/")[-1]) != "wall":
            continue
        written = "wallShearStressMean" if velocity.endswith("Mean") else "wallShearStress"
        if written in patch.cell_data:
            patch.cell_data["wallShearStress"] = np.asarray(patch.cell_data[written])
            out[name] = patch
            continue
        if nu is None:
            raise FoamLoadError("molecular viscosity not found in constant/")
        faces = np.asarray(patch.cell_centers().points)
        normals = np.asarray(patch.compute_normals(cell_normals=True, point_normals=False,
                                                   auto_orient_normals=False).cell_data["Normals"])
        cell = np.asarray(mesh.find_closest_cell(faces))
        d = np.abs(np.einsum("ij,ij->i", centres[cell] - faces, normals))
        d = np.maximum(d, 1e-30)
        wall_u = np.asarray(patch.cell_data[velocity]) if velocity in patch.cell_data else 0.0
        rel = U[cell] - wall_u
        tangential = rel - np.einsum("ij,ij->i", rel, normals)[:, None] * normals
        tau = -((nu + nut[cell]) / d)[:, None] * tangential
        patch.cell_data["wallShearStress"] = tau
        patch.cell_data["wallDistance"] = d
        out[name] = patch
    return out


def install_beside(script_dir: Path | str) -> Optional[Path]:
    """Copy this module next to a generated script so it can import it.

    A generated script runs as a subprocess, and Python puts that script's own
    directory on sys.path -- not the caller's. Every generator that writes a
    script which loads an OpenFOAM case calls this first, so the script can
    ``import foam_load`` with no path setup and in any sandbox.

    Returns the installed path, or None if it could not be written.
    """
    source = Path(__file__).resolve()
    target = Path(script_dir) / source.name
    try:
        Path(script_dir).mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        return target
    except OSError:
        return None
