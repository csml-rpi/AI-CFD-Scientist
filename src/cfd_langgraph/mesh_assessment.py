"""Judging whether a mesh level's solution can be trusted, before and during refinement.

A mesh gate that only compares numbers between refinement levels can converge
on a wrong solution: two levels that both suppress the physics agree with each
other perfectly. So each level is judged on two things:

  * whether its quantities have stopped changing under refinement, and
  * whether the flow itself is physically sensible for the stated case --
    judged by looking at it, alongside the numbers.

Neither uses reference data. Matching a reference is the study's job, after
the mesh is settled; the gate only establishes that the solution is resolved
and plausible.

For a transient case, quantities are read over the statistically settled part
of the run -- a single snapshot of an oscillating flow is a random phase, and
comparing two random phases between meshes measures nothing.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# Fraction of each monitored history treated as settled: the final part of the
# run, after start-up transients have passed.
WINDOW_BLOCKS = 20          # a history is split into this many equal-time blocks
SETTLED_TOLERANCE = 0.10    # a block's fluctuation matches the end of the run within this fraction
MIN_WINDOW_FRACTION = 0.2   # the window never covers less than this share of the run

# Relative fluctuation below which a monitored signal is treated as constant.
NOISE_FLOOR = 1e-5

_TRANSIENT_DDT = ("euler", "backward", "crankNicolson", "localEuler", "CoEuler", "SLTS")


# --------------------------------------------------------------------- histories

@dataclass
class SeriesStats:
    source: str          # the function-object output it came from
    column: str
    mean: float
    fluctuation_rms: float
    dominant_frequency: Optional[float]   # None when there is no clear periodic content
    window: tuple                          # (t_start, t_end) of the settled window
    samples: int
    halves: tuple = ()                     # ((mean, rms) first half, (mean, rms) second half)


def _read_history(path: Path) -> tuple[List[str], List[List[float]]]:
    """A function-object .dat file: last '#' header line names the columns."""
    header: List[str] = []
    rows: List[List[float]] = []
    for line in path.read_text(errors="ignore").splitlines():
        if not line.strip():
            continue
        if line.lstrip().startswith("#"):
            cols = line.lstrip("#").split()
            if cols and cols[0].lower().startswith("time"):
                header = cols
            continue
        parts = line.replace("(", " ").replace(")", " ").split()
        try:
            rows.append([float(p) for p in parts])
        except ValueError:
            continue
    return header, rows


def _dominant_frequency(t, y) -> Optional[float]:
    """Frequency of the strongest spectral peak, if the peak stands clear of the rest."""
    import numpy as np

    if len(t) < 32:
        return None
    tu = np.linspace(t[0], t[-1], 4096)
    yu = np.interp(tu, t, y - np.mean(y))
    if not np.any(yu):
        return None
    spec = np.abs(np.fft.rfft(yu * np.hanning(len(yu))))
    freq = np.fft.rfftfreq(len(yu), tu[1] - tu[0])
    spec, freq = spec[1:], freq[1:]
    k = int(np.argmax(spec))
    # A real oscillation puts a peak well above the median of the spectrum.
    if spec[k] < 10.0 * (np.median(spec) + 1e-30):
        return None
    return float(freq[k])


def _histories(case: Path) -> Dict[str, tuple]:
    """Every function-object history, one per output file name, with the
    pieces a restarted run writes (one folder per start time) joined in time
    order; where pieces overlap, the later start wins."""
    import numpy as np

    pieces: Dict[str, List[tuple]] = {}
    for dat in sorted(case.glob("postProcessing/**/*.dat")):
        rel = dat.relative_to(case / "postProcessing")
        parts = list(rel.parts)
        start = None
        if len(parts) >= 3:
            try:
                start = float(parts[-2])
                parts.pop(-2)
            except ValueError:
                pass
        header, rows = _read_history(dat)
        if not header or not rows:
            continue
        pieces.setdefault("/".join(parts), []).append((start if start is not None else 0.0, header, rows))
    merged: Dict[str, tuple] = {}
    for key, segs in pieces.items():
        segs.sort(key=lambda s: s[0])
        header = segs[-1][1]
        width = min(min(len(r) for r in rows) for _, _, rows in segs)
        chunks = []
        for i, (start, _, rows) in enumerate(segs):
            arr = np.array([r[:width] for r in rows])
            nxt = segs[i + 1][0] if i + 1 < len(segs) else None
            if nxt is not None:
                arr = arr[arr[:, 0] < nxt]
            chunks.append(arr)
        arr = np.vstack(chunks)
        merged[key] = (header, arr[np.argsort(arr[:, 0], kind="stable")])
    return merged


def _settled_start(t, y) -> float:
    """Start of the part of a history that looks like its end: the first
    block after which every block's mean and fluctuation stay within
    SETTLED_TOLERANCE of the last blocks. A start-up transient is excluded and
    all of the settled part is used; a history that never settles falls back
    to its last MIN_WINDOW_FRACTION, where the halves show it still changing."""
    import numpy as np

    edges = np.linspace(t[0], t[-1], WINDOW_BLOCKS + 1)
    stats = []
    for a, b in zip(edges[:-1], edges[1:]):
        seg = y[(t >= a) & (t <= b)]
        stats.append((float(seg.mean()), float(seg.std())) if seg.size else (np.nan, np.nan))
    stats = np.array(stats)
    ref_mean = float(np.nanmean(stats[-3:, 0]))
    ref_rms = float(np.nanmean(stats[-3:, 1]))
    # A block's mean carries phase noise from the partial cycles it contains,
    # so it is compared on the scale of the fluctuation; the fluctuation
    # itself must match closely. The floors keep a steady signal from
    # tolerating a ripple much larger than its own noise.
    mean_tol = 0.25 * max(ref_rms, 0.01 * abs(ref_mean), 1e-12)
    rms_tol = SETTLED_TOLERANCE * max(ref_rms, 1e-4 * abs(ref_mean), 1e-12)
    ok = (np.abs(stats[:, 0] - ref_mean) <= mean_tol) & (np.abs(stats[:, 1] - ref_rms) <= rms_tol)
    start = len(ok)
    while start > 0 and ok[start - 1]:
        start -= 1
    latest = t[-1] - MIN_WINDOW_FRACTION * (t[-1] - t[0])
    return float(min(edges[start], latest))


def history_statistics(case: Path | str) -> List[SeriesStats]:
    """Settled-window statistics for every scalar history the case wrote."""
    import numpy as np

    case = Path(case)
    out: List[SeriesStats] = []
    for source, (header, arr) in _histories(case).items():
        if len(arr) < 16:
            continue
        width = arr.shape[1]
        t = arr[:, 0]
        if t[-1] <= t[0]:
            continue
        for j in range(1, min(width, len(header))):
            if not np.all(np.isfinite(arr[:, j])):
                continue
            settled = t >= _settled_start(t, arr[:, j])
            y = arr[settled, j]
            if len(y) < 16:
                continue
            fluct = float(np.sqrt(np.mean((y - y.mean()) ** 2)))
            # Below this the "oscillation" is round-off: a frequency found in it
            # is noise, and reporting one would read as physics.
            freq = (_dominant_frequency(t[settled], y)
                    if fluct > NOISE_FLOOR * max(1.0, abs(float(y.mean()))) else None)
            half = len(y) // 2
            halves = tuple((float(p.mean()), float(np.sqrt(np.mean((p - p.mean()) ** 2))))
                           for p in (y[:half], y[half:]))
            out.append(SeriesStats(source=source, column=header[j], mean=float(y.mean()),
                                   fluctuation_rms=fluct, dominant_frequency=freq,
                                   window=(float(t[settled][0]), float(t[settled][-1])),
                                   samples=int(settled.sum()), halves=halves))
    return out


def history_quantities(stats: List[SeriesStats]) -> Dict[str, float]:
    """The statistics as named scalars a pairwise comparison can use."""
    q: Dict[str, float] = {}
    for s in stats:
        base = f"{Path(s.source).parent.parts[0] if Path(s.source).parent.parts else Path(s.source).stem}.{s.column}"
        q[f"{base}.mean"] = s.mean
        q[f"{base}.fluctuation_rms"] = s.fluctuation_rms
        if s.dominant_frequency is not None:
            q[f"{base}.dominant_frequency"] = s.dominant_frequency
    return q


# --------------------------------------------------------------------- transient

def is_transient(case: Path | str) -> bool:
    """True when the case marches in time rather than iterating to a steady state."""
    f = Path(case) / "system" / "fvSchemes"
    if not f.is_file():
        return False
    text = f.read_text(errors="ignore")
    m = re.search(r"ddtSchemes\s*\{(.*?)\}", text, re.S)
    body = m.group(1) if m else ""
    return any(k.lower() in body.lower() for k in _TRANSIENT_DDT)


def _end_time(case: Path) -> Optional[float]:
    cd = case / "system" / "controlDict"
    if not cd.is_file():
        return None
    m = re.search(r"\bendTime\s+([0-9.eE+-]+)\s*;", cd.read_text(errors="ignore"))
    return float(m.group(1)) if m else None


def ensure_time_averaging(case: Path | str) -> bool:
    """Add a fieldAverage function object to a transient case that lacks one.

    Averages start halfway through the run so start-up transients are excluded.
    Returns True if anything was added. A steady case, or one that already
    averages, is left alone.
    """
    case = Path(case)
    cd = case / "system" / "controlDict"
    if not cd.is_file() or not is_transient(case):
        return False
    text = cd.read_text(errors="ignore")
    if "fieldAverage" in text:
        return False
    end = _end_time(case)
    if end is None:
        return False
    block = (
        "    gateFieldAverage\n    {\n"
        '        type fieldAverage; libs ("libfieldFunctionObjects.so");\n'
        f"        writeControl writeTime; timeStart {0.5 * end:g};\n"
        "        fields ( U { mean on; prime2Mean on; base time; } "
        "p { mean on; prime2Mean off; base time; } );\n"
        "    }\n"
    )
    m = re.search(r"^functions\s*\{", text, re.M)
    if m:
        text = text[: m.end()] + "\n" + block + text[m.end():]
    else:
        text = text.rstrip() + "\n\nfunctions\n{\n" + block + "}\n"
    cd.write_text(text)
    return True


# --------------------------------------------------------------------- rendering

def _views(case: Path, plane, normal) -> Dict[str, Optional[tuple]]:
    """Regions of the flow plane worth a figure each: the whole domain, the
    region where the flow departs from the bulk, and the neighbourhood of the
    walls. Colour limits are set per view, so a small wake is not washed out by
    a large domain or by peak values at a wall."""
    import numpy as np

    views: Dict[str, Optional[tuple]] = {"whole domain": None}
    full = np.asarray(plane.bounds, dtype=float).reshape(3, 2)
    ax = int(np.argmax(normal))

    def box(lo, hi, pad):
        span = np.maximum(hi - lo, 1e-12)
        lo, hi = np.maximum(lo - pad * span, full[:, 0]), np.minimum(hi + pad * span, full[:, 1])
        lo[ax], hi[ax] = full[ax, 0], full[ax, 1]
        if np.allclose(lo, full[:, 0]) and np.allclose(hi, full[:, 1]):
            return None
        return (lo[0], hi[0], lo[1], hi[1], lo[2], hi[2])

    if "U" in plane.point_data:
        speed = np.linalg.norm(np.asarray(plane.point_data["U"]), axis=1)
        bulk = float(np.median(speed))
        disturbed = np.abs(speed - bulk) > 0.05 * max(abs(bulk), 1e-12)
        if disturbed.any():
            pts = np.asarray(plane.points)[disturbed]
            b = box(pts.min(axis=0), pts.max(axis=0), 0.05)
            if b is not None:
                views["disturbed region"] = b

    from cfd_langgraph.foam_load import load_patches
    from cfd_langgraph.wall_resolution import _patch_types

    walls = [n for n, t in _patch_types(case).items() if t == "wall"]
    try:
        patches = load_patches(case)
    except Exception:  # noqa: BLE001
        patches = {}
    pts = [np.asarray(m.points) for n, m in patches.items() if n.split("/")[-1] in walls]
    if pts:
        pts = np.vstack(pts)
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        b = box(lo, hi, 1.0)
        if b is not None:
            views["near walls"] = b
    return views


def _view_up(plane, normal) -> List[float]:
    """Screen-up direction: a positive coordinate axis, so the picture keeps the
    case's own orientation (never tilted or upside down), chosen perpendicular
    to the bulk flow where there is one, so that flow reads across the screen."""
    import numpy as np

    n = np.asarray(normal, dtype=float)
    k = int(np.argmax(np.abs(n)))
    in_plane = [i for i in range(3) if i != k]
    flow = None
    if "U" in plane.point_data:
        flow = np.abs(np.median(np.asarray(plane.point_data["U"]), axis=0))
    if flow is None or max(flow[i] for i in in_plane) < 1e-12:
        b = np.asarray(plane.bounds).reshape(3, 2)
        flow = b[:, 1] - b[:, 0]
    along = max(in_plane, key=lambda i: flow[i])
    up = [0.0, 0.0, 0.0]
    up[next(i for i in in_plane if i != along)] = 1.0
    return up


def render_level(case: Path | str, out_dir: Path | str) -> List[Path]:
    """Figures a reviewer needs to judge the flow: the flow plane at the latest
    solved time -- instantaneous, because an average of an oscillating flow can
    look steady -- and the monitored histories over time."""
    import numpy as np

    from cfd_langgraph.foam_load import empty_axis, load_case, new_plotter

    case, out_dir = Path(case), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    images: List[Path] = []
    mesh = load_case(case)
    axis = empty_axis(case)
    normal = [0.0, 0.0, 0.0]
    b = np.asarray(mesh.bounds).reshape(3, 2)
    normal[axis if axis is not None else int(np.argmin(b[:, 1] - b[:, 0]))] = 1.0
    field_mesh = mesh
    scalars = []
    if "U" in mesh.point_data:
        mesh.point_data["velocity magnitude"] = np.linalg.norm(np.asarray(mesh.point_data["U"]), axis=1)
        scalars.append("velocity magnitude")
        if axis is not None:
            try:
                d = mesh.compute_derivative(scalars="U", vorticity=True)
                d.point_data["out-of-plane vorticity"] = np.asarray(d.point_data["vorticity"])[:, int(np.argmax(normal))]
                field_mesh = d
                scalars.append("out-of-plane vorticity")
            except Exception:  # noqa: BLE001
                pass
    plane = field_mesh.slice(normal=normal, origin=field_mesh.center)
    views = _views(case, plane, normal)
    near_walls = views.get("near walls")
    up = _view_up(plane, normal)
    for view, bounds in views.items():
        region = plane if bounds is None else plane.clip_box(bounds, invert=False)
        if region.n_points == 0:
            continue
        for s in scalars:
            vals = np.asarray(region.point_data[s])
            if "vorticity" in s:
                # Away from the walls the wall shear layers would set the scale
                # and hide everything else, so they are left to saturate.
                pick = np.abs(vals)
                if view != "near walls" and near_walls is not None:
                    pts = np.asarray(region.points)
                    lo, hi = np.array(near_walls[0::2]), np.array(near_walls[1::2])
                    outside = ~np.all((pts >= lo) & (pts <= hi), axis=1)
                    if outside.any():
                        pick = pick[outside]
                lim = float(np.nanpercentile(pick, 99 if pick.size < vals.size else 95)) or 1.0
                clim = (-lim, lim)
            else:
                clim = (float(np.nanmin(vals)), float(np.nanmax(vals)))
            p = new_plotter(window_size=(1500, 700))
            p.add_mesh(region, scalars=s, cmap="RdBu_r" if "vorticity" in s else "viridis", clim=clim,
                       scalar_bar_args={"title": f"{s} ({view})", "vertical": False, "height": 0.06,
                                        "position_x": 0.25, "position_y": 0.03})
            p.camera_position = [tuple(np.asarray(region.center) + np.asarray(normal)),
                                 tuple(region.center), tuple(up)]
            p.reset_camera()
            p.camera.zoom(1.2)
            p.add_axes()
            path = out_dir / f"{case.name}_{s.replace(' ', '_')}_{view.replace(' ', '_')}.png"
            p.screenshot(str(path)); p.close()
            images.append(path)

    stats = history_statistics(case)
    if stats:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        histories = _histories(case)
        by_file: Dict[str, List[str]] = {}
        for s in stats:
            by_file.setdefault(s.source, []).append(s.column)
        n = min(sum(len(v) for v in by_file.values()), 6)
        fig, axes = plt.subplots(n, 1, figsize=(9, 1.8 * n), sharex=True, squeeze=False)
        i = 0
        for source, cols in by_file.items():
            header, arr = histories[source]
            t = arr[:, 0]
            # Start-up transients can be orders of magnitude larger than the
            # signal; scale each axis to the run after its first tenth.
            later = t >= t[0] + 0.1 * (t[-1] - t[0])
            for c in cols:
                if i >= n:
                    break
                y = arr[:, header.index(c)]
                axes[i][0].plot(t, y, lw=0.8)
                lo, hi = float(np.min(y[later])), float(np.max(y[later]))
                pad = 0.1 * (hi - lo) or 0.1 * max(abs(hi), 1e-12)
                axes[i][0].set_ylim(lo - pad, hi + pad)
                window = next((s.window for s in stats if s.source == source and s.column == c), None)
                if window:
                    axes[i][0].axvspan(window[0], window[1], color="0.9", zorder=0)
                axes[i][0].set_ylabel(c, fontsize=8)
                i += 1
        axes[-1][0].set_xlabel("time")
        fig.suptitle("monitored histories (shaded: window the statistics are taken over)")
        fig.tight_layout()
        path = out_dir / f"{case.name}_histories.png"
        fig.savefig(path, dpi=110); plt.close(fig)
        images.append(path)
    return images


# --------------------------------------------------------------------- judgement

ASSESS_SYSTEM = """You are a senior CFD engineer checking whether one mesh level's \
solution can be trusted before any study is run on it.

You judge three things, and only these three:

1. PHYSICAL PLAUSIBILITY. Is this flow what the stated case must produce? Use your \
knowledge of the physics of this configuration and regime -- which structures, \
instabilities, separation, unsteadiness or symmetry a flow like this must or must not \
show. Look at the figures: a solution can converge perfectly and still be the wrong \
flow -- a regime, symmetry or structure the physics of this configuration does not \
allow, or flow features damped out by a mesh too coarse to carry them.

2. NUMERICAL ADEQUACY. Is the resolution sufficient where the flow needs it -- near \
walls, in shear layers, in wakes, around separation -- and is the near-wall spacing \
suitable for the wall treatment in use?

3. SETTLED IN TIME. Have the monitored quantities stopped changing in time over the \
window the statistics are taken from (shaded in the history plot)? An oscillation whose \
amplitude is still growing or decaying, or a mean that is still drifting, has not settled; \
compare the first and second halves of the window. For a steady solver, settled means \
the iterations have converged: the residuals met their targets or the solution no \
longer changes between saved iterations; a run that simply reached its last iteration \
has not settled. A solution that has not settled cannot \
be compared with another level yet: it needs a longer run, not a different mesh. Do not \
count an unsettled history against numerical adequacy -- judge adequacy from what the \
fields show.

You do NOT compare against reference data or target values, even if any appear in the \
description. Matching a reference is the study's job after the mesh is settled.

When coarser levels of the same mesh sequence are given, use them. A feature that no \
longer changes when the mesh is refined is set by the model and the physics, not by \
resolution, and more cells will not change it: judge numerical adequacy by whether the \
solution is still changing under refinement where the flow needs it, not by how the \
flow would look with a different model or in three dimensions.

If the gate's quantities drift the same way at every level without the changes \
shrinking, the solution is not converging with refinement: look for the cause -- an \
unconverged iterative solve, a setup error -- instead of calling the level adequate.

WHERE TO REFINE. If the level is not acceptable, decide from the flow field, the \
geometry and the case parameters where the flow is under-resolved, and refine only \
there -- the regions and the directions that need it, not the whole domain. A cell \
count is a budget: one refinement step may produce a mesh with at most about twice the \
cells of this one, so spend the added cells where they change the solution most. Say \
which regions, in which directions, and roughly by how much.

CHECK BEFORE YOU CONCLUDE. A figure can mislead: its orientation, its colour range or \
the region it shows. Before you call anything in a figure wrong, check it with the tools: \
probe_fields gives the solution's values at coordinates you choose, and redraw draws a \
field again over a region you choose, with labelled coordinate axes. Compare what the \
checks show with the case parameters (which boundary moves, where the walls are). Judge \
the flow from the checks; if a figure turned out to be misleading, say so in issues and \
do not count it against the solution."""


@dataclass
class LevelAssessment:
    physically_plausible: bool
    numerically_adequate: bool
    issues: List[str] = field(default_factory=list)
    refine_where: str = ""
    refine_how: str = ""
    reason: str = ""
    statistically_settled: bool = True

    @property
    def acceptable(self) -> bool:
        return self.physically_plausible and self.numerically_adequate and self.statistically_settled

    def refinement_instruction(self) -> str:
        if not (self.refine_where or self.refine_how):
            return ""
        parts = [p.strip().rstrip(".") for p in (self.refine_where, self.refine_how) if p.strip()]
        return ("Refine the mesh as follows. " + ". ".join(parts)
                + ". Keep the domain, block topology and patch names identical; change "
                "only cell counts and grading. This is a resolution change only. The new "
                "mesh must have at most about twice as many cells as the current one: add "
                "them only where stated above.")


def _numbers_text(stats: List[SeriesStats], wall: Optional[Any]) -> str:
    lines = []
    for s in stats[:24]:
        freq = f", dominant frequency {s.dominant_frequency:.4g}" if s.dominant_frequency else ", no clear periodic content"
        lines.append(f"  {s.source} :: {s.column}: mean {s.mean:.5g}, fluctuation rms "
                     f"{s.fluctuation_rms:.3g}{freq} (window t={s.window[0]:.4g}..{s.window[1]:.4g})"
                     + (f"; first half of window mean {s.halves[0][0]:.4g} rms {s.halves[0][1]:.3g}, "
                        f"second half mean {s.halves[1][0]:.4g} rms {s.halves[1][1]:.3g}" if s.halves else ""))
    if wall is not None and getattr(wall, "walls", None):
        lines.append("  near-wall spacing:")
        for w in wall.walls.values():
            lines.append(f"    {w.patch}: y+ min {w.yplus_min:.3g}, mean {w.yplus_mean:.3g}, "
                         f"max {w.yplus_max:.3g}; wall treatment {w.treatment}")
    return "\n".join(lines) or "  (no monitored histories were written)"


def _cell_count(case: Path) -> Optional[int]:
    owner = Path(case) / "constant" / "polyMesh" / "owner"
    if not owner.is_file():
        return None
    with owner.open(errors="ignore") as fh:
        head = fh.read(2000)
    m = re.search(r"nCells:\s*(\d+)", head)
    return int(m.group(1)) if m else None


def case_parameters(case: Path | str) -> str:
    """What a reviewer needs to reason about where a flow needs resolution: the
    solver, the turbulence treatment, the viscosity, every boundary's velocity
    condition, the domain extent and the size of each wall, from the case
    itself."""
    import re as _re

    case = Path(case)
    lines = []

    def read(rel):
        p = case / rel
        return p.read_text(errors="ignore") if p.is_file() else ""

    app = _re.search(r"\bapplication\s+(\w+)\s*;", read("system/controlDict"))
    if app:
        lines.append(f"solver: {app.group(1)}")
    mt = read("constant/momentumTransport")
    sim = _re.search(r"\bsimulationType\s+(\w+)\s*;", mt)
    model = _re.search(r"\bmodel\s+(\w+)\s*;", mt)
    if sim:
        lines.append(f"turbulence treatment: {sim.group(1)}" + (f", model {model.group(1)}" if model else ""))
    for name in ("physicalProperties", "transportProperties"):
        nu = _re.search(r"\bnu\b[^;]*?([-+0-9.eE]+)\s*;", read(f"constant/{name}"))
        if nu:
            lines.append(f"kinematic viscosity nu: {nu.group(1)}")
            break
    u = read("0/U")
    for patch, body in _re.findall(r"\n\s*(\w+)\s*\{([^{}]*)\}", u.split("boundaryField", 1)[-1]):
        kind = _re.search(r"type\s+(\w+)", body)
        vals = _re.findall(r"\(([^()]*)\)", body)
        lines.append(f"velocity boundary {patch}: {kind.group(1) if kind else '?'}"
                     + (f", value ({vals[0].strip()})" if vals else ""))
    n = _cell_count(case)
    if n:
        lines.append(f"cells: {n}")
    try:
        from cfd_langgraph.foam_load import load_patches, patch_types

        types = patch_types(case)
        for name, mesh in load_patches(case).items():
            short = name.split("/")[-1]
            if types.get(short) == "wall":
                b = mesh.bounds
                lines.append(f"wall {short}: extent x {b[1]-b[0]:.4g}, y {b[3]-b[2]:.4g}, z {b[5]-b[4]:.4g}")
    except Exception:  # noqa: BLE001
        pass
    try:
        from cfd_langgraph.case_inventory import collect

        inv = collect(case)
        if inv.bounds:
            b = inv.bounds
            lines.append(f"domain extent: x {b[0]:.4g}..{b[1]:.4g}, y {b[2]:.4g}..{b[3]:.4g}, z {b[4]:.4g}..{b[5]:.4g}")
    except Exception:  # noqa: BLE001
        pass
    return "\n".join(f"  {l}" for l in lines) or "  (not readable from the case)"


def _earlier_text(earlier: List[Path]) -> str:
    if not earlier:
        return ""
    lines = ["COARSER LEVELS OF THIS MESH SEQUENCE (coarsest first):"]
    for lvl in earlier:
        lvl = Path(lvl)
        lines.append(f"- {lvl.name}: {_cell_count(lvl) or 'unknown'} cells")
        for q in _numbers_text(history_statistics(lvl), None).splitlines():
            lines.append(f"    {q}")
    return "\n".join(lines) + "\n\n"


def probe_fields(case: Path | str, points: List[List[float]]) -> str:
    """The solution's point fields at the latest time, at each given coordinate."""
    import numpy as np

    from cfd_langgraph.foam_load import _pyvista, load_case

    mesh = load_case(case)
    pts = np.asarray(points, dtype=float).reshape(-1, 3)[:20]
    sampled = _pyvista().PolyData(pts).sample(mesh)
    valid = np.asarray(sampled.point_data.get("vtkValidPointMask", np.ones(len(pts))))
    lines = []
    for i, pt in enumerate(pts):
        if not valid[i]:
            lines.append(f"{tuple(round(float(v), 6) for v in pt)}: outside the domain")
            continue
        vals = []
        for name in mesh.point_data.keys():
            v = np.asarray(sampled.point_data[name])[i]
            vals.append(f"{name}={np.array2string(np.atleast_1d(v), precision=4, separator=' ')}")
        lines.append(f"{tuple(round(float(v), 6) for v in pt)}: " + ", ".join(vals))
    return "\n".join(lines)


def redraw(case: Path | str, out_path: Path | str, *, field: str,
           bounds: Optional[List[float]] = None) -> Path:
    """One field on the mid-plane (2-D) or on the plane through the domain's
    thinnest direction, over ``bounds`` (xmin xmax ymin ymax zmin zmax) if
    given, drawn with coordinate axes and tick labels so positions can be read
    off the picture. A vector field is drawn as its magnitude, or one
    component when ``field`` is written as ``U_x``, ``U_y`` or ``U_z``."""
    import numpy as np

    from cfd_langgraph.foam_load import empty_axis, load_case, new_plotter

    case, out_path = Path(case), Path(out_path)
    mesh = load_case(case)
    axis = empty_axis(case)
    b = np.asarray(mesh.bounds).reshape(3, 2)
    k = axis if axis is not None else int(np.argmin(b[:, 1] - b[:, 0]))
    normal = [0.0, 0.0, 0.0]
    normal[k] = 1.0
    base, comp = field, None
    if field[-2:] in ("_x", "_y", "_z"):
        base, comp = field[:-2], "xyz".index(field[-1])
    if base not in mesh.point_data:
        raise ValueError(f"no field {base!r}; the case has {sorted(mesh.point_data.keys())}")
    vals = np.asarray(mesh.point_data[base])
    if vals.ndim > 1:
        vals = vals[:, comp] if comp is not None else np.linalg.norm(vals, axis=1)
    mesh.point_data[field] = vals
    plane = mesh.slice(normal=normal, origin=mesh.center)
    if bounds:
        bb = [float(v) for v in bounds][:6]
        bb[2 * k], bb[2 * k + 1] = b[k, 0] - 1, b[k, 1] + 1
        plane = plane.clip_box(bb, invert=False)
    if plane.n_points == 0:
        raise ValueError("the region holds no part of the domain")
    shown = np.asarray(plane.point_data[field])
    p = new_plotter(window_size=(1500, 900))
    p.add_mesh(plane, scalars=field, cmap="viridis", clim=(float(np.nanmin(shown)), float(np.nanmax(shown))),
               scalar_bar_args={"title": field, "vertical": False, "height": 0.05,
                                "position_x": 0.25, "position_y": 0.02})
    up = [0.0, 0.0, 0.0]
    up[[i for i in range(3) if i != k][1]] = 1.0
    p.camera_position = [tuple(np.asarray(plane.center) + np.asarray(normal)), tuple(plane.center), tuple(up)]
    p.reset_camera()
    p.show_bounds(location="outer", ticks="outside", font_size=10, all_edges=False,
                  xtitle="x", ytitle="y", ztitle="z")
    p.add_axes()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    p.screenshot(str(out_path)); p.close()
    return out_path


MAX_CHECKS = 6


def _checked_messages(llm: Any, case: Path, messages: List[Any], out_dir: Path) -> List[Any]:
    """Let the reviewer check what the figures show before it gives a verdict:
    up to MAX_CHECKS tool calls (probe_fields, redraw). Returns the
    conversation with the checks restated as plain text and figures, which
    every provider accepts in a call that binds no tools; unchanged if the
    model cannot call tools or calls none."""
    import base64

    from langchain_core.messages import HumanMessage, ToolMessage
    from pydantic import BaseModel, Field

    class probe_fields_args(BaseModel):
        """The solution's values at the latest time at up to 20 coordinates."""
        points: List[List[float]] = Field(description="Coordinates [[x, y, z], ...]")

    class redraw_args(BaseModel):
        """Draw one field again over a region you choose, with labelled coordinate axes."""
        field: str = Field(description="A point field, e.g. U, p; U_x / U_y / U_z for one component")
        bounds: Optional[List[float]] = Field(default=None, description="[xmin, xmax, ymin, ymax, zmin, zmax]; omit for the whole domain")

    try:
        bound = llm.bind_tools([probe_fields_args, redraw_args])
    except Exception:  # noqa: BLE001
        return messages
    msgs = list(messages)
    used = 0
    while used < MAX_CHECKS:
        try:
            reply = bound.invoke(msgs)
        except Exception:  # noqa: BLE001
            break
        calls = list(getattr(reply, "tool_calls", None) or [])
        if not calls:
            break
        msgs.append(reply)
        images = []
        for call in calls:
            used += 1
            args = call.get("args") or {}
            try:
                if call.get("name") == "probe_fields_args":
                    result = probe_fields(case, args.get("points") or [])
                elif call.get("name") == "redraw_args" and used <= MAX_CHECKS:
                    path = redraw(case, out_dir / f"check_{used}_{args.get('field', 'field')}.png",
                                  field=str(args.get("field", "")), bounds=args.get("bounds"))
                    images.append(path)
                    result = f"Drawn: {path.name}; the picture follows."
                else:
                    result = "Not run: no checks left." if used > MAX_CHECKS else f"Unknown tool {call.get('name')}."
            except Exception as exc:  # noqa: BLE001
                result = f"Failed: {type(exc).__name__}: {str(exc)[:500]}"
            msgs.append(ToolMessage(content=result, tool_call_id=call.get("id") or ""))
        if images:
            content: List[Dict[str, Any]] = []
            for img in images:
                content.append({"type": "text", "text": f"Figure: {img.name}"})
                content.append({"type": "image_url", "image_url": {
                    "url": "data:image/png;base64," + base64.b64encode(img.read_bytes()).decode()}})
            msgs.append(HumanMessage(content=content))
    if len(msgs) == len(messages):
        return messages
    record: List[Dict[str, Any]] = [{"type": "text", "text": "YOUR CHECKS AND WHAT THEY SHOWED:"}]
    for m in msgs[len(messages):]:
        if isinstance(m, ToolMessage):
            record.append({"type": "text", "text": f"result: {m.content}"})
        elif isinstance(m, HumanMessage):
            record.extend(m.content)
        else:
            for call in getattr(m, "tool_calls", None) or []:
                name = str(call.get("name", "")).removesuffix("_args")
                record.append({"type": "text", "text": f"{name}({json.dumps(call.get('args') or {})})"})
    record.append({"type": "text", "text": (
        "Give your verdict now. Where a check and a figure disagree, the check is right.")})
    return list(messages) + [HumanMessage(content=record)]


# Relative change of the velocity field between the last two saved
# iterations above which a steady solve is not finished.
STEADY_CHANGE_TOL = 1e-3


# A steady run stopped at a fixed iteration count is not settled because it ended:
# each finer level was further from converged and the gate refined to 4.8M cells
# (qwen_retest_20261005/cavity_r8).
def steady_unsettled(case: Path | str) -> Optional[str]:
    """Why a steady solve has not converged, or None if it has or it cannot be
    told. Converged means the solver met its own residual targets, or the
    velocity field no longer changes between the last two saved iterations."""
    import numpy as np

    from cfd_langgraph.foam_load import load_case

    case = Path(case)
    if not (case / "system" / "controlDict").is_file() or is_transient(case):
        return None
    for log in case.glob("log.*"):
        try:
            if "solution converged in" in log.read_text(errors="ignore")[-20000:]:
                return None
        except OSError:
            continue
    times = sorted(t for t in (_time_value(d.name) for d in case.iterdir() if d.is_dir()) if t and t > 0)
    if len(times) < 2:
        return None
    try:
        new, old = load_case(case, times[-1]), load_case(case, times[-2])
        u_new = np.asarray(new.point_data["U"], dtype=float)
        u_old = np.asarray(old.point_data["U"], dtype=float)
    except Exception:  # noqa: BLE001
        return None
    scale = float(np.linalg.norm(u_new)) or 1.0
    change = float(np.linalg.norm(u_new - u_old)) / scale
    if change <= STEADY_CHANGE_TOL:
        return None
    return (f"The steady solve has not converged: the velocity field still changed by {change:.2%} "
            f"between iterations {times[-2]:g} and {times[-1]:g}, and the solver's own residual "
            "targets were not met. Values measured on this level are not final.")


def _time_value(name: str) -> Optional[float]:
    from cfd_langgraph.foam_native.case_clean import parse_time_dir

    return parse_time_dir(name)


def _measured_text(measured: Optional[Dict[str, Dict[str, float]]]) -> str:
    if not measured:
        return ""
    lines = ["GATE QUANTITIES MEASURED ON EACH LEVEL SO FAR (coarsest first):"]
    for name, values in measured.items():
        shown = ", ".join(f"{k} {v:.5g}" for k, v in values.items() if isinstance(v, (int, float)))
        lines.append(f"- {name}: {shown or '(not measured)'}")
    return "\n".join(lines) + "\n\n"


def assess_level(llm: Any, case: Path | str, *, case_description: str,
                 images: List[Path], stats: List[SeriesStats],
                 wall: Optional[Any] = None, transient: Optional[bool] = None,
                 earlier: Optional[List[Path]] = None,
                 earlier_images: Optional[List[Path]] = None,
                 measured: Optional[Dict[str, Dict[str, float]]] = None) -> Optional[LevelAssessment]:
    """Look at one level and decide whether it can be trusted. None on failure --
    an absent judgement must never count as approval. ``earlier`` lists the
    coarser levels of the same sequence, coarsest first, for comparison;
    ``earlier_images`` are figures of the previous level."""
    import base64

    from langchain_core.messages import HumanMessage, SystemMessage
    from pydantic import BaseModel, Field

    from cfd_langgraph.utils import structured_output

    case = Path(case)
    transient = is_transient(case) if transient is None else transient

    class _Verdict(BaseModel):
        physically_plausible: bool = Field(description="Is this the flow the stated case must produce?")
        numerically_adequate: bool = Field(description="Is resolution sufficient where the flow needs it?")
        statistically_settled: bool = Field(description=(
            "Have the monitored quantities stopped changing in time over the statistics window?"))
        issues: List[str] = Field(default_factory=list)
        refine_where: str = Field(default="", description="Region needing more resolution, if any")
        refine_how: str = Field(default="", description="Direction and rough factor, if any")
        reason: str = ""

    text = (
        f"CASE:\n{case_description.strip()[:4000]}\n\n"
        f"TIME TREATMENT: {'transient -- figures of the flow are instantaneous at the latest time' if transient else 'steady'}\n\n"
        f"CASE PARAMETERS (read from the case):\n{case_parameters(case)}\n\n"
        f"THIS LEVEL: {_cell_count(case) or 'unknown'} cells\n"
        f"MONITORED QUANTITIES:\n{_numbers_text(stats, wall)}\n\n"
        + _earlier_text(earlier or [])
        + _measured_text(measured)
        + (f"ITERATIVE CONVERGENCE: {steady_note}\n\n" if (steady_note := steady_unsettled(case)) else "")
        + "The figures follow. Check anything that looks wrong with the tools, then judge."
    )
    content: List[Dict[str, Any]] = [{"type": "text", "text": text}]
    compare = [Path(p) for p in (earlier_images or []) if Path(p).is_file()][:3]
    for img in images[:8] + compare:
        b64 = base64.b64encode(Path(img).read_bytes()).decode()
        label = f"Figure: {Path(img).stem}" + (" (previous, coarser level)" if img in compare else "")
        content.append({"type": "text", "text": label})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
    messages = [SystemMessage(content=ASSESS_SYSTEM), HumanMessage(content=content)]
    check_dir = Path(images[0]).parent / "checks" if images else case / "review_checks"
    messages = _checked_messages(llm, case, messages, check_dir)
    try:
        out = structured_output(llm, _Verdict).invoke(messages)
        return LevelAssessment(physically_plausible=bool(out.physically_plausible),
                               numerically_adequate=bool(out.numerically_adequate),
                               issues=[str(i) for i in out.issues], refine_where=str(out.refine_where),
                               refine_how=str(out.refine_how), reason=str(out.reason),
                               statistically_settled=bool(out.statistically_settled))
    except Exception:  # noqa: BLE001
        return None


# ------------------------------------------------------------ how later runs start

START_SYSTEM = """You decide how the later runs of a CFD study start. They all use \
the mesh and setup of the level shown here; each changes something -- a model, a \
coefficient, a parameter -- and is then compared with the others.

There are two choices.

FROM THE SETTLED STATE. Every later run starts from this level's flow at its final \
time and runs for a length you set. Choose this only if that flow has settled: the \
monitored quantities no longer drift, an oscillation keeps a steady amplitude and \
frequency, the first and second halves of the statistics window agree. It must also be \
the right flow for this case -- not a start-up transient, not a symmetric or steady \
branch where the physics requires something else. Look at the figures and the numbers, \
and check anything doubtful with the tools.

FROM REST. Every later run starts from the case's initial conditions and runs to the \
case's own end time, as this level did. Choose this if the flow has not settled, or if \
starting from it could carry this level's state into runs that should find their own.

If you choose the settled state, set run_length in the case's own time units (or \
iterations, for a steady solver). It must cover two things: the time a changed model \
needs to move the flow from this state to its own settled state, and then the window \
the study's quantities are averaged over. Reason in the flow's own time scales -- \
oscillation periods from the dominant frequencies, flow-through times from the domain \
length and velocity, iterations a steady solve needed -- and say how you got the \
number. It must be well short of the time this level took from rest, or starting from \
the settled state saves nothing. Set averaging_start to the time within such a run from \
which averages and statistics are taken: after the adjustment, before the end."""


@dataclass
class StartDecision:
    from_settled_state: bool
    run_length: float = 0.0
    reason: str = ""
    settled_time: Optional[float] = None
    averaging_start: Optional[float] = None


def latest_time(case: Path | str) -> Optional[float]:
    from cfd_langgraph.foam_native.case_clean import parse_time_dir

    case = Path(case)
    times = [t for t in (parse_time_dir(d.name) for d in case.iterdir() if d.is_dir()) if t is not None]
    return max(times) if times else None


def decide_start(llm: Any, case: Path | str, *, case_description: str, images: List[Path],
                 stats: List[SeriesStats]) -> Optional[StartDecision]:
    """Look at the chosen level and decide whether later runs start from its
    final flow, and for how long they then run. None on failure -- the caller
    then starts from rest, which is never wrong, only slower."""
    import base64

    from langchain_core.messages import HumanMessage, SystemMessage
    from pydantic import BaseModel, Field

    from cfd_langgraph.utils import structured_output

    case = Path(case)
    final = latest_time(case)
    if not final:
        return StartDecision(False, reason="the level has no solved time to start from")

    class _Start(BaseModel):
        from_settled_state: bool = Field(description="Start later runs from this level's final flow?")
        run_length: float = Field(default=0.0, description=(
            "If from the settled state: how long each later run lasts, in the case's time units "
            "or iterations. 0 otherwise."))
        averaging_start: float = Field(default=0.0, description=(
            "If from the settled state: the time within each later run from which averages "
            "and statistics are taken. 0 otherwise."))
        reason: str = Field(description="What you saw, and how you got run_length")

    text = (
        f"CASE:\n{case_description.strip()[:4000]}\n\n"
        f"TIME TREATMENT: {'transient' if is_transient(case) else 'steady'}\n"
        f"THIS LEVEL RAN FROM 0 TO t={final:g} (end time in its controlDict: {_end_time(case)})\n\n"
        f"CASE PARAMETERS (read from the case):\n{case_parameters(case)}\n\n"
        f"MONITORED QUANTITIES:\n{_numbers_text(stats, None)}\n\n"
        "The figures follow: the flow at the final time and the monitored histories. Check "
        "anything doubtful with the tools, then decide."
    )
    content: List[Dict[str, Any]] = [{"type": "text", "text": text}]
    for img in images[:8]:
        content.append({"type": "text", "text": f"Figure: {Path(img).stem}"})
        content.append({"type": "image_url", "image_url": {
            "url": "data:image/png;base64," + base64.b64encode(Path(img).read_bytes()).decode()}})
    messages = [SystemMessage(content=START_SYSTEM), HumanMessage(content=content)]
    check_dir = Path(images[0]).parent / "start_checks" if images else case / "start_checks"
    messages = _checked_messages(llm, case, messages, check_dir)
    try:
        out = structured_output(llm, _Start).invoke(messages)
    except Exception:  # noqa: BLE001
        return None
    length = float(out.run_length or 0)
    warm = bool(out.from_settled_state) and length > 0
    if not warm:
        return StartDecision(False, reason=str(out.reason))
    start = float(out.averaging_start or 0)
    if not 0 <= start < length:
        start = 0.5 * length
    return StartDecision(True, length, str(out.reason), final, start)


def _set_entry(dict_file: Path, entry: str, value: str, env: Optional[Dict[str, str]]) -> None:
    """Set one entry with OpenFOAM's own dictionary editor."""
    import subprocess

    proc = subprocess.run(["foamDictionary", "-entry", entry, "-set", value, str(dict_file)],
                          env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"foamDictionary could not set {entry} in {dict_file}: "
                           f"{(proc.stderr or proc.stdout)[-500:]}")


def set_averaging_start(control_dict: Path, start: float, env: Optional[Dict[str, str]]) -> List[str]:
    """Set timeStart of every fieldAverage function object in a controlDict,
    so averages cover the decided window of the run. Returns their names."""
    import subprocess

    def value(entry: str) -> str:
        proc = subprocess.run(["foamDictionary", "-entry", entry, "-value", str(control_dict)],
                              env=env, capture_output=True, text=True)
        return proc.stdout.strip() if proc.returncode == 0 else ""

    proc = subprocess.run(["foamDictionary", "-entry", "functions", "-keywords", str(control_dict)],
                          env=env, capture_output=True, text=True)
    names = [n.strip() for n in proc.stdout.splitlines() if n.strip()] if proc.returncode == 0 else []
    averaged = [n for n in names if value(f"functions/{n}/type") == "fieldAverage"]
    for n in averaged:
        _set_entry(control_dict, f"functions/{n}/timeStart", f"{start:g}", env)
    return averaged


def make_start_case(level: Path | str, dest: Path | str, decision: StartDecision, *,
                    env: Optional[Dict[str, str]] = None) -> Path:
    """The case every later run copies: the level's mesh and setup, starting
    at t=0 either from the case's own initial fields (from rest) or from the
    level's settled flow, with the end time set to the decided run length.

    The settled flow goes into 0/ rather than being restarted from its own
    time: a run then always starts at 0, whatever its startFrom says and
    whatever it rebuilds, and averages and histories begin afresh instead of
    carrying the level's. Only fields the case starts from are taken; outputs
    of function objects (averages, wall quantities) are not."""
    import shutil

    level, dest = Path(level), Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for item in ("0", "constant", "system"):
        if (level / item).is_dir():
            shutil.copytree(level / item, dest / item)
    from cfd_langgraph.foam_native.loop import drop_unread_transport_properties

    drop_unread_transport_properties(dest)
    taken: List[str] = []
    if decision.from_settled_state and decision.settled_time is not None:
        from cfd_langgraph.foam_native.case_clean import parse_time_dir

        final = next((d for d in level.iterdir()
                      if d.is_dir() and parse_time_dir(d.name) == decision.settled_time), None)
        if final is None:
            raise RuntimeError(f"{level} has no time directory {decision.settled_time:g}")
        for f in sorted((dest / "0").iterdir()):
            if f.is_file() and (final / f.name).is_file():
                shutil.copy2(final / f.name, f)
                taken.append(f.name)
        if not taken:
            raise RuntimeError(f"no field of {dest / '0'} was found at {final}")
    cd = dest / "system" / "controlDict"
    _set_entry(cd, "startFrom", "startTime", env)
    _set_entry(cd, "startTime", "0", env)
    averaged: List[str] = []
    if decision.from_settled_state:
        _set_entry(cd, "endTime", f"{decision.run_length:g}", env)
        averaged = set_averaging_start(cd, decision.averaging_start or 0.0, env)
    (dest / "start_decision.json").write_text(json.dumps({
        "source_level": str(level), "from_settled_state": decision.from_settled_state,
        "settled_time": decision.settled_time, "run_length": decision.run_length,
        "averaging_start": decision.averaging_start, "averaging_set_in": averaged,
        "fields_taken_from_settled_state": taken, "reason": decision.reason,
    }, indent=2))
    return dest


# ------------------------------------------------------------ solution quantities

QUANTITY_SYSTEM = """You choose which quantities a mesh-independence check should \
monitor for one CFD case.

The study is scored on metrics that may compare the solution with reference data (an \
RMSE against measurements, a percentage error). A mesh check must not use those: it \
asks whether the SOLUTION stops changing under refinement, which is a question about \
the solution alone. So translate each study metric into the solution quantities it is \
built from -- the drag coefficient rather than its error against a measurement, the \
separation and reattachment locations rather than a profile RMSE, a wall-quantity \
profile sampled at stations rather than its mismatch with data.

WHICH FIELDS. In a steady case, define every quantity on the fields at the final time. \
In a transient case, a single instant depends on where in its cycle the run happened to \
stop and never settles under refinement, so define every quantity on the time-averaged \
fields (for example UMean and pMean, and wall quantities derived from the time-averaged \
velocity) -- never on the instantaneous ones, and not on a field the case writes only \
instantaneously. Statistics over time of the monitored \
histories -- means, fluctuation levels and dominant frequencies of forces, probes and \
other function-object outputs -- are measured separately and automatically, so do not \
include those; choose only quantities that need the fields.

WHAT THE CASE WRITES. You are given the case's inventory: the fields it writes, on the \
interior and on each boundary, and every monitored output with its columns. Choose only \
quantities that can be computed from those fields, directly or by derivation (gradients, \
integrals, wall shear from velocity and viscosity, samples along lines). Never ask for a \
field the case does not write. Anything a monitored output already records -- a force \
coefficient column, a probe -- is measured automatically from its history, so do not \
recompute it from the fields.

The check is of the flow as the case is set up now. If the description mentions changes \
the study intends to try (a modified model, a new term), ignore them: measure the flow, \
not a proposed change.

NUMBERS THAT CANNOT BE NEAR ZERO. Each quantity is one number, and levels are compared \
by its relative change, so its value must stay well away from zero for this flow: a \
small difference of a near-zero number reads as a huge percentage change and the check \
can never settle. Choose quantities whose size is set by the flow itself. If a quantity \
summarises something that varies -- along a wall, across a wake, over time -- summarise \
it in a way that cannot cancel: positive and negative parts must not offset each other. \
Say in the definition how the single number is formed.

NO MAXIMA OR MINIMA OVER THE DOMAIN OR ALONG A WALL. Such an extreme lands where the \
solution is least resolved -- in the cell touching a wall, or at a corner or edge where \
boundary conditions meet and the exact solution has no finite value -- so it moves with \
every refinement and the check never settles. Measure away from walls and corners: \
values at fixed interior points or stations along a line, the extremes of a profile \
across the interior and where they occur, the location and strength of a flow feature \
(a vortex centre, a separation or reattachment point), integrals over the domain or a \
boundary.

Give each quantity a short snake_case name and a precise definition someone could \
compute from the solution fields alone. Prefer a handful of quantities that are \
sensitive to resolution where this flow needs it. Never reference any external data."""


def quantity_spec(name: str, definition: str, *, transient: bool, from_inventory: bool) -> Dict[str, Any]:
    """A gate quantity as the metric-spec entry the extractor computes."""
    return {
        "name": name,
        "description": definition.strip(),
        "computation_hint": (
            definition.strip() + " Compute from this case's solution fields only. "
            + ("This is a transient case: use the time-averaged fields written at the latest "
               "time (UMean, pMean and the like; foam_load.wall_shear(case, velocity='UMean') "
               "for wall shear), never the instantaneous U or p. If no time-averaged field "
               "exists, return null and say so in <name>__why_null."
               if transient else
               "This is a steady case: use the fields at the final time.")
            + " Do not read or compare against any reference data."
        ),
        "reference_files": [],
        "time_basis": "time-averaged" if transient else "final time",
        "chosen_from_inventory": from_inventory,
    }


def _snake(name: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", name.strip().lower()).strip("_")


def solution_quantities(llm: Any, *, metric_specs: List[Dict[str, Any]], case_description: str,
                        monitored: Optional[List[str]] = None,
                        transient: bool = False,
                        inventory: str = "") -> List[Dict[str, str]]:
    """Study metrics translated into solution-only quantities, as metric-spec
    entries the existing extractor can compute. ``monitored`` names the history
    columns already measured, so they are not asked for twice. Empty list on
    failure."""
    from langchain_core.messages import HumanMessage, SystemMessage
    from pydantic import BaseModel

    from cfd_langgraph.utils import structured_output

    class _Q(BaseModel):
        name: str
        definition: str

    class _Qs(BaseModel):
        quantities: List[_Q]

    metrics = "\n".join(
        f"- {m.get('name')}: {str(m.get('description', ''))[:300]} {str(m.get('computation_hint', ''))[:600]}"
        for m in metric_specs if m.get("name"))
    try:
        out = structured_output(llm, _Qs).invoke([
            SystemMessage(content=QUANTITY_SYSTEM),
            HumanMessage(content=f"CASE:\n{case_description.strip()[:3000]}\n\n"
                         f"TIME TREATMENT: {'transient -- use time-averaged fields' if transient else 'steady -- use the final time'}\n\n"
                         + (f"WHAT THIS CASE WRITES:\n{inventory.strip()[:6000]}\n\n" if inventory else "")
                         + f"STUDY METRICS:\n{metrics}"
                         + ("\n\nALREADY MEASURED FROM MONITORED HISTORIES (do not repeat):\n"
                            + "\n".join(f"- {m}" for m in monitored) if monitored else "")),
        ])
    except Exception:  # noqa: BLE001
        return []
    specs = []
    for q in out.quantities[:8]:
        name = _snake(q.name)
        if name:
            specs.append(quantity_spec(name, q.definition, transient=transient,
                                       from_inventory=bool(inventory)))
    return specs


# ------------------------------------------------------------ quantity review

QUANTITY_REVIEW_SYSTEM = """You review the quantities a mesh-independence check will \
compare between mesh levels, before any refinement is run. You see the flow on the \
first mesh, each proposed quantity with its definition and the value measured on that \
mesh, and the statistics of the histories the case records.

The check compares every quantity you keep between successive meshes and calls the \
mesh converged when each changes by no more than the tolerance. So keep only \
quantities that describe this flow and can settle under refinement. Drop a quantity, \
and say why, when what it is makes it unable to settle or uninformative:
- a maximum or minimum over the whole domain or along a wall: it lands where the \
solution is least resolved -- the cell touching a wall, or a corner or edge where \
boundary conditions meet and the exact solution has no finite value -- and moves with \
every refinement. Probe the fields to see where it sits if you are unsure;
- a number that is zero or near zero for this flow (by symmetry, or because it is a \
residual or an error estimate): a relative change of it means nothing;
- a value the measurement could not produce (shown as missing, with its reason);
- a history statistic of an output that is not a physical quantity of the flow \
(solver residuals, iteration counts, time-step sizes, continuity errors).
Never drop a quantity because of its value or because you expect it to change under \
refinement: finding that out is what the check is for. Each reason must be about what \
the quantity is.

You may add up to three quantities, defined from the solution fields alone, that \
capture what the study measures and are measured away from walls and corners: values \
at fixed interior points or stations along a line, the extremes of a profile across the \
interior and where they occur, the location and strength of a flow feature (a vortex \
centre, a separation or reattachment point), integrals over the domain or a boundary. \
Each must be well away from zero for this flow -- probe the fields to check -- because \
a small number turns any change into a large percentage. Pressure in an \
incompressible flow is fixed only up to its reference level, so use a pressure \
difference between two points, never the pressure at one point. Give each a snake_case \
name and a definition precise enough to compute. Never reference external data.

At least two quantities must remain."""


def review_quantities(llm: Any, case: Path | str, *, case_description: str,
                      quantities: List[Dict[str, Any]], values: Dict[str, Any],
                      history: Dict[str, float], images: List[Path]) -> Optional[Dict[str, Any]]:
    """Look at the first level's flow and the proposed gate quantities, and
    decide which to drop and what to add. None on failure."""
    import base64

    from langchain_core.messages import HumanMessage, SystemMessage
    from pydantic import BaseModel, Field

    from cfd_langgraph.utils import structured_output

    class _Drop(BaseModel):
        name: str
        reason: str = Field(description="What the quantity is that keeps it from settling")

    class _Add(BaseModel):
        name: str
        definition: str

    class _Review(BaseModel):
        drop: List[_Drop] = Field(default_factory=list)
        add: List[_Add] = Field(default_factory=list)
        reason: str = Field(description="What you saw in the flow and why the set that remains measures it")

    def _value(name: str) -> str:
        v = values.get(name)
        if isinstance(v, (int, float)):
            return f"{v:.6g}"
        why = values.get(f"{name}__why_null")
        return "missing" + (f" ({str(why)[:200]})" if why else "")

    field_lines = "\n".join(
        f"- {q['name']} = {_value(q['name'])}: {str(q.get('description', ''))[:400]}" for q in quantities)
    history_lines = "\n".join(f"- {k} = {v:.6g}" for k, v in sorted(history.items())) or "(none)"
    text = (
        f"CASE:\n{case_description.strip()[:3000]}\n\n"
        f"TIME TREATMENT: {'transient' if is_transient(case) else 'steady'}\n\n"
        f"PROPOSED FIELD QUANTITIES, with the value on this mesh:\n{field_lines}\n\n"
        f"HISTORY STATISTICS, also compared unless you drop them (name the entry, or its "
        f"output and column to drop all three statistics):\n{history_lines}\n\n"
        "The figures follow: the flow on this mesh."
    )
    content: List[Dict[str, Any]] = [{"type": "text", "text": text}]
    for img in images[:8]:
        content.append({"type": "text", "text": f"Figure: {Path(img).stem}"})
        content.append({"type": "image_url", "image_url": {
            "url": "data:image/png;base64," + base64.b64encode(Path(img).read_bytes()).decode()}})
    messages = [SystemMessage(content=QUANTITY_REVIEW_SYSTEM), HumanMessage(content=content)]
    check_dir = Path(images[0]).parent / "quantity_checks" if images else Path(case) / "quantity_checks"
    messages = _checked_messages(llm, Path(case), messages, check_dir)
    try:
        out = structured_output(llm, _Review).invoke(messages)
    except Exception:  # noqa: BLE001
        return None
    return {
        "drop": [{"name": d.name.strip(), "reason": d.reason.strip()} for d in out.drop if d.name.strip()],
        "add": [{"name": _snake(a.name), "definition": a.definition.strip()}
                for a in out.add[:3] if _snake(a.name) and a.definition.strip()],
        "reason": out.reason.strip(),
    }


def is_history_dropped(key: str, dropped: set) -> bool:
    """A history statistic is dropped by its own name or by its output and
    column (``forces.Cl`` drops ``forces.Cl.mean``, ``.fluctuation_rms`` and
    ``.dominant_frequency``)."""
    return key in dropped or key.rsplit(".", 1)[0] in dropped


def apply_quantity_review(specs: List[Dict[str, Any]], history_names: List[str],
                          review: Dict[str, Any], *, transient: bool,
                          min_keep: int = 2) -> tuple:
    """(field specs to measure, history names dropped, record of what was
    applied). A review that would leave fewer than ``min_keep`` quantities is
    not applied at all."""
    wanted = {str(d.get("name", "")).strip() for d in review.get("drop") or []}
    kept = [s for s in specs if s["name"] not in wanted]
    known = {s["name"] for s in specs}
    added = [quantity_spec(a["name"], a["definition"], transient=transient, from_inventory=True)
             for a in review.get("add") or [] if a["name"] not in known]
    hist_dropped = sorted({h for h in history_names if is_history_dropped(h, wanted)})
    remaining = len(kept) + len(added) + len(history_names) - len(hist_dropped)
    record = {
        "dropped": [d for d in review.get("drop") or []
                    if d.get("name") in known or any(is_history_dropped(h, {d.get("name")}) for h in history_names)],
        "added": [{"name": a["name"], "definition": a["description"]} for a in added],
        "reason": review.get("reason", ""),
    }
    if remaining < min_keep or not (kept or added):
        record = {"dropped": [], "added": [], "reason": review.get("reason", ""),
                  "not_applied": f"it would leave {remaining} quantities (fewer than {min_keep}) "
                                 "or no field quantity"}
        return specs, [], record
    return kept + added, hist_dropped, record


# ------------------------------------------------------------ pair verdict

GATE_TOLERANCE_PERCENT = float(os.environ.get("CFD_SCIENTIST_GATE_TOLERANCE") or 5.0)


def relative_changes(q_a: Dict[str, Any], q_b: Dict[str, Any], keys: List[str]) -> Dict[str, float]:
    """Percent change parent -> child of each key. The mean of a monitored
    history is measured against the larger of its size and its fluctuation,
    so an oscillation about zero is judged on its amplitude, not on a mean
    that is zero by symmetry."""
    out: Dict[str, float] = {}
    for k in keys:
        a, b = float(q_a[k]), float(q_b[k])
        scale = abs(a)
        if k.endswith(".mean"):
            fluct = q_a.get(k[:-len(".mean")] + ".fluctuation_rms")
            if isinstance(fluct, (int, float)) and math.isfinite(float(fluct)):
                scale = max(scale, abs(float(fluct)))
        out[k] = abs(b - a) / max(scale, 1e-12) * 100.0
    return out


def pair_verdict(pct: Dict[str, float], q_a: Dict[str, Any], q_b: Dict[str, Any],
                 tolerance: float = GATE_TOLERANCE_PERCENT) -> Dict[str, Any]:
    """Converged when every compared quantity changed by at most ``tolerance``
    percent; decided here, not by a model."""
    over = sorted((k for k, p in pct.items() if p > tolerance), key=lambda k: -pct[k])
    converged = bool(pct) and not over
    if not pct:
        reason = "No quantity could be compared between the two levels."
    elif converged:
        reason = f"All {len(pct)} quantities changed by at most {tolerance:g}%."
    else:
        reason = "Changed by more than {:g}%: ".format(tolerance) + "; ".join(
            f"{k} {pct[k]:.1f}% ({float(q_a[k]):.6g} -> {float(q_b[k]):.6g})" for k in over[:6])
    return {
        "converged": converged,
        "tolerance_percent": tolerance,
        "changes_percent": {k: round(p, 3) for k, p in sorted(pct.items())},
        "over_tolerance": over,
        "reason": reason,
    }


# ------------------------------------------------------------ refinement check

def _read_mesh(case: Path, *, solved: bool):
    """(internal mesh, {patch: mesh}) at the latest solved time, or at the
    initial time for a mesh that has not been solved yet."""
    import pyvista as pv

    from cfd_langgraph.foam_load import marker_for

    reader = pv.OpenFOAMReader(str(marker_for(case)))
    try:
        reader.skip_zero_time = False
    except Exception:  # noqa: BLE001
        pass
    times = [float(t) for t in (reader.time_values or [])]
    if times:
        reader.set_active_time_value(max(times) if solved else min(times))
    for switch in ("enable_all_patch_arrays", "enable_all_cell_arrays"):
        try:
            getattr(reader, switch)()
        except Exception:  # noqa: BLE001
            pass
    block = reader.read()
    internal = block["internalMesh"]
    patches = {}
    if "boundary" in list(block.keys()):
        for name in block["boundary"].keys():
            p = block["boundary"][name]
            if p is not None and p.n_cells:
                patches[name] = p
    return internal, patches


def _cell_length(mesh, axis: Optional[int]):
    """Linear cell size: sqrt(area) in a 2-D case, cube root of volume in 3-D."""
    import numpy as np

    vol = np.abs(np.asarray(mesh.compute_cell_sizes(length=False, area=False, volume=True).cell_data["Volume"]))
    if axis is None:
        return np.cbrt(vol)
    b = np.asarray(mesh.bounds).reshape(3, 2)
    thickness = max(b[axis, 1] - b[axis, 0], 1e-30)
    return np.sqrt(vol / thickness)


def _cell_extents(mesh, axis: Optional[int]):
    """Longest and shortest extent of each cell, along its own principal
    directions, in the flow plane of a 2-D case (all three axes in 3-D).
    Area alone hides a cell that got thinner one way and no finer the other."""
    import numpy as np

    pts_all = np.asarray(mesh.points)
    keep = [i for i in range(3) if i != axis] if axis is not None else [0, 1, 2]
    offsets = np.asarray(mesh.offset)
    conn = np.asarray(mesh.cell_connectivity)
    counts = np.diff(offsets)
    longest = np.zeros(mesh.n_cells)
    shortest = np.zeros(mesh.n_cells)
    for k in np.unique(counts):
        idx = np.where(counts == k)[0]
        cell_pts = pts_all[conn[offsets[idx][:, None] + np.arange(k)]][:, :, keep]
        centred = cell_pts - cell_pts.mean(axis=1, keepdims=True)
        cov = np.einsum("nki,nkj->nij", centred, centred)
        _, vecs = np.linalg.eigh(cov)
        proj = np.einsum("nki,nij->nkj", centred, vecs)
        span = proj.max(axis=1) - proj.min(axis=1)
        span = np.where(span < 1e-12 * max(span.max(), 1e-30), np.nan, span)
        longest[idx] = np.nanmax(span, axis=1)
        shortest[idx] = np.nanmin(span, axis=1)
    return longest, shortest


def _first_cell_distance(mesh, patch):
    import numpy as np

    faces = np.asarray(patch.cell_centers().points)
    normals = np.asarray(patch.compute_normals(cell_normals=True, point_normals=False,
                                               auto_orient_normals=False).cell_data["Normals"])
    centres = np.asarray(mesh.cell_centers().points)
    cell = np.asarray(mesh.find_closest_cell(faces))
    return np.abs(np.einsum("ij,ij->i", centres[cell] - faces, normals))


def compare_meshes(parent: Path | str, child: Path | str) -> str:
    """How a new mesh differs from the solved mesh it was refined from, in
    numbers: cell counts, local cell size where the parent's flow is active
    and over the whole domain, first-cell distance on each wall, and the new
    mesh's quality report."""
    import numpy as np

    from cfd_langgraph.foam_load import empty_axis, patch_types

    parent, child = Path(parent), Path(child)
    axis = empty_axis(parent)
    p_mesh, p_patches = _read_mesh(parent, solved=True)
    c_mesh, c_patches = _read_mesh(child, solved=False)
    lines = [f"cells: parent {p_mesh.n_cells}, new {c_mesh.n_cells} "
             f"(x{c_mesh.n_cells / max(p_mesh.n_cells, 1):.2f})"]

    p_pts = np.asarray(p_mesh.cell_centers().points)
    at_p = np.asarray(c_mesh.find_closest_cell(p_pts))
    p_long, p_short = _cell_extents(p_mesh, axis)
    c_long, c_short = _cell_extents(c_mesh, axis)
    ratios = {
        "cell size": _cell_length(c_mesh, axis)[at_p] / np.maximum(_cell_length(p_mesh, axis), 1e-30),
        "longest cell extent": c_long[at_p] / np.maximum(p_long, 1e-30),
        "shortest cell extent": c_short[at_p] / np.maximum(p_short, 1e-30),
    }

    def summary(mask, label):
        if not mask.any():
            return
        lines.append(f"{label} ({int(mask.sum())} parent cells), new/parent ratio at the same location:")
        for what, ratio in ratios.items():
            r = ratio[mask]
            r = r[np.isfinite(r)]
            if r.size == 0:
                continue
            lines.append(
                f"  {what}: median {np.median(r):.2f}, 10th pct {np.percentile(r, 10):.2f}, "
                f"90th pct {np.percentile(r, 90):.2f}, max {r.max():.2f}; "
                f"new coarser (> 1.05) in {100 * np.mean(r > 1.05):.0f}%")

    summary(np.ones(p_mesh.n_cells, bool), "whole domain")
    U = p_mesh.cell_data.get("U")
    if U is not None:
        speed = np.linalg.norm(np.asarray(U), axis=1)
        bulk = float(np.median(speed))
        summary(np.abs(speed - bulk) > 0.05 * max(abs(bulk), 1e-12),
                "where the parent's flow departs from the bulk by > 5%")

    for name, kind in patch_types(parent).items():
        if kind != "wall" or name not in p_patches or name not in c_patches:
            continue
        dp = _first_cell_distance(p_mesh, p_patches[name])
        dc = _first_cell_distance(c_mesh, c_patches[name])
        lines.append(f"wall '{name}': first-cell distance parent mean {dp.mean():.3g} (min {dp.min():.3g}), "
                     f"new mean {dc.mean():.3g} (min {dc.min():.3g}), ratio {dc.mean() / max(dp.mean(), 1e-30):.3g}; "
                     f"faces parent {len(dp)}, new {len(dc)}")

    log = child / "log.checkMesh"
    if log.is_file():
        keep = ("aspect ratio", "non-orthogonality", "skewness", "Failed", "***", "Mesh OK")
        quality = [l.strip() for l in log.read_text(errors="ignore").splitlines() if any(k in l for k in keep)]
        lines.append("new mesh checkMesh: " + " | ".join(quality[:8]))
    return "\n".join(lines)


REFINEMENT_SYSTEM = """You check one mesh refinement step of a CFD mesh-independence study \
before any solver time is spent on it.

You are given the instruction the mesh editor was asked to follow and numbers comparing \
the new mesh with the solved mesh it was derived from. A valid refinement step:
- makes cells smaller, or leaves them unchanged, wherever the parent's flow is active, and \
is not coarser there; the refinement should be substantial where the instruction asked for it, \
and in the directions it asked for -- a region to be refined in all directions needs both the \
longest and the shortest cell extent to drop there, not just the cell area;
- changes near-wall spacing only in the way the instruction asked (if it asked to keep the \
first-cell distance, it stays close; if it asked to reduce it, it goes down by roughly that \
factor, not by orders of magnitude);
- has no mesh-quality defects: no failed checks, no extreme non-orthogonality or skewness, \
no aspect ratios far beyond the parent's.

The total cell count and its budget are checked separately, before you see the mesh: do \
not judge them. Judge only where the cells went and how good they are.

If the step is not valid, write a correction for the mesh editor: what is wrong, in numbers, \
and what to change in the mesh definition (cell counts, grading) to fix it."""


def review_refinement(llm: Any, *, parent: Path | str, child: Path | str, instruction: str) -> Optional[str]:
    """None if the new mesh is a valid refinement of its parent, otherwise a
    correction for the mesh editor. A failed check returns None rather than
    blocking: the level review after solving still judges the result."""
    from langchain_core.messages import HumanMessage, SystemMessage
    from pydantic import BaseModel, Field

    from cfd_langgraph.utils import structured_output

    class _Check(BaseModel):
        valid: bool = Field(description="Is this a valid refinement step of the parent mesh?")
        problems: List[str] = Field(default_factory=list)
        correction: str = Field(default="", description="Instruction for the mesh editor if not valid")

    try:
        numbers = compare_meshes(parent, child)
    except Exception as exc:  # noqa: BLE001
        print(f"[mesh-gate] could not compare {Path(child).name} with its parent: {exc}", flush=True)
        return None
    print(f"[mesh-gate] {Path(child).name} vs {Path(parent).name}:\n{numbers}", flush=True)
    try:
        out = structured_output(llm, _Check).invoke([
            SystemMessage(content=REFINEMENT_SYSTEM),
            HumanMessage(content=f"INSTRUCTION GIVEN TO THE MESH EDITOR:\n{instruction.strip()[:3000]}\n\n"
                                 f"NEW MESH COMPARED WITH ITS PARENT:\n{numbers}"),
        ])
    except Exception:  # noqa: BLE001
        return None
    if out.valid:
        return None
    return ("The previous edit did not produce a valid refinement. "
            + " ".join(out.problems) + " " + out.correction + "\nMeasured:\n" + numbers)
