"""What an OpenFOAM case actually contains, for anything planning work on it.

A visualisation plan built from time folders and field names alone can ask for
data the case never wrote -- probe histories for a case with no probes, a wall
quantity read from the interior where it is zero. The plan is then impossible
to satisfy, the reviewer marks the figures incomplete, and the retry loop
regenerates the same gap until it runs out of attempts.

So: report the solved times, the fields and where each one lives, the
boundaries and their types, what the function objects actually wrote and under
which column names, and the plane a 2-D case lies in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class Inventory:
    case: str
    times: List[str] = field(default_factory=list)
    internal_fields: List[str] = field(default_factory=list)
    patches: Dict[str, str] = field(default_factory=dict)
    wall_fields: Dict[str, List[str]] = field(default_factory=dict)
    post_processing: Dict[str, List[str]] = field(default_factory=dict)
    columns: Dict[str, List[str]] = field(default_factory=dict)
    bounds: Optional[tuple] = None
    plane_axis: Optional[int] = None
    note: str = ""

    def describe(self) -> str:
        """The inventory as text for a prompt, saying what exists and where."""
        axis_name = {0: "x", 1: "y", 2: "z"}.get(self.plane_axis or -1)
        lines = [f"SOLVED TIMES: {', '.join(self.times) or 'none'}"]
        if self.bounds:
            lo, hi = self.bounds[0::2], self.bounds[1::2]
            lines.append("DOMAIN BOUNDS: x {:.4g}..{:.4g}, y {:.4g}..{:.4g}, z {:.4g}..{:.4g}".format(
                lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]))
        if axis_name:
            lines.append(
                f"THIS IS A 2-D CASE, one cell thick along {axis_name}. Plot the plane "
                f"perpendicular to {axis_name}; any other view shows the geometry edge-on.")
        lines.append(f"FIELDS ON THE INTERNAL MESH: {', '.join(self.internal_fields) or 'none'}")
        if self.patches:
            lines.append("BOUNDARIES: " + ", ".join(f"{n} ({t})" for n, t in self.patches.items()))
        for patch, names in self.wall_fields.items():
            lines.append(f"FIELDS ON BOUNDARY {patch}: {', '.join(names) or 'none'} "
                         f"-- read wall quantities here, not from the interior, where they are ~0")
        if self.post_processing:
            for name, files in self.post_processing.items():
                cols = self.columns.get(name)
                suffix = f"; columns: {', '.join(cols)}" if cols else ""
                lines.append(f"FUNCTION OBJECT OUTPUT '{name}': {', '.join(files[:4])}{suffix}")
        else:
            lines.append("FUNCTION OBJECT OUTPUT: none -- the case wrote no postProcessing data, "
                         "so there are no probe histories or sampled sets to plot")
        if self.note:
            lines.append(f"NOTE: {self.note}")
        return "\n".join(lines)


def _read_columns(path: Path) -> List[str]:
    """Column names from a function-object data file's last header line."""
    try:
        header = ""
        with path.open("r", errors="ignore") as fh:
            for line in fh:
                if line.startswith("#"):
                    header = line
                    continue
                break
        return [c for c in header.lstrip("#").split() if c]
    except OSError:
        return []


def collect(case: Path | str) -> Inventory:
    case = Path(case)
    inv = Inventory(case=str(case))
    if not case.is_dir():
        inv.note = "case directory not found"
        return inv

    for d in sorted((d for d in case.iterdir() if d.is_dir()), key=lambda d: d.name):
        try:
            value = float(d.name)
        except ValueError:
            continue
        if value > 0:
            inv.times.append(d.name)
    inv.times.sort(key=float)

    latest = case / inv.times[-1] if inv.times else case / "0"
    if latest.is_dir():
        inv.internal_fields = sorted(f.name for f in latest.iterdir() if f.is_file())

    boundary = case / "constant" / "polyMesh" / "boundary"
    if boundary.is_file():
        name = None
        for line in boundary.read_text(errors="ignore").splitlines():
            t = line.strip()
            if t and not t.startswith(("{", "}", "//", "(", ")")) and " " not in t and not t.endswith(";"):
                name = t
            elif t.startswith("type") and name:
                inv.patches[name] = t.split()[-1].rstrip(";")
                name = None

    post = case / "postProcessing"
    if post.is_dir():
        for d in sorted(post.iterdir()):
            if not d.is_dir():
                continue
            files = sorted(f.name for f in d.rglob("*") if f.is_file())
            if files:
                inv.post_processing[d.name] = files
                sample = next((f for f in d.rglob("*") if f.is_file() and f.suffix in (".dat", ".csv", ".xy")), None)
                if sample is not None:
                    cols = _read_columns(sample)
                    if cols:
                        inv.columns[d.name] = cols

    try:
        from cfd_langgraph.foam_load import empty_axis, load_case, load_patches

        mesh = load_case(case)
        inv.bounds = tuple(float(b) for b in mesh.bounds)
        inv.plane_axis = empty_axis(case)
        for patch_name, patch in load_patches(case).items():
            if inv.patches.get(patch_name) == "wall":
                inv.wall_fields[patch_name] = sorted(set(patch.point_data.keys()) | set(patch.cell_data.keys()))
    except Exception as exc:  # noqa: BLE001 -- inventory is best-effort
        inv.note = f"mesh could not be read for bounds/patch fields: {type(exc).__name__}: {exc}"
    return inv
