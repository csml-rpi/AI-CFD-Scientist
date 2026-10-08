"""Opening an OpenFOAM case for plotting.

Covers the details that decide whether a figure appears at all: the ``.foam``
marker, enabling field arrays, cell-to-point conversion, extracting
``internalMesh``, off-screen rendering, and refusing a case whose only time is
its initial condition.

Run: python scripts/test_foam_load.py [a solved case dir]
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cfd_langgraph.foam_load import (  # noqa: E402
    FoamLoadError, load_case, marker_for, new_plotter, solved_times,
)

FAILURES: list[str] = []


def check(name: str, cond: object, detail: str = "") -> None:
    if cond:
        print(f"[PASS] {name}")
    else:
        FAILURES.append(name)
        print(f"[FAIL] {name}" + (f" — {detail}" if detail else ""))


def _is_time(name: str) -> bool:
    try:
        float(name)
        return True
    except ValueError:
        return False


def find_case() -> Path | None:
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    root = Path("/mnt/sda1/somasn_cfd_scientist_experiments")
    if not root.is_dir():
        return None
    for pm in root.rglob("constant/polyMesh"):
        case = pm.parent.parent
        if any(d.is_dir() and d.name.isdigit() and int(d.name) > 0 for d in case.iterdir()):
            return case
    return None


case = find_case()
if case is None:
    print("no solved case available to test against; skipping")
    sys.exit(0)

tmp = Path(tempfile.mkdtemp())
work = tmp / "probe_case"
shutil.copytree(case, work, symlinks=True)
print(f"case under test: {work}  (copied from {case})")


print("== 1. the marker")
m = marker_for(work)
check("a .foam marker is created inside the case", m.is_file() and m.parent == work, str(m))
check("it is named after the case", m.name == f"{work.name}.foam", m.name)
check("passing the marker back is idempotent", marker_for(m) == m)
try:
    marker_for(tmp / "does_not_exist")
    check("a missing path raises", False)
except FoamLoadError as exc:
    check("a missing path raises, with the path named", "does_not_exist" in str(exc), str(exc))


print("== 2. times")
ts = solved_times(work)
check("solved times found", len(ts) > 0, str(ts))
check("time 0 is excluded — it is the initial condition", 0.0 not in ts, str(ts))


print("== 3. loading the latest solved time")
mesh = load_case(work)
check("a mesh comes back with cells", getattr(mesh, "n_cells", 0) > 0, str(getattr(mesh, "n_cells", None)))
check("it is the internal mesh, not the MultiBlock", not hasattr(mesh, "n_blocks"))
names = set(mesh.point_data.keys())
check("fields are enabled and present as POINT data (contourable)",
      "U" in names, str(sorted(names)[:8]))
check("mesh['U'] does not raise", mesh["U"] is not None)


print("== 4. explicit time selection")
mesh_first = load_case(work, time=ts[0])
check("an explicit solved time loads", getattr(mesh_first, "n_cells", 0) > 0)
try:
    load_case(work, time=123456.0)
    check("an absent time raises", False)
except FoamLoadError as exc:
    check("an absent time raises and lists what exists", "not among" in str(exc), str(exc)[:90])


print("== 5. a case that never advanced is an ERROR, not a picture of time 0")
stub = tmp / "never_ran"
stub.mkdir()
for sub in ("constant", "system", "0"):
    src = work / sub
    if src.is_dir():
        shutil.copytree(src, stub / sub, symlinks=True)
try:
    load_case(stub)
    check("it refuses", False, "returned a mesh for a case with only time 0")
except FoamLoadError as exc:
    check("it refuses", True)
    check("and says why in words a reader can act on",
          "did not advance" in str(exc), str(exc)[:120])


print("== 6. rendering with no display")
p = new_plotter(window_size=(400, 300))
p.add_mesh(mesh.slice(normal="z"), scalars="U")
png = tmp / "shot.png"
p.screenshot(str(png))
p.close()
check("a screenshot is written", png.is_file())
check("and it is a real image, not a blank stub",
      png.stat().st_size > 5000, f"{png.stat().st_size} bytes")


print("== 7. wall shear computed from the fields matches OpenFOAM's own")
from cfd_langgraph.foam_load import load_patches, patch_types, wall_shear  # noqa: E402

walls = {n for n, t in patch_types(case).items() if t == "wall"}
written = {n: p for n, p in load_patches(case).items()
           if n.split("/")[-1] in walls and "wallShearStress" in p.cell_data}
if not written:
    print("[SKIP] this case wrote no wallShearStress to compare against")
else:
    import numpy as np

    name, ref_patch = next(iter(written.items()))
    ref = np.asarray(ref_patch.cell_data["wallShearStress"])
    copy = tmp / "no_wss"
    latest = max(solved_times(case))
    time_dir = next(d for d in case.iterdir() if d.is_dir() and _is_time(d.name) and float(d.name) == latest)
    for sub in ("constant", "system"):
        shutil.copytree(case / sub, copy / sub, symlinks=True)
    shutil.copytree(time_dir, copy / time_dir.name, ignore=shutil.ignore_patterns("wallShearStress*"))
    got = wall_shear(copy).get(name)
    check("the wall patch is found", got is not None)
    if got is not None:
        mine = np.asarray(got.cell_data["wallShearStress"])
        ratio = np.linalg.norm(mine, axis=1).mean() / np.linalg.norm(ref, axis=1).mean()
        check("magnitude within 5%", abs(ratio - 1) < 0.05, f"ratio {ratio:.3f}")
        check("same direction", np.corrcoef(ref[:, 0], mine[:, 0])[0, 1] > 0.98)
    check("a written field is used as is",
          np.allclose(np.asarray(wall_shear(case)[name].cell_data["wallShearStress"]), ref))


print()
print("FAILURES:", FAILURES if FAILURES else "none")
shutil.rmtree(tmp, ignore_errors=True)
sys.exit(1 if FAILURES else 0)
