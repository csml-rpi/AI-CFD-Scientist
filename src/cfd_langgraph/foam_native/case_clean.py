"""Stale time-directory removal for OpenFOAM cases about to be solved.

Why this exists. A case directory that already holds time directories from an
*earlier* solve is not merely untidy — it silently corrupts the next run's
analysis. PyVista's ``OpenFOAMReader`` selects ``max(time_values)``, which is
what every QoI extractor in this repo is told to do ("Select the latest
simulation time"). If the case was re-meshed between runs, the leftover
directory belongs to the *old* mesh and its ``internalField`` length no longer
matches ``nCells``, so the reader attaches no fields at all and the extractor
reports every metric as null.

Measured, not hypothetical. In ``codex_sol56_none/cavity_r2`` the mesh gate's
baseline case carried times 100..1000 from a 4096-cell solve alongside the
converged answer at t=415 on the current 1024-cell mesh::

    415/U   internalField nonuniform List<vector> 1024   (matches polyMesh)
    1000/U  internalField nonuniform List<vector> 4096   (stale, old mesh)

The extractor read t=1000, found "fields found: none", returned null for both
requested metrics, and the gate refused to judge convergence — correctly, since
converging on a substituted quantity is worse than not converging. The run died
after 21 hours having solved the case perfectly well.

The fix belongs here, before the solver runs, not in a downstream guard that
detects the mismatch: a case that is about to be solved from its initial
condition has no business carrying another solve's output.

Restarts are the one legitimate exception, and they are explicit in the case
itself: ``startFrom latestTime``, or ``startFrom startTime`` with a non-zero
``startTime``, both mean "continue from fields already on disk". Those cases are
left alone.

Stdlib only, deliberately: the standalone runner scripts that need this logic
(``scripts/foam_run_simple.py``, ``scripts/code_mod_runtime.py``) resolve their
own environment and must keep importing cleanly inside the build sandbox.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import List, Optional

# Directories that are never a time directory, checked before the float parse
# so a case with an oddly-named folder does not get probed as one.
_NON_TIME_DIRS = {"constant", "system", "postProcessing", "VTK", "dynamicCode", "customModels", "lib", "bin"}


def parse_time_dir(name: str) -> Optional[float]:
    """The simulation time a directory name denotes, or None if it is not one.

    ``"0"`` -> 0.0, ``"0.05"`` -> 0.05, ``"1000"`` -> 1000.0, ``"constant"`` -> None.
    """
    if name in _NON_TIME_DIRS or name.startswith("processor") or name.startswith("."):
        return None
    try:
        return float(name)
    except ValueError:
        return None


def restart_start_time(case_dir: Path) -> Optional[float]:
    """The time this case intends to restart from, or None when it starts fresh.

    Returns a float for ``startFrom startTime`` with a non-zero ``startTime``,
    and ``float("inf")`` for ``startFrom latestTime`` (whatever is newest on
    disk, which we cannot name in advance but must not delete). ``None`` means
    the case starts from its initial condition and any non-zero time directory
    present is a leftover.
    """
    control = Path(case_dir) / "system" / "controlDict"
    if not control.is_file():
        # No controlDict is not a case we should be second-guessing; treat it
        # as a restart so nothing is deleted on a half-written directory.
        return float("inf")
    try:
        text = control.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return float("inf")
    # Strip comments so a commented-out `startFrom latestTime;` does not
    # protect a case that actually starts from zero.
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    text = re.sub(r"//[^\n]*", " ", text)
    m = re.search(r"\bstartFrom\s+(\w+)\s*;", text)
    start_from = (m.group(1) if m else "startTime").strip()
    if start_from == "latestTime":
        return float("inf")
    if start_from == "firstTime":
        return None
    m = re.search(r"\bstartTime\s+([-+0-9.eE]+)\s*;", text)
    if not m:
        return None
    try:
        value = float(m.group(1))
    except ValueError:
        return None
    return value if value != 0.0 else None


def remove_stale_time_dirs(case_dir: Path, *, force: bool = False) -> List[str]:
    """Delete time directories left by a previous solve; return their names.

    Keeps every directory whose time is 0 (``0``, ``0.000000``) because that is
    the initial condition, not output. Does nothing when the case declares a
    restart, unless ``force`` is set — used by the copy paths, which always
    want a candidate solved from its own initial condition rather than from a
    converged field that shipped with the starter.
    """
    case_dir = Path(case_dir)
    if not case_dir.is_dir():
        return []
    if not force and restart_start_time(case_dir) is not None:
        return []
    removed: List[str] = []
    for child in sorted(case_dir.iterdir()):
        if not child.is_dir() or child.is_symlink():
            continue
        t = parse_time_dir(child.name)
        if t is None or t == 0.0:
            continue
        shutil.rmtree(child, ignore_errors=True)
        removed.append(child.name)
    return removed


def copy_ignore_time_dirs(directory: str, names: List[str], case_root: Path) -> List[str]:
    """``shutil.copytree`` ignore hook dropping non-zero time dirs at the case root.

    Only at the root: a file named ``100`` nested inside ``constant/`` or a
    sampling directory under ``postProcessing/`` is not a time directory and
    must survive the copy.
    """
    try:
        if Path(directory).resolve() != Path(case_root).resolve():
            return []
    except OSError:
        return []
    out: List[str] = []
    for n in names:
        t = parse_time_dir(n)
        if t is not None and t != 0.0 and (Path(directory) / n).is_dir():
            out.append(n)
    return out
