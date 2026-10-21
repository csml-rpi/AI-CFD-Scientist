"""Make a starter's OpenFOAM cases run on the installation the study has.

A starter folder ships cases written for whatever OpenFOAM its author used, and
the user may have another: an openfoam.org 10 case does not run on openfoam.com
(no turbulenceProperties, no transportProperties), and one written for v2312
does not run on openfoam.org 13. Whether a case needs changing is not decided
from its banner or its file names. It is decided by trying it on the user's
installation: every dictionary must be read by its foamDictionary, the mesh must
build and pass checkMesh, and the solver must run a few steps.

A case that passes is used as it is. One that fails is copied and handed to an
agent (scripts/translate_case.py) that changes only what the installed version
requires, with the installation's own tutorials for the same solver as the
reference, records every change, and may not finish until the same checks pass.
The starter itself is never modified. scripts/port_closure_cases.py is the
hand-written, single-benchmark predecessor of this.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from cfd_langgraph.openfoam_version import case_solver, get_profile, run_command

# What a case is: its input folders and the scripts beside them. Solved time
# folders, processor folders, logs and post-processing output are not copied.
_INPUT_DIRS = ("0", "0.orig", "0.org", "constant", "system")
_MAX_DICT_BYTES = 2_000_000


def copy_inputs(case_dir: Path, dest: Path) -> Path:
    """Copy a case's inputs (``0``, ``0.orig``, ``constant``, ``system`` and
    top-level scripts) to ``dest``; an existing mesh comes with ``constant``."""
    case_dir, dest = Path(case_dir), Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for name in _INPUT_DIRS:
        if (case_dir / name).is_dir():
            shutil.copytree(case_dir / name, dest / name, dirs_exist_ok=True, symlinks=True)
    for p in case_dir.iterdir():
        if p.is_file() and (p.name.startswith("All") or p.suffix in (".sh", ".md", ".txt")):
            shutil.copy2(p, dest / p.name)
    return dest


def _run(cmd: str, cwd: Path, env: Dict[str, str], timeout: int) -> Dict[str, Any]:
    try:
        proc = subprocess.run(["bash", "-c", cmd], cwd=str(cwd), env=env, capture_output=True,
                              text=True, timeout=timeout)
        return {"rc": proc.returncode, "out": proc.stdout[-4000:], "err": proc.stderr[-4000:]}
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout if isinstance(exc.stdout, str) else ""
        return {"rc": -9, "out": out[-4000:], "err": f"timed out after {timeout}s"}


def _dictionaries(case_dir: Path) -> List[Path]:
    """The case's text dictionaries and field files, skipping mesh data."""
    out: List[Path] = []
    for name in _INPUT_DIRS:
        folder = case_dir / name
        if not folder.is_dir():
            continue
        for p in sorted(folder.rglob("*")):
            if not p.is_file() or "polyMesh" in p.parts or "triSurface" in p.parts:
                continue
            try:
                if p.stat().st_size > _MAX_DICT_BYTES:
                    continue
                head = p.read_bytes()[:4000]
            except OSError:
                continue
            if b"FoamFile" in head:
                out.append(p)
    return out


def _first_error(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for i, ln in enumerate(lines):
        if "FOAM FATAL" in ln or "Error" in ln or "error" in ln:
            return " | ".join(lines[i:i + 6])[:900]
    return " | ".join(lines[-6:])[:900]


def with_paths(env: Dict[str, str], extra_path: str = "", extra_lib: str = "") -> Dict[str, str]:
    """``env`` with folders put first on PATH and LD_LIBRARY_PATH (a shelf
    solver's bin/ and lib/, say)."""
    env = dict(env)
    if extra_path:
        env["PATH"] = f"{extra_path}:{env.get('PATH', '')}"
    if extra_lib:
        env["LD_LIBRARY_PATH"] = f"{extra_lib}:{env.get('LD_LIBRARY_PATH', '')}"
    return env


def check_case(case_dir: Path, *, openfoam_path: str = "", steps: int = 3, timeout: int = 900,
               extra_path: str = "", extra_lib: str = "") -> Dict[str, Any]:
    """Whether a case runs on the installation at ``openfoam_path``, tried on a
    scratch copy: dictionaries read, mesh built and checked, ``steps`` solver
    steps run. ``extra_path``/``extra_lib`` put a third-party solver's programs
    and libraries on the search paths. Returns {ok, stage, problems, warnings,
    solver, ...}."""
    from cfd_langgraph.solver_shelf import run_env as case_env

    profile = get_profile(openfoam_path)
    env = with_paths(case_env(profile.get("project_dir", openfoam_path)), extra_path, extra_lib)
    result: Dict[str, Any] = {"ok": False, "version": profile.get("label"), "problems": [], "warnings": []}
    with tempfile.TemporaryDirectory(prefix="case_check_") as td:
        work = copy_inputs(case_dir, Path(td) / "case")
        for initial in ("0.orig", "0.org"):
            if not (work / "0").is_dir() and (work / initial).is_dir():
                shutil.copytree(work / initial, work / "0")

        for p in _dictionaries(work):
            r = _run(f'foamDictionary "{p}" > /dev/null', work, env, 120)
            if r["rc"] != 0:
                result["problems"].append(f"{p.relative_to(work)}: {_first_error(r['err'] + r['out'])}")
        if result["problems"]:
            result["stage"] = "dictionaries"
            return result

        solver = case_solver(work, profile)
        command = "foamRun" if solver["application"] == "foamRun" else run_command(profile, solver["application"])
        result["solver"] = command
        if not command:
            result.update(stage="solver", problems=["system/controlDict names no application or solver"])
            return result
        _shorten(work, env, steps)

        mesh = work / "constant" / "polyMesh"
        has_mesh = (mesh / "points").exists() or (mesh / "points.gz").exists()
        simple_mesh = (work / "system" / "blockMeshDict").is_file() and not (work / "system" / "snappyHexMeshDict").is_file()
        if not has_mesh and not simple_mesh:
            # The case builds its mesh with its own script: run that script,
            # which now stops after a few steps, and judge it by its logs.
            if not (work / "Allrun").is_file():
                result.update(stage="mesh", problems=[
                    "the case has no mesh and no Allrun to build one with; it cannot be tried"],
                    unverifiable=True)
                return result
            r = _run("chmod +x Allrun && ./Allrun > log.Allrun 2>&1", work, env, timeout)
            return _judge_logs(work, r, result, command)
        if not has_mesh:
            r = _run("blockMesh > log.blockMesh 2>&1", work, env, timeout)
            if r["rc"] != 0:
                log = (work / "log.blockMesh").read_text(errors="replace")
                result.update(stage="mesh", problems=[f"blockMesh: {_first_error(log)}"])
                return result
        r = _run("checkMesh > log.checkMesh 2>&1", work, env, timeout)
        mesh_log = (work / "log.checkMesh").read_text(errors="replace")
        if r["rc"] != 0:
            result.update(stage="mesh", problems=[f"checkMesh: {_first_error(mesh_log)}"])
            return result
        result["mesh_ok"] = "Mesh OK" in mesh_log

        r = _run(f"{command} > log.solver 2>&1", work, env, timeout)
        log = (work / "log.solver").read_text(errors="replace")
        result["warnings"] = _warnings(log)
        if r["rc"] == -9 and "Time = " in log and "FOAM FATAL" not in log:
            result.update(ok=True, stage="passed", note=f"stopped after {timeout}s while still stepping")
            return result
        if r["rc"] != 0 or not log.rstrip().endswith(("End", "Finalising parallel run")):
            result.update(stage="solver", problems=[f"{command}: {_first_error(log)}"])
            return result
        result.update(ok=True, stage="passed")
    return result


def _shorten(work: Path, env: Dict[str, str], steps: int) -> None:
    """Stop the case ``steps`` time steps after it starts, and write nothing."""
    start = _run("foamDictionary -entry startTime -value system/controlDict", work, env, 60)["out"].strip()
    delta = _run("foamDictionary -entry deltaT -value system/controlDict", work, env, 60)["out"].strip()
    try:
        end = float(start or 0) + steps * float(delta or 1)
    except ValueError:
        end = float(steps)
    _run(f"foamDictionary -entry endTime -set {end:g} system/controlDict && "
         f"foamDictionary -entry writeControl -set timeStep system/controlDict && "
         f"foamDictionary -entry writeInterval -set {steps * 1000} system/controlDict", work, env, 60)


def _warnings(log: str) -> List[str]:
    return [ln.strip() for ln in log.splitlines()
            if "warning" in ln.lower() or "deprecated" in ln.lower() or "not used" in ln.lower()][:12]


def _judge_logs(work: Path, run: Dict[str, Any], result: Dict[str, Any], command: str) -> Dict[str, Any]:
    """The verdict on a case run through its own Allrun: no fatal error in any
    log, and the solver's log ends normally (or was still stepping at the cut)."""
    logs = {p.name: p.read_text(errors="replace") for p in work.glob("log.*")}
    fatal = [f"{name}: {_first_error(text)}" for name, text in logs.items() if "FOAM FATAL" in text]
    solver_log = logs.get(f"log.{command.split()[0]}", "")
    result["warnings"] = _warnings(solver_log)
    if fatal:
        result.update(stage="allrun", problems=fatal[:4])
    elif not solver_log:
        result.update(stage="allrun", problems=[f"Allrun ran but wrote no log.{command.split()[0]}: "
                                                 + _first_error(logs.get("log.Allrun", ""))])
    elif solver_log.rstrip().endswith(("End", "Finalising parallel run")) or (run["rc"] == -9 and "Time = " in solver_log):
        result.update(ok=True, stage="passed")
    else:
        result.update(stage="solver", problems=[f"{command}: {_first_error(solver_log)}"])
    return result


def translated_path(out_dir: Path, case_dir: Path, starter_dir: Optional[Path]) -> Path:
    """Where a starter case's translated copy lives in the study folder."""
    case_dir = Path(case_dir).resolve()
    try:
        rel = case_dir.relative_to(Path(starter_dir).resolve()) if starter_dir else Path(case_dir.name)
    except ValueError:
        rel = Path(case_dir.name)
    return Path(out_dir) / "starter_translated" / rel
