#!/usr/bin/env python3
"""Build a registry solver for the installed OpenFOAM onto the shelf.

  python scripts/shelf_install.py <name> [--openfoam <WM_PROJECT_DIR>] [--jobs N]

Checks out the solver's pinned commit for this version into
<shelf>/builds/<name>/<fork>_<version>/src, builds it with its own build command
and $FOAM_USER_APPBIN / $FOAM_USER_LIBBIN pointed at that build's bin/ and lib/
(the OpenFOAM installation is never written to), inside a sandbox where only
the build folder is writable, then checks that every program and library the
registry promises exists, loads, and starts. The outcome is recorded in
shelf.json. A version the solver has no build for is recorded as needing a port.
See cfd_langgraph.solver_shelf.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from cfd_langgraph import solver_shelf as shelf  # noqa: E402
from cfd_langgraph.foam_native.openfoam_env import resolve_openfoam_env  # noqa: E402
from cfd_langgraph.openfoam_version import get_profile  # noqa: E402


def build_env(profile: Dict[str, Any], bdir: Path, jobs: int, extra: Dict[str, str],
              recipe_dir: str = "") -> Dict[str, str]:
    """The installation's environment with user build targets redirected into
    this build, and this build's own bin/ and lib/ found first."""
    env = dict(resolve_openfoam_env(profile["project_dir"]))
    for key in ("FOAM_USER_APPBIN", "FOAM_MODULE_APPBIN"):
        env[key] = str(bdir / "bin")
    for key in ("FOAM_USER_LIBBIN", "FOAM_MODULE_LIBBIN"):
        env[key] = str(bdir / "lib")
    env["PATH"] = f"{bdir / 'bin'}:{env.get('PATH', '')}"
    env["LD_LIBRARY_PATH"] = f"{bdir / 'lib'}:{env.get('LD_LIBRARY_PATH', '')}"
    env["WM_NCOMPPROCS"] = str(jobs)
    env.update({k: str(v) for k, v in extra.items()})
    env["SHELF_RECIPE_DIR"] = recipe_dir
    return env


def sandboxed(cmd: str, cwd: Path, writable: Path) -> List[str]:
    """Run ``cmd`` with the machine read-only except ``writable`` and a private
    /tmp: a third-party build script cannot touch anything else."""
    bwrap = shutil.which("bwrap")
    inner = ["bash", "-c", cmd]
    if not bwrap:
        return inner
    return [bwrap, "--die-with-parent", "--ro-bind", "/", "/", "--tmpfs", "/tmp",
            "--bind", str(writable), str(writable), "--proc", "/proc", "--dev", "/dev",
            "--chdir", str(cwd), *inner]


def checkout(solver: Dict[str, Any], ref: str, dest: Path) -> None:
    src = shelf.source_dir(solver["name"])
    if not (src / ".git").is_dir():
        src.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "-q", solver["repo"], str(src)], check=True)
    if dest.exists():
        shutil.rmtree(dest)
    subprocess.run(["git", "clone", "-q", "--no-hardlinks", str(src), str(dest)], check=True)
    subprocess.run(["git", "-C", str(dest), "checkout", "-q", ref], check=True)


def _missing_libs(path: Path, env: Dict[str, str]) -> List[str]:
    ldd = subprocess.run(["ldd", str(path)], capture_output=True, text=True, env=env)
    return [line.split()[0] for line in ldd.stdout.splitlines() if "not found" in line]


def verify(solver: Dict[str, Any], bdir: Path, env: Dict[str, str]) -> Dict[str, Any]:
    """Every promised program and library exists, resolves its libraries, and
    the programs start."""
    provides = solver.get("provides") or {}
    found: Dict[str, Any] = {"applications": {}, "libraries": {}}
    problems: List[str] = []
    for app in provides.get("applications") or []:
        path = bdir / "bin" / app
        if not path.is_file():
            problems.append(f"program {app} was not built")
            continue
        missing = _missing_libs(path, env)
        if missing:
            problems.append(f"{app} cannot load {missing}")
        # How a program shows it starts: its own `-help`, or the registry's
        # command for one that takes no such flag (a wrapper, a script).
        command = (provides.get("start") or {}).get(app)
        start = subprocess.run(["bash", "-c", command] if command else [str(path), "-help"],
                               capture_output=True, text=True, env=env, timeout=600)
        if start.returncode != 0:
            problems.append(f"{app} -help exited {start.returncode}: {start.stderr[-300:]}")
        found["applications"][app] = {"path": str(path), "starts": start.returncode == 0, "missing_libs": missing}
    for lib in provides.get("libraries") or []:
        path = bdir / "lib" / f"{lib}.so"
        if not path.is_file():
            problems.append(f"library {lib} was not built")
            continue
        missing = _missing_libs(path, env)
        if missing:
            problems.append(f"{lib} cannot load {missing}")
        found["libraries"][lib] = {"path": str(path), "missing_libs": missing}
    problems += clashes(bdir, env)
    return {"ok": not problems, "problems": problems, "found": found}


def clashes(bdir: Path, env: Dict[str, str]) -> List[str]:
    """Libraries of this build named like one of the installation's own. The
    build's lib/ goes on every case's library path, so such a library would
    replace the stock one in cases that never use this solver."""
    stock_dir = Path(env.get("FOAM_LIBBIN", ""))
    if not stock_dir.is_dir() or not (bdir / "lib").is_dir():
        return []
    stock = {p.name for p in stock_dir.rglob("*.so")}
    return [f"library {p.relative_to(bdir)} has the name of an OpenFOAM library ({p.name}) and would "
            f"replace it in every case; the build must give it its own name"
            for p in sorted((bdir / "lib").rglob("*.so")) if p.name in stock]


def install(name: str, openfoam_path: str, jobs: int, timeout: int) -> Dict[str, Any]:
    profile = get_profile(openfoam_path)
    if not profile.get("ok"):
        raise SystemExit(f"could not read the OpenFOAM installation at {openfoam_path!r}")
    solver = shelf.spec(name)
    name = solver["name"]
    entry = shelf.build_entry(solver, profile)
    bdir = shelf.build_dir(name, profile)
    bdir.mkdir(parents=True, exist_ok=True)
    if not entry:
        return shelf.write_record(name, profile, status="failed", reason="the registry lists no build")
    if entry["match"] != "exact":
        return shelf.write_record(
            name, profile, status="needs_port", ref=entry.get("ref"), port_from=entry.get("versions"),
            reason=f"{name} has no build for {profile['label']}; the nearest is {entry.get('fork')} "
                   f"{entry.get('versions')}")
    src = bdir / "src"
    checkout(solver, str(entry["ref"]), src)
    for d in ("bin", "lib"):
        (bdir / d).mkdir(exist_ok=True)
    env = build_env(profile, bdir, jobs, entry.get("env") or {}, recipe_dir=solver["_dir"])
    log = bdir / "build.log"
    with log.open("w") as handle:
        try:
            proc = subprocess.run(sandboxed(str(entry["build"]), src, bdir), cwd=str(src), env=env,
                                  stdout=handle, stderr=subprocess.STDOUT, timeout=timeout)
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            rc = -9
    tail = log.read_text(errors="replace")[-3000:]
    check = verify(solver, bdir, env)
    return shelf.write_record(
        name, profile, status="built" if check["ok"] else "failed", version=profile["label"],
        ref=entry["ref"], build=entry["build"], build_rc=rc, build_log=str(log),
        build_log_tail=tail[-1500:], provides=check["found"], problems=check["problems"], checks=[])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("name")
    ap.add_argument("--openfoam", default="", help="WM_PROJECT_DIR (default: the configured installation)")
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--timeout", type=int, default=4 * 3600)
    a = ap.parse_args()
    doc = install(a.name, a.openfoam, a.jobs, a.timeout)
    print(json.dumps({k: doc.get(k) for k in ("status", "version", "ref", "build_rc", "problems", "reason")},
                     indent=2))
    return 0 if doc.get("status") == "built" else 1


if __name__ == "__main__":
    raise SystemExit(main())
