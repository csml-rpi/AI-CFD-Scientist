from __future__ import annotations

import functools
import os
import subprocess
from typing import Dict


@functools.lru_cache(maxsize=8)
def resolve_openfoam_env(openfoam_path: str) -> Dict[str, str]:
    """A full subprocess environment with OpenFOAM's ``etc/bashrc`` sourced.

    ``Allrun``'s own ``. "$WM_PROJECT_DIR/bin/tools/RunFunctions"`` line
    needs ``WM_PROJECT_DIR`` just to source, and ``RunFunctions`` alone
    doesn't put solver binaries (``simpleFoam``, ``blockMesh``, ...) on
    ``PATH`` — that's ``etc/bashrc``'s job. Every subprocess this workflow
    spawns (this package's own ``run_foam_case``, and ``scripts/foam_run.py``
    via the manager's ``_foamagent_env``) previously assumed the launching
    shell had already sourced OpenFOAM — true for some setups, silently
    false for others, and the failure mode is an immediate, confusing
    Allrun failure with no clear cause. This sources it explicitly, once per
    ``openfoam_path`` per process (``lru_cache`` — ``etc/bashrc`` doesn't
    change mid-run), and returns a full environment dict any caller can pass
    straight to ``subprocess.run(..., env=...)``.

    Falls back to this process's own unmodified environment if sourcing
    fails (wrong path, missing install) — callers still get a usable dict,
    just without OpenFOAM on PATH, so the eventual failure is a clear
    "command not found" rather than a confusing error from *this* function.
    """
    base = os.environ.copy()
    if not openfoam_path:
        return base
    try:
        proc = subprocess.run(
            ["bash", "-c", f"source {openfoam_path}/etc/bashrc 2>/dev/null && env -0"],
            capture_output=True,
            timeout=60,
        )
        if proc.returncode != 0 or not proc.stdout:
            return base
        resolved: Dict[str, str] = {}
        for entry in proc.stdout.split(b"\x00"):
            if not entry:
                continue
            key, _, value = entry.partition(b"=")
            try:
                resolved[key.decode()] = value.decode()
            except UnicodeDecodeError:
                continue
        return resolved or base
    except Exception:
        return base
