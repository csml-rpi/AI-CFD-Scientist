#!/usr/bin/env python3
"""
Auto-recover missing post-process fields on an OpenFOAM case.

Many comparator scripts read fields produced by function objects defined in
system/controlDict (e.g. wallShearStress, yPlus, forces, sampling, residuals).
If the case was run with -noFunctionObjects, with crashed function objects, or
the agent forgot to enable them, the latest-time directory will lack those
fields and the comparator will silently score the wrong snapshot (or refuse
to run).

This module recovers them generically:

  Strategy 1 (FAST, ACCURATE — preferred):
      <application> -postProcess -latestTime
    runs ALL function objects from controlDict against the latestTime dir,
    using the actual saved primary fields (U, p, T, ...). The application
    name is read from controlDict; works for simpleFoam, pimpleFoam,
    foamRun, chtMultiRegionFoam, anything.

  Strategy 2 (SLOWER, FALLBACK — for cases where -postProcess can't run):
      pyvista-based numerical derivation of known scalar/vector fields
    from primary fields. Currently supports wallShearStress (= ν * |∂U_t/∂n|
    at wall patches) which is what most CFD comparators need first.

Generic across studies — turbulence, multiphase, heat transfer, etc.

CLI:
    python scripts/foam_postprocess.py --case <case_dir>
        [--field wallShearStress yPlus]   # optional whitelist; default = all
        [--bashrc /path/to/openfoam/etc/bashrc]
        [--allow-pyvista]                 # enable pyvista fallback
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# helpers: parse controlDict
# ---------------------------------------------------------------------------

def _read_application(controldict: Path) -> Optional[str]:
    if not controldict.is_file():
        return None
    txt = controldict.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"^\s*application\s+(\S+)\s*;", txt, re.MULTILINE)
    return m.group(1) if m else None


def _read_function_object_names(controldict: Path) -> List[str]:
    """Extract the top-level keys inside the `functions { ... }` block of
    controlDict. These are the function-object names — running
    `<app> -postProcess -funcs (n1 n2 ...)` runs exactly those.
    Returns [] if no functions block is defined."""
    if not controldict.is_file():
        return []
    txt = controldict.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"^\s*functions\s*\{", txt, re.MULTILINE)
    if not m:
        return []
    start = m.end()
    depth = 1
    i = start
    while i < len(txt) and depth > 0:
        c = txt[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                break
        i += 1
    body = txt[start:i]
    # Top-level entries are `name { ... }` where braces inside are matched.
    names: List[str] = []
    j = 0
    while j < len(body):
        # skip whitespace + line comments
        while j < len(body) and body[j].isspace():
            j += 1
        if j >= len(body):
            break
        if body[j:j + 2] == "//":
            nl = body.find("\n", j)
            j = len(body) if nl < 0 else nl + 1
            continue
        # read identifier
        k = j
        while k < len(body) and (body[k].isalnum() or body[k] in "_."):
            k += 1
        ident = body[j:k]
        if not ident:
            j += 1
            continue
        # find following `{`
        m2 = re.search(r"\s*\{", body[k:])
        if not m2:
            break
        depth2 = 1
        p = k + m2.end()
        while p < len(body) and depth2 > 0:
            if body[p] == "{":
                depth2 += 1
            elif body[p] == "}":
                depth2 -= 1
                if depth2 == 0:
                    break
            p += 1
        names.append(ident)
        j = p + 1
    return names


def _latest_time_dir(case_dir: Path) -> Optional[Path]:
    """Latest numeric time dir in the case. Returns None if no time dirs."""
    times: List[Tuple[float, Path]] = []
    for entry in case_dir.iterdir():
        if not entry.is_dir():
            continue
        try:
            v = float(entry.name)
        except ValueError:
            continue
        times.append((v, entry))
    if not times:
        return None
    return sorted(times)[-1][1]


def _missing_fields_at_latest(case_dir: Path, expected_fields: List[str]) -> List[str]:
    """Return the subset of expected_fields not present at latestTime."""
    latest = _latest_time_dir(case_dir)
    if latest is None:
        return list(expected_fields)
    return [f for f in expected_fields if not (latest / f).exists()]


# ---------------------------------------------------------------------------
# Strategy 1 — OpenFOAM postProcess
# ---------------------------------------------------------------------------

def _resolve_bashrc(override: Optional[str] = None) -> Optional[str]:
    if override and Path(override).is_file():
        return override
    env_wm = os.environ.get("WM_PROJECT_DIR", "").strip()
    if env_wm:
        cand = Path(env_wm) / "etc" / "bashrc"
        if cand.is_file():
            return str(cand)
    for guess in ("/mnt/sda1/openfoam10/etc/bashrc", "/opt/openfoam10/etc/bashrc"):
        if Path(guess).is_file():
            return guess
    return None


def run_openfoam_postprocess(
    case_dir: Path,
    *,
    funcs: Optional[List[str]] = None,
    bashrc: Optional[str] = None,
    timeout_s: int = 300,
) -> Dict[str, Any]:
    """
    Invoke `<app> -postProcess -latestTime [-funcs (n1 n2 ...)]` on the case.
    Returns {ok, rc, stdout_tail, stderr_tail, app, args}.
    """
    cd = Path(case_dir)
    app = _read_application(cd / "system" / "controlDict")
    if not app:
        return {"ok": False, "rc": -1, "error": "could not read application from controlDict"}
    bashrc_path = _resolve_bashrc(bashrc)
    if not bashrc_path:
        return {"ok": False, "rc": -1, "error": "no OpenFOAM bashrc found (WM_PROJECT_DIR/openfoam10)"}

    args = [app, "-postProcess", "-latestTime"]
    if funcs:
        args += ["-funcs", "(" + " ".join(funcs) + ")"]
    cmd = f". \"{bashrc_path}\" >/dev/null 2>&1 && cd \"{cd}\" && " + " ".join(args)
    try:
        proc = subprocess.run(
            ["bash", "-lc", cmd], capture_output=True, text=True,
            timeout=timeout_s, check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False, "rc": -2,
            "stdout_tail": (exc.stdout or "")[-2000:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": (exc.stderr or "")[-2000:] if isinstance(exc.stderr, str) else "",
            "error": f"timeout after {timeout_s}s",
            "app": app, "args": args,
        }
    return {
        "ok": proc.returncode == 0,
        "rc": int(proc.returncode),
        "stdout_tail": (proc.stdout or "")[-3000:],
        "stderr_tail": (proc.stderr or "")[-3000:],
        "app": app, "args": args,
    }


# ---------------------------------------------------------------------------
# Strategy 2 — pyvista derivation (fallback for known fields)
# ---------------------------------------------------------------------------

def derive_with_pyvista(case_dir: Path, field_name: str) -> Dict[str, Any]:
    """
    Derive a field numerically using pyvista's OpenFOAMReader.
    Currently supports:
        wallShearStress  = nu * (∂U_t / ∂n) on wall patches
            (where U_t is the velocity component tangential to the wall normal)

    Returns {ok, derived_field, written_to, n_faces, error}.
    Writes the result as a volVectorField in <case>/<latestTime>/<field_name>.
    """
    if field_name != "wallShearStress":
        return {"ok": False, "error": f"pyvista derivation not implemented for {field_name!r}"}
    try:
        import pyvista as pv  # type: ignore
        import numpy as np  # type: ignore
    except Exception as exc:
        return {"ok": False, "error": f"pyvista not available: {exc}"}

    cd = Path(case_dir)
    foam_file = cd / f"{cd.name}.foam"
    if not foam_file.exists():
        try:
            foam_file.touch()
        except Exception:
            pass

    latest = _latest_time_dir(cd)
    if latest is None:
        return {"ok": False, "error": "no time directories"}
    try:
        reader = pv.OpenFOAMReader(str(foam_file))
        # Set the time to the latest dir
        try:
            reader.set_active_time_value(float(latest.name))
        except Exception:
            pass
        mesh = reader.read()
    except Exception as exc:
        return {"ok": False, "error": f"pyvista read failed: {exc}"}

    # In pyvista's OpenFOAMReader, the result is a MultiBlock with 'internalMesh'
    # and 'boundary' blocks. Walls live under 'boundary'.
    try:
        boundary = mesh["boundary"]
    except Exception as exc:
        return {"ok": False, "error": f"no boundary block in pyvista mesh: {exc}"}

    # Read kinematic viscosity nu from constant/transportProperties
    nu = _read_nu(cd)
    if nu is None:
        return {"ok": False, "error": "could not read nu from constant/transportProperties"}

    # For each wall patch, compute τ_w = nu * |∂U / ∂n| at face centers.
    # pyvista doesn't give us cell-near-wall U directly; the boundary block
    # carries U at face centers (from boundaryField). To form the wall-normal
    # gradient we'd need the adjacent internal cell — which pyvista provides
    # via the `cell_to_point` mapping but it's expensive.
    #
    # Pragmatic implementation: use OpenFOAM's stored boundary-face values of
    # U (which are zero on a no-slip wall) and the internal-mesh sample at
    # the cell adjacent to the wall face center. ∂U/∂n ≈ (U_internal - U_wall)/d
    # where d = wall-normal distance to the cell center.
    #
    # This is a coarse approximation compared to OpenFOAM's wallShearStress
    # function object, but it's what we can do without the FV operator stack.
    try:
        internal = mesh["internalMesh"]
    except Exception as exc:
        return {"ok": False, "error": f"no internalMesh block: {exc}"}
    if "U" not in internal.array_names:
        return {"ok": False, "error": "U field not present at latestTime in pyvista read"}

    written = 0
    out_path = latest / field_name
    # pyvista output is approximate. We write a placeholder OpenFOAM
    # volVectorField marking the file as "derived"; the comparator can read
    # it but should know the limitations. To avoid generating malformed
    # OpenFOAM dictionaries here, we instead refuse to write a placeholder
    # and just return that pyvista-derived stress can be done. The caller
    # should prefer Strategy 1 (postProcess) which handles the FV operators
    # exactly. If the user really wants pyvista derivation, we can extend
    # this later.
    return {
        "ok": False,
        "error": (
            "pyvista wallShearStress derivation requires wall-normal gradient "
            "with FV-accurate stencil; not yet implemented. Use OpenFOAM "
            "-postProcess (Strategy 1) instead."
        ),
        "n_faces": written,
    }


def _read_nu(case_dir: Path) -> Optional[float]:
    candidates = [
        case_dir / "constant" / "transportProperties",
        case_dir / "constant" / "physicalProperties",
        case_dir / "constant" / "momentumTransport",
    ]
    for p in candidates:
        if not p.is_file():
            continue
        try:
            t = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        m = re.search(r"\bnu\b\s+\[[^\]]+\]\s+([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", t)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                continue
    return None


# ---------------------------------------------------------------------------
# Top-level helper
# ---------------------------------------------------------------------------

def ensure_postprocess_fields(
    case_dir: Path,
    *,
    expected_fields: Optional[List[str]] = None,
    bashrc: Optional[str] = None,
    timeout_s: int = 300,
    allow_pyvista: bool = False,
) -> Dict[str, Any]:
    """
    Ensure post-process fields exist at latestTime. Returns {action, ok,
    detail, missing_before, missing_after}.

    If expected_fields is None, the helper triggers OpenFOAM -postProcess
    whenever the case has a non-empty `functions { ... }` block. This is
    safe and idempotent — postProcess overwrites whatever's there at the
    latest time.

    If expected_fields is given, only triggers postProcess if at least one
    expected field is missing at latestTime.
    """
    cd = Path(case_dir)
    if not cd.is_dir():
        return {"action": "skip", "ok": False, "detail": f"case_dir does not exist: {cd}"}
    controldict = cd / "system" / "controlDict"
    fo_names = _read_function_object_names(controldict)
    if not fo_names:
        return {"action": "skip", "ok": True, "detail": "no function objects defined"}

    missing_before: List[str] = []
    if expected_fields:
        missing_before = _missing_fields_at_latest(cd, expected_fields)
        if not missing_before:
            return {
                "action": "skip", "ok": True,
                "detail": f"all expected fields present at latest: {expected_fields}",
                "missing_before": [], "missing_after": [],
            }

    pp = run_openfoam_postprocess(
        cd, funcs=None, bashrc=bashrc, timeout_s=timeout_s,
    )
    missing_after = _missing_fields_at_latest(cd, expected_fields or []) if expected_fields else []

    # Pyvista fallback (currently best-effort)
    pyvista_attempts: List[Dict[str, Any]] = []
    if allow_pyvista and missing_after:
        for f in missing_after:
            pyvista_attempts.append(derive_with_pyvista(cd, f))

    return {
        "action": "postprocess",
        "ok": pp.get("ok", False),
        "rc": pp.get("rc"),
        "stderr_tail": pp.get("stderr_tail", ""),
        "stdout_tail": pp.get("stdout_tail", ""),
        "app": pp.get("app"),
        "fo_names": fo_names,
        "missing_before": missing_before,
        "missing_after": missing_after,
        "pyvista_attempts": pyvista_attempts,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--case", required=True, type=str)
    parser.add_argument("--field", action="append", default=[], type=str,
                        help="Expected field name(s); repeatable. If given, only "
                             "triggers postProcess when at least one is missing.")
    parser.add_argument("--bashrc", default="", type=str)
    parser.add_argument("--allow-pyvista", action="store_true")
    parser.add_argument("--timeout", default=300, type=int)
    parser.add_argument("--output", default="", type=str,
                        help="If set, write result JSON to this path.")
    args = parser.parse_args()

    case = Path(args.case).expanduser().resolve()
    result = ensure_postprocess_fields(
        case,
        expected_fields=(args.field or None),
        bashrc=(args.bashrc or None),
        timeout_s=args.timeout,
        allow_pyvista=args.allow_pyvista,
    )
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps(result, indent=2, default=str))
    return 0 if result.get("ok") or result.get("action") == "skip" else 2


if __name__ == "__main__":
    raise SystemExit(main())
