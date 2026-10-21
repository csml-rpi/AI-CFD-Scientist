"""Frozen validation checks for DAFoam (see scripts/shelf_validate.py).

What DAFoam adds to OpenFOAM is the gradient: the adjoint of a converged flow
solution with respect to the design variables. The check is the one an adjoint
must pass whatever the case: its gradient equals the central finite
difference of the quantity, which needs nothing but flow solutions at moved
design variables and so shares none of the adjoint's code.

The case is the convergent channel of DAFoam's own regression tests
(reg_test_files at a pinned commit, SHA-256 checked), set up as its
runRegTests_DASimpleFoamForward.py sets it up: DASimpleFoam, Spalart-Allmaras
with wall functions, inlet 10 m/s, flow solved to residual 1e-12, adjoint to
GMRES tolerance 1e-12. The wall force CD is differentiated with respect to the
inlet velocity and one FFD shape variable (solvers/DAFoam/fd_check.py).

The mesh is warped by IDWarp with useRotations off (fd_check.py). The first
run, with IDWarp's default rotations on, failed on shape: adjoint 1.7437
against finite differences 1.6961 (2.8 %), the same ratio with SA, k-epsilon
and laminar flow, while the inflow gradient matched to 4e-10. OpenMDAO's
check_partials put the whole error in the mesh warper: IDWarp's derivative
leaves out the rotation terms its warp applies. With rotations off the
warper's derivative matches to 3e-10 and the shape gradient to 3e-6. Changed
on 2026-10-09 at the user's request to make DAFoam work; tolerances
unchanged. Studies must use the same setting (solver.yaml, usage).

Tolerances: each adjoint gradient within 0.1 % of the finite difference at the
smaller step. Both sides are converged far below that, and central
differences err as h^2, so a correct adjoint passes with room; an adjoint that
leaves out part of the linearisation (frozen turbulence, a missing boundary
term) misses by percents. The finite differences at the two steps must agree
within 0.1 % too, or the comparison would mean nothing.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
import urllib.request
from pathlib import Path
from typing import Any, Dict

REG_FILES_COMMIT = "927974649273962a7f60388a781e330e2a85e6f9"
REG_FILES_SHA256 = "af21851e0ba08f6668bbed62447cc4f1096a935d884f2371b2d6e3d88e2ddff5"
REG_FILES_URL = f"https://github.com/DAFoam/reg_test_files/archive/{REG_FILES_COMMIT}.tar.gz"
CHANNEL = "reg_test_files-main/ConvergentChannel"


def _archive(ctx: Dict[str, Any]) -> Path:
    """The pinned regression data, downloaded once beside the shelf's sources."""
    from cfd_langgraph.solver_shelf import shelf_root

    path = shelf_root() / "src" / "dafoam_reg_test_files" / f"{REG_FILES_COMMIT}.tar.gz"
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(REG_FILES_URL, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != REG_FILES_SHA256:
        raise RuntimeError(f"{path} has SHA-256 {digest}, expected {REG_FILES_SHA256}")
    return path


def prepare(case: Path, ctx: Dict[str, Any]) -> None:
    target = case / "reg_test_files-main"
    if target.exists():
        shutil.rmtree(target)
    with tarfile.open(_archive(ctx)) as tar:
        tar.extractall(case, filter="data")
    (case / f"reg_test_files-{REG_FILES_COMMIT}").rename(target)
    channel = case / CHANNEL
    # As runRegTests_DASimpleFoamForward.py does before it runs.
    shutil.rmtree(channel / "0", ignore_errors=True)
    for stale in channel.glob("processor*"):
        shutil.rmtree(stale)
    shutil.copytree(channel / "0.incompressible", channel / "0")
    for f in (channel / "system.incompressible").iterdir():
        shutil.copy2(f, channel / "system" / f.name)
    shutil.copy2(channel / "constant" / "turbulenceProperties.sa", channel / "constant" / "turbulenceProperties")


RUN = (f'dafoam-run mpirun --oversubscribe -np 4 python "$SHELF_RECIPE_DIR/fd_check.py" {CHANNEL} '
       "fd_results.json > log.fd_check 2>&1 && test -s fd_results.json")


def measure(case: Path, ctx: Dict[str, Any]) -> Dict[str, Any]:
    doc = json.loads((case / "fd_results.json").read_text())
    values: Dict[str, Any] = {"CD": doc["CD"]}
    for name in ("u_in", "shape"):
        coarse, fine = doc["finite_difference"][name]
        adjoint = doc["adjoint"][name]
        values[f"dCD_d{name}_adjoint"] = adjoint
        values[f"dCD_d{name}_fd"] = fine["gradient"]
        values[f"dCD_d{name}_adjoint_vs_fd"] = abs(adjoint - fine["gradient"]) / abs(fine["gradient"])
        values[f"dCD_d{name}_fd_steps_agree"] = abs(coarse["gradient"] - fine["gradient"]) / abs(fine["gradient"])
    return values


SOURCE = ("central finite differences of converged flow solutions (no adjoint code), "
          "DAFoam regression case ConvergentChannel at reg_test_files " + REG_FILES_COMMIT[:12])

CHECKS = [
    {
        "name": "adjoint gradients against finite differences",
        "tutorial": "tests",
        "translate": False,          # DAFoam runs its own OpenFOAM v2506, not the installed one
        "prepare": prepare,
        "run": RUN,
        "timeout": 2 * 3600,
        "measure": measure,
        "source": SOURCE,
        "expect": [
            {"quantity": f"dCD_d{name}_{what}", "reference": 0.0, "abs_tol": 1e-3, "source": SOURCE}
            for name in ("u_in", "shape") for what in ("adjoint_vs_fd", "fd_steps_agree")
        ],
    },
]
