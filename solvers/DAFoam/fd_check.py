#!/usr/bin/env python
"""Adjoint gradients against finite differences, for solvers/DAFoam/checks.py.

  dafoam-run mpirun -np 4 python fd_check.py <ConvergentChannel case> <results.json>

The convergent-channel case of DAFoam's own regression tests, set up as its
runRegTests_DASimpleFoamForward.py sets it up (incompressible, Spalart-Allmaras
with wall functions, inlet 10 m/s), with two design variables: the inlet
velocity and one FFD shape variable. The drag-like wall force CD is solved
once and its gradient taken by the adjoint (OpenMDAO compute_totals, reverse
mode); then each variable is moved by +-h and the flow solved again, for two
step sizes. Central differences use nothing but converged flow solutions, so
they check the adjoint without sharing any of its code.
"""

import json
import os
import sys

import numpy as np
import openmdao.api as om
from dafoam.mphys import DAFoamBuilder
from mphys import MPhysVariables
from mphys.core import Multipoint
from mphys.scenarios import ScenarioAerodynamic
from mpi4py import MPI
from pygeo import geo_utils
from pygeo.mphys import OM_DVGEOCOMP

gcomm = MPI.COMM_WORLD
case, out = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
os.chdir(case)

U0 = 10.0
STEPS = {"u_in": (1e-2, 1e-3), "shape": (1e-3, 1e-4)}

daOptions = {
    "designSurfaces": ["walls"],
    "solverName": "DASimpleFoam",
    "primalMinResTol": 1.0e-12,
    "primalMinResTolDiff": 1e4,
    "printDAOptions": False,
    "primalBC": {
        "U0": {"variable": "U", "patches": ["inlet"], "value": [U0, 0.0, 0.0]},
        "p0": {"variable": "p", "patches": ["outlet"], "value": [0.0]},
        "nuTilda0": {"variable": "nuTilda", "patches": ["inlet"], "value": [4.5e-5]},
        "useWallFunction": True,
        "transport:nu": 1.5e-5,
    },
    "function": {
        "CD": {
            "type": "force",
            "source": "patchToFace",
            "patches": ["walls"],
            "directionMode": "fixedDirection",
            "direction": [1.0, 0.0, 0.0],
            "scale": 0.1,
        },
    },
    "adjEqnOption": {"gmresRelTol": 1.0e-12, "pcFillLevel": 1, "jacMatReOrdering": "rcm"},
    "normalizeStates": {"U": U0, "p": U0 * U0 / 2.0, "phi": 1.0, "nuTilda": 1e-3},
    "inputInfo": {
        "aero_vol_coords": {"type": "volCoord", "components": ["solver", "function"]},
        "u_in": {
            "type": "patchVar",
            "varName": "U",
            "varType": "vector",
            "patches": ["inlet"],
            "components": ["solver", "function"],
        },
    },
}

# useRotations off: IDWarp's warp applies its surface-node rotations but its
# derivative (warpDeriv) leaves them out, so with rotations on the shape
# gradient is consistently ~2.8 % off the true one (the warper's reverse
# Jacobian differs from its finite-difference Jacobian by 10 % in norm; with
# rotations off they agree to 3e-10). Measured on this case 2026-10-09.
meshOptions = {"gridFile": os.getcwd(), "fileType": "OpenFOAM", "symmetryPlanes": [], "useRotations": False}


class Top(Multipoint):
    def setup(self):
        builder = DAFoamBuilder(daOptions, meshOptions, scenario="aerodynamic")
        builder.initialize(self.comm)
        self.add_subsystem("dvs", om.IndepVarComp(), promotes=["*"])
        self.add_subsystem("mesh", builder.get_mesh_coordinate_subsystem())
        self.add_subsystem("geometry", OM_DVGEOCOMP(file="FFD/FFD.xyz", type="ffd"))
        self.mphys_add_scenario("cruise", ScenarioAerodynamic(aero_builder=builder))
        self.connect("mesh.x_aero0", "geometry.x_aero0_geometry_input")
        self.connect("geometry.x_aero0_geometry_output", "cruise.x_aero")

    def configure(self):
        points = self.mesh.mphys_get_surface_mesh()
        self.geometry.nom_add_discipline_coords(MPhysVariables.Aerodynamics.Surface.Geometry, points)
        self.cruise.coupling.solver.add_dvgeo(self.geometry.nom_getDVGeo())
        pts = self.geometry.nom_getDVGeo().getLocalIndex(0)
        self.geometry.nom_addLocalDV(dvName="shape", pointSelect=geo_utils.PointSelect("list", pts[1, 0, 1].flatten()))
        self.dvs.add_output("shape", val=np.zeros(1))
        self.dvs.add_output("u_in", val=np.array([U0, 0.0, 0.0]))
        self.connect("shape", "geometry.shape")
        self.connect("u_in", "cruise.u_in")
        self.add_design_var("shape", lower=-10.0, upper=10.0, scaler=1.0)
        self.add_design_var("u_in", lower=-50.0, upper=50.0, scaler=1.0, indices=[0])
        self.add_objective("cruise.aero_post.CD", scaler=1.0)


prob = om.Problem()
prob.model = Top()
prob.setup(mode="rev")
prob.run_model()
cd0 = float(prob.get_val("cruise.aero_post.CD")[0])
totals = prob.compute_totals(of=["cruise.aero_post.CD"], wrt=["shape", "u_in"])
adjoint = {"shape": float(np.ravel(totals[("cruise.aero_post.CD", "shape")])[0]),
           "u_in": float(np.ravel(totals[("cruise.aero_post.CD", "u_in")])[0])}

base = {"shape": np.zeros(1), "u_in": np.array([U0, 0.0, 0.0])}
fd = {}
for name, steps in STEPS.items():
    fd[name] = []
    for h in steps:
        values = []
        for sign in (1.0, -1.0):
            moved = base[name].copy()
            moved[0] += sign * h
            prob.set_val(name, moved)
            prob.run_model()
            values.append(float(prob.get_val("cruise.aero_post.CD")[0]))
        prob.set_val(name, base[name])
        fd[name].append({"h": h, "cd_plus": values[0], "cd_minus": values[1],
                         "gradient": (values[0] - values[1]) / (2.0 * h)})

if gcomm.rank == 0:
    with open(out, "w") as f:
        json.dump({"CD": cd0, "adjoint": adjoint, "finite_difference": fd}, f, indent=2)
    print(json.dumps({"CD": cd0, "adjoint": adjoint,
                      "fd": {k: [e["gradient"] for e in v] for k, v in fd.items()}}))
