#!/usr/bin/env python
"""DAFoam's ConvergentChannel regression case, run as a case of its own.

  dafoam-run mpirun -np <n> python runScript.py

The case set up as DAFoam's runRegTests_DASimpleFoamForward.py sets it up
(incompressible, Spalart-Allmaras with wall functions, inlet 10 m/s), solved
once with DASimpleFoam. The flow is written to the case's time directories
and the wall force CD to results.json. A study that differentiates or
optimises starts from this script: add design variables, functions and
inputs (a field input such as betaFINuTilda) as DAFoam's other regression
scripts in tests/ do.
"""

import json
import os

import openmdao.api as om
from dafoam.mphys import DAFoamBuilder
from mphys.core import Multipoint
from mphys.scenarios import ScenarioAerodynamic
from mpi4py import MPI

U0 = 10.0

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
    "normalizeStates": {"U": U0, "p": U0 * U0 / 2.0, "phi": 1.0, "nuTilda": 1e-3},
    "inputInfo": {
        "aero_vol_coords": {"type": "volCoord", "components": ["solver", "function"]},
    },
}

# For shape derivatives keep IDWarp's rotations off (see the shelf's DAFoam
# usage notes): its derivative omits the rotation terms its warp applies.
meshOptions = {"gridFile": os.getcwd(), "fileType": "OpenFOAM", "symmetryPlanes": [], "useRotations": False}


class Top(Multipoint):
    def setup(self):
        builder = DAFoamBuilder(daOptions, meshOptions, scenario="aerodynamic")
        builder.initialize(self.comm)
        self.add_subsystem("mesh", builder.get_mesh_coordinate_subsystem())
        self.mphys_add_scenario("cruise", ScenarioAerodynamic(aero_builder=builder))
        self.connect("mesh.x_aero0", "cruise.x_aero")


prob = om.Problem()
prob.model = Top()
prob.setup(mode="rev")
prob.run_model()
cd = float(prob.get_val("cruise.aero_post.CD")[0])
if MPI.COMM_WORLD.rank == 0:
    with open("results.json", "w") as f:
        json.dump({"CD": cd}, f, indent=2)
    print(f"CD = {cd}")
