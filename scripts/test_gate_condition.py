#!/usr/bin/env python3
"""The mesh gate checks the mesh at the condition its group runs at.

The gate used to solve the starter case as shipped and hand its mesh to every
case of the group, and to share it with every group built from the same
starter case. hyStrath's DSMC cases at Knudsen 0.05 so ran on the mesh checked
at the starter's 0.25, with cells several mean free paths across, while Claude
Code, building a mesh per Knudsen number, got the study right
(shelf_demo_hystrath_20261011_1602, claude_code_hystrath_20261015).

The solver runs and the model's comparison are stubbed: this checks what the
gate asks for, not OpenFOAM.

Run: python scripts/test_gate_condition.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cfd_langgraph.manager.tools as mt  # noqa: E402
from cfd_langgraph import foam_native  # noqa: E402
from cfd_langgraph.config import get_settings  # noqa: E402
from cfd_langgraph.foam_native import review  # noqa: E402

FAILURES: list[str] = []


def check(name: str, got, want=True) -> None:
    ok = got == want
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + ("" if ok else f" — got {got!r}, want {want!r}"))
    if not ok:
        FAILURES.append(name)


KN005 = [{"quantity": "Knudsen number", "starter": "0.25", "required": "0.05", "file": "constant/dsmcProperties"}]
asked: list = []      # requirements the comparison was asked about
runs: list = []       # (case_dir, requirement, seed_only) of every gate run
answer: dict = {"items": []}


def fake_condition_changes(llm, *, foamfiles_xml, user_requirement, solver_notes=""):
    asked.append(user_requirement)
    return list(answer["items"])


def fake_run_foam_case(llm, case_dir, user_requirement, **kw):
    runs.append((Path(case_dir), user_requirement, kw.get("seed_only")))
    return {"status": "failed", "case_dir": str(case_dir)}


review.condition_changes = fake_condition_changes
foam_native.run_foam_case = fake_run_foam_case

with tempfile.TemporaryDirectory() as tmp:
    out = Path(tmp) / "study"
    starter = Path(tmp) / "starter" / "dsmc_cylinder_Kn0p25"
    (starter / "system").mkdir(parents=True)
    (starter / "constant").mkdir()
    (starter / "system" / "controlDict").write_text("application dsmcFoam+;\n")
    (starter / "system" / "blockMeshDict").write_text("blocks ( hex (0 1 2 3 4 5 6 7) (10 10 1) simpleGrading (1 1 1) );\n")
    (starter / "constant" / "dsmcProperties").write_text("numberDensity 1.7e20;\n")
    out.mkdir()
    (out / "starter_understanding.json").write_text(json.dumps(
        {"starter_dir": str(starter.parent), "base_case_path": str(starter)}))
    (out / "requirements.json").write_text(json.dumps([
        {"case_id": "case_001", "user_requirement_text": "DSMC of the cylinder at Kn 0.05."},
        {"case_id": "case_002", "user_requirement_text": "DSMC of the cylinder at Kn 0.25."}]))
    (out / "study_metrics.json").write_text(json.dumps([{"name": "drag"}]))
    (out / "study_mode.json").write_text(json.dumps({"mode": "solver"}))
    tools ={fn.__name__: fn for fn in mt.build_manager_tools(get_settings(), out)["manager_tools"]}
    gate = tools["run_mesh_gate"]

    print("== 1. a group at another condition than the starter's")
    answer["items"] = KN005
    spec = gate("dsmc_kn005", case_id="case_001", metrics=["drag"])
    check("the comparison is asked with the group's requirement", asked, ["DSMC of the cylinder at Kn 0.05."])
    check("one baseline is run", len(runs), 1)
    case_dir, requirement, seed_only = runs[0]
    check("written for the condition, not copied as shipped", seed_only, False)
    check("its requirement names the condition", "Knudsen number 0.05 (the starter case has 0.25" in requirement)
    check("and lets the mesh, domain and time step be sized for it",
          "cell sizes of the mesh, the extent of the domain and the time step" in requirement)
    check("its level is named after the group, not the starter's condition",
          case_dir.name, "dsmc_kn005__baseline")
    check("the gate's record carries the condition", spec.get("condition"), KN005)
    check("the answer is kept", json.loads((out / "mesh_gate/dsmc_kn005/condition.json").read_text())["items"],
          KN005)
    gate("dsmc_kn005", case_id="case_001", metrics=["drag"])
    check("and not asked again on a re-call", len(asked), 1)

    print("== 2. a group at the starter's own condition")
    runs.clear()
    answer["items"] = []
    gate("dsmc_kn025", case_id="case_002", metrics=["drag"])
    check("the starter case is run as shipped", runs[0][2], True)
    check("under the starter's name", runs[0][0].name.endswith("baseline") and "Kn0p25" in runs[0][0].name)

    print("== 3. a converged gate is shared only at the same condition")
    selected = out / "mesh_gate" / "done" / "level"
    selected.mkdir(parents=True)
    (out / "mesh_gate" / "done" / "selected_mesh_spec.json").write_text(json.dumps({
        "physics_group": "done", "converged": True, "selected_level": str(selected),
        "seed_fingerprint": mt._case_fingerprint(starter), "condition": []}))
    runs.clear()
    answer["items"] = KN005
    gate("dsmc_kn005_b", case_id="case_001", metrics=["drag"])
    check("a group at Kn 0.05 does not take the mesh checked at the starter's Kn 0.25", len(runs), 1)
    runs.clear()
    answer["items"] = []
    reused = gate("dsmc_kn025_b", case_id="case_002", metrics=["drag"])
    check("a group at the starter's condition does", (len(runs), reused.get("reused_from_group")), (0, "done"))

print()
print("FAILURES:", FAILURES if FAILURES else "none")
sys.exit(1 if FAILURES else 0)
