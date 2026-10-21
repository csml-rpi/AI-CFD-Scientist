#!/usr/bin/env python3
"""The setup check: a case's files are checked against its requirement before
the first run, and what differs is fixed.

Every hyStrath case of shelf_demo_hystrath_20261011_1602 kept its starter's
density, ran cleanly at the wrong Knudsen number and was passed: the only
review in the case loop read error logs, and a clean run has none.

  1. what the check is shown: every case file, seeded or written, long field
     lists shortened, given files and the study's mesh marked
  2. the check-fix-check loop, with the model stubbed
  3. edits of compressed fields land as plain text
  4. end to end: the OpenFOAM 10 cavity seeded with nu 0.01, a requirement
     asking for 0.02, a writer that leaves the seed's value; the check finds
     it, the fix lands, icoFoam runs with 0.02
  5. live, when a provider is configured: the real check prompt on that case

Run: python scripts/test_setup_check.py
"""

from __future__ import annotations

import gzip
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cfd_langgraph.foam_native import allrun as allrun_mod  # noqa: E402
from cfd_langgraph.foam_native import decomposer, loop, parser, rag, review, writer  # noqa: E402

OF10 = Path("/mnt/sda1/openfoam10")
CAVITY = OF10 / "tutorials/incompressible/icoFoam/cavity/cavity"
FAILURES: list[str] = []


def check(name: str, got: object, want: object = True) -> None:
    ok = got == want
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + ("" if ok else f": got {got!r}, want {want!r}"))
    if not ok:
        FAILURES.append(name)


def field(n: int) -> str:
    values = "\n".join("(1 0 0)" for _ in range(n))
    return ("FoamFile { class volVectorField; object U; }\ndimensions [0 1 -1 0 0 0 0];\n"
            f"internalField nonuniform List<vector>\n{n}\n(\n{values}\n)\n;\n\n"
            "boundaryField\n{\n    inlet { type fixedValue; value uniform (10 0 0); }\n}\n")


print("== 1. what the check is shown")
with tempfile.TemporaryDirectory() as tmp:
    case = Path(tmp)
    for d in ("0.org", "constant/polyMesh", "system"):
        (case / d).mkdir(parents=True)
    (case / "system/controlDict").write_text("application simpleFoam;\nendTime 500;\n")
    (case / "system/blockMeshDict").write_text("blocks ( hex (0 1 2 3 4 5 6 7) (7 7 7) );\n")
    (case / "constant/transportProperties").write_text("nu 1.5e-05;\n")
    (case / "constant/waveDict").write_text("waveHeight 0.15;\n")
    (case / "constant/polyMesh/points").write_text("(0 0 0)\n" * 50)
    (case / "0.org/U").write_text(field(20000))
    with gzip.open(case / "0.org/p.gz", "wt") as f:
        f.write("FoamFile { object p; }\ninternalField uniform 0;\nboundaryField { outlet { type fixedValue; value uniform 0; } }\n")
    (case / "runScript.py").write_text("U0 = 10.0\n")
    (case / "log.simpleFoam").write_text("Time = 1\n")
    xml = loop._case_files_xml(case, given={"constant/waveDict"}, zero_template="0.org", mesh_fixed=True)
    check("a seeded file the writer never listed is shown", "nu 1.5e-05" in xml)
    check("controlDict comes first", xml.index("controlDict") < xml.index("transportProperties"))
    check("fields are read from the folder 0/ is built from", "<folder_name>0.org</folder_name>" in xml)
    check("a compressed field is read", "type fixedValue; value uniform 0" in xml)
    check("the case's script is shown", "U0 = 10.0" in xml)
    check("a long internalField is shortened", "characters of values left out" in xml)
    check("but its boundary conditions are kept", "value uniform (10 0 0)" in xml)
    check("and the whole file stays small", len(xml) < 8000)
    check("a given file is marked", "given by the study" in xml.split("waveDict")[1][:200])
    check("the study's mesh is marked", "selected mesh" in xml.split("blockMeshDict")[1][:200])
    check("the mesh itself is not shown", "polyMesh" in xml, False)
    check("logs are not shown", "log.simpleFoam" in xml, False)

print("== 2. check, fix, check")
saved = (review.check_setup, review.plan_rewrite, writer.edit_file)
try:
    with tempfile.TemporaryDirectory() as tmp:
        case = Path(tmp)
        (case / "constant").mkdir()
        (case / "constant/transportProperties").write_text("nu 1e-05;\n")
        (case / "constant/waveDict").write_text("waveHeight 0.10;\n")
        calls = {"check": 0}

        def fake_check(llm, *, foamfiles_xml, user_requirement, **kw):
            calls["check"] += 1
            nu_ok = "nu 1.5e-05;" in foamfiles_xml
            return [{"requirement": "nu = 1.5e-5 m2/s", "file": "constant/transportProperties",
                     "found": "1.5e-05" if nu_ok else "1e-05", "status": "match" if nu_ok else "mismatch",
                     "fix": "set nu 1.5e-05"},
                    {"requirement": "inlet U = 10 m/s", "file": "0/U", "found": "10", "status": "match", "fix": ""}]

        review.check_setup = fake_check
        review.plan_rewrite = lambda llm, **kw: [{"file": "constant/transportProperties",
                                                  "changes": "set nu 1.5e-05", "delete": False}]
        writer.edit_file = lambda llm, **kw: kw["current_content"].replace("1e-05", "1.5e-05")
        history: list[str] = []
        apply = lambda targets: loop._apply_edits(
            None, case, targets, history=history, zero_template="", given=set(), mesh_seed_case_dir=None,
            run_script_name="", subtasks=[], case_solver="simpleFoam")
        out = loop._setup_check(None, case, "nu = 1.5e-5, inlet 10 m/s",
                                files_xml=lambda: loop._case_files_xml(case, given=set()),
                                apply=apply, history=history)
        check("the mismatch is fixed in the file", (case / "constant/transportProperties").read_text(), "nu 1.5e-05;\n")
        check("checked again after the fix, then stopped", calls["check"], 2)
        check("nothing remains", out["remaining"], [])
        check("one fix round", out["fix_rounds"], 1)
        check("the reviewer's history says what the check changed", "Setup check before the first run" in history[0])
        record = json.loads((case / "setup_check.json").read_text())
        check("every round is recorded", len(record["rounds"]), 2)

        # A mismatch in a file the study gives cannot be fixed: it is reported.
        calls["check"] = 0
        review.check_setup = lambda llm, **kw: (calls.__setitem__("check", calls["check"] + 1) or
                                                [{"requirement": "wave height 0.15 m", "file": "constant/waveDict",
                                                  "found": "0.10", "status": "mismatch", "fix": "waveHeight 0.15"}])
        review.plan_rewrite = lambda llm, **kw: [{"file": "constant/waveDict", "changes": "waveHeight 0.15",
                                                  "delete": False}]
        history.clear()
        apply = lambda targets: loop._apply_edits(
            None, case, targets, history=history, zero_template="", given={"constant/waveDict"},
            mesh_seed_case_dir=None, run_script_name="", subtasks=[], case_solver="olaFlow")
        out = loop._setup_check(None, case, "wave height 0.15 m",
                                files_xml=lambda: loop._case_files_xml(case, given={"constant/waveDict"}),
                                apply=apply, history=history)
        check("a given file is never edited", (case / "constant/waveDict").read_text(), "waveHeight 0.10;\n")
        check("the edit was refused into the history", any("gives as it is" in h for h in history))
        check("bounded: two fixes, three checks", calls["check"], 3)
        check("what still differs is returned", [i["requirement"] for i in out["remaining"]], ["wave height 0.15 m"])

    print("== 3. a compressed field is edited as plain text")
    with tempfile.TemporaryDirectory() as tmp:
        case = Path(tmp)
        (case / "0").mkdir()
        with gzip.open(case / "0/U.gz", "wt") as f:
            f.write("internalField uniform (1 0 0);\n")
        writer.edit_file = lambda llm, **kw: kw["current_content"].replace("(1 0 0)", "(10 0 0)")
        loop._apply_edits(None, case, [{"file": "0/U.gz", "changes": "inlet 10", "delete": False}], history=[],
                          zero_template="", given=set(), mesh_seed_case_dir=None, run_script_name="",
                          subtasks=[], case_solver="simpleFoam")
        check("written as plain 0/U", (case / "0/U").read_text(), "internalField uniform (10 0 0);\n")
        check("the old compressed copy is gone", (case / "0/U.gz").exists(), False)

    print("== 3b. the mesh itself is never edited; a seed's mesh replaces the case's")
    with tempfile.TemporaryDirectory() as tmp:
        case = Path(tmp) / "case"
        (case / "constant/polyMesh").mkdir(parents=True)
        with gzip.open(case / "constant/polyMesh/points.gz", "wt") as f:
            f.write("3\n(\n(0 0 0)\n(1 0 0)\n(1 1 0)\n)\n")
        writer.edit_file = lambda llm, **kw: "a stub"
        history = []
        failed = loop._apply_edits(None, case, [{"file": "constant/polyMesh/points.gz", "changes": "fix",
                                                 "delete": False},
                                                {"file": "constant/polyMesh/owner", "changes": "x", "delete": True}],
                                   history=history, zero_template="", given=set(), mesh_seed_case_dir=None,
                                   run_script_name="", subtasks=[], case_solver="simpleFoam")
        check("a mesh file is not rewritten", (case / "constant/polyMesh/points.gz").is_file()
              and not (case / "constant/polyMesh/points").exists())
        check("nor deleted, and the reviewer is told why", sum("part of the mesh itself" in h for h in history), 2)
        seed = Path(tmp) / "seed"
        (seed / "constant/polyMesh").mkdir(parents=True)
        (seed / "constant/polyMesh/points.gz").write_bytes((case / "constant/polyMesh/points.gz").read_bytes())
        (case / "constant/polyMesh/points").write_text("stale stub")
        loop._copy_case_files(seed, case)
        check("re-seeding drops a stale plain mesh file beside the seed's .gz",
              (case / "constant/polyMesh/points").exists(), False)
        check("and brings the seed's mesh", (case / "constant/polyMesh/points.gz").is_file())

    print("== 3d. deletes are exact; prose is never saved as a field; seeding drops .gz twins")
    with tempfile.TemporaryDirectory() as tmp:
        case = Path(tmp) / "case"
        (case / "0").mkdir(parents=True)
        (case / "0/U").write_text("FoamFile {}\ninternalField uniform (1 0 0);\n")
        with gzip.open(case / "0/U.gz", "wt") as f:
            f.write("FoamFile {}\ninternalField uniform (1 0 0);\n")
        hist = []
        loop._apply_edits(None, case, [{"file": "0/U", "changes": "remove the plain duplicate", "delete": True}],
                          history=hist, zero_template="", given=set(), mesh_seed_case_dir=None,
                          run_script_name="", subtasks=[], case_solver="simpleFoam")
        check("deleting the plain copy keeps the compressed one",
              ((case / "0/U").exists(), (case / "0/U.gz").exists()), (False, True))
        (case / "0/nut").write_text("FoamFile {}\ninternalField uniform 0;\n")
        writer.edit_file = lambda llm, **kw: "I cannot generate 157464 values here."
        failed = loop._apply_edits(None, case, [{"file": "0/nut", "changes": "write the field", "delete": False}],
                                   history=hist, zero_template="", given=set(), mesh_seed_case_dir=None,
                                   run_script_name="", subtasks=[], case_solver="simpleFoam")
        check("a refusal is not saved as the field", (case / "0/nut").read_text(), "FoamFile {}\ninternalField uniform 0;\n")
        check("and counts as a failed edit, said in the history", (failed, "no OpenFOAM dictionary" in hist[-1]), (1, True))
        seed = Path(tmp) / "seed"
        (seed / "0").mkdir(parents=True)
        (seed / "0/k").write_text("FoamFile {}\n")
        with gzip.open(seed / "0/k.gz", "wt") as f:
            f.write("FoamFile {}\n")
        fresh = Path(tmp) / "fresh"
        loop._copy_case_files(seed, fresh)
        check("seeding keeps the plain field and drops its compressed twin",
              ((fresh / "0/k").exists(), (fresh / "0/k.gz").exists()), (True, False))

    print("== 3e. a file too large to rewrite whole is not sent to the model")
    with tempfile.TemporaryDirectory() as tmp:
        case = Path(tmp)
        (case / "0").mkdir()
        values = "\n".join("0.001" for _ in range(160000))
        with gzip.open(case / "0/nuTilda.gz", "wt") as f:
            f.write(f"FoamFile {{}}\ninternalField nonuniform List<scalar> 160000\n(\n{values}\n);\n")
        sent = []
        writer.edit_file = lambda llm, **kw: sent.append(len(kw["current_content"])) or kw["current_content"]
        hist = []
        failed = loop._apply_edits(None, case, [{"file": "0/nuTilda.gz", "changes": "inlet value 3e-5",
                                                 "delete": False}],
                                   history=hist, zero_template="", given=set(), mesh_seed_case_dir=None,
                                   run_script_name="", subtasks=[], case_solver="simpleFoam")
        check("the model is not asked to rewrite a per-cell field", sent, [])
        check("the field is left as it was", ((case / "0/nuTilda.gz").is_file(), (case / "0/nuTilda").exists()),
              (True, False))
        check("a failed edit, and the reviewer is told why", (failed, "too large to rewrite whole" in hist[-1]),
              (1, True))

    print("== 3c. no blockMesh for a case without a blockMeshDict")
    with tempfile.TemporaryDirectory() as tmp:
        case = Path(tmp)
        (case / "system").mkdir()
        script = allrun_mod.build_allrun_script("blockMesh\ncheckMesh\nsimpleFoam", "simpleFoam", case_dir=case)
        check("blockMesh is dropped", "blockMesh" in script.replace("checkMesh", ""), False)
        (case / "system/blockMeshDict").write_text("vertices ();\n")
        script = allrun_mod.build_allrun_script("blockMesh\nsimpleFoam", "simpleFoam", case_dir=case)
        check("and kept when there is one", "runApplication blockMesh" in script)
finally:
    review.check_setup, review.plan_rewrite, writer.edit_file = saved

print("== 4. end to end on the OpenFOAM 10 cavity")
if not CAVITY.is_dir():
    print("[SKIP] no OpenFOAM 10 cavity tutorial")
else:
    saved_all = (parser.parse_requirement, rag.retrieve_references, decomposer.decompose_subtasks,
                 writer.write_file_initial, writer.edit_file, allrun_mod.generate_allrun_commands,
                 review.check_setup, review.plan_rewrite, review.review_errors)
    requirement = "Lid-driven cavity, icoFoam, lid speed 1 m/s, kinematic viscosity nu = 0.02 m2/s, end time 0.1 s."
    try:
        parser.parse_requirement = lambda llm, req: {"case_name": "cavity", "case_solver": "icoFoam",
                                                     "case_domain": "incompressible", "case_category": "basic"}
        rag.retrieve_references = lambda *a, **k: {"dir_structure": "", "tutorial_reference": "",
                                                   "allrun_reference": ""}
        decomposer.decompose_subtasks = lambda llm, req, ds: [{"folder_name": "system", "file_name": "controlDict"},
                                                              {"folder_name": "constant/polyMesh", "file_name": "points"}]
        written: list = []
        # A writer that edits toward the requirement but misses the seeded
        # physicalProperties: it only ever sees controlDict.
        writer.edit_file = lambda llm, **kw: written.append(kw["folder_name"] + "/" + kw["file_name"]) or (
            kw["current_content"].replace("endTime         0.5;", "endTime         0.1;")
            if kw["file_name"] == "controlDict" else
            kw["current_content"].replace("0.01;", "0.02;") if "nu" in kw["changes"] else kw["current_content"])
        writer.write_file_initial = lambda llm, **kw: written.append(str(kw)) or ""
        allrun_mod.generate_allrun_commands = lambda llm, **kw: "blockMesh\nicoFoam"
        seen: dict = {"checks": []}

        def cavity_check(llm, *, foamfiles_xml, user_requirement, **kw):
            ok = "[0 2 -1 0 0 0 0] 0.02;" in foamfiles_xml
            seen["checks"].append(ok)
            return [{"requirement": "nu = 0.02 m2/s", "file": "constant/physicalProperties",
                     "found": "0.02" if ok else "0.01", "status": "match" if ok else "mismatch",
                     "fix": "set nu to 0.02 in constant/physicalProperties"}]

        review.check_setup = cavity_check
        review.plan_rewrite = lambda llm, **kw: [{"file": "constant/physicalProperties",
                                                  "changes": "set nu 0.02", "delete": False}]
        review.review_errors = lambda *a, **k: "no errors expected"
        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp) / "case_001"
            result = loop.run_foam_case(None, case, requirement, openfoam_path=str(OF10),
                                        base_case_seed_dir=CAVITY, max_loop=2, max_time_limit_s=600)
            check("the case ran", result.get("success"), True)
            check("the check found the seed's viscosity, then the fix", seen["checks"], [False, True])
            check("the case ran with nu 0.02", "0.02;" in (case / "constant/physicalProperties").read_text())
            check("the run result carries the check", result.get("setup_check", {}).get("remaining"), [])
            check("in one fix round", result.get("setup_check", {}).get("fix_rounds"), 1)
            check("on the first run", result.get("loop_count"), 1)
            check("a mesh file the planner listed is never written",
                  [w for w in written if "polyMesh" in w], [])

        print("== 4b. a failing case the review cannot change stops, keeping its error")
        writer.edit_file = lambda llm, **kw: (
            kw["current_content"].replace("deltaT          0.005;", "deltaT          abc;")
            if kw["file_name"] == "controlDict" else kw["current_content"])
        review.check_setup = lambda llm, **kw: []
        given_src = Path(tempfile.mkdtemp()) / "physicalProperties"
        given_src.write_text((CAVITY / "constant/physicalProperties").read_text())
        review.plan_rewrite = lambda llm, **kw: [{"file": "constant/physicalProperties",
                                                  "changes": "change nu", "delete": False}]
        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp) / "case_002"
            result = loop.run_foam_case(None, case, requirement, openfoam_path=str(OF10),
                                        base_case_seed_dir=CAVITY, max_loop=5, max_time_limit_s=600,
                                        given_files={"constant/physicalProperties": given_src})
            check("the case failed", result.get("success"), False)
            check("after one run, not five", result.get("loop_count"), 1)
            logs = " ".join(result.get("error_logs") or [])
            check("the record says nothing could be changed", "changed no file" in logs)
            check("and still shows the run's own error", "FOAM FATAL" in logs, True)
            check("the review was given a second round with the refusals in front of it",
                  any("Nothing in the case changed" in h for h in result.get("review_history") or []))
            check("the record keeps what the review tried and why it was refused",
                  any("study gives" in h and "rejected" in h for h in result.get("review_history") or []))
    finally:
        (parser.parse_requirement, rag.retrieve_references, decomposer.decompose_subtasks,
         writer.write_file_initial, writer.edit_file, allrun_mod.generate_allrun_commands,
         review.check_setup, review.plan_rewrite, review.review_errors) = saved_all

print("== 5. live: the real check prompt on the seeded cavity")
if not os.environ.get("CFD_SCIENTIST_LLM_PROVIDER") or not CAVITY.is_dir():
    print("[SKIP] no provider configured")
else:
    from cfd_langgraph.config import get_settings
    from cfd_langgraph.llm.factory import create_langchain_llm

    with tempfile.TemporaryDirectory() as tmp:
        case = Path(tmp) / "cavity"
        shutil.copytree(CAVITY, case)
        llm = create_langchain_llm(model=get_settings().model, temperature=0.0)  # as run_case_native does
        items = review.check_setup(
            llm, foamfiles_xml=loop._case_files_xml(case, given=set()),
            user_requirement="Lid-driven cavity with icoFoam: lid speed 2 m/s, kinematic viscosity "
                             "nu = 0.02 m2/s, walls no-slip, end time 0.5 s, time step 0.005 s.")
        print(json.dumps(items, indent=1)[:3000])
        bad = {i["requirement"].lower(): i["status"] for i in items}
        check("the wrong viscosity is caught",
              any("nu" in k or "viscosity" in k for k, s in bad.items() if s == "mismatch"))
        check("the wrong lid speed is caught",
              any("lid" in k or "speed" in k or "velocity" in k for k, s in bad.items() if s == "mismatch"))
        check("the end time and time step, which match, are not flagged",
              [k for k, s in bad.items() if s != "match" and ("end time" in k or "time step" in k)], [])

print()
print("FAILURES:", FAILURES if FAILURES else "none")
sys.exit(1 if FAILURES else 0)
