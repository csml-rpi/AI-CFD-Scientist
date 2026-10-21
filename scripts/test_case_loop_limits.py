#!/usr/bin/env python3
"""Case-loop fixes from the DAFoam study that never finished
(shelf_demo_dafoam_20261015_1318):

1. OpenFOAM's start-up line "trapFpe: Floating point exception trapping
   enabled" is not a crash (healthy runs were stopped for it).
2. A mesh inside a folder the case holds is protected like the case's own.
3. A run cut off by the time limit says so, and is not followed by 60 s runs.
4. The fingerprint that stops a rerun of an unchanged case sees edits and
   ignores outputs.
10. A case on a solver's own OpenFOAM is not given the installed version's
   file names (shelf_demo_dafoam_20261019_1057).
11. A stuck run's Python processes say where they wait before it is stopped.
12. A file named under the case's own folder name is the case's file, and the
   run script the framework runs cannot be removed.
13. The review is shown the small text files the run wrote (a script's own
   report of the check that stopped it).
14. An edit naming an entry inside a file as a path, or returning the file
   unchanged, is reported as such.
15. A long JSON file is shown to the writer and reviewer as its outline.

Run: python scripts/test_case_loop_limits.py
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cfd_langgraph.foam_native import loop  # noqa: E402

FAILURES: list[str] = []


def check(name: str, got, want=True) -> None:
    ok = got == want
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + ("" if ok else f" — got {got!r}, want {want!r}"))
    if not ok:
        FAILURES.append(name)


print("== 1. the FPE-trapping banner is not a crash")
with tempfile.TemporaryDirectory() as tmp:
    case = Path(tmp)
    steps = "".join(f"Time = {i}\n" for i in range(1, 400))
    (case / "log.simpleDAFoam").write_text("Build  : v2506\ntrapFpe: Floating point exception trapping enabled\n"
                                           + steps)
    check("a healthy log has no crash", loop._crash_in_logs(case), "")
    check("and its error summary does not quote the banner as the error",
          "trapping enabled" in loop.error_summary(case), False)
    (case / "log.simpleDAFoam").write_text("trapFpe: Floating point exception trapping enabled\nTime = 1\n"
                                           "#0  Foam::error::printStack\nFloating point exception (core dumped)\n")
    check("a real floating point exception still is", loop._crash_in_logs(case).startswith("Floating point exception"))

print("== 2. meshes in nested folders are protected")
check("the case's own mesh", loop._in_a_mesh(Path("constant/polyMesh/points")))
check("a mesh in a folder the case holds", loop._in_a_mesh(Path("field_inversion/constant/polyMesh/points")))
check("not a file merely named like it", loop._in_a_mesh(Path("constant/transportProperties")), False)
check("nor a system file", loop._in_a_mesh(Path("system/polyMesh")), False)

print("== 3. a run stopped by the time limit says so")
with tempfile.TemporaryDirectory() as tmp:
    case = Path(tmp)
    # The limit is checked once a minute, so a 3 s limit stops a 3 min run after about 60 s.
    (case / "Allrun").write_text("#!/bin/bash\nfor i in $(seq 1 180); do echo $i > progress; sleep 1; done\n")
    (case / "Allrun").chmod(0o755)
    t = time.monotonic()
    rc, out, err, timed_out = loop.run_script(["./Allrun"], case, None, 3)
    check("the run is stopped at its limit", (timed_out, time.monotonic() - t < 120), (True, True))
    check("and the reason names the time limit", loop.TIME_LIMIT_NOTE in err)

print("== 4. the unchanged-case fingerprint")
with tempfile.TemporaryDirectory() as tmp:
    case = Path(tmp)
    for d in ("0", "constant/polyMesh", "system", "100", "processor0"):
        (case / d).mkdir(parents=True)
    (case / "0/U").write_text("internalField uniform (1 0 0);\n")
    (case / "system/controlDict").write_text("endTime 100;\n")
    (case / "constant/polyMesh/points").write_text("(0 0 0)\n")
    (case / "runScript.py").write_text("print('run')\n")
    (case / "scripts").mkdir()
    (case / "scripts/helper.py").write_text("x = 1\n")
    first = loop._inputs_fingerprint(case, ["scripts/helper.py"])
    (case / "100/U").write_text("results\n")
    (case / "log.run").write_text("Time = 1\n")
    (case / "processor0/U").write_text("decomposed\n")
    check("outputs of a run do not change it", loop._inputs_fingerprint(case, ["scripts/helper.py"]), first)
    (case / "runScript.py").write_text("print('fixed')\n")
    check("an edited run script does", loop._inputs_fingerprint(case, ["scripts/helper.py"]) != first)
    second = loop._inputs_fingerprint(case, ["scripts/helper.py"])
    (case / "scripts/helper.py").write_text("x = 2\n")
    check("and so does an edited file the review named", loop._inputs_fingerprint(case, ["scripts/helper.py"]) != second)

print("== 5. the setup check is told which other files the case holds")
with tempfile.TemporaryDirectory() as tmp:
    case = Path(tmp)
    for d in ("0", "system", "FFD", "constant/polyMesh", "processor0", "100"):
        (case / d).mkdir(parents=True)
    (case / "system/controlDict").write_text("FoamFile {}\napplication simpleFoam;\n")
    (case / "FFD/FFD.xyz").write_text("1\n2 2 2\n")
    (case / "constant/polyMesh/points").write_text("(0 0 0)\n")
    (case / "processor0/U").write_text("x\n")
    (case / "100/U").write_text("x\n")
    xml = loop._case_files_xml(case, given=set())
    listed = xml.split("<other_files>")[-1] if "<other_files>" in xml else ""
    check("a file outside the shown folders is listed", "FFD/FFD.xyz" in listed)
    check("but not the mesh, processor folders or results", any(n in listed for n in ("polyMesh", "processor0", "100/U")),
          False)

print("== 6. DAFoam's 'already decomposed' notice is not a fatal error")
with tempfile.TemporaryDirectory() as tmp:
    case = Path(tmp)
    notice = ("--> FOAM FATAL ERROR: (openfoam-2506)\nCase is already decomposed with 2 domains, use the "
              "-force option or manually\nremove processor directories before decomposing.\n\n"
              "    From decomposePar\n")
    steps = "".join(f"Time = {i}\n" for i in range(2, 400))
    (case / "log.simpleDAFoam").write_text("Time = 1\n" + notice + steps + "adjoint done\nEnd\n")
    check("the notice alone is not fatal", loop._has_fatal(case), False)
    check("nor a crash", loop._crash_in_logs(case), "")
    check("nor quoted as the run's error", "already decomposed" in loop.error_summary(case), False)
    sys.path.insert(0, str(ROOT / "scripts"))
    import stage_gate_audit as audit  # noqa: E402
    check("the final audit agrees", audit._scan_log_for_fatal(case / "log.simpleDAFoam"), False)
    (case / "log.simpleDAFoam").write_text(notice + "--> FOAM FATAL IO ERROR:\nUnknown RAS model type X\n")
    check("a real fatal error after it still is", loop._has_fatal(case), True)
    check("for the audit too", audit._scan_log_for_fatal(case / "log.simpleDAFoam"), True)

print("== 7. a long file is edited by exact replacements, not sent back whole")
from cfd_langgraph.foam_native import writer  # noqa: E402
import cfd_langgraph.utils as utils  # noqa: E402

text = "a = 1\nb = 2\nb = 2\nc = 3\n"
out, problems = writer.apply_replacements(text, [writer._Replacement(old="a = 1", new="a = 10")])
check("an exact unique replacement is applied", (out.startswith("a = 10\n"), problems), (True, []))
out, problems = writer.apply_replacements(text, [writer._Replacement(old="b = 2", new="b = 20"),
                                                 writer._Replacement(old="zzz", new="y")])
check("an ambiguous or absent old text is reported, not guessed", (out, len(problems)), (text, 2))

replies = [writer._Patch(replacements=[writer._Replacement(old="not in file", new="x")]),
           writer._Patch(replacements=[writer._Replacement(old="c = 3", new="c = 30")])]
prompts: list = []


class _Fake:
    def invoke(self, messages):
        prompts.append(messages[-1].content)
        return replies.pop(0)


saved_so = utils.structured_output
utils.structured_output = lambda llm, schema: _Fake()
try:
    got = writer.edit_file_by_patches(None, file_name="runScript.py", folder_name="", changes="c to 30",
                                      current_content=text, written_files_ctx="", case_solver="DASimpleFoam")
    check("a failed first try is sent back and the second applied", got.endswith("c = 30\n"))
    check("with the reason it failed", "could not be applied" in prompts[-1])
    replies[:] = [writer._Patch(replacements=[]), writer._Patch(replacements=[])]
    try:
        writer.edit_file_by_patches(None, file_name="runScript.py", folder_name="", changes="x",
                                    current_content=text, written_files_ctx="", case_solver="DASimpleFoam")
        check("an edit that never applies fails", False)
    except ValueError:
        check("an edit that never applies fails", True)
finally:
    utils.structured_output = saved_so

with tempfile.TemporaryDirectory() as tmp:
    case = Path(tmp)
    (case / "runScript.py").write_text("x = 1\n" * 3000)
    (case / "small.py").write_text("y = 1\n")
    used = []
    saved = (writer.edit_file, writer.edit_file_by_patches)
    writer.edit_file = lambda llm, **kw: used.append(("whole", kw["file_name"])) or kw["current_content"]
    writer.edit_file_by_patches = lambda llm, **kw: used.append(("patch", kw["file_name"])) or kw["current_content"]
    try:
        loop._apply_edits(None, case, [{"file": "runScript.py", "changes": "x", "delete": False},
                                       {"file": "small.py", "changes": "y", "delete": False}],
                          history=[], zero_template="", given=set(), mesh_seed_case_dir=None,
                          run_script_name="", subtasks=[], case_solver="DASimpleFoam")
    finally:
        writer.edit_file, writer.edit_file_by_patches = saved
    check("a long file goes to replacements, a short one is rewritten whole",
          sorted(used), [("patch", "runScript.py"), ("whole", "small.py")])

print("== 8. the writer and reviewer see what the cases a requirement names have written")
with tempfile.TemporaryDirectory() as tmp:
    cases = Path(tmp)
    me, truth = cases / "case_010", cases / "case_007"
    for d in (me, truth / "1000", truth / "0", truth / "processor0"):
        d.mkdir(parents=True)
    (truth / "synthetic_truth.json").write_text("{}")
    (truth / "log.simpleDAFoam").write_text("x")
    (truth / "1000/U").write_text("x")
    (truth / "0/U").write_text("x")
    note = loop._other_cases_note(me, "Read the reference velocity of case_007, then compare with case_003.")
    check("the named case's result file is listed by name", "synthetic_truth.json" in note)
    check("with the fields of its latest time folder", "time folder 1000/: U" in note)
    check("but not its logs or initial fields", ("log.simpleDAFoam" in note, "0/U" in note), (False, False))
    check("and a named case with no folder yet is said so", "case_003: no folder yet" in note)
    check("a requirement naming no other case adds nothing", loop._other_cases_note(me, "Run the channel."), "")

print("== 9. a small mesh's run counts as stuck sooner")
with tempfile.TemporaryDirectory() as tmp:
    case = Path(tmp)
    (case / "constant/polyMesh").mkdir(parents=True)
    (case / "constant/polyMesh/owner").write_text('FoamFile { note "nPoints:512  nCells:343  nFaces:1176"; }\n')
    check("a 343-cell case: 15 min", loop._stall_limit_s(case), 900.0)
    (case / "constant/polyMesh/owner").write_text('FoamFile { note "nPoints:1  nCells:157464  nFaces:1"; }\n')
    check("a 157k-cell case: 45 min", loop._stall_limit_s(case), 2700.0)
    check("a case with no mesh to read: 45 min", loop._stall_limit_s(case / "nothing"), 2700.0)

print("== 10. a case on a solver's own OpenFOAM keeps that OpenFOAM's file names")
import os  # noqa: E402
import shutil  # noqa: E402

os.environ.setdefault("CFD_SCIENTIST_SHELF", "/mnt/sda1/cfd_solver_shelf")
starter = Path(os.environ["CFD_SCIENTIST_SHELF"]) / "builds/DAFoam/own_openfoam-v2506/starters/convergent_channel_reg343"
if starter.is_dir():
    with tempfile.TemporaryDirectory() as tmp:
        # DAFoam's v2506 reads transportProperties; removed by OpenFOAM 10's
        # rule, DAFoam stopped at start-up (shelf_demo_dafoam_20261019_1057, case_004).
        case = Path(tmp) / "c"
        shutil.copytree(starter, case)
        (case / "constant/physicalProperties").write_text("nu 1.5e-05;\n")
        loop.drop_unread_transport_properties(case)
        check("a DAFoam case keeps constant/transportProperties", (case / "constant/transportProperties").is_file())
    from cfd_langgraph.foam_native import prompts  # noqa: E402

    check("the writer is told such a case follows its own OpenFOAM",
          "write that case for its own OpenFOAM version" in prompts.with_version_facts(""))
else:
    print("[SKIP] no DAFoam starter on this machine")

print("== 11. a stuck run's Python processes say where they wait")
made: list = []  # the stack folders this test's own run makes (live runs may make theirs meanwhile)
_real_mkdtemp = loop.tempfile.mkdtemp
loop.tempfile.mkdtemp = lambda *a, **k: made.append(_real_mkdtemp(*a, **k)) or made[-1]
with tempfile.TemporaryDirectory() as tmp:
    case = Path(tmp)
    (case / "wait.py").write_text("import time\n\n\ndef wait_forever():\n    while True:\n        time.sleep(1)\n\n\n"
                                  "wait_forever()\n")
    (case / "Allrun").write_text(f"#!/bin/bash\n{sys.executable} wait.py\n")
    (case / "Allrun").chmod(0o755)
    saved_stall = os.environ.get("CFD_SCIENTIST_RUN_STALL_S")
    os.environ["CFD_SCIENTIST_RUN_STALL_S"] = "3"
    try:
        rc, out, err, timed_out = loop.run_script(["./Allrun"], case, None, 600)
    finally:
        if saved_stall is None:
            os.environ.pop("CFD_SCIENTIST_RUN_STALL_S")
        else:
            os.environ["CFD_SCIENTIST_RUN_STALL_S"] = saved_stall
    check("the stuck run is stopped", timed_out)
    check("and the review is told the line it waited at",
          "wait.py line 6 in wait_forever: time.sleep(1)" in err)
check("a rank whose Python had shut down is said to be exiting",
      "it was exiting" in loop._where_waiting("", Path(".")))
loop.tempfile.mkdtemp = _real_mkdtemp
check("the hook leaves no folder behind", (len(made) > 0, [m for m in made if Path(m).exists()]), (True, []))

print("== 12. a file named under the case's own folder name is the case's file")
with tempfile.TemporaryDirectory() as tmp:
    case = Path(tmp) / "case_010"
    (case / "system").mkdir(parents=True)
    (case / "system/fvSolution").write_text("solvers {}\n")
    (case / "runScript.py").write_text("print('run')\n")
    check("the prefix is dropped", loop._without_case_name(case, Path("case_010/system/fvSolution")),
          Path("system/fvSolution"))
    check("a folder of another name is kept", loop._without_case_name(case, Path("field_inversion/0/U")),
          Path("field_inversion/0/U"))
    edited = []
    saved = writer.edit_file
    writer.edit_file = lambda llm, **kw: edited.append((kw["folder_name"], kw["file_name"])) or "solvers { p {} }\n"
    history: list = []
    try:
        loop._apply_edits(None, case, [{"file": "case_010/system/fvSolution", "changes": "x", "delete": False},
                                       {"file": "case_010/runScript.py", "changes": "duplicate", "delete": True}],
                          history=history, zero_template="", given=set(), mesh_seed_case_dir=None,
                          run_script_name="runScript.py", subtasks=[], case_solver="DASimpleFoam")
    finally:
        writer.edit_file = saved
    check("the edit goes to the case's own file", (edited, (case / "case_010").exists()), ([("system", "fvSolution")], False))
    check("and the run script the framework runs is not removed", (case / "runScript.py").is_file())
    check("which the review is told", any("which the framework runs" in h for h in history))

print("== 13. the review sees the small files the run wrote")
import json  # noqa: E402
with tempfile.TemporaryDirectory() as tmp:
    case = Path(tmp)
    for d in ("0", "system", "processor0", "100", ".openmdao_out", "runScript_out"):
        (case / d).mkdir()
    (case / "old_notes.txt").write_text("from before this run")
    os.utime(case / "old_notes.txt", (time.time() - 3600, time.time() - 3600))
    since = time.time() - 1
    (case / "results.json").write_text('{"validation": {"status": "failed", "check": "mesh_checksum"}}')
    (case / "runScript_out/summary.txt").write_text("ran 0 iterations")
    for name in ("Allrun.out", "run_result.json", "setup_check.json", "log.simpleDAFoam", "0/U.json",
                 "system/notes.txt", "processor0/x.json", "100/y.json", ".openmdao_out/z.json", "big.bin"):
        (case / name).write_text("harness or solver file")
    shown = loop.collect_error_logs(case, since=since)
    check("the run's own report is shown with its content",
          "results.json, written by this run" in shown and '"check": "mesh_checksum"' in shown)
    check("and a small report one folder down", "runScript_out/summary.txt, written by this run" in shown)
    check("but not files from before the run, nor harness, field, log or hidden files",
          [n for n in ("old_notes", "run_result.json, written", "setup_check.json, written", "Allrun.out, written",
                       "0/U.json", "system/notes", "processor0", "100/y", ".openmdao_out", "big.bin") if n in shown], [])
    check("without a run start nothing extra is shown", "written by this run" in loop.collect_error_logs(case), False)
    (case / ".truth.failure.json").write_text('{"failed": ["pristine_velocity_misfit"]}')
    (case / ".foamagent_state.json").write_text('{"case_info": {}}')
    shown = loop.collect_error_logs(case, since=since)
    check("a hidden report the run wrote is shown, the harness's own state file is not",
          (".truth.failure.json, written by this run" in shown, ".foamagent_state.json, written" in shown),
          (True, False))
    report = {"validations": [{"name": f"check_{i}", "passed": i % 2 == 0} for i in range(31)],
              "Uref": {"values": [[1.0, 2.0, 3.0]] * 4000}}
    (case / "truth_report.json").write_text(json.dumps(report))
    shown = loop.collect_error_logs(case, since=since)
    check("a long JSON report is shown by its checks, its arrays shortened",
          '"name":"check_3","passed":false' in shown and "<4000 values" in shown)

print("== 14. edits that cannot change anything say why")
with tempfile.TemporaryDirectory() as tmp:
    case = Path(tmp)
    (case / "constant").mkdir()
    (case / "constant/fvOptions").write_text("meshDiagnostic { type X; }\n")
    (case / "system").mkdir()
    (case / "system/controlDict").write_text("FoamFile {}\napplication simpleFoam;\nlibs (x);\n")
    calls = []
    saved = writer.edit_file
    writer.edit_file = lambda llm, **kw: calls.append(kw["file_name"]) or kw["current_content"]
    history = []
    try:
        loop._apply_edits(None, case, [{"file": "constant/fvOptions/meshDiagnostic", "changes": "remove it",
                                        "delete": False},
                                       {"file": "system/controlDict", "changes": "drop libs", "delete": False}],
                          history=history, zero_template="", given=set(), mesh_seed_case_dir=None,
                          run_script_name="", subtasks=[], case_solver="DASimpleFoam")
    finally:
        writer.edit_file = saved
    check("an entry named as a path inside a file is refused before any edit",
          (any("constant/fvOptions is a file" in h for h in history), "meshDiagnostic" in calls), (True, False))
    check("and the case is not broken by it", (case / "constant/fvOptions").is_file())
    check("an edit that returns the file unchanged says so",
          any("Editing system/controlDict returned it unchanged" in h for h in history))

print("== 15. a long JSON file is shown by its layout, not by its first numbers")
import json  # noqa: E402

truth = {"Uref": {"cell_count": 343, "values": [[10.0 + i * 1e-3, 1.0, 0.5] for i in range(343)]},
         "beta_true": {"values": [1.0] * 343, "observed_maximum": 1.05},
         "schema_version": "twin-v1", "runs": [{"rank": 0, "status": "ok"}, {"rank": 1, "status": "ok"}]}
truth["checks"] = {f"check_{i}": "passed" for i in range(300)}  # kept whole by the compact form
text = json.dumps(truth, indent=2)
view = loop._file_view(Path("synthetic_truth.json"), text, 2000)
check("every key is named, the last ones too, by its path",
      all(k in view for k in ("Uref:", "Uref.values:", "beta_true:", "schema_version:", "runs:")))
check("with list sizes and element types", "Uref.values: list[343] of list[3] of float" in view)
check("and the outline is bounded (150 lines), saying when keys are left out",
      (len(view.splitlines()) <= 152, "deeper keys not shown" in view), (True, True))
big = json.dumps({**truth, "Uref": {"values": [[10.0, 1.0, 0.5]] * 3430}}, indent=2)
wide = loop._file_view(Path("synthetic_truth.json"), big, 30000)
check("given room, it is shown with its values and its arrays shortened",
      '"observed_maximum":1.05' in wide and "<3430 values, first [10.0, 1.0, 0.5]>" in wide)
check("a short JSON file is shown whole", loop._file_view(Path("a.json"), '{"a": 1}', 2000), '{"a": 1}')
check("a long file of another kind is cut as before",
      loop._file_view(Path("a.txt"), "x" * 5000, 2000).endswith("[truncated]"))
with tempfile.TemporaryDirectory() as tmp:
    cases = Path(tmp)
    me, other = cases / "case_010", cases / "case_007"
    me.mkdir()
    other.mkdir()
    (me / "synthetic_truth.json").write_text(text)
    (other / "synthetic_truth.json").write_text(text)
    ctx = loop._written_files_ctx(me, [{"folder_name": "", "file_name": "synthetic_truth.json"}], 1)
    check("the writer sees the given file's layout", "beta_true:" in ctx and "schema_version:" in ctx)
    for name in ("run_result.json", "setup_check.json", "a_config.json", "b_config.json"):
        (other / name).write_text('{"x": 1}')
    note = loop._other_cases_note(me, "Read case_007's reference.")
    check("and the layout of the data files of the cases a requirement names",
          "case_007/synthetic_truth.json, its layout" in note and "list[343] of list[3] of float" in note)
    check("its largest data file first, not the framework's own records",
          ("run_result.json, its layout" in note, "setup_check.json, its layout" in note), (False, False))
deep = {"diag": {f"d{i}": {"x": {"y": list(range(10))}} for i in range(80)}, "mesh": {"cell_centres_m": [[0.1, 0.2, 0.3]] * 343}}
check("a key at the second level is shown though the first key's subtree is deep",
      "mesh.cell_centres_m: list[343]" in loop._json_outline(deep, max_lines=150))
real = Path("/mnt/sda1/somasn_cfd_scientist_experiments/shelf_demo_dafoam_20261019_1057/cases/case_010/synthetic_truth.json")
if real.is_file():
    real_view = loop._file_view(real, real.read_text(), 2000)
    check("the real 202 KB truth file shows its beta_true and Uref values",
          "beta_true:" in real_view and "Uref.values: list[343] of list[3] of float" in real_view)
from cfd_langgraph.foam_native import prompts as foam_prompts  # noqa: E402

check("the reviewer is told a run that stopped on its own check has failed",
      "stopped on a check of its own" in foam_prompts.REVIEWER_SYSTEM_PROMPT)

print("== 16. the manager is told when the setup check undoes its edit")
with tempfile.TemporaryDirectory() as tmp:
    case = Path(tmp)
    (case / "runScript.py").write_text("CENTRE = (0.60, 0.50, 0.50)\n")  # put back by the check
    (case / "other.py").write_text("x = 1\n")
    carried_bytes = {"runScript.py": b"CENTRE = (0.783, 0.156, 0.071)\n", "other.py": b"x = 1\n"}
    record = {"rounds": [{"items": [
        {"requirement": "Bump centre (0.60, 0.50, 0.50) m", "file": "runScript.py", "status": "mismatch",
         "found": "centre (0.783, 0.156, 0.071)"},
        {"requirement": "beta_peak 1.4", "file": "runScript.py", "status": "match", "found": ""},
        {"requirement": "x is 2", "file": "other.py", "status": "mismatch", "found": "x = 1"}]}]}
    got = loop._carried_changed(case, carried_bytes, record)
    check("the edit the check put back is reported with the requirement that decided it",
          [(g["file"], g["requirement"]) for g in got], [("runScript.py", "Bump centre (0.60, 0.50, 0.50) m")])
    check("nothing is reported without carried files", loop._carried_changed(case, {}, record), [])

print("== 17. a long script can still be edited; a long field cannot")
with tempfile.TemporaryDirectory() as tmp:
    case = Path(tmp)
    (case / "0").mkdir()
    (case / "runScript.py").write_text("x = 1\n" * 50000)  # 300 KB
    (case / "0/U").write_text("(1 0 0)\n" * 40000)  # 320 KB
    used = []
    saved = (writer.edit_file, writer.edit_file_by_patches)
    writer.edit_file = lambda llm, **kw: used.append(("whole", kw["file_name"])) or kw["current_content"] + "y\n"
    writer.edit_file_by_patches = lambda llm, **kw: used.append(("patch", kw["file_name"])) or kw["current_content"] + "y = 2\n"
    history = []
    try:
        loop._apply_edits(None, case, [{"file": "runScript.py", "changes": "x", "delete": False},
                                       {"file": "0/U", "changes": "y", "delete": False}],
                          history=history, zero_template="", given=set(), mesh_seed_case_dir=None,
                          run_script_name="runScript.py", subtasks=[], case_solver="DASimpleFoam")
    finally:
        writer.edit_file, writer.edit_file_by_patches = saved
    check("a 300 KB script is edited by exact replacements", used, [("patch", "runScript.py")])
    check("a 320 KB field is still refused", any("0/U, which is more than 200 KB" in h for h in history))

print("== 18. a log with a huge line does not make a huge prompt")
with tempfile.TemporaryDirectory() as tmp:
    case = Path(tmp)
    (case / "log.simpleDAFoam").write_text("Time = 1\n" + "{" + "x" * 6_600_000 + "}\n" + "Traceback (most recent call last):\nValueError: bad\n")
    shown = loop.collect_error_logs(case)
    check("the logs shown stay small", len(shown) < 100_000)
    check("and keep the error at the end", "ValueError: bad" in shown)
    check("saying what was cut", "characters of this line cut" in shown)

print()
print("FAILURES:", FAILURES if FAILURES else "none")
sys.exit(1 if FAILURES else 0)
