"""The third-party solver shelf: which build a version gets, that only
validated builds reach a study (environment, solver list, prompt text,
tutorial references), and how a validation verdict is reached.

Uses a registry and a shelf in a temporary folder; no OpenFOAM is run.

Run: python scripts/test_solver_shelf.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT / "src"), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from cfd_langgraph import openfoam_version as ov  # noqa: E402
from cfd_langgraph import solver_shelf as shelf  # noqa: E402

OF10 = {"ok": True, "fork": "openfoam.org", "version": "10", "label": "OpenFOAM 10 (openfoam.org)",
        "solver_style": "applications", "solver_applications": ["simpleFoam", "interFoam"]}
V2512 = {"ok": True, "fork": "openfoam.com", "version": "v2512", "label": "OpenFOAM v2512",
         "solver_style": "applications", "solver_applications": ["simpleFoam"]}

WAVES = """name: waveSolver
summary: Waves in a flume.
domains: [water waves, wave flume, coastal]
builds:
  - fork: openfoam.org
    versions: ["9", "10"]
    ref: abc
    build: ./allMake
provides:
  applications: [waveFoam]
usage: Run waveFoam.
"""
TURBINE = """name: rotorLib
summary: Actuator lines.
domains: [wind turbine, actuator line]
builds:
  - fork: openfoam.com
    versions: [v2412, v2512]
    ref: def
    build: wmake libso src
provides:
  applications: []
  libraries: [librotor]
usage: Load librotor.so.
"""
OWN = """name: ownSolver
summary: Brings its own OpenFOAM.
domains: [rarefied gas]
runtime:
  kind: own_openfoam
  wrapper: own-run
  programs:
    plainFoam: {}
    particleFoam: {before: [particleInit], function_objects: false}
    scriptFoam: {run: python runScript.py, inputs: [runScript.py, FFD]}
builds:
  - fork: any
    versions: [any]
    key: own_openfoam-v1
    ref: ghi
    build: ./build.sh
provides:
  applications: [own-run]
usage: Name the program in controlDict.
"""
# The wrapper marks what it runs, and runs it as it is when already inside.
OWN_RUN = """#!/bin/bash
[ "${OWN_INSIDE:-}" = 1 ] && exec "$@"
exec env OWN_INSIDE=1 "$@"
"""


def _setup(tmp: Path) -> None:
    reg = tmp / "solvers"
    for name, text in (("waveSolver", WAVES), ("rotorLib", TURBINE), ("ownSolver", OWN)):
        (reg / name).mkdir(parents=True)
        (reg / name / "solver.yaml").write_text(text)
    shelf.REGISTRY_DIR = reg
    import os

    os.environ["CFD_SCIENTIST_SHELF"] = str(tmp / "shelf")


def _build(name: str, profile, status: str, tutorial: str = "") -> Path:
    b = shelf.build_dir(name, profile)
    for d in ("bin", "lib", "src/tutorials"):
        (b / d).mkdir(parents=True, exist_ok=True)
    if tutorial:
        case = b / "src" / "tutorials" / tutorial
        (case / "system").mkdir(parents=True)
        (case / "system" / "controlDict").write_text("application waveFoam;\n")
        (case / "constant").mkdir()
    shelf.write_record(name, profile, status=status)
    return b


def test_build_entry() -> None:
    wave = shelf.spec("waveSolver")
    assert shelf.build_entry(wave, OF10)["match"] == "exact"
    assert shelf.build_entry(wave, {**OF10, "version": "13"})["match"] == "nearest"
    rotor = shelf.spec("rotorlib")              # names match whatever the case
    assert shelf.build_entry(rotor, V2512)["match"] == "exact"


def test_only_validated_builds_reach_a_study() -> None:
    assert shelf.usable(OF10) == {} and shelf.describe(OF10) == ""
    b = _build("waveSolver", OF10, "built", tutorial="flume")
    assert shelf.usable(OF10) == {}, "a build that has not passed its checks is not offered"
    assert "waveFoam" not in ov.solver_choices(OF10)
    shelf.write_record("waveSolver", OF10, status="failed")
    assert shelf.applications(OF10) == []
    shelf.write_record("waveSolver", OF10, status="validated")
    assert shelf.applications(OF10) == ["waveFoam"]
    assert ov.solver_choices(OF10) == ["simpleFoam", "interFoam", "waveFoam"]
    env = shelf.env_additions(OF10)
    assert env["PATH"] == [str(b / "bin")] and env["LD_LIBRARY_PATH"] == [str(b / "lib")]
    assert shelf.shell_exports(OF10).startswith(f'export PATH="{b / "bin"}:$PATH"')
    text = shelf.describe(OF10)
    assert "waveSolver" in text and "waveFoam" in text and "Run waveFoam." in text
    # Not on another installation.
    assert shelf.usable(V2512) == {}


def test_which_solvers_a_case_calls_for() -> None:
    _build("rotorLib", V2512, "validated")
    names = lambda found: [s["name"] for s in found]  # noqa: E731
    assert names(shelf.solvers_for("a wake behind a rotor with rotorLib", "pimpleFoam", V2512)) == ["rotorLib"]
    assert names(shelf.solvers_for("a wake behind a rotor", "pimpleFoam", V2512)) == []
    assert names(shelf.solvers_for("regular waves", "waveFoam", OF10)) == ["waveSolver"]
    listed = shelf.status(OF10, "breaking water waves in a flume")
    assert listed[0]["name"] == "waveSolver" and listed[0]["matches_topic"] and listed[0]["usable"]
    rotor = next(s for s in listed if s["name"] == "rotorLib")
    assert rotor["status"] == "not built" and rotor["build_for_this_version"] == "nearest"


def test_ported_tutorial_is_the_reference() -> None:
    solver = shelf.usable(OF10)["waveSolver"]
    original = shelf.build_dir("waveSolver", OF10) / "src" / "tutorials" / "flume"
    assert shelf.tutorial_cases(solver) == [original]
    ported = shelf.build_dir("waveSolver", OF10) / "tutorials_ported" / "tutorials" / "flume"
    (ported / "system").mkdir(parents=True)
    (ported.parent / "flume.translation.json").write_text(json.dumps({"status": "failed"}))
    assert shelf.tutorial_cases(solver) == [original], "a failed port is not used"
    (ported.parent / "flume.translation.json").write_text(json.dumps({"status": "translated"}))
    assert shelf.tutorial_cases(solver) == [ported]


def test_verdict() -> None:
    from shelf_validate import judge, verdict

    checks = [{"name": "a"}, {"name": "b"}]
    ok = {"ran": True, "passed": True, "expect": [{"passed": True}]}
    status, results = verdict([{"name": "a", **ok}], checks)
    assert status == "failed" and results[-1] == {"name": "b", "passed": False, "ran": False,
                                                  "error": "not run yet"}
    assert verdict([{"name": "a", **ok}, {"name": "b", **ok}], checks)[0] == "validated"
    no_ref = {"ran": True, "passed": False, "expect": []}
    assert verdict([{"name": "a", **no_ref}, {"name": "b", **no_ref}], checks)[0] == "runs"
    got = judge({"x": 1.04, "y": 0.2, "z": float("nan")},
                [{"quantity": "x", "reference": 1.0, "rel_tol": 0.05},
                 {"quantity": "y", "reference": 0.0, "abs_tol": 0.1},
                 {"quantity": "z", "reference": 1.0, "rel_tol": 1.0},
                 {"quantity": "w", "reference": 1.0, "rel_tol": 1.0}])
    assert [e["passed"] for e in got] == [True, False, False, False]
    assert got[2]["why"] == got[3]["why"] == "not measured"


def _case(root: Path, application: str, *, transient: bool = False, block_mesh: bool = True) -> Path:
    case = root / application
    for d in ("system", "constant/polyMesh"):
        (case / d).mkdir(parents=True, exist_ok=True)
    (case / "system" / "controlDict").write_text(
        f"application {application};\nendTime 1;\ndeltaT 0.01;\nwriteInterval 0.1;\n")
    (case / "system" / "fvSchemes").write_text(
        "ddtSchemes\n{\n    default " + ("Euler" if transient else "steadyState") + ";\n}\n")
    if block_mesh:
        (case / "system" / "blockMeshDict").write_text("vertices ();\n")
    return case


def _runs_decompose(text: str) -> bool:
    """Whether an Allrun runs decomposePar (naming system/decomposeParDict is not running it)."""
    return any("decomposePar" in line.split()[:2] for line in text.splitlines() if line.strip())


def test_cases_on_a_solvers_own_openfoam() -> None:
    from cfd_langgraph import mesh_assessment
    from cfd_langgraph.foam_native import allrun, loop

    b = _build("ownSolver", OF10, "validated")
    wrapper = b / "bin" / "own-run"
    wrapper.write_text(OWN_RUN)
    wrapper.chmod(0o755)
    assert {"plainFoam", "particleFoam", "scriptFoam"} <= set(shelf.applications(OF10))
    assert "own-run" not in shelf.applications(OF10), "the wrapper is not a case's program"

    root = Path(tempfile.mkdtemp())
    stock = _case(root, "simpleFoam", transient=True)
    particle = _case(root, "particleFoam", transient=True)
    script = _case(root, "scriptFoam", block_mesh=False)
    assert shelf.own_runtime(stock, OF10) == {}
    got = shelf.own_runtime(particle, OF10)
    assert got["wrapper"] == str(wrapper) and got["before"] == ["particleInit"] and got["function_objects"] is False
    assert shelf.in_runtime(["./Allrun"], stock) == ["./Allrun"]
    assert shelf.in_runtime(["./Allrun"], particle) == [str(wrapper), "./Allrun"]

    text = allrun.build_mesh_and_solve_allrun("particleFoam", case_dir=particle)
    order = [text.index(s) for s in ("runApplication blockMesh", "runApplication checkMesh",
                                      "runApplication particleInit", "decomposePar", "runParallel particleFoam")]
    assert order == sorted(order), text
    text = allrun.build_mesh_and_solve_allrun("scriptFoam", case_dir=script)
    assert "blockMesh" not in text, "a shipped mesh is used as it is"
    assert "mpirun -np $np python -m mpi4py runScript.py > log.scriptFoam 2>&1 && echo End >> log.scriptFoam" in text
    assert '[ -n "$np" ] || np=1' in text, "with no decomposeParDict the script runs on one rank"
    assert not _runs_decompose(text), "one rank: nothing to decompose"
    # The case says how many ranks its script runs on, as DAFoam's own cases do.
    (script / "system" / "decomposeParDict").write_text("numberOfSubdomains 4;\nmethod scotch;\n")
    text = allrun.build_mesh_and_solve_allrun("scriptFoam", case_dir=script)
    # The script decomposes the case itself (pyDAFoam's runDecomposePar):
    # only an earlier run's processor folders are cleared before it.
    assert '[ -n "$np" ] || np=4' in text, text
    assert text.index("rm -rf processor*") < text.index(
        "mpirun -np $np python -m mpi4py runScript.py"), text
    # Read when the Allrun runs: a numberOfSubdomains the review changes later
    # is the one used (the review may not edit this Allrun).
    import subprocess as _sp
    import tempfile as _tf
    with _tf.TemporaryDirectory() as bindir:
        stub = Path(bindir) / "foamDictionary"
        stub.write_text("#!/bin/sh\necho 2\n")
        stub.chmod(0o755)
        lines = [l for l in text.splitlines() if l.startswith(("np=", "[ -n"))]
        got = _sp.run(["sh", "-c", "\n".join(lines) + "\necho $np"], capture_output=True, text=True,
                      env={"PATH": f"{bindir}:/usr/bin:/bin"}).stdout.strip()
        assert got == "2", got
        # A script that writes no time folder leaves reconstructPar nothing to
        # do; its non-zero exit must not fail the whole Allrun.
        fail = Path(bindir) / "reconstructPar"
        fail.write_text("#!/bin/sh\necho 'No times selected'\nexit 1\n")
        fail.chmod(0o755)
        line = [l for l in text.splitlines() if "reconstructPar" in l][0]
        with _tf.TemporaryDirectory() as casedir:
            (Path(casedir) / "processor0").mkdir()
            rc = _sp.run(["sh", "-c", "runApplication() { \"$@\"; }\n" + line], cwd=casedir,
                         capture_output=True, text=True, env={"PATH": f"{bindir}:/usr/bin:/bin"}).returncode
        assert rc == 0, rc
    assert not _runs_decompose(text), text
    assert not _runs_decompose(allrun.build_allrun_script(
        "blockMesh\ndecomposePar\nscriptFoam", "scriptFoam", case_dir=script)), "a listed decomposePar is dropped"
    assert "mpirun -np $np python -m mpi4py runScript.py" in allrun.build_allrun_script(
        "blockMesh\nscriptFoam", "scriptFoam", case_dir=script)
    (script / "system" / "decomposeParDict").unlink()
    assert "runApplication scriptFoam" not in text and "runParallel" not in text
    text = allrun.build_allrun_script("blockMesh\nscriptFoam", "scriptFoam", case_dir=script)
    assert "runApplication scriptFoam" not in text and "python -m mpi4py runScript.py" in text
    text = allrun.build_allrun_script("blockMesh\nparticleFoam", "particleFoam", case_dir=particle)
    assert text.index("runApplication particleInit") < text.index("runApplication particleFoam"), text

    assert loop._prescribed_mesh(script) and not loop._prescribed_mesh(particle)
    stock_shipped = _case(root / "shipped", "simpleFoam", block_mesh=False)
    assert not loop._prescribed_mesh(stock_shipped), "only a solver's own case takes its mesh as given"

    assert mesh_assessment.ensure_time_averaging(stock)
    assert not mesh_assessment.ensure_time_averaging(particle), "this solver forbids added function objects"

    # With no monitored histories the reviewer is shown the end of the solver's
    # own log, which is where such a case says it converged.
    (script / "log.scriptFoam").write_text("Time = 373\nMinimal residual 9.99e-09 satisfied the prescribed tolerance 1e-08\nEnd\n")
    shown = mesh_assessment._monitored_text(script, [], None)
    assert "satisfied the prescribed tolerance" in shown and "log.scriptFoam" in shown, shown

    # The case's script runs inside the wrapper, and with nothing to read:
    # `cat` returns at once instead of waiting on the terminal.
    (particle / "Allrun").write_text("#!/bin/sh\ncat\necho inside=$OWN_INSIDE\n" + str(wrapper) + " true\n")
    (particle / "Allrun").chmod(0o755)
    rc, out, err, timed_out = loop.run_script(["./Allrun"], particle, None, 30)
    assert (rc, out.strip(), timed_out) == (0, "inside=1", False), (rc, out, err)

    # A copied case runs the way its source does: its own Allrun and what
    # that uses come along, its run outputs do not. On a solver's own OpenFOAM
    # its Allrun runs in the wrapper like any script; only a case solved by a
    # command in place of its program keeps the written Allrun.
    (stock / "Allrun").write_text("#!/bin/sh\nblockMesh\nsnappyHexMesh -overwrite\nsetFields\n")
    for d in ("0.orig", "100", "processor0", "postProcessing"):
        (stock / d).mkdir(exist_ok=True)
    for f in ("helper.py", "log.simpleFoam", "Allrun.out"):
        (stock / f).write_text("x")
    copy = root / "copy"
    loop._copy_case_files(stock, copy)
    assert loop._copy_own_allrun(stock, copy)
    names = {p.name for p in copy.iterdir()}
    assert {"Allrun", "0.orig", "helper.py"} <= names, names
    assert not names & {"100", "processor0", "postProcessing", "log.simpleFoam", "Allrun.out"}, names
    assert os.access(copy / "Allrun", os.X_OK)
    particle_copy = root / "particle_copy"
    loop._copy_case_files(particle, particle_copy)
    assert loop._copy_own_allrun(particle, particle_copy)
    (script / "Allrun").write_text("#!/bin/sh\npython runScript.py\n")
    script_copy = root / "script_copy"
    loop._copy_case_files(script, script_copy)
    assert not loop._copy_own_allrun(script, script_copy)
    text = allrun.build_mesh_and_solve_allrun("simpleFoam", case_dir=stock, mesh_given=True)
    assert "blockMesh" not in text and "runApplication checkMesh" in text, text


def main() -> int:
    tests = [test_build_entry, test_only_validated_builds_reach_a_study, test_which_solvers_a_case_calls_for,
             test_ported_tutorial_is_the_reference, test_verdict, test_cases_on_a_solvers_own_openfoam]
    with tempfile.TemporaryDirectory() as tmp:
        _setup(Path(tmp))
        failed = 0
        for t in tests:
            try:
                t()
                print(f"ok   {t.__name__}")
            except AssertionError as exc:
                failed += 1
                print(f"FAIL {t.__name__}: {exc}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
