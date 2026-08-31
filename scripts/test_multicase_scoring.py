#!/usr/bin/env python3
"""Multi-case evaluation: running one candidate's model over a declared set
of cases and scoring it on their mean.

The search was single-case throughout: oed_run_code_mod_candidate built one
case and oed_score_candidate scored one case_dir. That fits a study whose
objective is one flow, and cannot express an objective that is generalisation
across several — a benchmark's held-out test set, say, where the score is the
mean over cases the model has never seen.

No OpenFOAM run and no LLM call: the helpers are driven directly against real
ported case directories where those exist, and the scoring guards are driven
through the real tool.

    python3 scripts/test_multicase_scoring.py
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cfd_langgraph.config import Settings  # noqa: E402
import cfd_langgraph.manager.tools as tools  # noqa: E402

FAILURES: list[str] = []

# Real ported benchmark cases when they are present; the OpenFOAM-shaped
# fixture below when they are not, so this suite runs anywhere.
PORTED = Path("/tmp/claude-166195/-home-somasn-Desktop/"
              "1648fc3a-6f7a-4fa3-afbc-dc04a697f734/scratchpad/ported")


def check(name: str, cond: object, detail: str = "") -> None:
    if cond:
        print(f"[PASS] {name}")
    else:
        FAILURES.append(name)
        print(f"[FAIL] {name}" + (f" — {detail}" if detail else ""))


def make_case(root: Path, application: str = "simpleFoam", ras: str = "kOmegaSST") -> Path:
    """A directory with just enough OpenFOAM shape for these helpers."""
    (root / "system").mkdir(parents=True, exist_ok=True)
    (root / "constant").mkdir(parents=True, exist_ok=True)
    (root / "0").mkdir(parents=True, exist_ok=True)
    (root / "system" / "controlDict").write_text(
        f"application     {application};\nendTime         100;\n"
    )
    (root / "constant" / "momentumTransport").write_text(
        f"simulationType RAS;\nRAS\n{{\n    RASModel        {ras};\n}}\n"
    )
    return root


def real_or_fixture(relative: str, fallback_root: Path) -> Path:
    candidate = PORTED / relative
    if (candidate / "system" / "controlDict").is_file():
        return candidate
    return make_case(fallback_root / relative.replace("/", "_"))


def test_declared_cases_resolve() -> None:
    disc = Path(tempfile.mkdtemp()) / "open_ended_discovery"
    disc.mkdir(parents=True)
    (disc / "search_config.json").write_text(json.dumps({"topic": "t", "total_budget": 10}))
    check(
        "a study that declares nothing stays single-case",
        tools._evaluation_cases(disc) == [],
    )

    scratch = Path(tempfile.mkdtemp())
    cases = [
        real_or_fixture("Parm_PH_29/alpha_15/alpha_15_13929_4048", scratch),
        real_or_fixture("DUCT/AR_1_Ret_360", scratch),
    ]
    (disc / "search_config.json").write_text(
        json.dumps({"topic": "t", "total_budget": 10,
                    "evaluation_cases": [str(c) for c in cases]})
    )
    resolved = tools._evaluation_cases(disc)
    check(
        "declared evaluation cases resolve to real directories",
        len(resolved) == 2 and all(p.is_dir() for p in resolved),
        detail=str(resolved),
    )


def test_case_application_is_read_not_assumed() -> None:
    scratch = Path(tempfile.mkdtemp())
    case = real_or_fixture("Parm_PH_29/alpha_15/alpha_15_13929_4048", scratch)
    check(
        "the solver comes from the case's own controlDict",
        tools._case_application(case) == "simpleFoam",
        detail=repr(tools._case_application(case)),
    )
    other = make_case(scratch / "compressible", application="rhoCentralFoam")
    check(
        "a different solver is read, not defaulted away",
        tools._case_application(other) == "rhoCentralFoam",
    )
    check("a missing case yields no application", tools._case_application(scratch / "nope") == "")


def test_model_install_repoints_the_case() -> None:
    scratch = Path(tempfile.mkdtemp())
    source = make_case(scratch / "source")
    model = source / "customModels" / "MyModel"
    model.mkdir(parents=True)
    (model / "MyModel.C").write_text("// model source")
    (model / "libMyModel.so").write_bytes(b"\x7fELF not really")

    target = make_case(scratch / "target")
    error = tools._install_model_into_case(source, target, "MyModel")
    check("installing a compiled model succeeds", error is None, detail=str(error))
    check("the model source is copied in", (target / "customModels" / "MyModel" / "MyModel.C").is_file())

    transport = (target / "constant" / "momentumTransport").read_text()
    check(
        "RASModel is repointed at the new model",
        "MyModel;" in transport,
        detail=[line for line in transport.splitlines() if "RASModel" in line],
    )
    control = (target / "system" / "controlDict").read_text()
    check("the built library is loaded via libs", "libMyModel.so" in control)
    check(
        "a source case with no customModels is an error, not a silent no-op",
        tools._install_model_into_case(scratch / "absent", target, "X") is not None,
    )


def _scoring_fixture(evaluation_cases: list[str] | None):
    out = Path(tempfile.mkdtemp())
    disc = out / "open_ended_discovery"
    disc.mkdir(parents=True)
    config = {"topic": "t", "total_budget": 10, "baseline_direction": "min"}
    if evaluation_cases:
        config["evaluation_cases"] = evaluation_cases
    (disc / "search_config.json").write_text(json.dumps(config))
    (disc / "baseline_score.json").write_text(
        json.dumps({"metric": "m", "value": 0.004, "direction": "min", "verified": True})
    )
    candidate = disc / "cand_x"
    (candidate / "case").mkdir(parents=True)
    (candidate / "agentic_result.json").write_text(json.dumps({
        "status": "OK", "case_dir": str((candidate / "case").resolve()),
        "compile_ok": True, "converged": True, "compiled_model_name": "M",
        "solver_invocations": 5,
    }))
    built = tools.build_manager_tools(Settings(), out)
    score = [f for f in built["oed_candidate_tools"]
             if getattr(f, "__name__", "") == "oed_score_candidate"][0]
    return score, candidate


def test_a_partial_mean_is_refused() -> None:
    """A mean over six cases is not comparable with a mean over eight.

    Averaging whatever happened to work rewards a candidate for crashing on
    the cases it finds hard, and the archive stores one number with no way to
    tell the two apart afterwards.
    """
    scratch = Path(tempfile.mkdtemp())
    declared = [
        str(real_or_fixture("Parm_PH_29/alpha_15/alpha_15_13929_4048", scratch)),
        str(real_or_fixture("DUCT/AR_1_Ret_360", scratch)),
    ]
    score, candidate = _scoring_fixture(declared)
    result = score(candidate_dir=str(candidate), case_dir=str(candidate / "case"),
                   action_type="code_mod", variant_name="x", model_description="d",
                   case_dirs=[str(candidate / "case")])
    check(
        "scoring 1 of 2 declared cases is refused",
        result.get("ok") is False and "not comparable" in (result.get("error") or ""),
        detail=str(result.get("error"))[:120],
    )


def test_scoring_outside_the_candidate_is_refused() -> None:
    """Otherwise one candidate can be scored on another's fields, or on the
    pristine declared case rather than its own replica of it."""
    scratch = Path(tempfile.mkdtemp())
    shared = str(real_or_fixture("Parm_PH_29/alpha_15/alpha_15_13929_4048", scratch))
    score, candidate = _scoring_fixture([shared, shared])
    result = score(candidate_dir=str(candidate), case_dir=str(candidate / "case"),
                   action_type="code_mod", variant_name="x", model_description="d",
                   case_dirs=[str(candidate / "case"), shared])
    check(
        "a case_dir outside the candidate directory is refused",
        result.get("ok") is False and "inside the candidate" in (result.get("error") or ""),
        detail=str(result.get("error"))[:120],
    )

    result = score(candidate_dir=str(candidate), case_dir=str(candidate / "case"),
                   action_type="code_mod", variant_name="x", model_description="d",
                   case_dirs=[str(candidate / "case"), str(candidate / "missing")])
    check(
        "a case_dir that does not exist is refused",
        result.get("ok") is False and "do not exist" in (result.get("error") or ""),
        detail=str(result.get("error"))[:120],
    )


def test_single_case_studies_are_untouched() -> None:
    score, candidate = _scoring_fixture(None)
    result = score(candidate_dir=str(candidate), case_dir=str(candidate / "case"),
                   action_type="code_mod", variant_name="x", model_description="d")
    check(
        "a study with no declared cases scores exactly as before",
        "not comparable" not in str(result.get("error", "")),
        detail=str(result)[:140],
    )
    check("and carries no multi-case bookkeeping", result.get("per_case_scores") is None)


# --- prescribed mesh --------------------------------------------------------
# Regression for run closure_20260824_codex, which stalled for 11 hours without
# launching a single candidate. oed_setup_search accepts a baseline only if it
# is a converged selected_level from run_mesh_gate; the closure benchmark
# supplies every mesh and forbids both modifying one and running a mesh
# independence study, so no such result could legitimately exist. The run
# correctly refused to bypass the guard and wrote a blocker report instead —
# and telling it to continue could not help, because the refusal was in code.
#
# The gate's purpose is that candidates are never scored on a mesh whose
# convergence nobody established. A prescribed mesh satisfies that differently:
# it is not the study's mesh to justify. So the bypass is allowed only for a
# case inside the starter folder — supplied input the study did not create and
# therefore cannot have refined.

def _setup_tool(starter: str):
    import json
    out = Path(tempfile.mkdtemp())
    (out / "starter_understanding.json").write_text(json.dumps({"starter_dir": starter}))
    built = tools.build_manager_tools(Settings(), out)
    return [f for f in built["manager_tools"]
            if getattr(f, "__name__", "") == "oed_setup_search"][0]


def test_prescribed_mesh_needs_a_stated_reason() -> None:
    starter = Path("starter_closure_challenge")
    if not starter.is_dir():
        check("prescribed-mesh tests need starter_closure_challenge", True,
              detail="skipped: starter folder absent")
        return
    setup = _setup_tool(str(starter))
    supplied = str((starter / "cases" / "training" / "PH_Breuer").resolve())
    result = setup(topic="t", baseline_case_dir=supplied, total_budget=10)
    check(
        "without a reason the mesh gate still refuses",
        result.get("ok") is False and "prescribed_mesh_reason" in (result.get("error") or ""),
        detail=str(result.get("error"))[:140],
    )


def test_prescribed_mesh_refuses_a_self_built_case() -> None:
    starter = Path("starter_closure_challenge")
    if not starter.is_dir():
        check("prescribed-mesh provenance test", True, detail="skipped: starter folder absent")
        return
    setup = _setup_tool(str(starter))
    elsewhere = str(Path(tempfile.mkdtemp()))
    result = setup(topic="t", baseline_case_dir=elsewhere, total_budget=10,
                   prescribed_mesh_reason="the benchmark prescribes the mesh")
    check(
        "a case outside the starter folder is refused even with a reason",
        result.get("ok") is False and "not inside the starter" in (result.get("error") or ""),
        detail=str(result.get("error"))[:140],
    )


def test_prescribed_mesh_accepts_supplied_input() -> None:
    starter = Path("starter_closure_challenge")
    if not starter.is_dir():
        check("prescribed-mesh acceptance test", True, detail="skipped: starter folder absent")
        return
    setup = _setup_tool(str(starter))
    supplied = str((starter / "cases" / "training" / "PH_Breuer").resolve())
    result = setup(topic="t", baseline_case_dir=supplied, total_budget=10,
                   prescribed_mesh_reason="The challenge supplies every mesh and grades on it.")
    check(
        "a supplied starter case with a reason gets past the mesh gate",
        not (result.get("ok") is False and "mesh gate" in str(result.get("error", ""))),
        detail=str(result.get("error"))[:160],
    )



def test_baseline_must_cover_the_same_cases() -> None:
    """A 32-case candidate mean against a one-case baseline is not a comparison.

    Regression for run closure_20260825_codex: oed_setup_search registered all
    32 evaluation cases and the prescribed-mesh exception correctly, then took
    its baseline from CBFS alone — 0.00783, roughly an order of magnitude below
    the 32-case mean. Every candidate would have been measured against the wrong
    number and looked catastrophic for purely arithmetic reasons. The run caught
    it and refused to continue, which was the right call; this guard makes the
    mismatch impossible to reach.
    """
    import json
    scratch = Path(tempfile.mkdtemp())
    declared = [
        str(real_or_fixture("Parm_PH_29/alpha_15/alpha_15_13929_4048", scratch)),
        str(real_or_fixture("DUCT/AR_1_Ret_360", scratch)),
    ]
    score, candidate = _scoring_fixture(declared)
    disc = candidate.parent
    (candidate / "a").mkdir(exist_ok=True)
    (candidate / "b").mkdir(exist_ok=True)
    pair = [str(candidate / "a"), str(candidate / "b")]

    (disc / "baseline_score.json").write_text(json.dumps(
        {"metric": "m", "value": 0.0078, "direction": "min", "verified": True}))
    result = score(candidate_dir=str(candidate), case_dir=str(candidate / "case"),
                   action_type="code_mod", variant_name="x", model_description="d",
                   case_dirs=pair)
    check(
        "a one-case baseline is refused for a multi-case candidate",
        result.get("ok") is False and "baseline covers" in (result.get("error") or ""),
        detail=str(result.get("error"))[:150],
    )

    (disc / "baseline_score.json").write_text(json.dumps(
        {"metric": "m", "value": 0.10, "direction": "min", "verified": True,
         "evaluation_cases_scored": 2}))
    result = score(candidate_dir=str(candidate), case_dir=str(candidate / "case"),
                   action_type="code_mod", variant_name="x", model_description="d",
                   case_dirs=pair)
    check(
        "a baseline covering the same case count passes",
        "baseline covers" not in str(result.get("error", "")),
        detail=str(result.get("error"))[:150],
    )


def test_single_case_studies_skip_the_baseline_guard() -> None:
    score, candidate = _scoring_fixture(None)
    result = score(candidate_dir=str(candidate), case_dir=str(candidate / "case"),
                   action_type="code_mod", variant_name="x", model_description="d")
    check(
        "a study with no declared cases never trips the baseline guard",
        "baseline covers" not in str(result.get("error", "")),
        detail=str(result.get("error"))[:150],
    )



def test_model_install_is_verified_not_assumed() -> None:
    """Three ways the installer silently produced a case that ran the WRONG model.

    All three were found by auditing rather than by a failing test, because the
    original tests asserted on strings in the file instead of on OpenFOAM being
    able to use the result:

      1. `libs` was written as a bare filename. OpenFOAM dlopens the string as
         given, so a case-local .so is never found. Every working candidate in
         this repository writes an absolute path.
      2. The first `libs (...)` line in controlDict was rewritten — but
         controlDict carries `libs` inside functionObject blocks too, so that
         clobbered a functionObject and still left the model unloaded.
      3. Only `RASModel` was substituted. OpenFOAM 10 spells the key `model` in
         constant/momentumTransport, so on an OF10 case nothing was replaced and
         the case ran its ORIGINAL closure while the framework recorded the
         result as the candidate's score.
    """
    import shutil

    source = Path("runs/oed_20260823_opus_low/open_ended_discovery/"
                  "cand_sa_neg_cb2_diff/sa_neg_cb2_diff")
    base = Path("runs/oed_20260823_opus_low/mesh_gate/sa_qcr_periodic_hill/baseline")
    if not (source / "customModels").is_dir() or not base.is_dir():
        check("model-install tests need a real compiled candidate", True,
              detail="skipped: reference run absent")
        return

    scratch = Path(tempfile.mkdtemp())

    target = scratch / "of10"
    shutil.copytree(base, target)
    error = tools._install_model_into_case(source, target, "SANegCb2Diff")
    check("installing into an OpenFOAM 10 case succeeds", error is None, detail=str(error))
    transport = (target / "constant" / "momentumTransport").read_text()
    check(
        "the OF10 `model` key is repointed, not just `RASModel`",
        "SANegCb2Diff;" in transport,
        detail=[l for l in transport.splitlines() if "model" in l.lower()][:2],
    )
    control = (target / "system" / "controlDict").read_text()
    check(
        "the library is referenced by absolute path, not a bare filename",
        str(target.resolve()) in control and ".so" in control,
        detail=[l for l in control.splitlines() if "libs" in l][:2],
    )

    bare = scratch / "bare"
    (bare / "system").mkdir(parents=True)
    (bare / "constant").mkdir()
    (bare / "system" / "controlDict").write_text("application simpleFoam;\n")
    (bare / "constant" / "momentumTransport").write_text(
        "simulationType RAS;\nRAS\n{\n    turbulence on;\n}\n"
    )
    error = tools._install_model_into_case(source, bare, "X")
    check(
        "a case with no model entry is refused rather than left on its original closure",
        error is not None and "Refusing" in str(error),
        detail=str(error)[:120],
    )

    with_fo = scratch / "fo"
    shutil.copytree(base, with_fo)
    control_path = with_fo / "system" / "controlDict"
    control_path.write_text(
        control_path.read_text().rstrip()
        + '\n\nfunctions\n{\n  probes\n  {\n    libs            ("libsampling.so");\n  }\n}\n'
    )
    error = tools._install_model_into_case(source, with_fo, "SANegCb2Diff")
    text = control_path.read_text()
    check("install works alongside a functionObject libs line", error is None, detail=str(error))
    check("and leaves that functionObject's own libs intact", "libsampling.so" in text)



def _cost_fixture(build_runs: int, eval_runs: int, cases: int):
    import json
    out = Path(tempfile.mkdtemp())
    disc = out / "open_ended_discovery"
    disc.mkdir(parents=True)
    config = {"topic": "t", "total_budget": 500, "baseline_direction": "min"}
    candidate = disc / "cand_x"
    (candidate / "case").mkdir(parents=True)
    case_dirs = []
    for i in range(cases):
        d = candidate / f"e{i}"
        d.mkdir()
        case_dirs.append(str(d))
    if cases:
        config["evaluation_cases"] = case_dirs
    (disc / "search_config.json").write_text(json.dumps(config))
    (disc / "baseline_score.json").write_text(json.dumps({
        "metric": "m", "value": 0.1, "direction": "min", "verified": True,
        "evaluation_cases_scored": cases or None,
    }))
    (candidate / "agentic_result.json").write_text(json.dumps({
        "status": "OK", "case_dir": str((candidate / "case").resolve()),
        "compile_ok": True, "converged": True, "compiled_model_name": "M",
        "solver_invocations": build_runs,
    }))
    if eval_runs:
        (candidate / "evaluation_run_result.json").write_text(
            json.dumps({"solver_invocations": eval_runs}))
    built = tools.build_manager_tools(Settings(), out)
    score = [f for f in built["oed_candidate_tools"]
             if getattr(f, "__name__", "") == "oed_score_candidate"][0]
    return score, candidate, case_dirs


def test_evaluation_runs_are_charged_too() -> None:
    """A multi-case candidate's evaluation solves are most of its real price.

    They run through subprocess.run inside oed_run_evaluation_cases, not through
    the candidate agent's run_bash, so the agent's own counter never sees them.
    Left uncounted, a candidate that solved 32 evaluation cases was charged only
    for the handful of solves it made while building the model — understating a
    multi-case candidate by an order of magnitude and making strategies
    incomparable, which is the exact failure measured cost exists to prevent.
    """
    import json

    score, candidate, case_dirs = _cost_fixture(build_runs=8, eval_runs=32, cases=32)
    score(candidate_dir=str(candidate), case_dir=str(candidate / "case"),
          action_type="code_mod", variant_name="x", model_description="d",
          case_dirs=case_dirs)
    record = json.loads((candidate / "candidate_record.json").read_text())
    check("cost is build runs plus evaluation runs", record.get("cost") == 40,
          detail=f"cost={record.get('cost')}")
    check(
        "and the split is recorded so the charge is auditable",
        record.get("solver_invocations_build") == 8
        and record.get("solver_invocations_evaluation") == 32,
        detail=f"{record.get('solver_invocations_build')}/"
               f"{record.get('solver_invocations_evaluation')}",
    )


def test_single_case_cost_is_unchanged() -> None:
    import json

    score, candidate, _ = _cost_fixture(build_runs=5, eval_runs=0, cases=0)
    score(candidate_dir=str(candidate), case_dir=str(candidate / "case"),
          action_type="code_mod", variant_name="x", model_description="d")
    record = json.loads((candidate / "candidate_record.json").read_text())
    check("a single-case candidate is charged its own solves only",
          record.get("cost") == 5, detail=f"cost={record.get('cost')}")

    score, candidate, _ = _cost_fixture(build_runs=0, eval_runs=0, cases=0)
    score(candidate_dir=str(candidate), case_dir=str(candidate / "case"),
          action_type="code_mod", variant_name="x", model_description="d")
    record = json.loads((candidate / "candidate_record.json").read_text())
    check("with no counter at all it falls back to the flat price",
          record.get("cost") == 2, detail=f"cost={record.get('cost')}")



def test_a_case_that_wrote_nothing_is_not_counted_as_run() -> None:
    """Reaching End is not the same as producing something to score.

    Found by running the evaluation loop for real: a case whose writeInterval
    never fires within endTime runs happily to completion and leaves only 0/ on
    disk. All three cases reported ok, and the scorer would then have been
    handed three cases containing nothing but the initial condition — at best
    rejected one by one, at worst scored as if the initial guess were a result.
    """
    scratch = Path(tempfile.mkdtemp())
    empty = scratch / "empty"
    (empty / "0").mkdir(parents=True)
    check("a case with only 0/ has no solved time",
          tools._latest_solved_time(empty) is None)

    (empty / "300").mkdir()
    check("a written time directory is found", tools._latest_solved_time(empty) == "300")
    (empty / "1000").mkdir()
    check("and the latest one wins", tools._latest_solved_time(empty) == "1000")
    check("a missing directory is handled", tools._latest_solved_time(scratch / "absent") is None)


def test_openfoam_failures_are_reported_from_stdout() -> None:
    """OpenFOAM writes FOAM FATAL ERROR to stdout, not stderr, and exits non-zero.

    Reporting only the stderr tail handed back an empty string for the most
    common failures. Observed while testing the evaluation loop: three cases
    died on "cannot find file .../0/nuTilda" and the tool reported nothing but
    "No evaluation case ran successfully", which the manager cannot act on.
    """
    class _Proc:
        def __init__(self, stdout="", stderr=""):
            self.stdout = stdout
            self.stderr = stderr

    fatal = _Proc(stdout="Selecting RAS model X\n\n--> FOAM FATAL ERROR: \ncannot find file \"0/nuTilda\"\n")
    reason = tools._openfoam_failure_reason(fatal)
    check("a fatal error in stdout is surfaced", "FOAM FATAL" in reason and "nuTilda" in reason,
          detail=reason[:120])

    check("a stderr-only failure still reports",
          "boom" in tools._openfoam_failure_reason(_Proc(stderr="boom")))
    check("a silent failure says so, rather than returning nothing",
          tools._openfoam_failure_reason(_Proc()) != "")



def test_replicas_preserve_the_declared_case_layout() -> None:
    """A set-scoring comparator finds its cases by relative path, not basename.

    Regression for run closure_20260826_codex. Its authored comparator scores
    all 32 cases at once: given one case it walks UP looking for `training/`
    and `validation/`, then maps every reference case across by the path
    relative to that root. Replicating the cases as a flat list of basenames
    breaks the mapping — find_cases_root returns None, the comparator emits
    NaN, and EVERY candidate in the study is unscoreable.

    Caught before any candidate was scored; it fails loudly rather than
    silently returning the baseline, but the study would have produced nothing.
    """
    import os

    root = Path("starter_closure_challenge/cases")
    if not root.is_dir():
        check("replica-layout test needs the closure starter", True,
              detail="skipped: starter absent")
        return
    cases = sorted(p.parent.parent for p in root.rglob("system/controlDict"))
    common = Path(os.path.commonpath([str(c) for c in cases]))
    relatives = [c.relative_to(common) for c in cases]

    check(
        "the declared cases share a root that keeps their structure",
        len(relatives) == len(cases) and any(len(r.parts) > 1 for r in relatives),
        detail=f"{len(cases)} cases, sample {relatives[0] if relatives else None}",
    )
    check(
        "and that structure carries the train/validation split the comparator needs",
        {r.parts[0] for r in relatives} == {"training", "validation"},
        detail=str({r.parts[0] for r in relatives}),
    )



def test_a_solver_timeout_is_a_result_not_an_exception() -> None:
    """A slow candidate must not tear down the manager's step.

    Regression for run closure_20260826_codex: a candidate's closure was slow
    enough to exceed the per-case fence, subprocess.run raised TimeoutExpired,
    it propagated out of oed_run_evaluation_cases, and the whole step failed —
    discarding a candidate that had already cost an hour of compile time. This
    is the third place the same exception class ended a step; grep_files and
    _run_script were fixed the same way earlier.
    """
    import subprocess
    import tempfile

    scratch = Path(tempfile.mkdtemp())
    env = {"PATH": "/usr/bin:/bin"}

    real = subprocess.run

    def fake(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=3600, output=b"partial log")

    tools.subprocess.run = fake
    try:
        result = tools._run_one_evaluation_case(scratch, "simpleFoam", env)
        check("a timeout does not propagate", True)
        check("it is flagged as timed out", getattr(result, "timed_out", False) is True)
        check("and marked failed", result.returncode == 124, detail=str(result.returncode))
        check("the reason names the fence", "timeout" in result.stderr.lower(),
              detail=result.stderr[-80:])
        check("whatever the solver printed is kept", "partial log" in result.stdout,
              detail=repr(result.stdout)[:60])
    except subprocess.TimeoutExpired:
        check("a timeout does not propagate", False, detail="TimeoutExpired escaped")
    finally:
        tools.subprocess.run = real

    ok = tools._run_one_evaluation_case(scratch, "echo End", env)
    check("a normal run still reports success", getattr(ok, "timed_out", False) is False)
    check("and carries its output", "End" in ok.stdout, detail=repr(ok.stdout)[:40])


def test_a_slow_model_is_abandoned_not_ground_through() -> None:
    """A closure that blows the fence on one case will blow it on all 32.

    At an hour apiece that is 32 hours to learn what the first two cases
    already said, on a budget measured in solver runs.
    """
    check(
        "the abandon threshold is small but not one",
        2 <= getattr(tools, "_OED_EVAL_TIMEOUT_ABANDON", 0) <= 3,
        detail=str(getattr(tools, "_OED_EVAL_TIMEOUT_ABANDON", None)),
    )
    check(
        "and the per-case fence is well above a healthy solve",
        getattr(tools, "_OED_EVAL_CASE_TIMEOUT_S", 0) >= 3600,
        detail=str(getattr(tools, "_OED_EVAL_CASE_TIMEOUT_S", None)),
    )



def _rescue_fixture(*, converged: bool, declared: int, succeeded: int):
    """A candidate whose build trial run may have failed, with an evaluation
    result covering `declared` cases of which `succeeded` solved."""
    out = Path(tempfile.mkdtemp())
    disc = out / "open_ended_discovery"
    disc.mkdir(parents=True)
    (disc / "search_config.json").write_text(json.dumps(
        {"topic": "t", "total_budget": 10, "baseline_direction": "min"}
    ))
    (disc / "baseline_score.json").write_text(json.dumps(
        {"metric": "m", "value": 0.004, "direction": "min", "verified": True}
    ))
    candidate = disc / "cand_x"
    (candidate / "case").mkdir(parents=True)
    (candidate / "agentic_result.json").write_text(json.dumps({
        "status": "OK" if converged else "FAILED",
        "case_dir": str((candidate / "case").resolve()),
        "compile_ok": True,
        "converged": converged,
        "compiled_model_name": "M",
        "solver_invocations": 5,
    }))
    (candidate / "evaluation_run_result.json").write_text(json.dumps({
        "cases_declared": declared, "cases_succeeded": succeeded,
        "solver_invocations": succeeded,
    }))
    built = tools.build_manager_tools(Settings(), out)
    score = [f for f in built["oed_candidate_tools"]
             if getattr(f, "__name__", "") == "oed_score_candidate"][0]
    return score, candidate


def _stub_diagnosis(**verdict):
    """Replace the LLM diagnosis with a fixed verdict, and count the calls."""
    calls = []
    base = {"ok": True, "cause": "stub", "category": "harness", "repairable": False,
            "alters_graded_setup": False, "repair_steps": [], "confidence": 0.9,
            "score_anyway": False}
    base.update(verdict)

    def fake(candidate_path, execution_doc, score_error, settings):
        calls.append(str(candidate_path))
        return dict(base)

    original = tools._diagnose_null_score
    tools._diagnose_null_score = fake
    return calls, original


def test_the_model_decides_whether_graded_cases_outrank_the_trial_run() -> None:
    """The rescue is a judgement, so the model makes it — not a count.

    Measured on run closure_20260826_codex: cdomega_f1_taper_045_065 failed the
    single trial run the build agent made on the starter geometry, solved all 32
    declared cases, and was recorded FAILED with a null score. Scored with the
    study's own bound comparator it is 0.108830 — +4.20% and the best model the
    search produced.

    The first fix for this was `succeeded == declared`, which is wrong in both
    directions: it discards 31 of 32 without asking why one case failed, and
    accepts 32 of 32 even if every one of them quietly diverged. Neither is a
    counting question, so this pins that the verdict drives the outcome.
    """
    calls, original = _stub_diagnosis(score_anyway=True, cause="all 32 graded cases solved")
    try:
        score, candidate = _rescue_fixture(converged=False, declared=32, succeeded=32)
        result = score(candidate_dir=str(candidate), case_dir=str(candidate / "case"),
                       action_type="code_mod", variant_name="x", model_description="d")
        check("a verdict of score_anyway lets scoring proceed",
              result.get("execution_ok") is True, detail=str(result)[:200])
        check("and the recorded reason is the model's own cause",
              "all 32 graded cases solved" in str(result.get("evaluation_rescue", "")),
              detail=repr(result.get("evaluation_rescue")))
        check("the model was actually consulted", len(calls) == 1, detail=f"{len(calls)} calls")
    finally:
        tools._diagnose_null_score = original


def test_a_verdict_against_scoring_is_respected() -> None:
    """Same evidence shape, opposite verdict — the outcome must follow the verdict.

    32/32 succeeded here, which the old count rule would have scored blindly.
    """
    calls, original = _stub_diagnosis(score_anyway=False, cause="every case diverged before End")
    try:
        score, candidate = _rescue_fixture(converged=False, declared=32, succeeded=32)
        result = score(candidate_dir=str(candidate), case_dir=str(candidate / "case"),
                       action_type="code_mod", variant_name="x", model_description="d")
        check("a verdict against scoring keeps it a failure",
              result.get("execution_ok") is False, detail=str(result)[:200])
        check("and no rescue is claimed",
              not result.get("evaluation_rescue"), detail=repr(result.get("evaluation_rescue")))
    finally:
        tools._diagnose_null_score = original


def test_every_null_score_carries_a_diagnosis() -> None:
    """A null score used to be a dead end: FAILED, score None, no reason.

    Nothing downstream — the archive, the manager, or a person reading the run
    — could tell a diverged closure from a scoring-plumbing bug, though the
    evidence to tell them apart was on disk either way.
    """
    calls, original = _stub_diagnosis(score_anyway=False, category="model_physics",
                                      cause="the closure diverged on every case",
                                      repairable=False)
    try:
        score, candidate = _rescue_fixture(converged=False, declared=32, succeeded=0)
        result = score(candidate_dir=str(candidate), case_dir=str(candidate / "case"),
                       action_type="code_mod", variant_name="x", model_description="d")
        record = json.loads((candidate / "candidate_record.json").read_text())
        check("the record carries a diagnosis", bool(record.get("failure_diagnosis")),
              detail=str(record.get("failure_diagnosis"))[:200])
        check("with the cause on it",
              "diverged" in str(record["failure_diagnosis"].get("cause", "")))
        check("and its category",
              record["failure_diagnosis"].get("category") == "model_physics")
        check("the model was consulted once, not twice, for one failure",
              len(calls) == 1, detail=f"{len(calls)} calls")
    finally:
        tools._diagnose_null_score = original


def test_a_converged_build_never_pays_for_a_diagnosis() -> None:
    """The ordinary path must not start making a model call per candidate."""
    calls, original = _stub_diagnosis(score_anyway=True)
    try:
        score, candidate = _rescue_fixture(converged=True, declared=32, succeeded=32)
        result = score(candidate_dir=str(candidate), case_dir=str(candidate / "case"),
                       action_type="code_mod", variant_name="x", model_description="d")
        check("a converged build is still execution_ok", result.get("execution_ok") is True)
        check("and carries no rescue note", not result.get("evaluation_rescue"),
              detail=repr(result.get("evaluation_rescue")))
    finally:
        tools._diagnose_null_score = original


def test_diagnosis_failure_leaves_the_candidate_as_it_was() -> None:
    """An unreachable model must not turn a failure into a crash, or a pass."""
    original = tools._diagnose_null_score

    def explode(*a, **k):
        raise RuntimeError("no model available")

    tools._diagnose_null_score = explode
    try:
        score, candidate = _rescue_fixture(converged=False, declared=32, succeeded=32)
        try:
            result = score(candidate_dir=str(candidate), case_dir=str(candidate / "case"),
                           action_type="code_mod", variant_name="x", model_description="d")
            check("scoring survives a diagnosis that raises", True)
            check("and does not rescue on a failed diagnosis",
                  result.get("execution_ok") is False, detail=str(result)[:200])
        except Exception as exc:
            check("scoring survives a diagnosis that raises", False,
                  detail=f"{type(exc).__name__}: {exc}")
    finally:
        tools._diagnose_null_score = original


def main() -> int:
    for test in (
        test_declared_cases_resolve,
        test_case_application_is_read_not_assumed,
        test_model_install_repoints_the_case,
        test_a_partial_mean_is_refused,
        test_scoring_outside_the_candidate_is_refused,
        test_single_case_studies_are_untouched,
        test_prescribed_mesh_needs_a_stated_reason,
        test_prescribed_mesh_refuses_a_self_built_case,
        test_prescribed_mesh_accepts_supplied_input,
        test_baseline_must_cover_the_same_cases,
        test_single_case_studies_skip_the_baseline_guard,
        test_model_install_is_verified_not_assumed,
        test_evaluation_runs_are_charged_too,
        test_single_case_cost_is_unchanged,
        test_a_case_that_wrote_nothing_is_not_counted_as_run,
        test_openfoam_failures_are_reported_from_stdout,
        test_replicas_preserve_the_declared_case_layout,
        test_a_solver_timeout_is_a_result_not_an_exception,
        test_a_slow_model_is_abandoned_not_ground_through,
        test_the_model_decides_whether_graded_cases_outrank_the_trial_run,
        test_a_verdict_against_scoring_is_respected,
        test_every_null_score_carries_a_diagnosis,
        test_a_converged_build_never_pays_for_a_diagnosis,
        test_diagnosis_failure_leaves_the_candidate_as_it_was,
    ):
        try:
            test()
        except Exception as exc:  # noqa: BLE001 — a raising test is a failing test
            check(test.__name__, False, detail=f"{type(exc).__name__}: {exc}")
    if FAILURES:
        print(f"\n{len(FAILURES)} FAILURE(S): " + ", ".join(FAILURES))
        return 1
    print("\nALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
