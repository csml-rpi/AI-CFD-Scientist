#!/usr/bin/env python3
"""Offline test for the two ways a finished OED candidate used to be lost.

Both were measured on run closure_20260826_codex:

  * `resume` replays a whole batch, because LangGraph checkpoints the model's
    tool-call message before the tool results exist. sst_a1_limiter_025 had
    already compiled and solved to t=30000 in 1525s, and the replay was about
    to rebuild it from scratch.
  * A candidate's record only enters history.json when the manager remembers
    to pass its directory to oed_record_candidate_results. sst_crossdiff_-
    scale_065 -- the best model in the study at +2.92% -- sat on disk with a
    complete record while the manager reported the batch had produced nothing.

Runs against real files in a temp directory. No OpenFOAM, no LLM.

    python3 scripts/test_oed_replay_safety.py
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

FAILURES = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global FAILURES
    print(f"[{'PASS' if condition else 'FAIL'}] {name}" + (f" — {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES += 1


def _read(path: Path):
    return json.loads(path.read_text())


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _candidate(disc: Path, name: str, *, hypothesis: str, plan: str, status: str = "OK") -> Path:
    cdir = disc / f"cand_{name}"
    case = cdir / name
    (case / "system").mkdir(parents=True, exist_ok=True)
    _write(cdir / "agentic_result.json", {
        "status": status,
        "case_dir": str(case),
        "compile_ok": True,
        "converged": True,
        "compiled_model_name": f"model_{name}",
        "compiled_so": str(case / "customModels" / "lib.so"),
    })
    _write(cdir / "candidate_invocation.json", {
        "hypothesis": hypothesis, "plan": plan, "variant_name": name,
    })
    return cdir


def _record(cdir: Path, *, score: float, status: str = "REVISE") -> None:
    _write(cdir / "candidate_record.json", {
        "action_type": "code_mod",
        "family": f"family of {cdir.name}",
        "strategy": "analytic",
        "status": status,
        "cost": 40,
        "case_dir": str(cdir / cdir.name.replace("cand_", "")),
        "score": {"metric": "velocity_mae", "value": score, "direction": "min"},
    })


def build(out_dir: Path):
    """Real manager tools bound to `out_dir`, with only the subprocess runner
    stubbed out, so "did it rebuild?" is observable without compiling."""
    from cfd_langgraph.manager import tools as T
    from cfd_langgraph.config import get_settings

    ran: list = []

    def fake_run_script(argv, **kw):
        variant = argv[argv.index("--variant-name") + 1]
        ran.append(variant)
        out = Path(argv[argv.index("--output") + 1])
        _write(out, {"status": "OK", "case_dir": str(out.parent / variant),
                     "compile_ok": True, "converged": True,
                     "compiled_model_name": f"rebuilt_{variant}"})

        class P:
            returncode = 0
            stdout = ""
            stderr = ""
        return P()

    T._run_script = fake_run_script
    T._foamagent_env = lambda *a, **k: {}
    T._oed_candidate_timeout = lambda *a, **k: 0
    built = T.build_manager_tools(get_settings(), out_dir)
    by_name = {f.__name__: f for group in ("manager_tools", "case_runner_tools",
                                           "oed_candidate_tools")
               for f in built[group]}
    return by_name, ran


def test_replay_reuses_a_finished_candidate() -> None:
    tmp = Path(tempfile.mkdtemp())
    try:
        disc = tmp / "open_ended_discovery"
        disc.mkdir(parents=True)
        starter = tmp / "starter"
        (starter / "system").mkdir(parents=True)
        _write(disc / "search_config.json", {
            "topic": "T", "baseline_case_dir": str(starter), "baseline_direction": "min",
        })
        H, P = "compile a1=0.25", "Strategy: solver_fit."
        _candidate(disc, "sst_a1_limiter_025", hypothesis=H, plan=P)

        tools, ran = build(tmp)
        run = tools["oed_run_code_mod_candidate"]

        out = run(topic="T", variant_name="sst_a1_limiter_025", hypothesis=H, plan=P)
        check("a replayed identical candidate is not rebuilt", ran == [], detail=f"ran={ran}")
        check("and it is reported as reused", out.get("reused") is True, detail=str(out)[:200])
        check("with the finished model name, not a rebuilt one",
              out.get("compiled_model_name") == "model_sst_a1_limiter_025", detail=str(out)[:200])
        check("and reported ok so the manager scores it", out.get("ok") is True)

        out2 = run(topic="T", variant_name="sst_a1_limiter_025",
                   hypothesis="a1=0.20 instead", plan=P)
        check("a CHANGED hypothesis under the same name does rebuild",
              ran == ["sst_a1_limiter_025"], detail=f"ran={ran}")
        check("and is not labelled reused", out2.get("reused") is not True)

        ran.clear()
        _write(disc / "cand_half_built" / "agentic_result.json",
               {"status": "FAILED", "case_dir": ""})
        _write(disc / "cand_half_built" / "candidate_invocation.json",
               {"hypothesis": "h", "plan": "p", "variant_name": "half_built"})
        run(topic="T", variant_name="half_built", hypothesis="h", plan="p")
        check("a candidate that died mid-build is rebuilt, not reused",
              ran == ["half_built"], detail=f"ran={ran}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_a_finished_candidate_is_never_stranded() -> None:
    tmp = Path(tempfile.mkdtemp())
    try:
        disc = tmp / "open_ended_discovery"
        disc.mkdir(parents=True)
        starter = tmp / "starter"
        (starter / "system").mkdir(parents=True)
        _write(disc / "search_config.json", {
            "topic": "T", "baseline_case_dir": str(starter),
            "baseline_direction": "min", "baseline_score": 0.1136,
            "total_budget": 4000,
        })
        passed = _candidate(disc, "passed_in", hypothesis="h1", plan="p1")
        _record(passed, score=0.120)
        orphan = _candidate(disc, "crossdiff_scale_065", hypothesis="h2", plan="p2")
        _record(orphan, score=0.11027841682049302)

        tools, _ran = build(tmp)
        record = tools["oed_record_candidate_results"]

        out = record(candidate_dirs=[str(passed)])
        history = json.loads((disc / "history.json").read_text())
        names = {Path(h["candidate_dir"]).name for h in history}
        check("the candidate that was passed in is recorded",
              "cand_passed_in" in names, detail=str(names))
        check("the finished candidate that was NOT passed in is recovered",
              "cand_crossdiff_scale_065" in names, detail=str(names))
        check("and the recovery is reported, not silent",
              any("crossdiff" in c for c in out.get("recovered_unrecorded_candidates", [])),
              detail=str(out.get("recovered_unrecorded_candidates")))
        check("its real score survives into history",
              any(h["score"]["value"] == 0.11027841682049302 for h in history))
        check("budget counts both", out.get("budget_used") == 80, detail=str(out.get("budget_used")))

        out2 = record(candidate_dirs=[str(passed), str(orphan)])
        history2 = json.loads((disc / "history.json").read_text())
        check("recording again does not duplicate either entry",
              len(history2) == 2, detail=f"{len(history2)} entries")
        check("and reports nothing new to recover",
              out2.get("recovered_unrecorded_candidates") == [])

        running = disc / "cand_still_running"
        (running / "x").mkdir(parents=True)
        _write(running / "agentic_result.json", {"status": "OK", "case_dir": str(running / "x")})
        out3 = record(candidate_dirs=[])
        check("a candidate with no record yet is not swept in early",
              out3.get("recovered_unrecorded_candidates") == [],
              detail=str(out3.get("recovered_unrecorded_candidates")))
        check("and history is unchanged by that call",
              len(json.loads((disc / "history.json").read_text())) == 2)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _record_raw(cdir: Path, **fields) -> None:
    """A candidate record with arbitrary fields (score may be None)."""
    base = {
        "action_type": "code_mod",
        "family": f"family of {cdir.name}",
        "strategy": "analytic",
        "status": "REVISE",
        "cost": 48,
        "solver_invocations": 48,
        "case_dir": str(cdir / cdir.name.replace("cand_", "")),
        "score": None,
    }
    base.update(fields)
    _write(cdir / "candidate_record.json", base)


def test_a_candidate_mid_repair_is_not_frozen_as_failed() -> None:
    """A null-scored candidate awaiting repair must not be swept into history.

    Its record is complete and its case_dir resolves, so it passed every guard
    the sweep applies -- and once in history as FAILED/score:null it could never
    be corrected, because the already_recorded check skipped it outright. The
    repaired result was written to disk and silently dropped, the arms were
    charged a loss, and the repair's solver runs went unbilled. The real run has
    iteration 29 (elite_sst_sas025) sitting in history exactly like this, with
    repair_attempts 1 and a repair_log.
    """
    tmp = Path(tempfile.mkdtemp())
    try:
        disc = tmp / "open_ended_discovery"; disc.mkdir(parents=True)
        starter = tmp / "starter"; (starter / "system").mkdir(parents=True)
        _write(disc / "search_config.json", {
            "topic": "T", "baseline_case_dir": str(starter),
            "baseline_direction": "min", "baseline_score": 0.1136, "total_budget": 4000,
        })
        good = _candidate(disc, "good", hypothesis="h1", plan="p1")
        _record(good, score=0.109)
        broken = _candidate(disc, "repairing", hypothesis="h2", plan="p2")
        _record_raw(broken, status="FAILED", score=None, repair_attempts=1,
                    repair_log=["first attempt"])

        tools, _ = build(tmp)
        record = tools["oed_record_candidate_results"]

        out1 = record(candidate_dirs=[str(good)])
        names = [Path(h.get("candidate_dir", "")).name for h in _read(disc / "history.json")]
        check("the mid-repair candidate is held back, not swept",
              "cand_repairing" not in names, detail=str(names))
        check("and the manager is told why",
              any("repairing" in d for d in out1.get("deferred_pending_repair", [])),
              detail=str(out1.get("deferred_pending_repair")))

        # The repair succeeds and oed_score_candidate rewrites the record.
        _record_raw(broken, status="REVISE", repair_attempts=1,
                    score={"metric": "velocity_mae", "value": 0.0740, "direction": "min"},
                    cost=96, solver_invocations=96)
        out2 = record(candidate_dirs=[str(broken)])
        hist = {Path(h.get("candidate_dir", "")).name: h
                for h in _read(disc / "history.json")}
        check("the repaired candidate lands with its real score",
              (hist.get("cand_repairing", {}).get("score") or {}).get("value") == 0.0740,
              detail=str(hist.get("cand_repairing", {}).get("score")))
        # 40 for `good` (the shared _record helper) + 96 for the repaired
        # candidate, whose cost rose from 48 when the repair re-ran it. The
        # bug billed 48 for it -- the pre-repair figure frozen at sweep time.
        check("and the repair's extra solver runs are billed",
              out2.get("budget_used") == 136, detail=str(out2.get("budget_used")))


    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_a_repair_that_lands_late_corrects_the_frozen_entry() -> None:
    """A candidate already recorded as null-scored is corrected in place.

    The deferral above prevents the usual case, but a candidate whose repair
    budget is exhausted is recorded (correctly) as FAILED, and a crash-resume
    can record one mid-flight. If a score later appears on disk, the stale
    entry must be updated rather than skipped -- and a rewrite that contradicts
    an EXISTING score must not be, or a late write could silently restate a
    result the search has already acted on.
    """
    tmp = Path(tempfile.mkdtemp())
    try:
        disc = tmp / "open_ended_discovery"; disc.mkdir(parents=True)
        starter = tmp / "starter"; (starter / "system").mkdir(parents=True)
        _write(disc / "search_config.json", {
            "topic": "T", "baseline_case_dir": str(starter),
            "baseline_direction": "min", "baseline_score": 0.1136, "total_budget": 4000,
        })
        frozen = _candidate(disc, "frozen", hypothesis="h", plan="p")
        # Repair budget exhausted, so this one is legitimately recorded.
        _record_raw(frozen, status="FAILED", score=None, repair_attempts=2,
                    repair_log=["a", "b"])
        tools, _ = build(tmp)
        record = tools["oed_record_candidate_results"]
        record(candidate_dirs=[str(frozen)])
        h = _read(disc / "history.json")
        check("an exhausted candidate IS recorded (else this proves nothing)",
              len(h) == 1 and h[0]["score"] is None, detail=str(h))
        iteration = h[0]["iteration"]

        _record_raw(frozen, status="REVISE", repair_attempts=2,
                    score={"metric": "velocity_mae", "value": 0.0740, "direction": "min"})
        out = record(candidate_dirs=[str(frozen)])
        h = _read(disc / "history.json")
        check("a late repair corrects the frozen entry",
              len(h) == 1 and (h[0]["score"] or {}).get("value") == 0.0740, detail=str(h))
        check("in place, keeping its iteration number",
              h[0]["iteration"] == iteration, detail=str(h[0]["iteration"]))
        check("and it is reported",
              any("frozen" in d for d in out.get("repaired_since_recorded", [])),
              detail=str(out.get("repaired_since_recorded")))

        _record_raw(frozen, status="REVISE", repair_attempts=2,
                    score={"metric": "velocity_mae", "value": 0.999, "direction": "min"})
        record(candidate_dirs=[str(frozen)])
        h = _read(disc / "history.json")
        check("a contradictory rewrite of an existing score is refused",
              (h[0]["score"] or {}).get("value") == 0.0740, detail=str(h[0]["score"]))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_the_repair_counter_survives_a_rescore() -> None:
    """oed_score_candidate must not reset repair_attempts.

    It builds `record` as a fresh literal and writes it, and the prescribed
    workflow is repair -> oed_note_repair_attempt -> re-run -> score, so the
    counter was destroyed on every cycle: after two attempts (cap reached,
    remaining 0) a single re-score put it back to "attempt 1, remaining 1",
    making the cap unenforceable and letting the search grind one broken
    closure indefinitely at ~48 solver runs a go.
    """
    import ast
    src = Path(__file__).resolve().parents[1] / "src/cfd_langgraph/manager/tools.py"
    text = src.read_text(encoding="utf-8")
    fn = next(n for n in ast.walk(ast.parse(text))
              if isinstance(n, ast.FunctionDef) and n.name == "oed_score_candidate")
    seg = ast.get_source_segment(text, fn) or ""
    # Not a count of the filename -- it already appeared twice before this fix,
    # so `>= 2` passed with the bug still in place. Assert the READ specifically.
    reads = [n for n in ast.walk(fn) if isinstance(n, ast.Call)
             and getattr(n.func, "id", "") == "_read_json"
             and "candidate_record.json" in (ast.get_source_segment(text, n) or "")]
    check("oed_score_candidate reads the existing record before writing",
          bool(reads), detail="no _read_json of candidate_record.json")
    check("and carries the repair counter forward",
          "repair_attempts" in seg, detail="no mention of repair_attempts")


def test_an_experiment_is_identified_by_its_parent_too() -> None:
    """The same coefficient on two different compiled parents is two experiments.

    The key was family|experiment|strategy|params, so Ccr=1.2 on a
    curvature-corrected SST fingerprinted identically to Ccr=1.2 on a
    stress-limited one and the second was dropped as "an identical proposal".

    Keyed on the parent's ITERATION, because the two call sites must be able to
    name the parent identically. Keying on a directory basename broke the guard
    in BOTH directions: the history side fell back to the record's own
    case_dir while the proposal side used the parent's, and every experiment's
    case dir is named the literal string "case" -- so a real repeat did not
    match while two different parents collided. This test drives the two call
    sites' own field derivations, not hand-written names, because that
    asymmetry was invisible to a test that passed `base=` directly.
    """
    import ast
    from cfd_langgraph.manager.tools import _oed_candidate_fingerprint as fp

    a = fp("SST", "experiment", "tune it", {"Ccr": 1.2}, strategy="sweep", parent_iteration=7)
    b = fp("SST", "experiment", "tune it", {"Ccr": 1.2}, strategy="sweep", parent_iteration=9)
    same = fp("SST", "experiment", "worded differently", {"Ccr": 1.2}, strategy="sweep",
              parent_iteration=7)
    other_val = fp("SST", "experiment", "tune it", {"Ccr": 3.6}, strategy="sweep",
                   parent_iteration=7)
    check("different parents are different experiments", a != b)
    check("the same parent and value is still one experiment", a == same)
    check("a different coefficient value is still a new experiment", a != other_val)
    check("an int and a str parent iteration agree",
          a == fp("SST", "experiment", "tune it", {"Ccr": 1.2}, strategy="sweep",
                  parent_iteration="7"))

    # Both call sites must derive the parent from the SAME quantity. Read the
    # keyword each one passes straight out of the source.
    src = Path(__file__).resolve().parents[1] / "src/cfd_langgraph/manager/tools.py"
    text = src.read_text(encoding="utf-8")
    calls = [n for n in ast.walk(ast.parse(text)) if isinstance(n, ast.Call)
             and getattr(n.func, "id", "") == "_oed_candidate_fingerprint"]
    check("both call sites are present", len(calls) == 2, f"{len(calls)} call sites")
    passed = []
    for call in calls:
        kw = next((k for k in call.keywords if k.arg == "parent_iteration"), None)
        passed.append(ast.get_source_segment(text, kw.value) if kw else None)
    check("neither call site omits the parent", all(p is not None for p in passed), str(passed))
    check("neither derives it from a directory path",
          not any("case_dir" in (p or "") for p in passed), str(passed))


def test_every_coefficient_idiom_is_recognised() -> None:
    """A model's tunable coefficients must be found however they are read.

    Matching only `lookupOrAddToDict` missed the other forms this study's own
    candidates use -- `lookupOrDefault<scalar>` and
    `readScalar(coeffs.lookup(...))`. Three of the 55 recorded candidates
    reported NO tunable coefficients while exposing cInner/cOuter, cW and
    cXG/rXG, and those are exactly the coefficients each modification adds. It
    hid because a model derived from kOmegaSST still reports the inherited
    stock names, so the function looked like it was working.

    The cost is what the function exists to prevent: the DEEPEN prompt falls
    back to "use action_type=code_mod", so nudging one coefficient pays a full
    rebuild instead of a no-build rerun.
    """
    from cfd_langgraph.manager.tools import _runtime_coefficients
    tmp = Path(tempfile.mkdtemp())
    try:
        src = tmp / "customModels" / "m"
        src.mkdir(parents=True)
        (src / "model.C").write_text("""
            alphaK1_(dimensioned<scalar>::lookupOrAddToDict("alphaK1", coeffDict_, 0.85)),
            cXG_(coeffDict_.lookupOrDefault<scalar>("cXG", 0.20)),
            rXG_(this->coeffDict().template lookupOrDefault<scalar>("rXG", 0.10)),
            cW_(readScalar(coeffDict_.lookup("cW"))),
            nBlend_(coeffDict_.getOrDefault<label>("nBlend", 2)),
            // not a coefficient read:
            const word name = "notACoefficient";
        """, encoding="utf-8")
        found = _runtime_coefficients(tmp)
        for name in ("alphaK1", "cXG", "rXG", "cW", "nBlend"):
            check(f"{name} is recognised", name in found, detail=str(found))
        check("and nothing that is not a coefficient read is picked up",
              "notACoefficient" not in found, detail=str(found))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_an_exhausted_endgame_says_so_instead_of_looping() -> None:
    """When the last-scraps branch has nothing to re-run, say the search is over.

    That branch is evaluated before the `force_new_families` branch, so once
    budget lands in [experiment_cost, code_mod_cost) it is entered on every
    call. With no reusable elite it used to `break` with no picks, and the tool
    then advertised "Call this tool again with force_new_families=True" -- which
    re-enters the same branch and breaks again. The manager loops, paying a
    proposer call each round, and the study never reaches the paper stage.
    """
    import ast
    src = Path(__file__).resolve().parents[1] / "src/cfd_langgraph/manager/tools.py"
    text = src.read_text(encoding="utf-8")
    fn = next(n for n in ast.walk(ast.parse(text))
              if isinstance(n, ast.FunctionDef) and "propose_candidates" in n.name)
    body = ast.get_source_segment(text, fn) or ""

    check("the exhausted endgame is distinguished from an affordable one",
          "endgame_exhausted" in body, detail="no endgame_exhausted flag")
    # `affordable` gates the force_new_families advice, so it must account for it.
    affordable = next((n for n in ast.walk(fn) if isinstance(n, ast.Assign)
                       and any(getattr(t, "id", "") == "affordable" for t in n.targets)), None)
    check("and it is folded into whether anything is still affordable",
          affordable is not None
          and "endgame_exhausted" in (ast.get_source_segment(text, affordable.value) or ""),
          detail=ast.get_source_segment(text, affordable.value) if affordable else "no assign")

    # The proposer must not be asked for zero candidates.
    invoke = next((n for n in ast.walk(fn) if isinstance(n, ast.Call)
                   and "with_structured_output" in (ast.get_source_segment(text, n) or "")), None)
    guarded = False
    for node in ast.walk(fn):
        if isinstance(node, ast.If) and "picks" in ast.dump(node.test):
            seg = ast.get_source_segment(text, node) or ""
            if "with_structured_output" in seg:
                guarded = True
    check("and no proposer call is paid for an empty batch",
          invoke is not None and guarded, detail="LLM invoked without a picks guard")


def test_a_candidates_parent_is_the_case_it_is_built_on() -> None:
    """parent_iteration must follow base_case_dir, not just the selection.

    The proposer's `base_case_dir` takes precedence over the selection's elite
    when the two disagree, so deriving parent_iteration from the selection let
    a candidate be built on one model and recorded as a child of another. That
    one field is what the lineage reconstruction, the deepen-vs-parent bar and
    the experiment fingerprint all trust.
    """
    import ast
    src = Path(__file__).resolve().parents[1] / "src/cfd_langgraph/manager/tools.py"
    text = src.read_text(encoding="utf-8")
    fn = next(n for n in ast.walk(ast.parse(text))
              if isinstance(n, ast.FunctionDef) and "propose_candidates" in n.name)

    # The fingerprint's parent and the recorded parent must be the SAME
    # expression -- the earlier version of this check only asserted that
    # neither mentioned case_dir, which is what let them differ.
    fp_parent = None
    for n in ast.walk(fn):
        if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "_oed_candidate_fingerprint":
            kw = next((k for k in n.keywords if k.arg == "parent_iteration"), None)
            if kw is not None:
                fp_parent = ast.get_source_segment(text, kw.value)
    recorded = None
    for n in ast.walk(fn):
        if isinstance(n, ast.Assign):
            tgt = ast.get_source_segment(text, n.targets[0]) or ""
            if tgt.endswith('["parent_iteration"]'):
                recorded = ast.get_source_segment(text, n.value)
    check("the fingerprint's parent is resolved, not taken from the selection",
          fp_parent is not None and "_resolved_parent" in fp_parent, detail=str(fp_parent))
    check("and the recorded parent is the same expression",
          recorded == fp_parent, detail=f"fingerprint={fp_parent!r} recorded={recorded!r}")


def test_an_experiment_cannot_set_a_coefficient_the_model_never_reads() -> None:
    """A misnamed coefficient must be refused, not written and run.

    The `remaining` guard cannot catch this: when a name is absent from the
    coeffs dictionary `_set_model_coefficients` APPENDS it and reports it
    written -- correct for a coefficient read via lookupOrAddToDict with a
    default, indistinguishable from one the model never reads. So `remaining`
    empties either way and the refusal is unreachable.

    A misnamed coefficient (C_bradshaw for bBradshaw -- the underscore variant
    this file has already seen as C_cr for Ccr) then writes a key OpenFOAM
    ignores and the case runs to convergence bit-identical to its parent. Worse
    than wasted: `no_op` compares against the BASELINE, not the parent, so the
    clone enters history with the parent's score and is booked as a
    non-improving refinement against a chain that did nothing wrong.
    """
    from cfd_langgraph.manager.tools import _set_model_coefficients, _runtime_coefficients
    tmp = Path(tempfile.mkdtemp())
    try:
        case = tmp / "case"
        (case / "constant").mkdir(parents=True)
        (case / "constant" / "momentumTransport").write_text(
            "simulationType RAS;\nRAS\n{\n    model kOmegaSSTCustom;\n"
            "    kOmegaSSTCustomCoeffs\n    {\n        bBradshaw 0.90;\n    }\n}\n",
            encoding="utf-8")
        src = case / "customModels" / "m"
        src.mkdir(parents=True)
        (src / "model.C").write_text(
            'bBradshaw_(coeffDict_.lookupOrDefault<scalar>("bBradshaw", 0.90)),\n',
            encoding="utf-8")

        # The underlying writer still reports success for a bogus name -- that
        # is why the check has to happen before it, against the model source.
        wrote = _set_model_coefficients(case, {"C_bradshaw": 0.80})
        check("the writer alone cannot tell a bogus name from a defaulted one",
              "C_bradshaw" in wrote, detail=str(wrote))

        exposed = set(_runtime_coefficients(case))
        check("but the model's own source can", exposed == {"bBradshaw"}, detail=str(exposed))
        check("so a misnamed coefficient is detectable before any solver runs",
              "C_bradshaw" not in exposed)
        check("and the correctly-named one is accepted", "bBradshaw" in exposed)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # And the guard is actually wired into the experiment runner.
    import ast
    src_path = Path(__file__).resolve().parents[1] / "src/cfd_langgraph/manager/tools.py"
    text = src_path.read_text(encoding="utf-8")
    fn = next(n for n in ast.walk(ast.parse(text))
              if isinstance(n, ast.FunctionDef) and n.name == "oed_run_experiment_candidate")
    body = ast.get_source_segment(text, fn) or ""
    check("the experiment runner cross-checks against the compiled model",
          "_runtime_coefficients" in body, detail="no cross-check in the runner")
    check("and only when the parent exposes something at all",
          "if exposed and unknown" in body, detail="unconditional refusal would block real work")


def test_the_strategy_the_proposal_chose_is_the_one_recorded() -> None:
    """Strategy must not be re-derived at score time.

    oed_propose_candidates already runs normalize_strategy with use_llm=True
    and stores the answer -- its own comment says "the model call is paid once
    here and the answer is stored; replay never re-asks". Asking again at score
    time can disagree with itself, because normalize_strategy degrades silently
    to a keyword table when the model call fails and the table gets exactly the
    case the LLM was added for wrong: a solver_fit plan reads as analytic.

    The elite would then land in a different (family, strategy) cell from the
    one select_action picked and paid for, and history's label would no longer
    match the fingerprint a re-proposal carries, so an identical repeat escapes
    dedup.
    """
    import ast
    src_path = Path(__file__).resolve().parents[1] / "src/cfd_langgraph/manager/tools.py"
    text = src_path.read_text(encoding="utf-8")
    fn = next(n for n in ast.walk(ast.parse(text))
              if isinstance(n, ast.FunctionDef) and n.name == "oed_score_candidate")
    assign = next((n for n in ast.walk(fn) if isinstance(n, ast.Assign)
                   and any(getattr(t, "id", "") == "strategy_label" for t in n.targets)), None)
    check("score time assigns a strategy label", assign is not None)
    expr = ast.get_source_segment(text, assign.value) if assign else ""
    check("and takes the proposal's stored choice first",
          expr.strip().startswith('proposal.get("strategy")'), detail=expr[:90])
    # It may still fall back for a record with no proposal on file.
    check("falling back only when there is no stored choice",
          "normalize_strategy" in expr, detail=expr[:90])


def _propose_with_stubbed_llm(out_dir: Path, history: list, elite_iter: int,
                              base_case_dir, monkey_candidate: dict):
    """Drive the REAL oed_propose_candidates with only the proposer stubbed.

    Returns the single candidate it produced, so the test can read the
    parent_iteration the tool actually recorded rather than reading the source
    and hoping. The round-6 tests for this are pure AST assertions -- they check
    that the identifier `_resolved_parent` appears in the right expressions, and
    would pass unchanged if it returned the wrong iteration.
    """
    from cfd_langgraph.manager import tools as T
    from cfd_langgraph.config import get_settings

    disc = out_dir / "open_ended_discovery"
    disc.mkdir(parents=True, exist_ok=True)
    _write(disc / "search_config.json", {
        "topic": "T", "baseline_case_dir": str(out_dir / "starter"),
        "baseline_direction": "min", "baseline_score": 0.1136, "total_budget": 4000,
    })
    _write(disc / "history.json", history)
    # oed_propose_candidates refuses to run without a verified baseline.
    _write(disc / "baseline_score.json", {
        "verified": True, "value": 0.1136, "direction": "min",
        "metric": "m", "per_case": {},
    })

    class _Batch:
        def __init__(self, cands): self.candidates = cands
    class _Spec:
        def __init__(self, d): self._d = d
        def model_dump(self): return dict(self._d)
    class _Stub:
        def with_structured_output(self, _): return self
        def invoke(self, _prompt): return _Batch([_Spec(monkey_candidate)])

    T._foamagent_env = lambda *a, **k: {}
    T.normalize_strategy = lambda *a, **k: "analytic"
    T._llm_duplicate_of = lambda *a, **k: None
    # foam_llm is constructed inside build_manager_tools, so the stub has to be
    # in place BEFORE the tools are built -- patching afterwards leaves the real
    # provider bound and the test reaches for a live token.
    orig = T.create_langchain_llm
    T.create_langchain_llm = lambda *a, **k: _Stub()
    try:
        built = T.build_manager_tools(get_settings(), out_dir)
        fns = {f.__name__: f for f in built["manager_tools"]}
        return fns["oed_propose_candidates"]("T", num_candidates=1)
    finally:
        T.create_langchain_llm = orig


def test_the_recorded_parent_is_the_case_actually_built_on() -> None:
    """Behavioural cover for _resolved_parent: read the number it records.

    The proposer's base_case_dir takes precedence over the selection's elite,
    so a candidate can be built on one model and recorded as a child of
    another. parent_iteration is the single field the lineage reconstruction,
    the deepen-vs-parent bar and the experiment fingerprint all trust.
    """
    tmp = Path(tempfile.mkdtemp())
    try:
        disc = tmp / "open_ended_discovery"
        (tmp / "starter" / "system").mkdir(parents=True)
        cases = {}
        history = []
        for it, score in ((2, 0.1100), (5, 0.1050)):
            cdir = disc / f"cand_v{it}" / f"v{it}"
            (cdir / "customModels" / "m").mkdir(parents=True)
            (cdir / "customModels" / "m" / "model.C").write_text(
                'c_(coeffDict_.lookupOrDefault<scalar>("cX", 0.2)),', encoding="utf-8")
            cases[it] = cdir
            history.append({
                "action_type": "code_mod", "variant_name": f"v{it}", "family": "F",
                "strategy": "analytic", "iteration": it, "case_dir": str(cdir),
                "score": {"metric": "m", "value": score, "direction": "min"},
                "cost": 48, "status": "REVISE", "no_op": False,
            })

        # The proposal names iteration 2's case. Both entries share one
        # (family, strategy) cell, so the archive's elite for that cell is
        # iteration 5 (the better score) -- the two MUST differ or the fixture
        # cannot tell the resolved parent from the selection's, and an earlier
        # version of this test pointed both at 5 and passed on revert.
        out = _propose_with_stubbed_llm(tmp, history, elite_iter=5, base_case_dir=cases[2],
            monkey_candidate={
                "variant_name": "exp_a", "action_type": "experiment", "strategy": "sweep",
                "plan": "vary cX", "hypothesis": "vary cX on the older parent",
                "parameters": {"cX": 0.30}, "base_case_dir": str(cases[2]),
                "model_name_to_reuse": "", "target_family": "F",
            })
        cands = out.get("candidates") or []
        check("a candidate was produced (else this proves nothing)",
              len(cands) == 1, detail=str(out.get("error"))[:160])
        if cands:
            check("its recorded parent is the case it was built on, not the selection's elite",
                  cands[0].get("parent_iteration") == 2,
                  detail=f"parent_iteration={cands[0].get('parent_iteration')}, "
                         f"base was iteration 2 and the cell's elite is iteration 5")

        # An unresolvable base must fall back rather than record a wrong number.
        out2 = _propose_with_stubbed_llm(tmp, history, elite_iter=2,
            base_case_dir="/nonexistent/nope",
            monkey_candidate={
                "variant_name": "exp_b", "action_type": "code_mod", "strategy": "analytic",
                "plan": "p", "hypothesis": "a fresh idea in F",
                "parameters": {}, "base_case_dir": "/nonexistent/nope",
                "model_name_to_reuse": "", "target_family": "F",
            })
        c2 = (out2.get("candidates") or [{}])[0]
        check("an unresolvable base_case_dir never invents a parent",
              c2.get("parent_iteration") in (None, 2, 5),
              detail=f"parent_iteration={c2.get('parent_iteration')}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    test_replay_reuses_a_finished_candidate()
    test_a_finished_candidate_is_never_stranded()
    test_a_candidate_mid_repair_is_not_frozen_as_failed()
    test_a_repair_that_lands_late_corrects_the_frozen_entry()
    test_the_repair_counter_survives_a_rescore()
    test_an_experiment_is_identified_by_its_parent_too()
    test_every_coefficient_idiom_is_recognised()
    test_an_exhausted_endgame_says_so_instead_of_looping()
    test_a_candidates_parent_is_the_case_it_is_built_on()
    test_an_experiment_cannot_set_a_coefficient_the_model_never_reads()
    test_the_strategy_the_proposal_chose_is_the_one_recorded()
    test_the_recorded_parent_is_the_case_actually_built_on()
    print()
    print("ALL PASS" if not FAILURES else f"{FAILURES} FAILURE(S)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
