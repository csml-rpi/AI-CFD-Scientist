"""Per-strategy time fencing, and the diagnose/extend/repair path for a build
agent that stopped before it finished.

What is under test is not "does a tool return a dict". It is the specific
failure this machinery exists for: on run closure_20260826_codex six candidates
were killed at the wall clock mid-fit, each left a compiled library whose
coefficient had never reached the case dictionary, and each was recorded as a
genuine evaluation scoring exactly the baseline. Nothing was null, so nothing
was diagnosed, and the search concluded that fitting does not work from six
experiments that never ran.
"""
from __future__ import annotations
import json, sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cfd_langgraph.manager import tools as T
from cfd_langgraph.manager import subagents
from cfd_langgraph.config import get_settings

F = 0


def check(name, cond, detail=""):
    global F
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        F += 1


def study(candidates):
    """A study directory with the given (name, strategy, duration, status) rows."""
    out = Path(tempfile.mkdtemp())
    disc = out / "open_ended_discovery"
    disc.mkdir(parents=True)
    (disc / "search_config.json").write_text(json.dumps(
        {"topic": "t", "total_budget": 100, "baseline_direction": "min",
         "baseline_case_dir": str(out)}))
    for name, strategy, duration, status in candidates:
        cand = disc / f"cand_{name}"
        cand.mkdir(parents=True)
        (cand / "agentic_result.json").write_text(json.dumps(
            {"status": status, "duration_s": duration}))
        (cand / "candidate_record.json").write_text(json.dumps({"strategy": strategy}))
    return out, disc


# ---------------------------------------------------------------------------
# 1. The per-strategy fence
# ---------------------------------------------------------------------------
print("\n--- per-strategy time fence ---")

# Four fast analytic successes and nothing else: everything is fenced at
# analytic pace, which is the bug.
out, disc = study([(f"a{i}", "analytic", d, "OK")
                   for i, d in enumerate([1500, 1600, 1650, 1700])])
pooled = T._oed_candidate_timeout(disc)
solver = T._oed_candidate_timeout(disc, "solver_fit")
check("pooled fence is set once four successes exist", pooled is not None and pooled > 0,
      f"got {pooled}")
check("a strategy with no successes of its own falls back to the pooled fence",
      solver == pooled, f"solver={solver} pooled={pooled}")

# Now the same study, plus four slow solver_fit successes. The pooled fence
# must not move much, and solver_fit must get its own, much larger, fence.
out, disc = study(
    [(f"a{i}", "analytic", d, "OK") for i, d in enumerate([1500, 1600, 1650, 1700])]
    + [(f"s{i}", "solver_fit", d, "OK") for i, d in enumerate([7000, 7500, 8000, 8600])])
pooled2 = T._oed_candidate_timeout(disc)
solver2 = T._oed_candidate_timeout(disc, "solver_fit")
analytic2 = T._oed_candidate_timeout(disc, "analytic")
check("a strategy with four successes gets its own fence", solver2 is not None)
check("the slow strategy's fence covers its own slowest success",
      solver2 >= 8600, f"got {solver2}")
check("the fast strategy is NOT dragged up by the slow one",
      analytic2 < solver2 and analytic2 <= 3000, f"analytic={analytic2}")
check("each strategy is fenced close to its own population, not to the other's",
      analytic2 < 7000 <= solver2, f"analytic={analytic2} solver={solver2}")
# The pooled fence on a mixed population is degenerate in BOTH directions, which
# is the whole argument for splitting it. With only fast successes it sits at
# their pace and guillotines every slow strategy; once both are present the
# spread in log space blows the Tukey bound out to something that fences
# nothing at all. Neither number is a useful bound on either kind of work.
check("the pooled fence over a bimodal population is uselessly loose",
      pooled2 > 5 * solver2, f"pooled={pooled2} solver={solver2}")

# A failed candidate contributes nothing: only successes set the pace.
out, disc = study([(f"s{i}", "solver_fit", d, "FAILED") for i, d in enumerate([7000, 7500, 8000, 8600])])
check("failed candidates do not set a strategy's fence",
      T._oed_candidate_timeout(disc, "solver_fit") is None)

# The strategy label is read from the record, and from the invocation when the
# record does not exist yet (a candidate killed before it was ever scored).
out, disc = study([("x", "offline_fit", 100, "OK")])
check("_candidate_strategy reads candidate_record.json",
      T._candidate_strategy(disc / "cand_x") == "offline_fit")
(disc / "cand_x" / "candidate_record.json").unlink()
(disc / "cand_x" / "candidate_invocation.json").write_text(json.dumps({"strategy": "solver_fit"}))
check("_candidate_strategy falls back to candidate_invocation.json",
      T._candidate_strategy(disc / "cand_x") == "solver_fit")
(disc / "cand_x" / "candidate_invocation.json").unlink()
check("_candidate_strategy returns empty when nothing records one",
      T._candidate_strategy(disc / "cand_x") == "")


# ---------------------------------------------------------------------------
# 2. The evidence put in front of the diagnosing model
# ---------------------------------------------------------------------------
print("\n--- unclean-finish evidence ---")

out = Path(tempfile.mkdtemp())
cand = out / "cand_fit"
case = cand / "fit"
(case / "constant").mkdir(parents=True)
(case / "customModels" / "myModel").mkdir(parents=True)
(case / "customModels" / "myModel" / "myModel.C").write_text(
    'sF1_(dimensioned<scalar>::lookupOrAddToDict("sF1", this->coeffDict_, 1.0))')
(case / "constant" / "turbulenceProperties").write_text(
    "RAS\n{\n    RASModel myModel;\n    turbulence on;\n}\n")
(cand / "frozen_alphaK2.json").write_text(json.dumps({"alphaK2": 2.5677, "success": True}))
(cand / "agentic_trajectory.log").write_text("turn 50\n" + "x" * 500 + "\nlast action here")
execution = {"status": "FAILED", "aborted_reason": "timeout after 2249s",
             "duration_s": 2249, "turns_used": 51, "solver_invocations": 40,
             "compile_ok": True, "converged": False, "case_dir": str(case)}

ev = T._unclean_finish_evidence(cand, execution, 2249, 120)
check("evidence names why the agent stopped", "timeout after 2249s" in ev)
check("evidence reports turns against the cap", "51 of a 120-turn cap" in ev, ev[:200])
check("evidence computes seconds per solver launch so an estimate has a basis",
      "per solver launch" in ev and "56s" in ev, [l for l in ev.splitlines() if "per solver" in l])
check("evidence includes the fit artifact the agent left behind",
      "frozen_alphaK2.json" in ev and "2.5677" in ev)
check("evidence includes the compiled model source verbatim",
      "lookupOrAddToDict" in ev and "myModel.C" in ev)
check("evidence includes the case's activation dictionary verbatim",
      "RASModel myModel" in ev)
check("evidence includes the trajectory tail", "last action here" in ev)

# The turn-cap case must be distinguishable from the timeout case.
ev_cap = T._unclean_finish_evidence(cand, {**execution, "turns_used": 120,
                                           "aborted_reason": ""}, 9999, 120)
check("hitting the turn cap is called out explicitly", "AT THE CAP" in ev_cap)

# Nothing here may raise on a candidate that left almost nothing behind.
bare = Path(tempfile.mkdtemp()) / "cand_bare"
bare.mkdir(parents=True)
try:
    T._unclean_finish_evidence(bare, {"status": "FAILED"}, 0, 120)
    check("evidence survives a candidate with an empty directory", True)
except Exception as exc:
    check("evidence survives a candidate with an empty directory", False, repr(exc))

# A diagnosis whose model call fails must return a verdict, never raise.
saved = T.create_langchain_llm if hasattr(T, "create_langchain_llm") else None
import cfd_langgraph.llm.factory as factory
orig_factory = factory.create_langchain_llm
factory.create_langchain_llm = lambda **k: (_ for _ in ()).throw(RuntimeError("provider down"))
try:
    d = T._diagnose_unclean_finish(cand, execution, 2249, 120, 0, get_settings())
    check("a failed diagnosis returns verdict=unknown rather than raising",
          d.get("ok") is False and d.get("verdict") == "unknown", d)
except Exception as exc:
    check("a failed diagnosis returns verdict=unknown rather than raising", False, repr(exc))
finally:
    factory.create_langchain_llm = orig_factory


# ---------------------------------------------------------------------------
# 3. The extension budget and the repair budget
# ---------------------------------------------------------------------------
print("\n--- extension and repair budgets ---")

out = Path(tempfile.mkdtemp())
disc = out / "open_ended_discovery"
disc.mkdir(parents=True)
(disc / "search_config.json").write_text(json.dumps(
    {"topic": "t", "total_budget": 100, "baseline_direction": "min",
     "baseline_case_dir": str(out)}))
cand = disc / "cand_fit"
cand.mkdir(parents=True)
(cand / "candidate_invocation.json").write_text(json.dumps(
    {"hypothesis": "H", "plan": "P", "variant_name": "fit", "strategy": "solver_fit"}))
(cand / "agentic_result.json").write_text(json.dumps(
    {"status": "FAILED", "duration_s": 2249, "turns_used": 51,
     "aborted_reason": "timeout after 2249s", "solver_invocations": 40}))

built = T.build_manager_tools(get_settings(), out)
by = {f.__name__: f for f in built["manager_tools"] + built["oed_candidate_tools"]}
check("oed_extend_candidate is registered on the candidate runner", "oed_extend_candidate" in by)
check("oed_apply_repair is registered on the candidate runner", "oed_apply_repair" in by)
extend, repair = by["oed_extend_candidate"], by["oed_apply_repair"]

# Never actually launch a build agent in a test.
calls = []
T_rerun = None
for name in ("_rerun_build_agent",):
    pass


def fake_rerun(candidate_dir, *, timeout_s, prior_attempt="", repair_goal=""):
    calls.append({"timeout_s": timeout_s, "prior_attempt": prior_attempt,
                  "repair_goal": repair_goal})
    return {"ok": True, "status": "OK", "candidate_dir": candidate_dir}


# The tools close over the real _rerun_build_agent, so patch through the
# closure cell rather than the module namespace.
import inspect
cells = dict(zip(extend.__wrapped__.__code__.co_freevars if hasattr(extend, "__wrapped__") else
                 extend.__code__.co_freevars,
                 (extend.__wrapped__.__closure__ if hasattr(extend, "__wrapped__") else
                  extend.__closure__) or ()))
target = cells.get("_rerun_build_agent")
patched = target is not None
if patched:
    real = target.cell_contents
    target.cell_contents = fake_rerun
check("the extension tool calls a rerun helper that a test can intercept", patched)

if patched:
    r = extend(str(cand), 0, "because")
    check("a non-positive extension is refused", r.get("ok") is False, r)
    r = extend(str(cand), 600, "")
    check("an extension with no stated arithmetic is refused", r.get("ok") is False, r)
    check("a refused extension consumes no budget",
          not (cand / "candidate_attempts.json").exists())

    r = extend(str(cand), 600, "8 of 12 iterations done at ~90s each, ~600s remain")
    check("a justified extension runs", r.get("ok") is True, r)
    check("the grant is the prior duration plus the extension, not the extension alone",
          calls and calls[-1]["timeout_s"] == 2249 + 600, calls[-1] if calls else None)
    check("the continuation is told what the previous attempt did",
          "51 turns" in calls[-1]["prior_attempt"] and "timeout after 2249s" in calls[-1]["prior_attempt"])
    check("the continuation carries the rationale it was granted on",
          "~600s remain" in calls[-1]["prior_attempt"])
    check("one extension is recorded", r.get("extensions_used") == 1)
    check("one extension remains", r.get("extensions_remaining") == 1)

    extend(str(cand), 600, "same again")
    r = extend(str(cand), 600, "a third time")
    check("a third extension is refused", r.get("ok") is False, r)
    check("the refusal says how many were used", r.get("extensions_used") == 2, r)

    # The attempt is banked before the run, so a crash cannot reset the count.
    cand2 = disc / "cand_crash"
    cand2.mkdir()
    (cand2 / "candidate_invocation.json").write_text(json.dumps(
        {"hypothesis": "H", "variant_name": "crash"}))
    (cand2 / "agentic_result.json").write_text(json.dumps({"status": "FAILED", "duration_s": 100}))
    target.cell_contents = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("died mid-run"))
    try:
        extend(str(cand2), 300, "estimate")
    except Exception:
        pass
    banked = json.loads((cand2 / "candidate_attempts.json").read_text())
    check("an extension that crashes mid-run still consumes its attempt",
          banked.get("extensions_used") == 1, banked)
    target.cell_contents = fake_rerun

# Repair
rcells = dict(zip(repair.__wrapped__.__code__.co_freevars if hasattr(repair, "__wrapped__") else
                  repair.__code__.co_freevars,
                  (repair.__wrapped__.__closure__ if hasattr(repair, "__wrapped__") else
                   repair.__closure__) or ()))
rtarget = rcells.get("_rerun_build_agent")
if rtarget is not None:
    rtarget.cell_contents = fake_rerun
    r = repair(str(cand), [], "why")
    check("a repair with no steps is refused", r.get("ok") is False, r)
    r = repair(str(cand), ["add the libs entry"], "")
    check("a repair with no rationale is refused", r.get("ok") is False, r)

    r = repair(str(cand), ["add libs (\"libX.so\") to controlDict"], "library never loaded")
    check("a diagnosed repair runs", r.get("ok") is True, r)
    check("the repair agent is given the steps verbatim",
          "libX.so" in calls[-1]["repair_goal"], calls[-1])
    check("the repair agent is given the reason",
          "library never loaded" in calls[-1]["repair_goal"])
    check("a repair is a repair, not a continuation",
          calls[-1]["prior_attempt"] == "")
    check("one repair attempt is recorded", r.get("repair_attempts_used") == 1)

    repair(str(cand), ["again"], "still broken")
    r = repair(str(cand), ["third"], "still")
    check("a third repair is refused", r.get("ok") is False, r)

# Path safety: neither tool may act outside this study's OED directory.
outside = Path(tempfile.mkdtemp()) / "cand_elsewhere"
outside.mkdir(parents=True)
check("extension refuses a candidate outside this study",
      extend(str(outside), 60, "x").get("ok") is False)
check("repair refuses a candidate outside this study",
      repair(str(outside), ["x"], "y").get("ok") is False)
check("extension refuses a directory that is not a cand_*",
      extend(str(disc), 60, "x").get("ok") is False)


# ---------------------------------------------------------------------------
# 4. The build-agent prompt actually changes shape for a continuation/repair
# ---------------------------------------------------------------------------
print("\n--- build agent prompt modes ---")

import importlib.util
spec = importlib.util.spec_from_file_location(
    "cma", Path(__file__).resolve().parents[1] / "scripts" / "code_mod_agentic.py")
cma = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cma)

base = dict(topic="t", hypothesis="H", variant_name="v",
            starter_case=Path("/tmp/s"), run_dir=Path("/tmp/r"), wm_project_dir=None)
plain = cma.build_agent_prompt(**base)
cont = cma.build_agent_prompt(**base, prior_attempt="ran 51 turns, timed out")
rep = cma.build_agent_prompt(**base, repair_goal="add the libs entry")

check("a plain build is not told it is a continuation", "CONTINUATION" not in plain)
check("a continuation is told so", "CONTINUATION OF AN EARLIER ATTEMPT" in cont)
check("a continuation is told what the earlier attempt did", "ran 51 turns, timed out" in cont)
check("a continuation is told to inspect and build on existing work",
      "INSPECT IT FIRST" in cont)
check("a repair carries the goal", "add the libs entry" in rep)
check("a repair is forbidden from changing the graded setup",
      "may NOT change the mesh" in rep and "closure itself" in rep)
check("a repair is told to say so rather than improvise when it cannot comply",
      "do NOT improvise" in rep)




# ---------------------------------------------------------------------------
# 5. The verdict is ENFORCED, not merely reported
# ---------------------------------------------------------------------------
print("\n--- scoring refuses an incomplete model ---")

out = Path(tempfile.mkdtemp())
disc = out / "open_ended_discovery"
disc.mkdir(parents=True)
(disc / "search_config.json").write_text(json.dumps(
    {"topic": "t", "total_budget": 100, "baseline_direction": "min",
     "baseline_case_dir": str(out)}))
(disc / "baseline_score.json").write_text(json.dumps(
    {"metric": "m", "value": 0.1136, "direction": "min", "verified": True}))
cand = disc / "cand_halffit"
(cand / "case").mkdir(parents=True)
(cand / "candidate_invocation.json").write_text(json.dumps(
    {"hypothesis": "H", "variant_name": "halffit"}))
(cand / "agentic_result.json").write_text(json.dumps({"status": "FAILED", "duration_s": 900}))

built = T.build_manager_tools(get_settings(), out)
by = {f.__name__: f for f in built["manager_tools"] + built["oed_candidate_tools"]}
score = by["oed_score_candidate"]

# No standing verdict: scoring proceeds as before (it will fail for its own
# reasons here, but NOT with the completeness refusal).
r = score(str(cand), str(cand / "case"), "code_mod", "halffit", "d")
check("with no standing verdict, scoring is not blocked by the new gate",
      "Refusing to score" not in str(r.get("error", "")), r)

# The diagnosing model judged the model incomplete -> scoring must refuse.
(cand / "unclean_finish_diagnosis.json").write_text(json.dumps({
    "ok": True, "verdict": "repair", "model_is_complete": False,
    "cause": "sF1 is read from the case dict and the case dict does not set it",
    "repair_steps": ["write sF1 2.5677 into constant/turbulenceProperties"],
    "extra_seconds_needed": 0}))
r = score(str(cand), str(cand / "case"), "code_mod", "halffit", "d")
check("scoring refuses when the verdict says the model is incomplete",
      r.get("ok") is False and "Refusing to score" in str(r.get("error", "")), r)
check("the refusal carries the cause through", "sF1" in str(r.get("cause", "")))
check("the refusal carries the repair steps through", r.get("repair_steps"), r)
check("the refusal says what to do next", "oed_apply_repair" in str(r.get("next_step", "")))
check("no candidate_record.json is written by a refused score",
      not (cand / "candidate_record.json").exists())

# A complete model with an unclean finish is NOT blocked -- the agent ran out
# of turns while tidying up, which is the case the rescue path exists for.
(cand / "unclean_finish_diagnosis.json").write_text(json.dumps({
    "ok": True, "verdict": "complete", "model_is_complete": True, "cause": "fine"}))
r = score(str(cand), str(cand / "case"), "code_mod", "halffit", "d")
check("a complete model is not blocked despite an unclean finish",
      "Refusing to score" not in str(r.get("error", "")), r)

# A diagnosis that itself failed must not block anything: ok=False means the
# model had no opinion, and a tool outage is not evidence of an incomplete run.
(cand / "unclean_finish_diagnosis.json").write_text(json.dumps({
    "ok": False, "verdict": "unknown", "model_is_complete": False,
    "cause": "provider down"}))
r = score(str(cand), str(cand / "case"), "code_mod", "halffit", "d")
check("a diagnosis that could not be made does not block scoring",
      "Refusing to score" not in str(r.get("error", "")), r)

# A later attempt supersedes the verdict.
(cand / "unclean_finish_diagnosis.json").write_text(json.dumps({
    "ok": True, "verdict": "repair", "model_is_complete": False, "cause": "x"}))
rcells = dict(zip(by["oed_apply_repair"].__code__.co_freevars,
                  by["oed_apply_repair"].__closure__ or ()))
t2 = rcells.get("_rerun_build_agent")
if t2 is not None:
    t2.cell_contents = lambda cd, **k: {"ok": True, "status": "OK"}
    by["oed_apply_repair"](str(cand), ["fix it"], "because")
    check("a repair clears the standing verdict so the retry can be scored",
          not (cand / "unclean_finish_diagnosis.json").exists())




# ---------------------------------------------------------------------------
# 6. The gaps found on the second pass: tool placement, and the rescue path
# ---------------------------------------------------------------------------
print("\n--- wiring gaps ---")

built = T.build_manager_tools(get_settings(), Path(tempfile.mkdtemp()))
mgr = {f.__name__ for f in built["manager_tools"]}
runner = {f.__name__ for f in built["oed_candidate_tools"]}
# The manager's own OED loop is instructed to act on these verdicts, so it has
# to hold the tools; a prompt naming a tool the agent does not have is worse
# than no instruction, because it reads as a capability it can plan around.
for tool in ("oed_apply_repair", "oed_extend_candidate"):
    check(f"{tool} is on the manager", tool in mgr)
    check(f"{tool} is on the candidate runner", tool in runner)

# The reconstruction path invents status=OK for a process that died before
# writing its verdict. That OK satisfies the reuse test and would skip every
# check after it, so the completeness question has to be asked there too.
out = Path(tempfile.mkdtemp())
disc = out / "open_ended_discovery"
disc.mkdir(parents=True)
(disc / "search_config.json").write_text(json.dumps(
    {"topic": "t", "total_budget": 100, "baseline_direction": "min",
     "baseline_case_dir": str(out)}))
cand = disc / "cand_died"
case = cand / "died"
(case / "customModels" / "M" / "lib").mkdir(parents=True)
(case / "customModels" / "M" / "lib" / "libM.so").write_bytes(b"\x7fELF" + b"\x00" * 64)
(case / "constant").mkdir(parents=True)
(case / "constant" / "turbulenceProperties").write_text("RAS\n{\n RASModel M;\n}\n")
(case / "system").mkdir(parents=True)
(case / "system" / "controlDict").write_text("application simpleFoam;\n")
(case / "log.simpleFoam").write_text("Starting\nEnd\n")

seen = {}
real_diag = T._diagnose_unclean_finish
T._diagnose_unclean_finish = lambda cp, doc, *a, **k: seen.setdefault(
    "called", {"ok": True, "verdict": "repair", "model_is_complete": False,
               "cause": "coefficient never reached the case"})
try:
    built = T.build_manager_tools(get_settings(), out)
    by = {f.__name__: f for f in built["oed_candidate_tools"]}
    (cand / "candidate_invocation.json").write_text(json.dumps(
        {"hypothesis": "H", "plan": "", "variant_name": "died"}))
    by["oed_run_code_mod_candidate"]("t", "died", "H", "")
    check("a reconstructed candidate is diagnosed rather than trusted",
          "called" in seen)
    check("the reconstruction's verdict is persisted where scoring will see it",
          (cand / "unclean_finish_diagnosis.json").exists())
    if (cand / "unclean_finish_diagnosis.json").exists():
        v = json.loads((cand / "unclean_finish_diagnosis.json").read_text())
        check("the persisted verdict is the diagnosis, not a placeholder",
              v.get("model_is_complete") is False, v)
finally:
    T._diagnose_unclean_finish = real_diag

# A clean finish must clear a verdict left by an earlier attempt.
(cand / "unclean_finish_diagnosis.json").write_text(json.dumps(
    {"ok": True, "verdict": "repair", "model_is_complete": False, "cause": "old"}))
(cand / "agentic_result.json").write_text(json.dumps({"status": "OK", "case_dir": str(case)}))
built = T.build_manager_tools(get_settings(), out)
by = {f.__name__: f for f in built["oed_candidate_tools"]}
r = by["oed_run_code_mod_candidate"]("t", "died", "H", "")
check("a reused clean candidate surfaces any standing verdict to the runner",
      r.get("reused") is True and "unclean_finish_diagnosis" in r, r)




# ---------------------------------------------------------------------------
# 7. Wiring, not just units.
#
# Section 6 existed because a green suite of unit tests said "tested" while two
# connections were missing: a prompt named tools the agent did not hold, and the
# reconstruction path skipped the gate entirely. Unit tests could not see either,
# because both are properties of how the pieces join rather than of any piece.
# Everything below tests a join.
# ---------------------------------------------------------------------------
print("\n--- wiring: every tool a prompt names is on the agent told to call it ---")

import re
from cfd_langgraph.manager.deep_agent import _build_manager_system_prompt

probe = Path(tempfile.mkdtemp())
b = T.build_manager_tools(get_settings(), probe)
sets = {"manager": {f.__name__ for f in b["manager_tools"]},
        "case-runner": {f.__name__ for f in b["case_runner_tools"]},
        "oed-candidate-runner": {f.__name__ for f in b["oed_candidate_tools"]}}
every = set().union(*sets.values())

prompts = {"manager": _build_manager_system_prompt(probe),
           "oed-candidate-runner": subagents._build_oed_candidate_runner_prompt(probe)}
for attr in ("_build_case_runner_prompt", "_build_case_runner_system_prompt"):
    if hasattr(subagents, attr):
        prompts["case-runner"] = getattr(subagents, attr)(probe)
        break

for label, text in prompts.items():
    mentioned = set(re.findall(r"\b((?:oed|foam|run)_[a-z][a-z0-9_]+)\b", text))
    ghosts = sorted(t for t in mentioned if t not in every)
    check(f"{label}: names no tool that does not exist", not ghosts, ghosts)
    if label != "manager":
        # A subagent can only call what it holds. The manager may additionally
        # *describe* a subagent's tools when delegating, so it is exempt here.
        unreachable = sorted(t for t in mentioned if t in every and t not in sets[label])
        check(f"{label}: names no tool it cannot call", not unreachable, unreachable)

# The specific miss: both new tools must be on both agents that are told about them.
for tool in ("oed_apply_repair", "oed_extend_candidate"):
    check(f"{tool} is on the manager", tool in sets["manager"])
    check(f"{tool} is on the candidate runner", tool in sets["oed-candidate-runner"])


print("\n--- wiring: the CLI contract between the tool and the script ---")

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "cma_cli", Path(__file__).resolve().parents[1] / "scripts" / "code_mod_agentic.py")
_cma = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_cma)

_seen = {}
_real_run = _cma.run
_cma.run = lambda **kw: _seen.update(kw) or {"status": "OK"}
_argv0 = sys.argv
_base = ["code_mod_agentic.py", "--hypothesis", "H", "--variant-name", "v",
         "--run-dir", "/tmp/r", "--starter-case", "/tmp/s", "--topic", "t",
         "--output", str(Path(tempfile.mkdtemp()) / "o.json")]
try:
    _seen.clear()
    sys.argv = _base + ["--timeout", "2849", "--max-turns", "120",
                        "--prior-attempt", "ran 51 turns, timed out"]
    _cma.main()
    check("a continuation argv reaches run() as prior_attempt",
          _seen.get("prior_attempt", "").startswith("ran 51"), _seen.get("prior_attempt"))
    check("a continuation is not also sent as a repair", _seen.get("repair_goal") == "")
    check("the granted timeout reaches run()", _seen.get("timeout_s") == 2849)
    check("the turn cap the diagnosis is told about is the one actually used",
          _seen.get("max_turns") == T._OED_MAX_TURNS, _seen.get("max_turns"))

    _seen.clear()
    sys.argv = _base + ["--repair-goal", "add libs entry"]
    _cma.main()
    check("a repair argv reaches run() as repair_goal", _seen.get("repair_goal") == "add libs entry")
    check("a repair is not also sent as a continuation", _seen.get("prior_attempt") == "")

    _seen.clear()
    sys.argv = list(_base)
    _cma.main()
    check("a plain build sets neither flag",
          _seen.get("prior_attempt") == "" and _seen.get("repair_goal") == "")
finally:
    sys.argv = _argv0
    _cma.run = _real_run

import inspect as _inspect
for _fn in (_cma.run, _cma.run_agent_loop, _cma.build_agent_prompt):
    check(f"{_fn.__name__} accepts both new parameters",
          {"prior_attempt", "repair_goal"} <= set(_inspect.signature(_fn).parameters))


print("\n--- wiring: the whole chain, in order ---")


def chain_fixture(diagnosis):
    """A study whose build subprocess fails once, then succeeds."""
    import types
    out = Path(tempfile.mkdtemp())
    disc = out / "open_ended_discovery"
    disc.mkdir(parents=True)
    starter = out / "starter"
    starter.mkdir()
    (disc / "search_config.json").write_text(json.dumps(
        {"topic": "t", "total_budget": 100, "baseline_direction": "min",
         "baseline_case_dir": str(starter)}))
    (disc / "baseline_score.json").write_text(json.dumps(
        {"metric": "m", "value": 0.1136, "direction": "min", "verified": True}))
    (disc / "history.json").write_text("[]")
    cand = disc / "cand_fit"
    case = cand / "fit"
    case.mkdir(parents=True)
    launches, n = [], {"i": 0}

    def fake(argv, **kw):
        n["i"] += 1
        launches.append({
            "timeout": int(argv[argv.index("--timeout") + 1]),
            "subprocess_timeout": kw.get("timeout"),
            "prior": argv[argv.index("--prior-attempt") + 1] if "--prior-attempt" in argv else "",
            "repair": argv[argv.index("--repair-goal") + 1] if "--repair-goal" in argv else "",
        })
        done = n["i"] > 1
        Path(argv[argv.index("--output") + 1]).write_text(json.dumps(
            {"status": "OK" if done else "FAILED", "case_dir": str(case),
             "duration_s": 2900 if done else 2249, "turns_used": 74 if done else 51,
             "aborted_reason": "" if done else "timeout after 2249s",
             "solver_invocations": 40, "compile_ok": True, "converged": done}))
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    T._run_script = fake
    T._diagnose_unclean_finish = lambda *a, **k: dict(diagnosis)
    built = T.build_manager_tools(get_settings(), out)
    return ({f.__name__: f for f in built["manager_tools"] + built["oed_candidate_tools"]},
            disc, cand, case, launches)


saved_run_script, saved_diag = T._run_script, T._diagnose_unclean_finish
try:
    by, disc, cand, case, launches = chain_fixture(
        {"ok": True, "verdict": "extend", "model_is_complete": False,
         "cause": "sF1 is read from the case dict and the case dict does not set it",
         "stopped_because": "timeout", "extra_seconds_needed": 600,
         "estimate_basis": "4 of 12 iterations left at ~150s each",
         "repair_steps": [], "alters_graded_setup": False, "confidence": 0.9})

    r = by["oed_run_code_mod_candidate"]("t", "fit", "H", "P", "solver_fit")
    check("a failed build returns ok=False", r.get("ok") is False)
    check("a failed build carries the diagnosis to the runner",
          r.get("unclean_finish_diagnosis", {}).get("verdict") == "extend")
    check("the candidate is fenced against its own strategy",
          r.get("fenced_against_strategy") == "solver_fit", r.get("fenced_against_strategy"))
    check("the runner is told to read the diagnosis first",
          "unclean_finish_diagnosis" in r.get("next_step", ""))
    check("the verdict is persisted for the scoring gate",
          (cand / "unclean_finish_diagnosis.json").exists())

    r = by["oed_score_candidate"](str(cand), str(case), "code_mod", "fit", "d")
    check("scoring an incomplete model is refused", "Refusing to score" in str(r.get("error", "")))
    check("the refusal carries the diagnosis's own numbers",
          r.get("extra_seconds_needed") == 600 and "sF1" in str(r.get("cause", "")))
    check("a refused score writes no record", not (cand / "candidate_record.json").exists())

    r = by["oed_extend_candidate"](str(cand), 600, "4 of 12 iterations left at ~150s each")
    check("the extension runs", r.get("ok") is True, r)
    check("the grant is prior duration + extension", launches[-1]["timeout"] == 2249 + 600,
          launches[-1]["timeout"])
    check("the continuation is told what the previous attempt did",
          "51 turns" in launches[-1]["prior"] and "timeout after 2249s" in launches[-1]["prior"])
    check("a clean finish clears the standing verdict",
          not (cand / "unclean_finish_diagnosis.json").exists())

    r = by["oed_score_candidate"](str(cand), str(case), "code_mod", "fit", "d")
    check("scoring is no longer blocked after a successful continuation",
          "Refusing to score" not in str(r.get("error", "")))

    # The ceiling: a wildly wrong estimate must not hold a coordinator slot.
    huge = disc / "cand_huge"
    huge.mkdir()
    (huge / "candidate_invocation.json").write_text(json.dumps({"hypothesis": "H", "variant_name": "huge"}))
    (huge / "agentic_result.json").write_text(json.dumps({"status": "FAILED", "duration_s": 3000}))
    r = by["oed_extend_candidate"](str(huge), 200000, "55 hours please")
    check("an over-large extension is clipped to the ceiling",
          r.get("total_granted_s") == T._OED_MAX_EXTENDED_S, r.get("total_granted_s"))
    check("the clip is reported rather than applied silently", "ceiling" in r.get("note", ""))
    check("the subprocess bound respects the ceiling too",
          launches[-1]["subprocess_timeout"] <= T._OED_MAX_EXTENDED_S + 1800,
          launches[-1]["subprocess_timeout"])

    # A repair-only record must not reach history as a scoreless evaluation.
    # The sweep's old comment claimed candidate_record.json only exists after
    # scoring; the repair path made that false, and what actually holds the
    # line is the case_dir check.
    orphan = disc / "cand_orphan"
    orphan.mkdir()
    (orphan / "candidate_record.json").write_text(json.dumps({"repair_attempts": 1}))
    by["oed_record_candidate_results"]([])
    hist = json.loads((disc / "history.json").read_text())
    check("a repair-only record is not swept into history as an evaluation",
          all(h.get("candidate_dir", "").endswith("cand_orphan") is False for h in hist),
          [h.get("candidate_dir") for h in hist])
finally:
    T._run_script, T._diagnose_unclean_finish = saved_run_script, saved_diag




# ---------------------------------------------------------------------------
# 8. The four fixes for why solver_fit never completed.
#
# Root cause, measured on run closure_20260826_codex: one objective evaluation
# costs ~23 minutes because it solves six cases to convergence, so a
# 32-evaluation differential evolution needs 12.4 hours against a 0.9-hour
# fence. The agent was killed, its BACKGROUNDED fit ran on for another 5.7
# hours orphaned, and the diagnosis then read the fit as "no converged trial"
# because it never looked at the ledger sitting beside the trajectory.
# ---------------------------------------------------------------------------
print("\n--- fit-cost fixes ---")

out = Path(tempfile.mkdtemp())
cand = out / "cand_fitted"
case = cand / "fitted"
(case / "constant").mkdir(parents=True)
(case / "constant" / "turbulenceProperties").write_text("RAS\n{\n RASModel M;\n}\n")
# A ledger of the shape a real fit writes, with a near-flat objective.
(cand / "objective_ledger.jsonl").write_text("\n".join(
    json.dumps({"cXG": c, "mean_velocity_mae": m, "elapsed_seconds": 1400})
    for c, m in [(0.059, 0.074431), (0.137, 0.074395), (0.176, 0.074378),
                 (0.236, 0.074351), (0.264, 0.074338), (0.270, 0.074336)]))
(cand / "fit_status_bounded.json").write_text(json.dumps(
    {"bounds": [0.35, 0.70], "selected_b": 0.6174,
     "optimizer": {"nfev": 3, "nit": 3, "message": "Solution found."}}))
(cand / "FIT_INCOMPLETE.json").write_text(json.dumps(
    {"status": "incomplete_do_not_score_as_fitted",
     "error": "incomplete differential-evolution ledger: nfev=16, expected=32"}))
(cand / "agentic_trajectory.log").write_text("turn 32\nlast action")

ev = T._unclean_finish_evidence(
    cand, {"status": "FAILED", "aborted_reason": "timeout after 3121s",
           "duration_s": 3336, "turns_used": 32, "solver_invocations": 31,
           "compile_ok": True, "converged": False, "case_dir": str(case)}, 3121, 120)

check("evidence surfaces the fit ledger at all", "objective_ledger.jsonl" in ev)
check("evidence says how many objective evaluations were recorded",
      "6 objective evaluation" in ev, [l for l in ev.splitlines() if "objective evaluation" in l])
check("evidence carries the objective values so flatness is visible",
      "0.074431" in ev and "0.074336" in ev)
check("evidence carries per-evaluation elapsed time so cost is computable",
      "1400" in ev)
check("evidence surfaces the optimiser's bounds and evaluation count",
      "0.35" in ev and "nfev" in ev, "fit_status not shown")
check("evidence surfaces a fit that declared itself incomplete",
      "incomplete_do_not_score_as_fitted" in ev)

# A candidate with no fit at all must not gain spurious ledger sections.
plain = Path(tempfile.mkdtemp()) / "cand_plain"
plain.mkdir(parents=True)
ev_plain = T._unclean_finish_evidence(plain, {"status": "FAILED"}, 0, 120)
check("a candidate with no fit gets no ledger section", "FIT LEDGER" not in ev_plain)

# The build agent must be told its deadline and the fit rules.
import importlib.util as _i2
_s2 = _i2.spec_from_file_location(
    "cma2", Path(__file__).resolve().parents[1] / "scripts" / "code_mod_agentic.py")
_c2 = _i2.module_from_spec(_s2)
_s2.loader.exec_module(_c2)
_base = dict(topic="t", hypothesis="H", variant_name="v",
             starter_case=Path("/tmp/s"), run_dir=Path("/tmp/r"), wm_project_dir=None)

budgeted = _c2.build_agent_prompt(**_base, timeout_s=3121)
unbudgeted = _c2.build_agent_prompt(**_base)
check("the agent is told its wall-clock budget in seconds", "3121 seconds" in budgeted)
check("the agent is told its budget in minutes too", "52 minutes" in budgeted, budgeted[:0])
check("the agent is told to cost a fit before starting it",
      "BEFORE you start any fit" in budgeted and "total seconds" in budgeted)
check("the agent is told to time one case rather than guess",
      "Time ONE case first" in budgeted)
check("the agent is told a finished small fit beats a killed thorough one",
      "FINISHES is worth more" in budgeted)
check("no budget block when no fence is set", "YOUR TIME BUDGET" not in unbudgeted)

withplan = _c2.build_agent_prompt(**_base, plan="fit b by solver-in-the-loop")
check("fit rules: never background", "NEVER background it" in withplan and "nohup" in withplan)
check("fit rules: probe the ends for flatness first",
      "Probe the ends of the range BEFORE optimising" in withplan)
check("fit rules: a value on a bound means the bounds are wrong",
      "sitting on a bound" in withplan)
check("fit rules: a handful of evaluations is not a fit",
      "not a fit" in withplan)
check("fit rules: write the ledger as it happens", "ledger file on disk AS IT" in withplan)
check("fit rules: write the fitted value into the case dictionary",
      "WRITE THE SELECTED COEFFICIENT INTO THE CASE" in withplan)

# And the sandbox must make orphaning impossible, not merely discouraged.
import inspect as _insp
src = _insp.getsource(_c2.Sandbox.run_bash)
check("the shell sandbox unshares the PID namespace so nothing outlives the call",
      '"--unshare-pid"' in src)
check("--die-with-parent is still set", '"--die-with-parent"' in src)




# ---------------------------------------------------------------------------
# 9. No agent is ever handed a deadline it cannot actually be given.
#
# The fence is an outlier bound, and on a strategy whose successes are spread
# wide it stops bounding: offline_fit reached 96721s (26.9 h) on the real study
# from four successes of very different cost. That mattered only once the agent
# started planning against the number, which is new.
# ---------------------------------------------------------------------------
print("\n--- the fence can never exceed the ceiling ---")

import types as _t
out = Path(tempfile.mkdtemp())
disc = out / "open_ended_discovery"
disc.mkdir(parents=True)
starter = out / "starter"
starter.mkdir()
(disc / "search_config.json").write_text(json.dumps(
    {"topic": "t", "total_budget": 100, "baseline_direction": "min",
     "baseline_case_dir": str(starter)}))
# Four successes spread widely enough to blow the log-Tukey bound out.
for i, d in enumerate([600, 1800, 9000, 40000]):
    c = disc / f"cand_w{i}"
    c.mkdir()
    (c / "agentic_result.json").write_text(json.dumps({"status": "OK", "duration_s": d}))
    (c / "candidate_record.json").write_text(json.dumps({"strategy": "offline_fit"}))

raw = T._oed_candidate_timeout(disc, "offline_fit")
check("a wide-spread strategy really does blow the raw fence past the ceiling",
      raw is not None and raw > T._OED_MAX_EXTENDED_S, f"raw={raw}")

seen = {}
def _capture(argv, **kw):
    seen["timeout_flag"] = int(argv[argv.index("--timeout") + 1])
    seen["subprocess_timeout"] = kw.get("timeout")
    Path(argv[argv.index("--output") + 1]).write_text(json.dumps({"status": "OK"}))
    return _t.SimpleNamespace(returncode=0, stdout="", stderr="")

saved = T._run_script
T._run_script = _capture
try:
    built = T.build_manager_tools(get_settings(), out)
    by = {f.__name__: f for f in built["oed_candidate_tools"]}
    r = by["oed_run_code_mod_candidate"]("t", "wide", "H", "P", "offline_fit")
    check("the agent is handed the ceiling, not the raw fence",
          seen.get("timeout_flag") == T._OED_MAX_EXTENDED_S,
          f"told {seen.get('timeout_flag')}s, ceiling {T._OED_MAX_EXTENDED_S}s")
    check("the deadline reported back matches what the agent was told",
          r.get("granted_timeout_s") == seen.get("timeout_flag"), r.get("granted_timeout_s"))
    # The subprocess must outlive the deadline the agent plans against, or the
    # agent budgets to one number and is killed at an earlier one -- the exact
    # failure the time-budget block exists to prevent.
    check("the subprocess timeout sits above the agent's own deadline",
          seen.get("subprocess_timeout", 0) > seen.get("timeout_flag", 0),
          f"subprocess={seen.get('subprocess_timeout')} agent={seen.get('timeout_flag')}")
finally:
    T._run_script = saved




# ---------------------------------------------------------------------------
# 10. The kOmegaSST derivation boilerplate is stated, not rediscovered.
#
# On run closure_gemini, sst_sensitized_bradshaw_limiter made 12 wmake attempts
# over 119 turns and never compiled once. Its 109 errors all read
# "'dimensionedScalar' does not name a type" and all pointed inside
# kOmegaSSTBase.H -- a file the agent had not written -- so it spent 58
# read_file calls searching OpenFOAM source for a type that was never missing.
# Five rules applied to its own two files took it 109 -> 27 -> 7 -> 1 -> 0.
# ---------------------------------------------------------------------------
print("\n--- kOmegaSST derivation guidance ---")

import importlib.util as _i3
_s3 = _i3.spec_from_file_location(
    "cma3", Path(__file__).resolve().parents[1] / "scripts" / "code_mod_agentic.py")
_c3 = _i3.module_from_spec(_s3)
_s3.loader.exec_module(_c3)
guide = _c3.build_agent_prompt(
    topic="t", hypothesis="derive from kOmegaSST", variant_name="v",
    starter_case=Path("/tmp/s"), run_dir=Path("/tmp/r"), wm_project_dir=None)

for label, needle in [
    ("prerequisite includes, not just kOmegaSST.H", '#include "eddyViscosity.H"'),
    ("kOmegaSSTBase.H named as the real base header", "kOmegaSSTBase.H"),
    ("two template arguments spelled out",
     "eddyViscosity<RASModel<BasicMomentumTransportModel>>"),
    ("the 1-vs-2 template-arg error named", "wrong number of template arguments"),
    ("type is the FIRST constructor argument", "type, alpha, rho, U, alphaRhoPhi"),
    ("the .C needs its own include guard", "#ifndef MyModel_C"),
    ("the redefinition symptom named", "redefinition of"),
    ("registration header and makeRASModel", "makeIncompressibleMomentumTransportModel.H"),
    ("the NoRepository pairing explained", "#ifdef NoRepository"),
    ("the key heuristic: errors inside WM_PROJECT_DIR mean YOUR bug",
     "THE BUG IS IN YOUR"),
    ("told to copy a working sibling rather than re-derive", "fastest reference"),
]:
    check(f"build prompt states: {label}", needle in guide, needle)

# It must not fire for unrelated modification families -- this is one section of
# a generic prompt, so it should be present but the deliverable must survive.
check("the generic deliverable is still intact alongside it",
      "DELIVERABLE (generic" in guide and "wmake libso" in guide)

print(f"\n{'ALL PASS' if F == 0 else str(F) + ' FAILED'}")
sys.exit(1 if F else 0)
