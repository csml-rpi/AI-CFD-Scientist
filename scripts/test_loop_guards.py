"""Loop guards and flagged requirements, as seen on the Qwen cavity and hill runs.

  1. the same call returning the same result 8 times in a row gets a warning
     (a case-runner wrote the same script 10,000 times on qwen cavity_r8)
  2. a call failing identically 12 times is handed back to the model as
     blocked; the study is not paused for a person (qwen periodic_hill sat 9h44m)
  3. the pause message no longer claims Ctrl-C was pressed
  4. a requirement the checker flagged keeps the checker's issues, run_case_native
     returns them, and accepts a clarification for that case only
"""
import json, os, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.environ.update(CFD_SCIENTIST_LLM_PROVIDER="openai", CFD_SCIENTIST_MODEL="fake/model",
                  OPENAI_API_KEY="fake", OPENAI_BASE_URL="http://127.0.0.1:9/v1")

fails = []
def check(label, got, want):
    if got == want:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label}\n        got={got}\n        want={want}")
        fails.append(label)

from cfd_langgraph.manager import tools as mt  # noqa: E402
from cfd_langgraph.manager.control import GLOBAL_INTERRUPT, build_interrupt_on  # noqa: E402

print("== 1. the same call with the same result, repeated")
def write_text_file(path: str, content: str) -> dict:
    return {"path": path, "bytes_written": len(content)}
w = mt._with_progress(write_text_file)
out = [w("/x/run.sh", "same") for _ in range(8)]
check("no warning before the 8th identical call", any("loop_warning" in r for r in out[:7]), False)
check("warning on the 8th", "loop_warning" in out[7], True)
check("the count is reported", out[7].get("repeated_identical_calls"), 8)
check("the original result is kept", out[7]["bytes_written"], 4)
check("a different argument starts a fresh count", "loop_warning" in w("/x/other.sh", "same"), False)

counter = {"n": 0}
def list_directory(path: str) -> dict:
    counter["n"] += 1
    return {"entries": counter["n"]}  # the result changes every call
l = mt._with_progress(list_directory)
check("a changing result never warns", any("loop_warning" in l("/x") for _ in range(12)), False)

def read_text_file(path: str) -> str:
    return "same text"
r = mt._with_progress(read_text_file)
texts = [r("/x/f") for _ in range(8)]
check("a text result gets the warning appended", "exact call 8 times" in texts[7], True)

def oed_candidate_status(candidate_dir: str) -> dict:
    return {"state": "running"}
st = mt._with_progress(oed_candidate_status)
check("status polling is never flagged", any("loop_warning" in st("/c") for _ in range(20)), False)

print("== 2. an identical failure, repeated, is handed back, not paused")
GLOBAL_INTERRUPT.clear()
def oed_propose_candidates(num_candidates: int) -> dict:
    return {"error": "No candidate survived validation this round."}
f = mt._with_progress(oed_propose_candidates)
res = [f(1) for _ in range(mt._REPEAT_FAILURE_ABORT)]
check("the study is not paused", GLOBAL_INTERRUPT.is_set(), False)
check("the model is told the call is blocked", "BLOCKED" in res[-1]["error"], True)
check("the original error is still there", res[-1]["error"].startswith("No candidate survived"), True)
check("not blocked before the threshold", "BLOCKED" in res[-2]["error"], False)

print("== 3. the pause message")
desc = build_interrupt_on([oed_propose_candidates])["oed_propose_candidates"]["description"]
check("does not claim Ctrl-C was pressed", "Ctrl-C was pressed" in desc, False)

print("== 4. a flagged requirement: issues kept, returned, and clarifiable")
from cfd_langgraph.config import get_settings  # noqa: E402
import cfd_langgraph.agents.hypothesis_agent as ha  # noqa: E402

with tempfile.TemporaryDirectory() as tmp:
    out_dir = Path(tmp)
    (out_dir / "hypotheses_approved.json").write_text(json.dumps({"approved_hypotheses": [{
        "candidate_id": "cand_01", "idea": {"study_id": "s1", "experiments": [
            {"experiment_id": "exp_001", "name": "a"}, {"experiment_id": "exp_002", "name": "b"}]}}]}))
    (out_dir / "hypotheses_ranked.json").write_text(json.dumps({"research_topic": "t"}))
    tools = {fn.__name__: fn for fn in mt.build_manager_tools(get_settings(), out_dir)["manager_tools"]}
    runner = {fn.__name__: fn for fn in mt.build_manager_tools(get_settings(), out_dir)["case_runner_tools"]}

    original = ha.HypothesisAgent.generate_validated_requirement
    def fake(self, idea, simulation, run_topic="", **kw):
        if simulation["simulation_id"] == "exp_001":
            return {"requirement": "Run the cavity at Re = 100, 400 and 1000.", "valid": False,
                    "final_verdict": {"valid": False, "issues": ["Lists three Reynolds numbers for one case."]}}
        return {"requirement": "Run the cavity at Re = 400.", "valid": True}
    ha.HypothesisAgent.generate_validated_requirement = fake
    try:
        gen = tools["generate_case_requirements"]()
    finally:
        ha.HypothesisAgent.generate_validated_requirement = original
    reqs = {r["case_id"]: r for r in json.loads((out_dir / "requirements.json").read_text())}
    check("the flagged requirement keeps the checker's issues",
          reqs["case_001"].get("requirement_issues"), ["Lists three Reynolds numbers for one case."])
    check("a passing requirement carries none", "requirement_issues" in reqs["case_002"], False)
    check("the manager is told each flagged case's issues",
          gen.get("unvalidated_issues"), {"case_001": ["Lists three Reynolds numbers for one case."]})

    seed = out_dir / "mesh_gate" / "g" / "baseline"
    seed.mkdir(parents=True)
    (out_dir / "mesh_gate" / "g" / "selected_mesh_spec.json").write_text(
        json.dumps({"converged": True, "selected_level": str(seed)}))
    seen = []
    real_run = mt.foam_native.run_foam_case
    mt.foam_native.run_foam_case = lambda llm, case_dir, requirement, **kw: (
        seen.append(requirement) or {"status": "success", "success": True})
    try:
        run = runner["run_case_native"]
        flagged_text = reqs["case_001"]["user_requirement_text"]
        r1 = run("case_001", flagged_text, physics_group="g")
        check("the run returns the checker's issues", r1.get("requirement_checker_issues"),
              ["Lists three Reynolds numbers for one case."])
        check("and says what to do about them", "clarification=" in r1.get("requirement_checker_note", ""), True)
        check("the writer got the requirement unchanged", seen[-1], flagged_text)
        r2 = run("case_001", flagged_text, physics_group="g", clarification="This case is Re = 400, nu = 0.0025.")
        check("a clarification reaches the writer", "This case is Re = 400" in seen[-1], True)
        check("after the verbatim requirement", seen[-1].startswith(flagged_text), True)
        check("and is recorded in the result", r2.get("clarification"), "This case is Re = 400, nu = 0.0025.")
        ok_text = reqs["case_002"]["user_requirement_text"]
        r3 = run("case_002", ok_text, physics_group="g", clarification="anything")
        check("a clarification on a passing requirement is refused", "only accepted" in r3.get("error", ""), True)
        r4 = run("case_002", ok_text, physics_group="g")
        check("a passing requirement reports no issues", "requirement_checker_issues" in r4, False)
        check("a changed requirement_text is still refused",
              "does not match" in run("case_001", "something else", physics_group="g").get("error", ""), True)
    finally:
        mt.foam_native.run_foam_case = real_run

print()
print("FAILURES:", fails if fails else "none")
sys.exit(1 if fails else 0)
