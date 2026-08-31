"""The null-score diagnosis and its two-attempt repair budget, end to end."""
from __future__ import annotations
import json, shutil, sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cfd_langgraph.manager import tools as T
from cfd_langgraph.config import get_settings

F = 0
def check(name, cond, detail=""):
    global F
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail and not cond else ""))
    if not cond: F += 1

def fixture(verdict):
    out = Path(tempfile.mkdtemp()); disc = out / "open_ended_discovery"; disc.mkdir(parents=True)
    (disc / "search_config.json").write_text(json.dumps({"topic":"t","total_budget":10,"baseline_direction":"min"}))
    (disc / "baseline_score.json").write_text(json.dumps({"metric":"m","value":0.004,"direction":"min","verified":True}))
    cand = disc / "cand_x"; (cand / "case").mkdir(parents=True)
    (cand / "agentic_result.json").write_text(json.dumps({
        "status":"FAILED","case_dir":str((cand/"case").resolve()),
        "compile_ok":True,"converged":False,"compiled_model_name":"M","solver_invocations":5}))
    (cand / "evaluation_run_result.json").write_text(json.dumps({
        "cases_declared":32,"cases_succeeded":30,"solver_invocations":30,
        "failures":[{"case":"CBFS","case_dir":str(cand/"case"),"error":"diverged"}],
        "timed_out_cases":0,"abandoned":False}))
    (cand / "candidate_record.json").write_text(json.dumps({"status":"FAILED","score":None,"score_error":""}))
    base = {"ok":True,"cause":"stub","category":"harness","repairable":False,
            "alters_graded_setup":False,"repair_steps":[],"confidence":0.9,"score_anyway":False}
    base.update(verdict)
    T._diagnose_null_score = lambda *a, **k: dict(base)
    built = T.build_manager_tools(get_settings(), out)
    by = {f.__name__: f for f in built["manager_tools"] + built["oed_candidate_tools"]}
    return by["oed_diagnose_candidate"], by["oed_note_repair_attempt"], cand, out

orig = T._diagnose_null_score
try:
    # 1. repairable + safe -> may repair, 2 attempts
    diag, note, cand, out = fixture({"repairable":True,"alters_graded_setup":False,
                                     "repair_steps":["relax under-relaxation"],"cause":"solver stiffness"})
    r = diag(candidate_dir=str(cand))
    check("a repairable, safe failure may be repaired", r["may_repair"] is True, str(r)[:160])
    check("with the full budget of 2", r["repair_attempts_remaining"] == 2)
    check("and concrete steps", r["diagnosis"]["repair_steps"] == ["relax under-relaxation"])
    check("the diagnosis is persisted to the record",
          bool(json.loads((cand/"candidate_record.json").read_text()).get("failure_diagnosis")))

    n1 = note(candidate_dir=str(cand), what_was_changed="relaxed factors")
    check("attempt 1 counted", n1["repair_attempts_used"] == 1 and n1["repair_attempts_remaining"] == 1)
    n2 = note(candidate_dir=str(cand), what_was_changed="tightened tolerance")
    check("attempt 2 counted", n2["repair_attempts_used"] == 2 and n2["repair_attempts_remaining"] == 0)
    check("the last attempt says so", "last attempt" in n2["next_step"])
    r3 = diag(candidate_dir=str(cand))
    check("budget exhausted blocks further repair", r3["may_repair"] is False, str(r3)[:160])
    check("and says to move on", "move on" in r3["next_step"].lower(), r3["next_step"])
    log = json.loads((cand/"candidate_record.json").read_text())["repair_log"]
    check("both attempts are logged with what changed",
          [e["change"] for e in log] == ["relaxed factors","tightened tolerance"], str(log))
    shutil.rmtree(out, ignore_errors=True)

    # 2. a repair that would alter the graded setup is refused, however confident
    diag, note, cand, out = fixture({"repairable":True,"alters_graded_setup":True,
                                     "repair_steps":["coarsen the mesh"],"confidence":0.99})
    r = diag(candidate_dir=str(cand))
    check("a repair touching the graded setup is refused", r["may_repair"] is False, str(r)[:160])
    check("and says so explicitly", "DO NOT repair" in r["next_step"], r["next_step"][:120])
    shutil.rmtree(out, ignore_errors=True)

    # 3. genuinely broken model -> no repair, move on
    diag, note, cand, out = fixture({"repairable":False,"category":"model_physics",
                                     "cause":"the closure diverged on every case"})
    r = diag(candidate_dir=str(cand))
    check("an unrepairable model is not retried", r["may_repair"] is False)
    check("and the advice is to spend budget elsewhere",
          "different mechanism" in r["next_step"], r["next_step"][:120])
    shutil.rmtree(out, ignore_errors=True)

    # 4. guards
    diag, note, cand, out = fixture({})
    bad = diag(candidate_dir=str(out))
    check("a path outside the OED dir is refused", bad.get("ok") is False, str(bad)[:120])
    missing = diag(candidate_dir=str(cand.parent / "cand_nope"))
    check("a missing candidate is refused", missing.get("ok") is False, str(missing)[:120])
    shutil.rmtree(out, ignore_errors=True)
finally:
    T._diagnose_null_score = orig

print()
print("ALL PASS" if not F else f"{F} FAILURE(S)")
sys.exit(1 if F else 0)
