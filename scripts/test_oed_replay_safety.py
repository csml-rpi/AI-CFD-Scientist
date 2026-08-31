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


def main() -> int:
    test_replay_reuses_a_finished_candidate()
    test_a_finished_candidate_is_never_stranded()
    print()
    print("ALL PASS" if not FAILURES else f"{FAILURES} FAILURE(S)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
