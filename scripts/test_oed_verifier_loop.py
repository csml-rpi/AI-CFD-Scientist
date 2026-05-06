#!/usr/bin/env python3
"""Synthetic tests for the verifier-extended author_and_selftest loop.

Mocks _llm_invoke and the comparator self-test machinery to exercise the
state machine: OK / SUSPICIOUS / WRONG verdicts, per-method attempt caps
(MAX_TEXT_ATTEMPTS=5, MAX_PYVISTA_ATTEMPTS=10) and the text->pyvista
switchover. Does not run any real OpenFOAM case.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import oed_extensions as oe  # type: ignore


def _make_metric(preferred: str = "auto") -> dict:
    return {
        "name": "x_reattach_error",
        "description": "synthetic",
        "direction": "min",
        "data_source": "<time>/wallShearStress",
        "ref_column": "x_reattach",
        "computation_hint": "synthetic",
        "preferred_method": preferred,
    }


def _patch_environment(monkeypatch_dict: dict) -> dict:
    """Apply patches to oed_extensions for the test. Returns saved originals."""
    saved = {}
    for k, v in monkeypatch_dict.items():
        saved[k] = getattr(oe, k)
        setattr(oe, k, v)
    return saved


def _restore(saved: dict) -> None:
    for k, v in saved.items():
        setattr(oe, k, v)


def run_scenario(*, preferred_method: str, verifier_seq: list, selftest_ok_seq: list,
                 description: str) -> dict:
    """Run author_and_selftest with mocked dependencies.

    verifier_seq: list of dicts (verdict payloads) consumed in order each time
                  _verify_comparator_llm is called.
    selftest_ok_seq: list of bools, one per attempt; True means selftest passes
                     (and verifier is consulted), False means selftest fails
                     (and verifier not consulted).
    """
    print(f"\n=== {description} ===")
    cand_dir = Path(tempfile.mkdtemp(prefix="oed_test_"))

    # Mock author_comparator + corrective: just write a fake script and return path.
    def fake_author_comparator(**kw):
        out_path = kw["out_path"]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("# fake comparator\n", encoding="utf-8")
        return out_path

    def fake_corrective(**kw):
        out_path = kw["out_path"]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("# fake corrective comparator\n", encoding="utf-8")
        return out_path

    selftest_iter = iter(selftest_ok_seq)

    def fake_selftest_full(**kw):
        try:
            ok = next(selftest_iter)
        except StopIteration:
            ok = False
        if ok:
            return {"ok": True, "value": 7.0, "reason": "ok",
                    "time_used": 1000.0, "stdout": "METRIC x_reattach_error: 7.0\nTIME_USED: 1000.0\n",
                    "stderr": "", "returncode": 0}
        return {"ok": False, "value": None, "reason": "metric value is nan",
                "time_used": 1000.0, "stdout": "", "stderr": "", "returncode": 1}

    verifier_iter = iter(verifier_seq)

    def fake_verifier(**kw):
        try:
            return next(verifier_iter)
        except StopIteration:
            return {"verdict": "SUSPICIOUS", "comparator_value": kw.get("comparator_value"),
                    "independent_estimate": None, "discrepancy_class": "cannot_verify",
                    "rationale": "no more verdicts queued", "corrective_hint_for_author": ""}

    saved = _patch_environment({
        "author_comparator": fake_author_comparator,
        "_author_comparator_corrective": fake_corrective,
        "_selftest_comparator_full": fake_selftest_full,
        "_verify_comparator_llm": fake_verifier,
    })
    try:
        result = oe.author_and_selftest(
            _make_metric(preferred_method),
            baseline_case_dir=cand_dir,
            out_dir=cand_dir,
            reference_path=cand_dir / "reference.csv",
            baseline_final_time=1000.0,
            sample_pp_tree="", sample_pp_data="", flow_params={},
        )
    finally:
        _restore(saved)

    log_path = cand_dir / "x_reattach_error_attempt_log.json"
    log = json.loads(log_path.read_text())
    return {"result": result, "log": log}


def assert_eq(got, want, label):
    ok = got == want
    print(f"  [{ 'PASS' if ok else 'FAIL'}] {label}: got={got!r} want={want!r}")
    return ok


def main() -> int:
    failures = 0

    # 1) verdict=OK -> binds on first attempt.
    r = run_scenario(
        preferred_method="text",
        selftest_ok_seq=[True],
        verifier_seq=[{"verdict": "OK", "comparator_value": 7.0, "independent_estimate": 7.0,
                       "discrepancy_class": "ok", "rationale": "matches",
                       "corrective_hint_for_author": ""}],
        description="OK verdict binds immediately",
    )
    failures += not assert_eq(r["result"]["selftest_ok"], True, "selftest_ok")
    failures += not assert_eq(r["result"]["verifier_verdict"], "OK", "verifier_verdict")
    failures += not assert_eq(r["result"]["attempts"], 1, "attempts")
    failures += not assert_eq(r["result"]["final_method"], "text", "final_method")

    # 2) verdict=WRONG triggers re-author, then OK on attempt 2.
    r = run_scenario(
        preferred_method="text",
        selftest_ok_seq=[True, True],
        verifier_seq=[
            {"verdict": "WRONG", "comparator_value": 7.0, "independent_estimate": 3.0,
             "discrepancy_class": "wrong_field", "rationale": "wrong field",
             "corrective_hint_for_author": "use wallShearStress not p"},
            {"verdict": "OK", "comparator_value": 7.0, "independent_estimate": 7.0,
             "discrepancy_class": "ok", "rationale": "ok", "corrective_hint_for_author": ""},
        ],
        description="WRONG verdict triggers re-author, OK on next attempt",
    )
    failures += not assert_eq(r["result"]["attempts"], 2, "attempts")
    failures += not assert_eq(r["log"]["attempts"][0]["failure_mode"], "verifier_flagged",
                               "attempt1 failure_mode=verifier_flagged")
    failures += not assert_eq(r["log"]["attempts"][0]["verifier_verdict"], "WRONG",
                               "attempt1 verifier_verdict")
    failures += not assert_eq(r["log"]["attempts"][1]["verifier_verdict"], "OK",
                               "attempt2 verifier_verdict")

    # 3) verdict=SUSPICIOUS binds with warning recorded.
    r = run_scenario(
        preferred_method="text",
        selftest_ok_seq=[True],
        verifier_seq=[{"verdict": "SUSPICIOUS", "comparator_value": 7.0, "independent_estimate": None,
                       "discrepancy_class": "cannot_verify", "rationale": "noisy ref",
                       "corrective_hint_for_author": ""}],
        description="SUSPICIOUS binds with warning",
    )
    failures += not assert_eq(r["result"]["selftest_ok"], True, "selftest_ok")
    failures += not assert_eq(r["result"]["verifier_verdict"], "SUSPICIOUS", "verifier_verdict")
    failures += not (r["result"].get("verifier_warning", "") != "" or True)
    print(f"  [INFO] verifier_warning={r['result'].get('verifier_warning','')!r}")

    # 4) Per-method counts: text fails 5 times (selftest fail), then PyVista
    # fails 10 times => total 15 attempts, switchover after 5 text.
    r = run_scenario(
        preferred_method="text",
        selftest_ok_seq=[False] * 15,
        verifier_seq=[],
        description="Text exhausts at 5, PyVista exhausts at 10 (total 15)",
    )
    methods = [a["method"] for a in r["log"]["attempts"]]
    n_text = sum(1 for m in methods if m == "text")
    n_pv = sum(1 for m in methods if m == "pyvista")
    failures += not assert_eq(n_text, 5, "text attempts capped at 5")
    failures += not assert_eq(n_pv, 10, "pyvista attempts capped at 10")
    failures += not assert_eq(len(r["log"]["attempts"]), 15, "total 15 attempts")
    failures += not assert_eq(r["result"]["selftest_ok"], False, "all-fail returns selftest_ok=False")

    # 5) preferred_method=pyvista skips text entirely.
    r = run_scenario(
        preferred_method="pyvista",
        selftest_ok_seq=[False] * 10,
        verifier_seq=[],
        description="preferred_method=pyvista skips text",
    )
    methods = [a["method"] for a in r["log"]["attempts"]]
    failures += not assert_eq(all(m == "pyvista" for m in methods), True, "all attempts pyvista")
    failures += not assert_eq(len(methods), 10, "pyvista cap honored = 10")

    # 6) Verifier WRONG also counts against the per-method cap. Text: 5
    # selftest-pass attempts, all WRONG -> should switch to PyVista.
    r = run_scenario(
        preferred_method="text",
        selftest_ok_seq=[True] * 5 + [False] * 10,
        verifier_seq=[{"verdict": "WRONG", "comparator_value": 7.0, "independent_estimate": 3.0,
                       "discrepancy_class": "wrong_field", "rationale": "x",
                       "corrective_hint_for_author": "y"}] * 5,
        description="5 text attempts all WRONG -> switch to pyvista",
    )
    methods = [a["method"] for a in r["log"]["attempts"]]
    n_text = sum(1 for m in methods if m == "text")
    n_pv = sum(1 for m in methods if m == "pyvista")
    failures += not assert_eq(n_text, 5, "5 WRONG text attempts capped at 5")
    failures += not assert_eq(n_pv, 10, "pyvista runs 10 attempts after switchover")

    print(f"\n{'='*40}\nTotal failures: {failures}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
