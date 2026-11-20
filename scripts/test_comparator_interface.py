#!/usr/bin/env python3
"""Running a starter's own scorer when it does not speak our contract.

Everything that reads a comparator assumed one interface: `--case <dir>
--reference <file>`, answering `METRIC <name>: <number>` on stdout. A
comparator this pipeline authors is told to do that. A scorer the starter
shipped has never heard of it.

On the three nemotron cavity runs the fallback correctly identified
`starter_lid_driven_cavity/scripts/evaluate.py` — the one script that computes,
by definition, the metric the mesh gate was missing — and could not use it. It
takes `--case RE=DIR`, has no `--reference` (it loads Ghia itself), and prints
`rms_u=0.00123` with a JSON summary beside it. The invocation died in argparse,
and there was no METRIC line either way. The component whose whole job is to
rescue a failed extractor sat the failure out.

Offline: a synthetic scorer with the same awkward shape. Live: the real
evaluate.py, described but not run (running it needs a solved case).

Run: python scripts/test_comparator_interface.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

FAILURES: list[str] = []


def check(name: str, cond: object, detail: str = "") -> None:
    if cond:
        print(f"[PASS] {name}")
    else:
        FAILURES.append(name)
        print(f"[FAIL] {name}" + (f" — {detail}" if detail else ""))


import oed_extensions as oedx  # noqa: E402


def _provider_live() -> bool:
    """Whether a model can actually be called, not merely configured.

    A configured provider with a dead key or an expired token would otherwise
    turn these into failures, which says nothing about the code under test.
    """
    if not (os.environ.get("CFD_SCIENTIST_LLM_PROVIDER") or os.environ.get("OPENAI_API_KEY")):
        return False
    try:
        from cfd_langgraph.config import get_settings
        from cfd_langgraph.llm.factory import create_langchain_llm

        create_langchain_llm(model=get_settings().model, temperature=0.0).invoke("reply with: ok")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[SKIP] provider configured but not usable: {type(exc).__name__}: {str(exc)[:120]}")
        return False



# A scorer shaped like the cavity one: RE=DIR cases, no --reference, its own
# names, a JSON summary, and a non-zero exit when the case misses tolerance.
FAKE_SCORER = '''#!/usr/bin/env python3
import argparse, json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--case", action="append", default=[], metavar="RE=DIR")
ap.add_argument("--out", default="evaluation_summary.json")
args = ap.parse_args()

cases = {}
for item in args.case:
    re_text, sep, path = item.partition("=")
    if not sep:
        ap.error("--case takes RE=DIR")
    cases[re_text] = path

summary = {"tolerance_rms": 0.01, "cases": {}}
for re_text, path in cases.items():
    value = float(Path(path).name.split("_")[-1])
    summary["cases"][re_text] = {"rms_u": value, "rms_v": value / 2, "pass": False}
    print(f"Re {re_text}: FAIL  rms_u={value:.5f}  rms_v={value / 2:.5f}")
summary["result"] = "FAIL"
Path(args.out).write_text(json.dumps(summary, indent=2))
print("RESULT: FAIL")
raise SystemExit(1)
'''

INTERFACE = {
    "argv": ["{python}", "{script}", "--case", "100={case}", "--out", "{out}"],
    "metric_keys": {"RMS_u": "rms_u", "RMS_v": "rms_v"},
    "json_out": "{out}",
    "status": "ok",
}


print("== 1. a number is found wherever the scorer nests it")
blob = {"tolerance_rms": 0.01, "cases": {"100": {"rms_u": 0.0042, "pass": False}}, "result": "FAIL"}
check("nested under its case", oedx._numbers_under_key(blob, "rms_u") == [0.0042])
check("a key that is not there gives nothing", oedx._numbers_under_key(blob, "RMS_u") == [])
check("booleans are not numbers", oedx._numbers_under_key(blob, "pass") == [])
check("every case is found, not just the first",
      sorted(oedx._numbers_under_key(
          {"cases": {"100": {"rms_u": 1.0}, "400": {"rms_u": 2.0}}}, "rms_u")) == [1.0, 2.0])
check("lists are walked too", oedx._numbers_under_key([{"rms_u": 3.0}], "rms_u") == [3.0])


print("== 2. the scorer runs its own way, and its number is read")
with tempfile.TemporaryDirectory() as tmp:
    work = Path(tmp)
    scorer = work / "evaluate_like.py"
    scorer.write_text(FAKE_SCORER)
    case = work / "case_0.00429"
    case.mkdir()

    ok, reason, value = oedx.score_with_comparator_interface(
        comparator=scorer, interface=INTERFACE, case_dir=case, reference_file=None,
        metric_name="RMS_u", work_dir=work / "iface",
    )
    check("it scores a case whose CLI we could not have guessed", ok, reason)
    check("and gives the scorer's own number", value == 0.00429, str(value))
    check("a failing verdict is still a measurement", ok and value is not None)

    ok2, reason2, value2 = oedx.score_with_comparator_interface(
        comparator=scorer, interface=INTERFACE, case_dir=case, reference_file=None,
        metric_name="RMS_v", work_dir=work / "iface",
    )
    check("a second metric maps to its own key", ok2 and value2 == 0.00429 / 2, f"{reason2} {value2}")

    bad, reason_bad, none_value = oedx.score_with_comparator_interface(
        comparator=scorer, interface={**INTERFACE, "metric_keys": {}}, case_dir=case,
        reference_file=None, metric_name="RMS_u", work_dir=work / "iface",
    )
    check("a metric the scorer does not report is refused", not bad and none_value is None)
    check("and says why", "no interface" in reason_bad, reason_bad)

    two = work / "case_0.00381"
    two.mkdir()
    mixed = {**INTERFACE,
             "argv": ["{python}", "{script}", "--case", f"100={case}",
                      "--case", f"400={two}", "--out", "{out}"]}
    m_ok, m_reason, m_value = oedx.score_with_comparator_interface(
        comparator=scorer, interface=mixed, case_dir=case, reference_file=None,
        metric_name="RMS_u", work_dir=work / "iface",
    )
    check("different values from several cases are refused, not guessed", not m_ok, str(m_value))
    check("and the reason says so", "more than" in m_reason, m_reason)


print("== 3. stdout is read when the scorer writes no file")
with tempfile.TemporaryDirectory() as tmp:
    work = Path(tmp)
    talker = work / "talker.py"
    talker.write_text(
        "import argparse\n"
        "ap = argparse.ArgumentParser(); ap.add_argument('--case'); ap.parse_args()\n"
        "print('Re 100: FAIL  rms_u=0.00429  rms_v=0.00214')\n"
    )
    stdout_iface = {"argv": ["{python}", "{script}", "--case", "{case}"],
                    "metric_keys": {"RMS_u": "rms_u"}, "json_out": "", "status": "ok"}
    ok, reason, value = oedx.score_with_comparator_interface(
        comparator=talker, interface=stdout_iface, case_dir=work, reference_file=None,
        metric_name="RMS_u", work_dir=work / "iface",
    )
    check("a printed value is read", ok and value == 0.00429, f"{reason} {value}")

    silent = work / "silent.py"
    silent.write_text(
        "import argparse\n"
        "ap = argparse.ArgumentParser(); ap.add_argument('--case'); ap.parse_args()\n"
        "print('done')\n"
    )
    ok_s, reason_s, value_s = oedx.score_with_comparator_interface(
        comparator=silent, interface=stdout_iface, case_dir=work, reference_file=None,
        metric_name="RMS_u", work_dir=work / "iface",
    )
    check("a scorer that reports nothing is refused", not ok_s and value_s is None)
    check("and the command it tried is named", "--case" in reason_s, reason_s)


print("== 4. the mesh gate does not score against reference data")
gate_src = (ROOT / "src" / "cfd_langgraph" / "manager" / "tools.py").read_text()
gate = gate_src.split("def run_mesh_gate(")[1].split("\n    def ")[0]
check("no comparator in the gate", "comparator" not in gate)
check("the gate measures solution-only quantities", "solution_quantities" in gate)


print("== 5. live: describing the real cavity scorer")
evaluate = ROOT / "starter_lid_driven_cavity" / "scripts" / "evaluate.py"
if not evaluate.is_file():
    print("[SKIP] the cavity starter is not present here")
elif not _provider_live():
    print("[SKIP] live check not run")
else:
    from comparator_classifier import describe_comparator_interface

    with tempfile.TemporaryDirectory() as tmp:
        cache = Path(tmp) / "comparator_interface.json"
        iface = describe_comparator_interface(
            comparator=evaluate, metrics=["RMS_u", "RMS_v"], cache_path=cache,
        )
        print(f"       argv: {iface.get('argv')}")
        print(f"       keys: {iface.get('metric_keys')}")
        check("an interface is produced", iface.get("status") == "ok", str(iface.get("status")))
        argv = " ".join(iface.get("argv") or [])
        check("it uses this scorer's RE=DIR shape, not --case DIR", "={case}" in argv, argv)
        check("it does not invent a --reference flag", "--reference" not in argv, argv)
        check("RMS_u is mapped to the name the script really prints",
              (iface.get("metric_keys") or {}).get("RMS_u") == "rms_u",
              str(iface.get("metric_keys")))
        check("RMS_v too", (iface.get("metric_keys") or {}).get("RMS_v") == "rms_v")

        cached = describe_comparator_interface(
            comparator=evaluate, metrics=["RMS_v", "RMS_u"], cache_path=cache,
        )
        check("asking again costs nothing", cached.get("cached") is True)


print()
print("FAILURES:", FAILURES if FAILURES else "none")
sys.exit(1 if FAILURES else 0)
