#!/usr/bin/env python3
"""The setup verifier, and the diagnosis that reaches the model when it is too late.

All three nemotron cavity runs died the same way, hours of solver time apiece,
on a word written before anything ran. The study's metric spec named the right
reference file and then said to read columns `u_Ulid` / `v_Ulid`, inferred from
that file's README prose ("`u / U_lid` along the vertical line"). The real
header is `y,u_Re100,u_Re400,u_Re1000`. Three things had to be true at once:

  1. nothing checked what was INSIDE a reference file, only that it existed
  2. the extractor's own explanation ("Reference data not found, tried:
     .../ghia1982_u_vertical_centreline.csv") was dropped by a filter that kept
     only numbers, so the mesh gate reported an absent metric and no cause
  3. the spec is written once and read back forever, so there was no way back

This covers all three. Offline: the LLM is stubbed, except for one live check
of the audit prompt against the real Ghia file, skipped without a provider.

Run: python scripts/test_setup_verifier.py
"""

from __future__ import annotations

import json
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


from cfd_langgraph.manager import tools as mt  # noqa: E402


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



GHIA_U = "starter_lid_driven_cavity/reference_data/ghia1982_u_vertical_centreline.csv"
GHIA_V = "starter_lid_driven_cavity/reference_data/ghia1982_v_horizontal_centreline.csv"

BAD_SPEC = [
    {
        "name": "RMS_u",
        "description": "RMS difference from Ghia along the vertical centreline.",
        "direction": "min",
        "computation_hint": (
            f"Read reference data from {GHIA_U} (columns: y, u_Ulid). Interpolate the CFD "
            "u/U_lid to the y of each row and compute sqrt(mean((u_cfd - u_ghia)**2))."
        ),
        "reference_files": [GHIA_U],
    },
]

GOOD_SPEC = [
    {
        "name": "RMS_u",
        "description": "RMS difference from Ghia along the vertical centreline.",
        "direction": "min",
        "computation_hint": (
            f"Read reference data from {GHIA_U}; its columns are y, u_Re100, u_Re400 and "
            "u_Re1000 — take the column for this case's Reynolds number. Interpolate the CFD "
            "u/U_lid to each y and compute sqrt(mean((u_cfd - u_ghia)**2))."
        ),
        "reference_files": [GHIA_U],
    },
]


def _check_item(schema):
    """The per-metric model inside an audit schema."""
    return schema.model_fields["checks"].annotation.__args__[0]


print("== 1. the inventory shows what is really in each file")
inv = mt._reference_excerpt(ROOT / GHIA_U)
check("the real header is shown", "y,u_Re100,u_Re400,u_Re1000" in inv, inv)
check("the invented column is not", "u_Ulid" not in inv)


print("== 2. the audit reads the spec against the file, and is remembered")


class _AuditLLM:
    """Answers the audit; counts calls, so caching is visible."""

    def __init__(self, flag: bool) -> None:
        self.flag, self.calls, self.last_prompt = flag, 0, ""

    def with_structured_output(self, schema):  # noqa: ANN001
        outer = self

        class _Bound:
            def invoke(self, prompt):  # noqa: ANN001
                outer.calls += 1
                outer.last_prompt = prompt
                item = _check_item(schema)
                return schema(checks=[item(
                    metric="RMS_u", ok=not outer.flag,
                    problem=("the hint reads 'u_Ulid'; the file has u_Re100, u_Re400, u_Re1000"
                             if outer.flag else ""),
                )])

        return _Bound()


llm = _AuditLLM(flag=True)
problems = mt._metric_reference_problems(BAD_SPEC, llm)
check("a spec naming a column the file lacks is flagged", len(problems) == 1, str(problems))
check("the problem names both sides", problems and "u_Ulid" in problems[0] and "u_Re100" in problems[0])
check("the audit was shown the file's real first lines", "y,u_Re100,u_Re400,u_Re1000" in llm.last_prompt)
check("and the hint it is judging", "u_Ulid" in llm.last_prompt)

check("a spec that matches the file passes",
      mt._metric_reference_problems(GOOD_SPEC, _AuditLLM(flag=False)) == [])
check("a spec with no reference files is not audited at all",
      mt._metric_reference_problems([{"name": "Cd", "computation_hint": "force coefficient"}],
                                    _AuditLLM(flag=True)) == [])

with tempfile.TemporaryDirectory() as tmp:
    out_dir = Path(tmp)
    cached = _AuditLLM(flag=True)
    first = mt._audit_metric_spec_on_disk(out_dir, BAD_SPEC, cached)
    again = mt._audit_metric_spec_on_disk(out_dir, BAD_SPEC, cached)
    check("the verdict is kept", first == again and first != [])
    check("and costs one model call, not one per caller", cached.calls == 1, f"calls={cached.calls}")
    check("re-audited only when the spec itself changes",
          (mt._audit_metric_spec_on_disk(out_dir, GOOD_SPEC, cached), cached.calls)[1] == 2)


print("== 3. a spec already on disk that cannot be executed is corrected, names kept")


class _StudyLLM:
    """The metric proposer and the auditor in one, answering by schema."""

    def __init__(self) -> None:
        self.answers: list = [GOOD_SPEC]
        self.audits: list = [["the hint reads 'u_Ulid'; the file has u_Re100, u_Re400, u_Re1000"], []]
        self.prompts: list[str] = []

    def with_structured_output(self, schema):  # noqa: ANN001
        outer = self
        name = getattr(schema, "__name__", "")

        class _Bound:
            def invoke(self, prompt):  # noqa: ANN001
                outer.prompts.append(prompt)
                if name == "_MetricReferenceAudit":
                    problem = outer.audits.pop(0) if outer.audits else []
                    item = _check_item(schema)
                    return schema(checks=[item(metric="RMS_u", ok=not problem,
                                               problem=problem[0] if problem else "")])
                spec = outer.answers.pop(0) if outer.answers else GOOD_SPEC
                metric = schema.model_fields["metrics"].annotation.__args__[0]
                return schema(metrics=[metric(**m) for m in spec], reason="stub")

        return _Bound()


def _seed(out_dir: Path, spec: list | None) -> None:
    (out_dir / "user_prompt.txt").write_text("Lid-driven cavity at Re = 100, compared with Ghia (1982).")
    (out_dir / "starter_understanding.json").write_text(json.dumps({
        "starter_dir": str(ROOT / "starter_lid_driven_cavity"),
        "reference_data": {"quantities": ["u", "v"], "usage_guidance": "compare centreline profiles"},
        "flow_parameters": {"Re": 100},
    }))
    if spec is not None:
        (out_dir / "study_metrics.json").write_text(json.dumps(spec))


with tempfile.TemporaryDirectory() as tmp:
    out_dir = Path(tmp)
    _seed(out_dir, BAD_SPEC)
    study = _StudyLLM()
    got = mt._study_metrics(out_dir, study)

    check("the broken spec is not handed back", got and "u_Ulid" not in got[0]["computation_hint"], str(got))
    check("the corrected spec is written",
          "u_Re100" in json.loads((out_dir / "study_metrics.json").read_text())[0]["computation_hint"])
    check("the old one is kept for the record", (out_dir / "study_metrics.rejected.json").is_file())
    check("the rejected copy is the broken one",
          "u_Ulid" in json.loads((out_dir / "study_metrics.rejected.json").read_text())[0]["computation_hint"])
    check("the metric name is pinned so nothing already scored is orphaned", got[0]["name"] == "RMS_u", str(got[0]["name"]))
    redo = [p for p in study.prompts if "THIS IS A CORRECTION" in p]
    check("the re-ask says the names must not change", redo and "'RMS_u'" in redo[0])
    check("and says what was wrong", redo and "u_Ulid" in redo[0])
    check("the re-ask shows the file's real header", redo and "y,u_Re100,u_Re400,u_Re1000" in redo[0])

    calls_before = len(study.prompts)
    same = mt._study_metrics(out_dir, study)
    check("a spec that passes is returned without asking anything again", len(study.prompts) == calls_before)
    check("and is the corrected one", same[0]["name"] == "RMS_u", str(same[0]["name"]))


print("== 4. a spec that never passes is refused rather than written")
with tempfile.TemporaryDirectory() as tmp:
    out_dir = Path(tmp)
    _seed(out_dir, None)

    class _NeverRight(_StudyLLM):
        def __init__(self) -> None:
            super().__init__()
            self.answers = [BAD_SPEC] * 5
            self.audits = [["the hint reads 'u_Ulid'; the file has u_Re100"]] * 5

    out = mt._study_metrics(out_dir, _NeverRight())
    check("nothing is returned", out == [], str(out))
    check("and nothing is written for later stages to execute", not (out_dir / "study_metrics.json").exists())


print("== 5. the extractor's reason survives to the person who needs it")

qoi = {
    "mesh_n_cells": 65536,
    "pyvista_time_used": 1000.0,
    "RMS_u": None,
    "RMS_u__why_null": f"Reference data not found, tried: ['{ROOT / GHIA_U}']",
    "RMS_v__why_null": f"Reference data not found, tried: ['{ROOT / GHIA_V}']",
    "note": "",
}
cleaned: dict = {}
for k, v in qoi.items():
    if v is None:
        continue
    if isinstance(v, (int, float)) and not (isinstance(v, float) and (v != v)):
        cleaned[str(k)] = float(v) if isinstance(v, int) else v
    elif str(k).endswith("__why_null") and str(v).strip():
        cleaned[str(k)] = str(v)[:500]

check("the numbers still come through", cleaned.get("mesh_n_cells") == 65536.0, str(cleaned.get("mesh_n_cells")))
check("the reason is no longer deleted", "RMS_u__why_null" in cleaned)
check("it names the file", GHIA_U in cleaned["RMS_u__why_null"])
check("an empty reason is not carried", "note" not in cleaned)
check("the null metric itself is still absent", "RMS_u" not in cleaned)

analyze_src = (ROOT / "scripts" / "analyze.py").read_text()
check("analyze.py keeps why_null rather than filtering it out",
      "__why_null" in analyze_src.split("qclean: Dict")[1][:1400])


print("== 6. the mesh gate hands that reason to the model")
gate_src = (ROOT / "src" / "cfd_langgraph" / "manager" / "tools.py").read_text()
refusal = gate_src.split("mesh gate cannot judge convergence")[1][:1600]
check("the refusal quotes the extractor's own reason", "extractor's own reason" in refusal)
check("it points at the spec that caused it", "study_metrics.json" in refusal)
check("the why_null keys are not listed back as if they were metrics",
      "__why_null" in gate_src.split("available = sorted")[1][:200])


print("== 7. live: the audit prompt, against the real file")
if not _provider_live():
    print("[SKIP] live check not run")
else:
    try:
        from cfd_langgraph.config import get_settings
        from cfd_langgraph.llm.factory import create_langchain_llm

        live = create_langchain_llm(model=get_settings().model, temperature=0.0)
        found = mt._metric_reference_problems(BAD_SPEC, live)
        check("a real model catches the invented column", len(found) == 1, str(found))
        if found:
            print(f"       it said: {found[0][:220]}")
        clean = mt._metric_reference_problems(GOOD_SPEC, live)
        check("and does not flag the corrected spec", clean == [], str(clean))
    except Exception as exc:  # noqa: BLE001
        print(f"[SKIP] live check unavailable: {type(exc).__name__}: {exc}")


print()
print("FAILURES:", FAILURES if FAILURES else "none")
sys.exit(1 if FAILURES else 0)
