#!/usr/bin/env python3
"""The study's metric decision (`manager/tools.py::_study_metrics`).

One decision, made once from the user's prompt and the starter folder, carried
with its full computation rule to every measurement in the study — mesh
independence, candidate scoring, final comparison.

The reason it is one decision and not one per call site: measured twice on the
same case within an hour, the mesh gate's extractor produced Cf = -2.755e-04
and then -3.939e-07, the second having used Ub = 1.0 instead of the case's
0.028 — a factor of 1276, and exactly the error the study's own scoring
contract warns about. A metric re-derived per call site is a different metric.

Offline: the LLM is stubbed. The live decision is exercised separately.

Run: python scripts/test_study_metrics.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

FAILURES: list[str] = []


def check(name: str, cond: object, detail: str = "") -> None:
    if cond:
        print(f"[PASS] {name}")
    else:
        FAILURES.append(name)
        print(f"[FAIL] {name}" + (f" — {detail}" if detail else ""))


class _StubLLM:
    """Returns a fixed multi-metric decision and counts how often it is asked."""

    def __init__(self, metrics):
        self.metrics = metrics
        self.calls = 0

    def with_structured_output(self, schema):
        outer = self

        class _R:
            def invoke(self, _prompt):
                outer.calls += 1
                outer.last_prompt = _prompt
                return schema(metrics=outer.metrics, reason="stub")

        return _R()


def _study(tmp: Path) -> Path:
    (tmp / "user_prompt.txt").write_text(
        "Reduce Cf RMSE against DNS for periodic-hill flow. Use only Cf.", encoding="utf-8"
    )
    (tmp / "starter_understanding.json").write_text(json.dumps({
        "flow_parameters": {"Re": 5600, "Ub": 0.028, "nu": 5e-06},
        "reference_data": {"quantities": ["Cf", "x/h"], "usage_guidance": "RMSE vs cf_dns_exactmatch"},
    }), encoding="utf-8")
    return tmp


def test_decided_once_and_reused() -> None:
    from cfd_langgraph.manager.tools import _StudyMetric, _study_metrics

    with tempfile.TemporaryDirectory() as raw:
        tmp = _study(Path(raw))
        llm = _StubLLM([_StudyMetric(name="cf_rmse", computation_hint="Cf=-2*tau_x/Ub^2, Ub=0.028 from transportProperties")])

        first = _study_metrics(tmp, llm)
        check("a metric is decided from the prompt and starter files", [m["name"] for m in first] == ["cf_rmse"])
        check("it is persisted", (tmp / "study_metrics.json").is_file())
        check("the computation rule travels with it", "Ub=0.028" in first[0]["computation_hint"])

        second = _study_metrics(tmp, llm)
        check("a later call reuses the decision", second == first)
        check("and does not ask the model again", llm.calls == 1, detail=f"{llm.calls} calls")


def test_multi_metric_is_carried_end_to_end() -> None:
    """Several metrics must survive as several — not collapse to the first."""
    from cfd_langgraph.manager.tools import _StudyMetric, _study_metrics

    with tempfile.TemporaryDirectory() as raw:
        tmp = _study(Path(raw))
        llm = _StubLLM([
            _StudyMetric(name="cf_rmse", direction="min", computation_hint="rule A, Ub=0.028"),
            _StudyMetric(name="reattachment_error", direction="min", computation_hint="rule B"),
            _StudyMetric(name="lift_to_drag", direction="max", computation_hint="rule C"),
        ])
        specs = _study_metrics(tmp, llm)
        check("every metric is kept", [m["name"] for m in specs]
              == ["cf_rmse", "reattachment_error", "lift_to_drag"], detail=str([m["name"] for m in specs]))
        check("per-metric direction is preserved",
              [m["direction"] for m in specs] == ["min", "min", "max"])
        check("each keeps its own computation rule",
              [m["computation_hint"][:6] for m in specs] == ["rule A", "rule B", "rule C"])

        on_disk = json.loads((tmp / "study_metrics.json").read_text())
        check("all of them are persisted", len(on_disk) == 3)


def test_extractor_prompt_carries_every_rule() -> None:
    """The generated PyVista script must be told each metric's definition, so
    no constant is ever re-derived."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import inspect

    import analyze

    source = inspect.getsource(analyze._llm_pyvista_batch_qoi)
    check("the hint block is built from every hint", "for h in metric_hints" in source)
    check("and injected into the prompt", "hint_block" in source)
    check("with an instruction not to re-derive constants",
          "do not re-derive" in source and "do not substitute a default for any constant" in source)
    check("reference/dictionary files named by the rule are allowed",
          "You MAY read files the metric definition below explicitly names" in inspect.getsource(analyze))


def test_no_metric_is_a_refusal_not_a_default() -> None:
    from cfd_langgraph.manager.tools import _study_metrics

    class _Broken:
        def with_structured_output(self, _s):
            raise RuntimeError("provider down")

    with tempfile.TemporaryDirectory() as raw:
        tmp = _study(Path(raw))
        check("a failed decision yields nothing, not a guess", _study_metrics(tmp, _Broken()) == [])
        check("and writes no spec file", not (tmp / "study_metrics.json").is_file())


def test_oed_reuses_the_decision_instead_of_re_deriving() -> None:
    """One decision, used throughout — including by the search.

    `oed_setup_search` had its own metric derivation from before the
    study-wide decision existed, so it answered the same question a second
    time. That is the pattern that produced Cf with Ub = 1.0 instead of the
    case's 0.028 when one derivation had less context than the other.
    """
    import inspect
    from pathlib import Path as _P

    source = _P("scripts/open_ended_discovery.py").read_text(encoding="utf-8")
    check("the search looks for the study's decision", 'run_dir / "study_metrics.json"' in source)
    check(
        "it is consulted BEFORE the local proposer",
        source.index('_study_specs_path = run_dir / "study_metrics.json"')
        < source.index("_oedx.propose_metric_set("),
    )
    check(
        "the local proposer is the fallback, not the default",
        "if not specs:" in source
        and source.index("if not specs:") < source.index("_oedx.propose_metric_set("),
    )
    check("the computation rule is carried across", '_m.setdefault("computation_hint", "")' in source)
    check("and the reuse is announced, not silent", "reusing the study's metric decision" in source)


def test_comparator_search_reaches_the_reference_directory() -> None:
    """The study's own working comparator must be found, not re-authored.

    `starter_dir` is the CASE directory; the authoritative comparator lives
    beside it in the reference-data directory. Searching only the case
    directory never sees it, so the study authored a fresh one — five
    attempts, and the result returned nan because it treated reference
    stations just outside the mesh range as fatal, where the existing script
    handles them.
    """
    from pathlib import Path as _P

    source = _P("scripts/open_ended_discovery.py").read_text(encoding="utf-8")
    block = source[source.index("# Resolve comparators for each metric"):][:2000]
    check("the starter's parent is searched", "starter_dir.resolve().parent" in block)
    check("every reference file's directory is searched", "for _rf in ref_files" in block)
    check("roots are de-duplicated", "not in search_roots" in block)

    # And prove discovery actually finds it with those roots.
    sys.path.insert(0, str(_P(__file__).resolve().parents[1] / "scripts"))
    starter = _P("starter_oed_turbulence/periodic_hill_sa")
    reference = _P("starter_oed_turbulence/reference_data/reference_exactmatch_cf.csv")
    if not starter.is_dir() or not reference.is_file():
        print("[SKIP] live discovery — starter files not present here")
        return
    from oed_extensions import discover_existing_comparators

    case_only = discover_existing_comparators(
        search_roots=[starter.resolve()], metrics=[{"name": "cf_rmse", "ref_column": "cf_dns_exactmatch"}]
    )
    check("searching only the case directory misses it", "cf_rmse" not in case_only, detail=str(case_only))

    with_refs = discover_existing_comparators(
        search_roots=[starter.resolve(), starter.resolve().parent, reference.parent.resolve()],
        metrics=[{"name": "cf_rmse", "ref_column": "cf_dns_exactmatch"}],
    )
    check(
        "adding the reference directory finds the authoritative script",
        with_refs.get("cf_rmse", "").endswith("compare_exactmatch_cf.py"),
        detail=str(with_refs),
    )



# --- reference-file validation -------------------------------------------
# Regression for run oed_20260823_opus_low: the metric spec named
# "reference_data/cf_dns_exactmatch" — a CSV *column* written as if it were a
# file. Nothing checked it, so the extractor could not compute the metric at
# all and the mesh gate failed after 506s of solver time having produced only
# generic field statistics.

GOOD_REF = "starter_oed_turbulence/reference_data/reference_exactmatch_cf.csv"
BAD_REF = "reference_data/cf_dns_exactmatch"


class _PathLLM:
    """Yields one reference path per call, so a correction can be observed."""

    def __init__(self, paths):
        self.paths = list(paths)
        self.calls = 0
        self.prompts: list[str] = []

    def with_structured_output(self, _schema):
        return self

    def invoke(self, prompt):
        self.prompts.append(prompt)
        self.calls += 1
        path = self.paths[min(self.calls - 1, len(self.paths) - 1)]

        class _M:
            @staticmethod
            def model_dump():
                return {
                    "name": "cf_rmse", "description": "", "direction": "min",
                    "computation_hint": "...",
                    "reference_files": ([path] if path else []),
                }
            name = "cf_rmse"

        class _R:
            metrics = [_M()]
            reason = "test"

        return _R()


def _decide_with_paths(paths):
    from cfd_langgraph.manager import tools as mtools
    tmp = Path(tempfile.mkdtemp())
    (tmp / "user_prompt.txt").write_text("minimise Cf RMSE", encoding="utf-8")
    (tmp / "starter_understanding.json").write_text(
        json.dumps({"starter_dir": "starter_oed_turbulence"}), encoding="utf-8"
    )
    llm = _PathLLM(paths)
    return mtools._study_metrics(tmp, llm), llm, tmp


def test_reference_files_are_verified_on_disk() -> None:
    out, llm, _ = _decide_with_paths([GOOD_REF])
    check("a valid reference path is accepted first time", llm.calls == 1 and bool(out))
    check("the file inventory is handed to the model", GOOD_REF in llm.prompts[0])

    out, llm, _ = _decide_with_paths([BAD_REF, GOOD_REF])
    check(
        "a nonexistent reference path is caught and corrected",
        llm.calls == 2 and out and out[0]["reference_files"] == [GOOD_REF],
        detail=f"calls={llm.calls} out={out}",
    )
    check("the correction names the offending path", BAD_REF in llm.prompts[1])

    out, llm, tmp = _decide_with_paths([BAD_REF, BAD_REF, BAD_REF])
    check(
        "no spec is written when the path never resolves",
        out == [] and not (tmp / "study_metrics.json").is_file(),
    )
    check("retries are bounded", llm.calls == 3, detail=f"calls={llm.calls}")

    out, llm, _ = _decide_with_paths([None])
    check(
        "a metric needing no reference data is unaffected",
        llm.calls == 1 and out and out[0]["reference_files"] == [],
    )


def test_inventory_stays_inside_the_starter_tree() -> None:
    from cfd_langgraph.manager import tools as mtools
    tmp = Path(tempfile.mkdtemp())
    (tmp / "starter_understanding.json").write_text(
        json.dumps({"starter_dir": "starter_oed_turbulence"}), encoding="utf-8"
    )
    inv = mtools._reference_file_inventory(tmp)
    check(
        "the inventory is the starter's data files, not the whole repository",
        0 < len(inv) < 50,
        detail=f"got {len(inv)} files",
    )
    check(
        "and it contains the real reference CSV",
        any(f.endswith("reference_data/reference_exactmatch_cf.csv") for f in inv),
    )


# --- literature search query ------------------------------------------------
# Regression for run closure_20260824_codex. The study objective was passed to
# the paper search verbatim — file paths, metric variable names, case counts,
# percentage targets and all — and Semantic Scholar returned TWO papers, one of
# them about aircraft intake S-ducts. Ideation had nothing to ground on, invented
# over-broad multi-mechanism studies, and the critic rejected all six for being
# under-specified. That repeated for three rounds at ~26 minutes each.

class _QueryLLM:
    """Stands in for the distiller, so this suite needs no model call."""

    def __init__(self, query: str, broader: str = ""):
        self.query = query
        self.broader = broader
        self.prompts: list[str] = []

    def with_structured_output(self, _schema):
        return self

    def invoke(self, prompt):
        self.prompts.append(prompt)
        outer = self

        class _R:
            query = outer.query
            broader_query = outer.broader

        return _R()


def test_the_search_query_is_the_research_question() -> None:
    objective = (
        "Open-ended discovery under a fixed prescribed mesh: find a compiled "
        "OpenFOAM 10 modification of k-omega SST that lowers the equally weighted "
        "mean velocity_mae by at least 15% versus unmodified SST across all 32 "
        "cases in starter_closure_challenge (27 training plus 5 validation)"
    )
    llm = _QueryLLM("data-driven RANS closure for separated flow and duct secondary flow",
                    "RANS turbulence modeling of separation")
    from cfd_langgraph.manager import tools as mtools
    query, broader = mtools._literature_query(objective, llm)
    check(
        "the objective is distilled, not passed through",
        query != objective and "starter_closure_challenge" not in query,
        detail=query,
    )
    check("a broader fallback is produced too", bool(broader), detail=repr(broader))
    check(
        "the distiller is told to drop run configuration",
        any("file paths" in p for p in llm.prompts),
    )


def test_a_failing_distiller_still_searches() -> None:
    """A bad search beats no search: if the distiller raises, the objective is
    used as written rather than the study stalling."""

    class _Broken:
        def with_structured_output(self, _schema):
            return self

        def invoke(self, _prompt):
            raise RuntimeError("model unavailable")

    from cfd_langgraph.manager import tools as mtools
    query, broader = mtools._literature_query("some objective text", _Broken())
    check("a distiller failure falls back to the objective", query == "some objective text")
    check("and offers no fallback query", broader == "")


def test_empty_topic_is_not_searched() -> None:
    from cfd_langgraph.manager import tools as mtools
    query, broader = mtools._literature_query("   ", _QueryLLM("x"))
    check("an empty topic distils to nothing", query == "" and broader == "")


def test_paper_count_has_a_floor() -> None:
    """Two papers is not a literature review. The floor is what stops a study
    grounding ideation on almost nothing."""
    from cfd_langgraph.manager import tools as mtools
    check(
        "a minimum paper count is defined",
        getattr(mtools, "_LIT_MIN_PAPERS", 0) >= 5,
        detail=str(getattr(mtools, "_LIT_MIN_PAPERS", None)),
    )



def main() -> int:
    for test in (
        test_decided_once_and_reused,
        test_multi_metric_is_carried_end_to_end,
        test_extractor_prompt_carries_every_rule,
        test_no_metric_is_a_refusal_not_a_default,
        test_oed_reuses_the_decision_instead_of_re_deriving,
        test_comparator_search_reaches_the_reference_directory,
        test_reference_files_are_verified_on_disk,
        test_inventory_stays_inside_the_starter_tree,
        test_the_search_query_is_the_research_question,
        test_a_failing_distiller_still_searches,
        test_empty_topic_is_not_searched,
        test_paper_count_has_a_floor,
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
