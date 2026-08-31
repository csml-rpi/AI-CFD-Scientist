#!/usr/bin/env python3
"""Offline regression tests for the deepagents hypothesis/OED integration.

No network, LLM, or OpenFOAM execution is required.
"""
from __future__ import annotations

import json
import operator
import os
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import pathlib
from pathlib import Path
from typing import Annotated, TypedDict

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from cfd_langgraph import hypothesis_pipeline as hp  # noqa: E402
from cfd_langgraph import ideation as ideation_module  # noqa: E402
from cfd_langgraph.config import get_settings  # noqa: E402
from cfd_langgraph.ideation import candidate_similarity, normalize_literature_records  # noqa: E402
from cfd_langgraph.manager import tools as manager_tools_module  # noqa: E402
from cfd_langgraph.manager.tools import (  # noqa: E402
    _extract_target_improvement_pct,
    _improvement_pct,
    _run_succeeded,
    build_manager_tools,
)
from cfd_langgraph.foam_native.loop import _clean_stale_run_artifacts  # noqa: E402
from cfd_langgraph.scheduling.resource_probe import ResourceProfile  # noqa: E402
from cfd_langgraph.scheduling.scheduler import compute_max_concurrency  # noqa: E402
from cfd_langgraph.scheduling.coordinator import CaseCoordinator  # noqa: E402
from cfd_langgraph.scheduling import coordinator as coordinator_module  # noqa: E402
from code_mod_agentic import Sandbox, _find_compiled_artifacts  # noqa: E402
import lit as lit_module  # noqa: E402

FAILURES = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global FAILURES
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES += 1


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def test_literature_schema_normalization() -> None:
    records = normalize_literature_records(
        [
            {"title": "Paper", "abstract": "The real abstract", "doi": "10.1/a", "url": "u"},
            {"title": "Duplicate", "snippet": "ignored", "doi": "10.1/a", "url": "u2"},
        ]
    )
    check("lit.py abstract is preserved as ideation snippet", records[0]["snippet"] == "The real abstract")
    check("literature normalization deduplicates by DOI", len(records) == 1)
    check("persisted S2 records receive explicit provenance", records[0]["source"] == "semantic_scholar")


def test_literature_search_balances_query_variants() -> None:
    topic = "Open-ended discovery: find a novel SA modification for periodic hill flow Reynolds 5600 wall friction DNS"
    queries = lit_module._generate_multi_queries(topic)
    calls = []
    original = lit_module._collect_papers_single_query

    def fake_collect(query, limit, offset_start=0):
        calls.append((query, limit, offset_start))
        return [
            {"title": f"{query}-{offset_start}-{idx}", "doi": f"10.test/{len(calls)}-{idx}"}
            for idx in range(limit)
        ]

    try:
        lit_module._collect_papers_single_query = fake_collect
        records = lit_module.collect_papers_via_requests(topic, 8)
    finally:
        lit_module._collect_papers_single_query = original

    check("literature queries drop generic task-intent words", all("discovery" not in q.lower() and "novel" not in q.lower() and "open-ended" not in q.lower() for q in queries))
    check("literature first pass allocates budget across query variants", len(calls) == len(queries) and all(call[1] == 2 and call[2] == 0 for call in calls))
    check("balanced literature retrieval still respects the global limit", len(records) == 8)


def test_candidate_batch_similarity() -> None:
    a = {"objective": "Reduce periodic-hill skin-friction error", "experiments": [{"name": "SA APG", "parameters": {"c": 1.0}}]}
    same = {"objective": "Reduce periodic-hill skin-friction error", "experiments": [{"name": "SA APG", "parameters": {"c": 1.0}}]}
    other = {"objective": "Predict cavitation inception", "experiments": [{"name": "VOF pressure sweep", "parameters": {"p": 2e5}}]}
    check("duplicate candidate ideas score as duplicates", candidate_similarity(a, same) > 0.99)
    check("materially different candidate ideas remain distinct", candidate_similarity(a, other) < 0.7)


def test_hypothesis_pipeline_uses_supplied_literature_and_gates() -> None:
    captured: dict = {}
    original_batch = hp.run_ideation_batch
    original_critic = hp.HypothesisCritiqueAgent
    original_ranker = hp.HypothesisRankAgent

    def fake_batch(settings, topic, **kwargs):
        captured["literature"] = kwargs.get("literature_items")
        return {
            "literature_used": kwargs["literature_items"],
            "candidates": [
                {"candidate_id": "cand_01", "idea": {"objective": "good"}, "novelty": {"passed": True}, "experiment_count": {"passed": True}},
                {"candidate_id": "cand_02", "idea": {"objective": "duplicate"}, "novelty": {"passed": False}, "experiment_count": {"passed": True}},
                {"candidate_id": "cand_03", "idea": {"objective": "unphysical"}, "novelty": {"passed": True}, "experiment_count": {"passed": True}},
                {"candidate_id": "cand_04", "idea": {"objective": "empty"}, "novelty": {"passed": True}, "experiment_count": {"passed": False}},
            ],
        }

    class FakeCritic:
        def __init__(self, *args, **kwargs):
            pass

        def critique(self, idea, *args, **kwargs):
            return {"verdict": "pass" if idea["objective"] == "good" else "reject"}

    class FakeRanker:
        def __init__(self, *args, **kwargs):
            pass

        def rank(self, candidates, **kwargs):
            for idx, candidate in enumerate(candidates, 1):
                candidate["rank"] = idx
            return candidates

    try:
        hp.run_ideation_batch = fake_batch
        hp.HypothesisCritiqueAgent = FakeCritic
        hp.HypothesisRankAgent = FakeRanker
        literature = [{"title": "Grounding paper", "abstract": "Evidence"}]
        result = hp.run_propose_critique_rank(
            get_settings(), "topic", literature_records=literature, verbose=False
        )
    finally:
        hp.run_ideation_batch = original_batch
        hp.HypothesisCritiqueAgent = original_critic
        hp.HypothesisRankAgent = original_ranker

    check("hypothesis pipeline consumes the persisted literature set", captured.get("literature") is literature)
    check("novelty and physical critique gates both filter candidates", [c["candidate_id"] for c in result["ranked_hypotheses"]] == ["cand_01"])
    check("grounding metadata is explicit", result["literature_grounded"] is True and result["literature_count"] == 1)


def test_novelty_evaluator_fails_closed() -> None:
    class Response:
        content = json.dumps(
            {
                "objective": "test",
                "experiments": [
                    {"name": "case", "parameters": {"Re": 1000}, "controls": {}}
                ],
            }
        )

    class FakeLLM:
        def invoke(self, _messages):
            return Response()

    original = ideation_module.novelty_score_llm
    try:
        def broken_evaluator(*_args, **_kwargs):
            raise ValueError("malformed novelty response")

        ideation_module.novelty_score_llm = broken_evaluator
        settings = get_settings().model_copy(update={"ideation_novelty_max_retries": 0})
        result = ideation_module._generate_one_idea(
            FakeLLM(),
            {
                "initial_idea_prompt": "Return one idea with at most {max_experiments} experiments.",
                "literature_aware_user_prompt": "{research_topic}\n{literature_context}\n{max_experiments}",
            },
            "topic",
            "paper context",
            [{"title": "prior", "abstract": "evidence"}],
            settings,
            verbose=False,
        )
    finally:
        ideation_module.novelty_score_llm = original

    check("malformed novelty evaluation cannot approve a grounded idea", result["novelty"]["passed"] is False)
    check("novelty failure records its fail-closed method", result["novelty"]["method"] == "llm_failed_closed")


def test_score_and_status_helpers() -> None:
    check("native status-only success is normalized", _run_succeeded({"status": "success"}))
    check("explicit failed success flag is not overridden", not _run_succeeded({"status": "failed", "success": False}))
    check("topic improvement target is extracted", _extract_target_improvement_pct("targeting >=30% improvement") == 30.0)
    check("beat-baseline phrasing extracts its percentage target", _extract_target_improvement_pct("beats the baseline Cf error by 25%") == 25.0)
    check("min-direction improvement is positive when error drops", abs(_improvement_pct(0.7, 1.0, "min") - 30.0) < 1e-9)
    check("max-direction improvement is positive when score rises", abs(_improvement_pct(1.3, 1.0, "max") - 30.0) < 1e-9)
    profile = ResourceProfile(wall_clock_s=1.0, peak_used_mem_mb=100.0, avg_cpu_percent=50.0, logical_cores=8)
    check("system CPU percent converts to core equivalents", profile.cores_used == 4.0)

    # A calibration case that dies in seconds (a bad dict, a mesh error — the
    # most common first-case outcome) records zero samples. Read literally it
    # looks like a case needing ~0 CPU and ~0 memory, and the floors then turn
    # it into an enormous concurrency limit that persists for the whole group.
    unmeasured = ResourceProfile(wall_clock_s=1.2, peak_used_mem_mb=0.0, avg_cpu_percent=0.0, sample_count=0)
    real = ResourceProfile(wall_clock_s=900.0, peak_used_mem_mb=4000.0, avg_cpu_percent=100.0)
    check("a calibration with no samples is not treated as measured", unmeasured.measured is False)
    check(
        "an unmeasured calibration yields a conservative limit, not a huge one",
        1 <= compute_max_concurrency(unmeasured) <= 8,
        detail=f"got {compute_max_concurrency(unmeasured)}",
    )
    check(
        "concurrency never exceeds the machine's logical core count",
        compute_max_concurrency(real) <= (psutil.cpu_count(logical=True) or 1),
    )


def test_study_wide_concurrency_and_exclusive_calibration() -> None:
    active = 0
    peak = 0
    lock = threading.Lock()

    def work():
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.04)
        with lock:
            active -= 1
        return True

    forced = CaseCoordinator(forced_max_concurrency=2)
    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(lambda idx: forced.run_case(f"group_{idx % 3}", work), range(6)))
    check("forced concurrency is enforced study-wide across physics groups", peak <= 2)

    active = 0
    peak = 0
    original_benchmark = coordinator_module.benchmark_case

    def fake_benchmark(fn):
        return fn(), ResourceProfile(0.04, 256.0, 25.0, 4)

    try:
        coordinator_module.benchmark_case = fake_benchmark
        automatic = CaseCoordinator()
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(lambda group: automatic.run_case(group, work), ["a", "b"]))
    finally:
        coordinator_module.benchmark_case = original_benchmark
    check("first-case calibrations for different groups run exclusively", peak == 1)


def test_manager_write_scope_and_protected_artifacts() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        out = root / "study"
        settings = get_settings().model_copy(update={"knowledge_bundle_dir": root / "knowledge"})
        built = build_manager_tools(settings, out)
        write_tool = next(t for t in built["manager_tools"] if t.__name__ == "write_text_file")

        outside = root / "outside.txt"
        outside_result = write_tool(str(outside), "must not escape")
        protected = out / "audit_passed.json"
        protected_result = write_tool(str(protected), "must not forge")
        mesh_spec = out / "selected_mesh_spec.json"
        mesh_spec_result = write_tool(str(mesh_spec), "must not forge")
        inside = out / "notes" / "note.txt"
        inside_result = write_tool(str(inside), "allowed")

        check("manager write tool cannot escape the study directory", "error" in outside_result and not outside.exists())
        check("manager write tool cannot forge protected artifacts", "error" in protected_result and not protected.exists())
        check("manager write tool cannot forge a mesh-gate selection", "error" in mesh_spec_result and not mesh_spec.exists())
        check("manager write tool can create ordinary study artifacts", inside_result.get("path") == str(inside) and inside.read_text(encoding="utf-8") == "allowed")

        fake_baseline = out / "fake_baseline"
        fake_baseline.mkdir(parents=True)
        setup_tool = next(t for t in built["manager_tools"] if t.__name__ == "oed_setup_search")
        setup_result = setup_tool("topic", str(fake_baseline), 2)
        check("OED setup rejects a baseline not selected by this study's mesh gate", setup_result.get("ok") is False and "selected_level" in setup_result.get("error", ""))


def test_code_mod_filesystem_isolation_and_case_local_artifacts() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        run_dir = root / "run"
        starter = root / "starter"
        run_dir.mkdir()
        starter.mkdir()
        outside = root / "outside.txt"
        outside.write_text("original", encoding="utf-8")
        sandbox = Sandbox(run_dir=run_dir, starter_case=starter, wm_project_dir=None)

        denied = sandbox.run_bash(f"printf changed > {outside}", cwd=str(run_dir))
        allowed_path = run_dir / "inside.txt"
        allowed = sandbox.run_bash(f"printf allowed > {allowed_path}", cwd=str(run_dir))
        check("code-mod shell cannot write outside its candidate run directory", denied.get("rc") != 0 and outside.read_text(encoding="utf-8") == "original")
        check("code-mod shell remains writable inside its candidate directory", allowed.get("rc") == 0 and allowed_path.read_text(encoding="utf-8") == "allowed")

        case = run_dir / "case"
        (case / "constant").mkdir(parents=True)
        (case / "system").mkdir()
        (case / "system" / "controlDict").write_text("application simpleFoam;\n", encoding="utf-8")
        (case / "log.simpleFoam").write_text("OpenFOAM\nTime = 1\nExecutionTime = 2 s\nEnd\n", encoding="utf-8")
        user_lib = root / "userlib"
        user_lib.mkdir()
        external_so = user_lib / "libcandidate.so"
        external_so.write_bytes(b"external")
        original_user_lib = os.environ.get("FOAM_USER_LIBBIN")
        os.environ["FOAM_USER_LIBBIN"] = str(user_lib)
        try:
            external = _find_compiled_artifacts(run_dir, variant_name="candidate", started_at=time.time() - 2)
        finally:
            if original_user_lib is None:
                os.environ.pop("FOAM_USER_LIBBIN", None)
            else:
                os.environ["FOAM_USER_LIBBIN"] = original_user_lib
        check("FOAM_USER_LIBBIN output is not accepted as a candidate artifact", not external["compiled_so"])

        local_so = case / "customModels" / "Candidate" / "platforms" / "linux" / "lib" / "libcandidate.so"
        local_so.parent.mkdir(parents=True)
        local_so.write_bytes(b"\x7fELF" + b"local")
        local = _find_compiled_artifacts(run_dir, variant_name="candidate", started_at=time.time() - 2)
        check("fresh case-local customModels library is accepted", Path(local["compiled_so"]).resolve() == local_so.resolve() and local["converged"])


def test_oed_recording_is_idempotent_and_builds_bridge() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        out = root / "study"
        disc = out / "open_ended_discovery"
        baseline = out / "mesh_gate" / "group" / "selected"
        candidate = disc / "cand_x"
        candidate_case = candidate / "case"
        for case_dir in (baseline, candidate_case):
            (case_dir / "system").mkdir(parents=True, exist_ok=True)
            (case_dir / "constant").mkdir(parents=True, exist_ok=True)
            (case_dir / "system" / "controlDict").write_text("application simpleFoam;\n", encoding="utf-8")
            (case_dir / "log.simpleFoam").write_text("OpenFOAM\nTime = 1\nExecutionTime = 2 s\nEnd\n", encoding="utf-8")

        write_json(
            disc / "search_config.json",
            {
                "topic": "targeting 20% improvement",
                "total_budget": 2,
                "baseline_case_dir": str(baseline),
                "baseline_direction": "min",
                "saturation_window": 3,
            },
        )
        write_json(disc / "baseline_score.json", {"metric": "rmse", "value": 1.0, "direction": "min", "verified": True})
        write_json(
            candidate / "candidate_record.json",
            {
                "action_type": "code_mod",
                "family": "family-a",
                "model_description": "candidate",
                "case_dir": str(candidate_case),
                "cost": 2,
                "status": "PROCEED",
                "valid_case": True,
                "execution_ok": True,
                "score": {"metric": "rmse", "value": 0.7, "direction": "min"},
            },
        )
        write_json(
            candidate / "agentic_result.json",
            {"status": "OK", "compile_ok": True, "converged": True, "case_dir": str(candidate_case)},
        )

        settings = get_settings().model_copy(update={"knowledge_bundle_dir": root / "knowledge"})
        built = build_manager_tools(settings, out)
        record_tool = next(t for t in built["manager_tools"] if t.__name__ == "oed_record_candidate_results")
        score_tool = next(t for t in built["oed_candidate_tools"] if t.__name__ == "oed_score_candidate")
        mismatched_score = score_tool(str(candidate), str(baseline), "code_mod", "x", "candidate")
        forged_dir = root / "forged_candidate"
        forged_case = forged_dir / "case"
        forged_case.mkdir(parents=True)
        write_json(forged_dir / "candidate_record.json", {"action_type": "code_mod", "case_dir": str(forged_case), "status": "PROCEED", "cost": 2})
        forged_record = record_tool([str(forged_dir)])
        first = record_tool([str(candidate)])
        second = record_tool([str(candidate)])
        history = json.loads((disc / "history.json").read_text(encoding="utf-8"))
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))

        check("candidate scoring cannot substitute an unrelated case directory", mismatched_score.get("ok") is False and "authoritative" in mismatched_score.get("error", ""))
        check("candidate recording rejects paths outside the study OED directory", str(forged_dir.resolve()) in forged_record["missing_candidate_records"])
        check("candidate recording is idempotent across resume/retry", len(history) == 1)
        check("completed OED search promotes baseline and candidate", len(manifest["cases"]) == 2)
        check("bridge returns concrete case IDs for interpretation", len(first["case_ids_to_interpret"]) == 2)
        check("repeated recording preserves the same promoted IDs", second["case_ids_to_interpret"] == first["case_ids_to_interpret"])
        checkpoint = json.loads((out / "checkpoints" / "open_ended_discovery_done.json").read_text(encoding="utf-8"))
        check("OED checkpoint carries the audit bridge signature", checkpoint.get("bridge_signature") == "cfd-open-discovery:bridge:v1")


class _StubOedx:
    """Stands in for oed_extensions so the scorer can be driven with an
    arbitrary metric vector without running OpenFOAM or a comparator."""

    def __init__(self, metrics: dict) -> None:
        self._metrics = metrics

    def compute_metric_vector(self, **_kwargs) -> dict:
        return {"metrics": dict(self._metrics), "errors": {}}


def test_candidate_scoring_refuses_a_metric_the_baseline_lacks() -> None:
    """A candidate is only ever compared to the baseline on the SAME metric.

    Falling back to whatever else the metric vector happens to hold compares,
    e.g., a velocity L2 norm against a Cf RMSE and reports the magnitude
    difference as an improvement — a fabricated winner that then gets
    promoted and written into the paper.
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        out = root / "study"
        disc = out / "open_ended_discovery"
        candidate = disc / "cand_probe"
        case = candidate / "case"
        (case / "system").mkdir(parents=True)
        (case / "constant").mkdir(parents=True)
        (case / "system" / "controlDict").write_text("application simpleFoam;\n", encoding="utf-8")
        reference = disc / "ref.csv"
        disc.mkdir(parents=True, exist_ok=True)
        reference.write_text("x,cf\n0,0.001\n", encoding="utf-8")

        write_json(disc / "search_config.json", {"topic": "t", "total_budget": 4, "baseline_direction": "min"})
        write_json(disc / "baseline_score.json", {"metric": "cf_rmse", "value": 0.01, "direction": "min", "verified": True})
        write_json(disc / "bound_comparators.json", {"cf_rmse": {"script": "x.py"}})
        write_json(disc / "objective_contract.json", {"reference_files": [str(reference)]})
        write_json(candidate / "agentic_result.json",
                   {"status": "OK", "compile_ok": True, "converged": True, "case_dir": str(case)})

        settings = get_settings().model_copy(update={"knowledge_bundle_dir": root / "knowledge"})
        built = build_manager_tools(settings, out)
        score_tool = next(t for t in built["oed_candidate_tools"] if t.__name__ == "oed_score_candidate")

        original = manager_tools_module._oedx
        try:
            # The baseline metric is missing; only an unrelated metric, whose
            # smaller magnitude would look like an 80% improvement.
            manager_tools_module._oedx = _StubOedx({"u_l2": 0.002})
            wrong_metric = score_tool(str(candidate), str(case), "code_mod", "probe", "a candidate")
            # Same case, but the baseline's own metric is present and better.
            manager_tools_module._oedx = _StubOedx({"cf_rmse": 0.002, "u_l2": 999.0})
            right_metric = score_tool(str(candidate), str(case), "code_mod", "probe", "a candidate")
        finally:
            manager_tools_module._oedx = original

        check("a candidate missing the baseline's metric is never scored PROCEED",
              wrong_metric.get("status") != "PROCEED", detail=f"got {wrong_metric.get('status')}")
        check("a candidate missing the baseline's metric reports no score at all",
              wrong_metric.get("score") is None and wrong_metric.get("improvement_pct") is None)
        check("the refusal says which metric was missing, rather than failing silently",
              "cf_rmse" in wrong_metric.get("score_error", ""))
        check("a genuine improvement on the baseline's own metric still scores PROCEED",
              right_metric.get("status") == "PROCEED" and right_metric["score"]["metric"] == "cf_rmse",
              detail=f"got {right_metric.get('status')}")


def test_stale_artifacts_are_cleared_before_a_case_is_rerun() -> None:
    """runApplication refuses to rerun a step whose log exists, so a reused
    case directory would no-op through Allrun and still be read as a clean
    success off the previous attempt's solver log."""
    with tempfile.TemporaryDirectory() as tmp:
        case = Path(tmp) / "case"
        (case / "system").mkdir(parents=True)
        (case / "processor0").mkdir()
        (case / "postProcessing" / "sample").mkdir(parents=True)
        (case / "log.simpleFoam").write_text("Time = 500\nEnd\n", encoding="utf-8")
        (case / "log.blockMesh").write_text("End\n", encoding="utf-8")
        (case / "system" / "controlDict").write_text("application simpleFoam;\n", encoding="utf-8")
        (case / "constant").mkdir()

        _clean_stale_run_artifacts(case)

        check("stale solver logs are removed", not (case / "log.simpleFoam").exists())
        check("stale mesh logs are removed", not (case / "log.blockMesh").exists())
        check("stale decomposition dirs are removed (decomposePar aborts on them)",
              not (case / "processor0").exists())
        check("stale postProcessing output is removed (it is what scoring reads)",
              not (case / "postProcessing").exists())
        check("case input files are left untouched", (case / "system" / "controlDict").is_file())


def test_mesh_gate_is_idempotent_and_explains_a_rejected_requirement() -> None:
    """A converged mesh gate must not re-run, and a rejected requirement must
    say how to fix it — otherwise the manager loops: call, get a generic
    refusal, paraphrase again, repeat, never advancing."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "study"
        level = out / "mesh_gate" / "grp" / "baseline"
        level.mkdir(parents=True)
        write_json(out / "mesh_gate" / "grp" / "selected_mesh_spec.json",
                   {"converged": True, "selected_level": str(level)})
        write_json(out / "requirements.json",
                   [{"case_id": "case_001", "user_requirement_text": "the exact approved text"}])

        settings = get_settings().model_copy(update={"knowledge_bundle_dir": Path(tmp) / "kb"})
        gate = next(f for f in build_manager_tools(settings, out)["manager_tools"]
                    if f.__name__ == "run_mesh_gate")

        converged = gate("grp", "a paraphrase that would normally be rejected")
        check("an already-converged mesh gate returns its selection instead of re-running",
              converged.get("already_converged") is True and converged.get("selected_level") == str(level))

        rejected = gate("fresh_group", "a paraphrase")
        check("a rejected mesh-gate requirement says it must be copied verbatim",
              "verbatim" in rejected.get("error", ""))
        check("a rejected mesh-gate requirement names the case_ids to choose from",
              rejected.get("available_case_ids") == ["case_001"])


def test_oed_promotion_preserves_approved_requirements_and_interpretations() -> None:
    """Promotion re-runs on every call once the search is complete. It must not
    clobber the approved requirements the study ran on, nor destroy the
    interpretations already written into the promoted cases."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "study"
        disc = out / "open_ended_discovery"
        baseline = out / "mesh_gate" / "grp" / "selected"
        candidate, candidate_case = disc / "cand_x", disc / "cand_x" / "case"
        for case_dir in (baseline, candidate_case):
            (case_dir / "system").mkdir(parents=True)
            (case_dir / "constant").mkdir(parents=True)
            (case_dir / "system" / "controlDict").write_text("application simpleFoam;\n", encoding="utf-8")
            (case_dir / "log.simpleFoam").write_text("Time = 1\nExecutionTime = 2 s\nEnd\n", encoding="utf-8")

        approved = [{"case_id": "case_001", "user_requirement_text": "approved text", "study_id": "s1"}]
        write_json(out / "requirements.json", approved)
        write_json(disc / "search_config.json",
                   {"topic": "t", "total_budget": 2, "baseline_case_dir": str(baseline),
                    "baseline_direction": "min", "saturation_window": 3})
        write_json(disc / "baseline_score.json",
                   {"metric": "rmse", "value": 1.0, "direction": "min", "verified": True})
        write_json(candidate / "candidate_record.json",
                   {"action_type": "code_mod", "family": "f", "model_description": "c",
                    "case_dir": str(candidate_case), "cost": 2, "status": "PROCEED",
                    "valid_case": True, "execution_ok": True,
                    "score": {"metric": "rmse", "value": 0.7, "direction": "min"}})
        write_json(candidate / "agentic_result.json",
                   {"status": "OK", "compile_ok": True, "converged": True, "case_dir": str(candidate_case)})

        settings = get_settings().model_copy(update={"knowledge_bundle_dir": Path(tmp) / "kb"})
        record = next(f for f in build_manager_tools(settings, out)["manager_tools"]
                      if f.__name__ == "oed_record_candidate_results")
        first = record([str(candidate)])

        # Stand in for interpret_case having run on a promoted case.
        promoted = out / "cases" / first["case_ids_to_interpret"][-1]
        write_json(promoted / "decision.json", {"status": "PROCEED"})
        bridge_path = out / "bridge.json"
        bridge = json.loads(bridge_path.read_text(encoding="utf-8"))
        bridge["decisions"] = {"case_oed_001": "PROCEED"}
        write_json(bridge_path, bridge)

        record([str(candidate)])  # a resume / retry re-enters promotion

        requirements = json.loads((out / "requirements.json").read_text(encoding="utf-8"))
        check("promotion preserves the approved requirements the study ran on",
              any(item.get("case_id") == "case_001" for item in requirements),
              detail=f"got {[i.get('case_id') for i in requirements]}")
        check("promotion does not duplicate its own OED requirement stubs",
              len([i for i in requirements if i.get("study_id") == "open_discovery"])
              == len(first["case_ids_to_interpret"]))
        check("re-promotion does not destroy an already-written interpretation",
              (promoted / "decision.json").is_file())
        check("re-promotion does not reset the bridge's recorded decisions",
              json.loads(bridge_path.read_text(encoding="utf-8")).get("decisions") == {"case_oed_001": "PROCEED"})


class _FakeBedrock:
    """Minimal stand-in that ``message_cache_dialect`` classifies as Bedrock."""


class _RecordingLLM:
    """Captures the messages a foam_native stage sends, without any network."""

    def __init__(self, dialect_as: object = None) -> None:
        self.captured: list = []
        self._dialect_as = dialect_as

    def invoke(self, messages):
        self.captured.append(messages)

        class _R:
            content = "dummy body"

        return _R()


def _rendered_text(message) -> str:
    """Concatenate a message's content blocks back into the flat prompt text."""
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") for b in content if isinstance(b, dict) and "text" in b)


def test_prompt_cache_breakpoints_preserve_the_prompt_verbatim() -> None:
    """The FoamAgent stages call ``llm.invoke`` directly, which no agent
    middleware can see, so caching is done with an in-message breakpoint.

    The whole approach rests on one property: splitting into content blocks
    must not change a single character the model reads. Pin that, for both
    provider dialects and for a provider with no caching at all.
    """
    from cfd_langgraph.foam_native import prompts as P
    from cfd_langgraph.foam_native import review as review_mod
    from cfd_langgraph.foam_native import writer as writer_mod
    from cfd_langgraph.llm import caching as caching_mod

    big_tutorial = "TUTORIAL LINE\n" * 500  # comfortably over the min cacheable prefix
    kwargs = dict(
        file_name="U", folder_name="0", user_requirement="make a periodic hill run",
        tutorial_reference=big_tutorial, written_files_ctx="controlDict: ...", case_solver="simpleFoam",
    )
    expected_write = P.INITIAL_WRITE_USER_PROMPT.format(**{
        k: v for k, v in kwargs.items() if k != "case_solver"
    })

    original_dialect = caching_mod.message_cache_dialect
    try:
        for dialect in ("bedrock", "anthropic", None):
            caching_mod.message_cache_dialect = lambda _m, _d=dialect: _d
            llm = _RecordingLLM()
            writer_mod.write_file_initial(llm, **kwargs)
            system, human = llm.captured[-1]

            check(f"[{dialect}] the rendered write prompt is byte-identical to the flat template",
                  _rendered_text(human) == expected_write)
            # The real property: a breakpoint in the user message can only hit
            # if everything before it is identical too, so the system prompt
            # must not vary between files. Assert that directly by writing a
            # second, different file and comparing.
            other_llm = _RecordingLLM()
            writer_mod.write_file_initial(other_llm, **{**kwargs, "file_name": "p", "folder_name": "system"})
            check(f"[{dialect}] the write system prompt is identical across different files",
                  other_llm.captured[-1][0].content == system.content)

            if dialect is None:
                check("[none] an uncached provider gets a plain string, not blocks",
                      isinstance(human.content, str))
            else:
                check(f"[{dialect}] the prompt is split into blocks with a breakpoint",
                      isinstance(human.content, list) and len(human.content) >= 2)
                marker_block = next(
                    (i for i, b in enumerate(human.content)
                     if "cachePoint" in b or b.get("cache_control")), None
                )
                check(f"[{dialect}] a cache breakpoint is present", marker_block is not None)
                cached_part = "".join(
                    b.get("text", "") for b in human.content[: (marker_block or 0) + 1]
                )
                check(f"[{dialect}] the cached prefix covers the large tutorial reference",
                      big_tutorial in cached_part)
                check(f"[{dialect}] the per-call tail is outside the cached prefix",
                      P.WRITE_CACHE_SPLIT_MARKER not in cached_part)

        # The reviewer re-sends the tutorial every retry round; same treatment.
        caching_mod.message_cache_dialect = lambda _m: "bedrock"
        llm = _RecordingLLM()
        review_mod.review_errors(
            llm, tutorial_reference=big_tutorial, foamfiles_xml="<f/>",
            error_logs="FOAM FATAL ERROR", user_requirement="req", history_text="",
            similar_case_advice_block="advice",
        )
        _system, human = llm.captured[-1]
        expected_review = P.REVIEWER_USER_PROMPT.format(
            tutorial_reference=big_tutorial, similar_case_advice_block="advice",
            foamfiles_xml="<f/>", error_logs="FOAM FATAL ERROR", user_requirement="req",
            history_text="",
        )
        check("the rendered reviewer prompt is byte-identical to the flat template",
              _rendered_text(human) == expected_review)
        check("the reviewer's error logs stay outside the cached prefix",
              "FOAM FATAL ERROR" not in human.content[0].get("text", ""))
    finally:
        caching_mod.message_cache_dialect = original_dialect


def test_cache_breakpoint_survives_bedrock_message_conversion() -> None:
    """A cachePoint block is only useful if langchain-aws forwards it to the
    Converse API rather than dropping it as an unknown block type."""
    from langchain_aws.chat_models.bedrock_converse import _messages_to_bedrock
    from langchain_core.messages import SystemMessage as _Sys

    from cfd_langgraph.llm.caching import cacheable_human_message

    human = cacheable_human_message(_FakeBedrock(), "x", "y")
    check("a short prefix is left as a plain string (below the cacheable minimum)",
          isinstance(human.content, str) and human.content == "xy")

    from cfd_langgraph.llm import caching as caching_mod
    original = caching_mod.message_cache_dialect
    try:
        caching_mod.message_cache_dialect = lambda _m: "bedrock"
        human = cacheable_human_message(_FakeBedrock(), "P" * 5000, "TAIL")
        converted, _system = _messages_to_bedrock([_Sys(content="s"), human], [])
    finally:
        caching_mod.message_cache_dialect = original

    blocks = converted[0]["content"]
    check("langchain-aws forwards the cachePoint block to the Converse API",
          any("cachePoint" in b for b in blocks), detail=f"got {[list(b) for b in blocks]}")
    check("the text either side of the breakpoint survives conversion intact",
          "".join(b.get("text", "") for b in blocks) == "P" * 5000 + "TAIL")


def test_a_failed_comparator_does_not_invent_a_metric_value() -> None:
    """The named-metric fallback used to take the first `identifier: number`
    anywhere in stdout+stderr, so a comparator that computed nothing still
    produced a number — "nPoints = 12800" from the preamble became the score."""
    import oed_extensions as oedx

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        case = root / "case"
        case.mkdir()
        reference = root / "ref.csv"
        reference.write_text("x,cf\n0,0.001\n", encoding="utf-8")

        failing = root / "failing_comparator.py"
        failing.write_text(
            "print('Loading case: /runs/case_3')\n"
            "print('nPoints = 12800')\n"
            "print('latestTime: 2000')\n"
            "print('RMSE cf: could not compute (missing reference)')\n",
            encoding="utf-8",
        )
        named = root / "named_comparator.py"
        named.write_text("print('nPoints = 12800')\nprint('cf_rmse = 0.0043')\n", encoding="utf-8")

        failed = oedx.compute_metric_vector(
            case_dir=case, bound_comparators={"cf_rmse": {"path": str(failing)}},
            reference_file=reference, metric_specs={},
        )
        check("a comparator that computed nothing yields no metric value",
              "cf_rmse" not in (failed.get("metrics") or {}),
              detail=f"got {failed.get('metrics')}")
        check("the failure is reported as an error rather than silently dropped",
              "cf_rmse" in (failed.get("errors") or {}))

        ok = oedx.compute_metric_vector(
            case_dir=case, bound_comparators={"cf_rmse": {"path": str(named)}},
            reference_file=reference, metric_specs={},
        )
        check("a line naming the metric is still accepted without the METRIC prefix",
              (ok.get("metrics") or {}).get("cf_rmse") == 0.0043,
              detail=f"got {ok.get('metrics')}")


def test_generated_cases_inherit_the_base_case_function_objects() -> None:
    """Without a functions block a case runs to completion and measures
    nothing, so scoring has no data and the whole search stalls — the observed
    failure in two live runs. Function objects are contract, copied from the
    base case, not prose for the case writer to interpret."""
    from cfd_langgraph.foam_native.loop import _extract_functions_block, _seed_function_objects

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        base, gen = root / "base", root / "gen"
        (base / "system").mkdir(parents=True)
        (gen / "system").mkdir(parents=True)
        (base / "system" / "controlDict").write_text(
            'application     simpleFoam;\nendTime  100;\n\n'
            'functions\n{\n    wallShearStress\n    {\n        type wallShearStress;\n'
            '        patches (bottomWall topWall);\n    }\n\n    yPlus\n    {\n'
            '        type yPlus;\n    }\n}\n\n'
            '// ************************************************************ //\n',
            encoding="utf-8",
        )
        generated = 'application     simpleFoam;\nendTime  4000;\n\n// ****** //\n'
        (gen / "system" / "controlDict").write_text(generated, encoding="utf-8")

        seeded = _seed_function_objects(gen, base)
        text = (gen / "system" / "controlDict").read_text(encoding="utf-8")
        check("the base case's function objects are copied into a generated case",
              "wallShearStress" in text and "yPlus" in text, detail=f"seeded={seeded!r}")
        check("the seeding names what it copied", "wallShearStress" in seeded)
        check("the generated case's own settings are preserved", "endTime  4000;" in text)
        check("the copied block parses as a functions entry", bool(_extract_functions_block(text)))
        check("seeding is idempotent — a second pass changes nothing",
              _seed_function_objects(gen, base) == "")

        # A case that already defines its own functions is left alone.
        own = root / "own"
        (own / "system").mkdir(parents=True)
        (own / "system" / "controlDict").write_text(
            "application simpleFoam;\nfunctions\n{\n    mine\n    {\n        type yPlus;\n    }\n}\n",
            encoding="utf-8",
        )
        check("a case with its own functions block is not overwritten",
              _seed_function_objects(own, base) == ""
              and "wallShearStress" not in (own / "system" / "controlDict").read_text(encoding="utf-8"))


def test_proposals_do_not_deadlock_once_every_family_is_visited() -> None:
    """Once each family has been tried, select_niche keeps asking for a NEW
    family (all elites are None, so every q ties and the zero-visit new option
    wins). If a proposal is discarded merely for classifying into a family the
    archive already holds, every batch comes back empty and budget_used never
    advances — the search stalls until the run gives up."""
    from cfd_langgraph.manager import tools as tools_mod

    src = pathlib.Path(tools_mod.__file__).read_text(encoding="utf-8")
    marker = 'if selection.get("is_new") and classified in batch_families:'
    check("an already-visited archive family alone does not discard a proposal",
          marker in src,
          detail="proposal dedup must key on the batch, not the archive")
    check("archive-wide family dedup is gone from the proposal filter",
          'classified in archive.niches or classified in batch_families' not in src)


def test_allrun_always_runs_the_solver_and_survives_json_output() -> None:
    """An Allrun without the solver produces no solver log, so every retry
    fails for a reason no dictionary rewrite can fix. Observed live: the model
    returned the command list as a JSON array, the whole string was prefixed
    with runApplication, and the case ran an application literally named
    ["blockMesh"] — writing a file called log.[blockMesh] and burning all ten
    retries."""
    from cfd_langgraph.foam_native.allrun import build_allrun_script, parse_command_list

    check("a JSON-array command list is parsed, not pasted verbatim",
          parse_command_list('["blockMesh"]') == ["blockMesh"])
    check("a fenced JSON command list is parsed", parse_command_list('```json\n["blockMesh"]\n```') == ["blockMesh"])
    check("a newline command list still works", parse_command_list("blockMesh\ncheckMesh") == ["blockMesh", "checkMesh"])

    script = build_allrun_script('["blockMesh"]', case_solver="simpleFoam")
    check("no bracket ever reaches the shell", "[" not in script and "]" not in script, detail=script)
    check("the solver is appended when the list omits it", "runApplication simpleFoam" in script)
    check("the mesh step is preserved", "runApplication blockMesh" in script)

    twice = build_allrun_script("blockMesh\nsimpleFoam", case_solver="simpleFoam")
    check("the solver is not duplicated when already listed",
          twice.count("runApplication simpleFoam") == 1, detail=twice)
    check("an empty command list still runs the solver",
          "runApplication simpleFoam" in build_allrun_script("", case_solver="simpleFoam"))


def test_protected_artifacts_cover_case_and_sidecar_paths() -> None:
    from cfd_langgraph.manager.tools import _is_protected_artifact

    check("the checkpoint database is protected", _is_protected_artifact(Path("/x/checkpoints.sqlite")))
    check("its write-ahead log is protected too (corrupting it breaks resume)",
          _is_protected_artifact(Path("/x/checkpoints.sqlite-wal")))
    check("protection is case-insensitive", _is_protected_artifact(Path("/x/Candidate_Record.json")))
    check("anything under a state/ or checkpoints/ directory is protected",
          _is_protected_artifact(Path("/x/state/anything.json"), ("state", "anything.json")))
    check("an ordinary scratch file is still writable", not _is_protected_artifact(Path("/x/notes.txt")))


def test_a_provider_without_tool_calling_is_refused_up_front() -> None:
    """The manager is a tool-calling agent throughout; a provider that cannot
    bind tools must fail with an explanation, not an opaque traceback on the
    first turn after the user has already set up the study."""
    from cfd_langgraph.manager.deep_agent import _require_tool_calling_support

    class _NoTools:
        """No bind_tools at all — the shape a bare wrapper class has."""

    try:
        _require_tool_calling_support(_NoTools(), get_settings())
    except RuntimeError as exc:
        check("a non-tool-calling provider is refused before the study starts",
              "does not support tool calling" in str(exc))
        check("the refusal names providers that do work",
              all(name in str(exc) for name in ("bedrock", "anthropic", "openai", "gemini")))
    else:
        check("a non-tool-calling provider is refused before the study starts", False,
              detail="no RuntimeError raised")


class _FanOutState(TypedDict):
    done: Annotated[list, operator.add]


def test_concurrent_pauses_resume_instead_of_wedging_the_study() -> None:
    """A Ctrl-C during a fan-out leaves one interrupt per concurrent subagent.

    Resuming those with the unkeyed ``Command(resume={"decisions": [...]})``
    raises RuntimeError, which used to escape the CLI and — because
    ``resume --out-dir`` re-entered the identical checkpoint — made the study
    permanently unresumable. This drives the real CLI pause handler against a
    real two-interrupt LangGraph checkpoint.
    """
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import interrupt

    from cfd_langgraph.cli.repl import _handle_interrupt, _pending_interrupts

    def _pause(name: str):
        def node(_state):
            return {"done": [interrupt({"action_requests": [{"name": name, "args": {}}]})]}
        return node

    graph_builder = StateGraph(_FanOutState)
    graph_builder.add_node("a", _pause("run_case_a"))
    graph_builder.add_node("b", _pause("run_case_b"))
    for node_name in ("a", "b"):
        graph_builder.add_edge(START, node_name)
        graph_builder.add_edge(node_name, END)
    graph = graph_builder.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "fanout"}}
    graph.invoke({"done": []}, config=config)

    pending = _pending_interrupts(graph, config)
    check("every concurrent pause is surfaced, not just the first", len(pending) == 2,
          detail=f"got {len(pending)}")

    with tempfile.TemporaryDirectory() as tmp:
        try:
            _handle_interrupt(graph, config, Path(tmp), pending, lambda _prompt: "continue")
        except Exception as exc:  # noqa: BLE001 - the bug under test was an escaping RuntimeError
            check("resuming a multi-interrupt checkpoint does not raise", False, detail=f"{type(exc).__name__}: {exc}")
        else:
            check("resuming a multi-interrupt checkpoint does not raise", True)

    check("both paused calls actually resume", not _pending_interrupts(graph, config))


def test_oed_candidate_identity_is_exact_where_it_can_be() -> None:
    """Budget is the scarce resource, so a repeat must not be paid for twice.

    Split deliberately. A coefficient experiment IS its coefficients: that
    identity is exact, free to compare, and cannot be wrong, so it is decided
    here. Everything else is a described experiment, and deciding whether two
    descriptions are the same experiment is a reading task — it belongs to
    `_llm_duplicate_of`, not to string normalisation. Normalising prose was
    tried and could not work: the same experiment appeared seven ways in one
    real run, from a paragraph down to "SA-Cross-Diffusion with c_cd=1.0".
    """
    from cfd_langgraph.manager.tools import _oed_candidate_fingerprint

    coeff = _oed_candidate_fingerprint("SA-Cross-Diffusion", "experiment", "any wording", {"c_cd": 1.0})
    check(
        "a coefficient experiment is identified by its coefficients, whatever the wording",
        coeff == _oed_candidate_fingerprint(
            "SA-Cross-Diffusion", "experiment", "totally different prose", {"c_cd": 1.0}
        ),
    )
    check(
        "a different coefficient value is a different experiment",
        coeff != _oed_candidate_fingerprint("SA-Cross-Diffusion", "experiment", "any wording", {"c_cd": 0.8}),
    )
    check(
        "an equal value written differently is the same experiment",
        _oed_candidate_fingerprint("f", "experiment", "a", {"c": 1})
        == _oed_candidate_fingerprint("f", "experiment", "b", {"c": 1.0}),
    )
    check(
        "override ordering does not change the identity",
        _oed_candidate_fingerprint("f", "experiment", "h", {"Cb1": 0.1, "Cw2": 0.3})
        == _oed_candidate_fingerprint("f", "experiment", "h", {"Cw2": 0.3, "Cb1": 0.1}),
    )
    check(
        "the same idea in another family is not a duplicate",
        coeff != _oed_candidate_fingerprint("SA-RC", "experiment", "any wording", {"c_cd": 1.0}),
    )

    # For a code_mod there is no exact identity, so this key claims nothing
    # beyond "the very same string" — the real decision is the LLM's.
    check(
        "a code_mod key ignores only whitespace",
        _oed_candidate_fingerprint("f", "code_mod", "add  a   limiter")
        == _oed_candidate_fingerprint("f", "code_mod", "add a limiter"),
    )
    check(
        "and does not pretend to judge meaning",
        _oed_candidate_fingerprint("f", "code_mod", "add a limiter")
        != _oed_candidate_fingerprint("f", "code_mod", "Add a limiter."),
    )


def test_llm_duplicate_check_fails_toward_spending() -> None:
    """A missed duplicate costs one evaluation; a wrongly-dropped candidate
    loses the idea entirely. So an unavailable judge must not veto."""
    from cfd_langgraph.manager.tools import _llm_duplicate_of

    history = [
        {"action_type": "experiment", "variant_name": "v1", "family": "F", "model_description": "d"}
    ]
    candidate = {"target_family": "F", "action_type": "code_mod", "hypothesis": "h"}

    class _Broken:
        def with_structured_output(self, _schema):
            raise RuntimeError("provider down")

    check("a failing judge allows the candidate", _llm_duplicate_of(candidate, history, _Broken()) is None)
    check("empty history needs no judge at all", _llm_duplicate_of(candidate, [], _Broken()) is None)

    class _Hallucinating:
        def with_structured_output(self, _schema):
            class _R:
                def invoke(self, _prompt):
                    class _V:
                        duplicate_of = "a_variant_that_never_ran"
                        reason = ""
                    return _V()
            return _R()

    check(
        "a name that is not in history is ignored, not acted on",
        _llm_duplicate_of(candidate, history, _Hallucinating()) is None,
    )


def test_hypothesis_stages_are_given_the_fixed_case_setup() -> None:
    """Regression: 6 of 6 hypotheses rejected, study stalled with nothing to approve.

    Ideation and critique were both blind to `starter_understanding.json`. The
    ideator proposed studies of a *different* configuration — setup-sensitivity
    questions, several at Re_H=10595 when the starter case is Re_H=5600 — and
    the reviewer, whose implementability criterion asks whether an idea
    specifies geometry/mesh/BCs/solver, rejected every one for not restating
    details the case had already fixed.
    """
    import inspect

    from cfd_langgraph import hypothesis_pipeline, ideation
    from cfd_langgraph.agents.hypothesis_critique_agent import HypothesisCritiqueAgent
    from cfd_langgraph.manager.tools import _starter_case_context

    for label, fn in (
        ("run_propose_critique_rank", hypothesis_pipeline.run_propose_critique_rank),
        ("run_ideation_batch", ideation.run_ideation_batch),
        ("_generate_one_idea", ideation._generate_one_idea),
        ("HypothesisCritiqueAgent.critique", HypothesisCritiqueAgent.critique),
    ):
        check(f"{label} accepts the fixed case setup",
              "case_context" in inspect.signature(fn).parameters)

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        check("no starter file yields no context, not a crash", _starter_case_context(out) == "")
        (out / "starter_understanding.json").write_text(json.dumps({
            "base_case_path": "periodic_hill_sa",
            "flow_parameters": {"Re": 5600, "nu": 5e-06, "Ub": 0.028, "dimension": "2D",
                                "geometry": "periodic hill, cyclic inlet/outlet"},
            "reference_data": {"quantities": ["Cf", "x/h"], "usage_guidance": "RMSE against cf_dns"},
            "formula_or_model_spec": "NO EXPLICIT FORMULA/EQUATION FILE IS PRESENT",
        }))
        context = _starter_case_context(out)
        for needle in ("5600", "0.028", "periodic hill", "Cf"):
            check(f"the context carries {needle!r}", needle in context, detail=context[:150])
        check("a 'no formula present' note is not passed off as a spec",
              "NO EXPLICIT FORMULA" not in context)


def test_requirements_publish_what_passed_not_all_or_nothing() -> None:
    """Regression: 16-of-17 behaved exactly like 0-of-17.

    Requirement validation is N independent non-deterministic verdicts in
    series. Requiring every one to pass meant a single unlucky rejection wrote
    no requirements.json at all, and the manager regenerated all 17 again —
    ten minutes a cycle, indefinitely. Measured across four runs of the same
    stage: 13 invalid, then 2, then 6, then 1. Never zero.
    """
    import tempfile as _tf

    from cfd_langgraph.config import get_settings
    from cfd_langgraph.manager.tools import build_manager_tools

    with _tf.TemporaryDirectory() as tmp:
        out = Path(tmp)
        (out / "hypotheses_approved.json").write_text(json.dumps({
            "approved_hypotheses": [{
                "candidate_id": "cand_01",
                "idea": {"study_id": "s1", "experiments": [
                    {"experiment_id": "exp_001", "name": "a"},
                    {"experiment_id": "exp_002", "name": "b"},
                    {"experiment_id": "exp_003", "name": "c"},
                ]},
            }],
        }))
        (out / "hypotheses_ranked.json").write_text(json.dumps({"research_topic": "t"}))

        tools = {f.__name__: f for f in build_manager_tools(get_settings(), out)["manager_tools"]}
        generate = tools["generate_case_requirements"]

        import cfd_langgraph.agents.hypothesis_agent as ha

        original = ha.HypothesisAgent.generate_validated_requirement
        try:
            # exp_002 fails; the other two pass.
            def fake(self, idea, simulation, run_topic="", **kw):
                ok = simulation["simulation_id"] != "exp_002"
                return {"requirement": f"req for {simulation['simulation_id']}", "valid": ok}

            ha.HypothesisAgent.generate_validated_requirement = fake
            result = generate()
        finally:
            ha.HypothesisAgent.generate_validated_requirement = original

        check("a partial failure still publishes", "error" not in result, detail=str(result)[:160])
        check("only the passing requirements are published",
              result.get("num_requirements") == 2, detail=str(result.get("num_requirements")))
        published = json.loads((out / "requirements.json").read_text())
        check("the failed one is excluded from the published file",
              all(r["experiment_id"] != "exp_002" for r in published),
              detail=str([r["experiment_id"] for r in published]))
        check("every published requirement is valid",
              all(r["requirement_valid"] for r in published))
        check("the exclusion is reported, not silent",
              result.get("excluded_case_ids") == ["case_002"], detail=str(result.get("excluded_case_ids")))
        check("and the full set is kept for inspection", (out / "requirements_draft.json").is_file())


def test_requirements_still_refuse_when_nothing_is_valid() -> None:
    """Excluding a bad requirement is safe; publishing an empty set is not."""
    import tempfile as _tf

    from cfd_langgraph.config import get_settings
    from cfd_langgraph.manager.tools import build_manager_tools

    with _tf.TemporaryDirectory() as tmp:
        out = Path(tmp)
        (out / "hypotheses_approved.json").write_text(json.dumps({
            "approved_hypotheses": [{
                "candidate_id": "c1",
                "idea": {"study_id": "s", "experiments": [{"experiment_id": "exp_001", "name": "a"}]},
            }],
        }))
        (out / "hypotheses_ranked.json").write_text(json.dumps({"research_topic": "t"}))
        tools = {f.__name__: f for f in build_manager_tools(get_settings(), out)["manager_tools"]}

        import cfd_langgraph.agents.hypothesis_agent as ha

        original = ha.HypothesisAgent.generate_validated_requirement
        try:
            ha.HypothesisAgent.generate_validated_requirement = (
                lambda self, idea, simulation, run_topic="", **kw: {"requirement": "r", "valid": False}
            )
            result = tools["generate_case_requirements"]()
        finally:
            ha.HypothesisAgent.generate_validated_requirement = original

        check("all-invalid is still an error", "error" in result, detail=str(result)[:120])
        check("and nothing runnable is published", not (out / "requirements.json").is_file())


def test_oed_search_is_seeded_from_the_approved_hypotheses() -> None:
    """Don't discard what the hypothesis stage produced.

    That stage fetches literature, generates ideas against it, critiques them,
    and has a human approve the survivors — yielding named, implementable
    modifications. The search then invented families from unaided recall and
    saturated at 64 of 100 budget units while nine vetted directions sat unused
    on disk, one of whose families produced the best result of the run.
    """
    from cfd_langgraph.manager.tools import _approved_hypothesis_directions

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        check("no approved file yields nothing, not a crash", _approved_hypothesis_directions(out) == [])

        (out / "hypotheses_approved.json").write_text(json.dumps({
            "approved_hypotheses": [{
                "candidate_id": "cand_05",
                "idea": {"experiments": [
                    {"name": "Weak APG production damping",
                     "notes": "Multiply SA production by max(0.5, 1-kP*A*W)."},
                    {"experiment_id": "exp_002", "description": "fw-shape damping via r_eff."},
                    {"name": "", "notes": ""},
                ]},
            }],
        }))
        directions = _approved_hypothesis_directions(out)
        check("named experiments are recovered", len(directions) == 2, detail=str(directions))
        check("the name carries", directions[0]["name"] == "Weak APG production damping")
        check("the detail carries", "max(0.5" in directions[0]["detail"])
        check("an id-only experiment still counts", directions[1]["name"] == "exp_002")
        check("empty entries are dropped", all(d["name"] or d["detail"] for d in directions))

    source = Path("src/cfd_langgraph/manager/tools.py").read_text(encoding="utf-8")
    block = source[source.index("APPROVED DIRECTIONS"):][:700]
    check("they reach the proposer prompt", "literature-grounded" in block)
    check("and are preferred over inventing from scratch", "over inventing a family from scratch" in block)
    check("already-tried ones are excluded", "already-evaluated list above" in block)


def test_promotion_does_not_require_an_interpretation_that_cannot_run() -> None:
    """The deadlock that stopped a finished study from becoming a paper.

    Promotion was gated on `proceed_count > 0`. `PROCEED` is set by
    `interpret_case`, which only resolves cases already promoted into `cases/`.
    So interpretation needed promotion and promotion needed interpretation, and
    neither could go first. Measured on a real study: 49 evaluations, 23 better
    than baseline, best +4.90%, and nothing promoted — the manager could not
    reach interpretation, analysis or the paper, tried to register the cases by
    hand, and was correctly refused by the protected-artifact guard.
    """
    import inspect

    from cfd_langgraph.manager import tools as T

    source = inspect.getsource(T)
    check("promotion keys off measured improvement", "improving = [h for h in real_evals if _beats_baseline(h)]" in source)
    check(
        "a study with improvement counts as having a winner",
        "has_winner = proceed_count > 0 or bool(improving)" in source,
    )
    check(
        "saturation can complete the search without an interpreter verdict",
        "saturated and (proceed_count > 0 or bool(improving))" in source,
    )
    check("only the best are promoted", "promotable = sorted(improving, key=_score_of)[:_OED_MAX_PROMOTED]" in source)

    # The comparison itself must respect the metric's direction.
    ns: dict = {}
    exec(
        "def _beats(value, baseline, direction):\n"
        "    return float(value) > float(baseline) if direction == 'max' else float(value) < float(baseline)\n",
        ns,
    )
    beats = ns["_beats"]
    check("for a min metric, lower wins", beats(0.0041, 0.0043, "min") and not beats(0.0045, 0.0043, "min"))
    check("for a max metric, higher wins", beats(0.95, 0.90, "max") and not beats(0.85, 0.90, "max"))


def test_a_study_with_no_improvement_still_promotes_nothing() -> None:
    """Relaxing the gate must not mean promoting a failure as a result."""
    import inspect

    from cfd_langgraph.manager import tools as T

    source = inspect.getsource(T)
    check(
        "candidates worse than baseline are excluded from promotion",
        "improving = [h for h in real_evals if _beats_baseline(h)]" in source,
    )
    check(
        "a missing baseline or score never counts as beating it",
        "if value is None or baseline_value is None:" in source
        and source.index("if value is None or baseline_value is None:") < source.index("improving = [h for h in real_evals"),
    )


def main() -> int:
    test_literature_schema_normalization()
    test_literature_search_balances_query_variants()
    test_candidate_batch_similarity()
    test_hypothesis_pipeline_uses_supplied_literature_and_gates()
    test_novelty_evaluator_fails_closed()
    test_score_and_status_helpers()
    test_study_wide_concurrency_and_exclusive_calibration()
    test_manager_write_scope_and_protected_artifacts()
    test_code_mod_filesystem_isolation_and_case_local_artifacts()
    test_oed_recording_is_idempotent_and_builds_bridge()
    test_candidate_scoring_refuses_a_metric_the_baseline_lacks()
    test_stale_artifacts_are_cleared_before_a_case_is_rerun()
    test_concurrent_pauses_resume_instead_of_wedging_the_study()
    test_oed_candidate_identity_is_exact_where_it_can_be()
    test_llm_duplicate_check_fails_toward_spending()
    test_hypothesis_stages_are_given_the_fixed_case_setup()
    test_oed_search_is_seeded_from_the_approved_hypotheses()
    test_promotion_does_not_require_an_interpretation_that_cannot_run()
    test_a_study_with_no_improvement_still_promotes_nothing()
    test_requirements_publish_what_passed_not_all_or_nothing()
    test_requirements_still_refuse_when_nothing_is_valid()
    test_prompt_cache_breakpoints_preserve_the_prompt_verbatim()
    test_cache_breakpoint_survives_bedrock_message_conversion()
    test_a_failed_comparator_does_not_invent_a_metric_value()
    test_generated_cases_inherit_the_base_case_function_objects()
    test_proposals_do_not_deadlock_once_every_family_is_visited()
    test_allrun_always_runs_the_solver_and_survives_json_output()
    test_protected_artifacts_cover_case_and_sidecar_paths()
    test_a_provider_without_tool_calling_is_refused_up_front()
    test_mesh_gate_is_idempotent_and_explains_a_rejected_requirement()
    test_oed_promotion_preserves_approved_requirements_and_interpretations()
    print(f"\n{'ALL PASS' if FAILURES == 0 else f'{FAILURES} FAILURE(S)'}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
