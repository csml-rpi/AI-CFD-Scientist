#!/usr/bin/env python3
"""
Closed-loop open-ended CFD discovery.

Each iteration:
  1. Decision LLM looks at full history + reference data → decides next action
  2. Action is either:
       code_mod  — propose new model formula, compile new .so, run experiment (costs 2 budget units)
       experiment — re-run with existing compiled model + different parameters (costs 1 budget unit)
       stop      — LLM decides results are good enough or no promising directions remain
  3. Interpreter evaluates the run → compact summary added to history
  4. Repeat until budget exhausted or stop decision

Budget is controlled by --budget N (total units; code_mod = 2, experiment = 1).

Produces:
  {run_dir}/open_ended_discovery/history.json   — full iteration log
  {run_dir}/open_ended_discovery/summary.json   — best case + overall findings
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from timeline_logger import append_timeline_event, resolve_timeline_path
from oed_search_archive import SearchArchive

# run_validity is sibling-imported lazily inside helpers so the module remains
# safe to import even when the optional CFD-langgraph stack is unavailable.


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Suffixes that make a declared reference file a program rather than data. A
# study may name its authoritative comparator among its reference files -- that
# is useful, it is what the authored comparator is written from -- but it can
# never be the thing a comparator is pointed at with --reference.
_CODE_SUFFIXES = {".py", ".sh", ".pyc", ".ipynb"}


def _reference_data_file(ref_files, fallback=None):
    """The first declared reference file that is DATA, not a program.

    Every comparator is invoked as `--reference <this>` and reads it as a
    table. Taking ref_files[0] blindly hands over whatever happens to be
    listed first, and a study that declares its authoritative scorer among
    its reference files puts a .py there: the comparator then parses Python
    source as CSV, crashes, and the harness concludes the comparator is
    broken. Measured on ph_codex_20260902_1806 -- the starter's own scorer,
    which reproduces the study's stated baseline exactly, was discovered and
    thrown away on this alone, and a replacement authored in its place.

    Falls back to the old behaviour when every declared file is a program,
    so a study that declares nothing else is no worse off than before.
    """
    data = [f for f in (ref_files or []) if Path(f).suffix.lower() not in _CODE_SUFFIXES]
    if data:
        return data[0]
    if ref_files:
        return ref_files[0]
    return fallback


def _bootstrap(repo_root: Path) -> None:
    foam_src = repo_root / "Foam-Agent" / "src"
    lang_src = repo_root / "src"
    for p in (str(foam_src), str(lang_src), str(Path(__file__).parent)):
        if p not in sys.path:
            sys.path.insert(0, p)


def _read_json(p: Path, default: Any) -> Any:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _text_fingerprint(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8", errors="ignore")).hexdigest()[:12]


def _slugify_variant(text: str, max_len: int = 28) -> str:
    """Turn a model description / variant name into a filesystem-safe slug.

    Generic — works for any code_mod mode or topic. Used to make OED
    iteration directories human-readable (`iter_003_SA_RC_code_mod/`
    instead of `iter_003_code_mod/`) so the workspace is self-documenting.
    """
    if not text:
        return ""
    # Prefer an explicit TLA-like prefix in parens / dashes / first capitalized
    # token. For free-form descriptions, use the first 4-5 meaningful words.
    s = str(text).strip()
    # Grab the leading alphanumeric token (e.g. "SA-RC:" -> "SA-RC", "HillSA ..." -> "HillSA")
    m = re.match(r"\s*([A-Za-z][A-Za-z0-9_\-]{1,40})", s)
    head = m.group(1) if m else s[:40]
    # Replace non-word chars with underscore, compress, strip
    slug = re.sub(r"[^A-Za-z0-9]+", "_", head).strip("_")
    if len(slug) > max_len:
        slug = slug[:max_len].rstrip("_")
    return slug


def _fix_json_string_literals(txt: str) -> str:
    """Escape literal newlines/tabs inside JSON string values emitted by LLMs."""
    result: List[str] = []
    in_string = False
    i = 0
    while i < len(txt):
        c = txt[i]
        if c == "\\" and in_string:
            result.append(c)
            i += 1
            if i < len(txt):
                result.append(txt[i])
                i += 1
            continue
        if c == '"':
            in_string = not in_string
            result.append(c)
        elif c == "\n" and in_string:
            result.append("\\n")
        elif c == "\r" and in_string:
            result.append("\\r")
        elif c == "\t" and in_string:
            result.append("\\t")
        else:
            result.append(c)
        i += 1
    return "".join(result)


def _try_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def _parse_requested_model(case_dir: Path) -> str:
    mt = case_dir / "constant" / "momentumTransport"
    if not mt.is_file():
        return ""
    txt = mt.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"\bmodel\s+([A-Za-z0-9_]+)\s*;", txt)
    return m.group(1).strip() if m else ""


def _parse_selected_model_from_log(case_dir: Path) -> str:
    logp = case_dir / "log.simpleFoam"
    if not logp.is_file():
        return ""
    txt = logp.read_text(encoding="utf-8", errors="ignore")
    matches = re.findall(r"Selecting\s+RAS\s+turbulence\s+model\s+([A-Za-z0-9_]+)", txt)
    return matches[-1].strip() if matches else ""


def _extract_error_metrics(metrics_summary: str) -> List[Dict[str, float]]:
    """
    Extract objective-like numeric metrics from free-form summary text.
    This is intentionally generic and does not assume a specific QoI.
    """
    if not metrics_summary:
        return []
    out: List[Dict[str, float]] = []
    # Examples handled:
    # "RMSE: 0.0042", "L2 error = 1.2e-3", "mae vs ref 0.10",
    # "RMSE vs exact-match DNS reference: 0.004297"  (note hyphenated words!)
    # The non-greedy [^0-9]*? filler accepts hyphens between the metric name
    # and the number; the number itself still allows a leading [-+]?.
    pat = re.compile(
        r"(?i)\b(rmse|rms(?:\s*error)?|l2(?:\s*error)?|mae|mse|error)\b[^0-9]*?([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
    )
    for m in pat.finditer(metrics_summary):
        name = m.group(1).strip().lower().replace(" ", "_")
        val = _try_float(m.group(2))
        if val is not None:
            out.append({"name": name, "value": float(val)})
    return out


def _choose_primary_score(extracted: List[Dict[str, float]]) -> Optional[Dict[str, float]]:
    """
    Pick a single objective for ranking (lower is better), preferring RMSE/L2.
    """
    if not extracted:
        return None
    prio = ["rmse", "rms_error", "rms", "l2_error", "l2", "mae", "mse", "error"]
    by_name: Dict[str, List[float]] = {}
    for e in extracted:
        by_name.setdefault(e["name"], []).append(float(e["value"]))
    for k in prio:
        vals = by_name.get(k)
        if vals:
            return {"metric": k, "value": float(min(vals)), "direction": "min"}
    # Fallback: first available metric
    e0 = extracted[0]
    return {"metric": str(e0["name"]), "value": float(e0["value"]), "direction": "min"}


def _evaluate_open_ended_case_contract(
    *,
    case_dir: Path,
    status: str,
    interpreter_reason: str,
    metrics_summary: str,
    run_validity: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Generic contract for open-ended discovery cases:
      - integrity checks
      - score extraction
      - validity gating
    This path is intentionally scoped to open-ended runs only.
    """
    checks: List[Dict[str, Any]] = []
    artifacts = {
        "case_dir": str(case_dir),
        "decision_json": str(case_dir / "decision.json"),
        "run_result_json": str(case_dir / "run_result.json"),
        "log_simpleFoam": str(case_dir / "log.simpleFoam"),
    }

    def add_check(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    decision_ok = (case_dir / "decision.json").is_file()
    run_result_ok = (case_dir / "run_result.json").is_file()
    add_check("decision_present", decision_ok, "decision.json exists" if decision_ok else "decision.json missing")
    add_check("run_result_present", run_result_ok, "run_result.json exists" if run_result_ok else "run_result.json missing")

    rr_status = ""
    if run_result_ok:
        rr = _read_json(case_dir / "run_result.json", {})
        rr_status = str(rr.get("status", "")).strip().lower()
    add_check(
        "run_status_success",
        (rr_status == "success") if rr_status else True,  # do not fail hard if field unavailable
        f"run_result.status={rr_status or '(missing)'}",
    )

    requested_model = _parse_requested_model(case_dir)
    selected_model = _parse_selected_model_from_log(case_dir)
    if requested_model and selected_model:
        add_check(
            "model_loaded_matches_request",
            requested_model == selected_model,
            f"requested={requested_model}, selected={selected_model}",
        )
    else:
        add_check(
            "model_loaded_matches_request",
            True,
            f"requested={requested_model or '(n/a)'}, selected={selected_model or '(n/a)'}",
        )

    extracted = _extract_error_metrics(metrics_summary or "")
    primary = _choose_primary_score(extracted)
    add_check(
        "metrics_extractable",
        primary is not None,
        f"found {len(extracted)} objective-like values",
    )

    # `run_result.json` is written by `foam_run_simple.py` (used by
    # runtime/experiment paths). Class-derivation iterations run their solver
    # via the agentic loop's own bash and don't produce that file in this
    # location, so the gate trips on a *file-presence technicality* even
    # when the simulation actually ran (model loaded, metric extracted).
    # Treat it as advisory: still recorded in integrity_checks for
    # transparency, but not part of the validity decision. The two
    # substantive gates remain authoritative:
    #   - decision_present  (interpret produced something, even fallback)
    #   - run_status_success
    #   - model_loaded_matches_request
    #   - metrics_extractable
    _ADVISORY = {"run_result_present"}
    passed = all(c["ok"] for c in checks if c["name"] not in _ADVISORY)
    # Run-validity gate is authoritative: a RUN_INVALID iteration is never a
    # valid case, regardless of file-presence checks.
    if isinstance(run_validity, dict) and run_validity.get("status") == "RUN_INVALID":
        add_check(
            "run_validity_gate",
            False,
            f"RUN_INVALID: {run_validity.get('reason', '')[:240]}",
        )
        passed = False
    assessment = {
        "status": status,
        "interpreter_reason": interpreter_reason,
        "integrity_checks": checks,
        "valid": passed,
        "score": primary,
        "all_extracted_metrics": extracted,
        "artifacts": artifacts,
        "run_validity": run_validity or {},
    }
    return assessment


def _resolve_objective_contract(
    *,
    disc_dir: Path,
    starter_understanding: Dict[str, Any],
    starter_dir: Optional[Path],
    repo_root: Path,
) -> Dict[str, Any]:
    """
    Generic open-ended objective/evaluator binding:
    - bind to declared reference quantities/files when available
    - lock to a comparator script if one is discoverable
    """
    contract_path = disc_dir / "objective_contract.json"
    existing = _read_json(contract_path, {})
    if isinstance(existing, dict) and existing.get("version") == 1:
        # Only a contract that actually resolved something is worth keeping.
        # An empty one gets written whenever setup runs before
        # read_starter_folder's record is complete -- and because the cache
        # key was the version number alone, that emptiness then survived every
        # later attempt in the run, however good the starter understanding had
        # since become. Downstream, `if ... and ref_files:` silently skips
        # computing the baseline vector, so baseline_score stays null and the
        # search can never gate. Both ph_glm_20260902_2340 and
        # ph_gemini38_20260902_2349 died exactly there, hours apart, on
        # contracts stamped with the fingerprint of an empty topic and an
        # empty file list.
        #
        # Re-derive instead. If the starter understanding still declares
        # nothing, the rebuild writes the same empty contract and nothing is
        # lost but a few milliseconds.
        if existing.get("reference_files"):
            return existing

    ref_info = starter_understanding.get("reference_data", {}) if isinstance(starter_understanding, dict) else {}
    quantities = [str(q).strip() for q in (ref_info.get("quantities", []) or []) if str(q).strip()]
    declared_files = [str(f).strip() for f in (ref_info.get("files", []) or []) if str(f).strip()]

    # Reference-file lookup may use repo_root for declared paths that are
    # explicitly listed (those are user-asserted and safe). Comparator-script
    # auto-detection is restricted to the starter dir below to avoid
    # cross-starter contamination (e.g. binding a periodic-hill Cf comparator
    # to a multiphase droplet study just because both .py files exist somewhere
    # in the repo).
    ref_search_roots: List[Path] = []
    if starter_dir and starter_dir.is_dir():
        ref_search_roots.append(starter_dir.resolve())
        # The starter folder a study is pointed at is usually one case inside a
        # bundle, with the shared reference data a sibling of it -- and a
        # declared path is then written relative to the bundle, not the case.
        # Without this root, "reference_data/ref.csv" resolves under the case
        # and under the repo, misses at both, and the contract comes out with
        # no reference files at all: no baseline vector, baseline_score null,
        # search unable to gate. Observed on ph_glm_20260902_2340, which
        # declared exactly that path while ph_codex_20260902_1806 happened to
        # declare absolute ones and worked. The comparator search below
        # already walks this parent; the reference lookup did not.
        parent = starter_dir.resolve().parent
        if parent.is_dir() and parent not in ref_search_roots:
            ref_search_roots.append(parent)
    ref_search_roots.append(repo_root.resolve())

    ref_files_abs: List[str] = []
    for rel in declared_files:
        rp = Path(rel)
        cands = [rp] if rp.is_absolute() else [(root / rp) for root in ref_search_roots]
        for c in cands:
            if c.is_file():
                s = str(c.resolve())
                if s not in ref_files_abs:
                    ref_files_abs.append(s)

    # LLM-based content classification of all .py files in the starter.
    # No naming-convention assumptions: the LLM reads each script's content
    # and decides which (if any) is the comparator for THIS topic. Result is
    # cached at <disc_dir>/comparator_classification.json so other sites can
    # reuse it without re-spending tokens. Anti-contamination is enforced by
    # comparator_classifier itself: only paths INSIDE starter_dir are ever
    # returned, regardless of what the LLM picks.
    preferred: Optional[Path] = None
    try:
        _scripts_dir = Path(__file__).resolve().parent
        if str(_scripts_dir) not in sys.path:
            sys.path.insert(0, str(_scripts_dir))
        from comparator_classifier import classify_starter_scripts as _classify_scripts  # type: ignore
        classification = _classify_scripts(
            starter_dir=starter_dir,
            topic=str(starter_understanding.get("topic", "")),
            quantities=quantities,
            cache_path=disc_dir / "comparator_classification.json",
        )
        cp = classification.get("comparator_path")
        if cp:
            cp_path = Path(cp)
            if cp_path.is_file():
                preferred = cp_path
    except Exception:
        # Classifier unavailable (e.g., no LLM creds) — leave preferred=None.
        # The Phase 1 metric-author path (bound_comparators.json) covers
        # scoring downstream, so this is a graceful degradation.
        preferred = None

    contract = {
        "version": 1,
        "objective_quantities": quantities,
        "reference_files": ref_files_abs,
        "comparator_script": str(preferred) if preferred else "",
        "created_from": {
            "topic_hash": _text_fingerprint(str(starter_understanding.get("topic", ""))),
            "ref_desc_hash": _text_fingerprint(str(ref_info.get("description", ""))),
            "declared_files_hash": _text_fingerprint(json.dumps(declared_files, ensure_ascii=False)),
        },
    }
    _write_json(contract_path, contract)
    return contract


# Generic token regex for any error/agreement metric the comparator might
# print. Covers lower-is-better (rmse, l2, mae, mse, error, deviation) and
# higher-is-better (correlation, r2, agreement, accuracy). Kept broad on
# purpose — file-scan only requires presence, not classification.
_METRIC_TOKEN_RE = re.compile(
    r"(?i)\b(rmse|rms[\s_-]*error|l[12](?:[\s_-]*error)?|mae|mse|"
    r"max[\s_-]*error|deviation|residual|"
    r"correlation|pearson|spearman|r\s*\^?\s*2|r2|agreement|accuracy)\b"
)


def _run_bound_comparator(case_dir: Path, objective_contract: Dict[str, Any]) -> Optional[str]:
    """
    Run the bound comparator script (e.g. compare_exactmatch_cf.py) and
    extract its result text. Robust to:
      - empty objective_contract.reference_files (auto-detect ref CSV next to script)
      - comparator scripts that write files but produce empty stdout
      - comparator scripts that fail returncode but still leave a partial output
      - missing post-process fields at latestTime (auto-recovered via
        OpenFOAM `<app> -postProcess -latestTime` BEFORE comparator runs)
    Always tries to surface RMSE-like text from disk if stdout doesn't have it.
    """
    # Defensive auto-recovery: if the case has function objects defined but
    # their outputs aren't at the latest time (e.g. case ran with
    # -noFunctionObjects, or function objects crashed mid-run), trigger
    # OpenFOAM's own postProcess mode to derive them from saved primary
    # fields. Generic across applications and function-object types.
    try:
        scripts_dir = Path(__file__).resolve().parent
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        import foam_postprocess as _fpp  # type: ignore
        pp_result = _fpp.ensure_postprocess_fields(case_dir, timeout_s=180)
        if pp_result.get("action") == "postprocess":
            print(f"[OED][postprocess] action={pp_result.get('action')} ok={pp_result.get('ok')} "
                  f"app={pp_result.get('app')} fo_names={pp_result.get('fo_names')}")
    except Exception as exc:
        print(f"[OED][postprocess] non-fatal: {exc}")

    script = str(objective_contract.get("comparator_script", "") or "").strip()
    if not script:
        return None
    sp = Path(script)
    if not sp.is_file():
        return None

    # Resolve reference file. Prefer objective_contract; else auto-detect the
    # first .csv next to the script (most starters ship one).
    refs = [Path(p) for p in (objective_contract.get("reference_files", []) or []) if Path(p).is_file()]
    if not refs:
        for c in sorted(sp.parent.glob("*.csv")):
            if c.stat().st_size > 0:
                refs.append(c)
                break
    ref = refs[0] if refs else None

    out_dir = case_dir / "_oed_comparison_bound"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmds: List[List[str]] = []
    if ref is not None:
        cmds.append([sys.executable, str(sp), "--case", str(case_dir), "--time", "5000",
                     "--reference", str(ref), "--out", str(out_dir)])
        cmds.append([sys.executable, str(sp), "--case", str(case_dir),
                     "--reference", str(ref), "--out", str(out_dir)])
        cmds.append([sys.executable, str(sp), "--case", str(case_dir), "--reference", str(ref)])
    cmds.append([sys.executable, str(sp), "--case", str(case_dir), "--out", str(out_dir)])
    cmds.append([sys.executable, str(sp), "--case", str(case_dir)])

    last_blob = ""
    for cmd in cmds:
        try:
            res = subprocess.run(cmd, cwd=str(sp.parent), capture_output=True,
                                 text=True, timeout=180)
        except Exception:
            continue
        blob = (res.stdout or "") + "\n" + (res.stderr or "")
        last_blob = blob
        if res.returncode == 0 and blob.strip():
            return f"QoI comparison (bound comparator {sp.name}):\n{blob.strip()[:2000]}"

    # Scan the WHOLE case for any text artifacts the comparator wrote. We do
    # not assume a particular output directory name (older code hard-coded
    # `comparison_exactmatch/` which only fits one specific comparator).
    # Generic across any comparator that dumps markdown / csv / json /
    # dat / txt / log files anywhere under case_dir. Skips OpenFOAM time-step
    # field directories and bulky log.* files via size + suffix gates.
    found_blob = ""
    files_scanned = 0
    for p in case_dir.rglob("*"):
        if files_scanned > 400:
            break
        if not p.is_file():
            continue
        if p.suffix.lower() not in (".md", ".txt", ".json", ".csv", ".dat", ".log"):
            continue
        # Skip OpenFOAM time-step field files (large numeric blobs unrelated
        # to comparator output) and parallel-decomposed processor* directories.
        # We check ALL path parts, not just the first, so processor0/100/U is
        # also skipped (the time-step dir lives below processor*).
        rel_parts = p.relative_to(case_dir).parts
        if any(part.replace(".", "").isdigit() for part in rel_parts):
            continue
        if any(part.startswith("processor") for part in rel_parts):
            continue
        try:
            sz = p.stat().st_size
            if sz > 200_000:  # comparator outputs are short; skip huge logs
                continue
            text = p.read_text(encoding="utf-8", errors="replace")[:8000]
        except Exception:
            continue
        files_scanned += 1
        if _METRIC_TOKEN_RE.search(text):
            found_blob += f"\n--- {p.relative_to(case_dir)} ---\n{text[:1500]}"
    if found_blob.strip():
        return f"QoI comparison (bound comparator {sp.name} — files):\n{found_blob[:3000]}"

    if last_blob.strip():
        return f"QoI comparison (bound comparator {sp.name} — partial):\n{last_blob.strip()[:2000]}"
    return None


def _call(cmd: List[str], cwd: Path, timeout: int = 86400, env: Optional[Dict[str, str]] = None) -> Tuple[int, str, str]:
    r = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, timeout=timeout, env=env)
    return r.returncode, r.stdout, r.stderr


# ---------------------------------------------------------------------------
# Run-validity gate + Allrun pre-flight wrappers (lazy-import sibling module).
# Used by every runtime / agentic / code_mod runner before scoring.
# ---------------------------------------------------------------------------

# Module-level toggle for Allrun pre-flight (CLI-controlled). Default ON.
_ALLRUN_PREFLIGHT_ENABLED: bool = True


def _run_validity_gate(
    *,
    case_dir: Path,
    baseline_metrics: Optional[Dict[str, Any]],
    runtime_run_result: Optional[Dict[str, Any]],
    base_case: Optional[Path],
) -> Dict[str, Any]:
    """Thin wrapper around run_validity.gate that won't crash the loop on
    import errors. Returns a degraded RUN_OK result when run_validity is
    unavailable so legacy installs still proceed."""
    try:
        scripts_dir = Path(__file__).resolve().parent
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        import run_validity as _rv  # type: ignore
        return _rv.gate(
            case_dir=case_dir, baseline_metrics=baseline_metrics,
            runtime_run_result=runtime_run_result, base_case=base_case,
        )
    except Exception as exc:
        return {
            "valid": True, "status": "RUN_OK",
            "max_time": 0.0, "baseline_final_time": None,
            "min_required_time": 0.0,
            "reason": f"run_validity unavailable ({exc}); gate skipped.",
            "diagnostic_bundle_path": "",
        }


def _build_run_invalid_override(history: List[Dict[str, Any]]) -> str:
    """If the most recent real-action history entry is RUN_INVALID, build a
    CRITICAL OVERRIDE prompt fragment nudging the decision LLM to choose
    `investigate_runtime` (or justify a fresh hypothesis instead)."""
    if not history:
        return ""
    last_real = None
    for h in reversed(history):
        if not isinstance(h, dict):
            continue
        at = h.get("action_type")
        if at in ("code_mod", "experiment", "investigate_runtime"):
            last_real = h
            break
    if not last_real:
        return ""
    if str(last_real.get("status", "")).upper() != "RUN_INVALID":
        return ""
    rv = last_real.get("run_validity") or {}
    bundle_path = rv.get("diagnostic_bundle_path", "") or ""
    allrun_head = ""
    bundle = {}
    if bundle_path:
        try:
            bundle = json.loads(Path(bundle_path).read_text(encoding="utf-8"))
            allrun_head = (bundle.get("allrun_contents", "") or "")[:1500]
        except Exception:
            bundle = {}
    fragment = (
        "\n\nCRITICAL OVERRIDE: Previous iteration was flagged RUN_INVALID by "
        "the run-validity gate (the flow solver did not advance the case to a "
        "meaningful time). You SHOULD choose action_type=\"investigate_runtime\" "
        "this turn to diagnose the failure (target_iteration="
        f"{int(last_real.get('iteration', 0))}, target_case_dir=\""
        f"{last_real.get('case_dir', '')}\"), OR explain in `rationale` why a "
        "fresh hypothesis is warranted instead.\n"
        f"Diagnostic summary: max_time={rv.get('max_time')}; "
        f"baseline_final_time={rv.get('baseline_final_time')}; "
        f"log_simpleFoam_present={bundle.get('log_simpleFoam_present')}; "
        f"reason={(rv.get('reason') or '')[:240]}.\n"
        f"Allrun head:\n```\n{allrun_head}\n```\n"
    )
    return fragment


def _maybe_preflight_allrun(case_dir: Path, repo_root: Path) -> Optional[Dict[str, Any]]:
    """Run LLM-driven Allrun pre-flight if enabled. Returns the verdict dict
    or None when the feature is disabled or run_validity is unavailable."""
    if not _ALLRUN_PREFLIGHT_ENABLED:
        return None
    try:
        scripts_dir = Path(__file__).resolve().parent
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        import run_validity as _rv  # type: ignore
        out = _rv.preflight_allrun(case_dir=case_dir, repo_root=repo_root)
        try:
            (case_dir / "allrun_preflight.json").write_text(
                json.dumps(out, indent=2, default=str), encoding="utf-8"
            )
        except Exception:
            pass
        return out
    except Exception as exc:
        print(f"[OED][preflight] non-fatal: {exc}")
        return None


_HISTORY_PROMPT_MAX_ENTRIES = 12


def _compact_history(history: List[Dict[str, Any]], max_entries: int = _HISTORY_PROMPT_MAX_ENTRIES) -> str:
    """Build a concise text summary of past iterations for the decision LLM.

    Bounded: this string is rebuilt into every decision prompt, and each entry
    can carry up to ~1500 characters of script output. Rendering the whole
    history meant the prompt grew without limit as the search ran, so a long
    study spent an increasing share of every call re-reading its own early,
    least relevant attempts. The most recent ``max_entries`` are what the next
    decision actually turns on; the rest are summarised by count so nothing is
    silently pretended away.
    """
    if not history:
        return "No experiments run yet."
    lines = []
    omitted = len(history) - max_entries
    if omitted > 0:
        history = history[-max_entries:]
        lines.append(
            f"[{omitted} earlier iteration(s) omitted from this prompt for length; "
            f"the full record is in history.json and the archive summary covers "
            f"the best result per family.]"
        )
    for h in history:
        it = h.get("iteration", "?")
        atype = h.get("action_type", "?")
        if atype == "python_script":
            desc = h.get("script_description", "")
            output = h.get("script_output", "")[:1500]
            status = h.get("status", "?")
            lines.append(
                f"Iteration {it} [python_script]: {desc}\n"
                f"  Status: {status}\n"
                f"  Output:\n{output}"
            )
        else:
            desc = h.get("model_description", "")
            status = h.get("status", "?")
            reason = h.get("interpreter_reason", "")[:300]
            metrics = h.get("metrics_summary", "")
            formula = h.get("formula", "")[:200]
            params = json.dumps(h.get("parameters", {}), ensure_ascii=False)[:120]
            mcat = h.get("modification_category") or ""
            rae = h.get("runtime_apply_error") or ""
            ceh = h.get("compile_error_hint") or ""
            extras = ""
            if mcat:
                extras += f"\n  Category: {mcat}"
            if rae:
                extras += f"\n  Runtime apply error: {rae[:300]}"
            if ceh:
                extras += f"\n  Compile error: {str(ceh)[:300]}"

            # Explicit numeric score + delta-vs-baseline so the planner LLM
            # doesn't have to re-parse prose. Generic across QoIs and metric
            # directions; only included when the iteration produced a score.
            score_obj = h.get("score") if isinstance(h.get("score"), dict) else None
            if score_obj is not None and score_obj.get("value") is not None:
                metric_name = score_obj.get("metric", "?")
                metric_val = score_obj.get("value")
                metric_dir = score_obj.get("direction", "min")
                bs = h.get("baseline_score")
                delta = h.get("score_delta_vs_baseline")
                derived_flag = " [derived-from-score]" if h.get("status_derived_from_score") else ""
                valid_flag = "" if h.get("valid_case", True) else " [integrity:invalid]"
                if bs is not None and isinstance(metric_val, (int, float)):
                    pct = ((metric_val - bs) / bs * 100.0) if bs not in (0, None) else None
                    pct_str = f", Δ%={pct:+.2f}%" if pct is not None else ""
                    if metric_dir == "max":
                        beat = metric_val > bs
                    else:
                        beat = metric_val < bs
                    beat_word = "BEATS" if beat else "does NOT beat"
                    extras += (
                        f"\n  Score: {metric_name}={metric_val} (dir={metric_dir}) — "
                        f"baseline={bs} | Δ={delta}{pct_str} | {beat_word} baseline{derived_flag}{valid_flag}"
                    )
                else:
                    extras += f"\n  Score: {metric_name}={metric_val} (dir={metric_dir}){derived_flag}{valid_flag}"

            lines.append(
                f"Iteration {it} [{atype}]: {desc}\n"
                f"  Formula/spec: {formula}\n"
                f"  Parameters: {params}\n"
                f"  Result: {status} — {reason}{extras}\n"
                f"  Metrics: {metrics}"
            )
    return "\n\n".join(lines)


def _extract_compile_error_hint(apply_result: Dict[str, Any], fallback_err: str = "") -> str:
    """Extract a concise first-fatal compile hint from code_mod_apply_result."""
    logs = apply_result.get("compile_logs", []) if isinstance(apply_result, dict) else []
    if isinstance(logs, list):
        for entry in logs:
            if not isinstance(entry, dict):
                continue
            stderr = str(entry.get("stderr", "") or "")
            stdout = str(entry.get("stdout", "") or "")
            txt = f"{stderr}\n{stdout}"
            for line in txt.splitlines():
                ls = line.strip()
                if not ls:
                    continue
                low = ls.lower()
                if "fatal error:" in low or " error:" in low:
                    return ls[:500]
    return (fallback_err or "")[-500:]


def _recent_successful_code_mod_examples(history: List[Dict[str, Any]], limit: int = 2) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for h in reversed(history):
        if h.get("action_type") != "code_mod":
            continue
        if h.get("status") not in ("PROCEED", "RERUN", "UNKNOWN"):
            continue
        formula = str(h.get("formula", "") or "").strip()
        if not formula:
            continue
        out.append(
            {
                "iteration": str(h.get("iteration", "")),
                "model_description": str(h.get("model_description", "")),
                "formula_excerpt": formula[:1200],
            }
        )
        if len(out) >= limit:
            break
    return list(reversed(out))


def _llm_refine_code_mod_spec(
    *,
    action: Dict[str, Any],
    topic: str,
    starter_understanding: Dict[str, Any],
    history: List[Dict[str, Any]],
    repo_root: Path,
    compile_error_hint: str = "",
) -> Dict[str, Any]:
    """
    Open-ended-only compileability gate / build-engineer retry:
      - First call (no compile_error_hint): rewrite oversized/full-file specs into
        minimal equation-delta specs.
      - Subsequent calls (compile_error_hint set): switch to BUILD-ENGINEER mode —
        the LLM must FIX THE IMPLEMENTATION ONLY, not change the physical mechanism.
    """
    # Skip refinement for runtime / dict_only categories — those specs are
    # structured JSON (not free-form C++) and don't need compactification.
    mcat = str(action.get("modification_category") or "").strip().lower()
    if mcat in ("runtime_source", "runtime_bc", "runtime_field", "dict_only"):
        return action

    try:
        _bootstrap(repo_root)
        from langchain_core.messages import HumanMessage, SystemMessage  # type: ignore
        from cfd_langgraph.config import get_settings  # type: ignore
        from cfd_langgraph.llm.factory import create_langchain_llm  # type: ignore
        from cfd_langgraph.utils import strip_json_fences  # type: ignore

        flow = starter_understanding.get("flow_parameters", {}) if isinstance(starter_understanding, dict) else {}
        ref = starter_understanding.get("reference_data", {}) if isinstance(starter_understanding, dict) else {}
        examples = _recent_successful_code_mod_examples(history, limit=2)
        raw_spec = str(action.get("formula_or_modification", "") or "")
        is_build_engineer = bool(compile_error_hint and compile_error_hint.strip())

        # Tier-classify the compile error so the LLM gets focused coaching
        # rather than a 1500-line stderr blob.
        tier_block = ""
        if is_build_engineer:
            try:
                _scripts_dir = repo_root / "scripts"
                if str(_scripts_dir) not in sys.path:
                    sys.path.insert(0, str(_scripts_dir))
                import compile_error_classifier as _cec  # type: ignore
                tier = _cec.classify(compile_error_hint, "")
                tier_block = (
                    f"\nBUILD ERROR TIER: {tier['tier']} ({tier['tier_label']})\n"
                    f"COACHING:\n{tier['coaching']}\n"
                )
                if tier.get("missing_headers"):
                    tier_block += f"missing_headers: {tier['missing_headers']}\n"
                if tier.get("undeclared"):
                    tier_block += f"undeclared_identifiers: {tier['undeclared'][:6]}\n"
                if tier.get("missing_libs"):
                    tier_block += f"missing_libs: {tier['missing_libs']}\n"
                if tier.get("make_files_csources"):
                    tier_block += f"missing_csources_in_Make_files: {tier['make_files_csources']}\n"
            except Exception:
                pass

        llm = create_langchain_llm(model=get_settings().model, temperature=0.0)
        if is_build_engineer:
            sys_msg = (
                "You are a CFD BUILD ENGINEER, not a researcher.\n"
                "A code_mod spec for a hypothesis FAILED to compile. Your job is to\n"
                "FIX THE IMPLEMENTATION so it compiles, while PRESERVING the\n"
                "physical mechanism the hypothesis encodes.\n\n"
                "STRICT RULES:\n"
                "- DO NOT change the physical mechanism, the equation modification,\n"
                "  the inserted term, or the underlying idea. The next experiment\n"
                "  must still test the SAME hypothesis.\n"
                "- DO change: identifier names, header includes, type wrapping\n"
                "  (volScalarField vs dimensionedScalar), tmp<>/.ref() handling,\n"
                "  fvm:: vs fvc:: choice, Make/files target name, Make/options\n"
                "  EXE_INC paths, LIB_LIBS list — whatever the build error says\n"
                "  is broken.\n"
                "- Use the COACHING block below for tier-specific guidance.\n"
                "- Apply the SMALLEST possible fix. Do not refactor.\n"
                "- Output STRICT JSON only, no markdown.\n\n"
                "Output JSON keys:\n"
                "{\n"
                '  "model_description": "(unchanged unless renamed identifier)",\n'
                '  "formula_or_modification": "compact delta spec with fix applied",\n'
                '  "fix_summary": "1-2 lines: what was broken, what you changed"\n'
                "}"
                + tier_block
            )
        else:
            sys_msg = (
                "You are a CFD code-mod compileability reviewer.\n"
                "Rewrite the incoming code_mod spec into a compact, compile-focused DELTA spec.\n\n"
                "REQUIREMENTS:\n"
                "- Keep novelty in PHYSICS TERM only; keep class scaffold close to prior successful pattern.\n"
                "- Return equation-level and insertion-site instructions, NOT full source files.\n"
                "- Do NOT emit complete .H/.C/Make file templates.\n"
                "- Keep formula_or_modification under 2500 chars.\n"
                "- Output JSON only, no markdown.\n\n"
                "Output STRICT JSON with keys:\n"
                "{\n"
                '  "model_description": "...",\n'
                '  "formula_or_modification": "compact delta spec",\n'
                '  "why_compileable": "1-3 lines"\n'
                "}"
            )
        user_msg = (
            f"TOPIC: {topic}\n"
            f"FLOW: {json.dumps(flow, ensure_ascii=False)}\n"
            f"REFERENCE: {json.dumps(ref, ensure_ascii=False)[:1500]}\n"
            f"compile_error_hint: {compile_error_hint or '(none)'}\n\n"
            f"Recent successful code_mod examples:\n{json.dumps(examples, ensure_ascii=False)[:4000]}\n\n"
            f"Incoming action model_description: {action.get('model_description', '')}\n"
            f"Incoming formula_or_modification:\n{raw_spec[:12000]}"
        )

        raw = llm.invoke([SystemMessage(content=sys_msg), HumanMessage(content=user_msg)])
        txt = strip_json_fences(str(getattr(raw, "content", raw)).strip())
        s, e = txt.find("{"), txt.rfind("}")
        if s != -1 and e != -1 and e > s:
            txt = txt[s : e + 1]
        obj = json.loads(_fix_json_string_literals(txt))
        if not isinstance(obj, dict):
            return action

        refined = dict(action)
        md = str(obj.get("model_description", "") or "").strip()
        fm = str(obj.get("formula_or_modification", "") or "").strip()
        if md:
            refined["model_description"] = md[:220]
        if fm:
            refined["formula_or_modification"] = fm[:2500]
        return refined
    except Exception:
        return action


# ---------------------------------------------------------------------------
# Decision LLM
# ---------------------------------------------------------------------------

def _llm_decide_next_action(
    *,
    topic: str,
    starter_understanding: Dict[str, Any],
    history: List[Dict[str, Any]],
    budget_remaining: int,
    budget_total: int,
    compiled_models: List[Dict[str, Any]],
    repo_root: Path,
    force_real_action: bool = False,
    allow_stop: bool = True,
    blocked_compile_hints: Optional[List[str]] = None,
    recon_context: Optional[Dict[str, Any]] = None,
    force_class_derivation: bool = False,
    baseline_metrics: Optional[Dict[str, Any]] = None,
    lit_path: Optional[Path] = None,
    extension_context: str = "",
) -> Dict[str, Any]:
    """
    Ask the LLM what to do next based on full history + reference data.

    Returns dict with keys:
      action_type: "code_mod" | "experiment" | "stop"
      rationale: str
      model_description: str           (for code_mod)
      formula_or_modification: str     (for code_mod — full C++ spec text)
      model_name_to_reuse: str         (for experiment — name of compiled model)
      parameters: dict                 (for experiment — parameter overrides)
      stop_reason: str                 (for stop)
    """
    _bootstrap(repo_root)
    from langchain_core.messages import HumanMessage, SystemMessage  # type: ignore
    from cfd_langgraph.config import get_settings  # type: ignore
    from cfd_langgraph.llm.factory import create_langchain_llm  # type: ignore
    from cfd_langgraph.utils import strip_json_fences  # type: ignore

    llm = create_langchain_llm(model=get_settings().model, temperature=0.0)

    fp = starter_understanding.get("flow_parameters", {}) or {}
    ref = starter_understanding.get("reference_data", {}) or {}
    base_case = starter_understanding.get("base_case_path", "")

    compiled_summary = ""
    if compiled_models:
        compiled_summary = "Previously compiled models available for parameter experiments:\n" + "\n".join(
            f"  - {m['name']}: {m['description']}" for m in compiled_models
        )

    # Infer a brief one-line study goal from the topic rather than hardcoding model type
    sys_msg = (
        "You are an expert CFD scientist conducting open-ended, hypothesis-driven discovery.\n"
        "Your goal: iteratively improve CFD simulation accuracy for the given topic "
        "by proposing and testing model modifications or parameter changes, guided by "
        "comparison against the provided reference data.\n"
        "You are not limited to turbulence models — the modification can be a viscosity model, "
        "a flux scheme, a source term, boundary condition, or any physical model component "
        "relevant to the topic.\n"
        "You propose ONE action at a time, guided by results of all previous attempts.\n\n"
        "ACTION TYPES:\n"
        "  python_script — write and run a Python script for any exploratory calculation,\n"
        "               data analysis, curve fitting, visualization, or sanity check.\n"
        "               Use this like a scientist's scratchpad: inspect reference data,\n"
        "               analyse postProcessing output, fit a curve, compare two cases, etc.\n"
        "               Costs 0 budget units. Output is captured and added to history.\n"
        "               The script runs with access to all files in RUN_DIR and STARTER_DIR.\n"
        "  code_mod   — propose a model/scheme/BC/transport modification. Sub-categories\n"
        "               via the `modification_category` field (see below). Cost depends on\n"
        "               category: runtime/dict_only = 1 unit (fast, no compile);\n"
        "               class_derivation = 2 units (compiled C++ library).\n"
        "  experiment — re-run a previously compiled model with different parameter values.\n"
        "               Costs 1 budget unit. Use for parameter sensitivity on a promising model.\n"
        "  investigate_runtime — investigate a previous RUN_INVALID iteration. Reads the\n"
        "               diagnostic bundle, Allrun, logs, original action JSON; classifies\n"
        "               root cause; if the bug is in the harness (Allrun missing solver\n"
        "               invocation, etc.), patches and reruns; if the bug is in the model,\n"
        "               returns REVISE so the next planner iteration produces a corrected\n"
        "               code_mod. Costs 1 budget unit.\n"
        "               Required fields: target_iteration (int, optional — defaults to most\n"
        "               recent RUN_INVALID), target_case_dir (optional).\n"
        "  stop       — no more promising directions or budget too low to be useful.\n\n"
        "MODIFICATION CATEGORIES (set `modification_category` on every code_mod):\n"
        "  runtime_source   — adds a source term to ANY transport equation via\n"
        "                     codedFvOption (scalar/vector/tensor). OpenFOAM JIT-\n"
        "                     compiles the snippet at solver start. NO wmake. Use\n"
        "                     for: SA/k/omega/epsilon/nut source terms; momentum\n"
        "                     forcing on U; energy/temperature sources; scalar\n"
        "                     transport sources; mass-transfer terms in multiphase;\n"
        "                     any added/removed term in any transport equation.\n"
        "  runtime_bc       — replaces a BC with codedFixedValue / codedMixed.\n"
        "                     Use for: profile inlets, time/space-dependent BCs,\n"
        "                     sensor-dependent walls, custom wall functions.\n"
        "  runtime_field    — adds a derived field / sensor / monitor via coded\n"
        "                     functionObject. Use for: derived QoIs, in-loop\n"
        "                     diagnostics, conditional probes.\n"
        "  dict_only        — pure OpenFOAM dictionary edits, no code at all.\n"
        "                     Use for: numerical scheme swaps (fvSchemes,\n"
        "                     fvSolution); switching between built-in models in\n"
        "                     turbulenceProperties / momentumTransport / thermo;\n"
        "                     transport-property dictionary edits\n"
        "                     (transportProperties, viscosity laws like Bingham/\n"
        "                     PowerLaw/Carreau when expressible as a built-in dict);\n"
        "                     thermal property changes; control parameter changes.\n"
        "  class_derivation — Derive a new C++ class from an OpenFOAM parent\n"
        "                     class (e.g. SpalartAllmaras, kEpsilon, viscosityModel,\n"
        "                     fvPatchField, fvOption). RUN BY AN AGENTIC CODER\n"
        "                     with full shell access: it reads $WM_PROJECT_DIR/src\n"
        "                     to study the parent class, writes\n"
        "                     <case>/customModels/<ClassName>/{ClassName.H,.C,Make/...},\n"
        "                     runs `wmake libso`, reads stderr, edits, retries —\n"
        "                     all in one session. Use this for STRUCTURAL changes\n"
        "                     (replace fv1, replace Stilda, replace destruction\n"
        "                     formulation, introduce new state variables, novel\n"
        "                     correction architectures). The agent has freedom to\n"
        "                     iterate on compile errors — give it a clear physics\n"
        "                     hypothesis, not specific C++. Wall time ~5-15 min.\n\n"
        "WHEN TO USE WHICH CATEGORY:\n"
        "  - 'Add a new term to the existing transport equation' → runtime_source\n"
        "    (codedFvOption / codedFvModel; no compile, fastest).\n"
        "  - 'Replace the boundary value with a custom expression' → runtime_bc.\n"
        "  - 'Swap a numerical scheme / change solver tolerance' → dict_only.\n"
        "  - 'Change the STRUCTURE of an equation, replace a model component,\n"
        "    introduce new state, or write a novel SA correction architecture\n"
        "    (e.g. sensor-gated production/destruction multipliers, novel\n"
        "    closure)' → class_derivation. The agent will compile it.\n"
        "  - True novelty (publishable model variants) typically requires\n"
        "    class_derivation. Don't avoid it just because it's slower —\n"
        "    the agentic runner makes it RELIABLE now.\n\n"
        "REQUIRED FIELDS by category:\n"
        "  runtime_source: provide a `runtime_source` object with:\n"
        "      target_field, value_type ('scalar'|'vector'|'tensor'),\n"
        "      code_add_sup (C++ body modifying fvMatrix& eqn — e.g.\n"
        "      \"eqn += fvm::Sp(scalar(0.1), eqn.psi());\"),\n"
        "      code_include (optional headers), coefficients (optional named scalars),\n"
        "      selection_mode ('all'|'cellSet'|'cellZone', default 'all').\n"
        "      OPENFOAM 10 (org) NOTE: the runtime applier emits the modern\n"
        "      `type coded; field <name>;` form into constant/fvModels (NOT the\n"
        "      ESI `scalarCodedSource; fields (<name>)` form into fvOptions).\n"
        "      Available hooks in code_add_sup body: `eqn` is the fvMatrix,\n"
        "      `eqn.psi()` is the field, `mesh()` is the fvMesh, `coeffs()` is\n"
        "      the dictionary for named scalars (use\n"
        "      `coeffs().lookupOrDefault<scalar>(\"name\", 1.0)`). Use `fvm::Sp`\n"
        "      for implicit sinks/sources, `fvc::div`/`fvc::grad` for explicit\n"
        "      derivatives. Lookup other fields with\n"
        "      `mesh().lookupObject<volScalarField>(\"U\")` etc.\n"
        "  runtime_bc: provide a `runtime_bc` object with:\n"
        "      field_file ('0/<field>'), patch_name, bc_type ('codedFixedValue' |\n"
        "      'codedMixed'), code (C++ body), value_default (e.g. 'uniform 0').\n"
        "  runtime_field: provide a `runtime_field` object with:\n"
        "      name, code_execute (C++ body), code_include (optional).\n"
        "  dict_only: provide a `dict_only.edits` array, each element either:\n"
        "      {target: '<rel-path>', unified_diff: '<patch>'}, OR\n"
        "      {target: '<rel-path>', key_path: 'a/b', new_value: '<value>'}.\n"
        "  class_derivation: provide `formula_or_modification` (compact equation-\n"
        "      delta spec — parent class, insertion site, term, coefficient names).\n\n"
        "RULES:\n"
        "  - Use python_script freely to understand data before committing to a code_mod.\n"
        "  - Think physically: what mechanism is failing? What term or formulation addresses it?\n"
        "  - Do not repeat a formula/approach that already failed.\n"
        "  - For code_mod, provide a COMPACT equation-delta spec in formula_or_modification:\n"
        "    objective, parent model, exact insertion site/function, equation change, coefficient names/defaults,\n"
        "    and required dictionary activation updates. Do NOT emit complete .H/.C/Make file templates.\n"
        "  - For python_script, provide complete runnable Python code in script_code.\n"
        "    The script can use numpy, pandas, matplotlib, pathlib, json, scipy.\n"
        "    Print results clearly — stdout is captured and added to history.\n"
        "    When inspecting files (C/H/py/OpenFOAM dictionaries), read FULL file content first\n"
        "    before concluding. Do not inspect only the first N lines or header unless the file\n"
        "    is extremely large; in that case, read all chunks sequentially and summarize findings.\n"
        "  - HARD SAFETY RULE: python_script MUST NOT write to or modify any file inside\n"
        "    the OpenFOAM installation directory (WM_PROJECT_DIR). Read-only access is fine.\n"
        "    All file writes must go to RUN_DIR or DISCOVERY_DIR only.\n"
        "  - HARD SAFETY RULE: code_mod changes are compiled as case-local libraries only.\n"
        "    Never instruct the builder to edit files under WM_PROJECT_DIR.\n"
        "  - Stop only if results are satisfactorily validated OR no real action is affordable.\n"
        "Return STRICT JSON only. No markdown. No commentary outside the JSON."
        + ("\n\nCRITICAL OVERRIDE: You have already run multiple python_script exploration steps. "
           "You MUST NOT choose python_script this turn. Choose code_mod, experiment, or stop."
           if force_real_action else "")
        + ("\n\nCRITICAL OVERRIDE: You MUST NOT choose stop this turn. Choose code_mod or experiment only."
           if not allow_stop else "")
        + ("\n\nCRITICAL OVERRIDE: The last 2+ code_mods used cheap categories "
           "(runtime_source / runtime_bc / runtime_field / dict_only) without "
           "achieving PROCEED. Cheap categories cannot express structural model "
           "novelty. This turn, if you choose code_mod, the modification_category "
           "MUST be \"class_derivation\" — propose a structurally novel correction "
           "(e.g. replace fv1 / Stilda / a destruction term, or introduce a new "
           "sensor-channel architecture). The agentic compiler will iterate on "
           "wmake errors for you. Do NOT pick a runtime category this turn."
           if force_class_derivation else "")
        + _build_run_invalid_override(history)
    )

    # Coded-snippet grounding: list real registered fields on the base case +
    # OpenFOAM dimensioned-arithmetic rules + wall-distance gotcha. Prevents
    # the runtime_source / runtime_bc / runtime_field paths from emitting C++
    # bodies that crash at solver startup with `lookupObject` errors or
    # dimension-mismatch FOAM FATAL aborts. Generic — applies to any coded
    # snippet category.
    runtime_grounding = ""
    try:
        from openfoam_grounding import build_runtime_snippet_grounding  # type: ignore
        bcp = (base_case or "").strip()
        runtime_grounding = build_runtime_snippet_grounding(Path(bcp) if bcp else None)
    except Exception:
        runtime_grounding = ""

    # Literature digest: distill the topic-relevant Semantic Scholar entries
    # (lit.json from the literature stage) into a compact title + abstract
    # block so the planner LLM is aware of model-modification ideas grounded
    # in published work for THIS topic. Generic — applies to any OED topic
    # whose lit.json exists. Capped to keep prompt size bounded.
    literature_digest = ""
    try:
        if lit_path is not None and Path(lit_path).is_file():
            lit_entries = _read_json(Path(lit_path), [])
            if isinstance(lit_entries, list) and lit_entries:
                MAX_PAPERS = 8
                ABS_CHARS = 280
                # Sort by citationCount desc to surface seminal work first.
                def _cite(e: Any) -> int:
                    try:
                        return int(e.get("citationCount", 0)) if isinstance(e, dict) else 0
                    except Exception:
                        return 0
                ranked = sorted([e for e in lit_entries if isinstance(e, dict)], key=_cite, reverse=True)[:MAX_PAPERS]
                bullets: List[str] = []
                for i, e in enumerate(ranked, 1):
                    title = str(e.get("title", "(no title)") or "(no title)").strip()
                    year = e.get("year") or ""
                    venue = str(e.get("venue", "") or "").strip()
                    cites = _cite(e)
                    abstract = str(e.get("abstract") or "")[:ABS_CHARS].strip()
                    if not abstract:
                        abstract = "(abstract unavailable)"
                    head = f"  [{i}] {title} ({year}, {venue}, {cites} cites)"
                    bullets.append(head + "\n      abstract: " + abstract.replace("\n", " "))
                literature_digest = (
                    "RELEVANT LITERATURE (top by citation count from this topic's Semantic Scholar search):\n"
                    "Use these as inspiration for what KIND of modification might be effective. They\n"
                    "describe physics ideas and prior model variants for this exact problem. Do NOT\n"
                    "literally re-implement a published model — propose novel directions, but informed\n"
                    "by what the field has tried.\n"
                    + "\n".join(bullets)
                    + "\n\n"
                )
    except Exception:
        literature_digest = ""

    # Recon context: verified OpenFOAM source paths for this installation. Fed
    # into the decision prompt so the hypothesis-writer doesn't bake hallucinated
    # include directories / class names / header filenames into `formula_or_modification`.
    # Generic — relevant for any code_mod mode.
    recon_block = ""
    if isinstance(recon_context, dict) and (
        recon_context.get("selected_files") or recon_context.get("verified_include_paths")
    ):
        vpaths = [p.get("make_options_form", "") for p in recon_context.get("verified_include_paths", [])]
        sfiles = [f.get("rel", "") for f in recon_context.get("selected_files", []) if isinstance(f, dict)]
        classes = []
        for cs in recon_context.get("class_signatures", []):
            if isinstance(cs, dict) and cs.get("class") and cs.get("defined_in"):
                classes.append(f"{cs['class']} (in {cs['defined_in']})")
        recon_block = (
            "OPENFOAM SOURCE RECON (verified against the installation actually present on this machine):\n"
            "  verified_include_paths (use these in any Make/options spec you write — do not invent):\n"
            + "\n".join(f"    {p}" for p in vpaths if p) + "\n"
            "  selected_header_files (confirmed to exist; use these names in #include lines and parent-class citations):\n"
            + "\n".join(f"    {f}" for f in sfiles if f) + "\n"
            "  class_signatures_found:\n"
            + "\n".join(f"    {c}" for c in classes) + "\n"
            "IMPORTANT: Your `formula_or_modification` spec MUST align with these verified paths and "
            "class names. Do NOT reference headers, directories, or classes that are not listed here "
            "unless they are well-known OpenFOAM core primitives (e.g. fvMatrix, volScalarField). "
            "This installation may differ from ESI / foam-extend / OpenFOAM-v24xx layouts you have "
            "seen in training data.\n\n"
        )

    # Existing persisted tool scripts — the decision LLM can reference these
    # (and invoke them inside future python_script actions) rather than
    # rewriting equivalent logic from scratch every time. Lightweight —
    # only the filename+purpose-header is shown.
    tool_lines: List[str] = []
    try:
        tools_dir = starter_understanding.get("__run_dir__")
        # starter_understanding is per-iteration; the real run_dir is resolvable
        # from the first entry in history that has a case_dir we know. Fall
        # back to scanning standard path.
        # Easier: derive from base_case location.
    except Exception:
        pass
    # Re-derive run_dir from starter_understanding.base_case_path if possible
    base_case_path = starter_understanding.get("base_case_path", "")
    _run_dir_candidate = None
    try:
        if base_case_path:
            # base_case is typically {run_dir}/canonical_base_case or starter/...
            p = Path(base_case_path).resolve()
            for anc in [p, *p.parents]:
                if (anc / "state.json").is_file() and (anc / "open_ended_discovery").is_dir():
                    _run_dir_candidate = anc
                    break
    except Exception:
        pass
    if _run_dir_candidate is not None:
        td = _run_dir_candidate / "oed_tools"
        if td.is_dir():
            for tf in sorted(td.glob("*.py")):
                try:
                    head = tf.read_text(encoding="utf-8", errors="ignore").splitlines()[:5]
                    purpose = next((l for l in head if l.startswith("# Purpose:")), "")
                    tool_lines.append(f"  {tf.name}  —  {purpose[len('# Purpose:'):].strip() if purpose else '(no purpose)'}")
                except Exception:
                    continue
    tools_block = ""
    if tool_lines:
        tools_block = (
            "AVAILABLE PERSISTED TOOLS (from previous python_script iterations —\n"
            "you may invoke or read these in future python_script actions instead\n"
            "of rewriting equivalent logic from scratch):\n" + "\n".join(tool_lines) + "\n\n"
        )

    # Baseline-target block: when baseline_setup produced a numeric score, the
    # planner sees the bar to beat. This makes the search goal concrete instead
    # of "make it look better than DNS" (subjective).
    baseline_block = ""
    if isinstance(baseline_metrics, dict) and baseline_metrics:
        ps = baseline_metrics.get("primary_score") if isinstance(baseline_metrics.get("primary_score"), dict) else None
        ps_str = ""
        if ps:
            ps_str = f" {ps.get('metric')}={ps.get('value')} (lower is better)"
        baseline_block = (
            "BASELINE TO BEAT (from baseline_setup stage):\n"
            f"  baseline_model: {baseline_metrics.get('baseline_name','(unknown)')}\n"
            f"  baseline_case_dir: {baseline_metrics.get('baseline_case_dir','(none)')}\n"
            f"  baseline_score:{ps_str}\n"
            "  Goal: produce a variant whose score (same metric) is strictly "
            "less than baseline_score. PROCEED is gated on that.\n\n"
        )

    user_msg = (
        f"TOPIC: {topic}\n\n"
        f"FLOW PARAMETERS: {json.dumps(fp, ensure_ascii=False)}\n"
        f"BASE CASE: {base_case}\n\n"
        f"REFERENCE DATA TARGET:\n"
        f"  Description: {ref.get('description', '')}\n"
        f"  Quantities: {ref.get('quantities', [])}\n"
        f"  Usage: {ref.get('usage_guidance', '')}\n"
        f"  Excerpt:\n{str(ref.get('data_excerpt', ''))[:1500]}\n\n"
        f"{baseline_block}"
        f"BUDGET: {budget_remaining} units remaining of {budget_total} total "
        f"(code_mod runtime/dict_only=1 unit, code_mod class_derivation=2 units, "
        f"experiment=1 unit, python_script=0 units)\n\n"
        f"{recon_block}"
        f"{runtime_grounding}"
        f"{literature_digest}"
        f"{tools_block}"
        f"{compiled_summary}\n\n"
        f"Blocked compile patterns (avoid repeating): {json.dumps(blocked_compile_hints or [], ensure_ascii=False)}\n\n"
        f"HISTORY OF ATTEMPTS:\n{_compact_history(history)}\n\n"
        "Return JSON with this structure:\n"
        "{\n"
        '  "action_type": "code_mod" | "code_mod_batch" | "experiment" | "investigate_runtime" | "python_script" | "stop",\n'
        '  "rationale": "why this action, what physical insight motivates it",\n'
        '  "variant_name": "short slug (≤20 chars, alphanumeric/underscore/dash only) identifying\n'
        '                   this specific variant or script (e.g. \\"SA_RC\\", \\"sepDrive\\",\n'
        '                   \\"check_Cf_extraction\\"). Used for directory naming and log clarity.",\n'
        '  "model_description": "short human-readable name/description of model variant",\n'
        '  "modification_category": "runtime_source"|"runtime_bc"|"runtime_field"|"dict_only"|"class_derivation",\n'
        '  "formula_or_modification": "compact equation-delta spec (REQUIRED for class_derivation; brief summary otherwise)",\n'
        '  "runtime_source": { ...required fields when category=runtime_source... },\n'
        '  "runtime_bc":     { ...required fields when category=runtime_bc... },\n'
        '  "runtime_field":  { ...required fields when category=runtime_field... },\n'
        '  "dict_only":      { "edits": [ ... ] },\n'
        '  "variants": [   // REQUIRED for action_type=code_mod_batch; ignored otherwise\n'
        '      {"variant_name":"v1","model_description":"...","formula_or_modification":"..."},\n'
        '      {"variant_name":"v2","model_description":"...","formula_or_modification":"..."},\n'
        '      ...  (up to 5; budget cost = 2 * N)\n'
        '  ],\n'
        '  "model_name_to_reuse": "name from compiled models list (for experiment only)",\n'
        '  "target_iteration": <int>,                    // for investigate_runtime only — defaults to most recent RUN_INVALID\n'
        '  "target_case_dir": "<absolute path>",         // for investigate_runtime only (optional)\n'
        '  "parameters": {"param_name": value, ...},\n'
        '  "script_code": "complete runnable Python script (for python_script only)",\n'
        '  "script_description": "one line: what this script computes/checks",\n'
        '  "stop_reason": "reason for stopping (for stop only)",\n'
        '  "budget_extension_request": {"units": <int>, "justification": "<short>"}   // optional\n'
        "}\n\n"
        "BUDGET-EXTENSION POLICY:\n"
        "  If you are converging on a promising family but need more runs to confirm,\n"
        "  you may request an extension via `budget_extension_request`. The orchestrator\n"
        "  applies it only when:\n"
        "    (a) at least one PROCEED case already exists in history, AND\n"
        "    (b) the requested units are ≤ 50% of the original total budget.\n"
        "  Otherwise the request is ignored and the normal budget cap holds.\n"
        "  Do not abuse this — use it when you have a concrete, measurable plan."
    )

    # OED extensions (Phase 1/2/3): if the caller built an extension_context
    # (multi-metric vector, diversity constraint, multi-flow score matrix),
    # append it to the user message verbatim. Empty when extensions are off,
    # so existing single-metric behaviour is unchanged.
    if extension_context:
        user_msg = user_msg + "\n\n--- EXTENSIONS CONTEXT ---\n" + extension_context

    raw = llm.invoke([SystemMessage(content=sys_msg), HumanMessage(content=user_msg)])
    txt = str(getattr(raw, "content", raw)).strip()
    cleaned = strip_json_fences(txt)
    s, e = cleaned.find("{"), cleaned.rfind("}")
    if s != -1 and e != -1 and e > s:
        cleaned = cleaned[s: e + 1]
    result = json.loads(_fix_json_string_literals(cleaned))
    if not isinstance(result, dict):
        raise ValueError("Decision LLM returned non-dict")
    return result


# ---------------------------------------------------------------------------
# Python script action
# ---------------------------------------------------------------------------

def _run_python_script_iteration(
    *,
    iteration: int,
    action: Dict[str, Any],
    run_dir: Path,
    repo_root: Path,
    starter_understanding: Dict[str, Any],
    starter_dir: Optional[Path] = None,
    timeline_path: Path,
) -> Dict[str, Any]:
    """Write and execute an LLM-authored Python script; return stdout/stderr."""
    # Include a slug for the script purpose if provided.
    slug = _slugify_variant(action.get("script_description") or action.get("script_name") or "")
    slug_part = f"_{slug}" if slug else ""
    iter_dir = run_dir / f"iter_{iteration:03d}_python_script{slug_part}"
    iter_dir.mkdir(parents=True, exist_ok=True)

    script_code = action.get("script_code", "")
    script_desc = action.get("script_description", f"analysis script iteration {iteration}")

    if not script_code.strip():
        return {"status": "SKIPPED", "output": "No script_code provided.", "script_path": ""}

    # Inject useful path variables and a safety guard at the top.
    # NOTE: `run_dir` here is disc_dir (the open_ended_discovery subdir).
    #       The actual study run dir is its parent.
    study_run_dir = run_dir.parent        # e.g. runs/open_ended_turbuelnce_model
    disc_dir_path = run_dir               # e.g. .../open_ended_discovery
    _eff_starter = starter_dir or Path(starter_understanding.get("starter_dir", "")) or (repo_root / "starter")
    starter_dir_str = str(_eff_starter)
    of_dir = os.environ.get("WM_PROJECT_DIR", "/opt/openfoam")
    preamble = (
        f"# ===== AUTO-INJECTED BY ORCHESTRATOR — USE THESE VARIABLES =====\n"
        f"# All key paths are already defined as pathlib.Path objects below.\n"
        f"# Do NOT redefine them with os.environ or hardcoded strings.\n"
        f"#\n"
        f"#   RUN_DIR       = {study_run_dir}\n"
        f"#   DISCOVERY_DIR = {disc_dir_path}\n"
        f"#   STARTER_DIR   = {starter_dir_str}\n"
        f"#   REPO_ROOT     = {repo_root}\n"
        f"# ================================================================\n"
        f"import sys, pathlib, os\n"
        f"RUN_DIR       = pathlib.Path({str(study_run_dir)!r})\n"
        f"DISCOVERY_DIR = pathlib.Path({str(disc_dir_path)!r})\n"
        f"REPO_ROOT     = pathlib.Path({str(repo_root)!r})\n"
        f"STARTER_DIR   = pathlib.Path({starter_dir_str!r})\n"
        f"_OF_DIR       = pathlib.Path({of_dir!r})\n"
        f"# ================================================================\n\n"
        # Runtime guard: patch open() to block writes to OpenFOAM installation
        f"_orig_open = open\n"
        f"def open(file, mode='r', *a, **kw):\n"
        f"    _f = pathlib.Path(str(file)).resolve()\n"
        f"    if any(m in mode for m in ('w', 'a', 'x')) and _f.is_relative_to(_OF_DIR):\n"
        f"        raise PermissionError(f'Safety guard: scripts may not write to OpenFOAM installation: {{_f}}')\n"
        f"    return _orig_open(file, mode, *a, **kw)\n\n"
    )
    full_script = preamble + script_code
    script_path = iter_dir / "analysis_script.py"
    script_path.write_text(full_script, encoding="utf-8")
    print(f"[OED] Running python_script: {script_desc}")

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True, text=True, timeout=120, cwd=str(repo_root),
        )
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()
        rc = result.returncode
    except subprocess.TimeoutExpired:
        stdout, stderr, rc = "", "Script timed out after 120s", 1

    output = stdout if stdout else "(no output)"
    if stderr and rc != 0:
        output += f"\nSTDERR: {stderr[-500:]}"

    # Save outputs
    (iter_dir / "output.txt").write_text(output, encoding="utf-8")
    if rc == 0 and (iter_dir / "analysis_script.py").exists():
        print(f"[OED] Script output:\n{output[:1000]}")

    # Persist successful python_script outputs as reusable tools in
    # <run_dir>/oed_tools/ so later iterations can `cat` or invoke them
    # instead of the LLM re-generating equivalent logic. Generic — works
    # for any topic or script purpose.
    if rc == 0 and script_code.strip():
        try:
            tool_slug = _slugify_variant(
                action.get("script_name") or action.get("script_description") or f"script_iter_{iteration}"
            ) or f"script_iter_{iteration:03d}"
            tools_dir = run_dir / "oed_tools"
            tools_dir.mkdir(parents=True, exist_ok=True)
            tool_path = tools_dir / f"{tool_slug}.py"
            header = (
                f"# Auto-persisted from OED iter_{iteration:03d}_python_script\n"
                f"# Purpose: {script_desc[:200]}\n"
                f"# Reusable: later OED iterations may invoke this script\n"
                f"#           via `python {tool_path.relative_to(run_dir)}`\n\n"
            )
            tool_path.write_text(header + script_code, encoding="utf-8")
        except Exception:
            pass  # non-fatal

    return {
        "status": "OK" if rc == 0 else "FAILED",
        "output": output[:3000],  # cap for history context
        "script_path": str(script_path),
        "script_description": script_desc,
        "returncode": rc,
    }


# ---------------------------------------------------------------------------
# Extract compact metrics from a finished case
# ---------------------------------------------------------------------------

def _extract_case_metrics(
    case_dir: Path,
    starter_dir: Optional[Path] = None,
    repo_root: Optional[Path] = None,
    starter_understanding: Optional[Dict[str, Any]] = None,
    objective_contract: Optional[Dict[str, Any]] = None,
) -> str:
    """Pull interpreter decision + QoI vs reference error metrics from a finished case."""
    parts: list[str] = []

    # 1. interpreter decision if available
    decision_path = case_dir / "decision.json"
    if decision_path.is_file():
        d = _read_json(decision_path, {})
        parts.append(f"interpreter={d.get('status','?')}: {str(d.get('reason',''))[:300]}")

    # 2. QoI vs reference comparison (generic — uses any CSV reference found in starter_dir)
    cf_metrics = _compute_cf_metrics(
        case_dir,
        starter_dir=starter_dir,
        repo_root=repo_root,
        starter_understanding=starter_understanding,
        objective_contract=objective_contract,
    )
    if cf_metrics:
        parts.append(cf_metrics)

    # 3. residual convergence tail from simpleFoam log
    log = case_dir / "log.simpleFoam"
    if log.is_file():
        lines = log.read_text(errors="ignore").splitlines()
        # find last residual lines
        res_lines = [l for l in lines if "Solving for" in l or "SIMPLE solution converged" in l]
        tail = res_lines[-3:] if res_lines else lines[-3:]
        parts.append("residuals: " + " | ".join(tail)[:250])

    return "\n".join(parts) if parts else "no metrics available"


def _compute_cf_metrics(
    case_dir: Path,
    starter_dir: Optional[Path] = None,
    repo_root: Optional[Path] = None,
    starter_understanding: Optional[Dict[str, Any]] = None,
    objective_contract: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Generic QoI comparison: sim postProcessing vs any reference data.

    Works for any QoI (wall friction, heat flux, drag, viscosity profile,
    outlet flux, pressure drop, velocity profile, etc.) and any reference
    format (CSV, dat, txt). No hardcoded QoI names or field names.

    Strategy:
      1. Find all reference CSVs and any project compare script in starter_dir.
      2. Catalogue all postProcessing output files in the case.
      3. Use an LLM to write a Python comparison script that matches sim output
         to the reference, applies correct normalization, and computes error metrics.
      4. Execute the script and capture the result.
      5. Apply generic sign/scale anomaly detection on the raw output numbers.
    """
    _bootstrap(repo_root)
    starter_understanding = starter_understanding or {}
    objective_contract = objective_contract or {}
    flow_params = starter_understanding.get("flow_parameters", {}) or {}
    ref_info = starter_understanding.get("reference_data", {}) or {}
    quantities = ref_info.get("quantities", [])
    # 0) Bound evaluator path (if locked).
    bound_result = _run_bound_comparator(case_dir, objective_contract)
    if bound_result:
        return bound_result

    # ------------------------------------------------------------------ #
    # 1. Find reference files and the comparator script                    #
    #                                                                      #
    # Comparator binding source-of-truth (in priority order):              #
    #   (a) objective_contract.comparator_script — set by                  #
    #       _resolve_objective_contract via LLM-content classification     #
    #       (no naming-convention assumption; starter-local).              #
    #   (b) bound_comparators.json — Phase 1 metric-author output          #
    #       (handled by the OED extension layer, not here).                #
    # We do NOT rglob the repo for compare*.py: that previously leaked    #
    # cross-starter scripts (e.g. periodic-hill compare_exactmatch_cf.py   #
    # bound to a multiphase droplet study).                                #
    # Reference files (.csv, .dat) ARE allowed to come from repo_root      #
    # because those are user-asserted data and the binder never invokes    #
    # them as code.                                                        #
    # ------------------------------------------------------------------ #
    ref_search_roots: list[Path] = [p for p in [starter_dir, repo_root] if p and p.is_dir()]
    ref_candidates: list[Path] = []
    for root in ref_search_roots:
        for p in root.rglob("*.csv"):
            if p.stat().st_size > 0:
                ref_candidates.append(p)
        for p in root.rglob("*.dat"):
            if p.stat().st_size > 0 and "postProcessing" not in str(p):
                ref_candidates.append(p)

    # Comparator: use the locked contract path. If empty, no comparator —
    # caller falls through to bound_comparators.json or returns "".
    compare_scripts: list[Path] = []
    locked = str(objective_contract.get("comparator_script", "") or "").strip()
    if locked:
        lp = Path(locked)
        if lp.is_file():
            compare_scripts = [lp]

    # Rank reference files: prefer those whose names or parent dirs match known QoI names
    qoi_keywords = [q.lower() for q in quantities] + ["reference", "dns", "experiment", "ref"]
    ref_candidates.sort(key=lambda p: not any(k in (p.name + str(p.parent)).lower() for k in qoi_keywords))

    if not ref_candidates:
        return ""  # no reference data found — skip silently

    ref_csv = ref_candidates[0]

    # ------------------------------------------------------------------ #
    # 2. Catalogue postProcessing output                                   #
    # ------------------------------------------------------------------ #
    pp_dir = case_dir / "postProcessing"
    pp_tree = ""
    pp_files: list[Path] = []
    if pp_dir.is_dir():
        for p in sorted(pp_dir.rglob("*")):
            rel = p.relative_to(case_dir)
            pp_tree += f"  {rel}\n"
            if p.is_file() and p.suffix in (".dat", ".csv", ".txt", ""):
                pp_files.append(p)

    if not pp_tree:
        return "QoI comparison skipped: postProcessing directory is empty or missing"

    # ------------------------------------------------------------------ #
    # 3. Try project-provided compare script first (no arg assumptions)    #
    # ------------------------------------------------------------------ #
    for cscript in compare_scripts:
        try:
            res = subprocess.run(
                [sys.executable, str(cscript), "--case", str(case_dir)],
                capture_output=True, text=True, timeout=90,
                cwd=str(cscript.parent),
            )
            out = (res.stdout + res.stderr).strip()
            if out and res.returncode == 0:
                return f"QoI comparison (project script {cscript.name}):\n{out[:1000]}"
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # 4. LLM-written comparison script                                     #
    #    The LLM sees: reference file sample, postProcessing tree,         #
    #    flow parameters, and QoI targets. It decides how to extract       #
    #    and compare — no QoI-specific logic here.                         #
    # ------------------------------------------------------------------ #
    try:
        from langchain_core.messages import HumanMessage, SystemMessage  # type: ignore
        from cfd_langgraph.config import get_settings  # type: ignore
        from cfd_langgraph.llm.factory import create_langchain_llm  # type: ignore

        # Read a sample of the reference file
        ref_sample = ""
        try:
            ref_sample = ref_csv.read_text(errors="ignore")[:2000]
        except Exception:
            pass

        # Sample first postProcessing file content for context
        pp_sample = ""
        for pf in pp_files[:3]:
            try:
                pp_sample += f"\n--- {pf.relative_to(case_dir)} ---\n"
                pp_sample += pf.read_text(errors="ignore")[:800]
            except Exception:
                pass

        norm_params = {k: flow_params[k] for k in flow_params if k not in ("geometry", "dimension")}

        # If the user shipped a comparator script in starter_dir, surface it as
        # an exemplar — the LLM should mirror its data-loading and error-metric
        # patterns rather than reinvent fragile CSV sniffing from scratch.
        exemplar_block = ""
        for cscript in compare_scripts[:1]:
            try:
                etext = cscript.read_text(encoding="utf-8", errors="replace")
                exemplar_block = (
                    f"\nEXEMPLAR comparator from starter (mirror its "
                    f"file-reading + RMSE patterns; do not import it):\n"
                    f"--- {cscript.name} ---\n{etext[:6000]}\n--- end ---\n"
                )
                break
            except Exception:
                pass

        sys_msg = (
            "You are a CFD post-processing expert. Write a short, self-contained Python script "
            "that compares simulation output to reference data and prints error metrics.\n\n"
            "RULES:\n"
            "- Use only: pathlib, numpy, csv, json, re (stdlib + numpy). No other imports.\n"
            "- CASE_DIR and REF_FILE are injected as pathlib.Path variables.\n"
            "- FLOW_PARAMS dict is injected with authoritative case parameters for normalization.\n"
            "- Search postProcessing/ for the most relevant output file(s) matching the QoI.\n"
            "- For loading numeric data: prefer `np.loadtxt` / `np.genfromtxt` with "
            "`comments='#'`, `delimiter=','` (or whitespace) over `csv.Sniffer().sniff()`. "
            "Sniffer is fragile on whitespace-leading lines and short samples.\n"
            "- Wrap each file-read in try/except. On failure, print "
            "`PARSE_WARNING: <path> <reason>` and skip — never raise.\n"
            "- ALWAYS print at least one line of the form `RMSE: <number>` if any "
            "comparison is possible (even on a partial subset). The orchestrator "
            "regex looks for that token literally.\n"
            "- Compute: L2 error, Linf error, and any physically meaningful scalar (e.g. peak error location).\n"
            "- SIGN CHECK: if the sim-vs-ref Pearson correlation is negative, negate the sim values "
            "and print 'SIGN_CORRECTED: <reason>'.\n"
            "- SCALE CHECK: if sim/ref RMS ratio > 50 or < 0.02, print "
            "'SCALE_WARNING: sim/ref={ratio:.2e} — check normalization. "
            "Authoritative normalization params: {params}'.\n"
            "- Use FLOW_PARAMS for any normalization (velocity, length, density, etc.) — never hardcode 1.\n"
            "- Always include reference data in output: print reference column stats alongside sim.\n"
            "- Print results clearly on stdout. No plots. No file writes.\n"
            "- Output ONLY raw Python. No markdown fences."
        )

        user_msg = (
            f"QoI targets: {quantities}\n"
            f"Reference file: {ref_csv}\n"
            f"Reference sample:\n{ref_sample}\n\n"
            f"postProcessing tree:\n{pp_tree[:3000]}\n\n"
            f"postProcessing sample files:\n{pp_sample[:2000]}\n\n"
            f"Flow parameters (use for normalization): {json.dumps(norm_params, ensure_ascii=False)}\n"
            f"Reference data description: {ref_info.get('description', '')}\n"
            f"Usage guidance: {ref_info.get('usage_guidance', '')}"
            f"{exemplar_block}"
        )

        llm = create_langchain_llm(model=get_settings().model, temperature=0.0)
        raw = llm.invoke([SystemMessage(content=sys_msg), HumanMessage(content=user_msg)])
        script_code = str(getattr(raw, "content", raw)).strip()
        if script_code.startswith("```"):
            script_code = "\n".join(script_code.split("\n")[1:])
            script_code = script_code.rsplit("```", 1)[0]

        # Inject path bindings
        preamble = (
            f"import pathlib, numpy as np, csv, json, re\n"
            f"CASE_DIR = pathlib.Path({str(case_dir)!r})\n"
            f"REF_FILE = pathlib.Path({str(ref_csv)!r})\n"
            f"FLOW_PARAMS = {json.dumps(norm_params, ensure_ascii=False)}\n\n"
        )
        full_script = preamble + script_code

        # Write and run
        script_path = case_dir / "_oed_qoi_compare.py"
        script_path.write_text(full_script, encoding="utf-8")
        res = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True, text=True, timeout=90,
            cwd=str(case_dir),
        )
        out = (res.stdout + res.stderr).strip()
        # Keep the script on disk for debugging — leaving the file makes it
        # easier to diagnose future LLM-script issues.

        # Successful path: stdout/stderr already has metric numbers (RMSE etc).
        if out and re.search(r"(?i)\b(rmse|l2|mae|mse|error)\b", out):
            return f"QoI comparison (LLM script):\n{out[:1500]}"

        # Fallback: the LLM script crashed or printed nothing useful, but the
        # case may still have metric artifacts on disk from any comparator
        # that ran earlier. Scan the WHOLE case (no hardcoded subdir names)
        # for text files containing any metric token. Generic across studies
        # — pressure-drop comparator, heat-flux comparator, drag comparator,
        # etc. all write text artifacts that this picks up.
        scan_blob = ""
        files_scanned = 0
        for p in case_dir.rglob("*"):
            if files_scanned > 400:
                break
            if not p.is_file():
                continue
            if p.suffix.lower() not in (".md", ".txt", ".json", ".csv", ".dat", ".log"):
                continue
            if any(part.replace(".", "").isdigit() for part in p.relative_to(case_dir).parts[:1]):
                continue
            try:
                if p.stat().st_size > 200_000:
                    continue
                text = p.read_text(encoding="utf-8", errors="replace")[:6000]
            except Exception:
                continue
            files_scanned += 1
            if _METRIC_TOKEN_RE.search(text):
                scan_blob += f"\n--- {p.relative_to(case_dir)} ---\n{text[:1500]}"
        if scan_blob.strip():
            return (
                f"QoI comparison (recovered from on-disk artifacts after "
                f"LLM script {'crashed' if 'Traceback' in out else 'returned no metric'}):\n"
                f"{scan_blob[:3000]}"
            )

        if out:
            return f"QoI comparison (LLM script — no metric token, kept for diagnostics):\n{out[:1500]}"
        return "QoI comparison: LLM script ran but produced no output"

    except Exception as exc:
        return f"QoI comparison skipped (LLM script failed: {exc})"


# ---------------------------------------------------------------------------
# Run a single iteration: code_mod or experiment
# ---------------------------------------------------------------------------

def _build_experiment_requirement(
    *,
    iteration: int,
    kind: str,
    model_desc: str,
    model_name: str,
    params: Dict[str, Any],
    ref_info: Dict[str, Any],
    topic: str,
) -> str:
    """
    Build a FoamAgent requirement string that is:
      - Generic (no hardcoded QoI names, velocity symbols, or case-specific constants)
      - Self-contained (carries all normalization parameters so FoamAgent never guesses)
      - Explicit about post-processing convention rules
    """
    params_block = json.dumps(params, ensure_ascii=False, indent=2)
    ref_quantities = ref_info.get("quantities", [])
    ref_desc = ref_info.get("description", "")
    ref_usage = ref_info.get("usage_guidance", "")

    # Build a normalization note from params so FoamAgent uses the right scale
    norm_note_parts = []
    for key in ("Ub", "U_ref", "Uref", "U_inf", "nu", "rho", "L", "D", "Re"):
        if key in params:
            norm_note_parts.append(f"{key}={params[key]}")
    norm_note = (
        "NORMALIZATION: Use the following authoritative values for ANY post-processing "
        "or comparison script — do NOT hardcode 1.0 or any other default: "
        + ", ".join(norm_note_parts)
        if norm_note_parts else
        "NORMALIZATION: Use parameter values from AUTHORITATIVE_TARGET_PARAMETERS below."
    )

    return (
        f"Open-ended discovery iteration {iteration} ({kind}). "
        f"Topic: {topic}\n\n"
        f"Model: {model_desc} (compiled model: {model_name}).\n\n"
        f"AUTHORITATIVE_TARGET_PARAMETERS (single source of truth — override any conflicting value):\n"
        f"{params_block}\n\n"
        f"{norm_note}\n\n"
        f"SIGN & CONVENTION: For any extracted quantity (wall stress, heat flux, drag, viscosity, "
        f"pressure, flux, etc.), verify the sign and normalization convention matches the reference. "
        f"OpenFOAM may report quantities acting ON THE FLUID (opposite to wall-referenced convention), "
        f"or with solver-dependent sign choices. Always cross-check extracted values against physical "
        f"expectations (e.g., attached flow → positive streamwise wall friction; heat flows from hot "
        f"to cold; pressure drops in flow direction). If the reference uses a different convention, "
        f"apply the appropriate negation or scaling in post-processing — do NOT assume any sign.\n\n"
        f"REFERENCE QoIs to compare against: {ref_quantities}\n"
        f"Reference description: {ref_desc}\n"
        f"Reference usage guidance: {ref_usage}\n\n"
        f"Run to convergence. Write all required postProcessing output. "
        f"Include reference/DNS data alongside simulation results in any plots."
    )


def _run_code_mod_iteration(
    *,
    iteration: int,
    action: Dict[str, Any],
    run_dir: Path,
    repo_root: Path,
    base_case_dir: str,
    topic: str,
    lit_path: Path,
    timeline_path: Path,
    starter_understanding: Dict[str, Any],
    starter_dir: Optional[Path] = None,
    objective_contract: Optional[Dict[str, Any]] = None,
    env: Dict[str, str],
    attempt_tag: str = "",
) -> Dict[str, Any]:
    """Compile a new model and run one validation experiment."""
    suffix = f"_{attempt_tag}" if attempt_tag else ""
    # Named-variant directory layout: if the decision LLM supplied a short
    # `variant_name` slug (or one can be inferred from model_description),
    # include it in the iter-dir name so directories are interpretable later.
    # Generic — works for any mode/topic. Falls back to plain iter_NNN_code_mod
    # when no slug is available, preserving backward compatibility.
    variant_slug = _slugify_variant(
        action.get("variant_name")
        or action.get("model_description")
        or action.get("compiled_model_name")
        or ""
    )
    slug_part = f"_{variant_slug}" if variant_slug else ""
    iter_dir = run_dir / f"iter_{iteration:03d}_code_mod{slug_part}{suffix}"
    iter_dir.mkdir(parents=True, exist_ok=True)

    formula = action.get("formula_or_modification", "")
    model_desc = action.get("model_description", f"iteration_{iteration}_model")

    # Write a minimal starter_understanding override so code_mod_prepare picks up the formula
    iter_su = {
        "base_case_path": starter_understanding.get("base_case_path", ""),
        "formula_or_model_spec": formula,
        "formula_file": f"open_ended_iter_{iteration}",
        "flow_parameters": starter_understanding.get("flow_parameters", {}),
        "reference_data": starter_understanding.get("reference_data", {}),
        "status": "ok",
    }
    iter_su_path = iter_dir / "iter_starter_understanding.json"
    _write_json(iter_su_path, iter_su)

    # code_mod_prepare
    prepared_path = iter_dir / "code_mod_prepared.json"
    # Shared recon cache lives at the study root so every OED iteration reuses
    # the same discovered_paths.json — the OpenFOAM source tree doesn't change
    # across iterations, so recon runs at most once per study.
    shared_recon_cache = run_dir / "discovered_paths.json"
    rc, out, err = _call(
        [
            sys.executable, "scripts/code_mod_prepare.py",
            "--topic", topic,
            "--run-dir", str(iter_dir),
            "--literature", str(lit_path) if lit_path.is_file() else "",
            "--base-case-dir", base_case_dir,
            "--output", str(prepared_path),
            "--starter-understanding", str(iter_su_path),
            "--recon-cache", str(shared_recon_cache),
            *(["--starter-dir", str(starter_dir)] if starter_dir and starter_dir.is_dir() else []),
        ],
        cwd=repo_root,
        env=env,
    )
    print(out)
    if rc != 0:
        return {"status": "FAILED", "error": f"code_mod_prepare failed: {err[-500:]}", "case_dir": str(iter_dir)}

    prepared = _read_json(prepared_path, {})
    payload_raw = prepared.get("payload") if isinstance(prepared, dict) else {}
    if not isinstance(payload_raw, dict):
        return {"status": "FAILED", "error": "code_mod_prepare produced no payload", "case_dir": str(iter_dir)}
    payload_path = iter_dir / "code_mod_payload.json"
    _write_json(payload_path, payload_raw)

    # foam_code_builder
    build_result_path = iter_dir / "code_mod_build_result.json"
    rc, out, err = _call(
        [sys.executable, "scripts/foam_code_builder.py",
         "--payload", str(payload_path),
         "--output", str(build_result_path)],
        cwd=repo_root,
        env=env,
    )
    print(out)
    if rc != 0:
        return {"status": "FAILED", "error": f"foam_code_builder failed: {err[-500:]}", "case_dir": str(iter_dir)}

    # code_mod_apply_compile
    apply_result_path = iter_dir / "code_mod_apply_result.json"
    rc, out, err = _call(
        [sys.executable, "scripts/code_mod_apply_compile.py",
         "--build-result", str(build_result_path),
         "--output", str(apply_result_path),
         "--max-compile-attempts", "10"],
        cwd=repo_root,
        env=env,
    )
    print(out)
    apply_result = _read_json(apply_result_path, {})
    if rc != 0 or not apply_result.get("compile_ok"):
        compile_hint = _extract_compile_error_hint(apply_result, err)
        return {
            "status": "FAILED",
            "error": f"compile failed: {compile_hint}",
            "compile_error_hint": compile_hint,
            "case_dir": str(iter_dir),
        }

    compiled_case = apply_result.get("case_dir") or str(iter_dir / "canonical_base_case")
    class_name = apply_result.get("class_name", "CustomModel")

    # Build requirement for this experiment
    fp = starter_understanding.get("flow_parameters", {}) or {}
    params = {**fp, **action.get("parameters", {})}
    params.pop("geometry", None)
    ref_info = starter_understanding.get("reference_data", {}) or {}
    requirement = _build_experiment_requirement(
        iteration=iteration,
        kind="code_mod_validation",
        model_desc=model_desc,
        model_name=class_name,
        params=params,
        ref_info=ref_info,
        topic=topic,
    )

    # The compiled case is already complete — copy it, source OpenFOAM, run
    # Allrun. This used to go through foam_run.py, whose reviewer-led rewrite
    # loop needs Foam-Agent vendored; it also rewrites momentumTransport /
    # fvOptions / Make on a failed Allrun, which is the wrong repair for a
    # case whose whole point is the custom model already compiled into it.
    exp_dir = iter_dir / "experiment"
    validation_run_result_path = iter_dir / "validation_run_result.json"
    rc, out, err = _call(
        [
            sys.executable, "scripts/foam_run_simple.py",
            "--base-case", compiled_case,
            "--output-dir", str(exp_dir),
            "--output", str(validation_run_result_path),
            "--timeout", "21600",
        ],
        cwd=repo_root,
        timeout=22000,
        env=env,
    )
    print(out)

    # interpret
    _run_interpret(exp_dir, repo_root, timeline_path, env=env, objective_contract=objective_contract)
    metrics = _extract_case_metrics(
        exp_dir,
        starter_dir=starter_dir,
        repo_root=repo_root,
        starter_understanding=starter_understanding,
        objective_contract=objective_contract,
    )
    decision = _read_json(exp_dir / "decision.json", {})

    return {
        "compiled_model_name": class_name,
        "compiled_model_description": model_desc,
        "compiled_case_dir": compiled_case,
        "case_dir": str(exp_dir),
        "status": decision.get("status", "UNKNOWN"),
        "interpreter_reason": str(decision.get("reason", ""))[:500],
        "metrics_summary": metrics,
    }


def _resolve_starter_case_dir(
    starter_dir: Optional[Path],
    starter_understanding: Dict[str, Any],
    base_case_dir: str = "",
) -> Optional[Path]:
    """Find a clean OpenFOAM case (constant/ + system/) to clone from.
    Generic across topics — scans starter_dir for the first complete case."""
    cand = (base_case_dir or "").strip() or str(starter_understanding.get("base_case_path", "") or "").strip()
    if cand and Path(cand).exists() and (Path(cand) / "constant").is_dir():
        return Path(cand)
    if starter_dir is not None and Path(starter_dir).is_dir():
        for d in sorted(Path(starter_dir).rglob("*")):
            if not d.is_dir():
                continue
            if (d / "constant").is_dir() and (d / "system").is_dir():
                return d
            if d.relative_to(starter_dir).parts.__len__() > 3:
                break
    return None


def _run_agentic_code_mod_iteration(
    *,
    iteration: int,
    action: Dict[str, Any],
    run_dir: Path,
    repo_root: Path,
    base_case_dir: str,
    topic: str,
    timeline_path: Path,
    starter_understanding: Dict[str, Any],
    starter_dir: Optional[Path] = None,
    objective_contract: Optional[Dict[str, Any]] = None,
    env: Dict[str, str],
    attempt_tag: str = "",
    baseline_metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Class-derivation code_mod via the agentic runner. Spawns codex exec with
    full shell access; the agent reads OpenFOAM source, writes the custom
    library, runs wmake, fixes errors, retries — all in one session. Mirrors
    ARIS's code_mod mechanism. Generic across CFD modification kinds.
    """
    suffix = f"_{attempt_tag}" if attempt_tag else ""
    variant_slug = _slugify_variant(
        action.get("variant_name")
        or action.get("model_description")
        or "agentic"
    )
    slug_part = f"_{variant_slug}" if variant_slug else ""
    iter_dir = run_dir / f"iter_{iteration:03d}_code_mod{slug_part}{suffix}"
    iter_dir.mkdir(parents=True, exist_ok=True)

    # Hypothesis = the formula_or_modification spec the LLM produced. Pass
    # the topic separately so the agent has full domain context.
    hypothesis = (
        str(action.get("formula_or_modification") or "").strip()
        or str(action.get("model_description") or "").strip()
        or "Implement a small physically-motivated modification of the parent class."
    )
    starter_case = _resolve_starter_case_dir(starter_dir, starter_understanding, base_case_dir)
    if starter_case is None:
        return {
            "status": "FAILED",
            "error": "no starter OpenFOAM case found for agentic code_mod",
            "compile_error_hint": "no starter case",
            "case_dir": str(iter_dir),
        }

    agentic_result_path = iter_dir / "agentic_result.json"
    print(f"[OED][agentic] launching agentic exec for {variant_slug} (turn-bounded; no wall-clock cap)")
    rc, out, err = _call(
        [sys.executable, "scripts/code_mod_agentic.py",
         "--hypothesis", hypothesis,
         "--variant-name", variant_slug,
         "--run-dir", str(iter_dir),
         "--starter-case", str(starter_case),
         "--topic", topic,
         "--output", str(agentic_result_path),
         # Disable the agentic wall-clock cap; bound only by --max-turns
         # (default 120). Slower LLMs (e.g. claude-sonnet-4-6) make
         # turn-by-turn progress on class_derivation but exceed 30 min
         # wall-clock; turn-budget is a fairer cap.
         "--timeout", "0"],
        cwd=repo_root,
        # Outer subprocess fence: keep generous so no caller-side kill
        # before the agent's own turn budget naturally exhausts. 6 hours
        # is the same scale as the per-experiment fence elsewhere.
        timeout=21600,
        env=env,
    )
    print(out)
    agentic_result = _read_json(agentic_result_path, {})
    if not isinstance(agentic_result, dict):
        agentic_result = {}

    success = (agentic_result.get("status") == "OK")
    case_dir = agentic_result.get("case_dir") or str(iter_dir)
    class_name = agentic_result.get("class_name") or variant_slug
    compiled_so = agentic_result.get("compiled_so", "")
    compile_ok = bool(agentic_result.get("compile_ok"))
    converged = bool(agentic_result.get("converged"))

    # Build the failure hint by reading the trajectory log tail (where wmake
    # errors are visible) so the planner LLM sees what went wrong.
    err_hint = ""
    traj_log = agentic_result.get("trajectory_log", "")
    if not success and traj_log and Path(traj_log).is_file():
        try:
            tail = Path(traj_log).read_text(encoding="utf-8", errors="replace")[-6000:]
            for line in tail.splitlines():
                ls = line.strip()
                if not ls:
                    continue
                low = ls.lower()
                if (
                    "fatal error" in low
                    or " error:" in low
                    or "no matching function" in low
                    or "was not declared" in low
                    or "undefined reference" in low
                    or "make: ***" in low
                ):
                    err_hint = ls[:400]
                    break
            if not err_hint:
                err_hint = (agentic_result.get("error") or tail[-400:])[:400]
        except Exception:
            err_hint = (agentic_result.get("error") or "agentic run failed")[:400]
    elif not success:
        err_hint = (agentic_result.get("error") or "agentic run failed")[:400]

    if success:
        # Run-validity gate before scoring (skip preflight for agentic mode —
        # the agent authored its own Allrun and we don't want to second-guess
        # it). If the gate trips, skip interpret and return RUN_INVALID.
        gate_result = _run_validity_gate(
            case_dir=Path(case_dir), baseline_metrics=baseline_metrics,
            runtime_run_result=None,
            base_case=Path(base_case_dir) if base_case_dir else None,
        )
        if not gate_result.get("valid", True):
            print(f"[OED][gate] agentic RUN_INVALID — {gate_result.get('reason', '')[:240]}")
            return {
                "compiled_model_name": class_name,
                "compiled_model_description": str(action.get("model_description", ""))[:240],
                "compiled_case_dir": case_dir,
                "compiled_so": compiled_so,
                "case_dir": case_dir,
                "status": "RUN_INVALID",
                "interpreter_reason": gate_result.get("reason", "RUN_INVALID")[:500],
                "metrics_summary": "",
                "is_runtime": False,
                "is_agentic": True,
                "modification_category": "class_derivation",
                "trajectory_log": traj_log,
                "run_validity": gate_result,
            }
        # Run interpret on the converged case to score it.
        _run_interpret(Path(case_dir), repo_root, timeline_path, env=env, objective_contract=objective_contract)
        metrics = _extract_case_metrics(
            Path(case_dir),
            starter_dir=starter_dir,
            repo_root=repo_root,
            starter_understanding=starter_understanding,
            objective_contract=objective_contract,
        )
        decision = _read_json(Path(case_dir) / "decision.json", {})
        return {
            "compiled_model_name": class_name,
            "compiled_model_description": str(action.get("model_description", ""))[:240],
            "compiled_case_dir": case_dir,
            "compiled_so": compiled_so,
            "case_dir": case_dir,
            "status": decision.get("status", "UNKNOWN"),
            "interpreter_reason": str(decision.get("reason", ""))[:500],
            "metrics_summary": metrics,
            "is_runtime": False,
            "is_agentic": True,
            "modification_category": "class_derivation",
            "trajectory_log": traj_log,
            "run_validity": gate_result,
        }
    return {
        "status": "FAILED",
        "case_dir": str(iter_dir),
        "compiled_case_dir": case_dir,
        "compile_error_hint": err_hint,
        "interpreter_reason": (
            f"Agentic code_mod did not produce a converged case. "
            f"compile_ok={compile_ok}, converged={converged}. {err_hint}"
        )[:500],
        "metrics_summary": "",
        "is_runtime": False,
        "is_agentic": True,
        "modification_category": "class_derivation",
        "trajectory_log": traj_log,
    }


def _revise_runtime_snippet_with_llm(
    *,
    action: Dict[str, Any],
    error_hint: str,
    error_context: str,
    base_case: str,
    repo_root: Path,
) -> Optional[Dict[str, Any]]:
    """Ask the LLM to revise the C++ snippet body in a runtime action JSON
    given the FOAM error from the failed run. Returns a deep-copied action
    with only the snippet body fields replaced; or None if the LLM didn't
    produce a usable revision.

    Generic across runtime_source / runtime_bc / runtime_field. Surgical:
    only the C++ body fields (`code_add_sup`, `code_add_sup_rho`,
    `code_constrain`, `code_correct`, `code_include`, `code_execute`,
    `code`, `code_init`, `code_options`) get rewritten; coefficients,
    target_field, patch_name, etc. are preserved.
    """
    _bootstrap(repo_root)
    from langchain_core.messages import HumanMessage, SystemMessage  # type: ignore
    from cfd_langgraph.config import get_settings  # type: ignore
    from cfd_langgraph.llm.factory import create_langchain_llm  # type: ignore
    from cfd_langgraph.utils import strip_json_fences  # type: ignore

    cat = action.get("modification_category") or ""
    spec_key = {"runtime_source": "runtime_source",
                "runtime_bc": "runtime_bc",
                "runtime_field": "runtime_field"}.get(cat)
    if not spec_key:
        return None
    spec = action.get(spec_key) or {}
    if not isinstance(spec, dict):
        return None

    # Inject grounding so the revision LLM sees the same field-registry +
    # dimensioned-arithmetic facts as the planner.
    grounding = ""
    try:
        from openfoam_grounding import build_runtime_snippet_grounding  # type: ignore
        grounding = build_runtime_snippet_grounding(Path(base_case) if base_case else None)
    except Exception:
        grounding = ""

    sys_msg = (
        "You are an OpenFOAM coded-snippet repair tool. The user's runtime "
        "modification crashed at solver startup. Read the FOAM error and the "
        "original snippet, then produce a corrected snippet body. "
        "Change ONLY what is needed to fix the reported error. Preserve the "
        "physics intent. Do not rename coefficients or change the target field.\n\n"
        + grounding
        + "\nReturn STRICT JSON only with this shape (no markdown, no commentary):\n"
        "{\n"
        "  \"code_add_sup\":  \"<revised C++ body, optional — include only if changed>\",\n"
        "  \"code_add_sup_rho\": \"<...>\",\n"
        "  \"code_constrain\":   \"<...>\",\n"
        "  \"code_correct\":     \"<...>\",\n"
        "  \"code_include\":     \"<...>\",\n"
        "  \"code\":             \"<runtime_bc body, only for runtime_bc>\",\n"
        "  \"code_init\":        \"<runtime_bc init, only for runtime_bc>\",\n"
        "  \"code_execute\":     \"<runtime_field body, only for runtime_field>\",\n"
        "  \"explanation\":      \"<one sentence on what changed and why>\"\n"
        "}\n"
        "Omit any field you are not changing. Do not return the whole action."
    )
    user_msg = (
        f"Modification category: {cat}\n\n"
        f"FOAM error (one-line summary): {error_hint}\n\n"
        f"Captured log tail (last ~3500 chars):\n{error_context or '(empty)'}\n\n"
        f"Original snippet spec (JSON):\n{json.dumps(spec, indent=2, default=str)[:8000]}\n"
    )
    llm = create_langchain_llm(model=get_settings().model, temperature=0.0)
    try:
        raw = llm.invoke([SystemMessage(content=sys_msg), HumanMessage(content=user_msg)])
        text = getattr(raw, "content", str(raw))
        text = strip_json_fences(text or "")
        patch = json.loads(text)
    except Exception:
        return None
    if not isinstance(patch, dict) or not patch:
        return None

    # Whitelist of body fields that may be replaced.
    allowed = {"code_add_sup", "code_add_sup_rho", "code_constrain",
               "code_correct", "code_include", "code", "code_init",
               "code_execute", "code_options"}
    diffs = {k: v for k, v in patch.items() if k in allowed and isinstance(v, str)}
    if not diffs:
        return None

    # Build a deep-copied revised action with only the whitelisted body
    # fields swapped.
    revised = json.loads(json.dumps(action, default=str))
    revised_spec = revised.setdefault(spec_key, {})
    for k, v in diffs.items():
        revised_spec[k] = v
    revised["_runtime_revision_explanation"] = str(patch.get("explanation", ""))[:300]
    return revised


def _run_runtime_code_mod_iteration(
    *,
    iteration: int,
    action: Dict[str, Any],
    run_dir: Path,
    repo_root: Path,
    base_case_dir: str,
    topic: str,
    timeline_path: Path,
    starter_understanding: Dict[str, Any],
    starter_dir: Optional[Path] = None,
    objective_contract: Optional[Dict[str, Any]] = None,
    env: Dict[str, str],
    attempt_tag: str = "",
    baseline_metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Apply a runtime / dict_only modification (no wmake, no library build) and
    run one validation experiment. Generic across categories — works for any
    coded* / dictionary modification produced by code_mod_runtime.py.
    """
    suffix = f"_{attempt_tag}" if attempt_tag else ""
    variant_slug = _slugify_variant(
        action.get("variant_name")
        or action.get("model_description")
        or action.get("modification_category")
        or ""
    )
    slug_part = f"_{variant_slug}" if variant_slug else ""
    iter_dir = run_dir / f"iter_{iteration:03d}_code_mod{slug_part}{suffix}"
    iter_dir.mkdir(parents=True, exist_ok=True)

    action_path = iter_dir / "runtime_action.json"
    _write_json(action_path, action)

    # Resolve a base OpenFOAM case to copy. Order:
    #   1) explicit base_case_dir CLI arg
    #   2) starter_understanding.base_case_path
    #   3) any directory under starter_dir containing constant/ + system/
    # We DO NOT fall back to a prior iter's canonical_base_case — each code_mod
    # must start from a clean baseline. Prior iterations may have left runtime
    # modifications (e.g. a broken customSource in fvModels) that would
    # contaminate independent hypotheses with unrelated bugs.
    base = (base_case_dir or "").strip() or str(starter_understanding.get("base_case_path", "") or "").strip()
    if (not base or not Path(base).exists()) and starter_dir is not None:
        # Scan starter_dir (1-3 levels deep) for a complete OpenFOAM case.
        for d in sorted(Path(starter_dir).rglob("*")):
            if not d.is_dir():
                continue
            if (d / "constant").is_dir() and (d / "system").is_dir():
                base = str(d)
                break
            if d.relative_to(starter_dir).parts.__len__() > 3:
                break

    if not base or not Path(base).exists():
        msg = (
            f"runtime code_mod: no base OpenFOAM case found. "
            f"Tried base_case_dir={base_case_dir!r}, "
            f"starter_understanding.base_case_path={starter_understanding.get('base_case_path')!r}, "
            f"and recursive scan under starter_dir={starter_dir!s}. "
            f"A complete case (with constant/ and system/) is required."
        )
        print(f"[OED][runtime] {msg}")
        return {
            "status": "FAILED",
            "error": msg,
            "runtime_apply_error": msg[:500],
            "case_dir": str(iter_dir),
            "compile_error_hint": "",
        }
    print(f"[OED][runtime] using base case: {base}")

    runtime_result_path = iter_dir / "runtime_apply_result.json"
    rc, out, err = _call(
        [sys.executable, "scripts/code_mod_runtime.py",
         "--action", str(action_path),
         "--base-case", str(base),
         "--iter-dir", str(iter_dir),
         "--output", str(runtime_result_path)],
        cwd=repo_root,
        env=env,
    )
    print(out)
    runtime_result = _read_json(runtime_result_path, {})
    if rc != 0 or runtime_result.get("status") != "OK":
        msg = runtime_result.get("error") or err[-500:]
        # Runtime applier reports its own structured error — surface it as
        # `runtime_apply_error` (distinct from compile_error_hint) so the
        # decision LLM doesn't re-prompt the build-engineer; the error is in
        # the dictionary spec, not in C++.
        return {
            "status": "FAILED",
            "error": f"runtime apply failed: {msg}",
            "runtime_apply_error": str(msg)[:500],
            "case_dir": str(iter_dir),
            "compile_error_hint": "",
        }

    compiled_case = runtime_result.get("case_dir") or str(iter_dir / "canonical_base_case")
    # Allrun pre-flight: LLM audits the case's Allrun BEFORE we run it. Catches
    # commented-out solver-invocation lines and similar harness bugs that would
    # otherwise leave max_time=0 with rc=0 from foam_run_simple.
    try:
        _maybe_preflight_allrun(Path(compiled_case), repo_root)
    except Exception as _ex:
        print(f"[OED][preflight] non-fatal: {_ex}")
    class_name = runtime_result.get("class_name") or f"runtime_{variant_slug}"
    model_desc = action.get("model_description", f"iteration_{iteration}_runtime_model")

    fp = starter_understanding.get("flow_parameters", {}) or {}
    params = {**fp, **action.get("parameters", {})}
    params.pop("geometry", None)
    ref_info = starter_understanding.get("reference_data", {}) or {}

    # IMPORTANT: do NOT include a fancy "compiled model" name in the requirement
    # string for runtime modifications. FoamAgent's reviewer loop sees that and
    # tries to derive a custom turbulence-model C++ class with that name —
    # rewriting constant/momentumTransport to point at libRuntimeXxx.so.
    # For runtime mods the case's momentumTransport already points at the
    # standard built-in model; the modification lives in fvOptions / coded BC
    # / coded functionObject / dictionary edits.
    requirement = _build_experiment_requirement(
        iteration=iteration,
        kind="runtime_code_mod_validation",
        model_desc=(
            f"{model_desc}. "
            f"NOTE TO RUNNER: this case is a complete OpenFOAM case with a "
            f"runtime modification already applied via OpenFOAM's coded* "
            f"infrastructure (codedFvOption / codedFixedValue / coded "
            f"functionObject / dictionary edit). DO NOT modify "
            f"constant/momentumTransport. DO NOT generate a custom turbulence "
            f"model class. The standard built-in model is intentional; the "
            f"modification lives in constant/fvOptions, 0/<field> BCs, or "
            f"system/controlDict.functions. Just run the case to convergence "
            f"and report metrics."
        ),
        model_name="(built-in; runtime modification in fvOptions/BCs/functions)",
        params=params,
        ref_info=ref_info,
        topic=topic,
    )

    exp_dir = iter_dir / "experiment"
    # Runtime cases are already complete — bypass FoamAgent's reviewer-led
    # foam_run.py entirely. The reviewer assumes class-derivation and will
    # rewrite constant/momentumTransport / fvOptions / Make/* on Allrun
    # failure, which is exactly wrong for runtime / dict_only mods.
    # Use foam_run_simple.py: copy the case, source OpenFOAM, run Allrun.
    runtime_run_result_path = iter_dir / "runtime_run_result.json"
    rc, out, err = _call(
        [sys.executable, "scripts/foam_run_simple.py",
         "--base-case", compiled_case,
         "--output-dir", str(exp_dir),
         "--output", str(runtime_run_result_path),
         "--timeout", "21600"],
        cwd=repo_root,
        timeout=22000,
        env=env,
    )
    print(out)
    runtime_run_result = _read_json(runtime_run_result_path, {})
    # Did simpleFoam (or whatever app the case names) actually converge?
    run_ok = (rc == 0) and (str(runtime_run_result.get("status", "")).upper() == "OK")
    # If the case failed to run, scan ALL relevant outputs for the failure
    # signature: stderr/stdout tails AND the OpenFOAM log files (log.simpleFoam
    # / log.blockMesh etc.). The most informative error for runtime mods is
    # often inside log.simpleFoam (e.g. a JIT-compile error from
    # codedFvModel — tmp<>/template/operator-overload issues), which the
    # plain run-shell stderr never sees.
    if rc != 0 and isinstance(runtime_run_result, dict):
        run_err = (runtime_run_result.get("stderr_tail") or "")[-2000:]
        run_out = (runtime_run_result.get("stdout_tail") or "")[-2000:]
        run_msg = (runtime_run_result.get("error") or "")[:300]

        # Read the tails of every captured log file so JIT compile errors get
        # surfaced. Generic — works for any application's log.* output.
        log_blob = ""
        for lp in (runtime_run_result.get("log_paths") or [])[:8]:
            try:
                t = Path(lp).read_text(encoding="utf-8", errors="replace")[-4000:]
                log_blob += f"\n--- {Path(lp).name} ---\n{t}"
            except Exception:
                pass

        scan = (run_err + "\n" + run_out + "\n" + log_blob)
        # Pick the first occurrence of FATAL/error/Error: line for the hint —
        # that line is what the planner LLM and build-engineer prompt need.
        hint = ""
        for line in scan.splitlines():
            ls = line.strip()
            if not ls:
                continue
            low = ls.lower()
            # Prefer lines that look like compiler diagnostics or FOAM fatals.
            if (
                "fatal error" in low
                or " error:" in low
                or "fatal io error" in low
                or "no known conversion" in low
                or "no matching function" in low
                or "was not declared" in low
            ):
                hint = ls[:400]
                break
        if not hint:
            # Fall back to generic search.
            for line in scan.splitlines():
                ls = line.strip()
                if not ls:
                    continue
                if "error" in ls.lower() or "fatal" in ls.lower():
                    hint = ls[:400]
                    break
        if not hint:
            hint = (run_msg or run_err[-400:] or "OpenFOAM run failed")[:400]
        # Stash on the result dict that the OED loop is about to read.
        runtime_result.setdefault("runtime_apply_error", hint)
        runtime_result.setdefault("compile_error_hint", hint)

    # ------------------------------------------------------------------
    # Multi-round revise-and-retry (generic across runtime_source /
    # runtime_bc / runtime_field). If the run failed with a captured
    # FOAM/coded error, ask the LLM to revise the offending snippet body
    # given the latest error, re-render the case, re-run simpleFoam.
    # Each round builds on the prior round's revised snippet and the new
    # error it produced, so the LLM gets progressively better signal. Bound
    # at MAX_RETRIES rounds to prevent unbounded looping. Promotes the
    # successful attempt's artifacts as the iteration result; otherwise
    # keeps the original failure plus a per-round retry log.
    # ------------------------------------------------------------------
    MAX_RETRIES = 10
    if (not run_ok) and (runtime_result.get("compile_error_hint") or "") and \
            action.get("modification_category") in ("runtime_source", "runtime_bc", "runtime_field"):
        # Track the evolving action and the latest error context across rounds.
        current_action = action
        current_hint = runtime_result.get("compile_error_hint", "")
        current_ctx = log_blob[-3500:] if log_blob else ""
        retry_log: List[Dict[str, Any]] = []
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                revised = _revise_runtime_snippet_with_llm(
                    action=current_action,
                    error_hint=current_hint,
                    error_context=current_ctx,
                    base_case=base,
                    repo_root=repo_root,
                )
            except Exception as ex:
                print(f"[OED][runtime] retry {attempt}/{MAX_RETRIES}: revise step skipped "
                      f"({ex.__class__.__name__}: {ex})")
                retry_log.append({"attempt": attempt, "stage": "revise",
                                  "error": f"{ex.__class__.__name__}: {ex}"[:400]})
                break
            if revised is None:
                print(f"[OED][runtime] retry {attempt}/{MAX_RETRIES}: LLM produced no usable revision; stopping.")
                retry_log.append({"attempt": attempt, "stage": "revise",
                                  "error": "no usable revision"})
                break

            print(f"[OED][runtime] retry {attempt}/{MAX_RETRIES}: applying revised snippet …")
            retry_dir = iter_dir / f"retry_{attempt:02d}"
            retry_dir.mkdir(parents=True, exist_ok=True)
            retry_action_path = retry_dir / "runtime_action.json"
            _write_json(retry_action_path, revised)
            retry_apply_path = retry_dir / "runtime_apply_result.json"
            rc2, out2, err2 = _call(
                [sys.executable, "scripts/code_mod_runtime.py",
                 "--action", str(retry_action_path),
                 "--base-case", str(base),
                 "--iter-dir", str(retry_dir),
                 "--output", str(retry_apply_path)],
                cwd=repo_root,
                env=env,
            )
            print(out2)
            retry_apply = _read_json(retry_apply_path, {})
            if rc2 != 0 or retry_apply.get("status") != "OK":
                msg = (retry_apply.get("error") or err2[-400:] or "retry render failed")[:400]
                print(f"[OED][runtime] retry {attempt}: render FAILED — {msg[:120]}")
                retry_log.append({"attempt": attempt, "stage": "render", "error": msg})
                # Render-stage failure is usually a malformed action; feed it
                # back to the LLM as the next-round hint so it can correct
                # the spec, but advance the action so the next revision is
                # cumulative.
                current_action = revised
                current_hint = msg
                current_ctx = (out2 or "")[-3500:]
                continue

            retry_compiled = retry_apply.get("case_dir") or str(retry_dir / "canonical_base_case")
            # Allrun pre-flight on the freshly-rendered retry case.
            try:
                _maybe_preflight_allrun(Path(retry_compiled), repo_root)
            except Exception as _ex:
                print(f"[OED][preflight] non-fatal: {_ex}")
            retry_exp_dir = retry_dir / "experiment"
            retry_run_path = retry_dir / "runtime_run_result.json"
            rc2r, out2r, err2r = _call(
                [sys.executable, "scripts/foam_run_simple.py",
                 "--base-case", retry_compiled,
                 "--output-dir", str(retry_exp_dir),
                 "--output", str(retry_run_path),
                 "--timeout", "21600"],
                cwd=repo_root,
                timeout=22000,
                env=env,
            )
            print(out2r)
            retry_run = _read_json(retry_run_path, {})
            retry_ok = (rc2r == 0) and (str(retry_run.get("status", "")).upper() == "OK")
            if retry_ok:
                print(f"[OED][runtime] retry {attempt} SUCCEEDED — using retry case downstream.")
                rc = 0
                run_ok = True
                exp_dir = retry_exp_dir
                compiled_case = retry_compiled
                runtime_result = retry_apply
                runtime_run_result = retry_run
                runtime_result["retry_succeeded"] = True
                runtime_result["retry_attempts_used"] = attempt
                runtime_result.pop("runtime_apply_error", None)
                runtime_result.pop("compile_error_hint", None)
                action = revised
                retry_log.append({"attempt": attempt, "stage": "run", "result": "OK"})
                break

            # Run failed — extract the new FOAM error from this attempt's
            # logs and feed it into the next revision round.
            new_blob = ""
            for lp in (retry_run.get("log_paths") or [])[:8]:
                try:
                    t = Path(lp).read_text(encoding="utf-8", errors="replace")[-4000:]
                    new_blob += f"\n--- {Path(lp).name} ---\n{t}"
                except Exception:
                    pass
            new_scan = (retry_run.get("stderr_tail", "") or "") + "\n" \
                + (retry_run.get("stdout_tail", "") or "") + "\n" + new_blob
            new_hint = ""
            for line in new_scan.splitlines():
                ls = line.strip()
                if not ls:
                    continue
                low = ls.lower()
                if ("fatal error" in low or " error:" in low or "fatal io error" in low
                        or "no known conversion" in low or "no matching function" in low
                        or "was not declared" in low):
                    new_hint = ls[:400]
                    break
            if not new_hint:
                for line in new_scan.splitlines():
                    ls = line.strip()
                    if not ls:
                        continue
                    if "error" in ls.lower() or "fatal" in ls.lower():
                        new_hint = ls[:400]
                        break
            if not new_hint:
                new_hint = (retry_run.get("error") or err2r[-400:] or "retry simpleFoam failed")[:400]

            print(f"[OED][runtime] retry {attempt}: run FAILED — {new_hint[:120]}")
            retry_log.append({"attempt": attempt, "stage": "run", "error": new_hint})
            current_action = revised
            current_hint = new_hint
            current_ctx = new_blob[-3500:]

        # Persist retry trail on the runtime_result so the next planner
        # iteration sees the full progression.
        if retry_log:
            runtime_result["retry_attempted"] = True
            runtime_result["retry_log"] = retry_log
            if not run_ok:
                runtime_result["retry_error"] = retry_log[-1].get("error", "")[:400] \
                    if isinstance(retry_log[-1], dict) else ""

    # Run-validity gate: catch the harness-level "rc=0 but no flow solver ran"
    # failure mode (e.g. Allrun blockMesh-only, no serial fallback for parallel
    # block). Skip comparator entirely on RUN_INVALID.
    gate_result = _run_validity_gate(
        case_dir=exp_dir, baseline_metrics=baseline_metrics,
        runtime_run_result=runtime_run_result, base_case=Path(base) if base else None,
    )
    if not gate_result.get("valid", True):
        print(f"[OED][gate] RUN_INVALID — {gate_result.get('reason', '')[:240]}")
        return {
            "compiled_model_name": class_name,
            "compiled_model_description": model_desc,
            "compiled_case_dir": compiled_case,
            "case_dir": str(exp_dir),
            "status": "RUN_INVALID",
            "interpreter_reason": gate_result.get("reason", "RUN_INVALID")[:500],
            "metrics_summary": "",
            "is_runtime": True,
            "modification_category": runtime_result.get("category", ""),
            "runtime_apply_error": runtime_result.get("runtime_apply_error", ""),
            "compile_error_hint": runtime_result.get("compile_error_hint", ""),
            "run_validity": gate_result,
        }

    # Only run viz / interpret when the case actually converged. Running them
    # on a FAILED runtime case is wasteful — there is no converged solution
    # to visualize, and the viz/interpret LLM calls can stall indefinitely
    # on streaming responses (we've seen 15+ min hangs).
    if run_ok:
        _run_interpret(exp_dir, repo_root, timeline_path, env=env, objective_contract=objective_contract)
        metrics = _extract_case_metrics(
            exp_dir,
            starter_dir=starter_dir,
            repo_root=repo_root,
            starter_understanding=starter_understanding,
            objective_contract=objective_contract,
        )
        decision = _read_json(exp_dir / "decision.json", {})
        status = decision.get("status", "UNKNOWN")
        interp_reason = str(decision.get("reason", ""))[:500]
    else:
        # Skip viz/interpret. Construct a synthetic FAILED result with the
        # captured runtime error as the interpreter_reason so the planner LLM
        # learns from the failure on the next iteration.
        print(f"[OED][runtime] skipping viz/interpret — case did not converge (rc={rc}).")
        metrics = ""
        status = "FAILED"
        rae = runtime_result.get("runtime_apply_error", "") or runtime_result.get("compile_error_hint", "")
        interp_reason = (
            f"Runtime case failed before metrics could be extracted. "
            f"Captured error: {rae or '(none)'}"
        )[:500]

    return {
        "compiled_model_name": class_name,
        "compiled_model_description": model_desc,
        "compiled_case_dir": compiled_case,
        "case_dir": str(exp_dir),
        "status": status,
        "interpreter_reason": interp_reason,
        "metrics_summary": metrics,
        "is_runtime": True,
        "modification_category": runtime_result.get("category", ""),
        "runtime_apply_error": runtime_result.get("runtime_apply_error", ""),
        "compile_error_hint": runtime_result.get("compile_error_hint", ""),
        "run_validity": gate_result,
    }


def _run_investigate_runtime_iteration(
    *,
    iteration: int,
    action: Dict[str, Any],
    history: List[Dict[str, Any]],
    run_dir: Path,
    repo_root: Path,
    base_case_dir: str,
    topic: str,
    timeline_path: Path,
    starter_understanding: Dict[str, Any],
    starter_dir: Optional[Path] = None,
    objective_contract: Optional[Dict[str, Any]] = None,
    env: Dict[str, str],
    compiled_models: List[Dict[str, Any]],
    baseline_metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Investigate a previous RUN_INVALID iteration. LLM classifies root cause and
    proposes either a harness patch (apply + rerun + regate) or a model patch
    (downgrade to REVISE so the next planner iteration emits a corrected
    code_mod).
    """
    iter_dir = run_dir / f"iter_{iteration:03d}_investigate_runtime"
    iter_dir.mkdir(parents=True, exist_ok=True)

    # 1) Locate target history entry.
    target_iter = action.get("target_iteration")
    target = None
    if isinstance(target_iter, int):
        for h in history:
            if isinstance(h, dict) and int(h.get("iteration", -1)) == int(target_iter):
                target = h
                break
    if target is None:
        # Default to most recent RUN_INVALID.
        for h in reversed(history):
            if isinstance(h, dict) and str(h.get("status", "")).upper() == "RUN_INVALID":
                target = h
                break
    if target is None:
        return {
            "status": "FAILED",
            "case_dir": str(iter_dir),
            "interpreter_reason": "investigate_runtime: no RUN_INVALID iteration found in history."[:500],
            "metrics_summary": "",
        }

    target_case_dir = action.get("target_case_dir") or target.get("case_dir") or ""
    if not target_case_dir or not Path(target_case_dir).is_dir():
        return {
            "status": "FAILED",
            "case_dir": str(iter_dir),
            "interpreter_reason": (
                f"investigate_runtime: target_case_dir not found ({target_case_dir})."
            )[:500],
            "metrics_summary": "",
        }
    target_case = Path(target_case_dir)

    rv = target.get("run_validity") or {}
    bundle_path = rv.get("diagnostic_bundle_path", "") or str(target_case / "run_validity_diagnostic.json")
    bundle: Dict[str, Any] = {}
    try:
        bundle = json.loads(Path(bundle_path).read_text(encoding="utf-8"))
    except Exception:
        bundle = {}
    allrun_text = ""
    allrun_path = target_case / "Allrun"
    if allrun_path.is_file():
        try:
            allrun_text = allrun_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            allrun_text = ""

    # Original action JSON (runtime_action.json or agentic_result.json).
    action_json: Dict[str, Any] = {}
    for cand_name in ("runtime_action.json", "agentic_result.json"):
        # Look in target_case and its parent (iter_<NNN>_code_mod dir).
        for base in (target_case, target_case.parent):
            cand = Path(base) / cand_name
            if cand.is_file():
                try:
                    action_json = json.loads(cand.read_text(encoding="utf-8"))
                except Exception:
                    action_json = {}
                break
        if action_json:
            break

    log_tails = bundle.get("log_tails", {}) or {}

    # 2) LLM classification.
    try:
        scripts_dir = Path(__file__).resolve().parent
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        import run_validity as _rv  # type: ignore
        investigation = _rv.investigate_runtime_llm(
            diagnostic_bundle=bundle, allrun_text=allrun_text,
            action_json=action_json, log_tails=log_tails,
        )
    except Exception as exc:
        investigation = {
            "root_cause_class": "other",
            "explanation": f"investigator import/call failed: {exc}",
            "patch_target": "model",
            "patch": {"files": [], "rerun_strategy": "downgrade_to_revise"},
            "confidence": 0.0,
        }

    record: Dict[str, Any] = {
        "iteration": iteration,
        "target_iteration": target.get("iteration"),
        "target_case_dir": str(target_case),
        "investigation": investigation,
        "applied_patch_files": [],
        "rerun_result": None,
        "timestamp_utc": __import__("datetime").datetime.utcnow().isoformat() + "Z",
    }

    patch_target = str(investigation.get("patch_target", "model")).lower().strip()
    patch = investigation.get("patch", {}) or {}
    files = patch.get("files", []) if isinstance(patch, dict) else []
    rerun_strategy = str(patch.get("rerun_strategy", "downgrade_to_revise")).lower().strip()

    # 3) Apply patch.
    if patch_target == "harness" and isinstance(files, list) and files:
        applied = []
        for f in files:
            if not isinstance(f, dict):
                continue
            rel = str(f.get("path", "")).strip()
            new_content = f.get("new_content", "")
            if not rel or new_content is None:
                continue
            tgt = (target_case / rel).resolve()
            # Safety: tgt must be inside target_case.
            try:
                tgt.relative_to(target_case.resolve())
            except Exception:
                continue
            try:
                if tgt.is_file():
                    orig = tgt.read_text(encoding="utf-8", errors="replace")
                    backup = tgt.with_suffix(tgt.suffix + ".preflight_original")
                    try:
                        backup.write_text(orig, encoding="utf-8")
                    except Exception:
                        pass
                tgt.parent.mkdir(parents=True, exist_ok=True)
                tgt.write_text(new_content, encoding="utf-8")
                if rel.lower() == "allrun" or rel.endswith("/Allrun"):
                    try:
                        os.chmod(tgt, 0o755)
                    except Exception:
                        pass
                applied.append({"path": rel, "rationale": str(f.get("rationale", ""))[:400]})
            except Exception as exc:
                applied.append({"path": rel, "error": f"{exc}"[:200]})
        record["applied_patch_files"] = applied

        if rerun_strategy == "rerun_same_model":
            # Rerun foam_run_simple on the patched case.
            rerun_path = iter_dir / "rerun_runtime_run_result.json"
            try:
                _maybe_preflight_allrun(target_case, repo_root)
            except Exception as _ex:
                print(f"[OED][preflight] non-fatal: {_ex}")
            rc, out, err = _call(
                [sys.executable, "scripts/foam_run_simple.py",
                 "--base-case", str(target_case),
                 "--output-dir", str(target_case),
                 "--output", str(rerun_path),
                 "--timeout", "21600"],
                cwd=repo_root, timeout=22000, env=env,
            )
            print(out)
            rerun_result = _read_json(rerun_path, {})
            gate_result = _run_validity_gate(
                case_dir=target_case, baseline_metrics=baseline_metrics,
                runtime_run_result=rerun_result,
                base_case=Path(base_case_dir) if base_case_dir else None,
            )
            record["rerun_result"] = {"rc": rc, "gate": gate_result}
            _write_json(iter_dir / "investigation.json", record)
            if gate_result.get("valid", False):
                # Run interpret + score.
                _run_interpret(target_case, repo_root, timeline_path, env=env,
                               objective_contract=objective_contract)
                metrics = _extract_case_metrics(
                    target_case,
                    starter_dir=starter_dir,
                    repo_root=repo_root,
                    starter_understanding=starter_understanding,
                    objective_contract=objective_contract,
                )
                decision = _read_json(target_case / "decision.json", {})
                return {
                    "case_dir": str(target_case),
                    "status": decision.get("status", "UNKNOWN"),
                    "interpreter_reason": (
                        f"investigate_runtime: harness patch applied; rerun valid. "
                        f"{str(decision.get('reason', ''))[:300]}"
                    )[:500],
                    "metrics_summary": metrics,
                    "investigation": investigation,
                    "run_validity": gate_result,
                }
            # Still invalid after harness patch — surface FAILED.
            return {
                "case_dir": str(target_case),
                "status": "FAILED",
                "interpreter_reason": (
                    f"investigate_runtime: harness patch did not fix RUN_INVALID. "
                    f"{gate_result.get('reason', '')[:300]}"
                )[:500],
                "metrics_summary": "",
                "investigation": investigation,
                "run_validity": gate_result,
            }

    # Model patch (or empty patch list / downgrade): record + return REVISE.
    _write_json(iter_dir / "investigation.json", record)
    return {
        "case_dir": str(iter_dir),
        "status": "REVISE",
        "interpreter_reason": (
            f"investigate_runtime: classified root_cause={investigation.get('root_cause_class')} "
            f"patch_target={patch_target}. Downgraded to REVISE so the next iteration "
            f"can propose a corrected code_mod. Explanation: "
            f"{str(investigation.get('explanation', ''))[:300]}"
        )[:500],
        "metrics_summary": "",
        "investigation": investigation,
    }


def _run_experiment_iteration(
    *,
    iteration: int,
    action: Dict[str, Any],
    run_dir: Path,
    repo_root: Path,
    topic: str,
    timeline_path: Path,
    starter_understanding: Dict[str, Any],
    starter_dir: Optional[Path] = None,
    objective_contract: Optional[Dict[str, Any]] = None,
    compiled_models: List[Dict[str, Any]],
    env: Optional[Dict[str, str]] = None,
    baseline_metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run an experiment with an already-compiled model, different parameters."""
    # Include a slug for the reused model or variant purpose.
    slug = _slugify_variant(
        action.get("variant_name")
        or action.get("model_name_to_reuse")
        or action.get("model_description")
        or ""
    )
    slug_part = f"_{slug}" if slug else ""
    iter_dir = run_dir / f"iter_{iteration:03d}_experiment{slug_part}"
    iter_dir.mkdir(parents=True, exist_ok=True)

    # Find the compiled model to reuse
    model_name = action.get("model_name_to_reuse", "")
    compiled = next((m for m in compiled_models if m["name"] == model_name), None)
    if compiled is None and compiled_models:
        compiled = compiled_models[-1]  # fall back to most recent
    base_case = compiled["case_dir"] if compiled else ""
    is_runtime_model = bool(compiled and compiled.get("is_runtime"))

    fp = starter_understanding.get("flow_parameters", {}) or {}
    params = {**fp, **action.get("parameters", {})}
    params.pop("geometry", None)
    model_desc = action.get("model_description", f"parameter sweep iter {iteration}")
    ref_info = starter_understanding.get("reference_data", {}) or {}
    exp_dir = iter_dir / "experiment"

    if is_runtime_model and base_case:
        # Runtime experiment: copy the case, patch the fvModels coefficients
        # with the LLM's parameter overrides, run via the simple Allrun runner.
        # The FoamAgent reviewer would otherwise rewrite the runtime fvModels
        # entry into a (non-existent) class type and crash the run.
        print(f"[OED][exp] runtime parameter sweep on {model_name}; using foam_run_simple.")
        # Copy first so we can patch coefficients before running.
        try:
            from shutil import copytree, rmtree
            if exp_dir.exists():
                rmtree(exp_dir)
            # use code_mod_runtime's case copy helper for consistent ignore-rules
            scripts_dir = repo_root / "scripts"
            if str(scripts_dir) not in sys.path:
                sys.path.insert(0, str(scripts_dir))
            import code_mod_runtime as _crt  # type: ignore
            _crt._copy_case(Path(base_case), exp_dir)
            case_dir_resolved = _crt._resolve_case_dir(exp_dir)
            # Patch fvModels coefficient values from action.parameters.
            # Generic: iterate every numeric key in action.parameters and
            # update its top-level entry inside constant/fvModels.
            fvmodels = case_dir_resolved / "constant" / "fvModels"
            if fvmodels.is_file():
                txt = fvmodels.read_text(encoding="utf-8")
                user_params = action.get("parameters", {}) or {}
                patched = []
                for k, v in user_params.items():
                    if not re.match(r"^[A-Za-z_]\w*$", str(k)):
                        continue
                    try:
                        fv = float(v)
                    except Exception:
                        continue
                    pat = re.compile(rf"(^|\n)(\s*){re.escape(k)}\s+([^;]+);", re.MULTILINE)
                    replaced = [False]
                    def _sub(m: re.Match[str]) -> str:
                        replaced[0] = True
                        return f"{m.group(1)}{m.group(2)}{k}    {fv};"
                    txt = pat.sub(_sub, txt, count=1)
                    if replaced[0]:
                        patched.append(f"{k}={fv}")
                fvmodels.write_text(txt, encoding="utf-8")
                if patched:
                    print(f"[OED][exp] patched fvModels coefficients: {', '.join(patched)}")
        except Exception as exc:
            print(f"[OED][exp] runtime case prep failed (non-fatal): {exc}")

        runtime_run_result_path = iter_dir / "runtime_run_result.json"
        # Allrun pre-flight on the prepared experiment case before running.
        try:
            _maybe_preflight_allrun(Path(exp_dir), repo_root)
        except Exception as _ex:
            print(f"[OED][preflight] non-fatal: {_ex}")
        rc, out, err = _call(
            [sys.executable, "scripts/foam_run_simple.py",
             "--base-case", str(exp_dir),
             "--output-dir", str(exp_dir),
             "--output", str(runtime_run_result_path),
             "--timeout", "21600"],
            cwd=repo_root,
            timeout=22000,
            env=env,
        )
        print(out)
        runtime_run_result = _read_json(runtime_run_result_path, {})
        # Run-validity gate before scoring.
        gate_result = _run_validity_gate(
            case_dir=Path(exp_dir), baseline_metrics=baseline_metrics,
            runtime_run_result=runtime_run_result, base_case=Path(base_case) if base_case else None,
        )
        if not gate_result.get("valid", True):
            print(f"[OED][gate] RUN_INVALID — {gate_result.get('reason', '')[:240]}")
            return {
                "case_dir": str(exp_dir),
                "status": "RUN_INVALID",
                "interpreter_reason": gate_result.get("reason", "RUN_INVALID")[:500],
                "metrics_summary": "",
                "run_validity": gate_result,
            }
        run_ok = (rc == 0) and (str(runtime_run_result.get("status", "")).upper() == "OK")
        if run_ok:
            _run_interpret(exp_dir, repo_root, timeline_path, env=env, objective_contract=objective_contract)
            metrics = _extract_case_metrics(
                exp_dir,
                starter_dir=starter_dir,
                repo_root=repo_root,
                starter_understanding=starter_understanding,
                objective_contract=objective_contract,
            )
            decision = _read_json(exp_dir / "decision.json", {})
            return {
                "case_dir": str(exp_dir),
                "status": decision.get("status", "UNKNOWN"),
                "interpreter_reason": str(decision.get("reason", ""))[:500],
                "metrics_summary": metrics,
            }
        # Failed run — surface the captured error so the planner LLM learns.
        rae_blob = (runtime_run_result.get("stderr_tail") or "") + "\n" + (runtime_run_result.get("stdout_tail") or "")
        for lp in (runtime_run_result.get("log_paths") or [])[:5]:
            try:
                rae_blob += "\n" + Path(lp).read_text(encoding="utf-8", errors="replace")[-2000:]
            except Exception:
                pass
        hint = ""
        for line in rae_blob.splitlines():
            ls = line.strip()
            if not ls:
                continue
            if "fatal" in ls.lower() or " error:" in ls.lower() or "fatal io error" in ls.lower():
                hint = ls[:400]
                break
        return {
            "case_dir": str(exp_dir),
            "status": "FAILED",
            "interpreter_reason": f"Runtime experiment did not converge. Captured: {hint or 'unknown'}"[:500],
            "metrics_summary": "",
            "compile_error_hint": hint,
            "runtime_apply_error": hint,
        }

    # Class-derivation parameter sweep (legacy path). This went through
    # foam_run.py, whose LLM was what actually applied `params` to the case —
    # and foam_run.py needs Foam-Agent vendored. There is no safe drop-in:
    # copying and running the base case would execute the *unmodified*
    # configuration and still yield a real, scoreable metric, i.e. a confident
    # wrong answer attributed to a parameter set that was never applied.
    # Refuse instead. Runtime-model experiments (the branch above) are
    # unaffected and are what every current proposer emits.
    print("[OED] class-derivation parameter sweep is unsupported; use a runtime-coefficient model")
    return {
        "case_dir": str(exp_dir),
        "status": "FAILED",
        "interpreter_reason": (
            "Class-derivation parameter sweeps are no longer supported (they required "
            "Foam-Agent's reviewer to apply the parameters). Propose this as a runtime "
            "coefficient change instead, which the runtime path applies directly."
        ),
        "metrics_summary": "",
    }


def _latest_case_time(case_dir: Path) -> Optional[float]:
    """The largest numeric time directory in a finished OpenFOAM case.

    Time 0 is excluded deliberately: a case whose only time directory is 0 has
    not run, and scoring it would compare initial conditions against reference
    data. Returns None in that case so the caller can refuse rather than
    silently score nothing.
    """
    try:
        times = []
        for child in Path(case_dir).iterdir():
            if not child.is_dir():
                continue
            try:
                value = float(child.name)
            except ValueError:
                continue
            if value > 0:
                times.append(value)
        return max(times) if times else None
    except Exception:
        return None


def _comparator_exemplar_text(
    *, search_roots: List[Path], metrics: List[Dict[str, Any]], fallback: str = ""
) -> str:
    """Source of the study's own comparator, to author a new one from.

    A discovered script usually cannot be used as-is — this harness scores by
    reading a `METRIC <name>: <value>` line from stdout, and a hand-written
    comparator generally reports differently. But it is the authoritative
    statement of how the metric is computed, including the edge cases a fresh
    derivation gets wrong, so it is worth far more as an exemplar than as a
    rejected candidate.
    """
    try:
        from oed_extensions import discover_existing_comparators
    except Exception:
        return fallback
    try:
        found = discover_existing_comparators(search_roots=search_roots, metrics=metrics)
    except Exception:
        return fallback
    for path in found.values():
        try:
            text = Path(path).read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if text.strip():
            print(f"[OED-EXT][phase1] authoring from the study's own comparator: {path}", flush=True)
            return text
    return fallback


def _run_interpret(
    case_dir: Path,
    repo_root: Path,
    timeline_path: Path,
    env: Optional[Dict[str, str]] = None,
    *,
    objective_contract: Optional[Dict[str, Any]] = None,
) -> None:
    """Run viz + interpret on a finished case.

    Score-only fast path: if the OED has a bound comparator that produces a
    numeric primary_score for this case, skip the LLM-driven viz/interpret
    entirely and write a synthetic decision.json (status=UNKNOWN, scored).
    The downstream baseline gate already upgrades UNKNOWN→PROCEED/REVISE
    based on score-vs-baseline, so the planner gets the same signal at
    near-zero latency. This applies to every OED with a usable reference
    metric (turbulence/Cf, drag, heat flux, viscosity profile, etc.) —
    generic across topics.

    Vision-only path (when no comparator score is available): run the
    existing viz_creator + vision-LLM scoring pipeline.
    """
    decision_path = case_dir / "decision.json"

    # Score-only fast path. Try a deterministic comparator first.
    if objective_contract:
        try:
            comp_out = _run_bound_comparator(case_dir, objective_contract)
            if comp_out:
                extracted = _extract_error_metrics(comp_out)
                primary = _choose_primary_score(extracted)
                if primary is not None:
                    _write_json(decision_path, {
                        "status": "UNKNOWN",
                        "confidence": 0.0,
                        "reason": (
                            f"interpret skipped — using deterministic comparator score "
                            f"{primary.get('metric','?')}={primary.get('value','?')} "
                            f"(direction={primary.get('direction','min')}). "
                            f"Vision interpret is not invoked when a numeric score is available."
                        )[:500],
                        "suggested_changes": [],
                        "raw": {"comparator_output_head": str(comp_out)[:800]},
                        "score_only_path": True,
                    })
                    print(f"[OED][interpret] score-only path: skipping LLM interpret "
                          f"(score={primary})")
                    append_timeline_event(timeline_path, {
                        "stage": "interpret",
                        "case_id": case_dir.name,
                        "status": "score_only_skipped",
                        "score": primary,
                    })
                    return
        except Exception as ex:
            print(f"[OED][interpret] score-only short-circuit failed "
                  f"({type(ex).__name__}: {ex}); falling through to vision interpret")

    figs_dir = case_dir / "figs"
    # 1 hour per stage. Codex calls under load can take 30-120s each, and
    # viz_creator does up to N attempts × 2 LLM calls each plus a final
    # vision-review LLM call. The previous 10-min cap consistently killed
    # legitimate (slow but progressing) interpret runs mid-loop, leaving
    # the iteration with status=UNKNOWN. Generic across providers.
    INTERP_TIMEOUT_S = 3600
    decision_path = case_dir / "decision.json"
    interp_failure_reason = ""
    try:
        rc, out, _ = _call(
            [sys.executable, "scripts/viz.py", "--case", str(case_dir),
             "--mode", "interpret", "--output", str(figs_dir)],
            cwd=repo_root,
            env=env,
            timeout=INTERP_TIMEOUT_S,
        )
        print(out)
    except Exception as exc:
        msg = f"viz.py timed out or failed: {exc}"
        print(f"[OED][interpret] {msg}")
        interp_failure_reason = msg
    try:
        rc, out, _ = _call(
            [sys.executable, "scripts/interpret.py",
             "--case", str(case_dir),
             "--figs", str(figs_dir),
             "--output", str(decision_path),
             "--timeline", str(timeline_path)],
            cwd=repo_root,
            env=env,
            timeout=INTERP_TIMEOUT_S,
        )
        print(out)
    except Exception as exc:
        msg = f"interpret.py timed out or failed: {exc}"
        print(f"[OED][interpret] {msg}")
        interp_failure_reason = (interp_failure_reason + " | " + msg) if interp_failure_reason else msg

    # Fallback: if interpret.py never wrote decision.json, write a synthetic
    # one so downstream stages don't see a missing file. Status=UNKNOWN
    # signals to the planner that scoring failed (not the simulation).
    if not decision_path.is_file():
        try:
            _write_json(decision_path, {
                "status": "UNKNOWN",
                "confidence": 0.0,
                "reason": (interp_failure_reason or "interpret.py did not produce decision.json")[:500],
                "suggested_changes": [],
                "raw": {},
                "fallback": True,
            })
            print(f"[OED][interpret] wrote fallback decision.json (status=UNKNOWN).")
        except Exception as exc:
            print(f"[OED][interpret] could not write fallback decision.json: {exc}")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_open_ended_discovery(
    *,
    run_dir: Path,
    repo_root: Path,
    topic: str,
    budget: int,
    starter_understanding: Dict[str, Any],
    starter_dir: Optional[Path] = None,
    lit_path: Path,
    base_case_dir: str,
    timeline_path: Path,
    env: Dict[str, str],
    verbose: bool = True,
    baseline_metrics: Optional[Dict[str, Any]] = None,
    saturation_window: Optional[int] = None,
    setup_only: bool = False,
    # OED extensions:
    # - multi_metric is always-on (no flag): the LLM proposes metrics from
    #   topic+ref+baseline at startup; if ≥1 useful metric, the loop tracks
    #   the full vector and uses LLM-as-judge per iteration. Falls back to
    #   single-metric automatically when only one metric is relevant.
    # - diversity_mode and multi_flow remain explicit (they are search-policy
    #   choices, not data-driven decisions).
    multi_metric: bool = True,
    diversity_mode: str = "off",   # off | hybrid | aggressive
    diversity_far_ratio: float = 0.3,
    multi_flow: bool = False,
    starter_dirs: Optional[List[Path]] = None,
    metric_aggregator: str = "llm_judge",  # llm_judge (default) | weighted_sum | min_improvement | pareto_rank (legacy)
) -> Dict[str, Any]:
    baseline_metrics = baseline_metrics or {}
    baseline_score_value: Optional[float] = None

    # Hard cap on lifetime budget. Each extension request is also limited to
    # 50% of CURRENT budget (per-request), but accumulating extensions cannot
    # push the total past `_budget_hard_ceiling`. Prevents runaway resource
    # use even when the LLM keeps requesting more.
    _BUDGET_HARD_CAP_MULTIPLIER = 1.5
    _budget_original = int(budget)
    _budget_hard_ceiling = int(_budget_original * _BUDGET_HARD_CAP_MULTIPLIER)
    # Direction defaults to "min" (lower is better — RMSE / L2 / MAE family)
    # but is honored if the comparator declared otherwise (e.g. correlation,
    # accuracy, agreement → "max"). Generic across QoIs.
    baseline_direction: str = "min"
    # baseline_final_time: resolved from baseline_metrics. Plumbed into metric
    # specs and comparator invocations for time-pinned scoring.
    baseline_final_time: Optional[float] = None
    if isinstance(baseline_metrics, dict):
        ps = baseline_metrics.get("primary_score")
        if isinstance(ps, dict):
            try:
                baseline_score_value = float(ps.get("value"))
            except Exception:
                baseline_score_value = None
            d = str(ps.get("direction", "")).strip().lower()
            if d in ("min", "max"):
                baseline_direction = d
        try:
            bft = baseline_metrics.get("baseline_final_time")
            if bft is not None:
                baseline_final_time = float(bft)
        except Exception:
            baseline_final_time = None

    if baseline_final_time is None and base_case_dir:
        # Read it off the case. `baseline_metrics.json` is written by a later
        # stage, so on a fresh --setup-only there is nothing to read it from
        # and it stayed None — and a None here is not harmless: comparators are
        # required to pin scoring to an explicit time and to refuse rather than
        # guess, so the self-test invoked them with no --baseline-time, every
        # one correctly returned nan, and the study ended with a null baseline
        # score after five authoring attempts. The finished case knows its own
        # final time; nothing needs to be inferred.
        baseline_final_time = _latest_case_time(Path(base_case_dir))
        if baseline_final_time is not None:
            print(
                f"[OED-EXT][phase1] baseline final time read from the case: {baseline_final_time}",
                flush=True,
            )
        else:
            print(
                "[OED-EXT][phase1] WARNING: could not determine the baseline case's final time; "
                "comparators cannot be pinned and will refuse to score",
                flush=True,
            )
    disc_dir = run_dir / "open_ended_discovery"
    disc_dir.mkdir(parents=True, exist_ok=True)
    objective_contract = _resolve_objective_contract(
        disc_dir=disc_dir,
        starter_understanding=starter_understanding,
        starter_dir=starter_dir,
        repo_root=repo_root,
    )

    # ---------------------------------------------------------------- #
    # OED extensions setup (Phase 1: multi-metric, Phase 2: diversity,  #
    # Phase 3: multi-flow). All-default-off preserves prior behaviour.   #
    # ---------------------------------------------------------------- #
    ext_state: Dict[str, Any] = {
        "multi_metric": bool(multi_metric),
        "diversity_mode": str(diversity_mode or "off"),
        "diversity_far_ratio": float(diversity_far_ratio),
        "multi_flow": bool(multi_flow),
        "metric_aggregator": str(metric_aggregator or "weighted_sum"),
        "metric_specs": [],
        "bound_comparators": {},
        "baseline_metric_vector": {},
        "flows": {},                # flow_id -> Path
        "per_flow_metric_specs": {},
        "per_flow_bound_comparators": {},
        "per_flow_baseline_vector": {},
    }
    try:
        if str(Path(__file__).parent) not in sys.path:
            sys.path.insert(0, str(Path(__file__).parent))
        import oed_extensions as _oedx  # type: ignore
    except Exception as exc:
        _oedx = None  # type: ignore
        if multi_metric or multi_flow or (diversity_mode and diversity_mode != "off"):
            print(f"[OED-EXT] WARNING: oed_extensions import failed: {exc}. "
                  "Disabling extensions; falling back to single-metric mode.")
            ext_state["multi_metric"] = False
            ext_state["multi_flow"] = False
            ext_state["diversity_mode"] = "off"

    # Phase 3: detect flows
    if _oedx is not None and ext_state["multi_flow"]:
        flows = _oedx.detect_multi_flow_setup(
            starter_dir=starter_dir, starter_dirs=starter_dirs,
        )
        ext_state["flows"] = {fid: str(p) for fid, p in flows.items()}
        if len(flows) < 2:
            print("[OED-EXT][phase3] multi_flow requested but only one flow found; "
                  "running in single-flow mode.")
            ext_state["multi_flow"] = False

    # Phase 1: build metric specs + author missing comparators + baseline vector
    # FAST PATH: if the new unified `metric_setup` stage already populated
    # `<run_dir>/metric_specs.json` with a non-empty list AND the recorded
    # comparator script exists, skip the legacy proposer + per-metric author
    # + per-metric verifier loop entirely. Bind directly. The legacy code
    # below remains the FALLBACK when metric_setup is unavailable / failed.
    _metric_setup_specs_path = run_dir / "metric_specs.json"
    _metric_setup_used = False
    if _oedx is not None and ext_state["multi_metric"] and _metric_setup_specs_path.is_file():
        try:
            _ms_doc = json.loads(_metric_setup_specs_path.read_text(encoding="utf-8"))
        except Exception:
            _ms_doc = None
        _ms_metrics: List[Dict[str, Any]] = []
        _ms_comparator = ""
        if isinstance(_ms_doc, dict):
            _raw_specs = _ms_doc.get("metrics") or []
            if isinstance(_raw_specs, list):
                _ms_metrics = [s for s in _raw_specs if isinstance(s, dict) and s.get("name")]
            _ms_comparator = str(_ms_doc.get("comparator_script", "") or "")
        if _ms_metrics and _ms_comparator and Path(_ms_comparator).is_file():
            print(f"[OED-EXT][phase1] metric_setup output detected at {_metric_setup_specs_path}; "
                  f"binding {len(_ms_metrics)} metric(s) directly (fast path).")
            ref_files_fast = [Path(p) for p in (objective_contract.get("reference_files") or [])
                              if Path(p).is_file()]
            base_case_path_fast = ""
            if isinstance(baseline_metrics, dict):
                base_case_path_fast = str(baseline_metrics.get("baseline_case_dir", "") or "")
            if not base_case_path_fast:
                base_case_path_fast = base_case_dir
            base_case_p_fast = Path(base_case_path_fast).expanduser().resolve() if base_case_path_fast else None

            for s in _ms_metrics:
                s.setdefault("baseline_final_time", baseline_final_time)
            _write_json(disc_dir / "metric_specs.json", _ms_metrics)
            ext_state["metric_specs"] = _ms_metrics

            bound_fast: Dict[str, Dict[str, Any]] = {}
            for m in _ms_metrics:
                nm = m["name"]
                bound_fast[nm] = {
                    "path": _ms_comparator,
                    "origin": "metric_setup",
                    "selftest_ok": True,
                    "selftest_value": m.get("baseline_value"),
                    "selftest_reason": "bound by metric_setup",
                    "verifier_verdict": m.get("verifier_verdict", "AGREE"),
                    "binding_kind": "metric_setup",
                    "attempts": 1,
                    "final_method": "metric_setup",
                }
            _write_json(disc_dir / "bound_comparators.json", bound_fast)
            ext_state["bound_comparators"] = bound_fast

            if base_case_p_fast and base_case_p_fast.is_dir() and ref_files_fast:
                try:
                    bv = _oedx.compute_metric_vector(
                        case_dir=base_case_p_fast, bound_comparators=bound_fast,
                        reference_file=_reference_data_file(ref_files_fast),
                        baseline_final_time=baseline_final_time,
                        metric_specs=_ms_metrics,
                    )
                    _write_json(disc_dir / "baseline_metric_vector.json", bv)
                    ext_state["baseline_metric_vector"] = bv
                    print(f"[OED-EXT][phase1] (fast) baseline metric vector: {bv.get('metrics', {})}")
                except Exception as exc:
                    print(f"[OED-EXT][phase1] (fast) baseline vector compute failed: {exc}")
            _metric_setup_used = True

    if _oedx is not None and ext_state["multi_metric"] and not _metric_setup_used:
        try:
            ref_files = [Path(p) for p in (objective_contract.get("reference_files") or []) if Path(p).is_file()]
            # Pull a sample of postProcessing structure from baseline if available
            sample_pp_tree = ""
            sample_pp_data = ""
            base_case_path = ""
            if isinstance(baseline_metrics, dict):
                base_case_path = str(baseline_metrics.get("baseline_case_dir", "") or "")
            if not base_case_path:
                base_case_path = base_case_dir
            base_case_p = Path(base_case_path).expanduser().resolve() if base_case_path else None
            if base_case_p and (base_case_p / "postProcessing").is_dir():
                pp = base_case_p / "postProcessing"
                lines = []
                files = []
                for p in sorted(pp.rglob("*")):
                    rel = p.relative_to(base_case_p)
                    lines.append(f"  {rel}")
                    if p.is_file() and p.suffix in (".dat", ".csv", ".txt", ""):
                        files.append(p)
                sample_pp_tree = "\n".join(lines[:200])
                for pf in files[:3]:
                    try:
                        sample_pp_data += f"\n--- {pf.relative_to(base_case_p)} ---\n"
                        sample_pp_data += pf.read_text(errors="ignore")[:600]
                    except Exception:
                        pass

            _ref_search: List[Path] = []
            if starter_dir and Path(starter_dir).is_dir():
                _ref_search.append(Path(starter_dir))
            for _rf in ref_files:
                _rfp = Path(_rf).parent
                if _rfp.is_dir() and _rfp not in _ref_search:
                    _ref_search.append(_rfp)
            # The study already decided what it is judged on, before the mesh
            # gate ran, and wrote it with the full computation rule (see
            # manager/tools.py::_study_metrics). Re-deriving it here would be a
            # second, independent answer to a question already answered — the
            # pattern that produced Cf computed with Ub = 1.0 instead of the
            # case's 0.028, a factor of 1276, when one derivation had less
            # context than the other. Comparators are still authored below,
            # because the metric needs an executable form; what is reused is
            # the decision and its definition, so the authored comparator has
            # nothing to invent.
            specs = []
            _study_specs_path = run_dir / "study_metrics.json"
            if _study_specs_path.is_file():
                try:
                    _study_specs = json.loads(_study_specs_path.read_text(encoding="utf-8"))
                except Exception:
                    _study_specs = None
                if isinstance(_study_specs, list) and _study_specs:
                    specs = [dict(m) for m in _study_specs if isinstance(m, dict) and m.get("name")]
                    for _m in specs:
                        # propose_metric_set's schema, filled from the decision.
                        _m.setdefault("direction", "min")
                        _m.setdefault("description", "")
                        _m.setdefault("computation_hint", "")
                        _m.setdefault("data_source", "")
                        _m.setdefault("ref_column", "")
                        _m.setdefault("preferred_method", "pyvista")
                    print(
                        f"[OED-EXT][phase1] reusing the study's metric decision "
                        f"({[m['name'] for m in specs]}) from {_study_specs_path}",
                        flush=True,
                    )
            if not specs:
                specs = _oedx.propose_metric_set(
                    topic=topic,
                    starter_understanding=starter_understanding,
                    reference_files=ref_files,
                    sample_postprocessing=sample_pp_tree,
                    baseline_case_dir=base_case_p,
                    reference_search_paths=_ref_search,
                    out_dir=disc_dir,
                    baseline_metrics_path=run_dir / "baseline_metrics.json",
                    objective_contract=objective_contract,
                )
            if specs:
                # Stamp baseline_final_time into each spec so downstream
                # comparator invocations can pin scoring to that exact time.
                for s in specs:
                    if isinstance(s, dict):
                        s["baseline_final_time"] = baseline_final_time
                _write_json(disc_dir / "metric_specs.json", specs)
                ext_state["metric_specs"] = specs

                # Resolve comparators for each metric (existing or LLM-authored + self-test)
                #
                # `starter_dir` is the CASE directory
                # (starter_oed_turbulence/periodic_hill_sa). The study's own
                # working comparator lives beside it, not inside it
                # (starter_oed_turbulence/reference_data/compare_exactmatch_cf.py),
                # so searching only the case directory never sees it — and the
                # study then authors a fresh comparator instead of using the one
                # the prompt explicitly names as authoritative. Measured: five
                # authoring attempts, and the result returned nan because it
                # treated reference stations just outside the mesh range as
                # fatal, where the existing script handles them.
                #
                # The reference files are the reliable pointer: whatever
                # directory holds the reference data is where its reader lives.
                search_roots: List[Path] = []
                if starter_dir and starter_dir.is_dir():
                    search_roots.append(starter_dir.resolve())
                    if starter_dir.resolve().parent.is_dir():
                        search_roots.append(starter_dir.resolve().parent)
                for _rf in ref_files:
                    _rfp = Path(_rf).parent
                    if _rfp.is_dir() and _rfp.resolve() not in search_roots:
                        search_roots.append(_rfp.resolve())
                # Deliberately NOT repo_root. discover_existing_comparators
                # scores candidates by raw keyword counts, so a large unrelated
                # repo file outscores the study's own comparator: measured, it
                # picked scripts/skill_bootstrap.py over
                # reference_data/compare_exactmatch_cf.py, failed its self-test,
                # and authored a fresh comparator that returned nan. The
                # function's own docstring says it must not walk the repo; the
                # caller was adding it back. With no starter-local comparator,
                # finding nothing and authoring one is the correct outcome.
                # Every comparator is self-tested by being run with this as
                # its --reference, so it has to be the reference DATA. Taking
                # ref_files[0] blindly handed it whatever came first, and a
                # study that legitimately declares its authoritative scorer
                # among its reference files puts a .py there: the comparator
                # then tried to read Python source as CSV, crashed, was judged
                # broken, and a replacement was authored. Measured on
                # ph_codex_20260902_1806, where the starter's own scorer --
                # which reproduces the study's stated baseline exactly -- was
                # discovered and rejected on this alone, twice over two nights.
                ref_for_self = _reference_data_file(
                    ref_files, (starter_dir or repo_root) / "reference.csv")
                bound = _oedx.resolve_metric_comparators(
                    metrics=specs, search_roots=search_roots,
                    reference_file=ref_for_self, flow_params=starter_understanding.get("flow_parameters", {}) or {},
                    baseline_case_dir=base_case_p, out_dir=disc_dir / "authored_comparators",
                    sample_pp_tree=sample_pp_tree, sample_pp_data=sample_pp_data,
                    # The study's own working comparator, as the exemplar the
                    # authored one is written from.
                    #
                    # This was `[:0]` — truncated to an empty string, so the
                    # authoring never saw it. That is why a freshly authored
                    # comparator reinvented the method and got it subtly wrong:
                    # it treated reference stations lying just outside the mesh
                    # coordinate range as fatal and returned nan, where the
                    # existing script masks them and returns 0.004297.
                    #
                    # The existing script cannot be used directly — it writes a
                    # summary and a plot rather than printing the
                    # `METRIC <name>: <value>` line this harness scores from,
                    # so its self-test correctly fails. What it can do is show
                    # the authored script exactly how the quantity is computed.
                    exemplar_text=_comparator_exemplar_text(
                        search_roots=search_roots,
                        metrics=specs,
                        fallback=objective_contract.get("comparator_script") or "",
                    ),
                    baseline_final_time=baseline_final_time,
                    topic=topic,
                )
                _write_json(disc_dir / "bound_comparators.json", bound)
                ext_state["bound_comparators"] = bound

                # Compute baseline vector
                if base_case_p and base_case_p.is_dir() and ref_files:
                    bv = _oedx.compute_metric_vector(
                        case_dir=base_case_p, bound_comparators=bound,
                        reference_file=_reference_data_file(ref_files),
                        baseline_final_time=baseline_final_time,
                        metric_specs=specs,
                    )
                    _write_json(disc_dir / "baseline_metric_vector.json", bv)
                    ext_state["baseline_metric_vector"] = bv
                    print(f"[OED-EXT][phase1] baseline metric vector: {bv.get('metrics', {})}")
                else:
                    print("[OED-EXT][phase1] baseline case dir not found; skipping baseline vector.")
            else:
                print("[OED-EXT][phase1] LLM proposed no metrics; falling back to single-metric.")
                ext_state["multi_metric"] = False
        except Exception as exc:
            print(f"[OED-EXT][phase1] setup failed: {exc}. Disabling multi_metric.")
            ext_state["multi_metric"] = False

    _write_json(disc_dir / "ext_state.json", ext_state)
    # End extensions setup
    # ---------------------------------------------------------------- #

    if baseline_score_value is None:
        # Derive it from the vector setup just computed. `baseline_score_value`
        # is read from baseline_metrics.json, which a later stage writes — so
        # on a fresh --setup-only it is always None, and the search then has no
        # baseline to measure candidates against or to apply the improvement
        # target to. The value is already on disk in baseline_metric_vector.json
        # (measured: cf_rmse = 0.004321, self-test verified); nothing needs to
        # be recomputed, only read.
        _bv = (ext_state.get("baseline_metric_vector") or {}).get("metrics") or {}
        _specs_for_primary = ext_state.get("metric_specs") or []
        _primary_name = ""
        for _s in _specs_for_primary:
            if isinstance(_s, dict) and _s.get("primary") and _s.get("name") in _bv:
                _primary_name = str(_s["name"])
                break
        if not _primary_name:
            # No metric declared primary: with a single metric there is no
            # ambiguity, and with several the first spec is the one the study
            # named first.
            for _s in _specs_for_primary:
                if isinstance(_s, dict) and _s.get("name") in _bv:
                    _primary_name = str(_s["name"])
                    break
        if not _primary_name and len(_bv) == 1:
            _primary_name = next(iter(_bv))
        if _primary_name:
            try:
                baseline_score_value = float(_bv[_primary_name])
                for _s in _specs_for_primary:
                    if isinstance(_s, dict) and _s.get("name") == _primary_name:
                        _d = str(_s.get("direction", "")).strip().lower()
                        if _d in ("min", "max"):
                            baseline_direction = _d
                        break
                print(
                    f"[OED-EXT][phase1] baseline score taken from the measured vector: "
                    f"{_primary_name} = {baseline_score_value} ({baseline_direction})",
                    flush=True,
                )
            except Exception:
                baseline_score_value = None

    if setup_only:
        # Everything a per-candidate caller needs to score independently is
        # now on disk: objective_contract.json (reference files),
        # bound_comparators.json (scored-metric comparator scripts, already
        # authored/discovered — never re-derive this per candidate),
        # ext_state.json. Return here instead of entering the iteration
        # loop — the deepagents manager drives iteration itself now (see
        # manager/tools.py's oed_propose_candidates / oed_run_*_candidate /
        # oed_record_candidate_results), so this script's own while-loop
        # below is only reached when it's run standalone (e.g. via
        # orchestrator_run.py), not from that path.
        _write_json(disc_dir / "baseline_score.json", {
            "value": baseline_score_value, "direction": baseline_direction,
        })
        return {
            "status": "setup_complete",
            "disc_dir": str(disc_dir),
            "objective_contract_path": str(disc_dir / "objective_contract.json"),
            "bound_comparators_path": str(disc_dir / "bound_comparators.json"),
            "baseline_score_path": str(disc_dir / "baseline_score.json"),
            "ext_state_path": str(disc_dir / "ext_state.json"),
            "baseline_score": baseline_score_value,
            "baseline_direction": baseline_direction,
        }

    # Resume from previous run if history.json exists
    history_path = disc_dir / "history.json"
    if history_path.is_file():
        history = _read_json(history_path, [])
        if not isinstance(history, list):
            history = []
        # Reconstruct budget_used: prefer per-entry recorded `cost`, fall back
        # to legacy defaults so old histories resume correctly.
        budget_used = 0
        for h in history:
            c = h.get("cost")
            if isinstance(c, (int, float)):
                budget_used += int(c)
                continue
            at = h.get("action_type")
            if at == "code_mod":
                budget_used += 2
            elif at == "experiment":
                budget_used += 1
        compiled_models = [
            {"name": h["compiled_model_name"],
             "description": h.get("compiled_model_description", ""),
             "case_dir": h.get("compiled_case_dir", ""),
             "is_runtime": bool(h.get("is_runtime")),
             "family": h.get("family") or SearchArchive.classify(
                 h.get("compiled_model_description", ""), h["compiled_model_name"])}
            for h in history
            if h.get("action_type") == "code_mod" and h.get("compiled_model_name")
        ]
        iteration = max((h.get("iteration", 0) for h in history), default=0)
        print(f"[OED] Resuming from history: {len(history)} entries, "
              f"budget_used={budget_used}/{budget}, iteration={iteration}")
    else:
        history = []
        compiled_models = []
        budget_used = 0
        iteration = 0

    # Search-policy archive: one elite (best-scoring history entry) per model
    # family, replacing the old flat single-chain + fixed-period diversity
    # nudge. See scripts/oed_search_archive.py for the full design rationale.
    # Replaying resumed history keeps exploration progress across a
    # pause/resume instead of resetting the archive to empty every time.
    search_archive = SearchArchive()
    if history:
        search_archive.replay(history, baseline_direction=baseline_direction)

    consecutive_scripts = 0          # track back-to-back python_script actions
    MAX_CONSECUTIVE_SCRIPTS = 2      # force a real action sooner
    # Track consecutive cheap-category code_mods. The runtime/dict_only paths
    # are useful but they cannot express structural model novelty — after N
    # consecutive cheap code_mods without a PROCEED, force a class_derivation
    # so the search actually exercises the agentic compile path. Generic
    # across modification kinds.
    consecutive_cheap_code_mods = 0
    MAX_CONSECUTIVE_CHEAP_CODE_MODS = 2
    compile_fail_counts: Dict[str, int] = {}
    blocked_compile_hints: List[str] = []

    append_timeline_event(timeline_path, {
        "stage": "open_ended_discovery",
        "event": "start" if not history else "resume",
        "budget": budget,
        "budget_used_at_start": budget_used,
        "topic": topic,
    })

    # One-time recon of $WM_PROJECT_DIR/src so the hypothesis-generation LLM
    # (and every downstream code_mod call) has ground-truth include paths and
    # class signatures for the installation actually present. Cached at the
    # discovery-root so all iterations share one result. Topic-agnostic — the
    # recon module reads the topic as a free-form string and navigates
    # whatever source tree is present.
    recon_context: Optional[Dict[str, Any]] = None
    try:
        recon_cache = disc_dir / "discovered_paths.json"
        wm_dir = os.environ.get("WM_PROJECT_DIR", "").strip()
        foam_src = Path(wm_dir).expanduser().resolve() / "src" if wm_dir else None
        if foam_src and foam_src.is_dir():
            sys.path.insert(0, str(repo_root / "scripts"))
            import source_recon  # type: ignore
            fp_for_recon = starter_understanding.get("flow_parameters", {}) or {}
            recon_context = source_recon.run_slate_search(
                foam_src=foam_src,
                task={
                    "topic": topic,
                    "mode": "",  # OED is open-ended — no fixed mode
                    "parent_class": "",
                    "formula": (starter_understanding.get("formula_or_model_spec") or "")[:4000],
                    "flow_parameters": fp_for_recon,
                },
                cache_path=recon_cache,
                history_path=disc_dir / "discovered_paths.history.json",
                allow_cache_hit=True,
            )
            if recon_context.get("cache_hit"):
                print(f"[OED][recon] cache HIT: {recon_cache} "
                      f"({len(recon_context.get('selected_files', []))} files, "
                      f"{len(recon_context.get('verified_include_paths', []))} include paths)")
            else:
                print(f"[OED][recon] regenerated: {recon_cache} "
                      f"rounds={recon_context.get('rounds_run')} "
                      f"stopped={recon_context.get('stopped_reason')} "
                      f"({len(recon_context.get('selected_files', []))} files, "
                      f"{len(recon_context.get('verified_include_paths', []))} include paths)")
        else:
            print("[OED][recon] skipped — $WM_PROJECT_DIR/src not found; hypothesis LLM will operate without verified path context")
    except Exception as exc:
        print(f"[OED][recon] FAILED (non-fatal, continuing): {exc}")
        recon_context = None

    # ---------------------------------------------------------------- #
    # OED extensions: lazy post-hoc enrichment before each decide call.  #
    # Computes metric vectors for iterations that produced a case_dir,   #
    # updates family tracker, runs multi-flow validation if requested,   #
    # and renders a prompt fragment to feed to _llm_decide_next_action.   #
    # ---------------------------------------------------------------- #
    def _build_extension_context() -> str:
        parts: List[str] = []
        _run_metric_enrichment = _oedx is not None and (ext_state["multi_metric"] or ext_state["multi_flow"])
        # Phase 1 / 3: enrich any history entry that has a case_dir but no
        # metric_vector yet.
        ref_files = [Path(p) for p in (objective_contract.get("reference_files") or []) if Path(p).is_file()]
        bound = ext_state.get("bound_comparators") or {}
        baseline_vec = ext_state.get("baseline_metric_vector") or {}
        specs = ext_state.get("metric_specs") or []
        for h in (history if _run_metric_enrichment else []):
            if not isinstance(h, dict):
                continue
            cdir = h.get("case_dir") or h.get("compiled_case_dir") or ""
            if not cdir:
                continue
            cp = Path(cdir)
            if not cp.is_dir():
                continue
            if h.get("metric_vector"):
                continue
            if ext_state["multi_metric"] and bound and ref_files:
                try:
                    mv = _oedx.compute_metric_vector(
                        case_dir=cp, bound_comparators=bound, reference_file=_reference_data_file(ref_files),
                        baseline_final_time=baseline_final_time,
                        metric_specs=specs,
                    )
                    # LLM-as-judge replaces fixed aggregator. We pass the LLM
                    # everything needed to make a real call: topic, baseline
                    # vector, candidate metric vector, what modification was
                    # done (formula, category, parameters), why the proposer
                    # chose it, the run-log tail, interpreter physics summary
                    # (if available), recent history, prior PROCEED candidates,
                    # and the diversity policy / families seen.
                    best_so_far = None
                    prior_proceed: List[Dict[str, Any]] = []
                    for hh in history:
                        if (isinstance(hh, dict) and hh.get("status") == "PROCEED"
                                and isinstance(hh.get("metric_vector"), dict)):
                            if best_so_far is None:
                                best_so_far = hh.get("metric_vector")
                            agg_h = hh.get("metric_aggregated") or {}
                            prior_proceed.append({
                                "iteration": hh.get("iteration"),
                                "model_description": hh.get("model_description", ""),
                                "primary_score": agg_h.get("primary"),
                                "metrics": (hh.get("metric_vector") or {}).get("metrics", {}),
                            })
                    history_snip = _compact_history(history)[:3000]
                    # Pull a tail from the case's solver log for run-log context
                    run_log_tail = ""
                    try:
                        for cand_log in cp.glob("log.*"):
                            if cand_log.is_file() and cand_log.stat().st_size > 0:
                                run_log_tail = cand_log.read_text(
                                    encoding="utf-8", errors="ignore"
                                )[-2000:]
                                break
                    except Exception:
                        pass
                    interp = ""
                    try:
                        d = h.get("decision") or h.get("interpreter_decision")
                        if isinstance(d, dict):
                            interp = (d.get("reason") or d.get("rationale") or "")[:1500]
                    except Exception:
                        pass
                    families_seen_list: List[str] = []
                    if ext_state.get("diversity_mode") and ext_state["diversity_mode"] != "off":
                        try:
                            families_seen_list = _oedx.families_seen(disc_dir)
                        except Exception:
                            pass
                    judge = _oedx.llm_judge_iteration(
                        topic=topic,
                        model_description=h.get("model_description", "")
                            or h.get("compiled_model_description", ""),
                        metric_vector=mv,
                        baseline_vector=baseline_vec,
                        metric_specs=specs,
                        history_summary=history_snip,
                        best_so_far=best_so_far,
                        formula_or_modification=str(h.get("formula", "") or h.get("formula_or_modification", "")),
                        modification_category=str(h.get("modification_category", "")),
                        parameters=h.get("parameters") if isinstance(h.get("parameters"), dict) else {},
                        propose_rationale=str(h.get("rationale", ""))[:1500],
                        compiled_model_name=str(h.get("compiled_model_name", "")),
                        flow_parameters=(starter_understanding or {}).get("flow_parameters", {}) or {},
                        reference_description=str(((starter_understanding or {}).get("reference_data") or {}).get("description", "")),
                        interpreter_summary=interp,
                        run_log_excerpt=run_log_tail,
                        iteration=int(h.get("iteration", 0)),
                        budget_total=budget,
                        budget_used=budget_used,
                        prior_proceed_summaries=prior_proceed,
                        diversity_mode=str(ext_state.get("diversity_mode", "off")),
                        families_seen_list=families_seen_list,
                    )
                    # Backwards-compat shape: keep `metric_aggregated` field
                    # but populate it from the judge instead of fixed math.
                    agg = {
                        "primary": judge.get("primary_score"),
                        "aggregator": "llm_judge",
                        "direction": "min",
                        "decision": judge.get("decision"),
                        "is_improvement_over_baseline": judge.get("is_improvement_over_baseline"),
                        "is_best_so_far": judge.get("is_best_so_far"),
                        "rationale": judge.get("rationale"),
                        "strengths": judge.get("strengths", []),
                        "weaknesses": judge.get("weaknesses", []),
                    }
                    h["metric_vector"] = mv
                    h["metric_aggregated"] = agg
                    h["judge_decision"] = judge.get("decision")
                    h["judge_is_improvement"] = judge.get("is_improvement_over_baseline")
                    h["judge_rationale"] = judge.get("rationale", "")[:600]
                    _oedx.write_metric_artifact(
                        disc_dir=disc_dir, iteration=int(h.get("iteration", 0)),
                        case_dir=cp, metric_vector=mv, aggregated=agg,
                    )
                except Exception as exc:
                    h["metric_vector_error"] = f"{exc}"
            if ext_state["multi_flow"]:
                # Validate against other flows by running comparator on each
                # flow's reference (we do not re-run the simulation per flow
                # here — multi-flow simulation requires running the candidate
                # on each flow's base case, which is a future extension. Today
                # we score the same case against multiple reference datasets if
                # they exist.)
                per_flow_metrics = {}
                per_flow_specs = ext_state.get("per_flow_metric_specs") or {}
                per_flow_baseline = ext_state.get("per_flow_baseline_vector") or {}
                for fid, fpath in (ext_state.get("flows") or {}).items():
                    fp = Path(fpath)
                    fref = sorted((fp / "reference_data").glob("*.csv")) if (fp / "reference_data").is_dir() else []
                    if not fref:
                        continue
                    try:
                        mv = _oedx.compute_metric_vector(
                            case_dir=cp, bound_comparators=bound,
                            reference_file=_reference_data_file(fref),
                            baseline_final_time=baseline_final_time,
                            metric_specs=specs,
                        )
                        per_flow_metrics[fid] = mv
                    except Exception:
                        continue
                if per_flow_metrics:
                    flow_agg = _oedx.aggregate_flow_scores(
                        per_flow_metrics=per_flow_metrics,
                        per_flow_baseline=per_flow_baseline,
                        metric_specs_per_flow={fid: specs for fid in per_flow_metrics},
                        aggregator=ext_state.get("metric_aggregator", "weighted_sum"),
                    )
                    h["per_flow_metrics"] = per_flow_metrics
                    h["flow_aggregated"] = flow_agg
        # Persist enriched history
        try:
            _write_json(disc_dir / "history.json", history)
        except Exception:
            pass

        # Build prompt fragment from the most recent enriched iteration
        recent = next((h for h in reversed(history)
                       if isinstance(h, dict) and h.get("metric_vector")), None)
        if recent and ext_state["multi_metric"]:
            parts.append(_oedx.render_metric_vector_for_prompt(
                metric_vector=recent.get("metric_vector", {}),
                baseline_vector=baseline_vec, metric_specs=specs,
            ))
            agg = recent.get("metric_aggregated") or {}
            if agg.get("primary") is not None:
                parts.append(
                    f"Aggregated primary score (lower is better): {agg['primary']:.6g} "
                    f"(aggregator={agg.get('aggregator')})"
                )
        if recent and ext_state["multi_flow"] and recent.get("per_flow_metrics"):
            parts.append(_oedx.render_score_matrix(
                per_flow_metrics=recent["per_flow_metrics"],
                per_flow_baseline=ext_state.get("per_flow_baseline_vector") or {},
                metric_specs_per_flow={fid: specs for fid in recent["per_flow_metrics"]},
            ))

        # Search-policy archive: always included (this is the standing search
        # policy now, not an optional extension — see oed_search_archive.py).
        # Tells the LLM the best known result per model family tried so far,
        # and which niche this iteration's proposal should build on (picked
        # by search_archive.select_niche(...) at the top of this loop
        # iteration, before this function was called).
        parts.append(search_archive.render_summary(
            baseline_score=baseline_score_value, baseline_direction=baseline_direction,
        ))
        if selected_niche.get("is_new"):
            parts.append(
                "NEXT PROPOSAL SHOULD TARGET A NEW MODEL FAMILY not listed in the "
                "archive above — every existing family is either well-explored or "
                "not currently promising enough to justify another attempt right "
                "now. Propose a structurally different mechanism (a different "
                "equation term or physical effect), not a parameter tweak of "
                "something already in the archive."
            )
        else:
            elite = selected_niche.get("elite") or {}
            elite_formula = str(
                elite.get("formula") or elite.get("model_description")
                or elite.get("compiled_model_description") or ""
            )[:800]
            parts.append(
                f"NEXT PROPOSAL SHOULD BUILD ON family '{selected_niche.get('family')}' "
                f"(its current best result, from iteration {elite.get('iteration', '?')}): "
                f"{elite_formula}. Refine or extend this specific direction rather than "
                "starting over from scratch."
            )
        if search_saturated:
            parts.append(
                f"SEARCH ARCHIVE APPEARS SATURATED: the best score across every family "
                f"has not improved over the last {effective_saturation_window} real evaluations. "
                "If a valid PROCEED case already exists, strongly consider stopping now "
                "rather than continuing to spend budget on marginal variation."
            )
        return "\n\n".join(p for p in parts if p)
    # End extensions hook
    # ---------------------------------------------------------------- #

    while budget_used < budget:
        iteration += 1
        budget_remaining = budget - budget_used
        print(f"\n[OED] === Iteration {iteration} | budget used={budget_used}/{budget} ===")

        # Which niche (model family) the next proposal should be conditioned
        # on — computed once per iteration, before the decision call, so
        # _build_extension_context can tell the LLM which elite to mutate
        # from (or that it should try a family never attempted yet).
        selected_niche = search_archive.select_niche(budget_remaining, budget)
        # Saturation-based stop signal: nudges the LLM toward stopping (it
        # still writes the auditable stop_reason itself via allow_stop=True
        # below) rather than force-quitting — see SearchArchive.is_saturated.
        effective_saturation_window = (
            saturation_window if saturation_window is not None else max(3, budget // 4)
        )
        search_saturated = search_archive.is_saturated(effective_saturation_window)

        # Ask LLM what to do next.
        # Retry transient parse/network errors a few times before giving up;
        # a single malformed JSON response should NOT abandon remaining budget.
        force_class_deriv = consecutive_cheap_code_mods >= MAX_CONSECUTIVE_CHEAP_CODE_MODS
        action = None
        decision_attempts = 0
        MAX_DECISION_ATTEMPTS = 3
        while decision_attempts < MAX_DECISION_ATTEMPTS:
            decision_attempts += 1
            try:
                action = _llm_decide_next_action(
                    topic=topic,
                    starter_understanding=starter_understanding,
                    history=history,
                    budget_remaining=budget_remaining,
                    budget_total=budget,
                    compiled_models=compiled_models,
                    repo_root=repo_root,
                    force_real_action=(consecutive_scripts >= MAX_CONSECUTIVE_SCRIPTS),
                    allow_stop=True,
                    blocked_compile_hints=blocked_compile_hints,
                    recon_context=recon_context,
                    force_class_derivation=force_class_deriv,
                    baseline_metrics=baseline_metrics,
                    lit_path=lit_path,
                    extension_context=_build_extension_context(),
                )
                break  # success
            except (json.JSONDecodeError, ValueError) as exc:
                # Transient parse failure (LLM emitted malformed JSON, often
                # from token-truncation on long contexts). Retry — output is
                # non-deterministic so the next call usually parses cleanly.
                print(f"[OED] Decision LLM parse error (attempt {decision_attempts}/{MAX_DECISION_ATTEMPTS}): {exc}.")
                if decision_attempts >= MAX_DECISION_ATTEMPTS:
                    print(f"[OED] Skipping iteration {iteration} after {MAX_DECISION_ATTEMPTS} parse failures (budget preserved).")
                    action = None
            except Exception as exc:
                # Genuine bug / unrecoverable error — stop.
                print(f"[OED] Decision LLM failed (unrecoverable): {exc}. Stopping.")
                action = None
                break

        if action is None:
            # If we exhausted retries on parse errors, skip this iteration and
            # continue the loop — DON'T break out and abandon remaining budget.
            # If it was an unrecoverable error, the inner except already broke.
            if decision_attempts >= MAX_DECISION_ATTEMPTS:
                continue
            break

        atype = action.get("action_type", "stop")
        rationale = action.get("rationale", "")
        print(f"[OED] Decision: {atype} — {rationale[:200]}")

        append_timeline_event(timeline_path, {
            "stage": "open_ended_discovery",
            "event": "decision",
            "iteration": iteration,
            "action_type": atype,
            "rationale": rationale[:300],
            "budget_remaining": budget_remaining,
        })

        if atype == "stop":
            valid_proceed_cases = [
                h for h in history
                if h.get("status") == "PROCEED"
                and h.get("action_type") != "python_script"
                and bool(h.get("valid_case", False))
            ]
            can_afford_real_action = budget_remaining >= 1 if compiled_models else budget_remaining >= 2
            if valid_proceed_cases or not can_afford_real_action:
                print(f"[OED] LLM decided to stop: {action.get('stop_reason', '')}")
                history.append({
                    "iteration": iteration,
                    "action_type": "stop",
                    "rationale": rationale,
                    "stop_reason": str(action.get("stop_reason", ""))[:500],
                    "status": "STOPPED_SUCCESS" if valid_proceed_cases else "STOPPED_BUDGET_LIMIT",
                })
                _write_json(disc_dir / "history.json", history)
                break

            print("[OED] Stop decision blocked: no valid winner yet and budget remains for real actions.")
            try:
                action = _llm_decide_next_action(
                    topic=topic,
                    starter_understanding=starter_understanding,
                    history=history,
                    budget_remaining=budget_remaining,
                    budget_total=budget,
                    compiled_models=compiled_models,
                    repo_root=repo_root,
                    force_real_action=True,
                    allow_stop=False,
                    blocked_compile_hints=blocked_compile_hints,
                    recon_context=recon_context,
                    baseline_metrics=baseline_metrics,
                    lit_path=lit_path,
                    extension_context=_build_extension_context(),
                )
                atype = action.get("action_type", "stop")
                rationale = action.get("rationale", "")
                print(f"[OED] Replacement decision: {atype} — {rationale[:200]}")
            except Exception as exc:
                print(f"[OED] Replacement decision failed: {exc}.")
                if compiled_models and budget_remaining >= 1:
                    action = {
                        "action_type": "experiment",
                        "rationale": "Fallback experiment: budget remains, no valid winner exists, and stop is not allowed.",
                        "model_description": "fallback_parameter_sweep",
                        "model_name_to_reuse": compiled_models[-1]["name"],
                        "parameters": {},
                    }
                elif budget_remaining >= 2:
                    history.append({
                        "iteration": iteration,
                        "action_type": "stop",
                        "rationale": "Forced abort due to repeated invalid stop/replan behavior.",
                        "stop_reason": "Unable to obtain non-stop action from decision LLM while budget remained.",
                        "status": "FAILED_DECISION_POLICY",
                    })
                    _write_json(disc_dir / "history.json", history)
                    break
                atype = action.get("action_type", "stop")
                rationale = action.get("rationale", "")

        # Execute action
        history_entry: Dict[str, Any] = {
            "iteration": iteration,
            "action_type": atype,
            "rationale": rationale,
        }

        # Budget-extension request (generic, optional). If the LLM asks for
        # more units and the run has already produced at least one PROCEED
        # case, honor it up to a 50% cap of the original budget. Prevents
        # runaway resource use while letting a converging study finish.
        bext = action.get("budget_extension_request")
        if isinstance(bext, dict) and isinstance(bext.get("units"), int) and bext["units"] > 0:
            already_proceed = any(
                h.get("status") == "PROCEED" and h.get("action_type") != "python_script"
                for h in history
            )
            # Per-request cap: ≤50% of current budget.
            # Hard ceiling: total budget cannot exceed 1.5× original (no matter how many requests).
            per_req_cap = budget // 2
            ceiling_room = max(0, _budget_hard_ceiling - budget)
            cap = min(per_req_cap, ceiling_room)
            req = int(bext.get("units", 0))
            if already_proceed and req > 0 and req <= cap:
                budget += req
                print(f"[OED] Budget extension GRANTED: +{req} units "
                      f"(justification: {str(bext.get('justification',''))[:120]}). "
                      f"New total budget: {budget} (hard ceiling: {_budget_hard_ceiling}).")
                history_entry["budget_extension_granted"] = req
            else:
                if not already_proceed:
                    reason = "no PROCEED case yet"
                elif ceiling_room == 0:
                    reason = f"hard ceiling reached ({_budget_hard_ceiling}, original {_budget_original}× 1.5)"
                else:
                    reason = f"exceeds per-request cap (req={req}, cap={cap}; per-req={per_req_cap}, ceiling-room={ceiling_room})"
                print(f"[OED] Budget extension DENIED: {reason}")
                history_entry["budget_extension_denied"] = True
                history_entry["budget_extension_denial_reason"] = reason

        if atype == "python_script":
            # Hard cap: if the LLM ignored the CRITICAL OVERRIDE and chose python_script
            # anyway, refuse to execute and inject a synthetic "BLOCKED" history entry
            # so the next call sees it and is forced to pick a real action.
            if consecutive_scripts >= MAX_CONSECUTIVE_SCRIPTS:
                print(f"[OED] python_script BLOCKED (cap {MAX_CONSECUTIVE_SCRIPTS} reached). "
                      "Forcing next iteration to be code_mod/experiment/stop.")
                history.append({
                    "iteration": iteration,
                    "action_type": "python_script",
                    "script_description": "[BLOCKED by consecutive-script cap]",
                    "script_output": (
                        "This python_script action was blocked because the maximum number of "
                        f"consecutive scripts ({MAX_CONSECUTIVE_SCRIPTS}) was reached. "
                        "You MUST choose code_mod, experiment, or stop next."
                    ),
                    "status": "BLOCKED",
                })
                _write_json(disc_dir / "history.json", history)
                continue

            result = _run_python_script_iteration(
                iteration=iteration,
                action=action,
                run_dir=disc_dir,
                repo_root=repo_root,
                starter_understanding=starter_understanding,
                starter_dir=starter_dir,
                timeline_path=timeline_path,
            )
            # python_script costs 0 budget units — free scratchpad
            consecutive_scripts += 1
            history_entry.update({
                "script_description": result.get("script_description", ""),
                "script_output": result.get("output", ""),
                "status": result.get("status", "UNKNOWN"),
            })
            history.append(history_entry)

            append_timeline_event(timeline_path, {
                "stage": "open_ended_discovery",
                "event": "iteration_done",
                "iteration": iteration,
                "action_type": atype,
                "status": result.get("status", "UNKNOWN"),
                "budget_used": budget_used,
            })

            _write_json(disc_dir / "history.json", history)
            print(f"[OED] Iteration {iteration} done: python_script status={result.get('status')} "
                  f"budget_used={budget_used}/{budget} (script costs 0 units)"
                  + (f" [consecutive={consecutive_scripts}/{MAX_CONSECUTIVE_SCRIPTS}]"
                     if consecutive_scripts < MAX_CONSECUTIVE_SCRIPTS
                     else " [cap reached — next must be code_mod/experiment/stop]"))
            # NOTE: no extra iteration += 1 here; the outer while-loop's top already does it
            continue  # do not deduct budget; go to next iteration

        # Real action taken — reset consecutive script counter
        consecutive_scripts = 0

        history_entry["model_description"] = action.get("model_description", "")
        formula_text = action.get("formula_or_modification")
        if formula_text is None:
            formula_text = ""
        history_entry["formula"] = str(formula_text)[:400]
        history_entry["parameters"] = action.get("parameters", {})

        if atype == "code_mod_batch":
            # Run up to 5 variants in a single "iteration". Each variant costs
            # 2 budget units. Variants are compiled + tested sequentially; sub-
            # iteration dirs are iter_<N>_code_mod_batch_<slug>/. A ranked
            # summary is recorded in history_entry.
            raw_variants = action.get("variants") or []
            if not isinstance(raw_variants, list) or not raw_variants:
                print("[OED] code_mod_batch missing variants array; downgrading to single code_mod.")
                atype = "code_mod"
            else:
                variants = [v for v in raw_variants if isinstance(v, dict)][:5]
                cost = 2 * len(variants)
                if cost > budget_remaining:
                    # Trim to fit remaining budget
                    variants = variants[: max(0, budget_remaining // 2)]
                    cost = 2 * len(variants)
                    if not variants:
                        print("[OED] Not enough budget for any variant in batch. Stopping.")
                        break
                    print(f"[OED] Trimmed batch to {len(variants)} variants (budget limit).")
                print(f"[OED] code_mod_batch: running {len(variants)} variants (cost {cost} units)")
                batch_results: List[Dict[str, Any]] = []
                for v_idx, v in enumerate(variants, 1):
                    v_action = dict(action)  # carry base fields
                    v_action["action_type"] = "code_mod"
                    v_action["variant_name"] = v.get("variant_name") or f"v{v_idx}"
                    v_action["model_description"] = v.get("model_description", "")
                    v_action["formula_or_modification"] = v.get("formula_or_modification", "")
                    v_action["parameters"] = v.get("parameters", action.get("parameters", {}))
                    sub_tag = _slugify_variant(v_action["variant_name"]) or f"v{v_idx}"
                    print(f"[OED]   batch variant {v_idx}/{len(variants)}: {v_action['variant_name']}")
                    try:
                        sub_result = _run_code_mod_iteration(
                            iteration=iteration,
                            action=v_action,
                            run_dir=disc_dir,
                            repo_root=repo_root,
                            base_case_dir=base_case_dir,
                            topic=topic,
                            lit_path=lit_path,
                            timeline_path=timeline_path,
                            starter_understanding=starter_understanding,
                            starter_dir=starter_dir,
                            objective_contract=objective_contract,
                            env=env,
                            attempt_tag=f"batch_{sub_tag}",
                        )
                    except Exception as exc:
                        sub_result = {"status": "FAILED", "error": str(exc)[:300]}
                    batch_results.append({
                        "variant_name": v_action["variant_name"],
                        "status": sub_result.get("status"),
                        "case_dir": sub_result.get("case_dir"),
                        "metrics_summary": sub_result.get("metrics_summary", ""),
                        "compiled_model_name": sub_result.get("compiled_model_name"),
                    })
                # Charge budget once for the whole batch
                budget_used += cost
                history_entry["cost"] = cost
                # Build synthetic result for this iteration
                proceed_count = sum(1 for r in batch_results if r.get("status") == "PROCEED")
                result = {
                    "status": "PROCEED" if proceed_count > 0 else "FAILED",
                    "metrics_summary": (
                        f"batch: {proceed_count}/{len(batch_results)} PROCEED. "
                        + "; ".join(f"{r['variant_name']}={r['status']}" for r in batch_results)[:400]
                    ),
                    "batch_results": batch_results,
                }
                # Each compiled variant also added to compiled_models list
                for r in batch_results:
                    if r.get("compiled_model_name"):
                        compiled_models.append({
                            "name": r["compiled_model_name"],
                            "description": f"(batch iter {iteration}) {r['variant_name']}",
                            "case_dir": r.get("case_dir", ""),
                            "family": SearchArchive.classify(
                                r.get("variant_name", ""), r["compiled_model_name"]),
                        })
                history_entry["batch_results"] = batch_results
                consecutive_scripts = 0
                # Skip the normal single-variant path below by falling through
                # to history-append. Set a flag so the normal code_mod/experiment
                # branches don't also run.
                atype = "code_mod_batch_done"

        if atype == "code_mod":
            # Category-aware dispatch. Runtime / dict_only paths are O(seconds)
            # and never invoke wmake — they cost 1 unit. The class-derivation
            # path keeps its 2-unit cost when it succeeds (or is exhausted).
            mcat = str(action.get("modification_category") or "").strip().lower()
            is_runtime = mcat in ("runtime_source", "runtime_bc", "runtime_field", "dict_only")
            # Default any unset/unrecognized modification_category to
            # class_derivation. The legacy foam_code_builder + apply_compile
            # pipeline (the `else` branch below) is being phased out — the
            # agentic class_derivation path is the right fallback when the
            # planner LLM forgot to set the field.
            if not is_runtime and mcat != "class_derivation":
                if mcat:
                    print(f"[OED] unrecognized modification_category={mcat!r}; "
                          "defaulting to class_derivation (agentic).")
                mcat = "class_derivation"
            min_cost = 1 if is_runtime else 2
            if budget_remaining < min_cost:
                print(f"[OED] Not enough budget for code_mod ({mcat} "
                      f"needs {min_cost}). Stopping.")
                break

            history_entry["modification_category"] = mcat
            history_entry["is_runtime"] = is_runtime

            if is_runtime:
                # Fast path — dictionary-only modification, no compile.
                history_entry["model_description"] = action.get("model_description", "")
                formula_text = action.get("formula_or_modification") or json.dumps(
                    {k: action.get(k) for k in ("modification_category", "runtime_source",
                                                "runtime_bc", "runtime_field", "dict_only")
                     if action.get(k) is not None},
                    default=str,
                )[:2000]
                history_entry["formula"] = str(formula_text)[:400]
                result = _run_runtime_code_mod_iteration(
                    iteration=iteration,
                    action=action,
                    run_dir=disc_dir,
                    repo_root=repo_root,
                    base_case_dir=base_case_dir,
                    topic=topic,
                    timeline_path=timeline_path,
                    starter_understanding=starter_understanding,
                    starter_dir=starter_dir,
                    objective_contract=objective_contract,
                    env=env,
                    baseline_metrics=baseline_metrics,
                )
                # Runtime applies cost 1 unit regardless of success — they're
                # already cheap, so we don't penalize failure further (a
                # malformed coded* spec is an LLM-output bug, not a search
                # signal).
                cost_charged = 1
                # Surface the runtime apply error so the planner LLM sees WHY
                # the runtime path failed and can adapt (or pick a different
                # category). Without this the LLM only sees status=FAILED and
                # blindly retries the same broken category.
                rae = result.get("runtime_apply_error", "")
                if rae:
                    history_entry["runtime_apply_error"] = str(rae)[:400]
                    rae_fp = str(rae).lower()[:220]
                    compile_fail_counts[rae_fp] = compile_fail_counts.get(rae_fp, 0) + 1
                    if compile_fail_counts[rae_fp] >= 2 and rae_fp not in blocked_compile_hints:
                        blocked_compile_hints.append(rae_fp)
                if result.get("compiled_model_name"):
                    compiled_models.append({
                        "name": result["compiled_model_name"],
                        "description": result.get("compiled_model_description", ""),
                        "case_dir": result.get("compiled_case_dir", ""),
                        "is_runtime": True,
                        "family": SearchArchive.classify(
                            result.get("compiled_model_description", ""), result["compiled_model_name"]),
                    })
            elif mcat == "class_derivation":
                # Class-derivation path — agentic runner via codex exec, with
                # full shell access. Mirrors how ARIS runs OpenFOAM model
                # development: read OpenFOAM source, write the library, run
                # wmake, fix errors, retry — all in one agent session.
                history_entry["model_description"] = action.get("model_description", "")
                formula_text = action.get("formula_or_modification") or ""
                history_entry["formula"] = str(formula_text)[:400]
                result = _run_agentic_code_mod_iteration(
                    iteration=iteration,
                    action=action,
                    run_dir=disc_dir,
                    repo_root=repo_root,
                    base_case_dir=base_case_dir,
                    topic=topic,
                    timeline_path=timeline_path,
                    starter_understanding=starter_understanding,
                    starter_dir=starter_dir,
                    objective_contract=objective_contract,
                    env=env,
                    baseline_metrics=baseline_metrics,
                )
                cost_charged = 2 if result.get("status") != "FAILED" else 1
                hint = str(result.get("compile_error_hint", "") or "").strip()
                if hint:
                    fp = hint.lower()[:220]
                    compile_fail_counts[fp] = compile_fail_counts.get(fp, 0) + 1
                    if compile_fail_counts[fp] >= 2 and fp not in blocked_compile_hints:
                        blocked_compile_hints.append(fp)
                if result.get("compiled_model_name"):
                    compiled_models.append({
                        "name": result["compiled_model_name"],
                        "description": result.get("compiled_model_description", ""),
                        "case_dir": result.get("compiled_case_dir", ""),
                        "is_runtime": False,
                        "family": SearchArchive.classify(
                            result.get("compiled_model_description", ""), result["compiled_model_name"]),
                    })
            else:
                # Legacy class-derivation path — fall-back when modification_category
                # is unset / unrecognized. Goes through the old foam_code_builder
                # + apply_compile pipeline. Kept for backward compatibility.
                action = _llm_refine_code_mod_spec(
                    action=action,
                    topic=topic,
                    starter_understanding=starter_understanding,
                    history=history,
                    repo_root=repo_root,
                )
                history_entry["model_description"] = action.get("model_description", "")
                formula_text = action.get("formula_or_modification") or ""
                history_entry["formula"] = str(formula_text)[:400]

                # Multi-attempt build-engineer retry loop. Up to MAX_BUILD_ATTEMPTS
                # tries on the SAME hypothesis. Each retry re-prompts with the
                # build error so the LLM fixes the implementation, NOT the physics.
                MAX_BUILD_ATTEMPTS = 4
                cur_action = action
                result = _run_code_mod_iteration(
                    iteration=iteration,
                    action=cur_action,
                    run_dir=disc_dir,
                    repo_root=repo_root,
                    base_case_dir=base_case_dir,
                    topic=topic,
                    lit_path=lit_path,
                    timeline_path=timeline_path,
                    starter_understanding=starter_understanding,
                    starter_dir=starter_dir,
                    objective_contract=objective_contract,
                    env=env,
                )
                attempts_done = 1
                while (
                    result.get("status") == "FAILED"
                    and result.get("compile_error_hint")
                    and attempts_done < MAX_BUILD_ATTEMPTS
                ):
                    print(f"[OED] code_mod compile failed (attempt {attempts_done}/"
                          f"{MAX_BUILD_ATTEMPTS}); invoking build-engineer retry.")
                    repaired_action = _llm_refine_code_mod_spec(
                        action=cur_action,
                        topic=topic,
                        starter_understanding=starter_understanding,
                        history=history,
                        repo_root=repo_root,
                        compile_error_hint=str(result.get("compile_error_hint", "")),
                    )
                    new_spec = str(repaired_action.get("formula_or_modification", "") or "").strip()
                    cur_spec = str(cur_action.get("formula_or_modification", "") or "").strip()
                    if not new_spec or new_spec == cur_spec:
                        # LLM produced no refinement — break out, no point retrying.
                        print("[OED] build-engineer retry produced identical spec — aborting retry loop.")
                        break
                    attempts_done += 1
                    retry_result = _run_code_mod_iteration(
                        iteration=iteration,
                        action=repaired_action,
                        run_dir=disc_dir,
                        repo_root=repo_root,
                        base_case_dir=base_case_dir,
                        topic=topic,
                        lit_path=lit_path,
                        timeline_path=timeline_path,
                        starter_understanding=starter_understanding,
                        starter_dir=starter_dir,
                        objective_contract=objective_contract,
                        env=env,
                        attempt_tag=f"retry{attempts_done - 1}",
                    )
                    cur_action = repaired_action
                    if retry_result.get("status") != "FAILED":
                        result = retry_result
                        history_entry["model_description"] = repaired_action.get("model_description", "")
                        history_entry["formula"] = str(
                            repaired_action.get("formula_or_modification", "")
                        )[:400]
                        break
                    # carry forward the latest compile_error_hint
                    result = retry_result

                history_entry["build_attempts"] = attempts_done
                # Cost: successful run charges 2; all-retries-failed charges 1
                # (so the search loop can still try a different hypothesis
                # without burning the full slot).
                cost_charged = 2 if result.get("status") != "FAILED" else 1
                # Block a recurring compile signature so the decision LLM
                # avoids the same trap on the NEXT idea.
                hint = str(result.get("compile_error_hint", "") or "").strip()
                if hint:
                    fp = hint.lower()[:220]
                    compile_fail_counts[fp] = compile_fail_counts.get(fp, 0) + 1
                    if compile_fail_counts[fp] >= 2 and fp not in blocked_compile_hints:
                        blocked_compile_hints.append(fp)
                if result.get("compiled_model_name"):
                    compiled_models.append({
                        "name": result["compiled_model_name"],
                        "description": result.get("compiled_model_description", ""),
                        "case_dir": result.get("compiled_case_dir", ""),
                        "is_runtime": False,
                        "family": SearchArchive.classify(
                            result.get("compiled_model_description", ""), result["compiled_model_name"]),
                    })

            budget_used += cost_charged
            history_entry["cost"] = cost_charged

        elif atype == "experiment":
            result = _run_experiment_iteration(
                iteration=iteration,
                action=action,
                run_dir=disc_dir,
                repo_root=repo_root,
                topic=topic,
                timeline_path=timeline_path,
                starter_understanding=starter_understanding,
                starter_dir=starter_dir,
                objective_contract=objective_contract,
                compiled_models=compiled_models,
                env=env,
                baseline_metrics=baseline_metrics,
            )
            budget_used += 1
            history_entry["cost"] = 1
        elif atype == "investigate_runtime":
            result = _run_investigate_runtime_iteration(
                iteration=iteration, action=action, history=history,
                run_dir=disc_dir, repo_root=repo_root, base_case_dir=base_case_dir,
                topic=topic, timeline_path=timeline_path,
                starter_understanding=starter_understanding, starter_dir=starter_dir,
                objective_contract=objective_contract, env=env,
                compiled_models=compiled_models,
                baseline_metrics=baseline_metrics,
            )
            budget_used += 1
            history_entry["cost"] = 1
        elif atype == "code_mod_batch_done":
            # Batch path already ran and set `result`, `budget_used` updated above.
            history_entry["cost"] = history_entry.get("cost") or 0
        else:
            print(f"[OED] Unknown action type {atype!r}. Stopping.")
            break

        history_entry.update({
            "status": result.get("status", "UNKNOWN"),
            "interpreter_reason": result.get("interpreter_reason", ""),
            "metrics_summary": result.get("metrics_summary", ""),
            "case_dir": result.get("case_dir", ""),
            "compile_error_hint": result.get("compile_error_hint", ""),
            "run_validity": result.get("run_validity") if isinstance(
                result.get("run_validity"), dict) else None,
            # Persist compiled-model fields so the resume-from-history path can
            # rebuild compiled_models on relaunch. Without these the experiment
            # iteration helper (which references compiled_models[-1]) would see
            # an empty registry on resume and the LLM's parameter-sweep actions
            # would have no model to reuse.
            "compiled_model_name": result.get("compiled_model_name", ""),
            "compiled_model_description": result.get("compiled_model_description", ""),
            "compiled_case_dir": result.get("compiled_case_dir", ""),
            "compiled_so": result.get("compiled_so", ""),
            "is_runtime": bool(result.get("is_runtime", False)),
            "is_agentic": bool(result.get("is_agentic", False)),
        })
        # Open-ended-only integrity/score contract: invalid cases are never "best".
        case_dir_str = str(result.get("case_dir", "") or "")
        if case_dir_str:
            assessment = _evaluate_open_ended_case_contract(
                case_dir=Path(case_dir_str),
                status=str(history_entry.get("status", "UNKNOWN")),
                interpreter_reason=str(history_entry.get("interpreter_reason", "")),
                metrics_summary=str(history_entry.get("metrics_summary", "")),
                run_validity=history_entry.get("run_validity") if isinstance(
                    history_entry.get("run_validity"), dict) else None,
            )
            assess_path = Path(case_dir_str) / "open_ended_case_assessment.json"
            _write_json(assess_path, assessment)
            history_entry["case_assessment_path"] = str(assess_path)
            history_entry["valid_case"] = bool(assessment.get("valid", False))
            history_entry["score"] = assessment.get("score")
            if not history_entry["valid_case"] and str(history_entry.get("status", "")).upper() == "PROCEED":
                history_entry["status"] = "INVALID"
                history_entry["interpreter_reason"] = (
                    (history_entry.get("interpreter_reason", "") + " | ")
                    + "Open-ended contract failed integrity checks; marked INVALID."
                )[:500]
            # Baseline gate: when a baseline score is available and the
            # variant produced a comparable primary_score, decide PROCEED /
            # REVISE from the score directly. Generic across QoIs and metric
            # directions:
            #   direction=min (default): variant < baseline is better
            #   direction=max:           variant > baseline is better
            #
            # This block does TWO things, both score-driven:
            #   (a) DEMOTE: if interpret said PROCEED but the score does not
            #       beat baseline, demote to REVISE (planner keeps searching).
            #   (b) UPGRADE: if interpret was unavailable (UNKNOWN / FAILED
            #       due to timeout / stall / I/O error) but the comparator
            #       still produced a valid primary_score and the case is
            #       otherwise valid, derive PROCEED (beats baseline) or
            #       REVISE (does not). Without this, a working experiment
            #       with a real score would be wasted just because the
            #       vision-LLM interpret stalled.
            if baseline_score_value is not None:
                v_score = history_entry.get("score")
                v_val: Optional[float] = None
                v_direction = baseline_direction
                if isinstance(v_score, dict):
                    try:
                        v_val = float(v_score.get("value"))
                    except Exception:
                        v_val = None
                    vd = str(v_score.get("direction", "")).strip().lower()
                    if vd in ("min", "max"):
                        v_direction = vd
                if v_val is not None:
                    history_entry["baseline_score"] = baseline_score_value
                    history_entry["baseline_direction"] = v_direction
                    history_entry["score_delta_vs_baseline"] = v_val - baseline_score_value
                    if v_direction == "max":
                        beats = v_val > baseline_score_value
                    else:  # default min
                        beats = v_val < baseline_score_value

                    cur_status = str(history_entry.get("status", "")).upper()

                    # (a) Demote spurious PROCEED.
                    if cur_status == "PROCEED" and not beats:
                        history_entry["status"] = "REVISE"
                        cmp_word = "≤" if v_direction == "max" else "≥"
                        history_entry["interpreter_reason"] = (
                            (history_entry.get("interpreter_reason", "") + " | ")
                            + f"Variant score {v_val:.6g} {cmp_word} baseline "
                              f"{baseline_score_value:.6g} (direction={v_direction}); "
                              f"demoted PROCEED→REVISE."
                        )[:500]

                    # (b) Upgrade UNKNOWN/FAILED when we have a real score
                    # AND the case passed the integrity check (valid_case).
                    # We rescue the iteration: a valid simulation with a
                    # measurable primary_score should not be wasted because
                    # the interpret LLM call timed out.
                    elif cur_status in ("UNKNOWN", "FAILED") and bool(history_entry.get("valid_case", False)) and cur_status != "RUN_INVALID":
                        new_status = "PROCEED" if beats else "REVISE"
                        cmp_word = ("<" if v_direction == "min" else ">") if beats else \
                                   ("≥" if v_direction == "min" else "≤")
                        history_entry["status"] = new_status
                        history_entry["status_derived_from_score"] = True
                        history_entry["interpreter_reason"] = (
                            (history_entry.get("interpreter_reason", "") + " | ")
                            + f"Interpreter unavailable; status derived from "
                              f"primary_score {v_val:.6g} {cmp_word} baseline "
                              f"{baseline_score_value:.6g} (direction={v_direction}) → "
                              f"{new_status}."
                        )[:500]
        # Track consecutive cheap-category code_mods for the rotation gate.
        # Cheap = anything that doesn't compile a class. Reset on class_derivation
        # or on a successful PROCEED (we don't want to force-rotate after a win).
        if atype == "code_mod":
            cat = (history_entry.get("modification_category") or "").lower()
            status_now = str(history_entry.get("status", "")).upper()
            if cat == "class_derivation" or status_now == "PROCEED":
                consecutive_cheap_code_mods = 0
            else:
                consecutive_cheap_code_mods += 1

        # Record this iteration's lineage and update the search archive. The
        # niche this iteration was actually conditioned on (`selected_niche`,
        # computed at the top of this loop iteration) becomes its
        # parent_iteration — the field that did not exist anywhere in this
        # file before this change (see oed_search_archive.py).
        if atype in ("code_mod", "experiment"):
            family = None
            if atype == "experiment":
                reused_name = action.get("model_name_to_reuse", "")
                reused = next((m for m in compiled_models if m.get("name") == reused_name), None)
                if reused is None and compiled_models:
                    reused = compiled_models[-1]
                family = (reused or {}).get("family")
            if not family:
                family = SearchArchive.classify(
                    history_entry.get("model_description", "")
                        or history_entry.get("compiled_model_description", ""),
                    history_entry.get("compiled_model_name", ""),
                )
            history_entry["family"] = family
            history_entry["parent_iteration"] = (
                (selected_niche.get("elite") or {}).get("iteration")
                if not selected_niche.get("is_new") else None
            )
            _score_for_archive = history_entry.get("score")
            _direction_for_archive = baseline_direction
            if isinstance(_score_for_archive, dict):
                _vd = str(_score_for_archive.get("direction", "")).strip().lower()
                if _vd in ("min", "max"):
                    _direction_for_archive = _vd
            search_archive.update(family, iteration, _score_for_archive, _direction_for_archive, history_entry)

        history.append(history_entry)

        append_timeline_event(timeline_path, {
            "stage": "open_ended_discovery",
            "event": "iteration_done",
            "iteration": iteration,
            "action_type": atype,
            "status": result.get("status", "UNKNOWN"),
            "budget_used": budget_used,
        })

        # Save incremental history after every iteration
        _write_json(disc_dir / "history.json", history)
        print(f"[OED] Iteration {iteration} done: status={result.get('status')} "
              f"budget_used={budget_used}/{budget}")

        # Interim milestone snapshot after each code_mod or experiment iteration
        # (not every python_script — too chatty). Pure python, no LLM cost.
        if atype in ("code_mod", "experiment"):
            try:
                sys.path.insert(0, str(repo_root / "scripts"))
                import milestone_summary  # type: ignore
                ms_path = milestone_summary.write_milestone(
                    run_dir,
                    tag=f"oed_iter_{iteration:03d}_{atype}",
                    note=(f"Iteration {iteration} ({atype}) status={result.get('status')}. "
                          f"Budget used={budget_used}/{budget}."),
                )
                print(f"[OED] milestone snapshot: {ms_path}")
            except Exception as exc:
                print(f"[OED] milestone snapshot failed (non-fatal): {exc}")

    # Build summary
    proceed_cases = [
        h for h in history
        if h.get("status") == "PROCEED"
        and h.get("action_type") != "python_script"
        and bool(h.get("valid_case", False))
    ]
    script_runs = [h for h in history if h.get("action_type") == "python_script"]
    scored_valid_cases = [
        h for h in proceed_cases
        if isinstance(h.get("score"), dict) and _try_float(h.get("score", {}).get("value")) is not None
    ]
    if scored_valid_cases:
        best = min(scored_valid_cases, key=lambda h: float(h["score"]["value"]))
    else:
        best = max(proceed_cases, key=lambda h: h["iteration"]) if proceed_cases else (history[-1] if history else {})
    summary = {
        "iterations_run": iteration,
        "budget_used": budget_used,
        "budget_total": budget,
        "total_attempts": len(history),
        "python_script_runs": len(script_runs),
        "proceed_count": len(proceed_cases),
        "best_case_dir": best.get("case_dir", ""),
        "best_model_description": best.get("model_description", ""),
        "best_score": best.get("score"),
        "history": history,
        "compiled_models": compiled_models,
    }
    _write_json(disc_dir / "summary.json", summary)

    append_timeline_event(timeline_path, {
        "stage": "open_ended_discovery",
        "event": "complete",
        "iterations_run": iteration,
        "budget_used": budget_used,
        "proceed_count": len(proceed_cases),
    })

    print(f"\n[OED] Discovery complete: {iteration} iterations, "
          f"{budget_used}/{budget} budget used, {len(proceed_cases)} PROCEED cases.")
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Closed-loop open-ended CFD discovery.")
    parser.add_argument("--run-dir", required=True, type=str)
    parser.add_argument("--topic", required=True, type=str)
    parser.add_argument("--budget", required=True, type=int,
                        help="Total budget units (code_mod=2, experiment=1)")
    parser.add_argument("--starter-understanding", default="", type=str)
    parser.add_argument("--starter-dir", default="", type=str,
                        help="Path to starter folder (reference data, equations, base case). "
                             "Overrides starter_understanding['starter_dir'] if provided.")
    parser.add_argument("--literature", default="", type=str)
    parser.add_argument("--base-case-dir", default="", type=str)
    parser.add_argument("--baseline-metrics", default="", type=str,
                        help="Path to baseline_metrics.json produced by baseline_setup. "
                             "Used to gate PROCEED on variant_score < baseline_score and "
                             "to surface the target to beat in the planner prompt.")
    parser.add_argument("--timeline", default="", type=str)
    # OED extensions — multi-metric is always-on (LLM auto-decides what to track
    # from topic+ref+baseline; degrades to single-metric if only one relevant).
    # `--single-metric` exists only as an emergency override to force the legacy
    # single-comparator path, e.g. for cost-constrained reruns.
    parser.add_argument("--no-allrun-preflight", action="store_true",
                        help="Disable LLM-driven Allrun pre-flight before each "
                             "runtime/experiment iteration. Default: enabled.")
    parser.add_argument("--single-metric", action="store_true",
                        help="Override: disable multi-metric vector tracking and use "
                             "the legacy single-comparator path. Default behaviour "
                             "(without this flag) is multi-metric + LLM-as-judge.")
    parser.add_argument("--diversity-mode", default="off",
                        choices=["off", "hybrid", "aggressive"],
                        help="DEPRECATED, no-op: the fixed-period close/far nudge this used "
                             "to control has been superseded by the family-niched search "
                             "archive's visit/quality-based selection (always on — see "
                             "scripts/oed_search_archive.py). Kept parseable only so existing "
                             "callers (e.g. orchestrator_run.py) don't break.")
    parser.add_argument("--diversity-far-ratio", default=0.3, type=float,
                        help="DEPRECATED, no-op — see --diversity-mode.")
    parser.add_argument("--saturation-window", default=None, type=int,
                        help="Stop-nudge window: recommend stopping once the search "
                             "archive's best score hasn't improved over this many real "
                             "evaluations. Default: max(3, budget // 4).")
    parser.add_argument("--setup-only", action="store_true",
                        help="Run only the one-time per-study setup (objective contract, "
                             "bound_comparators.json, baseline metric vector) and exit "
                             "before the iteration loop. For callers (e.g. the deepagents "
                             "manager) that drive iteration themselves and just need the "
                             "setup artifacts on disk.")
    parser.add_argument("--multi-flow", action="store_true",
                        help="Phase 3: validate candidates against multiple reference flows. "
                             "Use --starter-dirs to provide flow folders or place multiple "
                             "subdirs (each with reference_data/) under --starter-dir.")
    parser.add_argument("--starter-dirs", nargs="+", default=[],
                        help="Phase 3: list of starter directories, one per flow. "
                             "Each must contain reference_data/. Implies --multi-flow.")
    parser.add_argument("--metric-aggregator", default="llm_judge",
                        choices=["llm_judge", "weighted_sum", "min_improvement", "pareto_rank"],
                        help="How to reduce a metric vector to a primary score. "
                             "Default 'llm_judge' = LLM looks at the full vector + "
                             "baseline + history and decides PROCEED/REVISE/RERUN with "
                             "rationale. Fixed aggregators ('weighted_sum', "
                             "'min_improvement', 'pareto_rank') are legacy fallbacks.")
    args = parser.parse_args()

    # Allrun pre-flight toggle (module-level so the helpers can read it).
    global _ALLRUN_PREFLIGHT_ENABLED
    _ALLRUN_PREFLIGHT_ENABLED = not bool(getattr(args, "no_allrun_preflight", False))

    repo_root = Path(__file__).resolve().parent.parent
    run_dir = Path(args.run_dir).expanduser().resolve()
    timeline_path = resolve_timeline_path(args.timeline)

    # Bootstrap paths
    for p in (str(repo_root / "Foam-Agent" / "src"), str(repo_root / "src"), str(Path(__file__).parent)):
        if p not in sys.path:
            sys.path.insert(0, p)

    starter_understanding: Dict[str, Any] = {}
    if args.starter_understanding:
        su_p = Path(args.starter_understanding)
        if su_p.is_file():
            starter_understanding = _read_json(su_p, {})

    # Resolve starter_dir: CLI arg wins over starter_understanding, then default
    if args.starter_dir:
        starter_dir: Optional[Path] = Path(args.starter_dir).expanduser().resolve()
    elif starter_understanding.get("starter_dir"):
        starter_dir = Path(starter_understanding["starter_dir"]).expanduser().resolve()
    else:
        _default = repo_root / "starter"
        starter_dir = _default if _default.is_dir() else None

    lit_path = Path(args.literature) if args.literature else run_dir / "lit.json"

    import os
    env = {**os.environ}

    baseline_metrics: Dict[str, Any] = {}
    if args.baseline_metrics:
        bp = Path(args.baseline_metrics).expanduser().resolve()
        if bp.is_file():
            loaded = _read_json(bp, {})
            if isinstance(loaded, dict):
                baseline_metrics = loaded

    # Resolve --starter-dirs (implies --multi-flow)
    starter_dirs_paths: List[Path] = []
    for sd in (args.starter_dirs or []):
        sp = Path(sd).expanduser().resolve()
        if sp.is_dir():
            starter_dirs_paths.append(sp)
    multi_flow_flag = bool(args.multi_flow or starter_dirs_paths)

    summary = run_open_ended_discovery(
        run_dir=run_dir,
        repo_root=repo_root,
        topic=args.topic,
        budget=args.budget,
        starter_understanding=starter_understanding,
        starter_dir=starter_dir,
        lit_path=lit_path,
        base_case_dir=args.base_case_dir,
        timeline_path=timeline_path,
        env=env,
        baseline_metrics=baseline_metrics,
        saturation_window=args.saturation_window,
        setup_only=bool(args.setup_only),
        multi_metric=(not bool(args.single_metric)),
        diversity_mode=str(args.diversity_mode),
        diversity_far_ratio=float(args.diversity_far_ratio),
        multi_flow=multi_flow_flag,
        starter_dirs=starter_dirs_paths or None,
        metric_aggregator=str(args.metric_aggregator),
    )
    print(json.dumps({"status": "ok", "summary": summary.get("best_model_description", "")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
