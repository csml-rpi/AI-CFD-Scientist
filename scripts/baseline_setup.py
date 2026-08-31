#!/usr/bin/env python3
"""
Baseline setup — produce a baseline numeric reference the OED loop can compare
each new variant against.

Generic across CFD studies:
  - "novel SA modification beats baseline SA on Cf vs DNS" → baseline = vanilla SA
  - "non-Newtonian viscosity beats Newtonian on pressure drop"  → baseline = Newtonian
  - "novel BC beats default zeroGradient on heat flux"          → baseline = default BC
  - "new k-omega variant beats k-omega SST on reattachment"     → baseline = k-omega SST

Logic:
  1. LLM parses the topic to extract:
       - whether a baseline comparison is requested
       - the baseline name (built-in OpenFOAM model / scheme / BC)
       - the QoI being compared (Cf, pressure drop, heat flux, ...)
       - the reference (DNS / experiment / literature)
  2. If baseline data already exists somewhere (starter_dir/baseline_run/,
     a CSV named like "*baseline*.csv", a metrics file in starter):
        → use it
  3. Else:
        → choose a starter case as the baseline case (clone), make sure
          it's configured to use the baseline model/scheme/BC, run it
          via foam_run_simple.py, run the bound comparator (if any) to
          get the baseline RMSE-vs-reference
  4. Write `<run_dir>/baseline_metrics.json`:
       {
         "baseline_required": bool,
         "baseline_model": "<name>",
         "baseline_case_dir": "<absolute path>",
         "primary_score": {"metric": "rmse", "value": 0.0057},
         "secondary": {...},
         "comparator_outputs": [...],
         "source": "starter_data" | "freshly_run" | "skipped"
       }

If baseline_required is False (the topic doesn't ask for baseline comparison),
or if baseline data can't be produced, the file is still written with that
status and `primary_score: null`. Downstream stages handle either case.

CLI:
  python scripts/baseline_setup.py \
      --run-dir <run_dir> \
      --topic "<study topic>" \
      --starter-dir <starter folder> \
      --reference-data-manifest <run_dir>/reference_data_manifest.json \
      --objective-contract <run_dir>/open_ended_discovery/objective_contract.json \
      --output <run_dir>/baseline_metrics.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------

def _bootstrap_paths(repo_root: Path) -> None:
    for p in (repo_root / "src", repo_root / "Foam-Agent" / "src", repo_root / "scripts"):
        sp = str(p)
        if p.is_dir() and sp not in sys.path:
            sys.path.insert(0, sp)


def _read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def _resolve_starter_case(starter_dir: Path) -> Optional[Path]:
    """Find the first directory under starter_dir with both constant/ and system/."""
    if not starter_dir.is_dir():
        return None
    for d in sorted(starter_dir.rglob("*")):
        if not d.is_dir():
            continue
        if (d / "constant").is_dir() and (d / "system").is_dir():
            return d
        if d.relative_to(starter_dir).parts.__len__() > 3:
            break
    return None


# ---------------------------------------------------------------------------
# LLM topic parser — generic
# ---------------------------------------------------------------------------

def llm_classify_baseline_need(topic: str, repo_root: Path) -> Dict[str, Any]:
    """
    Parse the topic for baseline-comparison intent.
    Returns: {required: bool, baseline_name: str, qoi: [...], reference_kind: str}
    Falls back to a permissive 'required=True, baseline_name=auto' if the LLM
    fails — we'd rather over-produce a baseline than skip a needed comparison.
    """
    _bootstrap_paths(repo_root)
    try:
        from langchain_core.messages import HumanMessage, SystemMessage  # type: ignore
        from cfd_langgraph.config import get_settings  # type: ignore
        from cfd_langgraph.llm.factory import create_langchain_llm  # type: ignore
        from cfd_langgraph.utils import strip_json_fences  # type: ignore
    except Exception:
        return {"required": True, "baseline_name": "auto", "qoi": [], "reference_kind": ""}

    sys_msg = (
        "You are a CFD task classifier. Read the topic and return strict JSON "
        "describing whether a baseline comparison is requested.\n"
        "Output JSON keys:\n"
        '  "required": boolean — true if the topic explicitly or implicitly '
        "asks the new variant to be compared against a baseline (built-in) model.\n"
        '  "baseline_name": short string naming the baseline (e.g. '
        '"SpalartAllmaras", "kEpsilon", "kOmegaSST", "Newtonian", '
        '"zeroGradient", "default scheme"). Use "auto" if the topic implies '
        'a baseline but does not name it.\n'
        '  "qoi": list of quantities being compared (e.g. ["Cf"], '
        '["pressure_drop"], ["reattachment_length"]).\n'
        '  "reference_kind": "DNS" | "experiment" | "literature" | "" if not '
        "stated.\n"
        "Return JSON only, no markdown."
    )
    user_msg = f"Topic:\n{topic}\n"
    try:
        llm = create_langchain_llm(model=get_settings().model, temperature=0.0)
        raw = llm.invoke([SystemMessage(content=sys_msg), HumanMessage(content=user_msg)])
        text = strip_json_fences(str(getattr(raw, "content", raw)).strip())
        s, e = text.find("{"), text.rfind("}")
        if s != -1 and e > s:
            text = text[s:e + 1]
        obj = json.loads(text)
        if not isinstance(obj, dict):
            raise ValueError("not a dict")
        return {
            "required": bool(obj.get("required", False)),
            "baseline_name": str(obj.get("baseline_name", "")).strip() or "auto",
            "qoi": list(obj.get("qoi") or []),
            "reference_kind": str(obj.get("reference_kind", "")).strip(),
        }
    except Exception:
        return {"required": True, "baseline_name": "auto", "qoi": [], "reference_kind": ""}


# ---------------------------------------------------------------------------
# look for pre-existing baseline data
# ---------------------------------------------------------------------------

_BASELINE_HINT_PATTERNS = (
    re.compile(r"baseline", re.IGNORECASE),
    re.compile(r"\bvanilla\b", re.IGNORECASE),
    re.compile(r"reference[_-]?model", re.IGNORECASE),
)


def find_existing_baseline_data(starter_dir: Path) -> Dict[str, Any]:
    """Return a dict describing any pre-existing baseline artifacts in starter_dir."""
    out: Dict[str, Any] = {"found": False, "files": [], "case_dirs": []}
    if not starter_dir.is_dir():
        return out
    for d in sorted(starter_dir.rglob("*")):
        if not d.is_dir():
            continue
        if not any(p.search(d.name) for p in _BASELINE_HINT_PATTERNS):
            continue
        if (d / "constant").is_dir() or (d / "system").is_dir():
            out["case_dirs"].append(str(d))
    for ext in ("*.csv", "*.dat", "*.json", "*.md"):
        for f in starter_dir.rglob(ext):
            if any(p.search(str(f)) for p in _BASELINE_HINT_PATTERNS):
                if f.is_file():
                    out["files"].append(str(f))
    out["found"] = bool(out["case_dirs"]) or bool(out["files"])
    return out


# ---------------------------------------------------------------------------
# RMSE extractor — same regex set as open_ended_discovery
# ---------------------------------------------------------------------------

# Generic metric extractor — covers the lower-is-better family (RMSE, L2,
# MAE, MSE, max/relative error, deviation) and the higher-is-better family
# (correlation, R², agreement, accuracy). Direction is classified by metric
# name so any new study (drag agreement, heat-flux correlation, viscosity-
# profile RMSE, ...) gets the right gate downstream.
_METRIC_RE = re.compile(
    r"(?i)\b("
    r"rmse|rms[\s_-]*error|l[12](?:[\s_-]*error)?|mae|mse|"
    r"max[\s_-]*error|deviation|relative[\s_-]*error|"
    r"correlation|pearson|spearman|r\s*\^?\s*2|r2|agreement|accuracy"
    r")\b[^0-9]*?([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
    # Filler `[^0-9]*?` is NON-GREEDY and matches any non-digit (including
    # hyphens and `-` in words like "exact-match"). Earlier `[^0-9\-+]*` was
    # broken: it excluded `-`, so it stopped at the `-` inside "exact-match"
    # and never reached the number. Non-greedy + leading `[-+]?` on the number
    # means we still capture signed numbers correctly.
)

_MIN_METRICS = ("rmse", "rms", "l2", "l1", "mae", "mse", "maxerror", "deviation",
                "relativeerror", "error", "residual", "loss", "discrepancy")
_MAX_METRICS = ("correlation", "pearson", "spearman", "r2", "r^2", "agreement",
                "accuracy", "iou", "dice", "skill", "precision", "recall", "auc")


def _direction_for_metric(name: str) -> str:
    """Is higher better for this metric, or lower?

    Containment, not ``startswith``. The prefix test only ever saw the bare
    tokens this module's own regex captures, and silently returned "min" for
    every real metric name built around one of them:
    ``velocity_profile_shape_correlation`` and ``recirculation_region_iou``
    are both max-metrics that were classified as min, which would have driven
    a search to minimise a correlation. Both are live names in a study's
    metric_specs.json.

    Min-family is tested first so a compound like "correlation_error" reads as
    an error measure, which is what such a name means.

    A name matching neither family still falls back to "min" — the CFD
    convention — so this is a heuristic and not a substitute for the direction
    the metric proposer states explicitly in metric_specs.json. Prefer that
    field wherever it exists; this exists only for free-text comparator output
    that carries no such field.
    """
    n = re.sub(r"[\s_\-^]", "", name.lower())
    if any(k in n for k in _MIN_METRICS):
        return "min"
    if any(k in n for k in _MAX_METRICS):
        return "max"
    return "min"


def extract_primary_score(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    found: List[Tuple[str, float]] = []
    for m in _METRIC_RE.finditer(text):
        try:
            found.append((m.group(1).lower().replace(" ", ""), float(m.group(2))))
        except Exception:
            continue
    if not found:
        return None
    # Priority: prefer RMSE-family when both lower- and higher-is-better
    # metrics are present (RMSE is the conventional CFD QoI).
    priority = ["rmse", "rmserror", "rms", "l2", "l2error", "l1", "mae", "mse",
                "maxerror", "deviation", "relativeerror",
                "correlation", "pearson", "spearman", "r2", "agreement", "accuracy"]
    for key in priority:
        for name, val in found:
            if name.startswith(key):
                return {"metric": name, "value": val, "direction": _direction_for_metric(name)}
    name, val = found[0]
    return {"metric": name, "value": val, "direction": _direction_for_metric(name)}


# ---------------------------------------------------------------------------
# baseline case generation
# ---------------------------------------------------------------------------

def run_baseline_case(
    *,
    run_dir: Path,
    repo_root: Path,
    starter_case: Path,
    timeout_s: int = 1800,
) -> Dict[str, Any]:
    """
    Clone the starter case (which already names a baseline model — the user
    set it up that way), run it through foam_run_simple.py. Returns the run
    result dict.
    """
    baseline_dir = run_dir / "baseline_case"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    result_path = baseline_dir / "baseline_run_result.json"

    cmd = [
        sys.executable, str(repo_root / "scripts" / "foam_run_simple.py"),
        "--base-case", str(starter_case),
        "--output-dir", str(baseline_dir / "case"),
        "--output", str(result_path),
        "--timeout", str(timeout_s),
    ]
    print(f"[baseline] cloning starter and running: {starter_case} -> {baseline_dir/'case'}")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s + 60)
    if proc.stdout:
        print(proc.stdout[-1500:])
    if proc.stderr:
        print(proc.stderr[-1500:], file=sys.stderr)
    return _read_json(result_path, {})


def _autodetect_comparator(
    starter_dir: Optional[Path],
    *,
    topic: str = "",
    cache_path: Optional[Path] = None,
) -> Optional[Path]:
    """Detect THE comparator script in starter_dir using LLM-based content
    classification. No naming-convention assumption — the user may have named
    their comparator anything (compare_*.py, score_*.py, evaluate_K.py, …).
    Anti-contamination is enforced: only files INSIDE starter_dir can ever
    be returned, regardless of what the LLM picks.

    Cache: if cache_path is set, the classification is written there for
    downstream callers (e.g. open_ended_discovery) to reuse without
    re-spending tokens.

    Falls back to None on any failure (e.g. no LLM creds available); the
    bound_comparators.json path written by Phase 1 then takes over.
    """
    if starter_dir is None or not Path(starter_dir).is_dir():
        return None
    try:
        _scripts_dir = Path(__file__).resolve().parent
        if str(_scripts_dir) not in sys.path:
            sys.path.insert(0, str(_scripts_dir))
        from comparator_classifier import find_comparator_for_starter as _find  # type: ignore
        return _find(
            starter_dir=Path(starter_dir),
            topic=topic,
            cache_path=cache_path,
        )
    except Exception:
        return None


def run_bound_comparator_on_baseline(
    *,
    baseline_case: Path,
    objective_contract: Dict[str, Any],
    starter_dir: Optional[Path] = None,
    topic: str = "",
    classifier_cache_path: Optional[Path] = None,
) -> str:
    """
    Run the bound comparator script (if any) against the baseline case. Same
    logic as the OED bound comparator runner. Returns whatever text it
    produces (stdout/stderr or recovered file content).

    Resolution order for the comparator script:
      1. objective_contract.comparator_script (set by OED — may not exist
         when baseline_setup runs, since OED is a later stage)
      2. LLM-content-classified comparator under starter_dir
         (no naming-convention assumption; cached at classifier_cache_path
         so OED later reuses without re-spending tokens).
    """
    script = str(objective_contract.get("comparator_script", "") or "").strip()
    sp: Optional[Path] = None
    if script and Path(script).is_file():
        sp = Path(script)
    if sp is None:
        sp = _autodetect_comparator(
            starter_dir,
            topic=topic,
            cache_path=classifier_cache_path,
        )
    if sp is None:
        return ""

    refs = [Path(p) for p in (objective_contract.get("reference_files", []) or []) if Path(p).is_file()]
    if not refs:
        for c in sorted(sp.parent.glob("*.csv")):
            if c.stat().st_size > 0:
                refs.append(c)
                break
    ref = refs[0] if refs else None

    out_dir = baseline_case / "_baseline_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmds: List[List[str]] = []
    if ref:
        cmds.append([sys.executable, str(sp), "--case", str(baseline_case),
                     "--time", "5000", "--reference", str(ref), "--out", str(out_dir)])
        cmds.append([sys.executable, str(sp), "--case", str(baseline_case),
                     "--reference", str(ref), "--out", str(out_dir)])
        cmds.append([sys.executable, str(sp), "--case", str(baseline_case),
                     "--reference", str(ref)])
    cmds.append([sys.executable, str(sp), "--case", str(baseline_case),
                 "--out", str(out_dir)])
    cmds.append([sys.executable, str(sp), "--case", str(baseline_case)])

    # Defensive auto-recovery of post-process fields BEFORE running the
    # comparator. Same logic as OED uses — generic across studies.
    try:
        scripts_dir = Path(__file__).resolve().parent
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        import foam_postprocess as _fpp  # type: ignore
        pp = _fpp.ensure_postprocess_fields(baseline_case, timeout_s=180)
        if pp.get("action") == "postprocess":
            print(f"[baseline][postprocess] action={pp.get('action')} ok={pp.get('ok')} "
                  f"app={pp.get('app')}")
    except Exception as exc:
        print(f"[baseline][postprocess] non-fatal: {exc}")

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
            return blob.strip()

    # Fallback: scan the whole baseline case for any text artifacts a
    # comparator might have written. Generic across studies — does not
    # depend on a particular output directory name. Same shape as the OED
    # comparator-output scanner.
    metric_re = re.compile(
        r"(?i)\b(rmse|rms[\s_-]*error|l[12](?:[\s_-]*error)?|mae|mse|"
        r"max[\s_-]*error|deviation|residual|"
        r"correlation|pearson|spearman|r\s*\^?\s*2|r2|agreement|accuracy)\b"
    )
    found_blob = ""
    files_scanned = 0
    for p in baseline_case.rglob("*"):
        if files_scanned > 400:
            break
        if not p.is_file():
            continue
        if p.suffix.lower() not in (".md", ".txt", ".json", ".csv", ".dat", ".log"):
            continue
        # Skip OpenFOAM time-step field dirs and parallel processor* dirs.
        rel_parts = p.relative_to(baseline_case).parts
        if any(part.replace(".", "").isdigit() for part in rel_parts):
            continue
        if any(part.startswith("processor") for part in rel_parts):
            continue
        try:
            if p.stat().st_size > 200_000:
                continue
            text = p.read_text(encoding="utf-8", errors="replace")[:6000]
        except Exception:
            continue
        files_scanned += 1
        if metric_re.search(text):
            found_blob += f"\n--- {p.relative_to(baseline_case)} ---\n{text[:1500]}"
    if found_blob.strip():
        return found_blob
    return last_blob


# ---------------------------------------------------------------------------
# main pipeline
# ---------------------------------------------------------------------------

def run(
    *,
    run_dir: Path,
    topic: str,
    starter_dir: Optional[Path],
    reference_data_manifest: Optional[Path],
    objective_contract_path: Optional[Path],
    output_path: Path,
    timeout_s: int = 1800,
) -> Dict[str, Any]:
    repo_root = Path(__file__).resolve().parent.parent
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    classify = llm_classify_baseline_need(topic, repo_root=repo_root)
    print(f"[baseline] topic classifier: {classify}")

    objective_contract = _read_json(objective_contract_path, {}) if objective_contract_path else {}
    if not isinstance(objective_contract, dict):
        objective_contract = {}

    out: Dict[str, Any] = {
        "baseline_required": bool(classify.get("required", False)),
        "baseline_name": classify.get("baseline_name", "auto"),
        "qoi": classify.get("qoi", []),
        "reference_kind": classify.get("reference_kind", ""),
        "baseline_case_dir": "",
        "baseline_final_time": None,
        "primary_score": None,
        "secondary": {},
        "comparator_outputs": [],
        "source": "skipped",
        "notes": [],
    }

    # Helper to stamp baseline_final_time into the result dict from a case dir.
    # Used at every successful baseline-resolution path (pre-existing data,
    # freshly run). Time-pinned scoring downstream relies on this field.
    def _stamp_baseline_final_time(case_dir_path: Path) -> None:
        try:
            import run_validity as _rv  # type: ignore
            t = _rv.detect_max_time(case_dir_path)
            if t > 0:
                out["baseline_final_time"] = float(t)
        except Exception as _exc:
            out.setdefault("notes", []).append(
                f"baseline_final_time detect failed: {_exc}"
            )

    if not classify.get("required", False):
        out["notes"].append("topic classifier did not flag a baseline comparison; skipping.")
        _write_json(output_path, out)
        print(json.dumps(out, indent=2))
        return out

    # 1) any pre-existing baseline data?
    if starter_dir is not None and starter_dir.is_dir():
        pre = find_existing_baseline_data(starter_dir)
        if pre.get("found"):
            out["notes"].append(f"pre-existing baseline artifacts in starter: {pre}")
            # If a baseline case dir is shipped, prefer running the comparator
            # against it (same as a freshly-run baseline) so we get a number.
            if pre.get("case_dirs"):
                base_case = Path(pre["case_dirs"][0])
                if (base_case / "log.simpleFoam").is_file():
                    cmp_text = run_bound_comparator_on_baseline(
                        baseline_case=base_case, objective_contract=objective_contract,
                        starter_dir=starter_dir, topic=topic,
                        classifier_cache_path=run_dir / "open_ended_discovery" / "comparator_classification.json")
                    if cmp_text:
                        out["primary_score"] = extract_primary_score(cmp_text)
                        out["comparator_outputs"].append(cmp_text[:2000])
                        out["baseline_case_dir"] = str(base_case)
                        out["source"] = "starter_data"
                        _stamp_baseline_final_time(base_case)
                        if out["primary_score"] is not None:
                            _write_json(output_path, out)
                            print(json.dumps({k: v for k, v in out.items()
                                              if k != "comparator_outputs"}, indent=2))
                            return out

    # 2) generate fresh baseline by cloning the starter case (which already
    # carries the baseline-model dictionary configuration) and running it.
    starter_case = _resolve_starter_case(starter_dir) if starter_dir else None
    if starter_case is None:
        out["notes"].append("no starter case found; cannot generate baseline.")
        _write_json(output_path, out)
        return out

    run_result = run_baseline_case(
        run_dir=run_dir, repo_root=repo_root,
        starter_case=starter_case, timeout_s=timeout_s,
    )
    if run_result.get("status") != "OK":
        out["notes"].append(f"baseline run failed: {run_result.get('error', '(unknown)')}")
        out["source"] = "freshly_run_failed"
        _write_json(output_path, out)
        print(json.dumps({k: v for k, v in out.items()
                          if k != "comparator_outputs"}, indent=2))
        return out

    base_case_dir = Path(run_result.get("case_dir", run_dir / "baseline_case" / "case"))
    out["baseline_case_dir"] = str(base_case_dir)
    out["source"] = "freshly_run"
    _stamp_baseline_final_time(base_case_dir)

    cmp_text = run_bound_comparator_on_baseline(
        baseline_case=base_case_dir, objective_contract=objective_contract,
        starter_dir=starter_dir, topic=topic,
        classifier_cache_path=run_dir / "open_ended_discovery" / "comparator_classification.json")
    if cmp_text:
        out["primary_score"] = extract_primary_score(cmp_text)
        out["comparator_outputs"].append(cmp_text[:2000])
    else:
        out["notes"].append("no comparator script found; baseline ran but no RMSE extracted.")

    _write_json(output_path, out)
    print(json.dumps({k: v for k, v in out.items()
                      if k != "comparator_outputs"}, indent=2, default=str))
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Baseline setup stage.")
    parser.add_argument("--run-dir", required=True, type=str)
    parser.add_argument("--topic", required=True, type=str)
    parser.add_argument("--starter-dir", default="", type=str)
    parser.add_argument("--reference-data-manifest", default="", type=str)
    parser.add_argument("--objective-contract", default="", type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--timeout", default=1800, type=int)
    args = parser.parse_args()

    sd = Path(args.starter_dir).expanduser().resolve() if args.starter_dir else None
    rdm = Path(args.reference_data_manifest).expanduser().resolve() if args.reference_data_manifest else None
    ocp = Path(args.objective_contract).expanduser().resolve() if args.objective_contract else None

    result = run(
        run_dir=Path(args.run_dir).expanduser().resolve(),
        topic=args.topic,
        starter_dir=sd,
        reference_data_manifest=rdm,
        objective_contract_path=ocp,
        output_path=Path(args.output).expanduser().resolve(),
        timeout_s=args.timeout,
    )
    return 0 if result.get("baseline_required") and result.get("primary_score") else 0


if __name__ == "__main__":
    raise SystemExit(main())
