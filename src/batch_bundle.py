#!/usr/bin/env python3
"""Batch bundling + study interpretation (until WriterAgent).

Creates machine-readable JSON artifacts in a batch folder so a downstream WriterAgent
(or human) can draft a paper without re-parsing logs/dirs.

Outputs (in batch_dir):
- idea.json: raw ideation JSON (if provided)
- writer_input.json: per-run experiments + per-run results (writer-facing)
- study_interpretation.json: cross-run interpretation (study-facing)

This module is deterministic: it only uses files produced by the pipeline.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


TIME_TAG_RE = re.compile(r"_t([0-9]+p[0-9]+)$")


def _read_text(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8")
    except Exception:
        return ""


def _read_json(p: Path) -> Any:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_json(p: Path, obj: Any) -> None:
    p.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _rel(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve())).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def _parse_log_pimplefoam(log_path: Path) -> Dict[str, Any]:
    if not log_path.exists():
        return {}

    txt = log_path.read_text(errors="ignore")

    means: List[float] = []
    maxs: List[float] = []
    for m in re.finditer(r"Courant Number mean:\s*([0-9.eE+-]+)\s*max:\s*([0-9.eE+-]+)", txt):
        means.append(float(m.group(1)))
        maxs.append(float(m.group(2)))

    dts = [float(x) for x in re.findall(r"deltaT\s*=\s*([0-9.eE+-]+)", txt)]
    ex = re.findall(r"ExecutionTime\s*=\s*([0-9.]+)\s*s", txt)

    return {
        "courant_entries": len(maxs),
        "co_mean_median": float(np.median(means)) if means else None,
        "co_mean_max": float(max(means)) if means else None,
        "co_max_median": float(np.median(maxs)) if maxs else None,
        "co_max_peak": float(max(maxs)) if maxs else None,
        "deltaT_min": float(min(dts)) if dts else None,
        "deltaT_median": float(np.median(dts)) if dts else None,
        "deltaT_max": float(max(dts)) if dts else None,
        "execution_time_s": float(ex[-1]) if ex else None,
    }


def _summarize_centerline(csv_path: Path) -> Optional[Dict[str, float]]:
    if not csv_path.exists():
        return None

    try:
        data = np.genfromtxt(csv_path, delimiter=",", names=True)
        y = np.asarray(data["y"], dtype=float)
        Uy = np.asarray(data["Uy"], dtype=float)
        if Uy.size == 0:
            return None

        def near(val: float) -> float:
            i = int(np.argmin(np.abs(y - val)))
            return float(Uy[i])

        return {
            "Uy_min": float(np.min(Uy)),
            "Uy_max": float(np.max(Uy)),
            "Uy_y0": near(0.0),
            "Uy_yMid": near(0.1),
            "Uy_yTop": near(0.2),
        }
    except Exception:
        return None


def _extract_time_from_filename(p: Path) -> Optional[float]:
    # umag_t0p10.png -> 0.10
    stem = p.stem
    if "_t" not in stem:
        return None
    m = TIME_TAG_RE.search(stem)
    if not m:
        return None
    try:
        return float(m.group(1).replace("p", "."))
    except Exception:
        return None


def build_writer_input(
    *,
    batch_dir: Path,
    study_goal: str,
    idea_json: Optional[dict] = None,
) -> Tuple[dict, dict, dict]:
    """Return (writer_input, study_interpretation, batch_archive)."""

    batch_dir = Path(batch_dir).resolve()

    # Discover experiments/runs from experiment_results.json (preferred)
    exp_results_path = batch_dir / "experiment_results.json"
    exp_results = _read_json(exp_results_path) if exp_results_path.exists() else None

    if not isinstance(exp_results, dict):
        exp_results = {}

    batch_name = str(exp_results.get("batch_name") or batch_dir.name)

    # Try load selection metadata
    selection = _read_json(batch_dir / "selector_selection.json")
    if not isinstance(selection, dict):
        selection = {}

    # Locate sim dir
    sim_dirs = sorted([d for d in batch_dir.iterdir() if d.is_dir() and d.name.startswith("sim_")])
    if not sim_dirs:
        raise RuntimeError(f"No sim_* directories found in batch: {batch_dir}")
    sim_dir = sim_dirs[0]

    # Index: requirement_index -> run_dir
    results_list = exp_results.get("results") or []
    if not isinstance(results_list, list):
        results_list = []

    runs_payload: List[dict] = []

    for entry in results_list:
        if not isinstance(entry, dict):
            continue
        req_i = entry.get("requirement_index")
        case_id = entry.get("case_id") or (entry.get("simulation_config") or {}).get("case_id")
        result = entry.get("result") or {}
        run_dir = Path(result.get("run_dir") or "")
        out_dir = Path(result.get("output_dir") or "")
        if not run_dir.exists():
            # fallback to sim_dir/run_###
            if isinstance(req_i, int):
                run_dir = sim_dir / f"run_{int(req_i):03d}"
        if not out_dir.exists():
            out_dir = run_dir / "output"

        ur_path = run_dir / "user_requirement.txt"
        ur_text = _read_text(ur_path).strip()

        artifacts_path = out_dir / "artifacts.json"
        artifacts = _read_json(artifacts_path) if artifacts_path.exists() else None
        if not isinstance(artifacts, dict):
            artifacts = {}

        files = artifacts.get("files") or {}
        if not isinstance(files, dict):
            files = {}

        umag_items = files.get("umag_pngs") or []
        p_items = files.get("p_pngs") or []
        uy_items = files.get("uy_csvs") or []

        # Centerline summaries per time
        uy_summary_rows = []
        for it in uy_items if isinstance(uy_items, list) else []:
            if not isinstance(it, dict):
                continue
            t = it.get("t")
            csv_p = Path(it.get("path") or "")
            s = _summarize_centerline(csv_p)
            if s is None:
                continue
            uy_summary_rows.append({"t": float(t) if t is not None else None, "path": _rel(csv_p, batch_dir), **s})
        uy_summary_rows.sort(key=lambda d: (d.get("t") is None, float(d.get("t") or 0.0)))

        # Log stats
        log_stats = _parse_log_pimplefoam(out_dir / "log.pimpleFoam")

        # Analysis verdict (if present)
        verdict_path = run_dir / "analysis_verdict.json"
        verdict = _read_json(verdict_path) if verdict_path.exists() else None

        # Extract parameters from selection if available
        param_info = None
        if isinstance(selection.get("selected"), list):
            for s in selection["selected"]:
                if isinstance(s, dict) and s.get("candidate_id") == case_id:
                    param_info = s
                    break
        if param_info is None and isinstance(selection.get("hero"), dict) and selection.get("hero", {}).get("candidate_id") == case_id:
            param_info = selection.get("hero")

        runs_payload.append(
            {
                "case_id": case_id,
                "requirement_index": req_i,
                "run_dir": _rel(run_dir, batch_dir),
                "output_dir": _rel(out_dir, batch_dir),
                "user_requirement_path": _rel(ur_path, batch_dir),
                "user_requirement": ur_text,
                "parameters": param_info or {},
                "artifacts": {
                    "artifacts_json": _rel(artifacts_path, batch_dir) if artifacts_path.exists() else None,
                    "requested_times": artifacts.get("requested_times"),
                    "used_times": artifacts.get("used_times"),
                    "umag_images": [
                        {"t": it.get("t"), "path": _rel(Path(it.get("path") or ""), batch_dir)}
                        for it in (umag_items if isinstance(umag_items, list) else [])
                        if isinstance(it, dict)
                    ],
                    "p_images": [
                        {"t": it.get("t"), "path": _rel(Path(it.get("path") or ""), batch_dir)}
                        for it in (p_items if isinstance(p_items, list) else [])
                        if isinstance(it, dict)
                    ],
                    "uy_csvs": [
                        {"t": it.get("t"), "path": _rel(Path(it.get("path") or ""), batch_dir)}
                        for it in (uy_items if isinstance(uy_items, list) else [])
                        if isinstance(it, dict)
                    ],
                    "uy_summary": uy_summary_rows,
                    "log_stats": log_stats,
                },
                "analysis_agent": {
                    "analysis_verdict_path": _rel(verdict_path, batch_dir) if verdict_path.exists() else None,
                    "analysis_verdict": verdict if isinstance(verdict, dict) else None,
                },
            }
        )

    # Writer-facing structure (experiment per run)
    idea_name = None
    if isinstance(idea_json, dict):
        idea_name = idea_json.get("study_id") or idea_json.get("description")
    if not idea_name:
        idea_name = str(exp_results.get("simulation_description") or "CFD study")

    experiments = []
    results = []

    for i, r in enumerate(runs_payload, 1):
        case_id = r.get("case_id") or f"run_{i:03d}"
        params = r.get("parameters") or {}

        exp_params = {
            "case_id": case_id,
            "fuel_velocity": params.get("fuel_velocity"),
            "inlet_box": params.get("box"),
            "inlet_box_width": params.get("box_width"),
            "inlet_box_height": params.get("box_height"),
        }

        experiments.append(
            {
                "experiment_id": i,
                "experiment_name": str(case_id),
                "experiment_description": f"Run {i} in batch {batch_name}: {study_goal}",
                "experiment_parameters": exp_params,
                "user_requirement": r.get("user_requirement"),
            }
        )

        # Per-run data table: centerline Uy summaries
        uy_sum = (r.get("artifacts") or {}).get("uy_summary") or []
        columns = ["t [s]", "Uy_min [m/s]", "Uy_max [m/s]", "Uy(y=0)", "Uy(y=0.10)", "Uy(y=0.20)"]
        values = []
        for row in uy_sum:
            if not isinstance(row, dict) or row.get("t") is None:
                continue
            values.append(
                [
                    float(row["t"]),
                    float(row.get("Uy_min")),
                    float(row.get("Uy_max")),
                    float(row.get("Uy_y0")),
                    float(row.get("Uy_yMid")),
                    float(row.get("Uy_yTop")),
                ]
            )

        # Image list (cite all times for both fields)
        imgs = []
        for it in (r.get("artifacts") or {}).get("umag_images") or []:
            imgs.append(it.get("path"))
        for it in (r.get("artifacts") or {}).get("p_images") or []:
            imgs.append(it.get("path"))

        # Key findings: strictly derived from CSV/log summary
        log_stats = (r.get("artifacts") or {}).get("log_stats") or {}
        uy_final = None
        if values:
            # last row by time
            values_sorted = sorted(values, key=lambda v: v[0])
            uy_final = values_sorted[-1]

        key_findings_parts = []
        if log_stats.get("co_max_peak") is not None:
            key_findings_parts.append(f"Courant peak (max) = {log_stats.get('co_max_peak'):.3g}.")
        if uy_final:
            key_findings_parts.append(
                f"At t={uy_final[0]:.2f}s, centerline Uy_max={uy_final[2]:.3g} m/s and Uy(y=0.10)={uy_final[4]:.3g} m/s."
            )
        key_findings = " ".join(key_findings_parts) if key_findings_parts else "See data_table and figures."

        results.append(
            {
                "experiment_name": str(case_id),
                "key_findings": key_findings,
                "data_table": {"columns": columns, "values": values},
                "images": [x for x in imgs if isinstance(x, str) and x],
                "evidence": {
                    "artifacts_json": (r.get("artifacts") or {}).get("artifacts_json"),
                    "analysis_verdict_path": (r.get("analysis_agent") or {}).get("analysis_verdict_path"),
                },
            }
        )

    writer_input = {
        "idea_name": idea_name,
        "short_hypothesis": study_goal,  # compatibility with existing writer expectations
        "experiments": experiments,
        "results": results,
    }

    # Cross-run interpretation (study-facing): simple deterministic comparisons
    study_interp = {
        "batch": batch_name,
        "study_goal": study_goal,
        "num_runs": len(runs_payload),
        "notes": [
            "This interpretation is deterministic and only uses CSV/log-derived numbers plus file citations.",
            "Qualitative flow-field interpretation should reference the cited UMag/p figures directly.",
        ],
        "comparisons": [],
    }

    # Try compare fuel steps at mid box if available
    def _get_metric(run: dict, metric: str = "Uy_max", time_s: float = 3.0) -> Optional[float]:
        uy_sum = (run.get("artifacts") or {}).get("uy_summary") or []
        best = None
        for r2 in uy_sum:
            if r2.get("t") is None:
                continue
            if abs(float(r2["t"]) - float(time_s)) < 1e-9:
                best = r2
                break
        if not best:
            return None
        return float(best.get(metric))

    # build lookup by (fuel, box)
    lookup = {}
    for r in runs_payload:
        p = r.get("parameters") or {}
        fuel = p.get("fuel_velocity")
        box = p.get("box")
        if fuel is None or box is None:
            continue
        lookup[(float(fuel), str(box))] = r

    # If we have at least 2 distinct fuels at same box, compare
    boxes = sorted(set([b for (_f, b) in lookup.keys()]))
    for b in boxes:
        fuels = sorted([f for (f, bb) in lookup.keys() if bb == b])
        if len(fuels) >= 2:
            f0, f1 = fuels[0], fuels[-1]
            r0, r1 = lookup[(f0, b)], lookup[(f1, b)]
            m0 = _get_metric(r0, "Uy_max", 3.0)
            m1 = _get_metric(r1, "Uy_max", 3.0)
            if m0 is not None and m1 is not None:
                study_interp["comparisons"].append(
                    {
                        "comparison": "fuel_velocity_effect_at_fixed_box",
                        "box": b,
                        "fuel_low": f0,
                        "fuel_high": f1,
                        "metric": "Uy_max at t=3.00s",
                        "value_low": m0,
                        "value_high": m1,
                        "delta": float(m1 - m0),
                        "evidence": {
                            "run_low": r0.get("case_id"),
                            "run_high": r1.get("case_id"),
                            "csv_low": next((x.get("path") for x in r0.get("artifacts", {}).get("uy_csvs", []) if float(x.get("t") or 0.0) == 3.0), None),
                            "csv_high": next((x.get("path") for x in r1.get("artifacts", {}).get("uy_csvs", []) if float(x.get("t") or 0.0) == 3.0), None),
                        },
                    }
                )
            break

    batch_archive = {
        "batch": batch_name,
        "study_goal": study_goal,
        "idea_json": idea_json,
        "selection": selection,
        "runs": runs_payload,
    }

    return writer_input, study_interp, batch_archive


def write_all(
    *,
    batch_dir: Path,
    study_goal: str,
    idea_json: Optional[dict] = None,
) -> Dict[str, str]:
    batch_dir = Path(batch_dir).resolve()

    # Persist idea.json
    if isinstance(idea_json, dict):
        _write_json(batch_dir / "idea.json", idea_json)

    writer_input, study_interp, batch_archive = build_writer_input(
        batch_dir=batch_dir,
        study_goal=study_goal,
        idea_json=idea_json,
    )

    _write_json(batch_dir / "writer_input.json", writer_input)
    _write_json(batch_dir / "study_interpretation.json", study_interp)
    _write_json(batch_dir / "batch_archive.json", batch_archive)

    return {
        "writer_input": str(batch_dir / "writer_input.json"),
        "study_interpretation": str(batch_dir / "study_interpretation.json"),
        "batch_archive": str(batch_dir / "batch_archive.json"),
        "idea": str(batch_dir / "idea.json") if isinstance(idea_json, dict) else "",
    }


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Bundle a CFD-Scientist batch for interpretation/writing")
    ap.add_argument("--batch", required=True, help="Batch name under data/experiments")
    ap.add_argument("--study-goal", required=True, help="Study goal / research question")
    ap.add_argument("--idea", default=None, help="Optional path to idea JSON")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    batch_dir = repo_root / "data" / "experiments" / args.batch

    idea_json = None
    if args.idea:
        idea_json = _read_json(Path(args.idea))
        if not isinstance(idea_json, dict):
            idea_json = None

    paths = write_all(batch_dir=batch_dir, study_goal=args.study_goal, idea_json=idea_json)
    print(json.dumps(paths, indent=2))


if __name__ == "__main__":
    main()
