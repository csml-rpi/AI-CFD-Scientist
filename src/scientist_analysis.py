#!/usr/bin/env python3
"""Deterministic, evidence-cited scientist analysis for a single batch.

Design goal: make it hard to hallucinate.

This module ONLY uses:
- files produced by the run (artifacts.json, uy_centerline_t*.csv, log.pimpleFoam)
- and batch metadata (selector_selection.json, experiment_results.json)

It does NOT attempt to interpret PNG images (that requires vision models). Instead, it:
- cites the exact figure filenames and times
- computes quantitative summaries from CSV/logs
- produces a verdict (supported/refuted/inconclusive)

For a single-case batch, the verdict for a parametric hypothesis should be 'inconclusive'.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Verdict:
    verdict: str  # supported|refuted|inconclusive
    rationale: str


def _read_json(p: Path) -> Dict[str, Any]:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _rel(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve())).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def parse_log_pimplefoam(log_path: Path) -> Dict[str, Any]:
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

    out: Dict[str, Any] = {
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
    return out


def summarize_centerline_csv(csv_path: Path) -> Optional[Dict[str, float]]:
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


def decide_verdict(*, hypothesis: str, num_cases: int) -> Verdict:
    # Deterministic policy: with one case, parameter-effect hypothesis cannot be supported/refuted.
    if num_cases <= 1:
        return Verdict(
            verdict="inconclusive",
            rationale=(
                "Only one case/run is present in this batch. The hypothesis concerns parameter effects "
                "(fuel velocity and inlet box size), which requires at least two distinct parameter settings "
                "to support or refute."
            ),
        )

    # Placeholder for future multi-case logic.
    return Verdict(verdict="inconclusive", rationale="Multi-case verdict logic not implemented yet.")


def build_scientist_analysis(batch_dir: Path) -> Dict[str, Any]:
    batch_dir = Path(batch_dir).resolve()

    selection = _read_json(batch_dir / "selector_selection.json")
    exp_results = _read_json(batch_dir / "experiment_results.json")

    hypothesis = str(selection.get("hypothesis", "")).strip() or "(missing hypothesis)"

    # Locate first run
    exp_dirs = sorted([d for d in batch_dir.iterdir() if d.is_dir() and d.name.startswith("sim_")])
    if not exp_dirs:
        raise RuntimeError(f"No sim_* experiment directories found under: {batch_dir}")

    exp_dir = exp_dirs[0]
    run_dirs = sorted([d for d in exp_dir.iterdir() if d.is_dir() and d.name.startswith("run_")])
    if not run_dirs:
        raise RuntimeError(f"No run_* directories found under: {exp_dir}")

    run_dir = run_dirs[0]
    out_dir = run_dir / "output"

    artifacts = _read_json(out_dir / "artifacts.json")

    requested_times = artifacts.get("requested_times") or []
    files = artifacts.get("files") or {}

    umag = files.get("umag_pngs") or []
    p_pngs = files.get("p_pngs") or []
    uy_csvs = files.get("uy_csvs") or []

    # Summarize Uy CSVs
    uy_rows: List[Dict[str, Any]] = []
    for item in uy_csvs:
        t = item.get("t")
        p = Path(item.get("path"))
        s = summarize_centerline_csv(p)
        if s is None:
            continue
        uy_rows.append({"t": float(t) if t is not None else None, "path": _rel(p, batch_dir), **s})

    uy_rows.sort(key=lambda d: (d.get("t") is None, float(d.get("t") or 0.0)))

    # Log stats
    log_stats = parse_log_pimplefoam(out_dir / "log.pimpleFoam")

    num_cases = int(exp_results.get("total_cases") or 0) or 0
    verdict = decide_verdict(hypothesis=hypothesis, num_cases=num_cases)

    return {
        "batch": batch_dir.name,
        "hypothesis": hypothesis,
        "hero": selection.get("hero"),
        "requested_times": requested_times,
        "evidence": {
            "umag_images": [{"t": x.get("t"), "path": _rel(Path(x.get("path")), batch_dir)} for x in umag],
            "p_images": [{"t": x.get("t"), "path": _rel(Path(x.get("path")), batch_dir)} for x in p_pngs],
            "uy_csvs": [{"t": x.get("t"), "path": _rel(Path(x.get("path")), batch_dir)} for x in uy_csvs],
            "uy_summary": uy_rows,
            "log_stats": log_stats,
        },
        "verdict": {"verdict": verdict.verdict, "rationale": verdict.rationale},
    }


def render_analysis_markdown(analysis: Dict[str, Any]) -> str:
    hero = analysis.get("hero") or {}
    lines: List[str] = []
    lines.append(f"# CFD Scientist Analysis — {analysis.get('batch','')}")
    lines.append("")
    lines.append("## Hypothesis")
    lines.append(str(analysis.get("hypothesis", "")).strip())
    lines.append("")

    if hero:
        lines.append("## Hero case")
        lines.append(f"- candidate_id: {hero.get('candidate_id')}")
        lines.append(f"- fuel_velocity: {hero.get('fuel_velocity')} m/s")
        lines.append(f"- inlet box: {hero.get('box')}")
        lines.append("")

    lines.append("## Evidence inventory (files)")
    ev = analysis.get("evidence") or {}
    for k, title in [("umag_images", "UMag images"), ("p_images", "Pressure images"), ("uy_csvs", "Centerline Uy CSVs")]:
        items = ev.get(k) or []
        lines.append(f"### {title}")
        if not items:
            lines.append("- (missing)")
        else:
            for it in items:
                t = it.get("t")
                tstr = f"t={float(t):.2f}s" if t is not None else "t=?"
                lines.append(f"- {tstr}: `{it.get('path')}`")
        lines.append("")

    # Quantitative Uy summary table
    uy_sum = ev.get("uy_summary") or []
    lines.append("## Quantitative evidence from centerline Uy")
    if not uy_sum:
        lines.append("(No CSV summaries available.)")
    else:
        lines.append("| t [s] | Uy_min [m/s] | Uy_max [m/s] | Uy(y=0) | Uy(y=0.10) | Uy(y=0.20) |")
        lines.append("|---:|---:|---:|---:|---:|---:|")
        for r in uy_sum:
            lines.append(
                f"| {r['t']:.2f} | {r['Uy_min']:.4g} | {r['Uy_max']:.4g} | {r['Uy_y0']:.4g} | {r['Uy_yMid']:.4g} | {r['Uy_yTop']:.4g} |"
            )
    lines.append("")

    # Log stats
    log_stats = ev.get("log_stats") or {}
    lines.append("## Numerical stability evidence (from log.pimpleFoam)")
    if not log_stats:
        lines.append("(log.pimpleFoam not found or could not be parsed.)")
    else:
        lines.append(f"- Courant peak (max): {log_stats.get('co_max_peak')}")
        lines.append(f"- Courant median (max): {log_stats.get('co_max_median')}")
        lines.append(f"- deltaT range [s]: [{log_stats.get('deltaT_min')}, {log_stats.get('deltaT_max')}] (median {log_stats.get('deltaT_median')})")
        lines.append(f"- ExecutionTime [s] (last): {log_stats.get('execution_time_s')}")
    lines.append("")

    # Verdict
    verdict = analysis.get("verdict") or {}
    lines.append("## Verdict")
    lines.append(f"**{verdict.get('verdict','').upper()}**")
    lines.append("")
    lines.append(str(verdict.get("rationale", "")).strip())
    lines.append("")

    lines.append("## Scientist narrative (grounded)")
    lines.append("- Use the evidence inventory above to make claims. For each claim, cite the relevant file(s) and time(s).")
    lines.append("- For qualitative flow-field interpretation, refer to the UMag/p images at the specific times.")
    lines.append("")

    return "\n".join(lines)


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Deterministic scientist analysis for a batch")
    ap.add_argument("--batch", required=True, help="Batch name under data/experiments")
    ap.add_argument("--out", default=None, help="Optional output markdown path")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    batch_dir = repo_root / "data" / "experiments" / args.batch

    analysis = build_scientist_analysis(batch_dir)
    md = render_analysis_markdown(analysis)

    out_path = Path(args.out) if args.out else (batch_dir / "scientist_analysis.md")
    out_path.write_text(md, encoding="utf-8")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
