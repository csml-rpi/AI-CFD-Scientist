#!/usr/bin/env python3
"""Retrospective ablation: walk decision.json files in gpt-5.5 cfd-scientist runs and
tabulate VLM verdicts. The REVISE rate is the silent-failure rate that VLM caught and
NoVerify would have propagated downstream.

Usage:
  python3 scripts/inventory_decisions.py [run_dirs...]
Default run set = the 5 gpt-5.5 cfd-scientist run dirs in runs/nips_2026/.
"""
from __future__ import annotations
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

DEFAULT_RUNS = [
    REPO / "runs" / "nips_2026" / "custom_viscosity_channel_codex_gpt55",
    REPO / "runs" / "nips_2026" / "jet_oscillation_re_codex_gpt55",
    REPO / "runs" / "nips_2026" / "oed_codex_gpt55_cf_only",
    REPO / "runs" / "nips_2026" / "sa_custom_periodic_hill_codex_gpt55",
    REPO / "runs" / "nips_2026" / "turb_model_sensitivity_bfs_codex_gpt55",
]


def classify_revise_reason(d: dict) -> str:
    """Bucket REVISE reasons into rough physics categories.

    Looks at decision['reason'], decision['suggested_changes'], and the inner
    raw['issues'] / raw['reasons'] fields.
    """
    raw = d.get("raw") or {}
    reason_blob = " ".join(
        str(d.get(k) or "") for k in ("reason", "suggested_changes")
    ) + " " + " ".join(
        str(raw.get(k) or "") for k in ("issues", "reasons", "summary")
    )
    blob = reason_blob.lower()

    # Order matters: more-specific buckets first
    if any(t in blob for t in ("geometry mismatch", "wrong geometry", "scenario mismatch",
                                "scenario does not match", "domain shape", "wrong domain",
                                "wrong scenario", "geometry does not match",
                                "geometry/scenario")):
        return "geometry_mismatch"
    if any(t in blob for t in ("reattach", "separation point", "recirculat",
                                "shear layer", "wake")):
        return "wrong_recirc_or_reattach"
    if any(t in blob for t in ("boundary condition", "bc ", "inlet", "outlet",
                                "wall function", "y+ out of range")):
        return "bc_or_wall"
    if any(t in blob for t in ("mesh", "cell count", "resolution", "y+")):
        return "mesh_artifact"
    if any(t in blob for t in ("diverg", "nan", "blow-up", "blow up", "unstable",
                                "oscillat", "courant")):
        return "numerical"
    if any(t in blob for t in ("unphysical", "non-physical", "unrealistic",
                                "physically implausible")):
        return "unphysical_field"
    if any(t in blob for t in ("incomplete", "no data", "no fields", "missing",
                                "no time direct", "case not found")):
        return "incomplete_run"
    return "other"


def walk_run(run_dir: Path) -> dict:
    decisions = list(run_dir.rglob("decision.json"))
    rows = []
    for p in decisions:
        try:
            d = json.loads(p.read_text())
        except Exception as e:
            rows.append({"path": p, "status": "UNREADABLE", "bucket": "_parse_error",
                         "snippet": str(e)[:80]})
            continue
        status = (d.get("status") or "").upper()
        # Some pipelines use "pending" or empty; treat blank as PROCEED only if
        # the inner raw says so, else "OTHER"
        if not status:
            raw = d.get("raw") or {}
            if raw.get("simulation_success") and raw.get("requirement_met"):
                status = "PROCEED"
            else:
                status = "OTHER"
        bucket = "ok" if status == "PROCEED" else classify_revise_reason(d)
        snippet = (d.get("reason") or (d.get("raw") or {}).get("summary") or "")[:120]
        # Determine path category (mesh-gate vs case vs OED iteration)
        relp = p.relative_to(run_dir)
        parts = relp.parts
        if "mesh_gate" in parts:
            path_cat = "mesh_gate"
        elif "open_ended_discovery" in parts:
            path_cat = "oed_iter"
        elif "code_mod_validation" in parts:
            path_cat = "code_mod_validation"
        elif "cases" in parts:
            path_cat = "case"
        else:
            path_cat = "other"
        rows.append({
            "path": str(relp),
            "path_cat": path_cat,
            "status": status,
            "bucket": bucket,
            "snippet": snippet,
        })
    return {"run_dir": str(run_dir.relative_to(REPO)), "n": len(rows), "rows": rows}


def render_run_table(report: dict) -> str:
    """Per-run summary."""
    lines = []
    lines.append(f"\n=== {report['run_dir']}  (n={report['n']}) ===")
    if report["n"] == 0:
        lines.append("  (no decision.json files)")
        return "\n".join(lines)
    by_status = Counter(r["status"] for r in report["rows"])
    by_path_cat = defaultdict(Counter)
    for r in report["rows"]:
        by_path_cat[r["path_cat"]][r["status"]] += 1
    lines.append("  Status (overall):")
    for s in ["PROCEED", "REVISE", "RERUN", "OTHER", "UNREADABLE"]:
        if by_status.get(s, 0):
            pct = 100.0 * by_status[s] / report["n"]
            lines.append(f"    {s:11s}: {by_status[s]:4d}  ({pct:5.1f}%)")
    lines.append("  By path category:")
    for cat, counts in sorted(by_path_cat.items()):
        total = sum(counts.values())
        bits = [f"{s}={n}" for s, n in counts.most_common() if n > 0]
        lines.append(f"    {cat:22s} (n={total:3d}): {', '.join(bits)}")
    # REVISE/RERUN bucket breakdown
    revise_buckets = Counter(r["bucket"] for r in report["rows"] if r["status"] in ("REVISE", "RERUN"))
    if revise_buckets:
        lines.append("  REVISE/RERUN reason buckets:")
        for b, n in revise_buckets.most_common():
            lines.append(f"    {b:30s}: {n}")
    return "\n".join(lines)


def render_overall_table(reports: list) -> str:
    total = sum(r["n"] for r in reports)
    rows_all = [row for r in reports for row in r["rows"]]
    by_status = Counter(r["status"] for r in rows_all)
    revise_buckets = Counter(r["bucket"] for r in rows_all if r["status"] in ("REVISE", "RERUN"))

    # Headline number: REVISE rate (NoVerify would have propagated these as PROCEED)
    n_revise = by_status.get("REVISE", 0)
    n_rerun = by_status.get("RERUN", 0)
    n_other = by_status.get("OTHER", 0)
    n_proceed = by_status.get("PROCEED", 0)

    lines = []
    lines.append("\n" + "=" * 70)
    lines.append(f"OVERALL — gpt-5.5 cfd-scientist runs in nips_2026/  (total decisions = {total})")
    lines.append("=" * 70)
    if total == 0:
        lines.append("(no decisions found)")
        return "\n".join(lines)
    lines.append(f"  PROCEED:   {n_proceed:5d}  ({100.0*n_proceed/total:5.1f}%) — both NoVerify and VLM keep")
    lines.append(f"  REVISE:    {n_revise:5d}  ({100.0*n_revise/total:5.1f}%) — VLM flags physics; NoVerify keeps")
    lines.append(f"  RERUN:     {n_rerun:5d}  ({100.0*n_rerun/total:5.1f}%) — VLM flags numerics")
    lines.append(f"  OTHER:     {n_other:5d}  ({100.0*n_other/total:5.1f}%) — blank/ambiguous status")
    lines.append("")
    silent_failure_rate = 100.0 * n_revise / total
    lines.append(f">>> Silent-failure rate (VLM-caught REVISE) = {silent_failure_rate:.1f}% ({n_revise}/{total})")
    lines.append(f">>> Total VLM-flagged (REVISE + RERUN)       = {100.0*(n_revise+n_rerun)/total:.1f}% ({n_revise+n_rerun}/{total})")
    if revise_buckets:
        lines.append("")
        lines.append("  REVISE/RERUN reason breakdown across all runs:")
        for b, n in revise_buckets.most_common():
            pct_of_flagged = 100.0 * n / max(1, n_revise + n_rerun)
            lines.append(f"    {b:30s}: {n:4d}  ({pct_of_flagged:5.1f}% of flagged)")

    # Per-run REVISE rate table
    lines.append("")
    lines.append("  Per-run breakdown:")
    lines.append(f"    {'run':50s} {'n':>4s}  {'PROC':>5s}  {'REV':>5s}  {'RER':>5s}  {'REV%':>6s}")
    for r in reports:
        s = Counter(row["status"] for row in r["rows"])
        n = r["n"]
        rev = s.get("REVISE", 0)
        rer = s.get("RERUN", 0)
        prc = s.get("PROCEED", 0)
        rev_pct = 100.0 * rev / n if n else 0.0
        name = Path(r["run_dir"]).name
        lines.append(f"    {name:50s} {n:>4d}  {prc:>5d}  {rev:>5d}  {rer:>5d}  {rev_pct:>5.1f}%")
    return "\n".join(lines)


def render_revise_examples(reports: list, n: int = 8) -> str:
    """Show sample REVISE reasons so the user can sanity-check the bucketing."""
    lines = ["\n  Sample REVISE/RERUN reasons (random spot-check):"]
    samples = [(rep["run_dir"], row) for rep in reports for row in rep["rows"]
               if row["status"] in ("REVISE", "RERUN")]
    if not samples:
        return "\n  (no REVISE/RERUN to sample)"
    for run_dir, row in samples[:n]:
        rname = Path(run_dir).name
        lines.append(f"    [{rname}] {row['path']}  status={row['status']} bucket={row['bucket']}")
        if row["snippet"]:
            lines.append(f"        reason: {row['snippet']!r}")
    return "\n".join(lines)


def main(argv: list) -> int:
    runs = [Path(p) for p in argv[1:]] if argv[1:] else DEFAULT_RUNS
    reports = [walk_run(r) for r in runs]
    for r in reports:
        print(render_run_table(r))
    print(render_overall_table(reports))
    print(render_revise_examples(reports, n=10))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
