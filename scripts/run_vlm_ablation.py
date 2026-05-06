#!/usr/bin/env python3
"""VLM verifier ablation driver — constructs 20 controlled cases and records VLM verdicts.

20 cases = 4 failure categories × 4 flow types + 4 controls.

For each case:
  1. Copy template dir from an existing PROCEED'd production case (read-only on the source).
  2. Apply category-specific post-hoc perturbation (no FoamAgent re-run).
  3. Generate diagnostic figures via scripts/viz.py --mode interpret.
  4. Run scripts/interpret.py to get the VLM verdict.
  5. Record decision.json + ground-truth label.

At the end, aggregate per-category recall and overall precision.

Existing production runs are read-only — never modified.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Optional

REPO = Path(__file__).resolve().parent.parent
ABL_DIR = REPO / "runs" / "nips_2026" / "vlm_ablation"

TEMPLATES = {
    "jet":  REPO / "runs/nips_2026/jet_oscillation_re_codex_gpt55/cases/case_001",
    "bfs":  REPO / "runs/nips_2026/turb_model_sensitivity_bfs_codex_gpt55/cases/case_001",
    "hill": REPO / "runs/nips_2026/sa_custom_periodic_hill_codex_gpt55/cases/case_001",
    "chan": REPO / "runs/nips_2026/custom_viscosity_channel_codex_gpt55/cases/case_001_reference",
}

# ---------------------------------------------------------------------------
# Generic well-posed requirement strings per flow type. Used directly for
# controls; planted failures append/modify a single failure-related sentence.

REQ_BASE = {
    "jet": (
        "Run a 2D laminar planar slot jet at Re=60 issuing into quiescent ambient air "
        "with a 1% lateral perturbation at the inlet. Solver: icoFoam, transient, "
        "deltaT=2e-4, endTime=8 s, writeInterval=0.05 s. Place a probe near the "
        "centerline a few slot-widths downstream and report the dominant lateral-velocity "
        "frequency from probe time-series spectral analysis. Mesh: ~6000 cells. "
        "Deliverable: probe time-series and the extracted dominant frequency in Hz."
    ),
    "bfs": (
        "Run a 2D backward-facing-step at Re_h=5100 (based on step height h) with the "
        "k-epsilon turbulence model and standard wall functions. Solver: simpleFoam, "
        "steady, endTime=2000 iterations, writeInterval=200. Use a structured mesh sized "
        "for y+ ~30. Deliverable: lower-wall skin-friction coefficient Cf along the "
        "lower wall as a function of x, sampled via a wallShearStress function object. "
        "Identify the reattachment x-location."
    ),
    "hill": (
        "Run periodic-hill flow at Re_b=10595 using the built-in Spalart-Allmaras "
        "turbulence model on the locked mesh-gate mesh. Solver: simpleFoam, steady, "
        "endTime=5000 iterations, writeInterval=1000. "
        "Deliverable: lower-wall skin-friction coefficient Cf as a function of x/H "
        "across the hill, compared against DNS reference (Frohlich et al). "
        "Report Cf RMSE vs DNS."
    ),
    "chan": (
        "Run 2D Newtonian channel flow with reference kinematic viscosity nu=1e-6 m^2/s. "
        "Solver: simpleFoam, steady, endTime=5000 iterations, writeInterval=500. "
        "Capture the fully-developed velocity profile and report the centerline U value "
        "and the wall shear stress."
    ),
}


@dataclass
class Spec:
    """One ablation case specification."""
    seed_id: str
    category: str   # missing_deliverable | wrong_magnitude_metric | broken_postprocessing | convergence_not_settled | control
    flow: str       # jet | bfs | hill | chan
    label: str      # FAIL (planted failure, expected verdict != PROCEED) or OK (control)
    requirement: str
    perturb_desc: str   # human-readable description for ground_truth.json


# ---------------------------------------------------------------------------
# Perturbation primitives.

def _list_time_dirs(case_dir: Path) -> list[tuple[float, Path]]:
    out = []
    for p in case_dir.iterdir():
        if p.is_dir():
            try:
                t = float(p.name)
                out.append((t, p))
            except ValueError:
                continue
    out.sort(key=lambda x: x[0])
    return out


def _zero_out_columnar_dat(path: Path) -> None:
    """Overwrite a probe/wallShearStress-style ASCII data file with zeros, preserving
    the time column and any header lines starting with #."""
    try:
        text = path.read_text()
    except Exception:
        return
    new_lines = []
    for ln in text.splitlines():
        if not ln.strip() or ln.lstrip().startswith("#"):
            new_lines.append(ln)
            continue
        # Keep the first whitespace-delimited token (time), zero the rest
        toks = ln.split()
        if not toks:
            new_lines.append(ln)
            continue
        n_other = len(toks) - 1
        new_lines.append(toks[0] + (" 0" * n_other))
    path.write_text("\n".join(new_lines) + "\n")


def perturb_missing_deliverable(case_dir: Path, flow: str) -> str:
    """Delete the deliverable artifact."""
    if flow == "jet":
        target = case_dir / "postProcessing" / "jetProbes"
        if target.exists():
            shutil.rmtree(target)
            return f"deleted {target.relative_to(case_dir)}"
        return "no postProcessing/jetProbes to delete"
    if flow == "bfs":
        for sub in ("wallShearStressLowerWall", "wallShearStress", "lineSamples"):
            tgt = case_dir / "postProcessing" / sub
            if tgt.exists():
                shutil.rmtree(tgt)
                return f"deleted {tgt.relative_to(case_dir)}"
        return "no postProcessing wall-shear dir to delete"
    if flow == "hill":
        for sub in ("wallShearStress", "yPlus"):
            tgt = case_dir / "postProcessing" / sub
            if tgt.exists():
                shutil.rmtree(tgt)
                return f"deleted {tgt.relative_to(case_dir)}"
        return "no postProcessing wall-shear dir to delete"
    if flow == "chan":
        # No postProcessing — instead, delete the latest two time dirs (the
        # "developed flow" the requirement asks for is gone).
        times = _list_time_dirs(case_dir)
        deleted = []
        for t, p in times[-2:]:
            shutil.rmtree(p)
            deleted.append(p.name)
        return f"deleted latest time dirs: {deleted}"
    return "noop"


def perturb_broken_postprocessing(case_dir: Path, flow: str) -> str:
    """Replace data file contents with zeros."""
    if flow == "jet":
        target = case_dir / "postProcessing" / "jetProbes"
        if not target.exists():
            return "no jetProbes to broke"
        n = 0
        for f in target.rglob("*"):
            if f.is_file():
                _zero_out_columnar_dat(f)
                n += 1
        return f"zeroed {n} files under {target.relative_to(case_dir)}"
    if flow in ("bfs", "hill"):
        # Zero out wallShearStress files
        n = 0
        for sub in ("wallShearStressLowerWall", "wallShearStress"):
            tgt = case_dir / "postProcessing" / sub
            if not tgt.exists():
                continue
            for f in tgt.rglob("*"):
                if f.is_file():
                    _zero_out_columnar_dat(f)
                    n += 1
        return f"zeroed {n} wallShearStress files"
    if flow == "chan":
        # Zero out U/p fields in the latest time directory
        times = _list_time_dirs(case_dir)
        if not times:
            return "no time dirs"
        last_t, last_p = times[-1]
        for fname in ("U", "p"):
            f = last_p / fname
            if f.exists():
                # OpenFOAM field file — replace internalField list with all zeros
                txt = f.read_text()
                # crude but effective: replace nonuniform List<vector/scalar> ... with uniform zero
                if "U" == fname:
                    new = _replace_internal_field(txt, "(0 0 0)", is_vector=True)
                else:
                    new = _replace_internal_field(txt, "0", is_vector=False)
                f.write_text(new)
        return f"zeroed U/p at t={last_t}"
    return "noop"


def _replace_internal_field(txt: str, uniform_value: str, is_vector: bool) -> str:
    """Crude replacement: find 'internalField   nonuniform List<...>' through the
    matching ');' and replace with 'internalField   uniform <value>;'."""
    import re
    if "nonuniform" not in txt:
        return txt
    pattern = re.compile(r"internalField\s+nonuniform\s+List<\w+>\s*\d+\s*\([^)]*\)\s*;", re.DOTALL)
    return pattern.sub(f"internalField   uniform {uniform_value};", txt, count=1)


def perturb_convergence_not_settled(case_dir: Path, flow: str) -> str:
    """Truncate time dirs, edit controlDict.endTime."""
    truncate_to = {"jet": 0.5, "bfs": 200, "hill": 500, "chan": 500}[flow]
    times = _list_time_dirs(case_dir)
    deleted = []
    for t, p in times:
        if t > truncate_to + 1e-9:
            shutil.rmtree(p)
            deleted.append(p.name)
    # Edit controlDict.endTime
    cd = case_dir / "system" / "controlDict"
    if cd.exists():
        txt = cd.read_text()
        import re
        new = re.sub(r"endTime\s+[0-9eE.+-]+\s*;", f"endTime         {truncate_to};", txt, count=1)
        cd.write_text(new)
    return f"truncated to t<={truncate_to}; deleted {len(deleted)} time dirs; controlDict.endTime updated"


def perturb_noop(case_dir: Path, flow: str) -> str:
    return "no perturbation (control)"


# ---------------------------------------------------------------------------
# Build the 20 specs.

def build_specs() -> list[Spec]:
    specs: list[Spec] = []
    for flow in ("jet", "bfs", "hill", "chan"):
        # missing_deliverable
        req = REQ_BASE[flow] + (
            "\n\nVALIDATION CONTRACT: the deliverable specified above (probe time-series / "
            "wallShearStress / latest time fields) MUST be present in the case's output. "
            "If absent, the run does not satisfy the requirement."
        )
        specs.append(Spec(
            seed_id=f"{flow}_missing_deliv",
            category="missing_deliverable",
            flow=flow,
            label="FAIL",
            requirement=req,
            perturb_desc="(filled at run time)",
        ))

        # wrong_magnitude_metric — modify the requirement to assert an impossible target.
        # Case data is not perturbed; the (requirement, case) pair is the failure.
        if flow == "jet":
            extra = ("\n\nVALIDATION CONTRACT: the dominant lateral-velocity frequency at the centerline "
                     "probe MUST exceed 50 Hz for this case to be considered valid (this is "
                     "deliberately set far above the physically plausible value at Re=60 — the "
                     "purpose is to test that the verifier flags impossible quantitative targets).")
        elif flow == "bfs":
            extra = ("\n\nVALIDATION CONTRACT: lower-wall Cf must remain BELOW 1e-6 along the entire wall "
                     "for this case to be valid (this is deliberately impossible for any real "
                     "BFS simulation at Re_h=5100; the verifier should detect that the case violates "
                     "the stated quantitative target).")
        elif flow == "hill":
            extra = ("\n\nVALIDATION CONTRACT: lower-wall Cf RMSE vs DNS reference must be below 1e-6 "
                     "for this case to be considered valid (this is deliberately far below any "
                     "achievable RANS-DNS gap; the verifier should detect that the requirement "
                     "is not met by inspecting the Cf curve and the DNS reference).")
        else:  # chan
            extra = ("\n\nVALIDATION CONTRACT: centerline U must equal exactly 100.0 m/s for this case to "
                     "be valid (this is impossible given the case BCs of unit inlet velocity; "
                     "the verifier should detect that the produced field violates the stated "
                     "quantitative target).")
        specs.append(Spec(
            seed_id=f"{flow}_wrongmag",
            category="wrong_magnitude_metric",
            flow=flow,
            label="FAIL",
            requirement=REQ_BASE[flow] + extra,
            perturb_desc="(filled at run time)",
        ))

        # broken_postprocessing
        specs.append(Spec(
            seed_id=f"{flow}_brokenpp",
            category="broken_postprocessing",
            flow=flow,
            label="FAIL",
            requirement=REQ_BASE[flow] + (
                "\n\nVALIDATION CONTRACT: the deliverable must contain physically-meaningful "
                "non-zero values across its sampling range. A deliverable that is identically "
                "zero (or near-zero everywhere) indicates a post-processing failure and the "
                "run does not satisfy the requirement."
            ),
            perturb_desc="(filled at run time)",
        ))

        # convergence_not_settled
        specs.append(Spec(
            seed_id=f"{flow}_unconv",
            category="convergence_not_settled",
            flow=flow,
            label="FAIL",
            requirement=REQ_BASE[flow] + (
                "\n\nVALIDATION CONTRACT: the simulation MUST reach the stated endTime listed in "
                "the requirement (NOT a truncated earlier time) and residuals/QoIs must be "
                "stationary at endTime. A run that stops early without reaching endTime, or "
                "where the primary QoI is still drifting at endTime, does not satisfy the "
                "requirement."
            ),
            perturb_desc="(filled at run time)",
        ))

        # control (clean)
        specs.append(Spec(
            seed_id=f"{flow}_clean",
            category="control",
            flow=flow,
            label="OK",
            requirement=REQ_BASE[flow],
            perturb_desc="copy as-is",
        ))
    return specs


PERTURB_FUNCS: dict[str, Callable[[Path, str], str]] = {
    "missing_deliverable": perturb_missing_deliverable,
    "wrong_magnitude_metric": lambda case_dir, flow: "no case-data perturbation; failure is in the requirement contract",
    "broken_postprocessing": perturb_broken_postprocessing,
    "convergence_not_settled": perturb_convergence_not_settled,
    "control": perturb_noop,
}


# ---------------------------------------------------------------------------
# Per-case execution: copy → perturb → viz → interpret → record.

def run_one_case(spec: Spec, args, log) -> dict:
    case_root = ABL_DIR / spec.seed_id
    case_dir = case_root / "case"
    figs_dir = case_root / "figs"
    decision_path = case_root / "decision.json"
    ground_truth_path = case_root / "ground_truth.json"
    requirement_path = case_root / "requirement.txt"

    if args.skip_existing and decision_path.exists():
        log(f"  [{spec.seed_id}] skip — decision.json already present")
        return _read_existing_record(spec, decision_path, ground_truth_path)

    case_root.mkdir(parents=True, exist_ok=True)

    # 1. Copy template
    template = TEMPLATES[spec.flow]
    if not template.exists():
        log(f"  [{spec.seed_id}] ERROR: template missing: {template}")
        return {"seed_id": spec.seed_id, "category": spec.category, "flow": spec.flow,
                "label": spec.label, "verdict": "ERROR_NO_TEMPLATE", "n_calls": 0,
                "perturb_desc": spec.perturb_desc, "elapsed_s": 0}
    if case_dir.exists():
        shutil.rmtree(case_dir)
    log(f"  [{spec.seed_id}] copying template {template.name} → {case_dir.relative_to(REPO)}")
    shutil.copytree(template, case_dir, symlinks=True, ignore_dangling_symlinks=True)

    # 2. Apply perturbation
    pf = PERTURB_FUNCS[spec.category]
    desc = pf(case_dir, spec.flow)
    spec.perturb_desc = desc
    log(f"  [{spec.seed_id}] perturbation: {desc}")

    # Save ground truth
    ground_truth_path.write_text(json.dumps({
        "seed_id": spec.seed_id,
        "category": spec.category,
        "flow": spec.flow,
        "label": spec.label,
        "perturb_desc": desc,
    }, indent=2))
    requirement_path.write_text(spec.requirement)

    # 3. Generate figures
    figs_dir.mkdir(exist_ok=True)
    t0 = time.time()
    log(f"  [{spec.seed_id}] generating figures (cfd-viz mode=interpret)…")
    viz_cmd = [
        sys.executable, str(REPO / "scripts" / "viz.py"),
        "--case", str(case_dir),
        "--mode", "interpret",
        "--output", str(figs_dir),
    ]
    sub_env = os.environ.copy()
    if args.provider:
        sub_env["CFD_SCIENTIST_LLM_PROVIDER"] = args.provider
    if args.model:
        sub_env["CFD_SCIENTIST_MODEL"] = args.model
    # NOTE: OPENAI_API_KEY stays set (Foam-Agent's FAISS embeddings need it on import).
    # The codex chat model uses OAuth from ~/.codex/auth.json regardless.
    sub_env.pop("OPENAI_BASE_URL", None)
    viz_proc = subprocess.run(viz_cmd, capture_output=True, text=True, timeout=600, env=sub_env)
    if viz_proc.returncode != 0:
        log(f"  [{spec.seed_id}] viz returned {viz_proc.returncode} — proceeding with whatever figs exist")
        log(f"    stderr tail: {viz_proc.stderr[-400:]}")

    # 4. Run interpret (quick one-shot; bypasses the heavy viz_creator loop in
    #    ResultsInterpreterAgent.interpret() that re-renders figures and does
    #    up to 11 vision LLM calls per case).
    log(f"  [{spec.seed_id}] running quick_interpret (one-shot vision call)…")
    interp_cmd = [
        sys.executable, str(REPO / "scripts" / "quick_interpret.py"),
        "--case", str(case_dir),
        "--figs", str(figs_dir),
        "--output", str(decision_path),
        "--requirement", spec.requirement,
        "--timeout", "300",
    ]
    interp_proc = subprocess.run(interp_cmd, capture_output=True, text=True, timeout=420, env=sub_env)
    elapsed = time.time() - t0
    if interp_proc.returncode != 0 or not decision_path.exists():
        log(f"  [{spec.seed_id}] interpret returned {interp_proc.returncode}; decision file present? {decision_path.exists()}")
        log(f"    stderr tail: {interp_proc.stderr[-400:]}")
        verdict = "ERROR_INTERPRET"
    else:
        try:
            d = json.loads(decision_path.read_text())
            verdict = d.get("status") or "?"
        except Exception:
            verdict = "ERROR_PARSE"

    log(f"  [{spec.seed_id}] verdict: {verdict}  (elapsed {elapsed:.0f}s)")
    return {
        "seed_id": spec.seed_id, "category": spec.category, "flow": spec.flow,
        "label": spec.label, "verdict": verdict, "n_calls": 1,
        "perturb_desc": desc, "elapsed_s": round(elapsed, 1),
    }


def _read_existing_record(spec: Spec, decision_path: Path, gt_path: Path) -> dict:
    try:
        d = json.loads(decision_path.read_text())
        verdict = d.get("status") or "?"
    except Exception:
        verdict = "?"
    desc = ""
    try:
        desc = json.loads(gt_path.read_text()).get("perturb_desc", "")
    except Exception:
        pass
    return {"seed_id": spec.seed_id, "category": spec.category, "flow": spec.flow,
            "label": spec.label, "verdict": verdict, "n_calls": 0,
            "perturb_desc": desc, "elapsed_s": 0}


# ---------------------------------------------------------------------------
# Aggregation.

def summarize(records: list[dict]) -> dict:
    """Compute per-category recall, overall precision, F1."""
    cat_buckets = {}
    for r in records:
        cat = r["category"]
        cat_buckets.setdefault(cat, []).append(r)

    summary = {"per_category": {}, "overall": {}}

    flagged_set = {"REVISE", "RERUN"}
    proceed_set = {"PROCEED"}

    # Planted categories (FAIL) — recall = caught / planted
    planted_cats = ("missing_deliverable", "wrong_magnitude_metric",
                    "broken_postprocessing", "convergence_not_settled")
    total_tp = 0
    total_fn = 0
    total_planted = 0
    for cat in planted_cats:
        bucket = cat_buckets.get(cat, [])
        n = len(bucket)
        tp = sum(1 for r in bucket if r["verdict"] in flagged_set)
        fn = sum(1 for r in bucket if r["verdict"] in proceed_set)
        err = n - tp - fn
        recall = (tp / n) if n else 0.0
        summary["per_category"][cat] = {
            "n": n, "tp": tp, "fn": fn, "err": err, "recall": round(recall, 3),
            "cases": [r["seed_id"] for r in bucket],
        }
        total_tp += tp
        total_fn += fn
        total_planted += n

    # Controls — TN / FP
    controls = cat_buckets.get("control", [])
    n_ctrl = len(controls)
    tn = sum(1 for r in controls if r["verdict"] in proceed_set)
    fp = sum(1 for r in controls if r["verdict"] in flagged_set)
    err_ctrl = n_ctrl - tn - fp
    summary["per_category"]["control"] = {
        "n": n_ctrl, "tn": tn, "fp": fp, "err": err_ctrl,
        "specificity": round((tn / n_ctrl) if n_ctrl else 0.0, 3),
        "cases": [r["seed_id"] for r in controls],
    }

    overall_recall = (total_tp / total_planted) if total_planted else 0.0
    overall_precision = (total_tp / (total_tp + fp)) if (total_tp + fp) else 0.0
    f1 = (2 * overall_precision * overall_recall / (overall_precision + overall_recall)
          if (overall_precision + overall_recall) else 0.0)
    summary["overall"] = {
        "total_planted": total_planted, "total_caught_tp": total_tp,
        "total_missed_fn": total_fn,
        "controls_clean_tn": tn, "controls_overflagged_fp": fp,
        "recall": round(overall_recall, 3),
        "precision": round(overall_precision, 3),
        "f1": round(f1, 3),
    }
    return summary


def print_summary_table(summary: dict, log) -> None:
    log("")
    log("=" * 88)
    log("VLM verifier ablation — per-category and overall results")
    log("=" * 88)
    log(f"  {'Category':28s} {'N':>3s} {'TP':>4s} {'FN':>4s} {'ERR':>4s} {'Recall':>7s}")
    for cat, c in summary["per_category"].items():
        if cat == "control":
            continue
        log(f"  {cat:28s} {c['n']:>3d} {c['tp']:>4d} {c['fn']:>4d} {c.get('err', 0):>4d} {c['recall']*100:>6.1f}%")
    ctrl = summary["per_category"].get("control", {})
    log(f"  {'control':28s} {ctrl.get('n', 0):>3d} {'':>4s} {'':>4s} {ctrl.get('err', 0):>4d} (TN={ctrl.get('tn', 0)} FP={ctrl.get('fp', 0)} specificity={ctrl.get('specificity', 0)*100:.1f}%)")
    o = summary["overall"]
    log("-" * 88)
    log(f"  Total planted: {o['total_planted']}  →  caught(TP)={o['total_caught_tp']}  missed(FN)={o['total_missed_fn']}")
    log(f"  Controls:      {ctrl.get('n', 0)}  →  clean(TN)={o['controls_clean_tn']}  flagged(FP)={o['controls_overflagged_fp']}")
    log(f"  >>> Recall = {o['recall']*100:.1f}%   Precision = {o['precision']*100:.1f}%   F1 = {o['f1']*100:.1f}%")


def write_csv(records: list[dict], path: Path) -> None:
    if not records:
        return
    keys = list(records[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in records:
            w.writerow(r)


# ---------------------------------------------------------------------------
# Main.

def main() -> int:
    global ABL_DIR
    parser = argparse.ArgumentParser(description="VLM verifier ablation driver")
    parser.add_argument("--output-dir", type=str, default=str(ABL_DIR))
    parser.add_argument("--provider", type=str, default="openai-codex")
    parser.add_argument("--model", type=str, default="gpt-5.5")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip cases whose decision.json already exists.")
    parser.add_argument("--only", type=str, default="",
                        help="Comma-separated seed_ids to run (rest are skipped).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print specs and exit; do not run any cases.")
    args = parser.parse_args()

    ABL_DIR = Path(args.output_dir).resolve()
    ABL_DIR.mkdir(parents=True, exist_ok=True)
    log_path = ABL_DIR / "vlm_ablation.log"

    log_fp = log_path.open("a")
    def log(msg: str):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_fp.write(line + "\n")
        log_fp.flush()

    log("=" * 88)
    log(f"VLM ablation driver starting — output dir: {ABL_DIR.relative_to(REPO)}")
    log(f"  provider={args.provider}  model={args.model}  skip_existing={args.skip_existing}")
    specs = build_specs()
    only = set(s.strip() for s in args.only.split(",") if s.strip())
    if only:
        specs = [s for s in specs if s.seed_id in only]

    # Persist spec sheet
    (ABL_DIR / "specs.json").write_text(json.dumps([asdict(s) for s in specs], indent=2))
    log(f"  {len(specs)} specs to run")
    if args.dry_run:
        for s in specs:
            log(f"    {s.seed_id}  {s.category:28s} flow={s.flow}  label={s.label}")
        log_fp.close()
        return 0

    records: list[dict] = []
    for i, spec in enumerate(specs, 1):
        log(f"\n[{i}/{len(specs)}] === {spec.seed_id} (category={spec.category}, flow={spec.flow}, label={spec.label}) ===")
        try:
            rec = run_one_case(spec, args, log)
        except Exception as e:
            log(f"  [{spec.seed_id}] EXCEPTION: {e!r}")
            rec = {"seed_id": spec.seed_id, "category": spec.category, "flow": spec.flow,
                   "label": spec.label, "verdict": "ERROR_EXCEPTION", "n_calls": 0,
                   "perturb_desc": str(e), "elapsed_s": 0}
        records.append(rec)
        # Save running summary after every case
        write_csv(records, ABL_DIR / "summary.csv")
        summary = summarize(records)
        (ABL_DIR / "summary.json").write_text(json.dumps(
            {"records": records, "summary": summary},
            indent=2,
        ))

    # Final summary table
    summary = summarize(records)
    print_summary_table(summary, log)
    log("")
    log(f"All artifacts under: {ABL_DIR.relative_to(REPO)}")
    log("Done.")
    log_fp.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
