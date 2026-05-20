#!/usr/bin/env python3
"""skill_bootstrap.py — state-machine driver for skill-mode CFD Scientist runs.

Slice 15. This script does **no CFD work** — that remains the job of the
per-stage skills under cfd-skills/. Bootstrap is the conductor:

1. `--init`     writes state.json + timeline.json + checkpoints/routing_done.json
                and prints the exact next command (preflight + which skill to
                invoke + which artifact must result).
2. `--advance`  validates the artifact the previous stage was supposed to produce
                (lit.json, hypotheses.json, paper/main.pdf, ...), enforces
                stage-specific shape checks (e.g. literature provenance), writes
                <stage>_done.json, advances state.json, prints the next command.
3. `--status`   prints what state.json says + what bootstrap would say next.

The agent reads cfd-skills/cfd-<stage>/SKILL.md and does the work. Bootstrap
hands it the next instruction. The agent cannot skip stages because each
--advance refuses to write _done.json unless the prior artifact is on disk
with the right shape.

Bootstrap NEVER edits cases/, never compiles code, never runs solvers, never
hits HTTP. State/checkpoint management only.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Per-route stage sequence. Each tuple:
#   (stage_name, artifact_relpath, skill_doc, artifact_kind)
# artifact_kind: "file" | "dir" | "dir_has_subdir" | "file_or_dir"
#
# Mirrors skills/cfd-orchestrator/SKILL.md "Required checkpoints per route"
# and the artifact contract documented in each cfd-skills/cfd-<stage>/SKILL.md.
ROUTES: Dict[str, List[Tuple[str, str, str, str]]] = {
    "research": [
        ("literature",                  "lit.json",                       "cfd-skills/cfd-literature/SKILL.md",   "file"),
        ("baseline_setup",              "baseline_metrics.json",          "cfd-skills/cfd-pipeline/SKILL.md (Step 5)", "file"),
        ("metric_setup",                "metric_definitions.json",        "cfd-skills/cfd-pipeline/SKILL.md (Step 6)", "file"),
        ("hypothesis",                  "hypotheses.json",                "cfd-skills/cfd-hypothesis/SKILL.md",   "file"),
        ("requirements",                "requirements.json",              "cfd-skills/cfd-requirements/SKILL.md", "file"),
        ("mesh_gate",                   "selected_mesh_spec.json",        "cfd-skills/cfd-mesh-gate/SKILL.md",    "file"),
        ("experiments",                 "cases",                          "cfd-skills/cfd-experiment/SKILL.md (one case_NNN per requirements.json entry; cfd-viz mode=interpret + cfd-interpret per case)", "dir_has_case"),
        ("analysis",                    "analysis.json",                  "cfd-skills/cfd-analyze/SKILL.md",      "file"),
        ("paper_experiment_plan",       "paper_experiment_plan.json",     "cfd-skills/cfd-paper/SKILL.md Stage 0", "file"),
        ("cross_experiment_analysis",   "cross_experiment_analysis",      "cfd-skills/cfd-cross-analyze/SKILL.md", "dir_has_png"),
        ("paper_review",                "paper/main.pdf",                 "cfd-skills/cfd-paper/SKILL.md (Stage 0.5 lit-enrichment + iterate)", "file_min_50kb"),
    ],
    "code_mod": [
        ("literature",                  "lit.json",                       "cfd-skills/cfd-literature/SKILL.md",   "file"),
        ("baseline_setup",              "baseline_metrics.json",          "cfd-skills/cfd-pipeline/SKILL.md (Step 5)", "file"),
        ("metric_setup",                "metric_definitions.json",        "cfd-skills/cfd-pipeline/SKILL.md (Step 6)", "file"),
        ("hypothesis",                  "hypotheses.json",                "cfd-skills/cfd-hypothesis/SKILL.md",   "file"),
        ("requirements",                "requirements.json",              "cfd-skills/cfd-requirements/SKILL.md", "file"),
        ("code_mod",                    "code_mod/build_result.json",     "cfd-skills/cfd-code-modify/SKILL.md",  "file"),
        ("mesh_gate",                   "selected_mesh_spec.json",        "cfd-skills/cfd-mesh-gate/SKILL.md",    "file"),
        ("experiments",                 "cases",                          "cfd-skills/cfd-experiment/SKILL.md",   "dir_has_case"),
        ("analysis",                    "analysis.json",                  "cfd-skills/cfd-analyze/SKILL.md",      "file"),
        ("paper_experiment_plan",       "paper_experiment_plan.json",     "cfd-skills/cfd-paper/SKILL.md Stage 0", "file"),
        ("cross_experiment_analysis",   "cross_experiment_analysis",      "cfd-skills/cfd-cross-analyze/SKILL.md", "dir_has_png"),
        ("paper_review",                "paper/main.pdf",                 "cfd-skills/cfd-paper/SKILL.md (Stage 0.5 lit-enrichment + iterate)", "file_min_50kb"),
    ],
    "mesh_gate": [
        ("mesh_gate",                   "selected_mesh_spec.json",        "cfd-skills/cfd-mesh-gate/SKILL.md",    "file"),
    ],
    "analysis_only": [
        ("analysis",                    "analysis.json",                  "cfd-skills/cfd-analyze/SKILL.md",      "file"),
    ],
    "paper_only": [
        ("paper_experiment_plan",       "paper_experiment_plan.json",     "cfd-skills/cfd-paper/SKILL.md Stage 0", "file"),
        ("cross_experiment_analysis",   "cross_experiment_analysis",      "cfd-skills/cfd-cross-analyze/SKILL.md", "dir_has_png"),
        ("paper_review",                "paper/main.pdf",                 "cfd-skills/cfd-paper/SKILL.md (Stage 0.5 lit-enrichment + iterate)", "file_min_50kb"),
    ],
    "open_discovery": [
        ("literature",                  "lit.json",                       "cfd-skills/cfd-literature/SKILL.md",   "file"),
        ("baseline_setup",              "baseline_metrics.json",          "cfd-skills/cfd-pipeline/SKILL.md (Step 5)", "file"),
        ("metric_setup",                "metric_definitions.json",        "cfd-skills/cfd-pipeline/SKILL.md (Step 6)", "file"),
        ("mesh_gate",                   "selected_mesh_spec.json",        "cfd-skills/cfd-mesh-gate/SKILL.md",    "file"),
        ("open_ended_discovery",        "open_ended_discovery",           "cfd-skills/cfd-open-discovery/SKILL.md", "dir"),
        ("analysis",                    "analysis.json",                  "cfd-skills/cfd-analyze/SKILL.md",      "file"),
        ("paper_experiment_plan",       "paper_experiment_plan.json",     "cfd-skills/cfd-paper/SKILL.md Stage 0", "file"),
        ("cross_experiment_analysis",   "cross_experiment_analysis",      "cfd-skills/cfd-cross-analyze/SKILL.md", "dir_has_png"),
        ("paper_review",                "paper/main.pdf",                 "cfd-skills/cfd-paper/SKILL.md (Stage 0.5 lit-enrichment + iterate)", "file_min_50kb"),
    ],
}


def _iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z") or time.strftime("%Y-%m-%dT%H:%M:%S")


def _read_json(p: Path) -> Any:
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _write_json_atomic(p: Path, data: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(p)


def _append_timeline(out_dir: Path, event: Dict[str, Any]) -> None:
    tl = out_dir / "timeline.json"
    events = _read_json(tl) or []
    if not isinstance(events, list):
        events = []
    events.append(event)
    _write_json_atomic(tl, events)


def _validate_artifact(out_dir: Path, artifact_rel: str, kind: str) -> Tuple[bool, str]:
    """Return (ok, message). False ok => skill did not produce expected artifact."""
    p = out_dir / artifact_rel
    if kind == "file":
        if not p.is_file():
            return False, f"missing artifact: {artifact_rel} (file)"
        return True, f"{artifact_rel} (file, {p.stat().st_size} B)"
    if kind == "file_min_50kb":
        if not p.is_file():
            return False, f"missing artifact: {artifact_rel} (file)"
        if p.stat().st_size < 50_000:
            return False, f"{artifact_rel} present but <50KB ({p.stat().st_size} B) — likely a stub PDF"
        return True, f"{artifact_rel} ({p.stat().st_size} B)"
    if kind == "dir":
        if not p.is_dir():
            return False, f"missing artifact: {artifact_rel}/ (directory)"
        return True, f"{artifact_rel}/ (dir)"
    if kind == "dir_has_png":
        if not p.is_dir():
            return False, f"missing artifact: {artifact_rel}/ (directory)"
        pngs = list(p.glob("*.png"))
        if len(pngs) < 1:
            return False, f"{artifact_rel}/ exists but contains 0 PNGs"
        return True, f"{artifact_rel}/ ({len(pngs)} PNG(s))"
    if kind == "dir_has_case":
        if not p.is_dir():
            return False, f"missing artifact: {artifact_rel}/ (directory)"
        case_dirs = [d for d in p.iterdir() if d.is_dir() and d.name.startswith("case_")]
        if not case_dirs:
            return False, f"{artifact_rel}/ exists but contains no case_NNN subdirs"
        # Per-case sanity: each case_NNN must have run_result.json + decision.json + figs/*.png
        missing = []
        for cd in case_dirs:
            if not (cd / "run_result.json").is_file():
                missing.append(f"{cd.name}: missing run_result.json")
            if not (cd / "decision.json").is_file():
                missing.append(f"{cd.name}: missing decision.json")
            if not (cd / "figs").is_dir() or not list((cd / "figs").glob("*.png")):
                missing.append(f"{cd.name}: figs/ empty (cfd-viz mode=interpret was skipped)")
        if missing:
            return False, f"{artifact_rel}/ has {len(case_dirs)} case dir(s) but: " + "; ".join(missing)
        return True, f"{artifact_rel}/ ({len(case_dirs)} case dir(s), all complete)"
    return False, f"unknown artifact kind: {kind}"


def _validate_literature_provenance(out_dir: Path) -> Tuple[bool, str]:
    """Slice 14 rule: lit.json must contain at least one record with
    source=='semanticscholar', or be [] paired with a timeline event."""
    lit = out_dir / "lit.json"
    data = _read_json(lit)
    if data is None:
        return False, "lit.json missing or unreadable"
    if isinstance(data, dict):
        recs = data.get("records") or data.get("papers") or data.get("results") or []
    else:
        recs = data
    if not isinstance(recs, list):
        return False, "lit.json has unexpected shape (expected list)"
    if len(recs) == 0:
        events = _read_json(out_dir / "timeline.json") or []
        if isinstance(events, list) and any(
            isinstance(e, dict) and e.get("event") in
                ("literature_empty_result", "literature_skipped_existing")
            for e in events
        ):
            return True, "lit.json is [] with documented empty-result timeline event"
        return False, ("lit.json is [] but no 'literature_empty_result' timeline event "
                       "was emitted — the agent must surface empty results explicitly")
    if not all(isinstance(r, dict) for r in recs):
        return False, "lit.json contains non-dict records"
    sources = [r.get("source") for r in recs]
    if not any(s == "semanticscholar" for s in sources):
        return False, (f"lit.json has {len(recs)} records but none have source=='semanticscholar'. "
                       f"Semantic Scholar is mandatory (cfd-literature Hard rule #1). "
                       f"Sources seen: {sorted({s for s in sources if s})!r}")
    n_s2 = sum(1 for s in sources if s == "semanticscholar")
    return True, f"lit.json: {len(recs)} records ({n_s2} from semanticscholar)"


def _stage_specific_validators(stage: str, out_dir: Path) -> Optional[Tuple[bool, str]]:
    """Hook for stage-specific extra validation beyond artifact existence."""
    if stage == "literature":
        return _validate_literature_provenance(out_dir)
    return None


def _print_next_command(out_dir: Path, mode: str, next_stage_idx: int, *, file=sys.stdout) -> None:
    seq = ROUTES[mode]
    if next_stage_idx >= len(seq):
        print(file=file)
        print("=" * 72, file=file)
        print("ALL STAGES COMPLETE.", file=file)
        print("=" * 72, file=file)
        print(f"  Run final audit:", file=file)
        print(f"    python scripts/stage_gate_audit.py --out-dir {out_dir}", file=file)
        print(f"  Expected: rc=0, writes {out_dir}/audit_passed.json", file=file)
        return
    stage, artifact, skill_doc, _kind = seq[next_stage_idx]
    print(file=file)
    print("=" * 72, file=file)
    print(f"NEXT STAGE  {next_stage_idx + 1}/{len(seq)}  →  {stage}", file=file)
    print("=" * 72, file=file)
    print(file=file)
    print(f"  1. Verify predecessors:", file=file)
    print(f"       python scripts/stage_gate_audit.py --out-dir {out_dir} \\", file=file)
    print(f"           --mode preflight --target-stage {stage}", file=file)
    print(f"     (must return rc=0 before invoking the skill)", file=file)
    print(file=file)
    print(f"  2. Invoke skill:  {skill_doc}", file=file)
    print(file=file)
    print(f"  3. Required artifact on disk when done:  {out_dir}/{artifact}", file=file)
    if stage == "literature":
        print(f"       — must contain at least one record with \"source\":\"semanticscholar\"", file=file)
        print(f"         (or be [] paired with a literature_empty_result timeline event)", file=file)
    if stage == "experiments":
        print(f"       — each case_NNN/ must have run_result.json + decision.json + figs/*.png", file=file)
        print(f"         (cfd-viz mode=interpret is non-negotiable per case)", file=file)
    if stage == "paper_review":
        print(f"       — also produce paper_audit.json + review.json with figure_physics_ok set", file=file)
        print(f"       — Stage 0.5 of cfd-paper re-queries Semantic Scholar for the bibliography", file=file)
    print(file=file)
    print(f"  4. Advance state:", file=file)
    print(f"       python scripts/skill_bootstrap.py --out-dir {out_dir} --advance", file=file)
    print(file=file)


def cmd_init(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir).resolve()
    if args.mode not in ROUTES:
        print(f"ERROR: unknown --mode {args.mode!r}. Valid: {sorted(ROUTES.keys())}", file=sys.stderr)
        return 2
    if (out_dir / "state.json").is_file() and not args.force:
        print(f"ERROR: {out_dir}/state.json already exists. Use --force to re-init, or --advance to continue.", file=sys.stderr)
        return 3
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(exist_ok=True)

    seq = ROUTES[args.mode]
    first_stage = seq[0][0]
    state = {
        "topic": args.topic,
        "mode": args.mode,
        "open_discovery": args.mode == "open_discovery",
        "status": "running",
        "run_dir": str(out_dir),
        "timeline_path": str(out_dir / "timeline.json"),
        "provider": args.provider,
        "model": args.model,
        "resume_from": "",
        "clarifications": {},
        "current_stage": first_stage,
        "current_stage_phase": "starting",
        "current_stage_index": 0,
        "current_stage_total": len(seq),
        "current_stage_progress": f"0/{len(seq)}",
        "next_stage": seq[1][0] if len(seq) > 1 else "finish",
        "current_stage_details": {
            "starter_dir": args.starter_dir or "",
        },
        "checkpoint": "routing",
        "checkpoint_extra": {},
        "failed_stage": "",
        "last_error": "",
        "starter_seed_case_dir": args.starter_dir or "",
    }
    _write_json_atomic(out_dir / "state.json", state)
    _write_json_atomic(out_dir / "timeline.json", [])
    _write_json_atomic(out_dir / "checkpoints" / "routing_done.json", {
        "stage": "routing",
        "event": "complete",
        "ts": _iso_now(),
        "mode": args.mode,
        "checkpoint": "routing_done",
        "via": "skill_bootstrap.py",
    })
    _append_timeline(out_dir, {
        "stage": "routing", "event": "complete", "ts": _iso_now(),
        "mode": args.mode, "topic": args.topic[:120],
    })
    print(f"INITIALIZED  out-dir = {out_dir}")
    print(f"  topic:  {args.topic[:80]}...")
    print(f"  mode:   {args.mode}  →  route has {len(seq)} stage(s)")
    if args.starter_dir:
        print(f"  starter: {args.starter_dir} (READ-ONLY)")
    _print_next_command(out_dir, args.mode, 0)
    return 0


def cmd_advance(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir).resolve()
    state_path = out_dir / "state.json"
    if not state_path.is_file():
        print(f"ERROR: {state_path} missing. Run --init first.", file=sys.stderr)
        return 2
    state = _read_json(state_path) or {}
    mode = state.get("mode")
    if mode not in ROUTES:
        print(f"ERROR: state.json#mode = {mode!r} not in {sorted(ROUTES.keys())}", file=sys.stderr)
        return 2
    seq = ROUTES[mode]
    idx = state.get("current_stage_index", 0)
    if idx >= len(seq):
        print(f"ALREADY COMPLETE: all {len(seq)} stages done for mode={mode}")
        _print_next_command(out_dir, mode, idx)
        return 0
    stage, artifact, skill_doc, kind = seq[idx]

    print(f"VALIDATING  stage {idx+1}/{len(seq)}  →  {stage}")
    print(f"  expected artifact: {out_dir}/{artifact}  (kind={kind})")
    ok, msg = _validate_artifact(out_dir, artifact, kind)
    print(f"  artifact check: {'OK' if ok else 'FAIL'} — {msg}")
    if not ok:
        print(f"\nCANNOT ADVANCE. The skill at {skill_doc} did not produce {artifact}.", file=sys.stderr)
        print(f"Re-invoke that skill, ensure the artifact is on disk, then re-run --advance.", file=sys.stderr)
        return 4

    extra = _stage_specific_validators(stage, out_dir)
    if extra is not None:
        ex_ok, ex_msg = extra
        print(f"  shape check:    {'OK' if ex_ok else 'FAIL'} — {ex_msg}")
        if not ex_ok:
            print(f"\nCANNOT ADVANCE. Artifact present but failed shape validation.", file=sys.stderr)
            return 4

    cp_name = f"{stage}_done"
    cp_path = out_dir / "checkpoints" / f"{cp_name}.json"
    _write_json_atomic(cp_path, {
        "stage": stage,
        "event": "complete",
        "ts": _iso_now(),
        "checkpoint": cp_name,
        "artifact": artifact,
        "validation": msg,
        "via": "skill_bootstrap.py --advance",
    })
    _append_timeline(out_dir, {
        "stage": stage, "event": "complete", "ts": _iso_now(),
        "checkpoint": cp_name, "artifact": artifact,
    })

    next_idx = idx + 1
    state["current_stage_index"] = next_idx
    state["current_stage_progress"] = f"{next_idx}/{len(seq)}"
    state["checkpoint"] = stage
    state["current_stage_phase"] = "done"
    if next_idx < len(seq):
        state["current_stage"] = seq[next_idx][0]
        state["next_stage"] = seq[next_idx + 1][0] if next_idx + 1 < len(seq) else "finish"
        state["current_stage_phase"] = "starting"
    else:
        state["current_stage"] = "finish"
        state["next_stage"] = "finish"
        state["status"] = "done"  # tentative — final audit_passed.json is the ground truth
    _write_json_atomic(state_path, state)

    print(f"  CHECKPOINT WRITTEN: {cp_path}")
    _print_next_command(out_dir, mode, next_idx)
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir).resolve()
    state_path = out_dir / "state.json"
    if not state_path.is_file():
        print(f"NOT INITIALIZED — {state_path} missing. Run --init first.", file=sys.stderr)
        return 2
    state = _read_json(state_path) or {}
    mode = state.get("mode", "?")
    seq = ROUTES.get(mode, [])
    idx = state.get("current_stage_index", 0)
    print(f"STATUS  out-dir = {out_dir}")
    print(f"  mode:    {mode}")
    print(f"  topic:   {state.get('topic','?')[:80]}")
    print(f"  stage:   {state.get('current_stage','?')} ({idx}/{len(seq)})")
    print(f"  status:  {state.get('status','?')}")
    print()
    print("  CHECKPOINTS:")
    cp_dir = out_dir / "checkpoints"
    for s, _a, _doc, _k in seq:
        cp = cp_dir / f"{s}_done.json"
        mark = "✓" if cp.is_file() else "·"
        print(f"    {mark} {s}_done")
    _print_next_command(out_dir, mode, idx)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description="Skill-mode state-machine driver. Does no CFD work — hands "
                    "the agent the next skill to invoke and validates artifacts.",
    )
    p.add_argument("--out-dir", required=True, help="run directory")
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--init",     action="store_true", help="initialize state.json + first checkpoint, print first command")
    grp.add_argument("--advance",  action="store_true", help="validate current stage's artifact, write _done.json, print next command")
    grp.add_argument("--status",   action="store_true", help="print current state + next command")
    # --init args
    p.add_argument("--topic",       help="(--init) user research topic verbatim")
    p.add_argument("--mode",        choices=sorted(ROUTES.keys()),
                   help="(--init) one of: research, code_mod, mesh_gate, analysis_only, paper_only, open_discovery")
    p.add_argument("--starter-dir", help="(--init) optional starter directory path (READ-ONLY)")
    p.add_argument("--provider",    default="openai-codex", help="(--init) LLM provider")
    p.add_argument("--model",       default="gpt-5.5",     help="(--init) LLM model")
    p.add_argument("--force",       action="store_true", help="(--init) overwrite existing state.json")
    args = p.parse_args()

    if args.init:
        if not args.topic or not args.mode:
            print("ERROR: --init requires --topic and --mode", file=sys.stderr)
            return 2
        return cmd_init(args)
    if args.advance:
        return cmd_advance(args)
    if args.status:
        return cmd_status(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
