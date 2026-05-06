#!/usr/bin/env python3
"""Emit a short human-readable milestone snapshot of a run's state.

Called from stages at natural checkpoints (e.g. after OED code_mod
iterations complete, after mesh-gate, after experiments stage). Produces
`<run_dir>/milestones/<tag>_<timestamp>.md` with a LLM-free linear summary
plus (optionally) a short LLM-written narrative.

Pure Python by default — no LLM required. Generic across all code-mod
modes and topics.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _read_json(p: Path, default: Any = None) -> Any:
    try:
        if p.is_file():
            return json.loads(p.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        pass
    return default


def _oed_summary(run_dir: Path) -> List[str]:
    oed = run_dir / "open_ended_discovery"
    hist = _read_json(oed / "history.json", [])
    if not isinstance(hist, list) or not hist:
        return []
    lines: List[str] = []
    budget = 0
    proceed_cases: List[Dict[str, Any]] = []
    for h in hist:
        if not isinstance(h, dict):
            continue
        cost = {"code_mod": 2, "experiment": 1, "python_script": 0}.get(h.get("action_type"), 0)
        budget += cost
        if h.get("status") == "PROCEED" and h.get("action_type") != "python_script":
            proceed_cases.append(h)
    lines.append(f"- Iterations: {len(hist)}")
    lines.append(f"- Budget used: {budget}")
    lines.append(f"- PROCEED cases: {len(proceed_cases)}")
    if proceed_cases:
        lines.append(f"- Proceeding variants:")
        for p in proceed_cases[:10]:
            name = p.get("compiled_model_name") or p.get("model_name_to_reuse") or "?"
            it = p.get("iteration")
            lines.append(f"    - iter {it}: {name}")
    return lines


def _compile_summary(run_dir: Path) -> List[str]:
    results = sorted(run_dir.rglob("code_mod_apply_result.json"))
    if not results:
        return []
    total, ok = 0, 0
    for r in results:
        d = _read_json(r, {})
        if d.get("compile_ok") is True:
            ok += 1
        total += 1
    return [f"- Compile attempts: {total}, succeeded: {ok}"]


def _case_summary(run_dir: Path) -> List[str]:
    cases = run_dir / "cases"
    if not cases.is_dir():
        return []
    counts = {"PROCEED": 0, "RERUN": 0, "REVISE": 0, "OTHER": 0}
    for c in cases.iterdir():
        if not c.is_dir():
            continue
        d = _read_json(c / "decision.json", {})
        s = (d.get("status") or "OTHER").upper()
        counts[s] = counts.get(s, 0) + 1
    total = sum(counts.values())
    if total == 0:
        return []
    return [f"- Cases: total={total}  " + "  ".join(f"{k}={v}" for k, v in counts.items() if v)]


def write_milestone(
    run_dir: Path,
    tag: str,
    *,
    note: str = "",
) -> Path:
    """Write a milestone snapshot markdown file. Returns the path written."""
    run_dir = run_dir.resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = run_dir / "milestones"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{tag}_{stamp}.md"

    state = _read_json(run_dir / "state.json", {}) or {}
    lines: List[str] = [
        f"# Milestone: {tag}",
        f"",
        f"- Timestamp (UTC): {stamp}",
        f"- Topic: {str(state.get('topic') or '')[:300]}",
        f"- Mode: {state.get('mode')}  provider={state.get('provider')}  model={state.get('model')}",
        f"- Stage: {state.get('current_stage')} / {state.get('current_stage_phase')}",
        f"- Checkpoint: {state.get('checkpoint')}",
    ]
    if state.get("failed_stage"):
        lines.append(f"- Last failed sub-step: {state.get('failed_stage')}")

    if note:
        lines.extend(["", "## Note", "", note])

    oed = _oed_summary(run_dir)
    if oed:
        lines.extend(["", "## Open-ended discovery", ""] + oed)

    comps = _compile_summary(run_dir)
    if comps:
        lines.extend(["", "## Compile results", ""] + comps)

    cases = _case_summary(run_dir)
    if cases:
        lines.extend(["", "## Experiment cases", ""] + cases)

    t = _read_json(run_dir / "llm_token_usage.json", {}) or {}
    tot = t.get("totals", {}) if isinstance(t, dict) else {}
    if tot:
        lines.extend(["", "## Token usage",
                      f"- calls: {tot.get('calls')}",
                      f"- input: {tot.get('input_tokens'):,}" if isinstance(tot.get('input_tokens'), int) else "",
                      f"- output: {tot.get('output_tokens'):,}" if isinstance(tot.get('output_tokens'), int) else "",
                      f"- total: {tot.get('total_tokens'):,}" if isinstance(tot.get('total_tokens'), int) else ""])

    out.write_text("\n".join(line for line in lines if line is not None) + "\n", encoding="utf-8")
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--tag", required=True, help="milestone tag (e.g. after_code_mod, after_mesh_gate)")
    p.add_argument("--note", default="", help="optional short note to include")
    args = p.parse_args()
    out = write_milestone(args.run_dir, args.tag, note=args.note)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
