#!/usr/bin/env python3
"""Linear human-readable summary of a cfd-scientist run directory.

Reads the scattered state/sidecar files (state.json, timeline.json,
history.json, decision.json per case, code_mod_apply_result.json per iter,
llm_token_usage.json) and prints a chronological narrative so you don't
have to grep a dozen files to understand what happened.

Pure Python, no LLM calls, generic across all mode/topic combinations.

Usage:
    python scripts/summarize_run.py <run_dir>
    python scripts/summarize_run.py runs/turbulence_model_change_codex
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _read_json(p: Path, default: Any = None) -> Any:
    try:
        if p.is_file():
            return json.loads(p.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return default
    return default


def _fmt_num(n: Optional[int]) -> str:
    return f"{n:,}" if isinstance(n, int) else "?"


def _fmt_ts(s: Optional[str]) -> str:
    if not s:
        return ""
    try:
        dt = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
        return dt.astimezone().strftime("%H:%M:%S")
    except Exception:
        return str(s)[:19]


def _summarize_timeline(run_dir: Path, limit: int = 60) -> List[str]:
    t = _read_json(run_dir / "timeline.json", {})
    events = t.get("events", []) if isinstance(t, dict) else []
    out: List[str] = []
    for e in events:
        if not isinstance(e, dict):
            continue
        ts = _fmt_ts(e.get("ts"))
        stage = e.get("stage", "?")
        event = e.get("event", "")
        extras = []
        if e.get("checkpoint"):
            extras.append(f"ckpt={e['checkpoint']}")
        if e.get("phase"):
            extras.append(f"phase={e['phase']}")
        if e.get("returncode") not in (None, 0):
            extras.append(f"rc={e['returncode']}")
        out.append(f"{ts}  {stage:<30} {event:<20} {' '.join(extras)}")
    return out[-limit:]


def _summarize_cases(run_dir: Path) -> List[str]:
    out: List[str] = []
    cases = run_dir / "cases"
    if not cases.is_dir():
        return out
    for case_dir in sorted(cases.iterdir()):
        if not case_dir.is_dir():
            continue
        decision = _read_json(case_dir / "decision.json", {})
        status = decision.get("status", "?")
        reason = (decision.get("reason", "") or "")[:200]
        out.append(f"  {case_dir.name:<12}  {status:<10}  {reason}")
    return out


def _summarize_oed(run_dir: Path) -> List[str]:
    oed = run_dir / "open_ended_discovery"
    if not oed.is_dir():
        return []
    hist = _read_json(oed / "history.json", [])
    if not isinstance(hist, list):
        return []
    out: List[str] = []
    budget = 0
    for h in hist:
        if not isinstance(h, dict):
            continue
        it = h.get("iteration", "?")
        atype = h.get("action_type", "?")
        status = h.get("status", "?")
        cost = {"code_mod": 2, "experiment": 1, "python_script": 0}.get(atype, 0)
        budget += cost
        name = h.get("compiled_model_name") or h.get("model_name_to_reuse") or h.get("model_description", "")[:60]
        extra = ""
        if h.get("metrics_summary"):
            extra = f" metrics={str(h['metrics_summary'])[:120]}"
        out.append(f"  iter {it:>3}  {atype:<14}  status={status:<10}  budget={budget}  {name}{extra}")
    return out


def _summarize_compiles(run_dir: Path) -> List[str]:
    """Scan for code_mod_apply_result.json files and report compile outcomes."""
    out: List[str] = []
    for p in sorted(run_dir.rglob("code_mod_apply_result.json")):
        d = _read_json(p, {})
        rel = str(p.parent.relative_to(run_dir))
        ok = d.get("compile_ok")
        cls = d.get("class_name", "?")
        attempts = len(d.get("compile_logs", []) or [])
        # Pull first real error line if failed
        err = ""
        if ok is False:
            logs = d.get("compile_logs", [])
            if logs:
                stderr = str(logs[-1].get("stderr", ""))
                for line in stderr.splitlines():
                    s = line.strip()
                    if "error:" in s or "No rule" in s or "fatal error" in s:
                        err = s[:200]
                        break
        status = "✓ compiled" if ok else ("✗ failed" if ok is False else "?")
        line = f"  {rel:<55}  {status:<12}  class={cls}  attempts={attempts}"
        if err:
            line += f"\n      └─ {err}"
        out.append(line)
    return out


def _summarize_tokens(run_dir: Path) -> List[str]:
    t = _read_json(run_dir / "llm_token_usage.json", {})
    tot = t.get("totals", {}) if isinstance(t, dict) else {}
    if not tot:
        return []
    out = [
        f"  calls:       {_fmt_num(tot.get('calls'))}",
        f"  input_tok:   {_fmt_num(tot.get('input_tokens'))}",
        f"  output_tok:  {_fmt_num(tot.get('output_tokens'))}",
        f"  total_tok:   {_fmt_num(tot.get('total_tokens'))}",
    ]
    by_model = tot.get("by_model", {})
    if isinstance(by_model, dict):
        for m, info in by_model.items():
            out.append(
                f"    - {m}: {_fmt_num(info.get('calls'))} calls, "
                f"{_fmt_num(info.get('total_tokens'))} tokens"
            )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--max-timeline", type=int, default=40)
    args = parser.parse_args()

    rd = args.run_dir.expanduser().resolve()
    if not rd.is_dir():
        print(f"ERROR: {rd} is not a directory")
        return 1

    state = _read_json(rd / "state.json", {})
    ce = state.get("checkpoint_extra", {}) or {}

    print(f"\n==== {rd.name} ====\n")
    print(f"Topic:  {(state.get('topic') or '')[:200]}")
    print(f"Mode:   {state.get('mode')}  provider={state.get('provider')}  model={state.get('model')}")
    print(f"Status: {state.get('status')}")
    print(f"Stage:  {state.get('current_stage')} / {state.get('current_stage_phase')}  (progress: {state.get('current_stage_progress')})")
    print(f"Ckpt:   {state.get('checkpoint')}")
    if state.get("failed_stage"):
        print(f"Failed: {state.get('failed_stage')}")
    if ce:
        if ce.get("budget_used") is not None:
            print(f"Budget: {ce.get('budget_used')} used, proceed={ce.get('proceed_cases')}")
        if ce.get("best_case_dir"):
            print(f"Best:   {ce.get('best_case_dir')}  score={ce.get('best_score')}")

    # OED
    oed = _summarize_oed(rd)
    if oed:
        print(f"\n-- Open-ended discovery iterations --")
        for line in oed:
            print(line)

    # Cases
    cases = _summarize_cases(rd)
    if cases:
        print(f"\n-- Experiment cases --")
        for line in cases:
            print(line)

    # Compiles (code_mod outcomes)
    comps = _summarize_compiles(rd)
    if comps:
        print(f"\n-- Code-mod compile attempts ({len(comps)}) --")
        for line in comps:
            print(line)

    # Tokens
    toks = _summarize_tokens(rd)
    if toks:
        print(f"\n-- Token usage --")
        for line in toks:
            print(line)

    # Timeline tail
    tl = _summarize_timeline(rd, limit=args.max_timeline)
    if tl:
        print(f"\n-- Recent timeline (last {len(tl)} events) --")
        for line in tl:
            print(line)

    # Paper / analysis hints
    if (rd / "paper").is_dir() or (rd / "paper_figs").is_dir():
        print("\n-- Paper/figures present --")
        for d in ("paper", "paper_figs", "figs", "analysis"):
            if (rd / d).is_dir():
                count = len(list((rd / d).rglob("*")))
                print(f"  {d}/  ({count} items)")

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
