#!/usr/bin/env python3
"""Token and call accounting for one study, broken down by stage, model and agent.

Reads the study's own log (see cli/repl.py, which points CFD_TOKEN_LOG_DIR at
the run directory). Everything is recomputed from the per-call rows in
``llm_token_usage_calls.jsonl``, not taken from the running totals in
``llm_token_usage.json``: the rows are complete, while a breakdown added to the
logger mid-study (per agent, per token source) only counts calls made after its
code was loaded. Checked on the paper-matrix logs: the call and token counts
agree exactly either way; only those later breakdowns differ.

    python3 scripts/token_report.py runs/<study>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from cfd_langgraph.llm.token_usage_logger import read_rows, rebuild_totals  # noqa: E402


def _fmt(n: int) -> str:
    return f"{n:,}"


def _normalise(row: dict) -> dict:
    """Rows written before per-agent labels, or with the old default label.

    Until 2026-09-18 an unlabelled call was filed under agent "manager" in
    every stage, including stages that are separate processes with no manager
    in them; before that, rows had no agent at all. Such a call is the stage's
    own work, which is how the logger files it now.
    """
    stage = row.get("stage") or "unattributed"
    agent = row.get("agent") or ""
    if not agent or (agent == "manager" and stage != "manager"):
        row = {**row, "agent": stage}
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path, help="study output folder")
    ap.add_argument("--price-in", type=float, default=0.0,
                    help="USD per 1M input tokens, to add a cost column")
    ap.add_argument("--price-out", type=float, default=0.0,
                    help="USD per 1M output tokens")
    args = ap.parse_args()

    path = args.run_dir / "llm_token_usage.json"
    calls_path = args.run_dir / "llm_token_usage_calls.jsonl"
    if not path.is_file() and not calls_path.is_file():
        print(f"no token log at {path}")
        print("A study writes one only if it ran after per-study accounting was added;")
        print("older studies logged to a shared llm_token_usage.json in the working directory.")
        return 1

    rows = [_normalise(r) for r in read_rows(calls_path)]
    if rows:
        totals = rebuild_totals(rows)
        source = f"{len(rows):,} per-call rows"
    else:
        totals = json.loads(path.read_text(encoding="utf-8")).get("totals") or {}
        source = "running totals (no per-call rows)"
    by_stage = totals.get("by_stage") or {}

    def cost(i: int, o: int) -> float:
        return i / 1e6 * args.price_in + o / 1e6 * args.price_out

    priced = args.price_in or args.price_out
    print(f"study: {args.run_dir}")
    print(f"from:  {source}")
    print()
    head = f"{'stage / model / agent':<44}{'calls':>8}{'input':>16}{'cached':>15}{'output':>13}"
    if priced:
        head += f"{'USD':>10}"
    print(head)
    print("-" * len(head))

    def line(label: str, v: dict) -> str:
        s = (f"{label[:44]:<44}{v.get('calls', 0):>8}{_fmt(v.get('input_tokens', 0)):>16}"
             f"{_fmt(v.get('cached_input_tokens', 0)):>15}{_fmt(v.get('output_tokens', 0)):>13}")
        if priced:
            s += f"{cost(v.get('input_tokens', 0), v.get('output_tokens', 0)):>10.2f}"
        return s

    for stage, v in sorted(by_stage.items(), key=lambda kv: -kv[1].get("input_tokens", 0)):
        print(line(stage, v))
        for m, mv in sorted((v.get("by_model") or {}).items(), key=lambda kv: -kv[1].get("input_tokens", 0)):
            print(line(f"   model  {m}", mv))
        for a, av in sorted((v.get("by_agent") or {}).items(), key=lambda kv: -kv[1].get("input_tokens", 0)):
            print(line(f"   agent  {a}", av))
    print("-" * len(head))
    print(line("TOTAL", totals))
    for m, mv in sorted((totals.get("by_model") or {}).items(), key=lambda kv: -kv[1].get("input_tokens", 0)):
        print(line(f"   model  {m}", mv))

    ratio = totals.get("input_tokens", 0) / max(totals.get("output_tokens", 0), 1)
    print()
    print(f"input:output ratio {ratio:.0f}:1"
          "  -- a high ratio is context being re-sent each agent turn, not work being done.")

    # Data quality: how much of the totals above is the provider's own accounting.
    # Before 2026-09-18 the Codex provider logged a tokenizer ESTIMATE of the message
    # text (missing tool schemas, instructions, caching and reasoning) under the label
    # "provider_usage", and its structured-output calls were not logged at all. Rows
    # written since then carry cached_input_tokens; a Codex row without it is one of
    # those estimates, whatever its label says.
    if rows:
        kinds = {"provider usage": 0, "pre-fix Codex estimate": 0, "labelled estimate": 0,
                 "no usage reported": 0}
        tok = {k: 0 for k in kinds}
        for r in rows:
            src = str(r.get("token_source") or "")
            if src == "missing" or not (int(r.get("input_tokens") or 0) or int(r.get("output_tokens") or 0)):
                # Rows logged before 2026-09-18 say "provider_usage" even here:
                # a streamed call on an OpenAI-compatible endpoint (Qwen,
                # OpenRouter) returned no usage and was written down as zero.
                k = "no usage reported"
            elif src.startswith("estimate"):
                k = "labelled estimate"
            elif r.get("provider") == "openai-codex" and "cached_input_tokens" not in r:
                k = "pre-fix Codex estimate"
            else:
                k = "provider usage"
            kinds[k] += 1
            tok[k] += int(r.get("input_tokens") or 0) + int(r.get("output_tokens") or 0)
        total = sum(kinds.values()) or 1
        print()
        print("data quality (per-call log):")
        for k in kinds:
            print(f"  {k:<26}{kinds[k]:>8} calls  {kinds[k]/total:>6.1%}   {_fmt(tok[k]):>16} tokens")
        if kinds["pre-fix Codex estimate"]:
            print("  WARNING: pre-fix Codex rows under-count input (tool schemas and instructions are")
            print("  missing: ~8.4k tokens per manager call) and structured-output calls from that period")
            print("  are absent entirely. Totals for this study are a lower bound.")
        if kinds["no usage reported"]:
            print(f"  WARNING: {kinds['no usage reported']} calls happened but reported no token counts;")
            print("  they add nothing to the totals above, which are a lower bound.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
