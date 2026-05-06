#!/usr/bin/env python3
"""Verify Codex OAuth for CFD Scientist (same path as CFD_SCIENTIST_LLM_PROVIDER=openai-codex).

Prerequisites:
  - Codex CLI installed and signed in (typically ``~/.codex/auth.json``), or Clawdbot OAuth cache.

Usage (from repo root, with conda env activated if you use one)::

  python scripts/test_codex_oauth.py
  python scripts/test_codex_oauth.py --model gpt-5-codex
  python scripts/test_codex_oauth.py --prompt "Say hi in 3 words."
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _setup_path() -> None:
    src = ROOT / "src"
    s = str(src)
    if s not in sys.path:
        sys.path.insert(0, s)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Test Codex OAuth + ChatGPT Codex Responses API (matches cfd_langgraph openai-codex path)."
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("CFD_SCIENTIST_MODEL", "gpt-5-codex"),
        help="Codex model id (default: $CFD_SCIENTIST_MODEL or gpt-5-codex).",
    )
    parser.add_argument(
        "--prompt",
        default="Reply with exactly one word: pong",
        help="User message to send.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for the test call.",
    )
    args = parser.parse_args()

    _setup_path()

    try:
        from cfd_langgraph.llm.factory import smoke_test_codex_oauth

        text = smoke_test_codex_oauth(
            model=args.model,
            temperature=args.temperature,
            prompt=args.prompt,
        )
    except FileNotFoundError as e:
        print("FAIL — no OAuth cache:", e, file=sys.stderr)
        print(
            "Install Codex CLI, then sign in (e.g. run `codex` and choose Sign in with ChatGPT).",
            file=sys.stderr,
        )
        return 1
    except Exception as e:
        print("FAIL —", type(e).__name__ + ":", e, file=sys.stderr)
        return 1

    print("OK — model replied:")
    print(text)
    print()
    print("PASS — Codex OAuth round-trip succeeded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
