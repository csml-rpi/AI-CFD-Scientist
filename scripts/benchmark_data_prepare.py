#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _score_relevance(topic: str, title: str, abstract: str) -> int:
    t = (topic or "").lower()
    text = f"{title} {abstract}".lower()
    score = 0
    for k in ["experimental", "experiment", "validation", "measured", "piv", "ldv", "benchmark", "data"]:
        if k in text:
            score += 2
    for k in ["backward", "step", "channel", "airfoil", "cavity", "pipe", "reattachment", "pressure drop", "drag", "lift"]:
        if k in t and k in text:
            score += 1
    return score


def _extract_numeric_hints(text: str) -> List[str]:
    vals = re.findall(r"\b(?:Re|Reynolds|Cd|Cl|Nu|Nusselt|Cp|Cf|pressure drop)\b[^.\n]{0,40}", text, flags=re.I)
    return vals[:10]


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare benchmark/experimental comparison candidates from literature.")
    parser.add_argument("--topic", required=True, type=str)
    parser.add_argument("--literature", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    args = parser.parse_args()

    lit = _read_json(Path(args.literature).expanduser().resolve(), [])
    if not isinstance(lit, list):
        lit = []

    candidates: List[Dict[str, Any]] = []
    for rec in lit:
        if not isinstance(rec, dict):
            continue
        title = str(rec.get("title", ""))
        abstract = str(rec.get("abstract", ""))
        score = _score_relevance(args.topic, title, abstract)
        if score <= 0:
            continue
        txt = f"{title}\n{abstract}"
        candidates.append(
            {
                "title": title,
                "doi": rec.get("doi", ""),
                "url": rec.get("url", ""),
                "year": rec.get("year", None),
                "relevance_score": score,
                "numeric_hints": _extract_numeric_hints(txt),
                "note": "Candidate experimental/benchmark source for CFD comparison.",
            }
        )

    candidates.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
    out = {
        "topic": args.topic,
        "benchmark_mode_enabled": bool(candidates),
        "selected_candidates": candidates[:10],
        "instructions": (
            "If benchmark_mode_enabled is true, analysis and paper stages must include "
            "explicit comparison against available experimental/benchmark references."
        ),
    }
    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({"benchmark_mode_enabled": out["benchmark_mode_enabled"], "candidates": len(out["selected_candidates"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

