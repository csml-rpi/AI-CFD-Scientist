"""A case's own result files as text, for the agents that write about the case."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

# The harness's own bookkeeping beside a case's result files.
_HARNESS_RECORDS = {"run_result.json", "setup_check.json", "decision.json", "failure.json", "figures.json"}


def case_records(case_dir: Path, *, each: int = 8000, total: int = 24000) -> Dict[str, str]:
    """A case's own result files (JSON and CSV at its top level and in
    results/) as text, ``total`` characters in all and at most ``each`` per
    file. The writer had only the cross-case summary and wrote that the
    gradient check, the training history and the final coefficients "were not
    supplied", though the cases had saved them (nn_in_solver_demo_20261022_1129,
    case_004 to case_006)."""
    texts: Dict[str, str] = {}
    found = [p for folder in (case_dir, case_dir / "results") if folder.is_dir()
             for p in sorted(folder.iterdir())
             if p.is_file() and p.suffix in (".json", ".csv")
             and not p.name.startswith(".") and p.name not in _HARNESS_RECORDS]
    for p in found:
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if p.suffix == ".json":
            try:
                text = json.dumps(json.loads(text), separators=(",", ":"))
            except ValueError:
                pass
        texts[str(p.relative_to(case_dir))] = text
    # Shared out smallest first: a small file is shown whole and a large one
    # gets an even share of what is left. Taken in name order, a 61-character
    # notice replaced a short training history after three larger files had
    # used the budget.
    out: Dict[str, str] = {}
    left = total
    order = sorted(texts, key=lambda name: len(texts[name]))
    for n, name in enumerate(order):
        text = texts[name]
        cut = min(each, left // (len(order) - n))
        out[name] = text if len(text) <= cut else text[:cut] + f"\n... ({len(text)} characters in all)"
        left -= min(len(text), cut)
    return {name: out[name] for name in texts}
