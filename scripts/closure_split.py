#!/usr/bin/env python3
"""The closure-challenge train / validation / test split, in one place.

The challenge's one strict rule is that nothing may be trained or validated
on a test case. The way this repository honours that is not by policing the
agent's file access but by never putting the test cases in front of it: a
study is set up over the training and validation cases only, and the held-out
set is evaluated once, afterwards, by scripts/eval_closure_heldout.py.

That makes the guarantee structural rather than behavioural. There is no
sandbox to escape and no rule to forget, because the data is not in the study.

Split as published in the benchmark README. TEST is the graded set; the
leaderboard scores exactly these eight.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

# The eight graded cases. Nothing in a study may touch these.
TEST_CASES: Dict[str, str] = {
    "alpha_15_13929_4048": "Parm_PH_29/alpha_15/alpha_15_13929_4048",
    "alpha_15_13929_2024": "Parm_PH_29/alpha_15/alpha_15_13929_2024",
    "alpha_05_4071_4048": "Parm_PH_29/alpha_05/alpha_05_4071_4048",
    "alpha_05_4071_2024": "Parm_PH_29/alpha_05/alpha_05_4071_2024",
    "AR_1_Ret_360": "DUCT/AR_1_Ret_360",
    "AR_3_Ret_360": "DUCT/AR_3_Ret_360",
    "AR_14_Ret_180": "DUCT/AR_14_Ret_180",
    "NASA_2DWMH": "NASA_2DWMH",
}

# The suggested validation split. Held back from fitting but visible to a
# study, which is what validation is for.
VALIDATION_CASES: Dict[str, str] = {
    "alpha_05_10071_4048": "Parm_PH_29/alpha_05/alpha_05_10071_4048",
    "alpha_05_10071_2024": "Parm_PH_29/alpha_05/alpha_05_10071_2024",
    "alpha_15_7929_4048": "Parm_PH_29/alpha_15/alpha_15_7929_4048",
    "alpha_15_7929_2024": "Parm_PH_29/alpha_15/alpha_15_7929_2024",
    "AR_7_Ret_180": "DUCT/AR_7_Ret_180",
}


def _case_dirs(root: Path) -> List[Path]:
    return sorted(p.parent.parent for p in root.rglob("system/controlDict"))


def classify(root: Path) -> Dict[str, List[Path]]:
    """Every case under ``root``, split into test / validation / training."""
    test_rel = set(TEST_CASES.values())
    validation_rel = set(VALIDATION_CASES.values())
    out: Dict[str, List[Path]] = {"test": [], "validation": [], "training": []}
    for case in _case_dirs(root):
        try:
            rel = str(case.relative_to(root))
        except ValueError:
            continue
        if rel in test_rel:
            out["test"].append(case)
        elif rel in validation_rel:
            out["validation"].append(case)
        else:
            out["training"].append(case)
    return out


def study_cases(root: Path) -> List[Path]:
    """What a study may see: everything that is not a graded test case."""
    split = classify(root)
    return sorted(split["training"] + split["validation"])


def main() -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path,
                        help="ported benchmark data directory")
    parser.add_argument("--json", action="store_true", help="emit JSON")
    args = parser.parse_args()

    split = classify(args.root)
    if args.json:
        print(json.dumps({k: [str(p) for p in v] for k, v in split.items()}, indent=2))
        return 0
    for name, cases in split.items():
        print(f"{name}: {len(cases)}")
        for case in cases:
            print(f"    {case.relative_to(args.root)}")
    print()
    print(f"a study may use {len(study_cases(args.root))} cases "
          f"({len(split['training'])} training + {len(split['validation'])} validation)")
    print(f"held out from every study: {len(split['test'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
