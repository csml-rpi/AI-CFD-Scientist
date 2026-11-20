"""Test-set targets withheld from candidate build agents: input-only copies,
hidden files, and the sandbox serving the copies to the shell and to read_file.

Run: python scripts/test_withheld_data.py   (needs bubblewrap)
"""

from __future__ import annotations

import csv
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from cfd_langgraph.withheld_data import MANIFEST_ENV, build_view, data_listing  # noqa: E402

FAILURES: list[str] = []


def check(name: str, cond: object, detail: str = "") -> None:
    if cond:
        print(f"[PASS] {name}")
    else:
        FAILURES.append(name)
        print(f"[FAIL] {name}" + (f" — {detail}" if detail else ""))


with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    starter = tmp / "starter"
    (starter / "data" / "processed").mkdir(parents=True)
    (starter / "data" / "raw" / "testData").mkdir(parents=True)
    np.savez(starter / "data" / "processed" / "test.npz", coords=np.ones((3, 4)), cp=np.full((3, 4), 7.0))
    np.savez(starter / "data" / "processed" / "train.npz", coords=np.ones((5, 4)), cp=np.zeros((5, 4)))
    with (starter / "data" / "processed" / "test_params.csv").open("w", newline="") as f:
        csv.writer(f).writerows([["sample", "mach", "c_l"], [500, 0.7, 0.42], [501, 0.8, 0.51]])
    (starter / "data" / "raw" / "testData" / "Sample500.dat").write_text("x y cp\n0 0 7.0\n")

    print("== 1. listing and input-only copies")
    listing = data_listing(starter)
    check("the listing shows archive keys and table columns",
          "cp[3, 4]" in listing and "'c_l'" in listing, listing)
    spec = [
        {"path": "data/processed/test.npz", "action": "keep_fields", "fields_to_keep": ["coords"]},
        {"path": "data/processed/test_params.csv", "action": "keep_fields", "fields_to_keep": ["sample", "mach"]},
        {"path": "data/raw/testData", "action": "hide"},
        {"path": "data/processed/missing.npz", "action": "hide"},
    ]
    view = tmp / "study" / "withheld_test_targets"
    m = build_view(starter, spec, view)
    by_name = {Path(e["path"]).name: e for e in m["entries"]}
    check("a file that does not exist is skipped", "missing.npz" not in by_name and len(m["entries"]) == 3,
          [e["path"] for e in m["entries"]])
    with np.load(by_name["test.npz"]["source"]) as z:
        check("the archive copy keeps only the input key", list(z.files) == ["coords"], z.files)
    with open(by_name["test_params.csv"]["source"], newline="") as f:
        rows = list(csv.reader(f))
    check("the table copy keeps only the input columns", rows[0] == ["sample", "mach"] and len(rows) == 3, rows)
    check("the folder that cannot be split is hidden", by_name["testData"]["action"] == "hide"
          and by_name["testData"]["is_dir"])
    with np.load(starter / "data" / "processed" / "test.npz") as z:
        check("the originals are untouched", "cp" in z.files)

    print("== 2. the sandbox serves the copies")
    os.environ[MANIFEST_ENV] = str(view / "manifest.json")
    from code_mod_agentic import Sandbox  # noqa: E402

    run_dir = tmp / "study" / "cand"
    run_dir.mkdir(parents=True)
    sb = Sandbox(run_dir=run_dir, starter_case=starter, wm_project_dir=None, starter_root=starter)
    py = sys.executable
    r = sb.run_bash(f"{py} -c \"import numpy; print(sorted(numpy.load('{starter}/data/processed/test.npz').files))\"",
                    cwd=str(run_dir))
    check("the shell sees only the input key of the test archive", "['coords']" in r.get("stdout", ""), r)
    r = sb.run_bash(f"cat {starter}/data/processed/test_params.csv", cwd=str(run_dir))
    check("the shell sees only the input columns of the test table",
          "mach" in r.get("stdout", "") and "c_l" not in r.get("stdout", "") and "0.42" not in r.get("stdout", ""), r)
    r = sb.run_bash(f"ls {starter}/data/raw/testData; cat {starter}/data/raw/testData/Sample500.dat", cwd=str(run_dir))
    check("the hidden folder is empty to the shell", "7.0" not in r.get("stdout", ""), r)
    r = sb.run_bash(f"{py} -c \"import numpy; print(numpy.load('{starter}/data/processed/train.npz')['cp'].shape)\"",
                    cwd=str(run_dir))
    check("training data is still readable in full", "(5, 4)" in r.get("stdout", ""), r)
    rf = sb.read_file(str(starter / "data" / "raw" / "testData" / "Sample500.dat"))
    check("read_file refuses a hidden file", not rf.get("ok") and "withheld" in rf.get("error", ""), rf)
    rf = sb.read_file(str(starter / "data" / "processed" / "test_params.csv"))
    check("read_file serves the input-only copy", rf.get("ok") and "c_l" not in rf.get("content", ""), rf)

    print("== 3. the candidate brief")
    import surrogate_agentic as sa  # noqa: E402

    brief = sa.build_surrogate_prompt(topic="t", hypothesis="h", variant_name="v", run_dir=run_dir, starter_case=starter)
    check("the brief says the targets are withheld and not to score on test",
          "target values are withheld" in brief and "do not try" in brief
          and "Run the study's scorer on your files" not in brief)
    os.environ.pop(MANIFEST_ENV)
    # A study folder with no withheld targets: the manifest is also found by
    # walking up from the candidate's folder, so this one is outside the study.
    plain = tmp / "plain_study" / "cand"
    plain.mkdir(parents=True)
    brief = sa.build_surrogate_prompt(topic="t", hypothesis="h", variant_name="v", run_dir=plain, starter_case=starter)
    check("without a manifest the brief is unchanged", "Run the study's scorer on your files" in brief)

print()
print("FAILURES:", FAILURES if FAILURES else "none")
sys.exit(1 if FAILURES else 0)
