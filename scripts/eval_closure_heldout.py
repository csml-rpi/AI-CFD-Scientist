#!/usr/bin/env python3
"""Evaluate a discovered model on the closure challenge's held-out test set.

Run this AFTER a study finishes, never during one. The study is set up over
training and validation cases only (scripts/closure_split.py), so the eight
graded cases are never visible to the search; this script is the single place
they are used, and it is invoked by hand.

What it does, per test case:
  1. copy the case, install the discovered model into the copy
  2. run the case's own solver from its initial condition
  3. sample U at the challenge's evaluation points for that case
  4. write predictions as CSV in the submission layout

Then, if the closure_challenge package is importable, it scores the result.
Scoring is optional on purpose: producing the CSVs is the deliverable, and a
missing scorer should not stop you getting them.

    python3 scripts/eval_closure_heldout.py \\
        --model-case runs/<study>/open_ended_discovery/cand_<x>/<x> \\
        --data /path/to/ported/benchmark/data \\
        --out submissions/mine
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from closure_split import TEST_CASES  # noqa: E402


def _openfoam_env(openfoam_dir: str) -> Dict[str, str]:
    """Environment with OpenFOAM sourced, read from the install itself."""
    import os

    bashrc = Path(openfoam_dir) / "etc" / "bashrc"
    if not bashrc.is_file():
        return dict(os.environ)
    probe = subprocess.run(
        ["bash", "-lc", f'. "{bashrc}" >/dev/null 2>&1 && env -0'],
        capture_output=True, text=True, timeout=120,
    )
    if probe.returncode != 0:
        return dict(os.environ)
    env = dict(os.environ)
    for chunk in probe.stdout.split("\0"):
        if "=" in chunk:
            key, _, value = chunk.partition("=")
            env[key] = value
    return env


def install_model(model_case: Path, target: Path, model_name: str) -> Optional[str]:
    """Reuses the manager's own installer so the study and this script put a
    model into a case in exactly the same way — a second implementation here
    is a second thing to drift."""
    from cfd_langgraph.manager.tools import _install_model_into_case

    return _install_model_into_case(model_case, target, model_name)


def case_application(case: Path) -> str:
    from cfd_langgraph.manager.tools import _case_application

    return _case_application(case) or "simpleFoam"


def sample_velocity(case: Path, points) -> "object":
    """U at the challenge's evaluation points, sampled from the solved mesh.

    pyvista rather than an OpenFOAM sampling dict: the evaluation points are
    an arbitrary scattered set, the reader gives the same interpolation the
    challenge's own tooling uses, and it needs nothing added to the case.
    """
    import numpy as np
    import pyvista as pv

    marker = case / "case.foam"
    marker.touch()
    reader = pv.OpenFOAMReader(str(marker))
    times = [t for t in reader.time_values if t > 0]
    if not times:
        raise RuntimeError(f"{case} has no solved time directory")
    reader.set_active_time_value(max(times))
    mesh = reader.read()["internalMesh"]
    probe = pv.PolyData(np.asarray(points, dtype=float)).sample(mesh)
    return np.asarray(probe["U"])


def evaluation_points(case_name: str, data_root: Path):
    """The 999 scored points for a case.

    Preferring the challenge package, which is authoritative, and falling back
    to the CSV the benchmark ships for convenience. They are the same points;
    the fallback exists so this runs without the package installed.
    """
    import numpy as np

    try:
        from closure_challenge.dataset_utils import evaluation_points as pkg_points

        return np.asarray(pkg_points(case_name), dtype=float)
    except Exception:
        csv = data_root / "evaluation_points" / f"{case_name}_points.csv"
        if not csv.is_file():
            raise RuntimeError(
                f"no evaluation points for {case_name}: install the closure_challenge "
                f"package or provide {csv}"
            )
        return np.loadtxt(csv, delimiter=",")


def main() -> int:
    import numpy as np

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-case", required=True, type=Path,
                        help="a case containing the discovered model's customModels/")
    parser.add_argument("--model-name", default="",
                        help="RASModel name; inferred from customModels/ when omitted")
    parser.add_argument("--data", required=True, type=Path,
                        help="ported benchmark data directory")
    parser.add_argument("--out", required=True, type=Path,
                        help="where to write the submission CSVs")
    parser.add_argument("--work", type=Path, default=None,
                        help="scratch directory for the case runs")
    parser.add_argument("--openfoam", default="/mnt/sda1/openfoam10")
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--cases", default="",
                        help="comma-separated subset of test case names (default: all eight)")
    args = parser.parse_args()

    model_name = args.model_name.strip()
    if not model_name:
        custom = args.model_case / "customModels"
        entries = [p.name for p in custom.iterdir() if p.is_dir()] if custom.is_dir() else []
        if len(entries) != 1:
            print(f"could not infer --model-name from {custom} (found {entries})", file=sys.stderr)
            return 1
        model_name = entries[0]
    print(f"model: {model_name}   from: {args.model_case}")

    wanted = [c.strip() for c in args.cases.split(",") if c.strip()] or list(TEST_CASES)
    work = args.work or (args.out / "_work")
    work.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)
    env = _openfoam_env(args.openfoam)

    written: List[str] = []
    skipped: Dict[str, str] = {}
    for name in wanted:
        relative = TEST_CASES.get(name)
        if relative is None:
            skipped[name] = "not a test case"
            continue
        source = args.data / relative
        if not (source / "system" / "controlDict").is_file():
            skipped[name] = f"case not found at {source}"
            print(f"  SKIP {name}: {skipped[name]}")
            continue

        target = work / name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target, symlinks=True)
        # Solve from the initial condition, not from a converged field the
        # benchmark shipped: restarting from the baseline solution measures
        # how far the model moves an already-converged baseline, which is a
        # different quantity from solving the case with it.
        for child in list(target.iterdir()):
            if child.is_dir() and child.name != "0":
                try:
                    float(child.name)
                except ValueError:
                    continue
                shutil.rmtree(child, ignore_errors=True)

        error = install_model(args.model_case, target, model_name)
        if error:
            skipped[name] = f"model install failed: {error}"
            print(f"  SKIP {name}: {skipped[name]}")
            continue

        application = case_application(target)
        print(f"  RUN  {name} ({application}) ...", flush=True)
        proc = subprocess.run(
            ["bash", "-lc", f'cd "{target}" && {application}'],
            capture_output=True, text=True, timeout=args.timeout, env=env,
        )
        if "End" not in (proc.stdout or "")[-2000:]:
            skipped[name] = "solver did not reach End"
            tail = (proc.stdout or proc.stderr or "").strip().splitlines()[-1:] or [""]
            print(f"  FAIL {name}: {skipped[name]} — {tail[0][:90]}")
            continue

        try:
            points = evaluation_points(name, args.data)
            velocity = sample_velocity(target, points)
        except Exception as exc:
            skipped[name] = f"sampling failed: {exc}"
            print(f"  FAIL {name}: {skipped[name]}")
            continue

        destination = args.out / f"{name}.csv"
        np.savetxt(destination, velocity, delimiter=",")
        written.append(name)
        print(f"  OK   {name}: wrote {destination.name} ({velocity.shape[0]} rows)")

    print()
    print(f"written: {len(written)}/{len(wanted)}")
    if skipped:
        print("skipped:")
        for name, why in skipped.items():
            print(f"  {name}: {why}")

    if len(written) == len(TEST_CASES):
        try:
            from closure_challenge import evaluate_from_csv_by_case, score_from_csv

            per_case = evaluate_from_csv_by_case(str(args.out))
            total = score_from_csv(str(args.out))
            print()
            print("SCORE (lower is better)")
            for case, value in per_case.items():
                print(f"  {case:<24} {value:.4f}")
            print(f"  {'OVERALL':<24} {total:.4f}")
            (args.out / "score.json").write_text(
                json.dumps({"overall": total, "per_case": per_case}, indent=2)
            )
        except Exception as exc:
            print(f"\n(scoring unavailable: {exc})")
    else:
        print("\nnot scoring: the challenge scores all eight cases together, "
              "and a partial set is not a submission.")
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
