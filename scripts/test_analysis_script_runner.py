#!/usr/bin/env python3
"""The cross-experiment analysis script runner (`agents/analysis_agent.py`).

Regression: the generated script was launched by a path relative to the repo
root while `cwd` was the case directory, so Python resolved it against that
directory and doubled it. Nothing matched, every attempt exited 2, and because
a non-zero exit is treated as "the script was wrong", all ten retries went on
asking the model to repair a script that had never run. Measured on a real
mesh gate: 10 attempts x 2 physics groups, ~30 minutes, identical error each
time, and the gate never produced a verdict.

Run: python scripts/test_analysis_script_runner.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

FAILURES: list[str] = []


def check(name: str, cond: object, detail: str = "") -> None:
    if cond:
        print(f"[PASS] {name}")
    else:
        FAILURES.append(name)
        print(f"[FAIL] {name}" + (f" — {detail}" if detail else ""))


def test_a_relative_script_path_breaks_under_a_different_cwd() -> None:
    """Demonstrates the mechanism, so the fix below is anchored to a real cause."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        case = root / "runs" / "study" / "group"
        proc_dir = case / "cross_experiment_analysis"
        proc_dir.mkdir(parents=True)
        script = proc_dir / "data_processing_script.py"
        script.write_text("print('ran')\n")

        relative = script.relative_to(root)
        broken = subprocess.run(
            [sys.executable, str(relative)], cwd=str(case), capture_output=True, text=True
        )
        check(
            "a repo-relative path under a case cwd fails to open",
            broken.returncode == 2 and "can't open file" in broken.stderr,
            detail=broken.stderr[:120],
        )
        check(
            "and the error shows the doubled path",
            str(case) in broken.stderr and broken.stderr.count("group") >= 2,
            detail=broken.stderr[:200],
        )

        fixed = subprocess.run(
            [sys.executable, str(script.resolve())], cwd=str(case), capture_output=True, text=True
        )
        check("an absolute path runs regardless of cwd", fixed.returncode == 0 and "ran" in fixed.stdout)


def test_runner_resolves_proc_dir_and_refuses_to_retry_invocation_faults() -> None:
    import inspect

    from cfd_langgraph.agents import analysis_agent

    source = inspect.getsource(analysis_agent)
    check(
        "the script directory is resolved to an absolute path",
        "proc_dir = Path(proc_dir).resolve()" in source,
    )
    check(
        "the script path is built from the resolved directory",
        source.index("proc_dir = Path(proc_dir).resolve()")
        < source.index('script_path = proc_dir / "data_processing_script.py"'),
    )
    check(
        "a missing-file exit is not treated as a repairable script error",
        "can't open file" in source and "invocation fault, not a script fault" in source,
    )


def test_the_qoi_batch_runner_is_also_immune() -> None:
    """The same bug existed in two places and only one was fixed.

    `analysis_agent.py` (cross-experiment) and `analyze.py`
    (`_llm_pyvista_batch_qoi`) both ran a generated script with `cwd` set to
    the script's own directory. Repairing only the first hid the second for
    four mesh-gate runs: every direct test passed `--output /tmp/...`, whose
    absolute work_dir never doubled, while the gate passes a repo-relative
    path and always did.
    """
    import inspect

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import analyze

    batch = inspect.getsource(analyze._llm_pyvista_batch_qoi)
    check("the batch work dir is resolved", "work_dir = Path(work_dir).resolve()" in batch)
    check(
        "resolution happens before the script path is built",
        batch.index("work_dir = Path(work_dir).resolve()")
        < batch.index('script_path = work_dir / "qoi_batch_script.py"'),
    )

    runner = inspect.getsource(analyze._run_python_script)
    check(
        "and the runner itself resolves whatever it is given",
        "script_path = Path(script_path).resolve()" in runner,
    )

    # Prove the runner is immune with a real relative path under a foreign cwd.
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        work = root / "runs" / "study" / ".qoi_llm_x"
        work.mkdir(parents=True)
        script = work / "qoi_batch_script.py"
        script.write_text("print('ok')\n")
        cwd_before = os.getcwd()
        try:
            os.chdir(root)
            rc, out, err = analyze._run_python_script(script.relative_to(root), cwd=work)
        finally:
            os.chdir(cwd_before)
        check("a relative script path runs under its own cwd", rc == 0 and "ok" in out, detail=f"rc={rc} {err[:120]}")


def main() -> int:
    for test in (
        test_a_relative_script_path_breaks_under_a_different_cwd,
        test_runner_resolves_proc_dir_and_refuses_to_retry_invocation_faults,
        test_the_qoi_batch_runner_is_also_immune,
    ):
        try:
            test()
        except Exception as exc:  # noqa: BLE001 — a raising test is a failing test
            check(test.__name__, False, detail=f"{type(exc).__name__}: {exc}")
    if FAILURES:
        print(f"\n{len(FAILURES)} FAILURE(S): " + ", ".join(FAILURES))
        return 1
    print("\nALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
