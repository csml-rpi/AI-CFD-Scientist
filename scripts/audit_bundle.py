#!/usr/bin/env python3
"""Bundle-level audit — item 2.

The per-run stage-gate audit (`scripts/stage_gate_audit.py`) writes a signed
`audit_passed.json` into a run directory only on rc=0. A *bundle* is a parent
directory holding several run directories (e.g. a smoke-test batch such as
`runs/skill_smoke_claude/`). This script verifies that EVERY task in a bundle
produced a valid final audit record.

The absence of `audit_passed.json` is itself treated as a process failure: a
task whose skill chain stopped before `cfd-orchestrator` Step 5, or whose audit
never reached rc=0, is not complete — even if its solver runs and paper look
fine. (This is exactly the Sonnet skill-smoke pattern where BFS/Jet/OED had
real runs but no audit record.)

Usage:
    python scripts/audit_bundle.py --bundle runs/skill_smoke_claude
    python scripts/audit_bundle.py --bundle <dir> --json
    python scripts/audit_bundle.py --bundle <dir> --rerun-audit   # run the
        per-run audit on tasks lacking a valid record, then re-verify

Exit codes:
    0 — every task subdirectory carries a valid signed audit_passed.json
    2 — one or more tasks missing / invalid / failed audit record
    6 — usage error (bundle dir missing, or no task subdirectories found)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

AUDIT_SIGNATURE = "stage_gate_audit.py:v1"
SCRIPT_DIR = Path(__file__).resolve().parent


def _read_json(p: Path) -> Any:
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _find_task_dirs(bundle: Path) -> List[Path]:
    """Immediate subdirectories that look like CFD-scientist run directories —
    they carry a state.json (written by cfd-orchestrator Step 2)."""
    return sorted(
        d for d in bundle.iterdir()
        if d.is_dir() and (d / "state.json").is_file()
    )


def _verify_task(task: Path) -> Tuple[bool, str]:
    """Verify a single task carries a valid signed, passing audit record."""
    ap = task / "audit_passed.json"
    if not ap.is_file():
        return False, ("audit_passed.json absent — the skill chain never "
                       "reached a passing final audit (process failure)")
    data = _read_json(ap)
    if not isinstance(data, dict):
        return False, "audit_passed.json present but unreadable as JSON"
    if data.get("audit_signature") != AUDIT_SIGNATURE:
        return False, (f"audit_passed.json signature {data.get('audit_signature')!r} "
                       f"!= {AUDIT_SIGNATURE!r} — hand-fabricated, not written "
                       f"by stage_gate_audit.py")
    if data.get("rc") != 0:
        return False, f"audit_passed.json rc={data.get('rc')!r} (expected 0)"
    nf = data.get("n_failures")
    if isinstance(nf, int) and nf != 0:
        return False, f"audit_passed.json n_failures={nf} (expected 0)"
    return True, "valid signed audit record (rc=0)"


def _run_audit(task: Path) -> None:
    audit = SCRIPT_DIR / "stage_gate_audit.py"
    try:
        subprocess.run(
            [sys.executable, str(audit), "--out-dir", str(task)],
            timeout=900,
        )
    except Exception as e:  # noqa: BLE001
        print(f"  (could not run stage_gate_audit.py on {task.name}: {e})",
              file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser(description="Bundle-level CFD-scientist audit.")
    ap.add_argument("--bundle", required=True,
                    help="parent directory holding several run directories")
    ap.add_argument("--rerun-audit", action="store_true",
                    help="run stage_gate_audit.py on tasks lacking a valid "
                         "record, then re-verify")
    ap.add_argument("--json", action="store_true", help="emit a JSON report")
    args = ap.parse_args()

    bundle = Path(args.bundle).resolve()
    if not bundle.is_dir():
        print(f"ERROR: bundle directory not found: {bundle}", file=sys.stderr)
        return 6
    tasks = _find_task_dirs(bundle)
    if not tasks:
        print(f"ERROR: no task subdirectories (with state.json) under {bundle}",
              file=sys.stderr)
        return 6

    if args.rerun_audit:
        for t in tasks:
            ok, _ = _verify_task(t)
            if not ok:
                print(f"re-running stage-gate audit on {t.name} ...")
                _run_audit(t)

    results: List[Dict[str, Any]] = []
    for t in tasks:
        ok, detail = _verify_task(t)
        results.append({"task": t.name, "ok": ok, "detail": detail})

    n_fail = sum(1 for r in results if not r["ok"])
    rc = 0 if n_fail == 0 else 2
    report = {
        "bundle": str(bundle),
        "n_tasks": len(tasks),
        "n_passed": len(tasks) - n_fail,
        "n_failed": n_fail,
        "rc": rc,
        "tasks": results,
    }
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"Bundle audit: {bundle}")
        for r in results:
            mark = "PASS" if r["ok"] else "FAIL"
            print(f"  [{mark}] {r['task']}: {r['detail']}")
        if rc == 0:
            print(f"OK — all {len(tasks)} task(s) carry a valid final audit record.")
        else:
            print(f"FAIL — {n_fail}/{len(tasks)} task(s) have no valid final "
                  f"audit record. A missing audit_passed.json is a process "
                  f"failure: re-run each failing task's chain through "
                  f"Skill cfd-orchestrator Step 5 until the audit reaches rc=0.",
                  file=sys.stderr)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
