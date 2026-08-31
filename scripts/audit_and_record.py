#!/usr/bin/env python3
"""Run the stage-gate audit for a study and, if it passes, record it into the
knowledge bundle — a lessons entry plus a new validation-suite case.

This is the standalone path into ``knowledge_bundle/`` for studies run any
way other than through the manager CLI (e.g. a study driven by hand, by
``orchestrator_run.py``, or by the skill-mode chain): the knowledge bundle
grows the same way regardless of how the study itself was run, because
recording only ever happens here or in the equivalent manager tool
(``run_audit_and_record`` in ``src/cfd_langgraph/manager/tools.py``) — never
anywhere else.

Usage:
    python scripts/audit_and_record.py --out-dir runs/<study>

Exit code mirrors stage_gate_audit.py's: 0 only if the audit passed (and the
study was recorded); non-zero otherwise, with nothing recorded.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from cfd_langgraph.knowledge_bundle import KnowledgeBundle  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--knowledge-bundle-dir", default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()

    proc = subprocess.run(
        [sys.executable, str(_REPO_ROOT / "scripts" / "stage_gate_audit.py"), "--out-dir", str(out_dir)],
        cwd=str(_REPO_ROOT),
    )
    if proc.returncode != 0:
        print(f"Audit failed (rc={proc.returncode}) — nothing recorded into the knowledge bundle.", file=sys.stderr)
        return proc.returncode

    bundle = KnowledgeBundle(Path(args.knowledge_bundle_dir) if args.knowledge_bundle_dir else None)
    entry = bundle.record_study(out_dir)
    n = len(bundle.list_validation_cases())
    print(f"Recorded {entry['study_id']} into the knowledge bundle. Validation suite now has {n} case(s).")
    if not bundle.is_bootstrapped():
        need = 3 - n
        print(f"({need} more audited stud{'y' if need == 1 else 'ies'} before self-evolution activates.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
