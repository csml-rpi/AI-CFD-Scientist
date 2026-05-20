#!/usr/bin/env python3
"""Thin FAISS query CLI for the cfd-foamagent skill.

The cfd-foamagent skill drives FoamAgent's case-generation loop in markdown
(planner → write → Allrun → run → review) without going through the full
foam_run.py wrapper. The one piece of FoamAgent that genuinely needs Python
is FAISS retrieval over the prebuilt OpenFOAM tutorial indices. This script
is the standalone wrapper: read a query, return top-K similar cases as JSON.

Usage:
    python scripts/rag_query.py --db <db_name> --query "<natural language>" \
                                --top 3 --output rag.json

DB names (must match Foam-Agent's prebuilt indices under
Foam-Agent/database/faiss/<embedding_model>/):
    openfoam_allrun_scripts        — Allrun examples
    openfoam_tutorials_structure   — directory + foamfile structure
    openfoam_tutorials_details     — full case content
    openfoam_command_help          — solver command help
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FOAM_AGENT_SRC = REPO / "Foam-Agent" / "src"


def main() -> int:
    parser = argparse.ArgumentParser(description="Query FoamAgent FAISS indices.")
    parser.add_argument("--db", required=True,
                        choices=["openfoam_allrun_scripts",
                                 "openfoam_tutorials_structure",
                                 "openfoam_tutorials_details",
                                 "openfoam_command_help"])
    parser.add_argument("--query", required=True, help="Natural-language query.")
    parser.add_argument("--top", type=int, default=3, help="Top-K matches to return.")
    parser.add_argument("--output", default="-", help="Output JSON path or '-' for stdout.")
    args = parser.parse_args()

    if str(FOAM_AGENT_SRC) not in sys.path:
        sys.path.insert(0, str(FOAM_AGENT_SRC))

    try:
        from utils import retrieve_faiss  # type: ignore
    except Exception as exc:
        print(f"failed to import FoamAgent FAISS retrieval: {exc!r}", file=sys.stderr)
        print("ensure FAISS DBs are built: python Foam-Agent/init_database.py", file=sys.stderr)
        return 2

    try:
        results = retrieve_faiss(args.db, args.query, topk=args.top)
    except Exception as exc:
        print(f"FAISS query failed: {exc!r}", file=sys.stderr)
        return 3

    payload = {
        "db": args.db,
        "query": args.query,
        "top": args.top,
        "results": results if isinstance(results, list) else [results],
    }

    out = json.dumps(payload, indent=2, default=str)
    if args.output == "-":
        print(out)
    else:
        Path(args.output).write_text(out)
        print(f"wrote {len(payload['results'])} results to {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
