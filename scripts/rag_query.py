#!/usr/bin/env python3
"""Query the prebuilt OpenFOAM FAISS indices.

Retrieval itself now lives in ``cfd_langgraph.foam_native.faiss_index`` — this
is the CLI wrapper the skill-driven path and ``foam_native/rag.py`` both use.
It no longer imports anything from Foam-Agent.

Two modes:

    # one question
    python scripts/rag_query.py --db openfoam_tutorials_details \
                                --query "simpleFoam periodic hill" --top 3

    # several questions, one process — see --batch below
    python scripts/rag_query.py --batch '[{"db": "...", "query": "...", "top": 3}, ...]'

``--batch`` exists because loading an index costs seconds and gigabytes, and a
single case asks three questions. Three separate invocations paid that cost
three times; one invocation pays it once.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cfd_langgraph.foam_native.faiss_index import DB_NAMES, retrieve  # noqa: E402


def _run(db: str, query: str, top: int) -> dict:
    return {"db": db, "query": query, "top": top, "results": retrieve(db, query, top)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Query the OpenFOAM FAISS indices.")
    parser.add_argument("--db", choices=list(DB_NAMES))
    parser.add_argument("--query", help="Natural-language query.")
    parser.add_argument("--top", type=int, default=3, help="Top-K matches to return.")
    parser.add_argument(
        "--batch",
        help='JSON list of {"db","query","top"} objects, answered in one process.',
    )
    parser.add_argument("--output", default="-", help="Output JSON path or '-' for stdout.")
    args = parser.parse_args()

    try:
        if args.batch:
            requests = json.loads(args.batch)
            payload = {
                "batch": [
                    _run(r["db"], r["query"], int(r.get("top", 3)))
                    for r in requests
                ]
            }
            count = sum(len(item["results"]) for item in payload["batch"])
        elif args.db and args.query:
            payload = _run(args.db, args.query, args.top)
            count = len(payload["results"])
        else:
            parser.error("give either --db/--query or --batch")
            return 2
    except FileNotFoundError as exc:
        print(f"{exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001
        print(f"FAISS query failed: {exc!r}", file=sys.stderr)
        return 3

    out = json.dumps(payload, indent=2, default=str)
    if args.output == "-":
        print(out)
    else:
        Path(args.output).write_text(out)
        print(f"wrote {count} results to {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
