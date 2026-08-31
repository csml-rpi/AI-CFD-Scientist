#!/usr/bin/env python3
"""Tests for the ported FAISS retrieval (`foam_native/faiss_index.py`).

This replaced an import of `Foam-Agent/src/utils.py`, so the thing worth
guarding is that it stays a self-contained port: same query preprocessing,
same result shape, no Foam-Agent module pulled in, and an index location that
can be pointed anywhere rather than assuming a vendored checkout.

The index files themselves are large prebuilt data, so the tests that need one
skip cleanly when none is present; the rest run everywhere.

Run: python scripts/test_faiss_index.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cfd_langgraph.foam_native import faiss_index  # noqa: E402

FAILURES: list[str] = []


def check(name: str, cond: object, detail: str = "") -> None:
    if cond:
        print(f"[PASS] {name}")
    else:
        FAILURES.append(name)
        print(f"[FAIL] {name}" + (f" — {detail}" if detail else ""))


def skip(name: str, why: str) -> None:
    print(f"[SKIP] {name} — {why}")


def test_tokenize_matches_the_index_build() -> None:
    """Queries must be preprocessed exactly as the documents were.

    A mismatch here does not fail — it silently returns worse matches, which
    is why it is pinned rather than left to drift.
    """
    check("underscores become spaces", faiss_index.tokenize("periodic_hill") == "periodic hill")
    check("camelCase is split", faiss_index.tokenize("simpleFoam") == "simple foam")
    check("result is lowercased", faiss_index.tokenize("RAS") == "ras")
    check(
        "both rules apply together",
        faiss_index.tokenize("momentumTransport_dict") == "momentum transport dict",
        detail=faiss_index.tokenize("momentumTransport_dict"),
    )


def test_index_root_is_configurable() -> None:
    """The indices are data. Nothing should require a Foam-Agent checkout to
    find them — pointing at a directory has to be enough."""
    prev = os.environ.get("CFD_SCIENTIST_FAISS_DIR")
    with tempfile.TemporaryDirectory() as tmp:
        try:
            os.environ["CFD_SCIENTIST_FAISS_DIR"] = tmp
            check("an explicit override wins", faiss_index.index_root() == Path(tmp))
            os.environ["CFD_SCIENTIST_FAISS_DIR"] = str(Path(tmp) / "does-not-exist")
            check(
                "a bad override falls through instead of breaking",
                faiss_index.index_root() != Path(tmp) / "does-not-exist",
            )
        finally:
            if prev is None:
                os.environ.pop("CFD_SCIENTIST_FAISS_DIR", None)
            else:
                os.environ["CFD_SCIENTIST_FAISS_DIR"] = prev


def test_unknown_database_is_refused() -> None:
    try:
        faiss_index.retrieve("openfoam_not_a_real_db", "x")
    except ValueError as exc:
        check("an unknown index name is refused by name", "openfoam_not_a_real_db" in str(exc))
    except Exception as exc:  # noqa: BLE001
        check("an unknown index name is refused by name", False, detail=f"{type(exc).__name__}: {exc}")
    else:
        check("an unknown index name is refused by name", False, detail="no error raised")


def test_missing_index_says_where_to_put_it() -> None:
    prev = os.environ.get("CFD_SCIENTIST_FAISS_DIR")
    with tempfile.TemporaryDirectory() as tmp:
        try:
            os.environ["CFD_SCIENTIST_FAISS_DIR"] = tmp
            faiss_index._load.cache_clear()
            faiss_index.retrieve("openfoam_allrun_scripts", "x")
        except FileNotFoundError as exc:
            check("a missing index is an explicit FileNotFoundError", "openfoam_allrun_scripts" in str(exc))
        except Exception as exc:  # noqa: BLE001
            check("a missing index is an explicit FileNotFoundError", False, detail=f"{type(exc).__name__}")
        else:
            check("a missing index is an explicit FileNotFoundError", False, detail="no error raised")
        finally:
            if prev is None:
                os.environ.pop("CFD_SCIENTIST_FAISS_DIR", None)
            else:
                os.environ["CFD_SCIENTIST_FAISS_DIR"] = prev
            faiss_index._load.cache_clear()


def test_live_retrieval_shape() -> None:
    root = faiss_index.index_root()
    model_dir = (root / faiss_index.embedding_model_name().replace("/", "_").replace(":", "_")) if root else None
    if model_dir is None or not model_dir.is_dir():
        skip("live retrieval returns FoamAgent's result shape", "no prebuilt index available here")
        return
    results = faiss_index.retrieve("openfoam_tutorials_details", "simpleFoam periodic hill", 2)
    check("a query returns the requested number of matches", len(results) == 2, detail=str(len(results)))
    entry = results[0]
    for field in ("index", "full_content", "case_name", "case_solver", "dir_structure", "tutorials", "score"):
        check(f"result carries `{field}`", field in entry)
    check("the match has real content", len(str(entry.get("tutorials", ""))) > 100)


def test_no_foam_agent_module_is_loaded() -> None:
    """The whole point of the port."""
    loaded = [
        name
        for name in sys.modules
        if name in ("utils", "config", "tracking_aws", "router_func") or name.startswith("services.")
    ]
    check("no Foam-Agent module is imported by retrieval", not loaded, detail=str(loaded))


def main() -> int:
    tests = (
        test_tokenize_matches_the_index_build,
        test_index_root_is_configurable,
        test_unknown_database_is_refused,
        test_missing_index_says_where_to_put_it,
        test_live_retrieval_shape,
        test_no_foam_agent_module_is_loaded,
    )
    for test in tests:
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
