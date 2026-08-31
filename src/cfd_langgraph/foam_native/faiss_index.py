"""FAISS retrieval over the prebuilt OpenFOAM tutorial indices.

FoamAgent Stage 2's data lookup, ported so the CLI stops importing the
vendored ``Foam-Agent/src/utils.py`` for it. That module builds an embedding
model and loads **all four** indices at import time, so asking one question
cost ~2.3 GB and ~9.5 s regardless of which index was wanted — and the CLI
asked three separate questions per case, in three separate subprocesses.

Only the index actually queried is loaded here, and it is cached for the life
of the process, so a batch of questions costs one load rather than one each.

The indices themselves are data, not code: they are large, prebuilt, and we
have no copy of our own. ``index_root`` therefore takes the location from the
environment first, so pointing at a directory of indices is all that is
needed — no Foam-Agent checkout.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parents[3]

DB_NAMES = (
    "openfoam_allrun_scripts",
    "openfoam_tutorials_structure",
    "openfoam_tutorials_details",
    "openfoam_command_help",
)

# Fields carried out of each index's document metadata, per index. Kept
# identical to what FoamAgent's own retrieve_faiss returned: the callers
# downstream (rag.py, and the skill-driven path via scripts/rag_query.py) read
# these keys by name.
_FIELDS: Dict[str, tuple] = {
    "openfoam_allrun_scripts": (
        "full_content", "case_name", "case_domain", "case_category",
        "case_solver", "dir_structure", "allrun_script",
    ),
    "openfoam_command_help": ("full_content", "command", "help_text"),
    "openfoam_tutorials_structure": (
        "full_content", "case_name", "case_domain", "case_category",
        "case_solver", "dir_structure",
    ),
    "openfoam_tutorials_details": (
        "full_content", "case_name", "case_domain", "case_category",
        "case_solver", "dir_structure", "tutorials",
    ),
}

_DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"


def embedding_model_name() -> str:
    return os.environ.get("CFD_SCIENTIST_EMBEDDING_MODEL", _DEFAULT_EMBEDDING_MODEL)


def index_root() -> Optional[Path]:
    """Directory holding ``<sanitised-model-name>/<index-name>/``.

    Checked in order: an explicit override, a repo-local copy, then the
    vendored Foam-Agent location. The last one is why nothing breaks today —
    but it is a fallback, not a requirement: drop the indices in
    ``<repo>/database/faiss/`` (or point ``CFD_SCIENTIST_FAISS_DIR`` anywhere)
    and no Foam-Agent checkout is involved at all.
    """
    override = os.environ.get("CFD_SCIENTIST_FAISS_DIR")
    candidates = [Path(override)] if override else []
    candidates.append(_REPO_ROOT / "database" / "faiss")
    candidates.append(_REPO_ROOT / "Foam-Agent" / "database" / "faiss")
    for path in candidates:
        if path.is_dir():
            return path
    return None


def tokenize(text: str) -> str:
    """Split camelCase and underscores, then lowercase.

    Must stay byte-identical to the function the indices were *built* with:
    queries are embedded the same way the documents were, and changing this
    silently degrades every match instead of failing.
    """
    text = text.replace("_", " ")
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
    return text.lower()


@lru_cache(maxsize=1)
def _embeddings() -> Any:
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=embedding_model_name())


@lru_cache(maxsize=len(DB_NAMES))
def _load(db_name: str) -> Any:
    from langchain_community.vectorstores import FAISS

    root = index_root()
    if root is None:
        raise FileNotFoundError(
            "No FAISS index directory found. Set CFD_SCIENTIST_FAISS_DIR, or place the "
            f"prebuilt indices under {_REPO_ROOT / 'database' / 'faiss'}."
        )
    path = root / embedding_model_name().replace("/", "_").replace(":", "_") / db_name
    if not path.is_dir():
        raise FileNotFoundError(f"FAISS index not found: {path}")
    return FAISS.load_local(str(path), _embeddings(), allow_dangerous_deserialization=True)


def retrieve(db_name: str, query: str, topk: int = 1) -> List[Dict[str, Any]]:
    """Top-``topk`` matches, in FoamAgent's own result shape."""
    if db_name not in _FIELDS:
        raise ValueError(f"Unknown database name: {db_name}")

    store = _load(db_name)
    tokenized = tokenize(query)
    try:
        pairs = store.similarity_search_with_score(tokenized, k=topk)
    except Exception:
        # Some index builds carry no scores; the documents are still usable
        # and are what the caller actually reads.
        pairs = [(doc, None) for doc in store.similarity_search(tokenized, k=topk)]

    results = []
    for doc, score in pairs:
        metadata = doc.metadata or {}
        entry: Dict[str, Any] = {"index": doc.page_content}
        for field in _FIELDS[db_name]:
            entry[field] = metadata.get(field, "unknown")
        entry["score"] = score
        results.append(entry)
    return results
