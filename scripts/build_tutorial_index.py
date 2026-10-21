#!/usr/bin/env python3
"""Build the case writer's tutorial indices from the installed OpenFOAM's own tutorials.

  python scripts/build_tutorial_index.py [--openfoam <WM_PROJECT_DIR>] [--out <dir>]

The prebuilt indices the case writer retrieves from (cfd_langgraph.foam_native
.faiss_index) were built from openfoam.org 10's tutorials, so on any other
version they hand the writer another version's file names and solver settings.
This builds the same three indices -- tutorial details, directory structure,
Allrun scripts -- in the same document format, from the installation the study
runs on, under database/faiss_versions/<fork>_<version>/, with an
openfoam_version.json naming the version. faiss_index.index_root prefers the
index matching the installed version when there is one.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from cfd_langgraph.foam_native.faiss_index import embedding_model_name, tokenize, versioned_index_dir  # noqa: E402
from cfd_langgraph.openfoam_version import _strip_comments, case_solver, get_profile  # noqa: E402

_INPUT_DIRS = ("0", "0.orig", "system", "constant")
_MAX_FILE_BYTES = 60_000


def _cases(tutorials: Path) -> List[Path]:
    out = []
    for root, dirs, files in os.walk(tutorials):
        if os.path.basename(root) == "system" and "controlDict" in files:
            out.append(Path(root).parent)
            dirs[:] = []
    return sorted(set(out))


def _record(case: Path, tutorials: Path, profile: Dict[str, Any]) -> Dict[str, str]:
    rel = case.relative_to(tutorials).parts
    solver = case_solver(case, profile)
    name = solver["module"] or solver["application"] or "None"
    domain = rel[0] if rel else "None"
    category = rel[1] if len(rel) > 2 else "None"
    index = (f"<index>\ncase name: {case.name}\ncase domain: {domain}\n"
             f"case category: {category}\ncase solver: {name}\n</index>")
    dirs, tutorial = [], []
    for sub in _INPUT_DIRS:
        folder = case / sub
        if not folder.is_dir():
            continue
        files = sorted(p for p in folder.iterdir() if p.is_file() and p.stat().st_size < _MAX_FILE_BYTES)
        if not files:
            continue
        dirs.append(f"<dir>directory name: {sub}. File names in this directory: "
                    f"[{', '.join(p.name for p in files)}]</dir>")
        parts = [f"<directory_begin>directory name: {sub}"]
        for p in files:
            try:
                text = _strip_comments(p.read_text(encoding="utf-8")).strip()
            except (UnicodeDecodeError, OSError):
                continue
            parts.append(f"<file_begin>file name: {p.name}\n<file_content>\n{text}\n</file_content>\n</file_end>")
        parts.append("</directory_end>")
        tutorial.append("\n\n".join(parts))
    allrun = case / "Allrun"
    return {
        "index": index,
        "structure": "<directory_structure>\n" + "\n".join(dirs) + "\n</directory_structure>",
        "tutorials": "\n\n".join(tutorial),
        "allrun": allrun.read_text(errors="replace") if allrun.is_file() else "",
        "case_name": case.name, "case_domain": domain, "case_category": category,
        "case_solver": name, "dir_structure": "\n".join(dirs),
    }


def build(openfoam_path: str, out: Path) -> Dict[str, Any]:
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document
    from langchain_huggingface import HuggingFaceEmbeddings

    profile = get_profile(openfoam_path)
    if not profile.get("ok"):
        raise SystemExit(f"could not read the OpenFOAM installation at {openfoam_path!r}")
    tutorials = Path(profile["tutorials_dir"])
    records = [_record(c, tutorials, profile) for c in _cases(tutorials)]

    def meta(r: Dict[str, str]) -> Dict[str, str]:
        return {k: r[k] for k in ("case_name", "case_domain", "case_category", "case_solver", "dir_structure")}

    docs = {
        "openfoam_tutorials_details": [Document(
            page_content=tokenize(r["index"] + "\n" + r["structure"]),
            metadata={**meta(r), "tutorials": r["tutorials"],
                      "full_content": f"{r['index']}\n\n{r['structure']}\n\n<tutorials>\n{r['tutorials']}\n</tutorials>"})
            for r in records],
        "openfoam_tutorials_structure": [Document(
            page_content=tokenize(r["index"]),
            metadata={**meta(r), "full_content": f"{r['index']}\n\n{r['structure']}"}) for r in records],
        "openfoam_allrun_scripts": [Document(
            page_content=tokenize(r["index"] + r["structure"]),
            metadata={**meta(r), "allrun_script": r["allrun"],
                      "full_content": f"{r['index']}\n\n{r['structure']}\n\n<allrun_script>\n{r['allrun']}\n</allrun_script>"})
            for r in records if r["allrun"]],
    }
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name())
    model_dir = out / embedding_model_name().replace("/", "_").replace(":", "_")
    for name, documents in docs.items():
        FAISS.from_documents(documents, embeddings).save_local(str(model_dir / name))
        print(f"{name}: {len(documents)} cases", flush=True)
    stamp = {"fork": profile["fork"], "version": profile["version"], "label": profile["label"],
             "cases": len(records), "tutorials_dir": str(tutorials)}
    (out / "openfoam_version.json").write_text(json.dumps(stamp, indent=2))
    return stamp


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--openfoam", default="", help="WM_PROJECT_DIR (default: the configured installation)")
    ap.add_argument("--out", default="", help="default: database/faiss_versions/<fork>_<version>")
    a = ap.parse_args()
    profile = get_profile(a.openfoam)
    out = Path(a.out) if a.out else versioned_index_dir(profile)
    out.mkdir(parents=True, exist_ok=True)
    print(json.dumps(build(a.openfoam or profile.get("project_dir", ""), out), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
