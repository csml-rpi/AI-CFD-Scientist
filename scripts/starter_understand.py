#!/usr/bin/env python3
"""
Unified starter-folder understanding via a single LLM call.

Reads every file in the starter directory (any format — OpenFOAM case files,
text/formula specs, CSVs, dat files, Python scripts, PDFs, images, etc.),
passes each one with its path to the LLM, and asks it to classify and extract:

  - base_case_path      : which sub-directory is the OpenFOAM case
  - formula_or_model_spec: the full equation/model change to implement
  - flow_parameters     : Re, nu, Ub, dimension, geometry, …
  - reference_data      : what validation data is available, quantities, excerpt
  - file_classifications: per-file role labels

The result is written to <run_dir>/starter_understanding.json and consumed by:
  - orchestrator_run.py  → flow_parameters (suppress Re clarification question)
  - code_mod_prepare.py  → formula_or_model_spec
  - reference_data_ingest.py → reference_data

Skipped:  polyMesh/, postProcessing/, dynamicCode/, processor*/, .git/
          binary extensions: .so .o .a .pyc .exe .bin .npy .npz .h5 .hdf5 .mat
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_SKIP_DIRS = {
    "polyMesh", "postProcessing", "dynamicCode", "processor",
    ".git", "__pycache__", ".venv", "node_modules",
}
_BINARY_SUFFIXES = {
    ".so", ".o", ".a", ".pyc", ".exe", ".bin",
    ".npy", ".npz", ".h5", ".hdf5", ".mat",
    ".pkl", ".pickle", ".pth",
}
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".tif", ".tiff", ".bmp"}
_MAX_FILE_CHARS = 6000   # per-file character cap
_MAX_TOTAL_CHARS = 60000  # total cap across all files sent to LLM


# ---------------------------------------------------------------------------
# File collection
# ---------------------------------------------------------------------------

def _should_skip_dir(d: str) -> bool:
    """True if directory name matches any skip pattern."""
    dl = d.lower()
    return dl in _SKIP_DIRS or dl.startswith("processor")


def _read_file_content(path: Path) -> Optional[str]:
    """Return text content of a file, or None if it is binary/unreadable."""
    sfx = path.suffix.lower()
    if sfx in _BINARY_SUFFIXES:
        return None
    if sfx in _IMAGE_SUFFIXES:
        # Describe image by path only — no binary content.
        return f"[image file — {path.stat().st_size} bytes]"
    # Try PDF text extraction.
    if sfx == ".pdf":
        try:
            from pypdf import PdfReader  # type: ignore
            reader = PdfReader(str(path))
            pages = [p.extract_text() or "" for p in reader.pages[:8]]
            return "\n".join(pages)[:_MAX_FILE_CHARS]
        except Exception:
            return "[PDF — could not extract text]"
    # Generic text read.
    try:
        raw = path.read_text(encoding="utf-8", errors="ignore")
        return raw[:_MAX_FILE_CHARS]
    except Exception:
        return None


def _collect_files(starter_dir: Path) -> List[Dict[str, Any]]:
    """
    Walk the starter directory, collect every readable file (skipping binary
    blobs and excluded directories), and return a list of
    {"rel_path": str, "content": str} dicts.
    """
    entries: List[Dict[str, Any]] = []
    total_chars = 0

    for fp in sorted(starter_dir.rglob("*")):
        if not fp.is_file():
            continue
        # Skip excluded directories anywhere in the path.
        if any(_should_skip_dir(part) for part in fp.parts):
            continue
        if total_chars >= _MAX_TOTAL_CHARS:
            break

        content = _read_file_content(fp)
        if content is None:
            continue

        try:
            rel = str(fp.relative_to(starter_dir))
        except ValueError:
            rel = str(fp)

        chunk = content[:min(_MAX_FILE_CHARS, _MAX_TOTAL_CHARS - total_chars)]
        entries.append({"rel_path": rel, "content": chunk})
        total_chars += len(chunk)

    return entries


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a CFD scientist assistant. You will be given the contents of a starter \
folder for a CFD study — it may contain an OpenFOAM case directory, a formula \
or model specification text file, reference/experimental data (CSV, dat, Python \
scripts that read data, etc.), PDFs, and other supporting files.

Your job is to read EVERY file provided and produce a single JSON object that \
classifies and extracts the key information. Do not guess — base your answers \
only on the file contents shown.

Return ONLY valid JSON (no markdown, no commentary) with this exact structure:

{
  "base_case_path": "<relative path to the OpenFOAM case directory, or null>",
  "formula_or_model_spec": "<full verbatim text of the model equation / modification \
to implement — copy the relevant lines exactly as written in the file>",
  "formula_file": "<filename that contains the formula, or null>",
  "flow_parameters": {
    "Re": <number or null>,
    "nu": <number or null>,
    "Ub": <number or null>,
    "dimension": "2D" or "3D" or null,
    "geometry": "<brief description of the flow domain, e.g. periodic hill, backward step, pipe, channel>"
  },
  "reference_data": {
    "description": "<what reference/DNS/LES/experimental data is available>",
    "quantities": ["Cf", "xr/h", "U", ...],
    "data_excerpt": "<verbatim key data rows from the reference file>",
    "usage_guidance": "<how CFD results should be compared with this data>",
    "files": ["<filenames that contain reference data>"]
  },
  "file_classifications": {
    "<rel_path>": "base_case" | "formula_spec" | "reference_data" | "literature" | "other"
  },
  "notes": "<any important observation about the starter folder>"
}
"""


def understand_starter_folder(
    starter_dir: Path,
    topic: str,
) -> Dict[str, Any]:
    """
    Main entry point.  Scans the starter folder, passes everything to the LLM,
    and returns the structured understanding dict.
    """
    entries = _collect_files(starter_dir)
    if not entries:
        return {"status": "empty_starter_dir", "starter_dir": str(starter_dir)}

    # Build the user message: topic + each file with its path and content.
    parts = [f"Study topic: {topic}\n\nFiles in starter folder ({len(entries)} readable files):\n"]
    for e in entries:
        parts.append(f"\n=== {e['rel_path']} ===\n{e['content']}")
    user_message = "".join(parts)[:_MAX_TOTAL_CHARS + 2000]  # slight headroom for header

    try:
        from langchain_core.messages import HumanMessage, SystemMessage  # type: ignore
        from cfd_langgraph.config import get_settings  # type: ignore
        from cfd_langgraph.llm.factory import create_langchain_llm  # type: ignore

        llm = create_langchain_llm(model=get_settings().model, temperature=0.0)
        raw = llm.invoke([SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=user_message)])
        txt = str(getattr(raw, "content", raw)).strip()

        # Strip optional markdown fences.
        if txt.startswith("```"):
            txt = txt.split("```", 1)[1].lstrip("json").strip()
            if "```" in txt:
                txt = txt.rsplit("```", 1)[0].strip()

        result = json.loads(txt)
        result["status"] = "ok"
        result["starter_dir"] = str(starter_dir)
        result["files_read"] = len(entries)
        print(f"[starter_understand] LLM classified {len(entries)} files.")
        print(f"[starter_understand] base_case_path : {result.get('base_case_path')}")
        print(f"[starter_understand] formula_file   : {result.get('formula_file')}")
        print(f"[starter_understand] flow_parameters: {result.get('flow_parameters')}")
        print(f"[starter_understand] ref quantities : {result.get('reference_data', {}).get('quantities')}")
        return result

    except json.JSONDecodeError as exc:
        print(f"[starter_understand] warning: LLM returned non-JSON: {exc}", file=sys.stderr)
        return {"status": "json_parse_error", "error": str(exc), "raw": txt[:500]}
    except Exception as exc:
        print(f"[starter_understand] warning: LLM call failed: {exc}", file=sys.stderr)
        return {"status": "llm_failed", "error": str(exc)}


# ---------------------------------------------------------------------------
# CLI entry point (called from orchestrator or standalone)
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Understand the starter folder via a single LLM call."
    )
    parser.add_argument("--starter-dir", required=True, type=str)
    parser.add_argument("--topic", required=True, type=str)
    parser.add_argument("--output", required=True, type=str,
                        help="Path to write starter_understanding.json")
    args = parser.parse_args()

    starter_dir = Path(args.starter_dir).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()

    result = understand_starter_folder(starter_dir, args.topic)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[starter_understand] written to {output}")
    return 0 if result.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
