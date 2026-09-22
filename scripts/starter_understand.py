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
          binary extensions: .so .o .a .pyc .exe .bin .pkl .pickle .pth
Data files (.npy .npz .h5 .hdf5 .mat .parquet) are described by structure only --
array names, shapes, dtypes, table columns -- read from their headers, never by
value. CSV/TSV files additionally get a row and column count.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
_MAX_FILE_CHARS = 10000  # per-file character cap shown to the model
_HARD_READ_CAP = 4_000_000  # memory guard when slurping a file off disk
_MAX_TOTAL_CHARS = 60000  # total cap across all files sent to LLM

# Data files are described by STRUCTURE, never by value.
#
# The reader used to drop every binary file, so a study whose data lives in
# arrays -- a surrogate benchmark, a field dataset -- reached the planning
# stages as a folder of prose with no data in it. Measured on the NASA CRM
# surrogate case: 5.8 GB of .npy/.npz/.h5 holding the entire problem, and the
# reader saw none of it; the planning steps knew the shapes only because the
# task brief happened to spell them out. A starter whose data is not described
# in prose would leave them blind.
#
# Only headers are read -- the .npy header, the zip member headers of an .npz,
# HDF5 and Parquet metadata, MATLAB's variable table -- so shapes and names are
# known while no value is ever loaded. That is deliberate: a starter may ship
# held-out targets alongside training data, and a structural summary must not
# become a way to see them.
_DATA_SUFFIXES = {".npy", ".npz", ".h5", ".hdf5", ".mat", ".parquet"}
_TABLE_SUFFIXES = {".csv", ".tsv"}
_MAX_DATA_FILES_PER_GROUP = 12    # per (folder, extension); the rest are listed by name, not described
_MAX_ARRAYS_PER_FILE = 40         # arrays / datasets / columns listed per file
_MAX_DATA_SUMMARY_CHARS = 12000   # separate budget, so text files cannot crowd data out


# ---------------------------------------------------------------------------
# File collection
# ---------------------------------------------------------------------------

def _should_skip_dir(d: str) -> bool:
    """True if directory name matches any skip pattern."""
    dl = d.lower()
    return dl in _SKIP_DIRS or dl.startswith("processor")


def _truncate_for_display(text: str, limit: int) -> str:
    """Shorten to ``limit`` on a line boundary, and say what was withheld.

    A hard character cut lands mid-value, and a data file ending in a
    half-written number reads as a corrupt file rather than a shortened one.
    Measured on ph_gemma_20260910e: the DNS reference is 767 complete lines,
    the model was shown the first 6,000 characters, and the last line it saw
    was "1.382812500000000000e+". It concluded the reference data was
    truncated and its RMSE would come out NaN, went hunting for an intact
    copy, and spent the rest of the study on a problem that did not exist.
    The file was never the issue; this reader was.
    """
    if len(text) <= limit:
        return text
    head = text[:limit]
    cut = head.rfind("\n")
    if cut > 0:
        head = head[:cut]
    shown = head.count("\n") + 1
    total = text.count("\n") + 1
    return head + (
        f"\n[... truncated for display: showing {shown} of {total} lines "
        f"({len(text)} characters total). THE FILE ON DISK IS COMPLETE AND INTACT — "
        "read it directly, or point a script at it, to use all of it.]"
    )


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
            return "\n".join(pages)
        except Exception:
            return "[PDF — could not extract text]"
    # Generic text read.
    try:
        # Read whole, truncate once, in _collect_files. Truncating here as
        # well meant the second pass measured the first cut, so the marker
        # reported "197 lines" for a 767-line file -- a truncation notice that
        # lies about the size is barely better than a silent cut.
        raw = path.read_text(encoding="utf-8", errors="ignore")
        return raw[:_HARD_READ_CAP]
    except Exception:
        return None


def _npy_header(fileobj: Any) -> Tuple[tuple, str]:
    """Shape and dtype from a .npy header. Reads the header bytes only."""
    import numpy as np

    fmt = np.lib.format
    version = fmt.read_magic(fileobj)
    if version == (1, 0):
        shape, _fortran, dtype = fmt.read_array_header_1_0(fileobj)
    elif version == (2, 0):
        shape, _fortran, dtype = fmt.read_array_header_2_0(fileobj)
    else:
        shape, _fortran, dtype = fmt._read_array_header(fileobj, version)  # noqa: SLF001
    return tuple(int(n) for n in shape), str(dtype)


def _describe_data_file(path: Path) -> str:
    """What a data file holds, from its header. Never its values."""
    sfx = path.suffix.lower()
    try:
        if sfx == ".npy":
            with open(path, "rb") as f:
                shape, dtype = _npy_header(f)
            return f"numpy array: shape {shape}, dtype {dtype}"

        if sfx == ".npz":
            import zipfile

            lines: List[str] = []
            with zipfile.ZipFile(path) as z:
                members = [n for n in z.namelist() if n.endswith(".npy")]
                for name in members[:_MAX_ARRAYS_PER_FILE]:
                    with z.open(name) as f:
                        shape, dtype = _npy_header(f)
                    lines.append(f"  {name[:-4]}: shape {shape}, dtype {dtype}")
            extra = len(members) - _MAX_ARRAYS_PER_FILE
            head = f"numpy archive with {len(members)} array(s):"
            return "\n".join([head, *lines] + ([f"  ... {extra} more arrays"] if extra > 0 else []))

        if sfx in (".h5", ".hdf5"):
            try:
                import h5py  # type: ignore
            except ImportError:
                return "HDF5 file (structure not readable here: h5py is not installed)"
            lines = []
            seen = [0]
            with h5py.File(path, "r") as h:
                def _visit(name: str, obj: Any) -> Optional[bool]:
                    if isinstance(obj, h5py.Dataset):
                        seen[0] += 1
                        if len(lines) < _MAX_ARRAYS_PER_FILE:
                            lines.append(f"  {name}: shape {tuple(obj.shape)}, dtype {obj.dtype}")
                        elif seen[0] > 10000:
                            return True  # stop walking a file with a huge number of datasets
                    return None
                stopped = h.visititems(_visit)
            count = f"at least {seen[0]}" if stopped else str(seen[0])
            return "\n".join([f"HDF5 file with {count} dataset(s):", *lines])

        if sfx == ".mat":
            from scipy.io import whosmat  # type: ignore

            items = whosmat(str(path))
            lines = [f"  {name}: shape {tuple(shape)}, class {cls}" for name, shape, cls in items[:_MAX_ARRAYS_PER_FILE]]
            return "\n".join([f"MATLAB file with {len(items)} variable(s):", *lines])

        if sfx == ".parquet":
            try:
                import pyarrow.parquet as pq  # type: ignore
            except ImportError:
                return "Parquet table (schema not readable here: pyarrow is not installed)"
            meta = pq.read_metadata(str(path))
            schema = meta.schema.to_arrow_schema()
            cols = [f"{field.name} ({field.type})" for field in schema][:_MAX_ARRAYS_PER_FILE]
            return f"Parquet table: {meta.num_rows} rows x {len(schema)} columns; columns: {', '.join(cols)}"
    except Exception as exc:  # noqa: BLE001
        return f"data file; its structure could not be read ({type(exc).__name__}: {str(exc)[:120]})"
    return "data file"


def _table_shape(path: Path) -> Optional[str]:
    """Row and column count of a delimited text table, plus its column names."""
    import csv

    try:
        delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
        with open(path, "r", encoding="utf-8", errors="ignore", newline="") as f:
            header = next(csv.reader(f, delimiter=delimiter), None)
        if not header:
            return None
        lines = 0
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                lines += block.count(b"\n")
        cols = [c.strip() for c in header]
        shown = ", ".join(cols[:_MAX_ARRAYS_PER_FILE]) + (
            f", ... {len(cols) - _MAX_ARRAYS_PER_FILE} more" if len(cols) > _MAX_ARRAYS_PER_FILE else "")
        return f"[table: about {max(lines - 1, 0)} data rows x {len(cols)} columns; first line: {shown}]"
    except Exception:  # noqa: BLE001
        return None


def _collect_data_summaries(starter_dir: Path) -> List[Dict[str, Any]]:
    """Structural descriptions of every data file in the starter, within budget.

    At most a few files per (folder, extension) are described one by one; the
    rest of a folder full of same-kind arrays is reported as a count, so a
    dataset stored as thousands of per-sample files cannot flood the prompt.
    """
    from collections import OrderedDict

    described: List[Dict[str, Any]] = []
    per_group: Dict[tuple, int] = {}
    not_described: "OrderedDict[tuple, List[str]]" = OrderedDict()
    total = 0
    for fp in sorted(starter_dir.rglob("*")):
        if not fp.is_file() or fp.suffix.lower() not in _DATA_SUFFIXES:
            continue
        if any(_should_skip_dir(part) for part in fp.parts):
            continue
        rel = fp.relative_to(starter_dir)
        key = (str(rel.parent), fp.suffix.lower())
        if per_group.get(key, 0) >= _MAX_DATA_FILES_PER_GROUP or total >= _MAX_DATA_SUMMARY_CHARS:
            not_described.setdefault(key, []).append(fp.name)
            continue
        try:
            size = fp.stat().st_size
        except OSError:
            size = 0
        text = (f"[data file, {size} bytes — structure read from the file header; "
                f"no values were read]\n{_describe_data_file(fp)}")[:2000]
        described.append({"rel_path": str(rel), "content": text})
        total += len(text)
        per_group[key] = per_group.get(key, 0) + 1
    # Named, not just counted: sorted order can put the files that matter most
    # past the cap -- on the CRM starter, test_*.npy sorted ahead of
    # train_*.npy and the training arrays became "3 more files".
    for (parent, sfx), names in not_described.items():
        shown = ", ".join(names[:30]) + (f", ... {len(names) - 30} more" if len(names) > 30 else "")
        described.append({"rel_path": f"{parent}/*{sfx}",
                          "content": f"[{len(names)} more {sfx} file(s) in this folder, not described individually: {shown}]"})
    return described


def _collect_files(starter_dir: Path) -> List[Dict[str, Any]]:
    """
    Walk the starter directory, collect every readable file (skipping binary
    blobs and excluded directories), and return a list of
    {"rel_path": str, "content": str} dicts.
    """
    entries: List[Dict[str, Any]] = []
    total_chars = 0

    def _looks_like_time_dir(name: str) -> bool:
        try:
            float(name)
            return True
        except ValueError:
            return False

    # Read order, not just alphabetical order, and breadth before depth.
    #
    # The walk used to be a plain `sorted(rglob("*"))` against a 60,000-char
    # budget. On a solved case that spends the whole budget inside one
    # subdirectory: measured on starter_oed_turbulence, 86 files of which 57
    # are the same handful of OpenFOAM fields repeated across 0/ 1000/ 2000/
    # 3000/ 4000/ 5000/. Because "periodic_hill_sa" sorts before
    # "reference_data", the budget was gone before the two files that define
    # the objective were ever opened, so the model was never shown the DNS
    # reference or the comparator and invented plausible filenames instead
    # (re_2800_cf.dat, re_5600_cf.dat). Those resolved to nothing, the
    # objective contract came out with zero reference_files, the baseline
    # could not be scored, and setup failed. Three different models failed
    # identically on the same starter: it is the read order, not the model.
    #
    # Now: repeated time snapshots go last (one is plenty to characterise the
    # fields), and the rest are interleaved round-robin across directories, so
    # every directory is represented before any directory gets a second file.
    # Deliberately no naming conventions -- nothing here looks for a folder
    # called "reference_data" or a file called "*_cf.csv". A starter that puts
    # its reference data anywhere, under any name, is seen just the same.
    from collections import OrderedDict

    def _bucket(fp: Path) -> tuple:
        # Grouped by TOP-LEVEL area of the starter, not by immediate parent.
        # Interleaving at every directory dilutes to nothing on a starter with
        # many cases: starter_closure_challenge has 4,975 files across
        # hundreds of directories, so a per-parent rotation still never
        # reaches reference_data/ within the budget. Rotating over the handful
        # of top-level areas instead gives each an equal share, which is the
        # granularity that actually distinguishes "the cases" from "the thing
        # that says how to score them".
        try:
            rel = fp.relative_to(starter_dir)
            top = rel.parts[0] if len(rel.parts) > 1 else ""
        except ValueError:
            top = str(fp.parent)
        return (0, top)

    by_dir: "OrderedDict[tuple, List[Path]]" = OrderedDict()
    def _is_solved_time(fp: Path) -> bool:
        """Inside a non-zero time directory, e.g. case/2000/U.

        Time 0 is kept: it carries the boundary and initial conditions, which
        are what actually characterise a case. Every later time holds the same
        field names with solved numbers in them and describes the case no
        better -- on starter_oed_turbulence that is 0/ repeated five times over
        as 1000/ 2000/ 3000/ 4000/ 5000/, 47 files of numeric noise competing
        for a 60,000-character budget against the two files that define the
        objective.
        """
        for part in fp.parts[:-1]:
            try:
                if float(part) != 0.0:
                    return True
            except ValueError:
                continue
        return False

    for fp in sorted(starter_dir.rglob("*")):
        if not fp.is_file():
            continue
        if any(_should_skip_dir(part) for part in fp.parts):
            continue
        if _is_solved_time(fp):
            continue
        by_dir.setdefault(_bucket(fp), []).append(fp)

    ordered: List[Path] = []
    for tier in (0, 1):
        groups = [v for k, v in sorted(by_dir.items()) if k[0] == tier]
        while any(groups):
            for g in groups:
                if g:
                    ordered.append(g.pop(0))

    for fp in ordered:
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

        budget = min(_MAX_FILE_CHARS, _MAX_TOTAL_CHARS - total_chars)
        chunk = _truncate_for_display(content, budget)
        if fp.suffix.lower() in _TABLE_SUFFIXES:
            shape_line = _table_shape(fp)
            if shape_line:
                chunk = shape_line + "\n" + chunk
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

Some entries, in a section of their own, describe DATA FILES by structure only: \
array names, shapes, dtypes, table columns and row counts, read from the file \
headers. Their values were deliberately not read. Use them to understand how the \
data is organised.

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
  "data_layout": "<how the data is organised: what each data file holds (inputs, targets, geometry, splits), its shape, and how inputs map to outputs; null if there are no data files>",
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
    data_entries = _collect_data_summaries(starter_dir)
    if not entries and not data_entries:
        return {"status": "empty_starter_dir", "starter_dir": str(starter_dir)}

    # Build the user message: topic + each file with its path and content.
    parts = [f"Study topic: {topic}\n\nFiles in starter folder ({len(entries)} readable files):\n"]
    for e in entries:
        parts.append(f"\n=== {e['rel_path']} ===\n{e['content']}")
    # Text is capped before the data section is added, so a starter full of
    # prose can never push the data summaries out of the message.
    text_part = "".join(parts)[:_MAX_TOTAL_CHARS + 2000]  # slight headroom for header
    data_parts = []
    if data_entries:
        data_parts.append(
            f"\n\nDATA FILES — structure only, values not read ({len(data_entries)} entries):\n")
        for e in data_entries:
            data_parts.append(f"\n=== {e['rel_path']} ===\n{e['content']}")
    user_message = text_part + "".join(data_parts)[:_MAX_DATA_SUMMARY_CHARS + 2000]

    try:
        from langchain_core.messages import HumanMessage, SystemMessage  # type: ignore
        from cfd_langgraph.config import get_settings  # type: ignore
        from cfd_langgraph.llm.factory import create_langchain_llm  # type: ignore

        llm = create_langchain_llm(model=get_settings().model, temperature=0.0)
        raw = llm.invoke([SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=user_message)])
        from cfd_langgraph.llm.reply import reply_text
        txt = reply_text(raw).strip()

        # Strip optional markdown fences.
        if txt.startswith("```"):
            txt = txt.split("```", 1)[1].lstrip("json").strip()
            if "```" in txt:
                txt = txt.rsplit("```", 1)[0].strip()

        result = json.loads(txt)
        result["status"] = "ok"
        result["starter_dir"] = str(starter_dir)
        result["files_read"] = len(entries)
        result["data_files_described"] = len(data_entries)
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
