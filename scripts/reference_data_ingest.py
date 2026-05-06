#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

from timeline_logger import append_timeline_event, resolve_timeline_path

TEXT_EXTS = {".txt", ".md", ".rst", ".log", ".dat", ".csv", ".tsv", ".json", ".yaml", ".yml"}
TABULAR_EXTS = {".csv", ".tsv"}
SPREADSHEET_EXTS = {".xlsx", ".xls"}
ARRAY_EXTS = {".npy", ".npz"}
MAT_EXTS = {".mat"}
HDF_EXTS = {".h5", ".hdf5"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif"}
PDF_EXTS = {".pdf"}


def _is_case_path(p: Path) -> bool:
    parts = {x.lower() for x in p.parts}
    return bool(parts & {"0", "constant", "system", "polymesh", "postprocessing"})


def _safe_text_sample(path: Path, max_chars: int = 2000) -> str:
    try:
        txt = path.read_text(encoding="utf-8", errors="ignore")
        return txt[:max_chars]
    except Exception:
        return ""


def _read_csv_like(path: Path, delimiter: str, max_rows: int = 12) -> Dict[str, Any]:
    info: Dict[str, Any] = {"columns": [], "sample_rows": [], "row_count_estimate": 0}
    try:
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
            reader = csv.reader(f, delimiter=delimiter)
            rows = []
            for i, row in enumerate(reader):
                if i >= max_rows:
                    break
                rows.append(row)
        if rows:
            info["columns"] = rows[0]
            info["sample_rows"] = rows[1:max_rows]
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            info["row_count_estimate"] = sum(1 for _ in f)
    except Exception as exc:
        info["error"] = str(exc)
    return info


def _read_json_brief(path: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {"top_level_type": "unknown", "keys": [], "sample": None}
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        if isinstance(data, dict):
            out["top_level_type"] = "object"
            out["keys"] = list(data.keys())[:30]
            sample: Dict[str, Any] = {}
            for k in list(data.keys())[:10]:
                v = data[k]
                sample[k] = type(v).__name__
            out["sample"] = sample
        elif isinstance(data, list):
            out["top_level_type"] = "array"
            out["sample"] = data[:3]
        else:
            out["top_level_type"] = type(data).__name__
            out["sample"] = str(data)[:300]
    except Exception as exc:
        out["error"] = str(exc)
    return out


def _infer_metric_hints(text: str) -> List[str]:
    """
    Placeholder — metric hint extraction is done by the LLM in _llm_interpret_reference_data.
    Returns empty list so the heuristic scoring path has no effect.
    """
    return []


def _analyze_file(path: Path) -> Dict[str, Any]:
    ext = path.suffix.lower()
    stat = path.stat()
    mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    rec: Dict[str, Any] = {
        "path": str(path),
        "name": path.name,
        "suffix": ext,
        "bytes": stat.st_size,
        "mime": mime,
        "kind": "unknown",
        "parser_status": "unparsed",
        "metadata": {},
    }
    if ext in TABULAR_EXTS:
        rec["kind"] = "tabular"
        rec["metadata"] = _read_csv_like(path, "," if ext == ".csv" else "\t")
        rec["parser_status"] = "ok" if "error" not in rec["metadata"] else "error"
        return rec
    if ext == ".json":
        rec["kind"] = "structured"
        rec["metadata"] = _read_json_brief(path)
        rec["parser_status"] = "ok" if "error" not in rec["metadata"] else "error"
        return rec
    if ext in {".yaml", ".yml", ".txt", ".md", ".rst", ".log", ".dat"}:
        rec["kind"] = "text"
        sample = _safe_text_sample(path)
        rec["metadata"] = {
            "sample": sample,
            "metric_hints": _infer_metric_hints(sample),
            "line_count_estimate": sample.count("\n") + 1 if sample else 0,
        }
        rec["parser_status"] = "ok"
        return rec
    if ext in PDF_EXTS:
        rec["kind"] = "document"
        meta: Dict[str, Any] = {}
        try:
            from pypdf import PdfReader  # type: ignore
            reader = PdfReader(str(path))
            meta["pages"] = len(reader.pages)
            if reader.pages:
                page0 = reader.pages[0]
                txt = (page0.extract_text() or "")[:1500]
                meta["sample"] = txt
                meta["metric_hints"] = _infer_metric_hints(txt)
            rec["parser_status"] = "ok"
        except Exception as exc:
            meta["error"] = str(exc)
            rec["parser_status"] = "error"
        rec["metadata"] = meta
        return rec
    if ext in IMAGE_EXTS:
        rec["kind"] = "image"
        meta: Dict[str, Any] = {}
        try:
            from PIL import Image  # type: ignore
            with Image.open(path) as im:
                meta["size"] = {"width": int(im.width), "height": int(im.height)}
                meta["mode"] = str(im.mode)
            rec["parser_status"] = "ok"
        except Exception as exc:
            meta["error"] = str(exc)
            rec["parser_status"] = "error"
        rec["metadata"] = meta
        return rec
    if ext in SPREADSHEET_EXTS:
        rec["kind"] = "tabular"
        meta: Dict[str, Any] = {}
        try:
            import pandas as pd  # type: ignore
            xls = pd.ExcelFile(path)
            meta["sheet_names"] = list(xls.sheet_names)[:20]
            if xls.sheet_names:
                df = pd.read_excel(path, sheet_name=xls.sheet_names[0], nrows=10)
                meta["columns"] = [str(c) for c in df.columns.tolist()]
                meta["sample_rows"] = df.head(5).to_dict(orient="records")
            rec["parser_status"] = "ok"
        except Exception as exc:
            meta["error"] = str(exc)
            rec["parser_status"] = "error"
        rec["metadata"] = meta
        return rec
    if ext in ARRAY_EXTS:
        rec["kind"] = "array"
        meta: Dict[str, Any] = {}
        try:
            import numpy as np  # type: ignore
            if ext == ".npy":
                arr = np.load(path, allow_pickle=False, mmap_mode="r")
                meta["shape"] = list(arr.shape)
                meta["dtype"] = str(arr.dtype)
            else:
                data = np.load(path, allow_pickle=False)
                meta["keys"] = list(data.files)[:30]
            rec["parser_status"] = "ok"
        except Exception as exc:
            meta["error"] = str(exc)
            rec["parser_status"] = "error"
        rec["metadata"] = meta
        return rec
    if ext in HDF_EXTS:
        rec["kind"] = "array"
        meta: Dict[str, Any] = {}
        try:
            import h5py  # type: ignore
            keys: List[str] = []
            with h5py.File(path, "r") as f:
                f.visit(lambda name: keys.append(name) if len(keys) < 100 else None)
            meta["keys"] = keys
            rec["parser_status"] = "ok"
        except Exception as exc:
            meta["error"] = str(exc)
            rec["parser_status"] = "error"
        rec["metadata"] = meta
        return rec
    if ext in MAT_EXTS:
        rec["kind"] = "array"
        meta: Dict[str, Any] = {}
        try:
            from scipy.io import loadmat  # type: ignore
            data = loadmat(path)
            keys = [k for k in data.keys() if not k.startswith("__")]
            meta["keys"] = keys[:40]
            rec["parser_status"] = "ok"
        except Exception as exc:
            meta["error"] = str(exc)
            rec["parser_status"] = "error"
        rec["metadata"] = meta
        return rec
    if ext in TEXT_EXTS:
        rec["kind"] = "text"
        sample = _safe_text_sample(path)
        rec["metadata"] = {"sample": sample, "metric_hints": _infer_metric_hints(sample)}
        rec["parser_status"] = "ok"
        return rec
    return rec


def _select_candidates(records: List[Dict[str, Any]], topic: str) -> List[Dict[str, Any]]:
    """
    Pass all non-image, non-binary records to the LLM for interpretation.
    No heuristic scoring — the LLM decides what is relevant given the topic.
    """
    skip_kinds = {"image"}
    skip_exts = IMAGE_EXTS | {".pdf"}
    candidates = [
        r for r in records
        if r.get("kind") not in skip_kinds
        and Path(r.get("path", "")).suffix.lower() not in skip_exts
    ]
    return candidates[:80]


def _llm_interpret_reference_data(
    starter_dir: Path,
    records: List[Dict[str, Any]],
    topic: str,
) -> Dict[str, Any]:
    """
    Ask the LLM to read every candidate reference file (any format: CSV, Python,
    text, etc.), understand what data is available, and produce a plain-English
    description plus raw content excerpt.  No format assumptions are made — the LLM
    figures it out from the actual file content.
    """
    _SKIP_DIRS = {"polyMesh", "postProcessing", "dynamicCode"}
    _MAX_FILE_CHARS = 8000
    _MAX_TOTAL_CHARS = 50000

    # Gather content of all non-binary files in the reference-data area.
    blobs: List[str] = []
    total = 0
    for rec in records:
        if total >= _MAX_TOTAL_CHARS:
            break
        path = Path(rec.get("path", ""))
        if not path.exists() or any(d in _SKIP_DIRS for d in path.parts):
            continue
        if rec.get("kind") in {"image", "array"} or rec.get("suffix", "").lower() in {
            ".png", ".jpg", ".jpeg", ".gif", ".webp", ".npy", ".npz", ".h5", ".hdf5", ".mat",
        }:
            continue
        try:
            raw = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        chunk = raw[:_MAX_FILE_CHARS]
        rel = path.relative_to(starter_dir) if path.is_relative_to(starter_dir) else path
        blob = f"\n\n=== {rel} ===\n{chunk}"
        blobs.append(blob)
        total += len(blob)

    if not blobs:
        return {"status": "no_readable_files"}

    files_text = "".join(blobs)[:_MAX_TOTAL_CHARS]

    system_prompt = (
        "You are a CFD data analyst. Given the contents of a reference-data folder "
        "(which may contain CSV files, Python scripts, text documents, or any other format), "
        "produce a comprehensive description of:\n"
        "1. What reference/experimental/DNS/LES data is available.\n"
        "2. What physical quantities (Cf, Cp, U, xr/h, etc.) are tabulated.\n"
        "3. How a CFD analysis agent should use this data to validate simulation results "
        "(including any units, normalisations, or coordinate conventions).\n"
        "4. Provide a verbatim excerpt of the most important data rows so the analysis "
        "agent can directly compare simulation values.\n\n"
        "Return valid JSON with keys:\n"
        '{"description": "...", "quantities": ["Cf", ...], "usage_guidance": "...", '
        '"data_excerpt": "...verbatim rows...", "files_found": ["filename", ...]}'
    )
    user_prompt = (
        f"Study topic: {topic}\n\n"
        "Reference data folder contents:\n"
        + files_text
    )

    try:
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from cfd_langgraph.llm.factory import create_langchain_llm  # type: ignore
        from cfd_langgraph.config import get_settings  # type: ignore
        from langchain_core.messages import HumanMessage, SystemMessage  # type: ignore

        llm = create_langchain_llm(model=get_settings().model, temperature=0.0)
        raw_resp = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
        txt = str(getattr(raw_resp, "content", raw_resp)).strip()
        if txt.startswith("```"):
            txt = txt.split("```")[1].lstrip("json").strip()
            txt = txt.rsplit("```", 1)[0].strip()
        result = json.loads(txt)
        print(f"[ref_ingest] LLM identified quantities: {result.get('quantities', [])}")
        return result
    except Exception as exc:
        print(f"[ref_ingest] warning: LLM reference interpretation failed: {exc}", file=sys.stderr)
        return {"status": "llm_failed", "error": str(exc)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Generic starter/reference data ingestion manifest.")
    parser.add_argument("--starter-dir", default="", type=str,
                        help="Path to starter folder; if absent or empty an empty manifest is written.")
    parser.add_argument("--topic", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--timeline", type=str, default="")
    parser.add_argument("--starter-understanding", default="", type=str,
                        help="Path to starter_understanding.json; its reference_data block "
                             "is merged into the manifest as llm_interpretation.")
    args = parser.parse_args()

    output = Path(args.output).expanduser().resolve()
    timeline_path = resolve_timeline_path(args.timeline)

    # Resolve starter_dir; produce an empty manifest when it does not exist.
    starter_dir_str = (args.starter_dir or "").strip()
    starter_dir = Path(starter_dir_str).expanduser().resolve() if starter_dir_str else None
    if starter_dir is None or not starter_dir.is_dir():
        print(f"[ref_ingest] starter_dir not found ({starter_dir_str!r}); writing empty manifest.")
        empty_manifest: Dict[str, Any] = {
            "starter_dir": starter_dir_str,
            "topic": args.topic,
            "scan_summary": {"total_files_seen": 0, "files_analyzed": 0, "candidate_reference_files": 0},
            "records": [],
            "selected_reference_candidates": [],
            "llm_interpretation": {},
            "loader_guidance": "No starter folder provided; reference data unavailable.",
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(empty_manifest, indent=2), encoding="utf-8")
        append_timeline_event(
            timeline_path,
            {"stage": "reference_data_ingest", "event": "skipped_no_starter_dir",
             "output": str(output)},
        )
        return 0

    files = [p for p in starter_dir.rglob("*") if p.is_file()]
    records: List[Dict[str, Any]] = []
    for p in files:
        if _is_case_path(p):
            continue
        records.append(_analyze_file(p))

    candidates = _select_candidates(records, args.topic)

    # Use starter_understanding.json if available (already contains full LLM analysis
    # of the reference data from the unified folder scan).  Only run the standalone
    # LLM interpretation if no prior understanding is cached.
    llm_interpretation: Dict[str, Any] = {}
    if args.starter_understanding:
        su_p = Path(args.starter_understanding)
        if su_p.exists():
            try:
                su = json.loads(su_p.read_text(encoding="utf-8"))
                ref_block = su.get("reference_data", {})
                if ref_block and isinstance(ref_block, dict):
                    llm_interpretation = ref_block
                    llm_interpretation["source"] = "starter_understanding"
                    print(f"[ref_ingest] reference_data from starter_understanding.json: "
                          f"quantities={ref_block.get('quantities')}")
            except Exception as e:
                print(f"[ref_ingest] warning: could not read starter_understanding: {e}", file=sys.stderr)

    if not llm_interpretation:
        # LLM reads every reference file (any format) and produces a structured
        # interpretation — quantities, usage guidance, verbatim data excerpt.
        llm_interpretation = _llm_interpret_reference_data(starter_dir, records, args.topic)

    manifest = {
        "starter_dir": str(starter_dir),
        "topic": args.topic,
        "scan_summary": {
            "total_files_seen": len(files),
            "files_analyzed": len(records),
            "candidate_reference_files": len(candidates),
        },
        "records": records,
        "selected_reference_candidates": candidates,
        "llm_interpretation": llm_interpretation,
        "loader_guidance": (
            "Use llm_interpretation for a plain-English description and data excerpt. "
            "Use selected_reference_candidates for raw file paths and metadata. "
            "The LLM has already read and understood the data regardless of format."
        ),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    append_timeline_event(
        timeline_path,
        {
            "stage": "reference_data_ingest",
            "event": "completed",
            "starter_dir": str(starter_dir),
            "output": str(output),
            "files_analyzed": len(records),
            "selected_candidates": len(candidates),
            "llm_quantities": llm_interpretation.get("quantities", []),
        },
    )
    print(json.dumps(manifest["scan_summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
