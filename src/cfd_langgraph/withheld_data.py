"""Test-set targets withheld from the agents that build candidates.

A data-fitted study supplies its test inputs and test targets in the starter
folder, often in the same file (an archive holding the test coordinates next to
the test pressures). Its rule is that test scores may steer which candidates are
tried, never anything fitted inside one. That rule used to rest on the agent's
word: on dlr_airfoil_codex_20260913b a candidate ran the scorer on 1,303 blends
and kept, for each test sample, the blend that scored best on it.

So the agents never see the targets. Once per study an LLM reads the starter's
data documentation and names every file that holds test or held-out targets,
with the fields in it that are legal inputs. Code checks each name, writes a copy
of the file holding only those fields, and records the substitution in a
manifest; files whose inputs cannot be separated are hidden. The build sandbox
(scripts/code_mod_agentic.py) mounts the copies over the originals. The
framework's own scoring reads the originals.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

MANIFEST_ENV = "CFD_SCIENTIST_WITHHELD_MANIFEST"
VALIDATION_VIEW_DIR = "validation_view"
_DATA_SUFFIXES = {".npz", ".npy", ".csv", ".dat", ".txt", ".json", ".h5", ".hdf5", ".vtk", ".vtu", ".pt"}
_DOC_NAMES = ("README.md", "README.txt", "TASK.md")

SPEC_SYSTEM = """You protect a machine-learning study's test set. Agents will build \
models from the starter folder below; they may read the training data and the test \
INPUTS, but never the test (or held-out, or out-of-distribution) TARGETS: the values \
the study is scored against.

From the folder's own documentation and the listing of its data files, name every file \
or folder that holds test or held-out target values. For each:
- if the same file also holds inputs a model may read at prediction time, and it is an \
.npz archive or a .csv table, give action "keep_fields" and list exactly the fields \
(archive keys or table columns) that are legal inputs. Everything else in it is withheld.
- if it is an .npz archive whose arrays each hold inputs and targets side by side as \
columns of their last axis (one array per sample, say), give action "mask_columns" and \
list in columns_to_mask the column indices that hold targets. The copy keeps every \
array with those columns replaced by NaN, so column positions do not change.
- otherwise give action "hide": the whole file or folder becomes unreadable. Do this \
when its inputs are also available elsewhere, or cannot be separated from its targets.
Do not list training data. List a combined file that mixes training and test rows as \
"hide" if the training rows are available in a separate file. Use paths exactly as they \
appear in the listing."""


def _summarise(path: Path, root: Path) -> str:
    rel = str(path.relative_to(root))
    try:
        if path.suffix == ".npz":
            import numpy as np

            with np.load(path, allow_pickle=False) as z:
                fields = ", ".join(f"{k}{list(z[k].shape)}" for k in z.files)
            return f"{rel}: npz keys {fields}"
        if path.suffix == ".csv":
            with path.open(newline="") as f:
                header = next(csv.reader(f), [])
            return f"{rel}: csv columns {header}"
    except Exception as exc:  # noqa: BLE001
        return f"{rel}: (could not open: {type(exc).__name__})"
    return f"{rel}: {path.suffix or 'file'}"


def data_listing(root: Path, limit: int = 120) -> str:
    """The starter's data files, one line each, with archive keys and table
    columns; a folder of many similar files is summarised as one line."""
    root = Path(root)
    lines: List[str] = []
    for folder in sorted({p.parent for p in root.rglob("*") if p.is_file() and p.suffix in _DATA_SUFFIXES}):
        files = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix in _DATA_SUFFIXES)
        if len(files) > 8:
            lines.append(f"{folder.relative_to(root)}/ : {len(files)} files, e.g. "
                         + ", ".join(p.name for p in files[:3]))
        else:
            lines.extend(_summarise(p, root) for p in files)
        if len(lines) >= limit:
            break
    return "\n".join(lines[:limit])


def documentation(root: Path, limit: int = 12000, per_doc: int = 6000) -> str:
    docs = []
    for p in sorted(Path(root).rglob("*")):
        if p.is_file() and p.name in _DOC_NAMES:
            docs.append(f"=== {p.relative_to(root)} ===\n{p.read_text(errors='replace')[:per_doc]}")
    return "\n\n".join(docs)[:limit]


def propose_spec(llm: Any, root: Path) -> List[Dict[str, Any]]:
    """The LLM's list of what to withhold. Empty on failure."""
    from langchain_core.messages import HumanMessage, SystemMessage
    from pydantic import BaseModel, Field

    from cfd_langgraph.utils import structured_output

    class _Entry(BaseModel):
        path: str = Field(description="file or folder, relative to the starter folder")
        action: str = Field(description="keep_fields or hide")
        fields_to_keep: List[str] = Field(default_factory=list)
        columns_to_mask: List[int] = Field(default_factory=list)
        reason: str = ""

    class _Spec(BaseModel):
        entries: List[_Entry]

    text = (f"STARTER FOLDER: {root}\n\nDOCUMENTATION:\n{documentation(root)}\n\n"
            f"DATA FILES:\n{data_listing(root)}")
    try:
        out = structured_output(llm, _Spec).invoke(
            [SystemMessage(content=SPEC_SYSTEM), HumanMessage(content=text)])
    except Exception:  # noqa: BLE001
        return []
    return [e.model_dump() for e in out.entries]


def input_only_copy(source: Path, copy: Path, *, keep: Optional[List[str]] = None,
                    mask: Optional[List[int]] = None) -> "tuple[List[str], List[str]]":
    """Write ``copy``: ``source`` with its targets removed, by keeping only the
    named fields (an .npz key or a .csv column) or by setting the named columns of
    every array's last axis to NaN. Returns (kept, withheld); nothing is written
    when nothing would be kept."""
    source, copy = Path(source), Path(copy)
    present: List[str] = []
    withheld: List[str] = []
    copy.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix == ".npz" and mask:
        import numpy as np

        cols = sorted({int(c) for c in mask})
        out = {}
        with np.load(source, allow_pickle=False) as z:
            for k in z.files:
                arr = z[k]
                if arr.ndim >= 1 and arr.shape[-1] > max(cols) and np.issubdtype(arr.dtype, np.floating):
                    arr = arr.copy()
                    arr[..., cols] = np.nan
                    withheld.append(k)
                out[k] = arr
            present = list(z.files)
        if not withheld:
            return [], []
        with copy.open("wb") as f:
            np.savez(f, **out)
        return present, [f"columns {cols} of {len(withheld)} arrays"]
    if source.suffix == ".npz" and keep:
        import numpy as np

        with np.load(source, allow_pickle=False) as z:
            present = [k for k in keep if k in z.files]
            withheld = [k for k in z.files if k not in present]
            if present:
                with copy.open("wb") as f:
                    np.savez(f, **{k: z[k] for k in present})
        return present, withheld
    if source.suffix == ".csv" and keep:
        with source.open(newline="") as f:
            rows = list(csv.reader(f))
        header = rows[0] if rows else []
        present = [k for k in keep if k in header]
        withheld = [k for k in header if k not in present]
        if present:
            idx = [header.index(k) for k in present]
            with copy.open("w", newline="") as f:
                csv.writer(f).writerows([[r[i] for i in idx] for r in rows if len(r) == len(header)])
        return present, withheld
    return [], []


def build_view(root: Path, spec: List[Dict[str, Any]], view_dir: Path) -> Dict[str, Any]:
    """Write the input-only copies and the manifest the sandbox reads. A field
    that does not exist in its file, or a file that cannot be split, is hidden
    instead: withholding too much costs a candidate, too little costs the study."""
    root, view_dir = Path(root).resolve(), Path(view_dir)
    view_dir.mkdir(parents=True, exist_ok=True)
    empty = view_dir / "withheld_empty"
    empty.write_bytes(b"")
    entries: List[Dict[str, Any]] = []
    for item in spec:
        target = (root / str(item.get("path", "")).strip().lstrip("/")).resolve()
        if not str(target).startswith(str(root)) or not target.exists() or target == root:
            continue
        keep = [str(k) for k in item.get("fields_to_keep") or []]
        mask = [int(c) for c in item.get("columns_to_mask") or [] if str(c).lstrip("-").isdigit() and int(c) >= 0]
        record: Dict[str, Any] = {"path": str(target), "reason": str(item.get("reason", ""))[:300]}
        action = item.get("action")
        if target.is_file() and ((action == "keep_fields" and keep and target.suffix in (".npz", ".csv"))
                                 or (action == "mask_columns" and mask and target.suffix == ".npz")):
            copy = view_dir / target.relative_to(root)
            present, withheld = input_only_copy(
                target, copy, keep=keep if action == "keep_fields" else None,
                mask=mask if action == "mask_columns" else None)
            if present:
                entries.append({**record, "action": "replace", "source": str(copy),
                                "kept": present, "withheld": withheld,
                                **({"masked_columns": sorted(set(mask))} if action == "mask_columns" else {})})
                continue
        entries.append({**record, "action": "hide", "is_dir": target.is_dir()})
    manifest = {"starter_root": str(root), "empty_file": str(empty), "entries": entries}
    (view_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def view_command(cmd: List[str], manifest_path: Optional[str]) -> List[str]:
    """``cmd`` run with a manifest's replacement files mounted over the originals
    and nothing else changed: how the framework runs a study's scorer against the
    validation copy of its data (cfd_langgraph.validation_split). Unchanged when
    there is no manifest."""
    import shutil

    manifest = load_manifest(manifest_path)
    binds: List[str] = []
    for entry in manifest.get("entries") or []:
        if entry.get("action") == "replace" and Path(entry.get("source", "")).is_file():
            binds += ["--ro-bind", entry["source"], entry["path"]]
    bwrap = shutil.which("bwrap")
    if not binds:
        return list(cmd)
    if not bwrap:
        raise RuntimeError("bubblewrap is required to score against the validation copy")
    return [bwrap, "--dev-bind", "/", "/", *binds, "--", *cmd]


def find_manifest(run_dir: Optional[Path] = None) -> Optional[str]:
    """This study's manifest: named in the environment, or else found in the
    study folder above ``run_dir``. The folder lookup does not depend on how a
    build process was launched: on dlr_airfoil_codex_20261008 the launcher's
    environment was captured before the manifest existed, the variable never
    reached two candidates, and both scored themselves on the test set."""
    import os

    named = os.environ.get(MANIFEST_ENV)
    if named and Path(named).is_file():
        return named
    if run_dir is not None:
        for folder in (Path(run_dir), *Path(run_dir).resolve().parents):
            # During the search the evaluation sets candidates see are the
            # validation stand-ins; the test manifest alone applies only where
            # no validation view was built.
            for name in (VALIDATION_VIEW_DIR, "withheld_test_targets"):
                found = folder / name / "manifest.json"
                if found.is_file():
                    return str(found)
    return None


def load_manifest(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}
