"""A study's file ledger: every file in the study, who wrote it, when, what it
is about, and where it is now.

The harness used to find the files others wrote by guessing their names and
places: logs named log.*, reports ending .json at most three folders deep,
nothing under processor*/ or previous_attempts/. Each guess missed something
a run wrote and the harness then could not show it. A training run's
failures.jsonl was never read; a script's own log.txt beside log.* files was
never shown; the export's checks of a trained network lived only in a manager
tool result, and an S809 case that was to require them refused ten runs
running (nn_airfoil_s814_20261026_1537, case_006); a folder renumbered into
previous_attempts/ left every path to it stale.

The ledger replaces the guesses. The harness, not the model, records each file
as it appears: a case run's writes by comparing the case folder before and
after the run, the harness's own writes (writer, reviewer, setup check, given
files) the same way around each step, a manager tool's writes around each
call, and every folder move. Whoever needs a file looks it up here, by who
wrote it, when, what it is about, or the path it once had.

Layout: <study>/.ledger/files.jsonl, one JSON record per line:
    {"at": ISO-8601 UTC, "by": "case_006 run 3", "event": "created",
     "source": "run", "path": "cases/case_006/results.json", "kind": "record",
     "size": 41644, "sha256": "..."}
events: created, changed, deleted, moved (with "from"), noted (an explicit
record, with "about": [paths]). OpenFOAM field data (files without a suffix in
time, processor or polyMesh folders) is counted, not listed: one "fields"
record per folder with "files" and "bytes".
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Optional, Tuple

LEDGER_DIR = ".ledger"
_FILE = "files.jsonl"
_HASH_LIMIT = 64 * 1024 * 1024
_LOCK = threading.Lock()
_RUNNING: Dict[str, int] = {}  # case folders whose run is in progress, and how many runs

Snapshot = Dict[str, Tuple[int, int, int]]  # relative path -> (size, mtime_ns, ctime_ns)

_LOG_SUFFIXES = {".log", ".out", ".err"}
_RECORD_SUFFIXES = {".json", ".jsonl", ".csv", ".dat", ".txt", ".tsv", ".xy", ".npz", ".npy", ".h5",
                    ".hdf5", ".yaml", ".yml", ".toml", ".md", ".tex"}
_SCRIPT_SUFFIXES = {".py", ".sh", ".bash", ".m", ".jl", ".c", ".cpp", ".C", ".H", ".h"}
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".svg", ".pdf", ".gif", ".vtk", ".vtp", ".vtu", ".foam"}


def _is_number(name: str) -> bool:
    try:
        float(name)
        return True
    except ValueError:
        return False


def is_field(rel: str) -> bool:
    """OpenFOAM field or mesh data: a file without a suffix (or compressed)
    inside a time folder, a processor folder or polyMesh."""
    parts = PurePosixPath(rel).parts
    suffix = PurePosixPath(rel).suffix
    if suffix not in ("", ".gz") or len(parts) < 2:
        return False
    return any(_is_number(p) or p.startswith("processor") or p == "polyMesh" for p in parts[:-1])


def kind(rel: str) -> str:
    """What a file is, from its name and place: field, log, record, script,
    dictionary, image or other."""
    path = PurePosixPath(rel)
    if is_field(rel):
        return "field"
    if path.name.startswith("log.") or path.suffix in _LOG_SUFFIXES:
        return "log"
    if path.suffix in _RECORD_SUFFIXES:
        return "record"
    if path.suffix in _SCRIPT_SUFFIXES or path.name.startswith("Allrun") or path.name.startswith("Allclean"):
        return "script"
    if path.suffix in _IMAGE_SUFFIXES:
        return "image"
    if path.suffix == "" and set(path.parts[:-1]) & {"system", "constant", "0", "0.orig"}:
        return "dictionary"
    return "other"


def ensure(root: Path) -> Path:
    """The study's ledger folder, made if missing."""
    folder = Path(root) / LEDGER_DIR
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def root_of(path: Path) -> Optional[Path]:
    """The study folder holding ``path``: the nearest folder above it with a ledger."""
    path = Path(path).resolve()
    for folder in (path, *path.parents):
        if (folder / LEDGER_DIR).is_dir():
            return folder
    return None


def snapshot(folder: Path, skip: Iterable[str] = ()) -> Snapshot:
    """Size and times of every file under ``folder`` (links not followed),
    leaving out the ledger and the relative folders and files in ``skip``."""
    folder = Path(folder)
    skipped = {LEDGER_DIR, *skip}
    found: Snapshot = {}
    if not folder.is_dir():
        return found
    for top, dirs, files in os.walk(folder):
        rel_top = os.path.relpath(top, folder)
        rel_top = "" if rel_top == "." else rel_top
        dirs[:] = [d for d in dirs if os.path.join(rel_top, d) not in skipped and d != LEDGER_DIR]
        for name in files:
            if os.path.join(rel_top, name) in skipped:
                continue
            full = os.path.join(top, name)
            try:
                st = os.lstat(full)
            except OSError:
                continue
            found[os.path.join(rel_top, name)] = (st.st_size, st.st_mtime_ns, st.st_ctime_ns)
    return found


def running(folder: Path, on: bool) -> None:
    """Mark a case folder's run as started (``on``) or ended. A manager tool's
    record leaves out folders being run: their files are the run's."""
    key = str(Path(folder).resolve())
    with _LOCK:
        _RUNNING[key] = _RUNNING.get(key, 0) + (1 if on else -1)
        if _RUNNING[key] <= 0:
            _RUNNING.pop(key)


def running_under(root: Path) -> List[str]:
    """The folders under ``root`` (relative to it) whose run is in progress."""
    root_key = str(Path(root).resolve())
    with _LOCK:
        keys = list(_RUNNING)
    return [os.path.relpath(k, root_key) for k in keys if k.startswith(root_key + os.sep)]


def changes(before: Snapshot, after: Snapshot) -> Dict[str, List[str]]:
    """The files created, changed and deleted between two snapshots."""
    return {
        "created": sorted(p for p in after if p not in before),
        "changed": sorted(p for p in after if p in before and after[p] != before[p]),
        "deleted": sorted(p for p in before if p not in after),
    }


def _sha256(path: Path) -> Optional[str]:
    try:
        if path.stat().st_size > _HASH_LIMIT:
            return None
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                digest.update(block)
        return digest.hexdigest()
    except OSError:
        return None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append(root: Path, records: List[Dict[str, Any]]) -> None:
    if not records:
        return
    text = "".join(json.dumps(r, sort_keys=True) + "\n" for r in records)
    with _LOCK:
        with open(ensure(root) / _FILE, "a", encoding="utf-8") as f:
            f.write(text)


def rel(root: Path, path: Path) -> str:
    """``path`` relative to the study folder (absolute if outside it)."""
    path = Path(path)
    if not path.is_absolute():
        path = Path(root) / path
    try:
        return str(path.resolve().relative_to(Path(root).resolve()))
    except ValueError:
        return str(path.resolve())


def record_changes(root: Path, folder: Path, before: Snapshot, after: Snapshot, by: str,
                   source: str = "") -> Dict[str, Any]:
    """Record what changed in ``folder`` between the two snapshots as written by
    ``by`` (``source``: "run" for a case's own scripts and solvers, "harness"
    for the framework's steps, "tool" for a manager tool), and return it: created/changed/deleted (study-relative paths, field
    data left out) and fields ({folder: count} of field files written)."""
    root, folder = Path(root), Path(folder)
    diff = changes(before, after)
    at = _now()
    base = rel(root, folder)
    inside = base not in ("", ".")
    records: List[Dict[str, Any]] = []
    fields: Dict[str, List[int]] = {}
    listed: Dict[str, List[str]] = {"created": [], "changed": [], "deleted": []}
    for event in ("created", "changed", "deleted"):
        for name in diff[event]:
            path = str(PurePosixPath(base) / name) if inside else name
            if is_field(name):
                if event != "deleted":
                    top = PurePosixPath(name).parts[0]
                    count = fields.setdefault(str(PurePosixPath(base) / top) if inside else top, [0, 0])
                    count[0] += 1
                    count[1] += after[name][0]
                continue
            listed[event].append(path)
            record: Dict[str, Any] = {"at": at, "by": by, "source": source, "event": event, "path": path,
                                      "kind": kind(name)}
            if event != "deleted":
                record["size"] = after[name][0]
                digest = _sha256(folder / name)
                if digest:
                    record["sha256"] = digest
            records.append(record)
    records += [{"at": at, "by": by, "source": source, "event": "fields", "path": top, "kind": "field", "files": n, "bytes": size}
                for top, (n, size) in sorted(fields.items())]
    _append(root, records)
    return {**listed, "fields": {top: n for top, (n, _) in fields.items()}}


def note(root: Path, path: Path, by: str, about: Iterable[Path] = (), what: str = "") -> None:
    """Record a file the caller wrote knowingly, with the files it is about
    (a network's check record is about its card)."""
    root = Path(root)
    full = Path(path) if Path(path).is_absolute() else root / path
    where = rel(root, full)
    record: Dict[str, Any] = {"at": _now(), "by": by, "event": "noted", "path": where, "kind": kind(where)}
    if full.is_file():
        record["size"] = full.stat().st_size
        digest = _sha256(full)
        if digest:
            record["sha256"] = digest
    about_rel = [rel(root, Path(a)) for a in about]
    if about_rel:
        record["about"] = about_rel
    if what:
        record["what"] = what
    _append(root, [record])


def moved(root: Path, src: Path, dst: Path, by: str) -> None:
    """Record that a file or folder moved from ``src`` to ``dst``."""
    root = Path(root)
    _append(root, [{"at": _now(), "by": by, "event": "moved", "from": rel(root, src), "path": rel(root, dst)}])


_PARSED: Dict[str, Tuple[Tuple[int, int], List[Dict[str, Any]]]] = {}


def entries(root: Path) -> List[Dict[str, Any]]:
    """Every record of the study's ledger, oldest first. Parsed again only
    when the file has changed: a review reads it half a dozen times."""
    path = Path(root) / LEDGER_DIR / _FILE
    try:
        st = path.stat()
    except OSError:
        return []
    key, stamp = str(path), (st.st_size, st.st_mtime_ns)
    cached = _PARSED.get(key)
    if cached and cached[0] == stamp:
        return cached[1]
    out = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    _PARSED[key] = (stamp, out)
    return out


def current_path(root: Path, path: Path) -> Optional[Path]:
    """Where a file that was at ``path`` is now: itself if it is still there,
    else followed through the recorded moves of it or a folder holding it."""
    root = Path(root)
    full = Path(path) if Path(path).is_absolute() else root / path
    if full.exists():
        return full
    at = rel(root, full)
    moves = [e for e in entries(root) if e.get("event") == "moved"]
    for _ in range(50):  # a file moved again and again, never in a loop
        for move in reversed(moves):
            src, dst = move.get("from", ""), move.get("path", "")
            if at == src or at.startswith(src.rstrip("/") + "/"):
                at = dst + at[len(src):]
                break
        else:
            return None
        found = root / at
        if found.exists():
            return found
    return None


def about(root: Path, path: Path) -> List[Path]:
    """The files recorded as being about ``path`` that still exist."""
    root = Path(root)
    target = rel(root, Path(path) if Path(path).is_absolute() else root / path)
    out: List[Path] = []
    for entry in entries(root):
        if target in (entry.get("about") or []):
            found = current_path(root, Path(entry["path"]))
            if found is not None and found.is_file() and found not in out:
                out.append(found)
    return out


def written_by(root: Path, by_prefix: str) -> List[Dict[str, Any]]:
    """The records whose writer starts with ``by_prefix`` ("case_006 run 3",
    "case_006", "manager"), oldest first."""
    return [e for e in entries(root) if str(e.get("by", "")).startswith(by_prefix)]


def run_writes(root: Path, folder: Path, since: float) -> Optional[List[str]]:
    """The files under ``folder`` that a run recorded at or after ``since``
    (epoch seconds) created or changed and that are still there, relative to
    ``folder``, newest record last; None when no run of it is recorded since."""
    root = Path(root)
    base = rel(root, folder).rstrip("/") + "/"
    start = datetime.fromtimestamp(since, timezone.utc).isoformat()
    found: Dict[str, None] = {}
    recorded = False
    for entry in entries(root):
        if entry.get("source") != "run" or str(entry.get("at", "")) < start:
            continue
        path = str(entry.get("path", ""))
        if not path.startswith(base):
            continue
        recorded = True
        if entry.get("event") in ("created", "changed") and (root / path).is_file():
            found.pop(path[len(base):], None)
            found[path[len(base):]] = None
    return list(found) if recorded else None


_KEEP_WHOLE = 20 * 1024 * 1024
_KEEP_HEAD, _KEEP_TAIL = 1024 * 1024, 5 * 1024 * 1024


def keep_run_logs(root: Path, case_dir: Path, label: str) -> List[Path]:
    """Copy a case's top-level logs (log.*, Allrun.out) as a run left them to
    <study>/.ledger/runs/<case>/<label>/, whole up to 20 MB, else their first
    1 MB and last 5 MB. The next run overwrites them: an S814 run that had
    finished all 21 of its conditions was judged failed, and its solver log,
    the one record of why, was gone when the second run began
    (nn_airfoil_s814_20261026_1537, case_004)."""
    root, case_dir = Path(root), Path(case_dir)
    dest = root / LEDGER_DIR / "runs" / rel(root, case_dir) / label
    kept: List[Path] = []
    for src in sorted(case_dir.glob("log.*")) + [case_dir / "Allrun.out"]:
        if not src.is_file():
            continue
        try:
            dest.mkdir(parents=True, exist_ok=True)
            out = dest / src.name
            size = src.stat().st_size
            with open(src, "rb") as f, open(out, "wb") as g:
                if size <= _KEEP_WHOLE:
                    g.write(f.read())
                else:
                    g.write(f.read(_KEEP_HEAD))
                    g.write(f"\n... [{size - _KEEP_HEAD - _KEEP_TAIL} bytes of the middle left out] ...\n".encode())
                    f.seek(size - _KEEP_TAIL)
                    g.write(f.read())
            kept.append(out)
        except OSError:
            continue
    for out in kept:
        note(root, out, f"{rel(root, case_dir)}: {label}", about=[case_dir / out.name],
             what="the log as this run left it")
    return kept
