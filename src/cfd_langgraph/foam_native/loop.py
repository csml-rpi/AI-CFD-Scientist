from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from . import allrun as allrun_mod
from . import decomposer, parser, rag, review, writer
from .case_clean import parse_time_dir, remove_stale_time_dirs
from .openfoam_env import resolve_openfoam_env  # noqa: F401  (re-exported)
from cfd_langgraph.solver_shelf import run_env as case_env

_CASE_FILE_DIRS = ("0", "constant", "system")


# Characters no case file name holds. A model naming "every field" as 0/*
# had a file literally called "*" written; decomposePar then failed on its
# header ("expected word, found '*'") on every later run of the case
# (shelf_demo_hystrath_20261011_1602, case_002).
_BAD_NAME_CHARS = set("*?[]{}<>|;\"'") | {" ", "\t"}


def _bad_case_file_name(name: str) -> bool:
    return not name or bool(_BAD_NAME_CHARS & set(name))


def _safe_case_relative_path(folder_name: str, file_name: str) -> Path:
    rel = Path(str(folder_name or "")) / str(file_name or "")
    if rel.is_absolute() or ".." in rel.parts or not rel.name:
        raise ValueError(f"Unsafe FoamAgent case path: {rel}")
    if any(_bad_case_file_name(part) for part in rel.parts):
        raise ValueError(f"Not a file name (a pattern, not one file): {rel}")
    return rel


def _without_case_name(case_dir: Path, rel: Path) -> Path:
    """``rel`` without a leading folder named like the case itself. Named
    "case_010/runScript.py", a DAFoam case's files were written into a
    case_010/ folder inside the case; the review then removed the case's own
    runScript.py as the duplicate, and the next six runs found no script
    (shelf_demo_dafoam_20261019_1057, case_010)."""
    parts = Path(rel).parts
    if len(parts) > 1 and parts[0] == Path(case_dir).name:
        return Path(*parts[1:])
    return Path(rel)


# What a solver's run script (DAFoam's runScript.py) may become when it is
# written or edited. Told only to compute what the requirement asks, the model
# grew the 77-line starter script to 1,700-2,600 lines with up to 109 checks of
# its own; those checks were what failed, and one that aborted on a rank other
# than 0 left no error in the log (shelf_demo_dafoam_20261011_1437, cases 001-005).
_RUN_SCRIPT_RULES = (
    "Keep the script's structure and add only what the requirement needs. Do not add "
    "checks of the environment, the folder or the case files: the framework starts it "
    "in the case folder with the solver's environment loaded. Do not catch errors to "
    "abort or exit: let them raise, so the error and its traceback reach the log the "
    "run is reviewed from. Do not write files named run_result.json, setup_check.json, "
    "Allrun or Allrun.out, or any log.* file: the framework writes those, and a script's "
    "own file under one of these names replaces the framework's record of the run. Give "
    "every file the script writes a plain name, not one starting with a dot: its results, "
    "reports and diagnostics are read by name between runs."
)


def _mesh_patches(case_dir: Path) -> List[Tuple[str, str]]:
    """(name, type) of every patch in the case's constant/polyMesh/boundary."""
    try:
        text = (case_dir / "constant" / "polyMesh" / "boundary").read_text(errors="ignore")
    except OSError:
        return []
    while "/*" in text:
        start = text.find("/*")
        end = text.find("*/", start + 2)
        text = text[:start] if end < 0 else text[:start] + text[end + 2:]
    text = "\n".join(line.split("//")[0] for line in text.splitlines())
    for ch in "{}();":
        text = text.replace(ch, f" {ch} ")
    toks = text.split()
    i = 0
    if "FoamFile" in toks:
        i, depth = toks.index("FoamFile") + 1, 0
        while i < len(toks):
            if toks[i] == "{":
                depth += 1
            elif toks[i] == "}":
                depth -= 1
                if depth == 0:
                    i += 1
                    break
            i += 1
    while i < len(toks) and toks[i] != "(":
        i += 1
    i += 1
    patches: List[Tuple[str, str]] = []
    while i < len(toks) and toks[i] != ")":
        name, i = toks[i], i + 1
        if i < len(toks) and toks[i] == "{":
            depth, kind, i = 1, "", i + 1
            while i < len(toks) and depth:
                if toks[i] == "{":
                    depth += 1
                elif toks[i] == "}":
                    depth -= 1
                elif toks[i] == "type" and depth == 1 and i + 1 < len(toks):
                    kind = toks[i + 1]
                i += 1
            patches.append((name, kind))
    return patches


def _patch_note(case_dir: Path, folder_name: str) -> str:
    """For a field file, the mesh's patches it must name. Guessing them, the
    writer left the mesh's `walls` out of DAFoam's betaFINuTilda and decomposePar
    stopped every case (shelf_demo_dafoam_20261011_1437)."""
    if str(folder_name).split("/")[0] not in ("0", "0.org", "0.orig"):
        return ""
    patches = _mesh_patches(case_dir)
    if not patches:
        return ""
    return ("\n\nMESH BOUNDARY: this case's mesh has these patches; a field file needs a "
            "boundaryField entry for every one, under exactly these names: "
            + ", ".join(f"{n} ({k})" if k else n for n, k in patches) + ".")


def _zero_template(case_dir: Path) -> str:
    """The folder an own Allrun rebuilds 0/ from (0.org or 0.orig), if it does."""
    try:
        allrun = (case_dir / "Allrun").read_text(errors="ignore")
    except OSError:
        return ""
    words = [w.strip("\"'") for w in allrun.replace("/", " ").replace('"', " ").split()]
    for name in ("0.org", "0.orig"):
        if (case_dir / name).is_dir() and name in words:
            return name
    return ""


def _compiled_python(llm: Any, file_name: str, folder_name: str, content: str,
                     written_files_ctx: str, case_solver: str, tries: int = 2) -> str:
    """A Python file as written, sent back to the writer with its syntax error
    until it compiles (at most ``tries`` times). Run as written, a 1,466-line
    run script with a SyntaxError on line 1101 cost a whole Allrun and a
    review round per attempt (shelf_demo_dafoam_20261011_1437, case_004)."""
    if not file_name.endswith(".py"):
        return content
    for _ in range(tries):
        try:
            compile(content, file_name, "exec")
            return content
        except SyntaxError as exc:
            lines = content.splitlines()
            line = lines[exc.lineno - 1].strip() if exc.lineno and exc.lineno <= len(lines) else ""
            print(f"[foam-native] {file_name}: does not compile ({exc.msg}, line {exc.lineno}); "
                  "sent back to the writer", flush=True)
            try:
                content = writer.edit_file(
                    llm, file_name=file_name, folder_name=folder_name,
                    changes=(f"This Python file does not compile: {exc.msg} at line {exc.lineno}: "
                             f"`{line}`. Fix that error and any other syntax error; change nothing "
                             "else. Return the complete file."),
                    current_content=content, written_files_ctx=written_files_ctx,
                    case_solver=case_solver,
                )
            except Exception:
                return content
    return content


def _remove_pattern_named_files(case_dir: Path) -> None:
    """Delete files in the case folders named like a pattern ("*"), left by an
    earlier run before such names were refused: OpenFOAM utilities read every
    file in a time folder and stop at one."""
    for sub in _CASE_FILE_DIRS:
        folder = case_dir / sub
        if not folder.is_dir():
            continue
        for p in folder.iterdir():
            if p.is_file() and _bad_case_file_name(p.name):
                p.unlink()
                print(f"[foam-native] {case_dir.name}: removed {sub}/{p.name}, a pattern written as a "
                      "file name", flush=True)


def _json_outline(value: Any, max_lines: int = 150) -> str:
    """The shape of a JSON document, level by level: every key as its full
    path (mesh.cell_centres_m), its type, the length of each list and its
    first value, all top-level keys first, then the next level down. Walked
    depth first, a 289 KB truth file's outline spent its lines on the first
    keys' deep diagnostics and never reached mesh.cell_centres_m, which a
    DAFoam script then failed to find (shelf_demo_dafoam_20261019_1057, case_006)."""
    from collections import deque

    lines: List[str] = []

    def short(v: Any) -> str:
        text = json.dumps(v)
        return text if len(text) <= 80 else text[:77] + "..."

    def kind(v: Any) -> str:
        if isinstance(v, list):
            return f"list[{len(v)}]" + (f" of {kind(v[0])}" if v else "")
        if isinstance(v, dict):
            return "object"
        return {bool: "bool", int: "int", float: "float", str: "string"}.get(type(v), "null")

    queue = deque([("", value)])
    while queue and len(lines) < max_lines:
        path, v = queue.popleft()
        name = path or "(the document)"
        if isinstance(v, dict):
            lines.append(f"{name}: object with {len(v)} keys")
            for k, x in v.items():
                queue.append((f"{path}.{k}" if path else str(k), x))
        elif isinstance(v, list) and v and isinstance(v[0], dict):
            lines.append(f"{name}: list[{len(v)}] of objects; the first is {path}[0]")
            queue.append((f"{path}[0]", v[0]))
        elif isinstance(v, list):
            lines.append(f"{name}: {kind(v)}" + (f", first {short(v[0])}" if v else ""))
        else:
            lines.append(f"{name}: null" if v is None else f"{name}: {kind(v)} {short(v)}")
    if queue:
        lines.append("... (deeper keys not shown)")
    return "\n".join(lines)


def _compact_json(value: Any) -> Any:
    """``value`` with its bulk left out: a list of more than six numbers (or
    of lists of numbers) becomes a note of its length and first value, a
    longer list keeps its first ten items, a long string its first 300
    characters. Keys, checks and messages stay as they are."""
    if isinstance(value, dict):
        return {k: _compact_json(x) for k, x in value.items()}
    if isinstance(value, list):
        if len(value) > 6 and all(isinstance(x, (int, float, list)) and not isinstance(x, bool) for x in value[:6]):
            return f"<{len(value)} values, first {json.dumps(value[0])[:60]}>"
        if len(value) > 12:
            return [_compact_json(x) for x in value[:10]] + [f"<{len(value) - 10} more>"]
        return [_compact_json(x) for x in value]
    if isinstance(value, str) and len(value) > 300:
        return value[:300] + f"...<{len(value)} characters>"
    return value


def _file_view(path: Path, text: str, limit: int) -> str:
    """``text`` as a prompt shows it: whole within ``limit`` characters; a
    longer JSON file as its outline (_json_outline); anything else cut at
    ``limit``. Cut, a 202 KB truth file showed only the numbers of its first
    array, and three DAFoam scripts guessed its layout ("solution.velocity",
    a "metadata" member) and stopped on their own checks, ten runs each
    (shelf_demo_dafoam_20261019_1057, cases 001, 009, 010)."""
    if len(text) <= limit:
        return text
    if path.suffix == ".json":
        try:
            data = json.loads(text)
        except ValueError:
            data = None
        if data is not None:
            # With its values when they fit, else by its keys alone.
            compact = json.dumps(_compact_json(data), separators=(",", ":"))
            if len(compact) <= limit:
                return (f"[{len(text)} characters of JSON, shown with long lists of numbers shortened to "
                        f"their length and first value]\n{compact}")
            return (f"[{len(text)} characters of JSON, shown as its outline: every key with its type, "
                    f"the length of each list and its first value]\n{_json_outline(data)}")
    return text[:limit] + "\n... [truncated]"


def _foamfiles_xml(case_dir: Path, subtasks: List[Dict[str, str]], max_file_kb: int = 30, max_total_kb: int = 300) -> str:
    parts: List[str] = []
    total = 0
    for st in subtasks:
        p = case_dir / st["folder_name"] / st["file_name"]
        if not p.is_file():
            continue
        text = _file_view(p, p.read_text(encoding="utf-8", errors="ignore"), max_file_kb * 1024)
        if total + len(text) > max_total_kb * 1024:
            break
        total += len(text)
        parts.append(
            f"<foamfile><file_name>{st['file_name']}</file_name>"
            f"<folder_name>{st['folder_name']}</folder_name>"
            f"<content>{text}</content></foamfile>"
        )
    return "\n".join(parts)


def _written_files_ctx(case_dir: Path, subtasks: List[Dict[str, str]], upto_index: int) -> str:
    parts: List[str] = []
    for st in subtasks[:upto_index]:
        p = case_dir / st["folder_name"] / st["file_name"]
        if p.is_file():
            parts.append(
                f"{st['folder_name']}/{st['file_name']}:\n"
                + _file_view(p, p.read_text(encoding="utf-8", errors="ignore"), 2000)
            )
    return "\n\n".join(parts)


# checkMesh flags quality complaints with the same "***" it uses for real
# errors. This one fires on any wall-resolved boundary-layer mesh — the
# study's own validated starter case reports max aspect ratio 1750 and still
# converges cleanly — so treating it as fatal would fail every case and burn
# the whole retry budget. It is reported to the reviewer, never fatal.
_BENIGN_MESH_CHECKS = ("high aspect ratio",)


def mesh_check_errors(case_dir: Path) -> List[str]:
    """checkMesh failures that actually invalidate the mesh, or [].

    checkMesh EXITS 0 even when it finds negative-volume cells, so neither the
    Allrun return code nor the "FOAM FATAL" scan notices a broken mesh. The
    solver then runs on it and diverges, and the review loop spends its
    retries rewriting physics files while the fault is in blockMeshDict.

    checkMesh marks errors with a leading '***' and warnings with a single
    '*', but not every '***' is solver-fatal — see _BENIGN_MESH_CHECKS.
    """
    return [
        line for line in _mesh_check_lines(case_dir)
        if not any(benign in line.lower() for benign in _BENIGN_MESH_CHECKS)
    ]


def _mesh_check_lines(case_dir: Path) -> List[str]:
    """Every '***' line checkMesh emitted, fatal or not."""
    log_path = case_dir / "log.checkMesh"
    if not log_path.is_file():
        return []
    try:
        text = log_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []
    return [ln.strip() for ln in text.splitlines() if ln.lstrip().startswith("***")]


# What a run may write that tells why it failed, besides its logs: a run
# script's own report of its checks.
_RUN_OUTPUT_SUFFIXES = (".json", ".txt", ".csv", ".dat", ".md", ".yaml", ".yml")
# Written by the harness, not by the run.
_HARNESS_FILES = ("Allrun.out", "run_result.json", "setup_check.json", ".foamagent_state.json")


def _run_outputs(case_dir: Path, since: float, per_file: int = 12000, total: int = 24000) -> List[str]:
    """The small text files a run wrote into the case (changed at or after
    ``since``), outside its dictionaries, fields, mesh, logs and results
    folders, the latest first. A DAFoam script that stopped itself on a
    failed check wrote which check and why into results.json; the review saw
    only "validation failed" in the log and the file's name, and the case
    failed ten runs in a row (shelf_demo_dafoam_20261019_1057, case_008)."""
    found = []
    for path in case_dir.rglob("*"):
        rel = path.relative_to(case_dir)
        top = rel.parts[0]
        if (len(rel.parts) > 2 or not path.is_file() or path.suffix not in _RUN_OUTPUT_SUFFIXES
                # A hidden report is shown (a DAFoam script's log pointed to
                # .synthetic_truth.coordinator.failure.json for its diagnostics);
                # hidden folders (.openmdao_out, caches) are not.
                or str(rel) in _HARNESS_FILES or any(part.startswith(".") for part in rel.parts[:-1])
                or top in ("0", "constant", "system", "postProcessing", "dynamicCode")
                or top.startswith(("processor", "log.")) or _is_number(top)):
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        if stat.st_mtime >= since:
            found.append((len(rel.parts), -stat.st_mtime, rel, stat.st_size))
    parts: List[str] = []
    left = total
    for _, _, rel, size in sorted(found):
        if left <= 0:
            break
        text = (case_dir / rel).read_text(encoding="utf-8", errors="ignore")
        if "\x00" in text[:4096]:
            continue
        # A JSON report is shown by its checks and values, its arrays shortened
        # (_file_view): cut at 12 KB, a DAFoam truth case's 265 KB report showed
        # its velocity numbers and none of the 31 checks that had stopped it
        # (shelf_demo_dafoam_20261019_1057, case_007).
        if rel.suffix == ".json" and len(text) > per_file:
            shown = _file_view(rel, text, left)
        else:
            shown = text[:min(per_file, left)]
            if len(shown) < len(text):
                shown += f"\n... [first {len(shown)} characters]"
        left -= len(shown)
        parts.append(f"--- {rel}, written by this run ({size} B) ---\n{shown}")
    return parts


def _bounded_tail(lines: List[str], line_chars: int = 2000, total_chars: int = 60_000) -> str:
    """A log's last lines as the reviewer is shown them: each cut at
    ``line_chars``, the whole at its last ``total_chars``. Shown whole, a
    DAFoam script's log holding a 6.6 MB line of JSON made a 15 MB prompt
    that the provider refused (shelf_demo_dafoam_20261019_1057, case_002)."""
    shown = [line if len(line) <= line_chars
             else line[:line_chars] + f" ... [{len(line) - line_chars} characters of this line cut]"
             for line in lines]
    text = "\n".join(shown)
    if len(text) > total_chars:
        text = f"... [{len(text) - total_chars} characters cut]\n" + text[-total_chars:]
    return text


def collect_error_logs(case_dir: Path, max_lines: int = 200, since: Optional[float] = None) -> str:
    """What the reviewer is shown of a failed run: mesh errors, every log's
    last lines, and with ``since`` (when the run started) the small text
    files the run wrote (_run_outputs)."""
    parts: List[str] = []
    mesh_errors = mesh_check_errors(case_dir)
    if mesh_errors:
        parts.append(
            "--- MESH VALIDITY (checkMesh) ---\n"
            "The mesh itself is invalid. Fix system/blockMeshDict; rewriting "
            "boundary conditions, schemes or relaxation factors cannot fix "
            "these, and the solver cannot converge on this mesh:\n"
            + "\n".join(mesh_errors)
        )
    solver = _read_solver_from_control_dict(case_dir)
    if solver and _solver_ran_no_steps(case_dir, solver):
        parts.append(
            f"--- {solver} ran no time steps ---\n"
            f"{_solver_log(case_dir, solver).name} ends normally but the solver never advanced past its start time, so "
            "nothing was solved. Check startTime, endTime, deltaT and stopAt in "
            "system/controlDict: endTime must be later than startTime."
        )
    candidates = sorted(case_dir.glob("log.*")) + [case_dir / "Allrun.out"]
    for log_path in candidates:
        if log_path.is_file():
            lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
            parts.append(f"--- {log_path.name} (last {min(len(lines), max_lines)} lines) ---\n"
                         + _bounded_tail(lines[-max_lines:]))
    if since is not None:
        parts += _run_outputs(case_dir, since)
    return "\n\n".join(parts)



_ERROR_MARKS = ("FOAM FATAL", "Traceback (most recent call last)", "ERROR", "Error:")


def error_summary(case_dir: Path, limit: int = 3000) -> str:
    """What a failed run's record says: the first error in the logs, with the
    lines around it, then the end of Allrun.out. The record used to keep the
    last 3000 characters of every log joined in name order, so a DAFoam case's
    "Unknown RAS model type" in log.DASimpleFoam never reached the manager,
    only the tails of the logs after it (shelf_demo_dafoam_20261011_1437)."""
    logs = sorted((q for q in case_dir.glob("log.*") if q.is_file()), key=lambda q: q.stat().st_mtime)
    parts: List[str] = []
    # Up to three excerpts, bad input and tracebacks first: the first error in
    # that log was a notice DAFoam carries on past ("already decomposed"); the
    # cause was a FOAM FATAL IO ERROR further down.
    found: List[Tuple[int, str, int, List[str]]] = []
    for log_path in logs:
        lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        real_fatal = set(_fatal_blocks(lines))
        for i, line in enumerate(lines):
            if _startup_notice(line):
                continue
            if "FOAM FATAL" in line and i not in real_fatal:
                continue  # a notice the run carries on past
            # A solver crash prints no FOAM error: an olaFlow case that blew up
            # left only "Floating point exception" and a stack, and its record
            # came out empty (shelf_demo_olaflow_20261011_1242, case_006).
            for rank, mark in enumerate(("FOAM FATAL IO ERROR", "Traceback (most recent call last)",
                                         "FOAM FATAL ERROR", "Floating point exception",
                                         "Segmentation fault", "exited on signal", "ERROR", "Error:")):
                if mark in line:
                    found.append((rank, log_path.name, i, lines))
                    break
    found.sort(key=lambda f: (f[0], f[2]))
    taken: List[Tuple[str, int]] = []
    said: List[str] = []
    for _, name, i, lines in found:
        # The same message from every rank ("[2] Unknown RAS model ...") counts once.
        message = " ".join(" ".join(lines[i:i + 3]).replace("[", " [").split())
        message = " ".join(w for w in message.split() if not (w.startswith("[") and w.endswith("]")))
        if len(taken) >= 3 or message in said or any(n == name and abs(i - j) < 30 for n, j in taken):
            continue
        said.append(message)
        taken.append((name, i))
        parts.append(f"--- {name}, line {i + 1} ---\n" + "\n".join(lines[max(0, i - 3):i + 15]))
    out = case_dir / "Allrun.out"
    if out.is_file():
        parts.append("--- Allrun.out (end) ---\n"
                     + "\n".join(out.read_text(encoding="utf-8", errors="ignore").splitlines()[-8:]))
    text = "\n\n".join(parts) or collect_error_logs(case_dir)
    return text[:limit] if len(text) > limit else text


# What a FOAM FATAL notice says when the run it is in carries on: pyDAFoam
# runs decomposePar each time it builds a problem, and a script that builds a
# second one (a gradient check's perturbed solves) gets "Case is already
# decomposed" from the second. Counted as fatal, a DAFoam gradient check that
# matched finite differences to 6e-8 was recorded as failed run after run
# (shelf_demo_dafoam_20261019_1057, case_003).
_NOT_FATAL = ("already decomposed",)


def _fatal_blocks(lines: List[str]) -> List[int]:
    """Line numbers of the FOAM FATAL errors in a log that are real errors."""
    found = []
    for i, line in enumerate(lines):
        if "FOAM FATAL ERROR" in line or "FOAM FATAL IO ERROR" in line:
            block = " ".join(lines[i:i + 6])
            if not any(note in block for note in _NOT_FATAL):
                found.append(i)
    return found


def _has_fatal(case_dir: Path) -> bool:
    for log_path in case_dir.glob("log.*"):
        try:
            text = log_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if ("FOAM FATAL ERROR" in text or "FOAM FATAL IO ERROR" in text) and _fatal_blocks(text.splitlines()):
            return True
    return False


# A run that ends at its start time still prints "End"; recorded as a success, its
# initial fields were later judged as if they were a solution (qwen_retest_20261005/cavity_r3).
def _solver_log(case_dir: Path, solver: str) -> Path:
    """The log the Allrun's run of ``solver`` writes: log.foamRun for a solver
    module (openfoam.org 11+), log.<application> otherwise."""
    from cfd_langgraph.openfoam_version import get_profile, log_name, solver_command

    return Path(case_dir) / (log_name(solver_command(get_profile(), solver)) or f"log.{solver}")


def _default_solver(name: str) -> str:
    """``name``, or the command that superseded it on this installation."""
    from cfd_langgraph.openfoam_version import get_profile, solver_command

    return solver_command(get_profile(), name)


def _solver_ran_no_steps(case_dir: Path, solver: str) -> bool:
    """Whether the solver's log ends normally without having advanced a time step
    (endTime at or before startTime)."""
    log_path = _solver_log(case_dir, solver)
    if not log_path.is_file():
        return False
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    return "\nTime = " not in text and text.rstrip().endswith(("End", "Finalising parallel run",
                                                                _log_end(case_dir)))


def _log_end(case_dir: Path) -> str:
    """The line a finished solver's log ends with: "End", or what a solver on
    its own OpenFOAM declares (dsmcFoam+ ends with "End main")."""
    from cfd_langgraph.solver_shelf import own_runtime

    return own_runtime(case_dir).get("log_end") or "End"


def _solver_ended_cleanly(case_dir: Path, solver: str) -> bool:
    """The solver advanced at least one time step and its log ended normally."""
    log_path = _solver_log(case_dir, solver)
    if not log_path.is_file():
        return False
    if _solver_ran_no_steps(case_dir, solver):
        return False
    tail = log_path.read_text(encoding="utf-8", errors="ignore")[-2000:].rstrip()
    # A parallel run prints "Finalising parallel run" after its "End".
    if tail.endswith("Finalising parallel run"):
        tail = tail[: -len("Finalising parallel run")].rstrip()
    return tail.endswith(_log_end(case_dir))


def _extract_functions_block(control_dict_text: str) -> str:
    """The whole top-level ``functions { ... }`` entry, or "" if absent."""
    match = re.search(r"(?m)^\s*functions\s*$|^\s*functions\s*\{", control_dict_text)
    if not match:
        return ""
    brace = control_dict_text.find("{", match.start())
    if brace < 0:
        return ""
    depth = 0
    for i in range(brace, len(control_dict_text)):
        ch = control_dict_text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "functions\n" + control_dict_text[brace:i + 1] + "\n"
    return ""


def _seed_function_objects(case_dir: Path, seed_case_dir: Path) -> str:
    """Copy the base case's ``functions`` block into a generated case.

    Function objects are how a case produces anything measurable —
    wallShearStress for Cf, yPlus, sampled sets. They are *measurement
    contract*, not physics the case writer should be inventing: a
    requirement saying "to evaluate wall shear stress" is prose, and
    FoamAgent has no reason to translate it into a specific function-object
    entry. When it doesn't, the case runs to completion, writes no
    postProcessing output, and the study only discovers it much later — at
    scoring time, when there is nothing to score and no way to tell a real
    null result from a case that never measured anything.

    Copied verbatim from the starter's controlDict, and only when the
    generated case has no functions block of its own, nor a solver that
    forbids added function objects (cfd_langgraph.solver_shelf.own_runtime).
    """
    from cfd_langgraph.solver_shelf import own_runtime

    target = case_dir / "system" / "controlDict"
    source = seed_case_dir / "system" / "controlDict"
    if not target.is_file() or not source.is_file():
        return ""
    if own_runtime(case_dir).get("function_objects") is False:
        return ""
    target_text = target.read_text(encoding="utf-8", errors="ignore")
    if _extract_functions_block(target_text):
        return ""
    block = _extract_functions_block(source.read_text(encoding="utf-8", errors="ignore"))
    if not block:
        return ""
    marker = "// ****"
    idx = target_text.rfind(marker)
    if idx < 0:
        idx = len(target_text)
    target.write_text(target_text[:idx] + "\n" + block + "\n" + target_text[idx:], encoding="utf-8")
    names = re.findall(r"(?m)^\s{4}(\w+)\s*$", block)
    return ", ".join(names) or "functions"


# Largest file an edit rewrites whole through the model, in characters.
_MAX_EDIT_CHARS = 200 * 1024
# Largest script an edit changes. A script is changed by exact replacements
# (edit_file_by_patches), which put the file in the request but do not ask for
# it back: held to the 200 KB limit meant for per-cell fields, a 219 KB DAFoam
# run script could not be fixed in its last runs, and a setup-check fix refused
# on it left the script half-changed (shelf_demo_dafoam_20261019_1057, case_007).
_MAX_SCRIPT_EDIT_CHARS = 800 * 1024
# Above this, an edit asks for exact replacements instead of the whole file
# back (writer.edit_file_by_patches).
_PATCH_EDIT_CHARS = 12_000


def _field_head(path: Path, size: int = 65536) -> str:
    """The start of a field file, read through gzip when it is compressed
    (DAFoam writes its fields as .gz)."""
    import gzip

    try:
        if path.name.endswith(".gz"):
            with gzip.open(path, "rt", errors="ignore") as f:
                return f.read(size)
        with path.open(errors="ignore") as f:
            return f.read(size)
    except OSError:
        return ""


def _field_length(path: Path) -> Optional[int]:
    """How many values a field's nonuniform internalField lists, or None when
    it is uniform or not a field."""
    head = _field_head(path)
    i = head.find("internalField")
    if i < 0:
        return None
    words = head[i:i + 400].split()
    if len(words) < 3 or words[1] != "nonuniform":
        return None
    for word in words[2:6]:
        count = word.split("(")[0]
        if count.isdigit():
            return int(count)
    return None


def _drop_old_mesh_fields(case_dir: Path, n_cells: Optional[int] = None) -> List[str]:
    """Remove initial-time files holding per-cell or per-face lists -- all of
    them, or with ``n_cells`` only those sized for another mesh.

    After re-meshing, a list sized for the old mesh cannot be read on the new
    one; a solver that needs such a field fails either way, and one that does
    not (function-object output such as yPlus, written at the first time)
    otherwise stops decomposePar, which reads every field in 0/. Compressed
    fields are read too: DAFoam's 343-cell 0/pData.gz passed every check
    unread and stopped decomposePar on an 8000-cell case
    (shelf_demo_dafoam_20261011_1437).
    """
    zero = Path(case_dir) / "0"
    dropped: List[str] = []
    if not zero.is_dir():
        return dropped
    for f in sorted(zero.iterdir()):
        if not f.is_file():
            continue
        length = _field_length(f)
        nonuniform = length is not None or "nonuniform" in _field_head(f)
        if nonuniform and (n_cells is None or (length is not None and length != n_cells)):
            f.unlink()
            dropped.append(f.name)
    # A field set by setFields keeps its uniform template beside it (F.org or
    # F.orig): put that back, for the Allrun's setFields to fill again. Dropped
    # without it, a refined olaFlow level had no alpha.water and failed at
    # decomposePar (shelf_demo_olaflow_20261011_1242).
    for name in dropped:
        stem = name[:-3] if name.endswith(".gz") else name
        for suffix in (".org", ".orig"):
            template = zero / f"{stem}{suffix}"
            if template.is_file() and _field_length(template) is None:
                shutil.copy2(template, zero / stem)
                break
    if dropped:
        print(f"[foam] {Path(case_dir).name}: dropped 0/ fields sized for the old mesh: "
              f"{', '.join(dropped)}", flush=True)
    return dropped


def _clean_stale_run_artifacts(case_dir: Path) -> None:
    """Remove everything a previous ``./Allrun`` attempt left behind that
    would make the next attempt skip work or read someone else's results.

    Four distinct hazards, all of which have bitten this pipeline:

    1. ``log.*`` — every step in an Allrun script runs via OpenFOAM's
       ``runApplication``, which refuses to rerun a step whose log file
       already exists: it prints e.g. ``"blockMesh already run... remove log
       file 'log.blockMesh' to re-run"`` and exits 0. Left in place, a retry
       silently no-ops through every step that got as far as writing a log,
       and — because the *stale* solver log still ends in ``End`` — the run
       is scored as a clean success even though none of the rewritten files
       were ever executed.
    2. ``processor*/`` — ``decomposePar`` without ``-force`` aborts with a
       FOAM FATAL ERROR on an already-decomposed case. Clearing the logs
       alone makes decomposePar rerun and hit exactly that, converting a
       fixable failure into a new unrelated one for the reviewer to chase.
    3. ``postProcessing/`` — this is what the scoring comparators read. A
       partially-rerun case that still carries the previous attempt's
       samples gets scored on results that did not come from the code
       currently in the case directory.
    4. non-zero time directories — every QoI extractor here reads the case
       with PyVista and is instructed to take ``max(time_values)``. A time
       directory left by an earlier solve *on a different mesh* then wins
       that selection, its ``internalField`` length no longer matches
       ``nCells``, and the reader attaches no fields at all. Measured on
       ``codex_sol56_none/cavity_r2``: the baseline carried 100..1000 from a
       4096-cell solve next to the converged t=415 on the current 1024-cell
       mesh, the extractor read t=1000, reported "fields found: none", and
       the mesh gate died after 21 hours on a case that had solved fine.
       See :mod:`cfd_langgraph.foam_native.case_clean`, which leaves genuine
       restarts (``startFrom latestTime``, or a non-zero ``startTime``)
       alone.

    Call this before the first attempt as well as between retries: case
    directories are reused across relaunches (a RERUN case is handed back to
    a subagent with the same ``case_id``, and the mesh gate reuses its
    baseline/refined dirs), so "first attempt" does not imply "clean dir".
    Every call site is positioned before a ``./Allrun``; the retry loops
    break out on success first, so a good result is never cleared.
    """
    for log_path in case_dir.glob("log.*"):
        if log_path.is_file():
            log_path.unlink(missing_ok=True)
    for proc_dir in case_dir.glob("processor*"):
        if proc_dir.is_dir() and not proc_dir.is_symlink():
            shutil.rmtree(proc_dir, ignore_errors=True)
    post = case_dir / "postProcessing"
    if post.is_dir() and not post.is_symlink():
        shutil.rmtree(post, ignore_errors=True)
    # A restart from an earlier time is only valid on the mesh that wrote it.
    # When this case's Allrun regenerates the mesh, those fields no longer
    # fit it, so they go regardless of startFrom.
    allrun = case_dir / "Allrun"
    regenerates_mesh = allrun.is_file() and any(
        tool in allrun.read_text(errors="ignore") for tool in ("blockMesh", "snappyHexMesh"))
    removed_times = remove_stale_time_dirs(case_dir, force=regenerates_mesh)
    if removed_times:
        # Say so: a silent delete of a previous solve's output is exactly the
        # kind of thing that should be visible in a run log when a result
        # later looks unexpected.
        print(
            f"[foam] cleared {len(removed_times)} stale time director"
            f"{'y' if len(removed_times) == 1 else 'ies'} in {case_dir.name}: "
            f"{', '.join(removed_times[:8])}{' ...' if len(removed_times) > 8 else ''}",
            flush=True,
        )


def _newest_change(case_dir: Path) -> float:
    """The latest modification time among a case's entries and the entries of
    its folders, one level down: a growing log, a new time directory or a
    file a script writes into logs/ all move it."""
    latest = 0.0
    try:
        with os.scandir(case_dir) as top:
            for entry in top:
                try:
                    latest = max(latest, entry.stat(follow_symlinks=False).st_mtime)
                    if entry.is_dir(follow_symlinks=False):
                        with os.scandir(entry.path) as inner:
                            for sub in inner:
                                latest = max(latest, sub.stat(follow_symlinks=False).st_mtime)
                except OSError:
                    continue
    except OSError:
        pass
    return latest


def _simulated_progress(case_dir: Path) -> Optional[Tuple[float, float]]:
    """(latest simulated time in the newest solver log, controlDict endTime),
    or None when either is not there to read."""
    end = None
    try:
        for line in (case_dir / "system" / "controlDict").read_text(errors="ignore").splitlines():
            words = line.replace(";", " ").split()
            if len(words) >= 2 and words[0] == "endTime":
                end = float(words[1])
                break
    except (OSError, ValueError):
        return None
    logs = [q for q in case_dir.glob("log.*") if q.is_file()]
    if end is None or not logs:
        return None
    newest = max(logs, key=lambda q: q.stat().st_mtime)
    try:
        with newest.open("rb") as fh:
            fh.seek(max(0, newest.stat().st_size - 65536))
            tail = fh.read().decode(errors="ignore")
    except OSError:
        return None
    for line in reversed(tail.splitlines()):
        words = line.split()
        if len(words) >= 3 and words[0] == "Time" and words[1] == "=":
            try:
                return float(words[2].rstrip("s")), end
            except ValueError:
                return None
    return None


# How run_script says a run was stopped by its time limit, and the least time
# a further run of the case is worth starting with.
# Review rounds after a failed run before the case stops for want of any change.
_REVIEW_ROUNDS = 2
TIME_LIMIT_NOTE = "Stopped: the run used up the case's time limit"
_MIN_RUN_S = 600


def _stall_limit_s(case_dir: Path) -> float:
    """How long a case's run may write nothing before it counts as stuck. 45
    min, the longest quiet stretch measured on a large mesh (a DAFoam adjoint's
    Jacobian colouring on 157k cells writes nothing for about 8 min), but 15 min
    on a mesh under 20,000 cells, whose solves take seconds: two hung
    343-cell DAFoam runs each held their case for the full 45 min
    (shelf_demo_dafoam_20261019_1057)."""
    cells = _n_cells(case_dir)
    return 900.0 if cells is not None and cells < 20000 else 2700.0


_STACK_HOOK = Path(__file__).resolve().parent / "stack_hook"


def _descendants(pid: int) -> List[int]:
    """Every process below ``pid``, read from /proc."""
    children: Dict[int, List[int]] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text()
        except OSError:
            continue
        # "pid (name) state ppid ...": the name may hold spaces, what follows it does not.
        fields = stat.rsplit(")", 1)[-1].split()
        if len(fields) > 1 and fields[1].isdigit():
            children.setdefault(int(fields[1]), []).append(int(entry.name))
    found, todo = [], [pid]
    while todo:
        for child in children.get(todo.pop(), []):
            found.append(child)
            todo.append(child)
    return found


def _mpi_rank(pid: int) -> str:
    """The MPI rank a process was started as ("" when it is not a rank)."""
    try:
        environ = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
    except OSError:
        return ""
    for item in environ:
        name, _, value = item.partition(b"=")
        if name in (b"OMPI_COMM_WORLD_RANK", b"PMI_RANK", b"PMIX_RANK"):
            return value.decode(errors="ignore")
    return ""


def _where_waiting(stack: str, case_dir: Path) -> str:
    """Where a process was, from the Python stack it wrote: its innermost call
    in a file of the case, with that line of code, else its innermost call."""
    frames = []
    for line in stack.splitlines():
        text = line.strip()
        if not text.startswith('File "'):
            continue
        name, _, rest = text[len('File "'):].partition('", line ')
        number, _, func = rest.partition(" in ")
        if number.isdigit():
            frames.append((name, int(number), func))
    if not frames:
        return "no Python code running: its script had finished and it was exiting"
    case = Path(case_dir).resolve()
    for name, number, func in frames:  # innermost first
        path = Path(name) if Path(name).is_absolute() else case / name
        try:
            rel = path.resolve().relative_to(case)
            code = path.read_text(errors="ignore").splitlines()[number - 1].strip()
        except (ValueError, OSError, IndexError):
            continue
        return f"{rel} line {number} in {func}: {code}"
    name, number, func = frames[0]
    return f"{name} line {number} in {func}"


def _process_stacks(proc: subprocess.Popen, stack_dir: Path, case_dir: Path) -> str:
    """Where each Python process of a stuck run was waiting, from the stacks
    stack_hook/sitecustomize.py has them write on SIGWINCH (a process without
    the hook ignores it)."""
    procs = [(pid, _mpi_rank(pid)) for pid in _descendants(proc.pid)]
    for pid, _ in procs:
        try:
            os.kill(pid, signal.SIGWINCH)
        except OSError:
            pass
    time.sleep(2.0)
    where: Dict[str, List[Tuple[int, str]]] = {}
    for pid, rank in procs:
        try:
            stack = (stack_dir / f"stack.{pid}").read_text(errors="ignore")
        except OSError:
            continue  # not a Python process
        # A process whose Python has shut down no longer answers: mpi4py's
        # ranks wait in MPI_Finalize after that.
        place = (_where_waiting(stack, case_dir) if stack.strip() else
                 "no answer: its Python had already shut down, so it had finished its script and was exiting")
        where.setdefault(place, []).append(
            (int(rank) if rank.isdigit() else 10**9 + pid, f"rank {rank}" if rank else f"pid {pid}"))
    if not where:
        return ""
    lines = [", ".join(label for _, label in sorted(who)) + ": " + place
             for place, who in sorted(where.items(), key=lambda kv: min(kv[1]))]
    return ("\n[harness] Where each Python process of the run was waiting when it was stopped:\n  "
            + "\n  ".join(lines[:20]))


def run_script(cmd: List[str], cwd: Path | str, env: Optional[Dict[str, str]],
               timeout: Optional[float]) -> tuple[int, str, str, bool]:
    """Run a case script in its own process group and return (returncode,
    stdout, stderr, timed_out). On a timeout, or if this process is
    interrupted, the whole group is stopped: killing only the script would
    leave mpirun and its solver ranks running, still writing into the case.

    The script reads nothing: with the terminal as its input, an Allrun that
    ran `bash` as the case's application sat waiting for typed commands for
    1 h 20 min (shelf_demo_dafoam_20261010_1356). A case on a solver's own
    OpenFOAM runs inside that solver's wrapper
    (cfd_langgraph.solver_shelf.in_runtime)."""
    from cfd_langgraph.solver_shelf import in_runtime

    # A run that writes nothing for this long is stuck, not slow: a DAFoam
    # script deadlocked between its ranks (rank 0 asleep, three spinning) and
    # would have held its case for the six-hour limit
    # (shelf_demo_dafoam_20261011_1437, case_002).
    # Only for a case's own scripts (Allrun, Allrun.extend), whose solvers log
    # as they go; a run_python computation may rightly write nothing for long.
    stall_s = (float(os.environ.get("CFD_SCIENTIST_RUN_STALL_S") or _stall_limit_s(Path(cwd)))
               if cmd and Path(str(cmd[0])).name.startswith("Allrun") else 0.0)
    # Such a script's Python processes can be asked where they wait before
    # it is stopped (stack_hook/sitecustomize.py). Set inside the command,
    # since a solver's wrapper starts from an empty environment.
    stack_dir = Path(tempfile.mkdtemp(prefix="cfd_stacks_")) if stall_s > 0 else None
    if stack_dir is not None:
        cmd = ["bash", "-c", 'export PYTHONPATH="$1${PYTHONPATH:+:$PYTHONPATH}" CFD_SCIENTIST_STACK_DIR="$2"; '
               'shift 2; exec "$@"', "with-stacks", str(_STACK_HOOK), str(stack_dir), *cmd]
    proc = subprocess.Popen(in_runtime(cmd, Path(cwd)), cwd=str(cwd), env=env, stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            start_new_session=True)
    # A run whose log already shows a crash and that has gone quiet is over:
    # one DAFoam rank raised, its eight ranks spun at full CPU and the run
    # held its cores until the 45-min stall limit
    # (shelf_demo_dafoam_20261015_1318, case_003).
    crash_quiet_s = (float(os.environ.get("CFD_SCIENTIST_CRASH_QUIET_S") or 300) if stall_s > 0 else 0.0)
    deadline = time.monotonic() + timeout if timeout else None
    mark, changed_at = _newest_change(Path(cwd)), time.monotonic()
    # A run that cannot reach its endTime in the time it has is stopped once
    # its pace is known (15 min of progress), and the numbers are reported: a
    # DSMC case at deltaT 1e-9 to endTime 0.002 made 1,576 of 2,000,000 steps
    # in 3 h and would have held 16 cores to the six-hour limit
    # (shelf_demo_hystrath_20261011_1602, case_003).
    # The pace is judged over the last window, not since the start: an olaFlow
    # run's time step collapsed from 5e-3 to 6e-7 s after 19 of 21 s, and the
    # average since the start still looked fine
    # (shelf_demo_olaflow_20261011_1242, case_002).
    samples: List[Tuple[float, float]] = []  # (wall, simulated time)
    pace_window_s = float(os.environ.get("CFD_SCIENTIST_PACE_WINDOW_S") or 900)
    hopeless = ""
    try:
        while True:
            try:
                out, err = proc.communicate(timeout=60)
                return proc.returncode, out or "", err or "", False
            except subprocess.TimeoutExpired:
                now = time.monotonic()
                latest = _newest_change(Path(cwd))
                if latest != mark:
                    mark, changed_at = latest, now
                stalled = stall_s > 0 and now - changed_at > stall_s
                crashed = (not stalled and crash_quiet_s > 0 and now - changed_at > crash_quiet_s
                           and _crash_in_logs(Path(cwd)))
                if stall_s > 0 and deadline is not None and not stalled:
                    progress = _simulated_progress(Path(cwd))
                    if progress is not None:
                        sim, end = progress
                        if samples and sim < samples[-1][1]:
                            samples.clear()  # a new run started in the same case
                        samples.append((now, sim))
                        while len(samples) > 1 and now - samples[1][0] >= pace_window_s:
                            samples.pop(0)
                        w0, s0 = samples[0]
                        if now - w0 >= pace_window_s and sim < end:
                            rate = (sim - s0) / (now - w0)
                            needed = (end - sim) / rate if rate > 0 else float("inf")
                            left = deadline - now
                            if needed > max(3 * left, 3600):
                                hopeless = (
                                    f"\n[harness] Stopped: at the pace of the last {(now - w0) / 60:.0f} min "
                                    f"(simulated time {s0:g} to {sim:g}), reaching endTime {end:g} would "
                                    + (f"take about {needed / 3600:.0f} h more" if rate > 0 else "never happen")
                                    + f", against {left / 3600:.1f} h left for this run. If the time step "
                                    "collapsed, the solution is blowing up; otherwise shorten endTime, "
                                    "enlarge deltaT, or make each step cheaper.")
                if not stalled and not crashed and not hopeless and (deadline is None or now < deadline):
                    continue
                stacks = _process_stacks(proc, stack_dir, Path(cwd)) if stalled and stack_dir else ""
                _stop_group(proc)
                out, err = proc.communicate()
                why = (f"\n[harness] Stopped: nothing in the case changed for {stall_s / 60:.0f} min, "
                       "so the run was stuck (a deadlock between ranks or an endless wait), not slow."
                       + stacks
                       if stalled else
                       f"\n[harness] Stopped: the log shows a crash ({crashed}) and nothing in the case "
                       f"changed for {crash_quiet_s / 60:.0f} min after it, so the run was over; the "
                       "processes left were waiting on the one that failed."
                       if crashed else hopeless or
                       f"\n[harness] {TIME_LIMIT_NOTE} ({timeout / 3600:.1f} h) while it was still "
                       "running; it was not stuck or crashed.")
                return -1, out or "", (err or "") + why, True
    except BaseException:
        _stop_group(proc)
        raise
    finally:
        # The script can exit while something it started in the background
        # is still running in its group.
        _stop_group(proc, quiet=True)
        if stack_dir is not None:
            shutil.rmtree(stack_dir, ignore_errors=True)


_CRASH_MARKERS = ("Traceback (most recent call last)", "FOAM FATAL", "MPI_ABORT",
                  "Segmentation fault", "Floating point exception")


def _in_a_mesh(rel: Path) -> bool:
    """Whether a case-relative path is inside a constant/polyMesh folder, the
    case's own or one in a folder the case holds. Checked at the top level
    only, a nested case's mesh was written as #include stubs and its run died
    with 'Cannot find file "points"' (shelf_demo_dafoam_20261015_1318)."""
    parts = Path(rel).parts
    return any(parts[i] == "constant" and parts[i + 1] == "polyMesh" for i in range(len(parts) - 1))


def _inputs_fingerprint(case_dir: Path, extra: Iterable[str] = ()) -> str:
    """A hash of what the case is made of: 0/, constant/ (not the mesh),
    system/, the Allrun and the case's scripts at its top level, and any other
    named files. Outputs (time folders, processor folders, logs, results) are
    not part of it, so two runs with the same fingerprint ran the same case."""
    import hashlib

    paths: List[Path] = []
    for folder in ("0", "constant", "system"):
        d = case_dir / folder
        if d.is_dir():
            paths += [p for p in d.rglob("*") if p.is_file() and not _in_a_mesh(p.relative_to(case_dir))]
    paths += [p for p in case_dir.iterdir() if p.is_file() and (p.name == "Allrun" or p.suffix in (".py", ".sh"))]
    paths += [case_dir / rel for rel in extra if (case_dir / rel).is_file()]
    digest = hashlib.sha1()
    for p in sorted(set(paths)):
        digest.update(str(p.relative_to(case_dir)).encode())
        try:
            digest.update(p.read_bytes())
        except OSError:
            continue
    return digest.hexdigest()


def _startup_notice(line: str) -> bool:
    """OpenFOAM's start-up banner line "trapFpe: Floating point exception
    trapping enabled", which names a crash marker and is no crash. Read as
    one, it had healthy DAFoam runs stopped after five quiet minutes and
    their records point at decomposition (shelf_demo_dafoam_20261015_1318)."""
    return "trapping enabled" in line


def _crash_in_logs(case_dir: Path) -> str:
    """The first crash marker in the tail of the case's log files, or ""."""
    for log in sorted(case_dir.glob("log.*")):
        try:
            with log.open("rb") as f:
                f.seek(max(0, log.stat().st_size - 200_000))
                tail = f.read().decode("utf-8", errors="ignore")
        except OSError:
            continue
        lines = tail.splitlines()
        real_fatal = set(_fatal_blocks(lines))
        for i, line in enumerate(lines):
            if _startup_notice(line):
                continue
            if "FOAM FATAL" in line and i not in real_fatal:
                continue  # a notice the run carries on past
            for marker in _CRASH_MARKERS:
                if marker in line:
                    return f"{marker} in {log.name}"
    return ""


def _stop_group(proc: subprocess.Popen, quiet: bool = False) -> None:
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        pgid = proc.pid
    for sig, wait in ((signal.SIGTERM, 10.0), (signal.SIGKILL, 5.0)):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return
        except PermissionError:
            if not quiet:
                raise
            return
        deadline = time.monotonic() + wait
        while time.monotonic() < deadline:
            try:
                os.killpg(pgid, 0)
            except ProcessLookupError:
                return
            time.sleep(0.2)


def _run_seeded_case(
    base_case_seed_dir: Path,
    case_dir: Path,
    *,
    openfoam_path: str,
    max_loop: int,
    max_time_limit_s: int,
    t0: float,
    prepare: Optional[Callable[[Path], Any]] = None,
) -> Dict[str, Any]:
    """Copy a validated case and run it. No LLM calls at all. ``prepare``, if
    given, may add output-only settings (extra function objects) to the copy
    before it runs; it must not change the physics or the mesh."""
    _copy_case_files(base_case_seed_dir, case_dir)
    own_allrun = _copy_own_allrun(base_case_seed_dir, case_dir)
    # A start case made from a level that ran its own Allrun carries that
    # level's final mesh and initial fields: meshing it again from
    # blockMeshDict alone would undo what the rest of that Allrun did.
    mesh_given = not own_allrun and bool(_read_state(base_case_seed_dir).get("own_allrun"))
    if prepare is not None:
        prepare(case_dir)
    solver = _read_solver_from_control_dict(case_dir) or _default_solver("simpleFoam")
    # Record the source case: a copy under a new name has lost the condition
    # its original directory name carried, which scoring depends on.
    (case_dir / ".foamagent_state.json").write_text(
        json.dumps(
            {
                "case_info": {"case_solver": solver},
                "seed_source": str(base_case_seed_dir),
                "seed_name": base_case_seed_dir.name,
                "own_allrun": own_allrun or mesh_given,
            },
            indent=2,
        )
    )
    allrun_path = case_dir / "Allrun"
    if not own_allrun:
        allrun_path.write_text(allrun_mod.build_mesh_and_solve_allrun(
            solver, case_dir=case_dir, mesh_given=mesh_given))
        allrun_path.chmod(0o755)
    allrun_env = case_env(openfoam_path)
    _clean_stale_run_artifacts(case_dir)

    remaining = max(60, max_time_limit_s - int(time.monotonic() - t0))
    returncode, out, err, _ = run_script(["./Allrun"], case_dir, allrun_env, remaining)
    allrun_out = out + "\n" + err
    (case_dir / "Allrun.out").write_text(allrun_out)

    mesh_errors = mesh_check_errors(case_dir)
    success = (
        returncode == 0
        and _solver_ended_cleanly(case_dir, solver)
        and not _has_fatal(case_dir)
        and not mesh_errors
    )
    # A failure here is NOT something a review loop can repair: the case was
    # not authored, it was copied from the study's own validated base case.
    # Report it plainly instead of rewriting files nobody wrote.
    result = {
        "status": "success" if success else "failed",
        "case_dir": str(case_dir),
        "case_solver": solver,
        "loop_count": 1,
        "seed_only": True,
        "base_case_seed_dir": str(base_case_seed_dir),
        "mesh_check_errors": mesh_errors,
        "error": "" if success else (
            "The validated base case did not run cleanly as copied. This was not "
            "generated, so it cannot be fixed by rewriting case files — check the "
            "base case itself."
        ),
    }
    (case_dir / "run_result.json").write_text(json.dumps(result, indent=2))
    return result


def _canonical_dict(path: Path, env: Optional[Dict[str, str]]) -> str:
    """A dictionary file's content in OpenFOAM's own canonical form, from its
    FoamFile header on (the banner above it names the file)."""
    proc = subprocess.run(["foamDictionary", "-expand", str(path)], env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        return f"unreadable:{path}"
    start = proc.stdout.find("FoamFile")
    return proc.stdout[start:] if start >= 0 else proc.stdout


def _apply_start_case(case_dir: Path, start_case: Path, seed_case: Path,
                      env: Optional[Dict[str, str]]) -> str:
    """Start this case the way the study's runs start (see
    mesh_assessment.make_start_case): from the settled flow when the start
    case carries it and this case's initial and boundary conditions are still
    the selected level's, from rest otherwise -- a settled field whose
    boundaries were set up differently would start the case from a state its
    own conditions never produced."""
    from cfd_langgraph.mesh_assessment import _set_entry, set_averaging_start

    (case_dir / "start_applied.json").unlink(missing_ok=True)  # from an earlier run of this case
    try:
        decision = json.loads((start_case / "start_decision.json").read_text())
    except (OSError, ValueError):
        return "from rest: the start case has no decision on record"
    if not decision.get("from_settled_state"):
        return "from rest, as the study's start decision says"
    names = sorted(f.name for f in (case_dir / "0").iterdir() if f.is_file())
    if names != sorted(f.name for f in (seed_case / "0").iterdir() if f.is_file()) or any(
            _canonical_dict(case_dir / "0" / n, env) != _canonical_dict(seed_case / "0" / n, env) for n in names):
        return "from rest: its initial or boundary conditions differ from the selected level's"
    for n in names:
        shutil.copy2(start_case / "0" / n, case_dir / "0" / n)
    cd = case_dir / "system" / "controlDict"
    length = float(decision["run_length"])
    _set_entry(cd, "startFrom", "startTime", env)
    _set_entry(cd, "startTime", "0", env)
    _set_entry(cd, "endTime", f"{length:g}", env)
    averaging = set_averaging_start(cd, float(decision.get("averaging_start") or 0.0), env)
    (case_dir / "start_applied.json").write_text(json.dumps({
        "from_settled_state": True,
        "fields_replaced": names,
        "endTime": length,
        "averaging_start": decision.get("averaging_start") if averaging else None,
        "note": (
            "Set on purpose by the study's start protocol: this case starts from the selected "
            "mesh level's settled flow, so 0/ holds that flow (not uniform initial fields) and "
            f"endTime is {length:g}, the run length decided for every run of this study, "
            "whatever the requirement text says. Do not change these or re-run to 'fix' them."
        ),
    }, indent=2))
    return f"from the settled flow and runs {length:g}"


def _not_a_dictionary(folder_name: str, file_name: str, content: str, before: str = "") -> bool:
    """True when what a writer returned for an OpenFOAM dictionary (a file
    without an extension under 0/, constant/ or system/) has no FoamFile
    header. Asked for a 157,464-value field, the writer answered with a
    sentence saying it could not, and the sentence was saved as the field
    (shelf_demo_dafoam_20261015_1318, case_003)."""
    top = Path(folder_name or ".").parts[:1]
    if not top or top[0] not in ("0", "0.orig", "0.org", "constant", "system") or "." in file_name:
        return False
    # A fragment that never had a header (an #include'd file) may lack one.
    if before.strip() and "FoamFile" not in before:
        return False
    return "FoamFile" not in content


def _other_cases_note(case_dir: Path, user_requirement: str, max_files: int = 40) -> str:
    """The files present in every other case of the study that the
    requirement names (case_007, ...), for the writer, the reviewer and the
    setup check. Told only to read case_007's results from its folder, three
    DAFoam inversion scripts each guessed a file name case_007 had not
    written (results.json, reference.json) and stopped at "reference data
    absent" (shelf_demo_dafoam_20261019_1057)."""
    words = "".join(ch if ch.isalnum() or ch == "_" else " " for ch in user_requirement).split()
    named = sorted({w for w in words if w.startswith("case_") and w[5:].isdigit()} - {case_dir.name})
    lines = []
    for other in named:
        folder = case_dir.parent / other
        if not folder.is_dir():
            lines.append(f"{other}: no folder yet ({folder})")
            continue
        # Its own top-level files (results, data, reports) and the fields of
        # its latest time folder; not its inputs, logs or figures.
        files = [f"{p.name} ({p.stat().st_size} B)" for p in sorted(folder.iterdir())
                 if p.is_file() and not p.name.startswith(("log.", "Allrun", "."))
                 and p.suffix not in (".foam", ".png", ".pyc")][:max_files]
        times = sorted((p for p in folder.iterdir() if p.is_dir() and p.name != "0" and _is_number(p.name)),
                       key=lambda p: float(p.name))
        if times:
            fields = sorted(q.name for q in times[-1].iterdir() if q.is_file())
            files.append(f"time folder {times[-1].name}/: " + " ".join(fields[:max_files]))
        lines.append(f"{other} ({folder}): " + (", ".join(files) or "no files yet"))
        # How its data files are laid out, so a script reads them by their
        # real keys (_file_view).
        # Its data files, the largest first; not the framework's own records,
        # which sorted first by name and pushed the truth file out.
        data_files = sorted((p for p in folder.iterdir() if p.is_file() and p.suffix == ".json"
                             and p.name not in _HARNESS_FILES), key=lambda p: -p.stat().st_size)
        for rank, data in enumerate(data_files[:3]):
            # The largest, the case's main result, gets room to reach its
            # second level: at 60 lines a truth file's mesh.cell_centres_m (line
            # 65) was left out (shelf_demo_dafoam_20261019_1057, case_006).
            try:
                outline = _json_outline(json.loads(data.read_text(encoding="utf-8", errors="ignore")),
                                        max_lines=150 if rank == 0 else 40)
            except (OSError, ValueError):
                continue
            lines.append(f"  {other}/{data.name}, its layout:\n" + "\n".join("    " + x for x in outline.splitlines()))
    if not lines:
        return ""
    return ("\n\nFILES OF THE OTHER CASES THIS REQUIREMENT NAMES (what is there now; read a case's "
            "results from these files, by these names):\n" + "\n".join(lines))


def _notes_block(notes: str) -> str:
    """The case's shelf-solver usage notes as a block for a writer or reviewer prompt."""
    return ("\n\nSOLVER NOTES (how this solver must be used on this machine; they hold):\n" + notes
            if notes else "")


def _examples_note(script_examples: str) -> str:
    """The solver's example scripts as a block for a writer or reviewer prompt."""
    if not script_examples:
        return ""
    return ("\n\nEXAMPLES of this solver's Python API, from its own regression tests and "
            "tutorials, chosen for this requirement. Read them for how the API is called "
            "(solver options, inputs such as field design variables, functions, derivatives, how "
            "the problem is built and run) and follow that API usage, option keys included; do not "
            "copy what belongs to their own cases (paths, cell counts, boundary values, the "
            "commands that set up their folders):\n" + script_examples)


def _abbreviated_field(text: str, keep: int = 2000) -> str:
    """A field file with a long nonuniform internalField shortened: the
    values say nothing a setup check can use, and a 157k-cell field would
    crowd every other file out."""
    start = text.find("internalField")
    if start < 0:
        return text
    end = text.find("boundaryField", start)
    values = text[start:end] if end > 0 else text[start:]
    if len(values) <= keep or "nonuniform" not in values[:200]:
        return text
    opening = values.find("(")
    head = values[:opening + 1] if 0 <= opening < 400 else values[:200]
    return (text[:start] + head + f"\n... {len(values)} characters of values left out ...\n)\n;\n\n"
            + (text[end:] if end > 0 else ""))


def _case_files_xml(case_dir: Path, *, given: set, zero_template: str = "", mesh_fixed: bool = False,
                    max_file_kb: int = 30, max_total_kb: int = 300) -> str:
    """Every dictionary of the case, seeded or written, for the setup check:
    system/controlDict, constant/, the initial fields, the rest of system/,
    and the case's own Python scripts. The reviewer's view
    (_foamfiles_xml) holds only the files the writer was asked for, and the
    values that went wrong (a starter's density, kept at the wrong Knudsen
    number) were in seeded files it never listed."""
    import gzip

    def files(folder: str, suffix: str = "") -> List[Path]:
        d = case_dir / folder if folder else case_dir
        return sorted(p for p in d.iterdir() if p.is_file() and p.name.endswith(suffix)) if d.is_dir() else []

    ordered = ([case_dir / "system" / "controlDict"] + files("constant") + files(zero_template or "0")
               + files("system") + files("", ".py"))
    summary = mesh_summary(case_dir)
    parts: List[str] = [f"<mesh>{summary}</mesh>"] if summary else []
    left_out: List[str] = []
    seen: set = set()
    total = 0
    for path in ordered:
        rel = path.relative_to(case_dir)
        if str(rel) in seen or not path.is_file():
            continue
        if path.name.endswith(".gz") and path.with_name(path.name[:-3]).is_file():
            continue  # OpenFOAM reads the plain copy
        seen.add(str(rel))
        try:
            if path.name.endswith(".gz"):
                with gzip.open(path, "rt", errors="ignore") as f:
                    text = f.read(20 * 1024 * 1024)
            elif path.stat().st_size > 50 * 1024 * 1024:
                left_out.append(str(rel))
                continue
            else:
                text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if "\x00" in text[:4096]:
            continue
        text = _abbreviated_field(text)
        if len(text) > max_file_kb * 1024:
            text = text[: max_file_kb * 1024] + "\n... [truncated]"
        if total + len(text) > max_total_kb * 1024:
            left_out.append(str(rel))
            continue
        total += len(text)
        note = ""
        if str(rel) in given:
            note = "<note>given by the study: stays as it is</note>"
        elif mesh_fixed and path.name == "blockMeshDict":
            note = "<note>the study's selected mesh: stays as it is</note>"
        parts.append(f"<foamfile><file_name>{path.name}</file_name>"
                     f"<folder_name>{str(rel.parent).strip('.')}</folder_name>{note}"
                     f"<content>{text}</content></foamfile>")
    if left_out:
        parts.append(f"<!-- not shown, too large: {', '.join(left_out)} -->")
    # Every other file of the case, by name: shown only the dictionaries and
    # top-level scripts, the check reported FFD/bumpFFD.xyz "absent" round
    # after round while it sat in the case (shelf_demo_dafoam_20261019_1057).
    others = []
    for path in sorted(case_dir.rglob("*")):
        rel = path.relative_to(case_dir)
        top = rel.parts[0]
        if (not path.is_file() or str(rel) in seen or _in_a_mesh(rel) or top.startswith(("processor", "log."))
                or top in ("postProcessing", "dynamicCode") or (top != "0" and _is_number(top) and len(rel.parts) > 1)):
            continue
        others.append(f"{rel} ({path.stat().st_size} B)")
        if len(others) >= 200:
            others.append("...")
            break
    if others:
        parts.append("<other_files>present in the case, not shown: " + ", ".join(others) + "</other_files>")
    return "\n".join(parts)


def _is_number(name: str) -> bool:
    try:
        float(name)
    except ValueError:
        return False
    return True


def _apply_edits(
    llm: Any, case_dir: Path, target_files: List[Dict[str, Any]], *, history: List[str],
    zero_template: str, given: set, mesh_seed_case_dir: Optional[Path], run_script_name: str,
    subtasks: List[Dict[str, str]], case_solver: str, script_examples: str = "",
    solver_notes: str = "",
) -> int:
    """Apply a rewrite plan to the case's files (Stage 8c, and the setup
    check's fixes). Edits the study forbids are refused into ``history``.
    Returns how many edits failed."""
    from cfd_langgraph.solver_shelf import own_runtime

    edit_failures = 0
    for t in target_files:
        rel = _without_case_name(case_dir, Path(t["file"]))
        if zero_template and rel.parts[:1] == ("0",) and len(rel.parts) > 1:
            rel = Path(zero_template, *rel.parts[1:])  # 0/ is rebuilt from it by the Allrun
        named_gz = rel.name.endswith(".gz")
        # A compressed field is edited as plain text: written back under the
        # .gz name it would be text in a file OpenFOAM reads as gzip.
        if rel.name.endswith(".gz"):
            rel = rel.with_name(rel.name[:-3])
        if str(rel) in given:
            history.append(f"Reviewer proposed editing {rel}, which the study gives as it is; "
                           "rejected. Fix the case another way.")
            continue
        try:
            _safe_case_relative_path("", str(rel))
        except ValueError as exc:
            history.append(f"Reviewer proposed a path that was rejected ({exc}). "
                           "Name each file to change.")
            continue
        if rel.name == "Allrun" and not str(rel.parent).strip(".") and own_runtime(case_dir):
            # Only the framework's Allrun runs every command inside the
            # solver's wrapper. Rewritten freely, one loaded OpenFOAM 10
            # and refused to run a hyStrath case at all
            # (shelf_demo_hystrath_20261011_1602, case_006).
            history.append(
                "Reviewer proposed editing Allrun, which for a solver with its own OpenFOAM "
                "is written by the framework to run inside that solver's wrapper; rejected. "
                "Fix the case files instead. That Allrun meshes the case, then runs its solver "
                "or run script on the number of MPI ranks set by numberOfSubdomains in "
                "system/decomposeParDict, read at each run: to change the rank count, edit that "
                "file (or what the script expects).")
            continue
        if _in_a_mesh(rel):
            # The mesh itself, made by the meshing tools or given with the
            # case: it is not text to edit. Rewritten, a 157k-cell mesh's
            # points became a 286-byte stub and every later run died reading
            # it (shelf_demo_dafoam_20261015_1318, case_003).
            history.append(f"Reviewer proposed editing {rel}, which is part of the mesh itself; "
                           "rejected. The mesh is never edited by hand: fix the case files.")
            continue
        if mesh_seed_case_dir is not None and rel.name == "blockMeshDict":
            # This case was seeded with the mesh gate's certified
            # blockMeshDict. Letting the retry loop rewrite it would run
            # the experiment on a mesh no independence study ever
            # examined, while run_result.json still advertises the
            # certified mesh_seed_case_dir — an unfalsifiable claim.
            # A genuine meshing problem here means the gate's selection
            # is wrong and belongs back in the gate, not patched per case.
            history.append(
                "Reviewer proposed editing blockMeshDict, which is fixed by this study's "
                "mesh-independence gate; rejected. Fix the case another way."
            )
            continue
        folder_name = str(rel.parent) if str(rel.parent) not in (".", "") else ""
        file_name = rel.name
        target_path = case_dir / rel
        gz_path = target_path.with_name(file_name + ".gz")
        # An entry of a dictionary named as if the dictionary were a folder
        # (constant/fvOptions/meshDiagnostic): written, it would end the case
        # run on the folder that cannot be made (shelf_demo_dafoam_20261019_1057,
        # case_007).
        inside_file = next((p for p in reversed(rel.parents) if str(p) not in (".", "")
                            and (case_dir / p).is_file()), None)
        if inside_file is not None:
            history.append(f"Reviewer proposed editing {rel}, but {inside_file} is a file, not a folder; "
                           f"rejected. To change an entry of it, name {inside_file} and say which entry.")
            continue
        if target_path.is_dir():
            # Writing it raised IsADirectoryError and ended the whole mesh
            # gate (shelf_demo_hystrath_20261010_1356, four times on 0/).
            history.append(f"Reviewer proposed editing {rel}, which is a folder; rejected. "
                           "Name the files inside it.")
            continue
        if t.get("delete"):
            if rel == Path("Allrun"):
                history.append("Reviewer proposed removing Allrun, which runs the case; rejected.")
            elif run_script_name and rel == Path(run_script_name):
                history.append(f"Reviewer proposed removing {run_script_name}, which the framework runs to "
                               "solve the case; rejected. Change it instead.")
            else:
                # Exactly the file named: removing a plain duplicate also took
                # its compressed twin, and 0/U and 0/nuTilda vanished from a
                # DAFoam case (shelf_demo_dafoam_20261015_1318, case_003).
                victim = gz_path if named_gz else target_path
                if victim.is_file():
                    victim.unlink()
                    history.append(f"Removed {victim.relative_to(case_dir)}: {t['changes']}")
            continue
        if target_path.is_file():
            current_content = target_path.read_text(encoding="utf-8", errors="ignore")
        elif gz_path.is_file():
            current_content = _field_head(gz_path, _MAX_EDIT_CHARS + 1)
        else:
            current_content = ""
        limit = _MAX_SCRIPT_EDIT_CHARS if rel.suffix in (".py", ".sh") else _MAX_EDIT_CHARS
        if len(current_content) > limit:
            # The edit sends the whole file and asks for the whole file back.
            # A per-cell field on a 157k-cell mesh is over a megabyte: the
            # request passed the model's context window and the case run
            # stalled on it (shelf_demo_dafoam_20261015_1318, case_001,
            # 2026-10-16).
            history.append(
                f"Reviewer proposed editing {rel}, which is more than {limit // 1024} KB "
                "(a field with a value per cell, or similar data): too large to rewrite whole; "
                "rejected. Change what writes this file, or change it with the case's own script.")
            edit_failures += 1
            continue
        try:
            # Long files are changed by exact replacements, not sent back whole.
            edit = writer.edit_file_by_patches if len(current_content) > _PATCH_EDIT_CHARS else writer.edit_file
            new_content = edit(
                llm,
                file_name=file_name, folder_name=folder_name,
                changes=(f"{t['changes']}\n\n{_RUN_SCRIPT_RULES}" + solver_notes + _examples_note(script_examples)
                         if run_script_name and file_name == run_script_name and not folder_name
                         else t["changes"] + solver_notes),
                current_content=current_content,
                written_files_ctx=_written_files_ctx(case_dir, subtasks, len(subtasks))
                + _patch_note(case_dir, folder_name),
                case_solver=case_solver,
            )
        except Exception as exc:
            # One file's edit failing (the model returning nothing for a
            # file it meant to remove, five times) ended the whole case
            # run; the other edits are still worth running.
            history.append(f"Editing {rel} failed and it was left as it was: {exc}")
            edit_failures += 1
            continue
        new_content = _compiled_python(llm, file_name, folder_name, new_content,
                                       _written_files_ctx(case_dir, subtasks, len(subtasks)),
                                       case_solver)
        if _not_a_dictionary(folder_name, file_name, new_content, current_content):
            history.append(f"Editing {rel} returned no OpenFOAM dictionary (it began: "
                           f"{new_content.strip()[:160]!r}); the file was left as it was.")
            edit_failures += 1
            continue
        # Said, a review learns its change was not made; unsaid, a DAFoam case
        # was stopped as "unchanged" with no reason the review could act on
        # (shelf_demo_dafoam_20261019_1057, case_007).
        if new_content.strip() == current_content.strip():
            history.append(f"Editing {rel} returned it unchanged: the change asked for "
                           f"({t['changes'][:200]!r}) was not made.")
            edit_failures += 1
            continue
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(new_content)
        gz_path.unlink(missing_ok=True)
    return edit_failures


def _carried_changed(case_dir: Path, carried_bytes: Dict[str, bytes],
                     setup_check: Dict[str, Any]) -> List[Dict[str, str]]:
    """What the first setup check found differing from the requirement in
    the files the manager carried into the run (run_case_native's
    kept_your_edits) and changed back. Untold, the manager moved a DAFoam
    truth case's beta bump in the script three times; each time the check put
    it back where the approved requirement places it, and the manager took the
    unmoved bump for its edits failing to persist (shelf_demo_dafoam_20261019_1057,
    case_007)."""
    rounds = (setup_check or {}).get("rounds") or []
    if not carried_bytes or not rounds:
        return []
    changed = {rel for rel, before in carried_bytes.items()
               if (case_dir / rel).is_file() and (case_dir / rel).read_bytes() != before}
    return [{"file": str(i.get("file") or ""), "requirement": str(i.get("requirement") or ""),
             "found": str(i.get("found") or "")}
            for i in rounds[0].get("items") or []
            if i.get("status") in review.SETUP_PROBLEMS and str(i.get("file") or "") in changed]


def _setup_check(llm: Any, case_dir: Path, user_requirement: str, *, files_xml: Callable[[], str],
                 apply: Callable[[List[Dict[str, Any]]], int], history: List[str],
                 rounds: int = 2, label: str = "before the first run",
                 solver_notes: str = "") -> Dict[str, Any]:
    """Before the first run: check the case's files against the requirement
    (parameters, boundary and initial conditions, models, run controls), fix
    what differs through the same edit path the reviewer uses, and check
    again, up to ``rounds`` fixes. Written to setup_check.json; what still
    differs after the last fix is returned for run_result.json."""
    record: Dict[str, Any] = {"label": label, "rounds": []}
    for attempt in range(rounds + 1):
        # The solver's notes too: without them the check enforced a requirement
        # naming DAFoam's legacy application, set the run script's solverName
        # to it, and every later run died on "Unknown DASolver type"
        # (shelf_demo_dafoam_20261015_1318, case_003).
        items = review.check_setup(llm, foamfiles_xml=files_xml(), user_requirement=user_requirement,
                                   solver_notes=solver_notes)
        entry: Dict[str, Any] = {"items": items}
        record["rounds"].append(entry)
        problems = [i for i in items if i["status"] in review.SETUP_PROBLEMS]
        if not problems or attempt == rounds:
            break
        analysis = (
            "The case has not run yet. Its files were checked against the requirement and these "
            "differ from it. Fix exactly these, and change nothing else:\n" + "\n".join(
                f"- {p['requirement']}: {p['status']} in {p['file'] or 'the case'} "
                f"(found: {p['found'] or 'nothing'}). Fix: {p['fix']}" for p in problems)) + solver_notes
        targets = review.plan_rewrite(llm, foamfiles_xml=files_xml(), error_logs="(the case has not run yet)",
                                      review_analysis=analysis, user_requirement=user_requirement)
        entry["edits"] = targets
        failed = apply(targets)
        history.append(
            f"Setup check {label}: {len(problems)} value(s) differed from the requirement "
            f"({'; '.join(p['requirement'] for p in problems)[:400]}); edited "
            f"{', '.join(t['file'] for t in targets) or 'nothing'}"
            + (f", {failed} edit(s) failed" if failed else "") + ".")
    final = record["rounds"][-1]["items"]
    record["checked"] = len(final)
    record["fix_rounds"] = sum(1 for r in record["rounds"] if r.get("edits"))
    record["remaining"] = [i for i in final if i["status"] in review.SETUP_PROBLEMS]
    record["unclear"] = [i for i in final if i["status"] == "unclear"]
    # Every check of this case run is kept; the newest is the one at the top.
    earlier = []
    if label != "before the first run" and (case_dir / "setup_check.json").is_file():
        try:
            previous = json.loads((case_dir / "setup_check.json").read_text())
            earlier = previous.pop("earlier", []) + [previous]
        except (OSError, ValueError):
            earlier = []
    (case_dir / "setup_check.json").write_text(json.dumps({**record, "earlier": earlier}, indent=2))
    for item in record["remaining"]:
        print(f"[foam-native] {case_dir.name}: setup still differs from the requirement: "
              f"{item['requirement']} ({item['status']}, found {item['found'] or 'nothing'})", flush=True)
    return {k: record[k] for k in ("checked", "fix_rounds", "remaining", "unclear")}


def run_foam_case(
    llm: Any,
    case_dir: Path,
    user_requirement: str,
    *,
    mesh_type: str = "standard_mesh",
    max_loop: int = 10,
    max_time_limit_s: int = 21600,
    openfoam_path: str = "",
    mesh_seed_case_dir: Path | None = None,
    functions_seed_case_dir: Path | None = None,
    base_case_seed_dir: Path | None = None,
    seed_only: bool = False,
    prepare: Optional[Callable[[Path], Any]] = None,
    start_case_dir: Path | None = None,
    given_files: Optional[Dict[str, Path]] = None,
    carried_files: Optional[Dict[str, Path]] = None,
) -> Dict[str, Any]:
    """The full FoamAgent loop — parse, RAG, decompose, write, Allrun, run,
    review/rewrite/retry, run_result.json — ported stage-by-stage from
    ``cfd-skills/cfd-foamagent/SKILL.md`` so it runs as first-class Python
    inside this workflow. Doesn't import Foam-Agent's ``services.*`` package
    or need ``scripts/foam_run.py`` for the core loop; RAG retrieval still
    prefers the vendored FAISS indices but degrades gracefully without them
    (see ``rag.py``).

    Known gaps vs. the full SKILL.md protocol, scoped out deliberately, not
    silently: mesh routing only fully handles ``standard_mesh`` (custom_mesh
    base-case-copy and gmsh .geo->.msh conversion aren't implemented);
    per-file ``foamDictionary`` syntax verification after each write isn't
    run; the "same file 3 loops in a row -> bail" stuck-loop detector isn't
    implemented, only the ``max_loop`` cap is.
    """
    t0 = time.monotonic()
    case_dir = Path(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)

    # ``seed_only``: the validated base case IS the case to run, so nothing is
    # authored — no parse, no RAG, no decompose, no per-file write, and no
    # Allrun generation. Used for the mesh-gate baseline, whose job is to
    # establish mesh independence OF THE VALIDATED SETUP; there is nothing
    # for a model to invent, and letting it try is actively harmful. Measured
    # on the real study: asked to "edit" the benchmarked periodic-hill case
    # toward a requirement that restated the geometry in metres, the model
    # rescaled convertToMeters and nu but left Ubar alone, turning Re=5600
    # into Re=404, and left 453 mesh edges misaligned. It still ran to "End".
    if seed_only and base_case_seed_dir is None:
        raise ValueError("seed_only requires base_case_seed_dir")
    if seed_only:
        return _run_seeded_case(
            Path(base_case_seed_dir), case_dir, openfoam_path=openfoam_path,
            max_loop=max_loop, max_time_limit_s=max_time_limit_s, t0=t0,
            prepare=prepare,
        )

    # The mesh is the one the study's mesh-independence check selected; a
    # requirement written from the starter can name another. Every DAFoam case
    # asked for the starter's 8000 cells, ran on the selected 58835, and its own
    # script stopped on the mismatch (shelf_demo_dafoam_20261011_1437).
    if mesh_seed_case_dir is not None:
        seed_cells = _n_cells(Path(mesh_seed_case_dir))
        if seed_cells:
            user_requirement = (
                f"{user_requirement}\n\nMESH: this case runs on the mesh the study's mesh-independence "
                f"check selected, {seed_cells} cells, copied with the case. Where the requirement above "
                "names another mesh, cell count or resolution, this mesh wins: use it as it is and do "
                "not check for another size.")

    # Stage 1 — parse
    case_info = parser.parse_requirement(llm, user_requirement)
    (case_dir / ".foamagent_state.json").write_text(json.dumps({"case_info": case_info}, indent=2))

    # Stage 2 — RAG retrieval (with fallback)
    refs = rag.retrieve_references(
        user_requirement, case_info["case_solver"], case_info.get("case_domain", ""), case_info.get("case_category", "")
    )

    # Stage 3 — decompose into subtasks
    subtasks = decomposer.decompose_subtasks(llm, user_requirement, refs.get("dir_structure", ""))
    # A file named outside the case is left out, not a crash of the whole
    # mesh gate: a requirement naming an absolute folder had the decomposer
    # list /mnt/.../Kn0p01/0/U (shelf_demo_hystrath_20261011_0054).
    safe = []
    for subtask in subtasks:
        # A file named by its full path inside this case is the same file:
        # dropped as "outside the case", a DAFoam case's fvSolution, fvSchemes
        # and decomposeParDict were never written nor shown to the reviewer
        # (shelf_demo_dafoam_20261011_1437, case_002).
        full = Path(str(subtask.get("folder_name") or "")) / str(subtask.get("file_name") or "")
        if not full.is_absolute() and _without_case_name(case_dir, full) != full:
            full = _without_case_name(case_dir, full)
            subtask = {**subtask, "folder_name": str(full.parent).strip("."), "file_name": full.name}
        if full.is_absolute():
            try:
                inside = full.resolve().relative_to(case_dir.resolve())
            except ValueError:
                inside = None
            if inside is None:
                # A case folder the requirement invented elsewhere: its 0/,
                # constant/ or system/ file is this case's. Left out instead,
                # constant/waveDict was never written and every olaFlow
                # breakwater case kept its seed's wave
                # (shelf_demo_olaflow_20261011_1242).
                parts = full.parts
                at = max((i for i, part in enumerate(parts) if part in _CASE_FILE_DIRS), default=-1)
                if 0 <= at < len(parts) - 1:
                    inside = Path(*parts[at:])
            if inside is not None and inside.name:
                subtask = {**subtask, "folder_name": str(inside.parent).strip("."),
                           "file_name": inside.name}
        try:
            _safe_case_relative_path(subtask.get("folder_name", ""), subtask.get("file_name", ""))
        except ValueError as exc:
            print(f"[foam-native] left out a file: {exc}", flush=True)
            continue
        # The mesh itself is made by the meshing tools or comes with the case,
        # never written as text. Listed for writing, a shipped mesh got plain
        # points and owner files holding a #codeStream stub, which OpenFOAM
        # read before the real points.gz (shelf_demo_dafoam_20261015_1318,
        # case_003).
        if _in_a_mesh(Path(str(subtask.get("folder_name") or "")) / str(subtask.get("file_name") or "")):
            print(f"[foam-native] left out {subtask.get('folder_name')}/{subtask.get('file_name')}: "
                  "the mesh is not written as text", flush=True)
            continue
        safe.append(subtask)
    subtasks = safe
    _remove_pattern_named_files(case_dir)
    given_mesh = any(d is not None and _prescribed_mesh(Path(d)) for d in (mesh_seed_case_dir, base_case_seed_dir))
    if given_mesh:
        subtasks = [s for s in subtasks if s.get("file_name") != "blockMeshDict"]
    elif mesh_type == "standard_mesh" and not any(s["file_name"] == "blockMeshDict" for s in subtasks):
        subtasks.append({"file_name": "blockMeshDict", "folder_name": "system"})

    # Stage 4b — seed from a known-good case, when the study supplied one.
    #
    # Without this every file is invented from the requirement PROSE, even
    # though the starter already ships a validated case for exactly this
    # physics. Observed consequences of writing from scratch: a hand-derived
    # hill profile that produced 16 negative-volume cells; the case rescaled
    # to dimensional metres so nu/Ubar no longer matched the reference data;
    # `constant/fvOptions` and `constant/turbulenceProperties` instead of the
    # OpenFOAM 10 names the starter uses. None of that is recoverable by the
    # review loop, because the reviewer only ever sees the solver diverging.
    own_allrun = False
    zero_template = ""
    if base_case_seed_dir is not None:
        base_case_seed_dir = Path(base_case_seed_dir)
        _copy_case_files(base_case_seed_dir, case_dir)
        print(f"[foam-native] seeded case from validated base case: {base_case_seed_dir}", flush=True)
        # Fields the seed carried for some other mesh (its own starter's) are of
        # no use on any mesh this case runs.
        seed_cells = _n_cells(base_case_seed_dir)
        if seed_cells:
            _drop_old_mesh_fields(case_dir, n_cells=seed_cells)
        # A seed that ran its own Allrun is meshed by it (snappyHexMesh,
        # createPatch, setFields after blockMesh): the case runs it too. Given
        # a written Allrun of blockMesh and setFields instead, every olaFlow
        # breakwater case solved on the bare block mesh, 33000 cells against
        # fields for the seed's 32610 (shelf_demo_olaflow_20261011_1242).
        if _read_state(base_case_seed_dir).get("own_allrun") and _copy_own_allrun(base_case_seed_dir, case_dir):
            own_allrun = True
            state = _read_state(case_dir)
            (case_dir / ".foamagent_state.json").write_text(
                json.dumps({**state, "own_allrun": True}, indent=2))
            print(f"[foam-native] {case_dir.name}: runs its seed's own Allrun", flush=True)
            # That Allrun rebuilds 0/ from its template folder on every run
            # (olaFlow's breakwater: rm -rf 0; cp -r 0.org 0), so a field
            # written into 0/ was thrown away: fields go to the template.
            zero_template = _zero_template(case_dir)
            if zero_template:
                subtasks = [{**s, "folder_name": zero_template} if s.get("folder_name") == "0" else s
                            for s in subtasks]
                print(f"[foam-native] {case_dir.name}: fields are written in {zero_template}/, which "
                      "its Allrun copies to 0/", flush=True)

    # Files the study gives as they are (run_case_native's given_files): copied
    # over the seed, shown to the reviewer, never rewritten.
    given: set = set()
    for rel, src in (given_files or {}).items():
        rel_path = Path(rel)
        if zero_template and rel_path.parts[:1] == ("0",) and len(rel_path.parts) > 1:
            rel_path = Path(zero_template, *rel_path.parts[1:])
        dest = case_dir / _safe_case_relative_path("", str(rel_path))
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        given.add(str(rel_path))
        if not any(str(Path(s.get("folder_name") or "") / s["file_name"]) == str(rel_path) for s in subtasks):
            subtasks.append({"folder_name": str(rel_path.parent).strip("."), "file_name": rel_path.name})
        print(f"[foam-native] {case_dir.name}: {rel_path} given as it is, from {src}", flush=True)

    # Files the manager changed after the case's previous run (run_case_native's
    # carried_files): copied over the seed and not rewritten by the writer, but,
    # unlike given files, still open to the review's fixes.
    carried: set = set()
    carried_bytes: Dict[str, bytes] = {}
    for rel, src in (carried_files or {}).items():
        rel_path = Path(rel)
        if zero_template and rel_path.parts[:1] == ("0",) and len(rel_path.parts) > 1:
            rel_path = Path(zero_template, *rel_path.parts[1:])
        if str(rel_path) in given or not Path(src).is_file():
            continue
        dest = case_dir / _safe_case_relative_path("", str(rel_path))
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        carried.add(str(rel_path))
        carried_bytes[str(rel_path)] = dest.read_bytes()
        if not any(str(Path(s.get("folder_name") or "") / s["file_name"]) == str(rel_path) for s in subtasks):
            subtasks.append({"folder_name": str(rel_path.parent).strip("."), "file_name": rel_path.name})
        print(f"[foam-native] {case_dir.name}: {rel_path} carried over as the manager left it", flush=True)

    # A case its solver runs by a script is defined as much by that script as
    # by its dictionaries: DAFoam's runScript.py holds what the run computes
    # (gradients, an inversion). Never among the files written, it stayed the
    # seed's one-solve script whatever the requirement asked, and every
    # gradient and inversion case of a DAFoam study ran a plain primal
    # (shelf_demo_dafoam_20261011_1437).
    from cfd_langgraph.solver_shelf import own_runtime

    run_script_name = ((own_runtime(case_dir).get("run") or "").split() or [""])[-1]
    if not run_script_name.endswith(".py"):
        run_script_name = ""
    if run_script_name and (case_dir / run_script_name).is_file() and not any(
            s.get("file_name") == run_script_name and not s.get("folder_name") for s in subtasks):
        subtasks.append({"file_name": run_script_name, "folder_name": ""})
    # The solver's own example scripts, for a case its solver runs by a
    # script: the writer and the reviewer otherwise know its API from memory
    # alone (see rag.shelf_example_scripts).
    script_examples = (rag.shelf_example_scripts(
        user_requirement, _read_solver_from_control_dict(case_dir) or case_info["case_solver"])
        if run_script_name else "")
    solver_notes = _notes_block(rag.shelf_usage_notes(
        user_requirement, _read_solver_from_control_dict(case_dir) or case_info["case_solver"]))
    solver_notes += _other_cases_note(case_dir, user_requirement)

    # Stage 5 — write each subtask file. A file that came from the seed is
    # EDITED toward the requirement, never overwritten from nothing, so the
    # validated numerics, boundary conditions and mesh survive.
    for i, st in enumerate(subtasks):
        out_path = case_dir / _safe_case_relative_path(st["folder_name"], st["file_name"])
        if str(out_path.relative_to(case_dir)) in given | carried:
            continue
        if out_path.is_dir():
            print(f"[foam-native] skipped writing {out_path.relative_to(case_dir)}: it is a folder, "
                  "not a file", flush=True)
            continue
        seeded_content = (
            out_path.read_text(encoding="utf-8", errors="ignore")
            if (base_case_seed_dir is not None and out_path.is_file())
            else ""
        )
        if seeded_content.strip() and run_script_name and st["file_name"] == run_script_name \
                and not st["folder_name"]:
            from .allrun import script_command, script_ranks

            content = writer.edit_file(
                llm,
                file_name=st["file_name"], folder_name=st["folder_name"],
                changes=(
                    "This is the script the case's solver runs: the whole run is "
                    f"`mpirun -np {script_ranks(case_dir)} {script_command(own_runtime(case_dir)['run'])}` from the "
                    "case folder (the rank count is system/decomposeParDict's "
                    "numberOfSubdomains). Make it compute and write everything the "
                    "requirement below asks of this run, using the solver's Python API as "
                    "the script already does, and keep what already works (the solver "
                    f"options, mesh options and how it builds and runs the problem). {_RUN_SCRIPT_RULES} It must "
                    "run on any rank count, and only rank 0 writes result files, inside the "
                    "case folder. It solves the case in the folder it is started from: it "
                    "never copies the case into another folder or runs one elsewhere, since "
                    "only this folder's files are checked and fixed between runs. Return "
                    "the complete Python script.\n\n"
                    f"CASE REQUIREMENT:\n{user_requirement}"
                    + solver_notes + _examples_note(script_examples)
                ),
                current_content=seeded_content,
                written_files_ctx=_written_files_ctx(case_dir, subtasks, i)
                + _patch_note(case_dir, st["folder_name"]),
                case_solver=case_info["case_solver"],
            )
        elif seeded_content.strip():
            content = writer.edit_file(
                llm,
                file_name=st["file_name"], folder_name=st["folder_name"],
                changes=(
                    "Adapt this validated file to the case requirement below. Keep every "
                    "value the requirement does not explicitly change — geometry, mesh "
                    "topology, physical properties, boundary conditions and numerics are "
                    "already correct and benchmarked. Change nothing you are not asked to "
                    "change, and keep the file in the same OpenFOAM version's format.\n\n"
                    "NEVER change the unit system or length scale. The requirement may "
                    "restate the geometry in different units than this case uses; that is "
                    "a restatement, not an instruction to rescale. Do not touch "
                    "convertToMeters, vertex coordinates, nu, or Ubar in order to match "
                    "the units it quotes. A partial rescale silently changes the Reynolds "
                    "number and invalidates the benchmark: rescaling geometry and nu but "
                    "not Ubar has already turned Re=5600 into Re=404 in this project.\n\n"
                    f"CASE REQUIREMENT:\n{user_requirement}" + solver_notes
                ),
                current_content=seeded_content,
                written_files_ctx=_written_files_ctx(case_dir, subtasks, i)
                + _patch_note(case_dir, st["folder_name"]),
                case_solver=case_info["case_solver"],
            )
        else:
            content = writer.write_file_initial(
                llm,
                file_name=st["file_name"], folder_name=st["folder_name"],
                user_requirement=user_requirement + solver_notes, tutorial_reference=refs.get("tutorial_reference", ""),
                written_files_ctx=_written_files_ctx(case_dir, subtasks, i)
                + _patch_note(case_dir, st["folder_name"]),
                case_solver=case_info["case_solver"],
            )
        content = _compiled_python(llm, st["file_name"], st["folder_name"], content,
                                   _written_files_ctx(case_dir, subtasks, i), case_info["case_solver"])
        if _not_a_dictionary(st["folder_name"], st["file_name"], content, seeded_content):
            print(f"[foam-native] {case_dir.name}: the writer returned no OpenFOAM dictionary for "
                  f"{out_path.relative_to(case_dir)} ({content.strip()[:120]!r}); "
                  + ("kept the seeded file" if seeded_content.strip() else "not written"), flush=True)
            if not seeded_content.strip():
                continue
            content = seeded_content
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(content)

    # Enforce the mesh selected by the mesh-independence gate.  The case is
    # still generated from its own experiment requirement, but its mesh
    # generator dictionary comes byte-for-byte from the selected level.
    # Previously the gate wrote selected_mesh_spec.json and normal cases
    # ignored it, so the paper simulations could run on unrelated meshes.
    if mesh_seed_case_dir is not None and _prescribed_mesh(Path(mesh_seed_case_dir)):
        # The mesh-independence gate took this mesh as given (a mesh shipped
        # with a case on a solver's own OpenFOAM, which it cannot refine):
        # the case runs on that mesh, as is.
        shutil.copytree(Path(mesh_seed_case_dir) / "constant" / "polyMesh",
                        case_dir / "constant" / "polyMesh", dirs_exist_ok=True)
        stray = case_dir / "system" / "blockMeshDict"
        if stray.is_file():
            stray.unlink()
    elif mesh_seed_case_dir is not None:
        mesh_seed_case_dir = Path(mesh_seed_case_dir)
        seed_bmd = mesh_seed_case_dir / "system" / "blockMeshDict"
        if mesh_type != "standard_mesh" or not seed_bmd.is_file():
            raise ValueError(
                f"Selected mesh cannot be applied: expected standard-mesh blockMeshDict at {seed_bmd}"
            )
        target_bmd = case_dir / "system" / "blockMeshDict"
        target_bmd.parent.mkdir(parents=True, exist_ok=True)
        base_bmd = Path(base_case_seed_dir) / "system" / "blockMeshDict" if base_case_seed_dir else None
        same_mesh = (base_bmd is not None and base_bmd.is_file()
                     and base_bmd.read_bytes() == seed_bmd.read_bytes())
        shutil.copy2(seed_bmd, target_bmd)
        # On a mesh other than the seed's, the seed's per-cell fields cannot be
        # read: DAFoam's 343-cell 0/pData stopped decomposePar on the selected
        # 58,835-cell mesh, and the run was failed ten times for it although
        # the solve ended cleanly (shelf_demo_dafoam_20261011_1437, case_001).
        if not same_mesh:
            _drop_old_mesh_fields(case_dir)

    if functions_seed_case_dir is not None:
        seeded = _seed_function_objects(case_dir, Path(functions_seed_case_dir))
        if seeded:
            print(f"[foam-native] seeded function objects from base case: {seeded}", flush=True)

    # What the caller needs on every case, authored as well as seeded -- the
    # mesh gate adds time averaging to a transient case this way. Applied only
    # on the seeded path, an authored baseline never wrote the averaged
    # fields its gate quantities are defined on (qwen_retest_20261005/cavity:
    # 6 of 16 failed gate calls).
    if prepare is not None:
        prepare(case_dir)
    drop_unread_transport_properties(case_dir)

    if start_case_dir is not None and base_case_seed_dir is not None:
        how = _apply_start_case(case_dir, Path(start_case_dir), Path(base_case_seed_dir),
                                case_env(openfoam_path))
        print(f"[foam-native] {case_dir.name} starts {how}", flush=True)

    # Stage 6 — Allrun
    # The case runs the application its own controlDict names. The parser's
    # reading of the requirement prose is only a guess: a hy2Foam case seeded
    # from a hy2Foam level was given a dsmcFoam+ Allrun because the requirement
    # mentioned DSMC, and failed ten times over 6.9 h
    # (shelf_demo_hystrath_20261011_1602, case_005).
    written_solver = _read_solver_from_control_dict(case_dir)
    if written_solver and written_solver != case_info.get("case_solver"):
        print(f"[foam-native] {case_dir.name}: runs {written_solver}, as its controlDict says "
              f"(the requirement was read as {case_info.get('case_solver')})", flush=True)
        case_info["case_solver"] = written_solver
    allrun_path = case_dir / "Allrun"
    # An Allrun the study gives is the one that runs. Written over here, the
    # manager's own DAFoam Allrun was replaced by the framework's, which ran in
    # all three cases of a batch, and every run was then rejected for the
    # commands it had added (shelf_demo_dafoam_20261015_1318).
    if "Allrun" in given:
        print(f"[foam-native] {case_dir.name}: Allrun given by the study, run as it is", flush=True)
    elif not own_allrun:
        command_text = allrun_mod.generate_allrun_commands(
            llm, dir_structure=refs.get("dir_structure", ""), case_info=case_info,
            allrun_reference=refs.get("allrun_reference", ""), mesh_type=mesh_type,
        )
        allrun_path.write_text(
            allrun_mod.build_allrun_script(
                command_text, case_solver=case_info["case_solver"],
                needs_blockmesh=(case_dir / "system" / "blockMeshDict").is_file()
                and not (case_dir / "constant" / "polyMesh" / "points").exists(),
                case_dir=case_dir,
            )
        )
        allrun_path.chmod(0o755)

    # Stages 7+8 — run, review-rewrite-retry loop
    loop_count = 0
    success = False
    history: List[str] = []

    touched: set = set()  # every file an edit named, for _inputs_fingerprint

    def apply_edits(target_files: List[Dict[str, Any]]) -> int:
        touched.update(str(t.get("file") or "") for t in target_files if t.get("file"))
        return _apply_edits(
            llm, case_dir, target_files, history=history, zero_template=zero_template, given=given,
            mesh_seed_case_dir=mesh_seed_case_dir, run_script_name=run_script_name,
            subtasks=subtasks, case_solver=case_info["case_solver"], script_examples=script_examples,
            solver_notes=solver_notes)

    # Stage 6b — setup check. The reviewer below reads error logs, so a case
    # that runs cleanly with the wrong values is never looked at again: every
    # hyStrath case of shelf_demo_hystrath_20261011_1602 kept its starter's
    # density, "succeeded" at the wrong Knudsen number and was passed. Before
    # the first run the case's files are checked against the requirement and
    # what differs is fixed; what cannot be fixed is recorded in the result.
    setup_check: Dict[str, Any] = {}
    if os.environ.get("CFD_SCIENTIST_SETUP_CHECK", "1") != "0":
        try:
            setup_check = _setup_check(
                llm, case_dir, user_requirement,
                files_xml=lambda: _case_files_xml(case_dir, given=given, zero_template=zero_template,
                                                  mesh_fixed=mesh_seed_case_dir is not None),
                apply=apply_edits, history=history,
                rounds=int(os.environ.get("CFD_SCIENTIST_SETUP_CHECK_ROUNDS", "2")), solver_notes=solver_notes)
        except Exception as exc:  # the check failing must not stop the case from running
            setup_check = {"error": f"setup check did not complete: {exc}"}
            print(f"[foam-native] {case_dir.name}: {setup_check['error']}", flush=True)
    carried_changed = _carried_changed(case_dir, carried_bytes, setup_check)
    # subprocess.run(["./Allrun"], ...) with no env= inherits this process's
    # environment unchanged; if the shell that launched the CLI never
    # sourced OpenFOAM, Allrun fails at its first line ("$WM_PROJECT_DIR"
    # empty), regardless of anything the model discovers via a separate
    # run_shell call (that's a different subprocess's environment, it
    # doesn't propagate here). Resolve the real environment once and reuse
    # it for every retry loop iteration.
    allrun_env = case_env(openfoam_path)
    # Case directories are reused across relaunches (run_case_native always
    # writes to out_dir/cases/<case_id>, and a RERUN/REVISE case is handed
    # back to a subagent under the same case_id), so the very first attempt
    # of *this* invocation can still be walking into another attempt's logs,
    # decomposition and postProcessing output. Clearing only between retries
    # left the worst case uncovered: a relaunched case whose Allrun no-ops
    # end to end, whose stale solver log still ends in "End", and which is
    # therefore recorded as a success without having run anything.
    _clean_stale_run_artifacts(case_dir)
    # A run cut off by the time limit is reported as that, and no run starts
    # with less than _MIN_RUN_S left. The loop used to review the cut-off run
    # as a failure, edit the case and start nine more runs of 60 s each: three
    # DAFoam inversions were stopped at 6 h while IPOPT was still improving,
    # and their records named a start-up line as the error
    # (shelf_demo_dafoam_20261015_1318).
    time_up = ""
    # A case the review could not change is not run again: every edit refused
    # (a given file) or failed left the same failing script, which was then
    # run ten times over (shelf_demo_dafoam_20261015_1318).
    unchanged = ""
    last_inputs = ""

    def _fingerprinted() -> List[str]:
        return [str(Path(t.get("folder_name") or "") / str(t.get("file_name") or ""))
                for t in subtasks if t.get("file_name")] + sorted(touched)
    for loop_count in range(1, max_loop + 1):
        if prepare is not None:
            prepare(case_dir)  # a reviewer's rewrite may have dropped it
        last_inputs = _inputs_fingerprint(case_dir, _fingerprinted())
        remaining = max_time_limit_s - int(time.monotonic() - t0)
        if loop_count > 1 and remaining < _MIN_RUN_S:
            time_up = (f"The case's time limit ({max_time_limit_s / 3600:g} h) is used up after "
                       f"{loop_count - 1} run(s); no further run was started.")
            loop_count -= 1
            break
        run_started = time.time()
        returncode, out, err, timed_out = run_script(["./Allrun"], case_dir, allrun_env, max(remaining, 60))
        allrun_out = out + "\n" + err
        (case_dir / "Allrun.out").write_text(allrun_out)
        if timed_out and TIME_LIMIT_NOTE in err:
            time_up = (f"Run {loop_count} used up the case's time limit ({max_time_limit_s / 3600:g} h) "
                       "while it was still running (not stuck or crashed); it was stopped there.")
            break

        clean = (
            returncode == 0
            and _solver_ended_cleanly(case_dir, case_info["case_solver"])
            and not _has_fatal(case_dir)
            and not mesh_check_errors(case_dir)
        )
        if clean:
            success = True
            break
        if loop_count >= max_loop:
            break

        # Stages 8a-8c, once more when nothing could be changed: the reviewer
        # then has the refusals in front of it. Stopped after one round, a
        # DAFoam reviewer that had found the cause (a script requiring 2 ranks)
        # proposed an Allrun edit, was refused, and the case ended before it
        # could act on the refusal (shelf_demo_dafoam_20261019_1057, case_002).
        for review_round in range(1, _REVIEW_ROUNDS + 1):
            # Stage 8a — reviewer
            analysis = review.review_errors(
                llm,
                tutorial_reference=refs.get("tutorial_reference", "") + solver_notes + _examples_note(script_examples),
                foamfiles_xml=_foamfiles_xml(case_dir, subtasks),
                error_logs=collect_error_logs(case_dir, since=run_started),
                user_requirement=user_requirement,
                history_text="\n".join(f"Prior attempt {i + 1}: {h}" for i, h in enumerate(history)),
            )
            history.append(analysis[:500])

            # Stage 8b — rewrite plan
            target_files = review.plan_rewrite(
                llm,
                foamfiles_xml=_foamfiles_xml(case_dir, subtasks),
                error_logs=collect_error_logs(case_dir, since=run_started),
                review_analysis=analysis,
                user_requirement=user_requirement,
            )

            # Stage 8c — apply edits
            apply_edits(target_files)
            # The files are checked against the requirement again before they run:
            # checked only before the first run, a DAFoam truth case passed that
            # check, had its truth field dropped by the fixes of runs 2 and 3, and
            # "succeeded" as the unmodified baseline (shelf_demo_dafoam_20261015_1318,
            # case_003).
            if os.environ.get("CFD_SCIENTIST_SETUP_CHECK", "1") != "0":
                try:
                    setup_check = _setup_check(
                        llm, case_dir, user_requirement,
                        files_xml=lambda: _case_files_xml(case_dir, given=given, zero_template=zero_template,
                                                          mesh_fixed=mesh_seed_case_dir is not None),
                        apply=apply_edits, history=history, rounds=1, label=f"before run {loop_count + 1}",
                        solver_notes=solver_notes)
                except Exception as exc:  # noqa: BLE001 - the check failing must not stop the run
                    print(f"[foam-native] {case_dir.name}: setup re-check did not complete: {exc}", flush=True)
            # Checked before the logs are cleared below, so the record keeps the
            # error of the run that will not be repeated. Checked after the clear,
            # a DAFoam case's record said only "nothing changed" and no longer
            # showed why its run had failed (shelf_demo_dafoam_20261019_1057).
            if _inputs_fingerprint(case_dir, _fingerprinted()) != last_inputs:
                break
            history.append("Nothing in the case changed: the review proposed no change to any file. Propose "
                           "the changes that fix the failure." if not target_files else
                           "Nothing in the case changed: every change proposed above was refused or "
                           "failed, for the reasons given. Propose changes that can be made.")
        else:
            unchanged = (f"After run {loop_count} the review changed no file of the case in "
                         f"{_REVIEW_ROUNDS} rounds (every proposed change was refused or failed: see "
                         "the history), so the same case was not run again.")
            break

        # Clear this attempt's log.* before the next ./Allrun — see
        # _clean_stale_run_artifacts's docstring for why this is required, not
        # cosmetic: without it, the edits just applied above never actually
        # get exercised.
        _clean_stale_run_artifacts(case_dir)

    # Stage 9 — run_result.json
    wall_time_s = round(time.monotonic() - t0, 1)
    status = "success" if success else "failed"
    run_result = {
        "status": status,
        "success": success,
        "case_dir": str(case_dir),
        "case_name": case_info.get("case_name", ""),
        "case_solver": case_info.get("case_solver", ""),
        "loop_count": loop_count,
        "max_loop": max_loop,
        "wall_time_s": wall_time_s,
        "error_logs": [] if success else [m for m in (time_up, unchanged) if m] + [error_summary(case_dir)],
        "stopped_by_time_limit": bool(time_up),
        # What the review tried, and what was refused or failed, run by run.
        # Kept only in memory, a record that said "the review changed no file"
        # could not show why (shelf_demo_dafoam_20261019_1057, case_002).
        "review_history": [h[:600] for h in history[-12:]],
        "rag_fallback": bool(refs.get("fallback")),
        "mesh_seed_case_dir": str(mesh_seed_case_dir) if mesh_seed_case_dir else "",
        "base_case_seed_dir": str(base_case_seed_dir) if base_case_seed_dir else "",
        "setup_check": setup_check,
        **({"carried_changed_by_setup_check": carried_changed} if carried_changed else {}),
    }
    (case_dir / "run_result.json").write_text(json.dumps(run_result, indent=2))
    return run_result


def _prescribed_mesh(case_dir: Path) -> bool:
    """A case on a solver's own OpenFOAM whose mesh is shipped with it
    (constant/polyMesh, no blockMeshDict): the mesh-independence gate cannot
    refine it and takes it as given (manager tools, run_mesh_gate)."""
    from cfd_langgraph.solver_shelf import own_runtime

    case_dir = Path(case_dir)
    return ((case_dir / "constant" / "polyMesh").is_dir()
            and not (case_dir / "system" / "blockMeshDict").is_file()
            and bool(own_runtime(case_dir)))


def _copy_case_files(base_case_dir: Path, case_dir: Path) -> None:
    """Copy a case's real input files (``0/``, ``constant/``, ``system/``) —
    not stale run outputs like ``postProcessing/``, ``log.*``, ``Allrun.out``."""
    case_dir.mkdir(parents=True, exist_ok=True)
    # The seed's mesh replaces the case's whole: merged in, its points.gz sat
    # beside a stale plain points, which OpenFOAM reads first
    # (shelf_demo_dafoam_20261015_1318, case_003).
    if (base_case_dir / "constant" / "polyMesh").is_dir():
        shutil.rmtree(case_dir / "constant" / "polyMesh", ignore_errors=True)
    for item in _CASE_FILE_DIRS:
        src = base_case_dir / item
        if src.is_dir():
            shutil.copytree(src, case_dir / item, dirs_exist_ok=True)
    # A tutorial that ships its initial fields as 0.orig/ (0.org/ in older
    # ones) has its own Allrun copy them to 0/; the Allrun written here does
    # not, and the seeded olaFlow baseWaveFlume failed on a missing 0/p_rgh
    # (shelf_demo_olaflow_20261010_1613).
    if not (base_case_dir / "0").is_dir():
        for name in ("0.orig", "0.org"):
            if (base_case_dir / name).is_dir():
                shutil.copytree(base_case_dir / name, case_dir / "0", dirs_exist_ok=True)
                break
    # A field both plain and compressed (DAFoam writes 0/*.gz beside the
    # starter's plain files): OpenFOAM reads the plain one, so the compressed
    # copy only misleads whoever reads the case next.
    for folder in ("0", "0.orig", "0.org"):
        d = case_dir / folder
        if d.is_dir():
            for gz in d.glob("*.gz"):
                if (d / gz.name[:-3]).is_file():
                    gz.unlink()
    # A case on a solver's own OpenFOAM may need more (DAFoam's runScript.py
    # and FFD/), as that solver declares.
    from cfd_langgraph.solver_shelf import own_runtime

    for item in own_runtime(case_dir).get("inputs", []):
        src = base_case_dir / item
        if src.is_dir():
            shutil.copytree(src, case_dir / item, dirs_exist_ok=True)
        elif src.is_file():
            shutil.copy2(src, case_dir / item)
    drop_unread_transport_properties(case_dir)


# What a case directory may hold besides its inputs; never carried to a copy.
_RUN_OUTPUTS = ("postProcessing", "dynamicCode", "VTK", "Allrun.out", "run_result.json",
                ".foamagent_state.json")


def _copy_own_allrun(base_case_dir: Path, case_dir: Path) -> bool:
    """Run a copied case the way its source runs: when the source has an
    Allrun, copy it with everything beside it that it may use (0.orig/,
    helper scripts, extra dictionaries) but not run outputs, and return True.
    The Allrun written here otherwise meshes with blockMesh alone, and the
    olaFlow breakwater starter, which also needs snappyHexMesh and setFields,
    crashed on its missing caisson patch (shelf_demo_olaflow_20261011_1055).

    A case on a solver's own OpenFOAM runs its Allrun inside that solver's
    wrapper like any other script (run_script); the hyStrath NASA MSL starter
    needs its createPatch (shelf_demo_hystrath_20261011_1055). Not for a case
    solved by a command in place of its program (DAFoam's runScript.py): the
    Allrun written here is what writes the log its run is judged by."""
    from cfd_langgraph.solver_shelf import own_runtime

    if not (base_case_dir / "Allrun").is_file() or own_runtime(case_dir).get("run"):
        return False
    for item in base_case_dir.iterdir():
        name = item.name
        if (name in _CASE_FILE_DIRS or name in _RUN_OUTPUTS or name.startswith(("log.", "processor"))
                or parse_time_dir(name) is not None):
            continue
        if item.is_dir():
            shutil.copytree(item, case_dir / name, dirs_exist_ok=True)
        else:
            shutil.copy2(item, case_dir / name)
    _parallel_own_allrun(case_dir)
    (case_dir / "Allrun").chmod(0o755)
    return True


def _parallel_own_allrun(case_dir: Path) -> bool:
    """In a copied Allrun, run the case's solver on the ranks its
    decomposeParDict asks for, when the script runs it serially: that one line
    becomes decompose, parallel solve, reconstruct, logging as before. A
    refined olaFlow level of 292,656 cells ran serially for hours under the
    starter's own Allrun, its case set up for 4 ranks
    (shelf_demo_olaflow_20261011_1242). Returns whether it changed the script."""
    from .allrun import script_ranks

    allrun, ranks = case_dir / "Allrun", script_ranks(case_dir)
    solver = _read_solver_from_control_dict(case_dir)
    if ranks < 2 or not solver or not allrun.is_file():
        return False
    lines, changed = allrun.read_text(errors="ignore").splitlines(), False
    for i, line in enumerate(lines):
        words = line.split()
        helper = words[:1] == ["runApplication"]
        program = words[1:2] if helper else words[:1]
        if program != [solver] or "-parallel" in words:
            continue
        if helper:
            lines[i] = "\n".join(["runApplication decomposePar -force",
                                  "runParallel " + " ".join(words[1:]),
                                  "runApplication reconstructPar"])
        else:
            rest = " ".join(words[1:])
            lines[i] = "\n".join([
                "decomposePar -force > log.decomposePar 2>&1",
                f"mpirun -np {ranks} {solver} -parallel {rest}".rstrip(),
                "reconstructPar > log.reconstructPar 2>&1",
            ])
        changed = True
    if changed:
        allrun.write_text("\n".join(lines) + "\n")
        print(f"[foam] {case_dir.name}: its Allrun runs {solver} on {ranks} ranks, as its "
              "decomposeParDict asks", flush=True)
    return changed


def _read_state(case_dir: Path) -> Dict[str, Any]:
    try:
        return json.loads((Path(case_dir) / ".foamagent_state.json").read_text())
    except (OSError, ValueError):
        return {}


# A second, unread copy misleads: a case-runner edited the unused file and
# "fixed" a viscosity that was already right (qwen_retest_20261005/cavity_r11).
def drop_unread_transport_properties(case_dir: Path) -> bool:
    """Remove constant/transportProperties when constant/physicalProperties is
    present and the installation reads physicalProperties (openfoam.org 10+).
    openfoam.org 9 and openfoam.com read transportProperties, so there both
    stay. Returns whether a file was removed."""
    from cfd_langgraph.openfoam_version import get_profile
    from cfd_langgraph.solver_shelf import own_runtime

    constant = Path(case_dir) / "constant"
    stale = constant / "transportProperties"
    # A case on a solver's own OpenFOAM is judged by that OpenFOAM, not the
    # machine's: DAFoam's v2506 reads transportProperties, and removed here by
    # OpenFOAM 10's rule, DAFoam stopped at start-up for want of it
    # (shelf_demo_dafoam_20261019_1057, case_004).
    if own_runtime(case_dir):
        return False
    if get_profile().get("files", {}).get("transport") != "physicalProperties":
        return False
    if (constant / "physicalProperties").is_file() and stale.is_file():
        stale.unlink()
        print(f"[foam] {Path(case_dir).name}: removed constant/transportProperties "
              "(physicalProperties is the file this OpenFOAM reads)", flush=True)
        return True
    return False


def _read_solver_from_control_dict(case_dir: Path) -> str:
    """What runs the case: its `application`, or foamRun when it names a
    solver module (openfoam.org 11+, where `application` is often absent)."""
    from cfd_langgraph.openfoam_version import case_solver

    return case_solver(case_dir)["application"]


def _n_cells(case_dir: Path) -> Optional[int]:
    """Cell count from the header OpenFOAM writes into polyMesh/owner (or
    owner.gz: DAFoam's OpenFOAM writes its mesh compressed)."""
    import gzip

    poly = Path(case_dir) / "constant" / "polyMesh"
    if (poly / "owner").is_file():
        with (poly / "owner").open(errors="ignore") as f:
            head = f.read(4000)
    elif (poly / "owner.gz").is_file():
        try:
            with gzip.open(poly / "owner.gz", "rt", errors="ignore") as f:
                head = f.read(4000)
        except OSError:
            return None
    else:
        return None
    i = head.find("nCells:")
    if i < 0:
        return None
    digits = ""
    for ch in head[i + len("nCells:"):].lstrip():
        if not ch.isdigit():
            break
        digits += ch
    return int(digits) if digits else None


def _mesh_extents(case_dir: Path) -> Optional[Tuple[List[float], List[float], str]]:
    """(min xyz, max xyz, source) of the case's mesh: from
    constant/polyMesh/points when there is one (ascii, plain or .gz),
    otherwise from blockMeshDict's vertices and scale."""
    import gzip

    import numpy as np

    def floats(text: str) -> Optional[List[float]]:
        try:
            return [float(w) for w in text.replace("(", " ").replace(")", " ").split()]
        except ValueError:
            return None

    poly = Path(case_dir) / "constant" / "polyMesh"
    for name in ("points", "points.gz"):
        path = poly / name
        if not path.is_file():
            continue
        try:
            opener = gzip.open if name.endswith(".gz") else open
            with opener(path, "rt", errors="ignore") as f:
                text = f.read()
        except OSError:
            return None
        header_end = text.find("}") if "FoamFile" in text[:2000] else -1
        if "binary" in text[:max(header_end, 0)]:
            return None
        body = text[header_end + 1:]
        start, end = body.find("("), body.rfind(")")
        values = floats(body[start + 1:end]) if 0 <= start < end else None
        if not values or len(values) % 3:
            return None
        xyz = np.array(values).reshape(-1, 3)
        return xyz.min(0).tolist(), xyz.max(0).tolist(), "the mesh points"
    bmd = Path(case_dir) / "system" / "blockMeshDict"
    if not bmd.is_file():
        return None
    text = "\n".join(line.split("//")[0] for line in bmd.read_text(errors="ignore").splitlines())
    scale = 1.0
    for key in ("convertToMeters", "scale"):
        i = text.find(key)
        if i >= 0:
            words = text[i + len(key):text.find(";", i)].split()
            got = floats(" ".join(words)) if words else None
            if not got or len(got) != 1:
                return None
            scale = got[0]
            break
    i = text.find("vertices")
    start = text.find("(", i) if i >= 0 else -1
    if start < 0:
        return None
    depth, end = 0, -1
    for j in range(start, len(text)):
        depth += {"(": 1, ")": -1}.get(text[j], 0)
        if depth == 0:
            end = j
            break
    values = floats(text[start + 1:end]) if end > start else None
    if not values or len(values) % 3:
        return None
    xyz = np.array(values).reshape(-1, 3) * scale
    return xyz.min(0).tolist(), xyz.max(0).tolist(), "the blockMeshDict vertices (curved edges may bulge past them)"


def mesh_summary(case_dir: Path) -> str:
    """The case's mesh as measured, for any prompt that must not guess its
    size: cell count, extents, patches. A requirement written from a README
    that said "54 x 54 x 54 cells" made the 1 m channel 54 m long and put the
    study's whole truth field outside it (shelf_demo_dafoam_20261015_1214)."""
    parts = []
    cells = _n_cells(case_dir)
    if cells:
        parts.append(f"{cells} cells")
    extents = _mesh_extents(case_dir)
    if extents:
        lo, hi, source = extents
        parts.append(", ".join(f"{axis} from {a:.4g} to {b:.4g}" for axis, a, b in zip("xyz", lo, hi))
                     + f" (in the mesh's units, normally m; measured from {source})")
    patches = _mesh_patches(case_dir)
    if patches:
        parts.append("patches: " + ", ".join(f"{name} ({kind})" for name, kind in patches))
    return "; ".join(parts)


# Most cells one refinement step may have, as a multiple of its parent's: a
# 1.5x step in each direction the mesh resolves, 2.25 in 2-D (REFINE_BUDGET)
# and 3.375 in 3-D (refine_budget). The 2-D figure also held 3-D meshes to
# 1.31x per direction, and a 3-D channel graded to both walls, 7x7x7 cells,
# could then not be refined without coarsening somewhere: eleven rewrites
# were rejected (shelf_demo_dafoam_20261011_1437).
REFINE_STEP = 1.5
REFINE_BUDGET = float(os.environ.get("CFD_SCIENTIST_REFINE_BUDGET") or REFINE_STEP ** 2)
# Below this many cells a step costs nothing worth limiting, so any step is
# allowed: refining by whole multiples is what keeps a graded mesh's cells
# from shifting.
REFINE_FREE_CELLS = 20000


def refine_budget(case_dir: Path) -> float:
    """The cell budget of one refinement step from ``case_dir``."""
    if os.environ.get("CFD_SCIENTIST_REFINE_BUDGET"):
        return REFINE_BUDGET
    try:
        from cfd_langgraph.foam_load import patch_types

        planar = "wedge" in set(patch_types(case_dir).values())
    except Exception:  # noqa: BLE001
        planar = False
    planar = planar or _layers_along_empty_axis(case_dir) == 1
    return REFINE_STEP ** (2 if planar else 3)


def _layers_along_empty_axis(case_dir: Path) -> Optional[int]:
    """Cells across the out-of-plane direction of a 2-D case (one with an
    empty patch), or None if the case is 3-D or its mesh cannot be read.
    Read from the mesh alone, so an unsolved level can be checked."""
    try:
        import numpy as np

        from cfd_langgraph.foam_load import patch_types
        from cfd_langgraph.mesh_assessment import _read_mesh

        if "empty" not in set(patch_types(case_dir).values()):
            return None
        points = np.asarray(_read_mesh(Path(case_dir), solved=False)[0].points)
        span = np.ptp(points, axis=0)
        span[span == 0] = 1.0
        counts = [len(np.unique(np.round((points[:, a] - points[:, a].min()) / span[a], 6)))
                  for a in range(3)]
        return max(1, min(counts) - 1)
    except Exception:  # noqa: BLE001
        return None


# Checked in code: a reviewer passed an unchanged blockMeshDict and a coarser mesh,
# and the gate then compared a mesh with itself (qwen_retest_20261005/cavity).
def _not_finer(case_dir: Path, parent_dir: Path, parent_cells: Optional[int] = None) -> Optional[str]:
    """Why a refined mesh is not a valid refinement step of its parent (unchanged,
    not finer, over the cell budget, or no longer 2-D), or None. ``parent_cells``
    replaces the parent's count when its final mesh is not a blockMesh mesh."""
    def bmd(d: Path) -> str:
        f = d / "system" / "blockMeshDict"
        return " ".join(f.read_text(errors="ignore").split()) if f.is_file() else ""

    if bmd(case_dir) and bmd(case_dir) == bmd(parent_dir):
        return ("The refined blockMeshDict is identical to the parent's: nothing was refined. "
                "Increase the cell counts where the refinement instruction says.")
    child = _n_cells(case_dir)
    parent = parent_cells if parent_cells is not None else _n_cells(parent_dir)
    if child is not None and parent is not None and child <= parent:
        return (f"The refined mesh has {child} cells and its parent {parent}: it is not finer. A "
                "refinement must add cells -- increase the cell counts where the refinement "
                "instruction says, and do not reduce them anywhere.")
    layers = _layers_along_empty_axis(parent_dir), _layers_along_empty_axis(case_dir)
    if layers[0] == 1 and layers[1] not in (None, 1):
        return (f"This is a 2-D case: the out-of-plane direction (its empty patches) must keep "
                f"exactly one cell, but the refined mesh has {layers[1]} there. Refine only the "
                "two in-plane directions.")
    # The budget is a number, so it is checked here rather than by a reviewer:
    # on qwen_retest_20261005/cavity_r4 the reviewer rejected a step of 1.98x
    # as over a budget of "about twice", five times, and the gate stopped.
    # 2.25 admits a uniform 1.5x step in two directions.
    budget = refine_budget(parent_dir)
    if child is not None and parent is not None and child > budget * parent and child > REFINE_FREE_CELLS:
        return (f"The refined mesh has {child} cells, {child / parent:.2f} times its parent's {parent}: "
                f"one refinement step may add at most {budget:g} times the cells. Refine only "
                "the regions and directions the instruction names, or by a smaller factor.")
    return None


def _foam_tokens(text: str) -> List[Any]:
    """An OpenFOAM list as nested Python lists of word tokens."""
    stack: List[List[Any]] = [[]]
    word = ""
    for ch in text + " ":
        if ch in "()" or ch.isspace():
            if word:
                stack[-1].append(word)
                word = ""
            if ch == "(":
                stack.append([])
            elif ch == ")":
                if len(stack) == 1:
                    raise ValueError("unbalanced ')'")
                done = stack.pop()
                stack[-1].append(done)
        else:
            word += ch
    if len(stack) != 1:
        raise ValueError("unbalanced '('")
    return stack[0]


def _foam_text(item: Any) -> str:
    if isinstance(item, list):
        return "(" + " ".join(_foam_text(x) for x in item) + ")"
    return str(item)


def refine_block_counts(block_mesh_dict: Path, env: Dict[str, str],
                        budget: float = REFINE_BUDGET, factor: Optional[float] = None) -> Optional[str]:
    """Refine every block of a blockMeshDict uniformly, in code: each direction
    with more than one cell is multiplied by the same factor -- ``factor`` if
    given, else chosen so the total grows by at most ``budget``. Directions with one cell (the empty or
    wedge direction) stay at one, and equal counts stay equal, so blocks that
    share a face still match. Returns None on success, else why it could not."""
    path = Path(block_mesh_dict)
    read = subprocess.run(["foamDictionary", "-entry", "blocks", "-value", "-expand", str(path)],
                          env=env, capture_output=True, text=True)
    if read.returncode != 0 or not read.stdout.strip():
        return f"foamDictionary could not read the blocks: {(read.stderr or '').strip()[-300:]}"
    try:
        top = _foam_tokens(read.stdout)
    except ValueError as exc:
        return f"the blocks entry could not be parsed: {exc}"
    if len(top) != 1 or not isinstance(top[0], list):
        return "the blocks entry is not a single list"
    blocks = top[0]
    counts: List[List[Any]] = []
    for i, item in enumerate(blocks):
        if item != "hex":
            continue
        # After the vertex list: an optional zone name, then the cell counts.
        found = None
        for nxt in blocks[i + 2:]:
            if nxt == "hex":
                break
            if isinstance(nxt, list) and len(nxt) == 3 and all(
                    isinstance(x, str) and x.isdigit() for x in nxt):
                found = nxt
                break
        if found is None:
            return f"block {len(counts)} has no cell counts this step can read"
        counts.append(found)
    if not counts:
        return "the blockMeshDict has no hex blocks"
    dims = max(sum(1 for x in c if int(x) > 1) for c in counts)
    if dims == 0:
        return "no block has more than one cell in any direction"
    factor = factor or budget ** (1.0 / dims)
    for c in counts:
        for k, x in enumerate(c):
            n = int(x)
            if n > 1:
                c[k] = str(max(n + 1, int(n * factor + 1e-9)))
    write = subprocess.run(["foamDictionary", "-entry", "blocks", "-set", _foam_text(blocks), str(path)],
                           env=env, capture_output=True, text=True)
    if write.returncode != 0:
        return f"foamDictionary could not write the blocks: {(write.stderr or '').strip()[-300:]}"
    return None


def refine_mesh_from_parent(
    llm: Any,
    case_dir: Path,
    base_case_dir: Path,
    refine_instruction: str,
    *,
    case_solver: str = "",
    max_loop: int = 10,
    max_time_limit_s: int = 21600,
    openfoam_path: str = "",
    mesh_review: Optional[Callable[[Path], Optional[str]]] = None,
    max_mesh_reviews: int = 4,
    uniform: bool = False,
) -> Dict[str, Any]:
    """Copy an existing case's files wholesale and refine only its mesh —
    with ``uniform``, every block is refined by one factor in code
    (:func:`refine_block_counts`) and the model is used only if that fails —
    the ``base_case_dir`` mesh-copy-and-edit capability
    ``scripts/foam_run.py --base-case-dir --mesh-gate-role refined`` used to
    provide, ported here so mesh-independence checking doesn't need that
    (currently broken — version-mismatched) vendored-Foam-Agent path.

    Unlike :func:`run_foam_case`'s normal flow (parse -> RAG -> decompose ->
    write every file from scratch), this reuses every file from
    ``base_case_dir`` unchanged except ``system/blockMeshDict``, which gets a
    single targeted edit asking for the requested refinement. That's the
    actual point: physics, boundary conditions, and solver settings carry
    over exactly, which a from-scratch rewrite of the whole case can't
    guarantee even when explicitly instructed to "keep everything the same
    except the mesh."
    """
    t0 = time.monotonic()
    case_dir = Path(case_dir)
    base_case_dir = Path(base_case_dir)
    _copy_case_files(base_case_dir, case_dir)
    _drop_old_mesh_fields(case_dir)
    # A refined level is the same case at another resolution, so it inherits
    # the parent's provenance; _copy_case_files carries only 0/, constant/
    # and system/.
    inherited = _read_state(base_case_dir)
    own_allrun = bool(inherited.get("own_allrun")) and _copy_own_allrun(base_case_dir, case_dir)
    carried = {k: inherited[k] for k in ("seed_source", "seed_name") if k in inherited}
    if carried or own_allrun:
        (case_dir / ".foamagent_state.json").write_text(
            json.dumps({**carried, "own_allrun": own_allrun}, indent=2))

    solver = case_solver or _read_solver_from_control_dict(case_dir) or _default_solver("simpleFoam")

    block_mesh_path = case_dir / "system" / "blockMeshDict"
    parent_bmd = base_case_dir / "system" / "blockMeshDict"
    allrun_env = case_env(openfoam_path)
    # The parent's own Allrun may change its mesh after blockMesh
    # (snappyHexMesh adds and removes cells), so its final mesh is no measure
    # of a refinement of blockMeshDict: count the parent's blockMesh mesh.
    parent_cells: Optional[int] = None
    if own_allrun and block_mesh_path.is_file():
        subprocess.run(["bash", "-c", "blockMesh > log.blockMesh 2>&1"],
                       cwd=str(case_dir), env=allrun_env, capture_output=True, text=True)
        parent_cells = _n_cells(case_dir)
    # Uniform refinement is arithmetic, so it is done in code: asked for 1.5x,
    # a model wrote 1458x1458 for a 192x192 parent (qwen_retest_20261005/cavity_r13).
    refined_in_code = False
    if uniform and block_mesh_path.is_file():
        why_not = refine_block_counts(block_mesh_path, allrun_env, budget=refine_budget(base_case_dir))
        refined_in_code = why_not is None
        print(f"[foam] {case_dir.name}: "
              + ("refined every block uniformly in code" if refined_in_code
                 else f"could not refine in code ({why_not}); asking the model"), flush=True)
    if refined_in_code:
        mesh_review = None
    else:
        current_bmd = parent_bmd.read_text(encoding="utf-8", errors="ignore") if parent_bmd.is_file() else ""
        block_mesh_path.parent.mkdir(parents=True, exist_ok=True)
        block_mesh_path.write_text(writer.edit_file(
            llm,
            file_name="blockMeshDict",
            folder_name="system",
            changes=refine_instruction,
            current_content=current_bmd,
            written_files_ctx="",
            case_solver=solver,
        ))

    def _reviewed_mesh() -> Optional[str]:
        """Mesh only and check the result against the parent before any
        solver time is spent -- first in code (a refinement must add cells),
        then with ``mesh_review`` -- and rewrite with the correction. Every
        version is checked, the last one included: returns None once a
        version passes, or the last correction if none did -- a mesh that
        failed the check is never solved."""
        correction: Optional[str] = None
        for attempt in range(max_mesh_reviews + 1):
            proc = subprocess.run(
                ["bash", "-c", "blockMesh > log.blockMesh 2>&1 && checkMesh > log.checkMesh 2>&1"],
                cwd=str(case_dir), env=allrun_env, capture_output=True, text=True,
            )
            if proc.returncode != 0:
                return None  # the solve loop below diagnoses a mesh that does not build
            correction = _not_finer(case_dir, base_case_dir, parent_cells) or (
                mesh_review(case_dir) if mesh_review is not None else None)
            if not correction:
                return None
            if attempt == max_mesh_reviews:
                break
            print(f"[foam] {case_dir.name}: refinement rejected before solving ({correction[:200]}); "
                  "rewriting the mesh from the parent's", flush=True)
            # From the parent's file each time: editing a rejected version
            # compounds its error (r13 above: four corrections, none passed).
            block_mesh_path.write_text(writer.edit_file(
                llm,
                file_name="blockMeshDict",
                folder_name="system",
                changes=(f"{refine_instruction}\n\nA previous attempt at this refinement was rejected: "
                         f"{correction}\nStart again from the parent's file below."),
                current_content=parent_bmd.read_text(encoding="utf-8", errors="ignore"),
                written_files_ctx="",
                case_solver=solver,
            ))
        print(f"[foam] {case_dir.name}: no version of the refined mesh passed the check; not solving it",
              flush=True)
        return correction

    mesh_rejection = _reviewed_mesh()
    # No targeted rewrite passed: refine every block uniformly in code from the
    # parent's file instead -- by a whole factor of 2 when the result is still
    # small, so a graded block's cells cannot shift. In a 3-D channel graded
    # to both walls, every targeted rewrite left some cells coarser, sixteen
    # over three calls (shelf_demo_dafoam_20261011_1437).
    if mesh_rejection and not refined_in_code and parent_bmd.is_file():
        shutil.copy2(parent_bmd, block_mesh_path)
        cells = parent_cells if parent_cells is not None else _n_cells(base_case_dir)
        doubled = cells is not None and cells * 8 <= REFINE_FREE_CELLS
        why_not = (refine_block_counts(block_mesh_path, allrun_env, factor=2.0) if doubled else
                   refine_block_counts(block_mesh_path, allrun_env, budget=refine_budget(base_case_dir)))
        if why_not is None:
            print(f"[foam] {case_dir.name}: no targeted rewrite passed; refined every block "
                  f"uniformly in code instead{' by 2' if doubled else ''}", flush=True)
            mesh_review = None
            mesh_rejection = _reviewed_mesh()

    allrun_path = case_dir / "Allrun"
    if not own_allrun:
        allrun_path.write_text(allrun_mod.build_mesh_and_solve_allrun(solver, case_dir=case_dir))
        allrun_path.chmod(0o755)

    loop_count = 0
    success = False
    history: List[str] = []
    mesh_subtask = [{"file_name": "blockMeshDict", "folder_name": "system"}]
    # Same reuse hazard as run_foam_case: the mesh gate reuses its
    # baseline/refined_* directories across re-runs, so a re-run gate could
    # otherwise "converge" on a mesh that was never actually built.
    _clean_stale_run_artifacts(case_dir)
    for loop_count in range(1, (0 if mesh_rejection else max_loop) + 1):
        remaining = max(60, max_time_limit_s - int(time.monotonic() - t0))
        returncode, out, err, _ = run_script(["./Allrun"], case_dir, allrun_env, remaining)
        allrun_out = out + "\n" + err
        (case_dir / "Allrun.out").write_text(allrun_out)

        clean = (
            returncode == 0
            and _solver_ended_cleanly(case_dir, solver)
            and not _has_fatal(case_dir)
            and not mesh_check_errors(case_dir)
        )
        if clean:
            success = True
            break
        if loop_count >= max_loop:
            break

        # The review/rewrite loop here only ever targets blockMeshDict — the
        # physics/BC/solver files copied from the parent are never touched,
        # matching "MESH CHANGE ONLY" from the same protocol run_foam_case
        # follows for its own refined-level requirement text.
        analysis = review.review_errors(
            llm,
            tutorial_reference="",
            foamfiles_xml=_foamfiles_xml(case_dir, mesh_subtask),
            error_logs=collect_error_logs(case_dir),
            user_requirement=refine_instruction,
            history_text="\n".join(f"Prior attempt {i + 1}: {h}" for i, h in enumerate(history)),
        )
        history.append(analysis[:500])
        current_bmd = block_mesh_path.read_text(encoding="utf-8", errors="ignore")
        refined_bmd = writer.edit_file(
            llm,
            file_name="blockMeshDict",
            folder_name="system",
            changes=f"Fix this mesh so it builds and runs cleanly, based on this diagnosis: {analysis[:1000]}",
            current_content=current_bmd,
            written_files_ctx="",
            case_solver=solver,
        )
        block_mesh_path.write_text(refined_bmd)
        mesh_rejection = _reviewed_mesh()
        if mesh_rejection:
            break
        _clean_stale_run_artifacts(case_dir)

    wall_time_s = round(time.monotonic() - t0, 1)
    status = "success" if success else "failed"
    run_result = {
        "status": status,
        "success": success,
        "case_dir": str(case_dir),
        "case_name": case_dir.name,
        "case_solver": solver,
        "base_case_dir": str(base_case_dir),
        "loop_count": loop_count,
        "max_loop": max_loop,
        "wall_time_s": wall_time_s,
        "error_logs": [] if success else [error_summary(case_dir)],
    }
    if mesh_rejection and not success:
        run_result["mesh_check_failed"] = mesh_rejection[:3000]
    (case_dir / "run_result.json").write_text(json.dumps(run_result, indent=2))
    return run_result


def extend_run(
    case_dir: Path,
    *,
    factor: float = 1.0,
    openfoam_path: str = "",
    max_time_limit_s: int = 21600,
) -> Dict[str, Any]:
    """Continue a solved transient case from its latest time on the same mesh,
    lengthening the run by ``factor`` times its current end time.

    For a solution whose monitored quantities have not settled yet: running
    longer, not a finer mesh, is what answers that. The mesh and the solved
    fields are left as they are; only endTime, startFrom and the start of the
    gate's own averaging window change.
    """
    case_dir = Path(case_dir)
    from cfd_langgraph.solver_shelf import own_runtime

    if own_runtime(case_dir).get("run"):
        return {"status": "failed", "error": "this case is solved by its own script, which cannot be "
                                             "continued from a later time"}
    cd_path = case_dir / "system" / "controlDict"
    text = cd_path.read_text(errors="ignore")
    m = re.search(r"\bendTime\s+([0-9.eE+-]+)\s*;", text)
    if not m:
        return {"status": "failed", "error": "endTime not found in controlDict"}
    end = float(m.group(1))
    times = [t for t in (parse_time_dir(p.name) for p in case_dir.iterdir() if p.is_dir()) if t]
    latest = max(times) if times else 0.0
    if latest <= 0.0:
        return {"status": "failed", "error": "no solved time to continue from"}
    new_end = end * (1.0 + factor)
    text = text[: m.start(1)] + f"{new_end:g}" + text[m.end(1):]
    text = re.sub(r"\bstartFrom\s+\w+\s*;", "startFrom latestTime;", text)
    # The gate's own averaging restarts at the continuation, so the averaged
    # fields cover the later, settled part of the run.
    block = re.search(r"gateFieldAverage\s*\{(.*?)\n\s*\}", text, re.S)
    if block:
        body = re.sub(r"timeStart\s+[0-9.eE+-]+\s*;", f"timeStart {latest:g};", block.group(1))
        if "restartOnRestart" not in body:
            body = body + "\n        restartOnRestart yes;"
        text = text[: block.start(1)] + body + text[block.end(1):]
    cd_path.write_text(text)

    solver = _read_solver_from_control_dict(case_dir) or _default_solver("pimpleFoam")
    parallel = (case_dir / "system" / "decomposeParDict").is_file() and any(case_dir.glob("processor*"))
    lines = ["#!/bin/sh", 'cd "${0%/*}" || exit 1', '. "$WM_PROJECT_DIR/bin/tools/RunFunctions"', ""]
    if parallel:
        lines += ["rm -rf processor*",
                  "runApplication -o decomposePar -force -latestTime",
                  f"runParallel -o {solver}",
                  "runApplication -o reconstructPar -newTimes"]
    else:
        lines += [f"runApplication -o {solver}"]
    script = case_dir / "Allrun.extend"
    script.write_text("\n".join(lines) + "\n")
    script.chmod(0o755)

    t0 = time.monotonic()
    returncode, out, err, timed_out = run_script(["./Allrun.extend"], case_dir,
                                                 case_env(openfoam_path), max_time_limit_s)
    out = out + "\n" + err + ("\ntimed out" if timed_out else "")
    (case_dir / "Allrun.extend.out").write_text(out)
    ok = returncode == 0 and _solver_ended_cleanly(case_dir, solver) and not _has_fatal(case_dir)
    result_path = case_dir / "run_result.json"
    try:
        result = json.loads(result_path.read_text()) if result_path.is_file() else {}
    except Exception:  # noqa: BLE001
        result = {}
    result.setdefault("extensions", []).append(
        {"from": latest, "to": new_end, "ok": ok, "wall_time_s": round(time.monotonic() - t0, 1)})
    if not ok:
        result["status"], result["success"] = "failed", False
    result_path.write_text(json.dumps(result, indent=2))
    print(f"[foam] {case_dir.name}: continued from t={latest:g} to t={new_end:g} "
          f"({'ok' if ok else 'FAILED'})", flush=True)
    return {"status": "success" if ok else "failed", "from": latest, "to": new_end}
