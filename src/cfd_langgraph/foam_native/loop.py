from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from . import allrun as allrun_mod
from . import decomposer, parser, rag, review, writer
from .case_clean import parse_time_dir, remove_stale_time_dirs
from .openfoam_env import resolve_openfoam_env

_CASE_FILE_DIRS = ("0", "constant", "system")


def _safe_case_relative_path(folder_name: str, file_name: str) -> Path:
    rel = Path(str(folder_name or "")) / str(file_name or "")
    if rel.is_absolute() or ".." in rel.parts or not rel.name:
        raise ValueError(f"Unsafe FoamAgent case path: {rel}")
    return rel


def _foamfiles_xml(case_dir: Path, subtasks: List[Dict[str, str]], max_file_kb: int = 30, max_total_kb: int = 300) -> str:
    parts: List[str] = []
    total = 0
    for st in subtasks:
        p = case_dir / st["folder_name"] / st["file_name"]
        if not p.is_file():
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
        if len(text) > max_file_kb * 1024:
            text = text[: max_file_kb * 1024] + "\n... [truncated]"
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
                f"{st['folder_name']}/{st['file_name']}:\n{p.read_text(encoding='utf-8', errors='ignore')[:2000]}"
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


def collect_error_logs(case_dir: Path, max_lines: int = 200) -> str:
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
            f"log.{solver} ends normally but the solver never advanced past its start time, so "
            "nothing was solved. Check startTime, endTime, deltaT and stopAt in "
            "system/controlDict: endTime must be later than startTime."
        )
    candidates = sorted(case_dir.glob("log.*")) + [case_dir / "Allrun.out"]
    for log_path in candidates:
        if log_path.is_file():
            lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
            parts.append(f"--- {log_path.name} (last {min(len(lines), max_lines)} lines) ---\n" + "\n".join(lines[-max_lines:]))
    return "\n\n".join(parts)


def _has_fatal(case_dir: Path) -> bool:
    for log_path in case_dir.glob("log.*"):
        try:
            text = log_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if "FOAM FATAL ERROR" in text or "FOAM FATAL IO ERROR" in text:
            return True
    return False


# A run that ends at its start time still prints "End"; recorded as a success, its
# initial fields were later judged as if they were a solution (qwen_retest_20261005/cavity_r3).
def _solver_ran_no_steps(case_dir: Path, solver: str) -> bool:
    """Whether the solver's log ends normally without having advanced a time step
    (endTime at or before startTime)."""
    log_path = case_dir / f"log.{solver}"
    if not log_path.is_file():
        return False
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    return "\nTime = " not in text and text.rstrip().endswith(("End", "Finalising parallel run"))


def _solver_ended_cleanly(case_dir: Path, solver: str) -> bool:
    """The solver advanced at least one time step and its log ended normally."""
    log_path = case_dir / f"log.{solver}"
    if not log_path.is_file():
        return False
    if _solver_ran_no_steps(case_dir, solver):
        return False
    tail = log_path.read_text(encoding="utf-8", errors="ignore")[-2000:].rstrip()
    # A parallel run prints "Finalising parallel run" after its "End".
    if tail.endswith("Finalising parallel run"):
        tail = tail[: -len("Finalising parallel run")].rstrip()
    return tail.endswith("End")


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
    generated case has no functions block of its own.
    """
    target = case_dir / "system" / "controlDict"
    source = seed_case_dir / "system" / "controlDict"
    if not target.is_file() or not source.is_file():
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


def _drop_old_mesh_fields(case_dir: Path) -> List[str]:
    """Remove initial-time files holding per-cell or per-face lists.

    After re-meshing, a list sized for the old mesh cannot be read on the new
    one; a solver that needs such a field fails either way, and one that does
    not (function-object output such as yPlus, written at the first time)
    otherwise stops decomposePar, which reads every field in 0/.
    """
    zero = Path(case_dir) / "0"
    dropped: List[str] = []
    if not zero.is_dir():
        return dropped
    for f in sorted(zero.iterdir()):
        if f.is_file() and "nonuniform" in f.read_text(errors="ignore"):
            f.unlink()
            dropped.append(f.name)
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


def run_script(cmd: List[str], cwd: Path | str, env: Optional[Dict[str, str]],
               timeout: Optional[float]) -> tuple[int, str, str, bool]:
    """Run a case script in its own process group and return (returncode,
    stdout, stderr, timed_out). On a timeout, or if this process is
    interrupted, the whole group is stopped: killing only the script would
    leave mpirun and its solver ranks running, still writing into the case."""
    proc = subprocess.Popen(cmd, cwd=str(cwd), env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, start_new_session=True)
    try:
        out, err = proc.communicate(timeout=timeout)
        return proc.returncode, out or "", err or "", False
    except subprocess.TimeoutExpired:
        _stop_group(proc)
        out, err = proc.communicate()
        return -1, out or "", err or "", True
    except BaseException:
        _stop_group(proc)
        raise
    finally:
        # The script can exit while something it started in the background
        # is still running in its group.
        _stop_group(proc, quiet=True)


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
    if prepare is not None:
        prepare(case_dir)
    solver = _read_solver_from_control_dict(case_dir) or "simpleFoam"
    # Record the source case: a copy under a new name has lost the condition
    # its original directory name carried, which scoring depends on.
    (case_dir / ".foamagent_state.json").write_text(
        json.dumps(
            {
                "case_info": {"case_solver": solver},
                "seed_source": str(base_case_seed_dir),
                "seed_name": base_case_seed_dir.name,
            },
            indent=2,
        )
    )
    allrun_path = case_dir / "Allrun"
    allrun_path.write_text(allrun_mod.build_mesh_and_solve_allrun(solver))
    allrun_path.chmod(0o755)
    allrun_env = resolve_openfoam_env(openfoam_path)
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

    # Stage 1 — parse
    case_info = parser.parse_requirement(llm, user_requirement)
    (case_dir / ".foamagent_state.json").write_text(json.dumps({"case_info": case_info}, indent=2))

    # Stage 2 — RAG retrieval (with fallback)
    refs = rag.retrieve_references(
        user_requirement, case_info["case_solver"], case_info.get("case_domain", ""), case_info.get("case_category", "")
    )

    # Stage 3 — decompose into subtasks
    subtasks = decomposer.decompose_subtasks(llm, user_requirement, refs.get("dir_structure", ""))
    for subtask in subtasks:
        _safe_case_relative_path(subtask.get("folder_name", ""), subtask.get("file_name", ""))
    if mesh_type == "standard_mesh" and not any(s["file_name"] == "blockMeshDict" for s in subtasks):
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
    if base_case_seed_dir is not None:
        base_case_seed_dir = Path(base_case_seed_dir)
        _copy_case_files(base_case_seed_dir, case_dir)
        print(f"[foam-native] seeded case from validated base case: {base_case_seed_dir}", flush=True)

    # Stage 5 — write each subtask file. A file that came from the seed is
    # EDITED toward the requirement, never overwritten from nothing, so the
    # validated numerics, boundary conditions and mesh survive.
    for i, st in enumerate(subtasks):
        out_path = case_dir / _safe_case_relative_path(st["folder_name"], st["file_name"])
        seeded_content = (
            out_path.read_text(encoding="utf-8", errors="ignore")
            if (base_case_seed_dir is not None and out_path.is_file())
            else ""
        )
        if seeded_content.strip():
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
                    f"CASE REQUIREMENT:\n{user_requirement}"
                ),
                current_content=seeded_content,
                written_files_ctx=_written_files_ctx(case_dir, subtasks, i),
                case_solver=case_info["case_solver"],
            )
        else:
            content = writer.write_file_initial(
                llm,
                file_name=st["file_name"], folder_name=st["folder_name"],
                user_requirement=user_requirement, tutorial_reference=refs.get("tutorial_reference", ""),
                written_files_ctx=_written_files_ctx(case_dir, subtasks, i),
                case_solver=case_info["case_solver"],
            )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(content)

    # Enforce the mesh selected by the mesh-independence gate.  The case is
    # still generated from its own experiment requirement, but its mesh
    # generator dictionary comes byte-for-byte from the selected level.
    # Previously the gate wrote selected_mesh_spec.json and normal cases
    # ignored it, so the paper simulations could run on unrelated meshes.
    if mesh_seed_case_dir is not None:
        mesh_seed_case_dir = Path(mesh_seed_case_dir)
        seed_bmd = mesh_seed_case_dir / "system" / "blockMeshDict"
        if mesh_type != "standard_mesh" or not seed_bmd.is_file():
            raise ValueError(
                f"Selected mesh cannot be applied: expected standard-mesh blockMeshDict at {seed_bmd}"
            )
        target_bmd = case_dir / "system" / "blockMeshDict"
        target_bmd.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(seed_bmd, target_bmd)

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
                                resolve_openfoam_env(openfoam_path))
        print(f"[foam-native] {case_dir.name} starts {how}", flush=True)

    # Stage 6 — Allrun
    command_text = allrun_mod.generate_allrun_commands(
        llm, dir_structure=refs.get("dir_structure", ""), case_info=case_info,
        allrun_reference=refs.get("allrun_reference", ""), mesh_type=mesh_type,
    )
    allrun_path = case_dir / "Allrun"
    allrun_path.write_text(
        allrun_mod.build_allrun_script(
            command_text, case_solver=case_info["case_solver"],
            needs_blockmesh=(case_dir / "system" / "blockMeshDict").is_file()
            and not (case_dir / "constant" / "polyMesh" / "points").exists(),
        )
    )
    allrun_path.chmod(0o755)

    # Stages 7+8 — run, review-rewrite-retry loop
    loop_count = 0
    success = False
    history: List[str] = []
    # subprocess.run(["./Allrun"], ...) with no env= inherits this process's
    # environment unchanged; if the shell that launched the CLI never
    # sourced OpenFOAM, Allrun fails at its first line ("$WM_PROJECT_DIR"
    # empty), regardless of anything the model discovers via a separate
    # run_shell call (that's a different subprocess's environment, it
    # doesn't propagate here). Resolve the real environment once and reuse
    # it for every retry loop iteration.
    allrun_env = resolve_openfoam_env(openfoam_path)
    # Case directories are reused across relaunches (run_case_native always
    # writes to out_dir/cases/<case_id>, and a RERUN/REVISE case is handed
    # back to a subagent under the same case_id), so the very first attempt
    # of *this* invocation can still be walking into another attempt's logs,
    # decomposition and postProcessing output. Clearing only between retries
    # left the worst case uncovered: a relaunched case whose Allrun no-ops
    # end to end, whose stale solver log still ends in "End", and which is
    # therefore recorded as a success without having run anything.
    _clean_stale_run_artifacts(case_dir)
    for loop_count in range(1, max_loop + 1):
        if prepare is not None:
            prepare(case_dir)  # a reviewer's rewrite may have dropped it
        remaining = max(60, max_time_limit_s - int(time.monotonic() - t0))
        returncode, out, err, _ = run_script(["./Allrun"], case_dir, allrun_env, remaining)
        allrun_out = out + "\n" + err
        (case_dir / "Allrun.out").write_text(allrun_out)

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

        # Stage 8a — reviewer
        analysis = review.review_errors(
            llm,
            tutorial_reference=refs.get("tutorial_reference", ""),
            foamfiles_xml=_foamfiles_xml(case_dir, subtasks),
            error_logs=collect_error_logs(case_dir),
            user_requirement=user_requirement,
            history_text="\n".join(f"Prior attempt {i + 1}: {h}" for i, h in enumerate(history)),
        )
        history.append(analysis[:500])

        # Stage 8b — rewrite plan
        target_files = review.plan_rewrite(
            llm,
            foamfiles_xml=_foamfiles_xml(case_dir, subtasks),
            error_logs=collect_error_logs(case_dir),
            review_analysis=analysis,
            user_requirement=user_requirement,
        )

        # Stage 8c — apply edits
        for t in target_files:
            rel = Path(t["file"])
            if rel.is_absolute() or ".." in rel.parts or not rel.name:
                history.append(f"Reviewer proposed unsafe path and it was rejected: {rel}")
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
            current_content = target_path.read_text(encoding="utf-8", errors="ignore") if target_path.is_file() else ""
            new_content = writer.edit_file(
                llm,
                file_name=file_name, folder_name=folder_name, changes=t["changes"],
                current_content=current_content,
                written_files_ctx=_written_files_ctx(case_dir, subtasks, len(subtasks)),
                case_solver=case_info["case_solver"],
            )
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_text(new_content)

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
        "error_logs": [] if success else [collect_error_logs(case_dir)[-3000:]],
        "rag_fallback": bool(refs.get("fallback")),
        "mesh_seed_case_dir": str(mesh_seed_case_dir) if mesh_seed_case_dir else "",
        "base_case_seed_dir": str(base_case_seed_dir) if base_case_seed_dir else "",
    }
    (case_dir / "run_result.json").write_text(json.dumps(run_result, indent=2))
    return run_result


def _copy_case_files(base_case_dir: Path, case_dir: Path) -> None:
    """Copy a case's real input files (``0/``, ``constant/``, ``system/``) —
    not stale run outputs like ``postProcessing/``, ``log.*``, ``Allrun.out``."""
    case_dir.mkdir(parents=True, exist_ok=True)
    for item in _CASE_FILE_DIRS:
        src = base_case_dir / item
        if src.is_dir():
            shutil.copytree(src, case_dir / item, dirs_exist_ok=True)
    drop_unread_transport_properties(case_dir)


# A second, unread copy misleads: a case-runner edited the unused file and
# "fixed" a viscosity that was already right (qwen_retest_20261005/cavity_r11).
def drop_unread_transport_properties(case_dir: Path) -> bool:
    """Remove constant/transportProperties when constant/physicalProperties is
    present, since OpenFOAM 10 reads only the latter. Returns whether a file was
    removed."""
    constant = Path(case_dir) / "constant"
    stale = constant / "transportProperties"
    if (constant / "physicalProperties").is_file() and stale.is_file():
        stale.unlink()
        print(f"[foam] {Path(case_dir).name}: removed constant/transportProperties "
              "(physicalProperties is the file OpenFOAM 10 reads)", flush=True)
        return True
    return False


def _read_solver_from_control_dict(case_dir: Path) -> str:
    path = case_dir / "system" / "controlDict"
    if not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"\bapplication\s+(\w+)\s*;", text)
    return m.group(1) if m else ""


def _n_cells(case_dir: Path) -> Optional[int]:
    """Cell count from the header OpenFOAM writes into polyMesh/owner."""
    owner = Path(case_dir) / "constant" / "polyMesh" / "owner"
    if not owner.is_file():
        return None
    with owner.open(errors="ignore") as f:
        head = f.read(4000)
    i = head.find("nCells:")
    if i < 0:
        return None
    digits = ""
    for ch in head[i + len("nCells:"):].lstrip():
        if not ch.isdigit():
            break
        digits += ch
    return int(digits) if digits else None


# Most cells one refinement step may have, as a multiple of its parent's.
REFINE_BUDGET = float(os.environ.get("CFD_SCIENTIST_REFINE_BUDGET") or 2.25)


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
def _not_finer(case_dir: Path, parent_dir: Path) -> Optional[str]:
    """Why a refined mesh is not a valid refinement step of its parent (unchanged,
    not finer, over the cell budget, or no longer 2-D), or None."""
    def bmd(d: Path) -> str:
        f = d / "system" / "blockMeshDict"
        return " ".join(f.read_text(errors="ignore").split()) if f.is_file() else ""

    if bmd(case_dir) and bmd(case_dir) == bmd(parent_dir):
        return ("The refined blockMeshDict is identical to the parent's: nothing was refined. "
                "Increase the cell counts where the refinement instruction says.")
    child, parent = _n_cells(case_dir), _n_cells(parent_dir)
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
    if child is not None and parent is not None and child > REFINE_BUDGET * parent:
        return (f"The refined mesh has {child} cells, {child / parent:.2f} times its parent's {parent}: "
                f"one refinement step may add at most {REFINE_BUDGET:g} times the cells. Refine only "
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
                        budget: float = REFINE_BUDGET) -> Optional[str]:
    """Refine every block of a blockMeshDict uniformly, in code: each direction
    with more than one cell is multiplied by the same factor, chosen so the
    total grows by at most ``budget``. Directions with one cell (the empty or
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
    factor = budget ** (1.0 / dims)
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
    parent_state = base_case_dir / ".foamagent_state.json"
    if parent_state.is_file():
        try:
            inherited = json.loads(parent_state.read_text())
        except Exception:
            inherited = {}
        carried = {k: inherited[k] for k in ("seed_source", "seed_name") if k in inherited}
        if carried:
            (case_dir / ".foamagent_state.json").write_text(json.dumps(carried, indent=2))

    solver = case_solver or _read_solver_from_control_dict(case_dir) or "simpleFoam"

    block_mesh_path = case_dir / "system" / "blockMeshDict"
    parent_bmd = base_case_dir / "system" / "blockMeshDict"
    allrun_env = resolve_openfoam_env(openfoam_path)
    # Uniform refinement is arithmetic, so it is done in code: asked for 1.5x,
    # a model wrote 1458x1458 for a 192x192 parent (qwen_retest_20261005/cavity_r13).
    refined_in_code = False
    if uniform and block_mesh_path.is_file():
        why_not = refine_block_counts(block_mesh_path, allrun_env)
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
            correction = _not_finer(case_dir, base_case_dir) or (
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

    allrun_path = case_dir / "Allrun"
    allrun_path.write_text(allrun_mod.build_mesh_and_solve_allrun(solver))
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
        "error_logs": [] if success else [collect_error_logs(case_dir)[-3000:]],
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

    solver = _read_solver_from_control_dict(case_dir) or "pimpleFoam"
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
                                                 resolve_openfoam_env(openfoam_path), max_time_limit_s)
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
