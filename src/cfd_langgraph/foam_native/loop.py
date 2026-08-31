from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List

from . import allrun as allrun_mod
from . import decomposer, parser, rag, review, writer
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


def _solver_ended_cleanly(case_dir: Path, solver: str) -> bool:
    log_path = case_dir / f"log.{solver}"
    if not log_path.is_file():
        return False
    tail = log_path.read_text(encoding="utf-8", errors="ignore")[-2000:]
    return tail.rstrip().endswith("End")


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


def _clean_stale_run_artifacts(case_dir: Path) -> None:
    """Remove everything a previous ``./Allrun`` attempt left behind that
    would make the next attempt skip work or read someone else's results.

    Three distinct hazards, all of which have bitten this pipeline:

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

    Call this before the first attempt as well as between retries: case
    directories are reused across relaunches (a RERUN case is handed back to
    a subagent with the same ``case_id``, and the mesh gate reuses its
    baseline/refined dirs), so "first attempt" does not imply "clean dir".
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


def _run_seeded_case(
    base_case_seed_dir: Path,
    case_dir: Path,
    *,
    openfoam_path: str,
    max_loop: int,
    max_time_limit_s: int,
    t0: float,
) -> Dict[str, Any]:
    """Copy a validated case and run it, unmodified. No LLM calls at all."""
    _copy_case_files(base_case_seed_dir, case_dir)
    solver = _read_solver_from_control_dict(case_dir) or "simpleFoam"
    (case_dir / ".foamagent_state.json").write_text(
        json.dumps({"case_info": {"case_solver": solver}}, indent=2)
    )
    allrun_path = case_dir / "Allrun"
    allrun_path.write_text(allrun_mod.build_allrun_script("blockMesh", case_solver=solver))
    allrun_path.chmod(0o755)
    allrun_env = resolve_openfoam_env(openfoam_path)
    _clean_stale_run_artifacts(case_dir)

    remaining = max(60, max_time_limit_s - int(time.monotonic() - t0))
    try:
        proc = subprocess.run(
            ["./Allrun"], cwd=str(case_dir), env=allrun_env,
            capture_output=True, text=True, timeout=remaining,
        )
        allrun_out = (proc.stdout or "") + "\n" + (proc.stderr or "")
        returncode = proc.returncode
    except subprocess.TimeoutExpired as exc:
        allrun_out = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
        returncode = -1
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
    return {
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

    # Stage 6 — Allrun
    command_text = allrun_mod.generate_allrun_commands(
        llm, dir_structure=refs.get("dir_structure", ""), case_info=case_info,
        allrun_reference=refs.get("allrun_reference", ""), mesh_type=mesh_type,
    )
    allrun_path = case_dir / "Allrun"
    allrun_path.write_text(
        allrun_mod.build_allrun_script(command_text, case_solver=case_info["case_solver"])
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
        remaining = max(60, max_time_limit_s - int(time.monotonic() - t0))
        try:
            proc = subprocess.run(
                ["./Allrun"], cwd=str(case_dir), env=allrun_env,
                capture_output=True, text=True, timeout=remaining,
            )
            allrun_out = (proc.stdout or "") + "\n" + (proc.stderr or "")
            returncode = proc.returncode
        except subprocess.TimeoutExpired as exc:
            allrun_out = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
            returncode = -1
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


def _read_solver_from_control_dict(case_dir: Path) -> str:
    path = case_dir / "system" / "controlDict"
    if not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"\bapplication\s+(\w+)\s*;", text)
    return m.group(1) if m else ""


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
) -> Dict[str, Any]:
    """Copy an existing case's files wholesale and refine only its mesh —
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

    solver = case_solver or _read_solver_from_control_dict(case_dir) or "simpleFoam"

    block_mesh_path = case_dir / "system" / "blockMeshDict"
    current_bmd = block_mesh_path.read_text(encoding="utf-8", errors="ignore") if block_mesh_path.is_file() else ""
    refined_bmd = writer.edit_file(
        llm,
        file_name="blockMeshDict",
        folder_name="system",
        changes=refine_instruction,
        current_content=current_bmd,
        written_files_ctx="",
        case_solver=solver,
    )
    block_mesh_path.parent.mkdir(parents=True, exist_ok=True)
    block_mesh_path.write_text(refined_bmd)

    allrun_path = case_dir / "Allrun"
    allrun_path.write_text(allrun_mod.build_allrun_script(f"blockMesh\ncheckMesh\n{solver}", case_solver=solver))
    allrun_path.chmod(0o755)

    allrun_env = resolve_openfoam_env(openfoam_path)
    loop_count = 0
    success = False
    history: List[str] = []
    mesh_subtask = [{"file_name": "blockMeshDict", "folder_name": "system"}]
    # Same reuse hazard as run_foam_case: the mesh gate reuses its
    # baseline/refined_* directories across re-runs, so a re-run gate could
    # otherwise "converge" on a mesh that was never actually built.
    _clean_stale_run_artifacts(case_dir)
    for loop_count in range(1, max_loop + 1):
        remaining = max(60, max_time_limit_s - int(time.monotonic() - t0))
        try:
            proc = subprocess.run(
                ["./Allrun"], cwd=str(case_dir), env=allrun_env,
                capture_output=True, text=True, timeout=remaining,
            )
            allrun_out = (proc.stdout or "") + "\n" + (proc.stderr or "")
            returncode = proc.returncode
        except subprocess.TimeoutExpired as exc:
            allrun_out = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
            returncode = -1
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
    (case_dir / "run_result.json").write_text(json.dumps(run_result, indent=2))
    return run_result
