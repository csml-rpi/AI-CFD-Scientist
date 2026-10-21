from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from . import prompts as P
from cfd_langgraph.llm.reply import reply_text

_DEFAULT_COMMANDS = [
    "blockMesh", "snappyHexMesh", "checkMesh", "decomposePar", "reconstructPar",
    "renumberMesh", "foamDictionary", "surfaceFeatureExtract", "topoSet",
    "setFields", "gmshToFoam",
]


def generate_allrun_commands(
    llm: Any,
    *,
    dir_structure: str,
    case_info: Dict[str, Any],
    allrun_reference: str,
    mesh_type: str = "standard_mesh",
    commands: Optional[List[str]] = None,
) -> str:
    """FoamAgent Stage 6 (verbatim prompt + mesh-type-conditional appendices):
    the OpenFOAM command list for the Allrun script."""
    system = P.with_version_facts(P.COMMAND_SYSTEM_PROMPT)
    if mesh_type == "copied_case":
        system += "\n" + P.COMMAND_APPENDIX_COPIED_CASE
    elif mesh_type == "custom_mesh":
        system += "\n" + P.COMMAND_APPENDIX_CUSTOM_MESH
    user = P.COMMAND_USER_PROMPT.format(
        commands=commands or _DEFAULT_COMMANDS,
        dir_structure=dir_structure,
        case_info=json.dumps(case_info),
        allrun_reference=allrun_reference,
    )
    raw = reply_text(llm.invoke([SystemMessage(content=system), HumanMessage(content=user)]))
    return (raw or "").strip()


_COMMAND_TOKEN_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.+-]*$")


def parse_command_list(command_list_text: str) -> List[str]:
    """Bare OpenFOAM application names from whatever shape the model replied in.

    The model is asked for a newline-separated list and does not reliably give
    one: a JSON array (``["blockMesh"]``) has been observed in a real run, and
    the old code prefixed ``runApplication`` to that entire string, producing
    ``runApplication ["blockMesh"]`` — an application name that cannot exist.
    The case then "ran", wrote a file called ``log.[blockMesh]``, never
    executed a solver, and burned all ten retries.

    So: try JSON first, fall back to lines, and strip the punctuation a list
    literal leaves behind. Anything that still doesn't look like a command
    name is dropped rather than passed to the shell.
    """
    text = (command_list_text or "").strip()
    if text.startswith("```"):
        text = "\n".join(line for line in text.splitlines() if not line.strip().startswith("```"))
    raw_items: List[str] = []
    try:
        loaded = json.loads(text)
        if isinstance(loaded, list):
            raw_items = [str(item) for item in loaded]
        elif isinstance(loaded, dict):
            for value in loaded.values():
                if isinstance(value, list):
                    raw_items = [str(item) for item in value]
                    break
    except Exception:
        raw_items = []
    if not raw_items:
        raw_items = text.splitlines()

    commands: List[str] = []
    for raw in raw_items:
        cmd = str(raw).strip().strip("[]").strip().strip(",").strip().strip('"\'').strip()
        cmd = cmd.lstrip("-").strip()
        if not cmd or cmd.startswith("#"):
            continue
        if cmd.startswith(("runApplication", "runParallel")):
            commands.append(cmd)
            continue
        head = cmd.split()[0]
        if not _COMMAND_TOKEN_RE.match(head):
            continue
        commands.append(cmd)
    return commands


def _with_mesh_check(commands: List[str]) -> List[str]:
    """Insert ``checkMesh`` after the mesher when the model left it out.

    Without it an invalid mesh — negative-volume cells, inverted faces — is
    never reported: blockMesh happily writes one and the solver simply
    diverges, so the review loop spends its retries on the wrong files. This
    only adds the diagnostic; checkMesh exits 0 either way, so an Allrun that
    used to succeed still succeeds (the verdict is read from log.checkMesh).
    """
    meshers = {"blockMesh", "snappyHexMesh", "gmshToFoam"}
    if any("checkMesh" in cmd.split() for cmd in commands):
        return commands
    last_mesher = -1
    for i, cmd in enumerate(commands):
        if meshers & set(cmd.split()):
            last_mesher = i
    if last_mesher < 0:
        return commands
    return commands[: last_mesher + 1] + ["checkMesh"] + commands[last_mesher + 1 :]


def build_allrun_script(command_list_text: str, case_solver: str = "", *,
                        needs_blockmesh: bool = False, case_dir: Optional[Path] = None) -> str:
    """Wrap the model's bare command list in the standard FoamAgent Allrun header.

    ``case_solver`` comes from the case's own controlDict ``application``
    entry, and is appended when the parsed list doesn't already run it. Which
    solver a case runs is a fact recorded in the case, not a judgement call —
    and a case whose Allrun omits it produces no solver log, so every retry
    fails for a reason no rewrite of the *dictionaries* can fix.

    ``needs_blockmesh`` is the same kind of fact about the mesh: the case has a
    blockMeshDict and no mesh yet, so blockMesh runs first if the list leaves
    it out; the retry loop rewrites dictionaries, never the Allrun, so a
    missing mesh step could not otherwise be repaired.

    A case on a solver's own OpenFOAM (``case_dir``, see
    cfd_langgraph.solver_shelf.own_runtime) runs the programs that solver
    puts between meshing and solving, and is solved by the command it names
    when its program is not itself run.
    """
    from cfd_langgraph.solver_shelf import own_runtime

    runtime = own_runtime(case_dir) if case_dir is not None else {}
    # A shell variable is not an application: `runApplication $application`
    # left an empty log file named "log." and ran nothing.
    commands = [c for c in parse_command_list(command_list_text) if "$" not in c]
    # The whole Allrun runs inside the solver's wrapper (run_script), so a
    # command naming the wrapper again is reduced to the program. Left in,
    # runApplication logged every step as log.hystrath-run and skipped all but
    # the first as "already run": dsmcFoam+ never ran
    # (shelf_demo_hystrath_20261011_1602, case_002).
    wrapper = Path(str(runtime.get("wrapper") or "")).name
    if wrapper:
        stripped = []
        for cmd in commands:
            words = cmd.split()
            if words[:1] == [wrapper]:
                words = words[1:]
            elif len(words) > 1 and words[0] in ("runApplication", "runParallel") and words[1] == wrapper:
                words = [words[0], *words[2:]]
            if words:
                stripped.append(" ".join(words))
        commands = stripped
    if needs_blockmesh and not any("blockMesh" in cmd.split() for cmd in commands):
        commands.insert(0, "blockMesh")
    # A case with no blockMeshDict has its mesh already (shipped with it, or
    # copied from the study's selected mesh): blockMesh can only fail there,
    # and its FATAL ERROR failed a DAFoam run whose solve was fine; the
    # reviewer then "fixed" the mesh files themselves
    # (shelf_demo_dafoam_20261015_1318, case_003).
    if case_dir is not None and not (Path(case_dir) / "system" / "blockMeshDict").is_file():
        commands = [c for c in commands if "blockMesh" not in c.split()]
    commands = _with_mesh_check(commands)
    meshing = {"blockMesh", "snappyHexMesh", "gmshToFoam", "checkMesh"}
    after_mesh = max((i + 1 for i, cmd in enumerate(commands) if meshing & set(cmd.split())), default=0)
    for k, program in enumerate(p for p in runtime.get("before", [])
                                if not any(p in cmd.split() for cmd in commands)):
        commands.insert(after_mesh + k, program)
    solver = (case_solver or "").strip()
    tail: List[str] = []
    if runtime.get("run"):
        # The program names the solver; the command runs it. Its script
        # decomposes and the tail reconstructs, so a listed decomposePar or
        # reconstructPar is dropped (see _script_solve).
        commands = [c for c in commands if runtime["program"] not in c.split()
                    and not {"decomposePar", "reconstructPar"} & set(c.split())]
        tail = _script_solve(runtime, script_ranks(case_dir))
    elif solver:
        # A solver module runs through foamRun (openfoam.org 11+); an Allrun
        # that already runs the solver under either name is left alone.
        from cfd_langgraph.openfoam_version import get_profile, solver_command

        run = solver_command(get_profile(), solver)
        names = {solver, run.split()[0]}
        if not any(names & set(cmd.split()) for cmd in commands):
            commands.append(run)
        # A case whose decomposeParDict asks for several domains runs on them,
        # decomposed just before its solver, whatever the model's list says:
        # left to the list, a DSMC case set up for 16 ranks ran on one.
        if script_ranks(case_dir) > 1 and not any("decomposePar" in cmd.split() for cmd in commands):
            at = next((i for i, cmd in enumerate(commands)
                       if names & set(cmd.split()[:2])), len(commands))
            commands.insert(at, "decomposePar -force")
        # A case split for a parallel run is solved in parallel and put back
        # together after the solve. The model's list for a hyStrath case ran
        # decomposePar, then reconstructPar, then a serial hy2Foam: one core of
        # eight on a ten-hour solve (shelf_demo_hystrath_20261011_1602, case_003).
        if any("decomposePar" in cmd.split() for cmd in commands):
            reconstruct = [c for c in commands if "reconstructPar" in c.split()]
            commands = [c for c in commands if "reconstructPar" not in c.split()]
            for i, cmd in enumerate(commands):
                words = cmd.split()
                if words and words[0] == "runApplication":
                    words = words[1:]
                if words and words[0] in names and "-parallel" not in words:
                    commands[i] = "runParallel " + " ".join(words)
            commands += reconstruct or ["reconstructPar -latestTime"]

    lines = [
        "#!/bin/sh",
        'cd "${0%/*}" || exit 1',
        '. "$WM_PROJECT_DIR/bin/tools/RunFunctions"',
        "",
    ]
    for cmd in commands:
        if cmd.startswith(("runApplication", "runParallel")):
            lines.append(cmd)
        else:
            lines.append(f"runApplication {cmd}")
    lines += tail
    lines.append("")
    return "\n".join(lines)


def script_ranks(case_dir: Optional[Path]) -> int:
    """How many ranks a case solved by its solver's command runs on: the
    numberOfSubdomains of its system/decomposeParDict, as DAFoam's own cases
    declare it, else 1. The case says it; the framework does not choose. A
    twin-experiment script written for four ranks was started on one, and
    nothing told the model why (shelf_demo_dafoam_20261011_1437)."""
    path = Path(case_dir) / "system" / "decomposeParDict" if case_dir is not None else None
    if path is None or not path.is_file():
        return 1
    words = path.read_text(errors="ignore").replace(";", " ; ").split()
    for word, value in zip(words, words[1:]):
        if word == "numberOfSubdomains" and value.isdigit():
            return max(1, int(value))
    return 1


def script_command(run: str) -> str:
    """A solver's run command as launched under mpirun: a Python script runs
    under ``python -m mpi4py`` (see _script_solve)."""
    words = run.split()
    if words[:1] in (["python"], ["python3"]) and words[1:2] != ["-m"]:
        words[1:1] = ["-m", "mpi4py"]
    return " ".join(words)


def _script_solve(runtime: Dict[str, Any], np: int) -> List[str]:
    """Allrun lines that solve a case by its solver's command (``run``) on
    ``np`` processes, decomposed first when more than one, logged as
    runApplication would log its program. The command is not an OpenFOAM
    program and prints no "End", so the line is added when, and only when,
    the command exits 0.

    A Python script runs under ``python -m mpi4py``, which ends every rank
    when one raises. Run plainly, the failed rank waited in MPI_Finalize and
    the others spun at full load in their next collective, so a script error
    on rank 0 sat as a hang until the stall guard stopped it 45 min later
    (shelf_demo_dafoam_20261011_1437, case_002)."""
    log = f"log.{runtime['program']}"
    # The script decomposes the case itself once it has written its fields
    # (pyDAFoam's runDecomposePar). Decomposed here first, its own decomposePar
    # refused ("already decomposed") and every field the script had written
    # since (varRefFieldInversion) was missing from the processor folders
    # (shelf_demo_dafoam_20261011_1437, case_002). An earlier run's are removed.
    #
    # The rank count is read from system/decomposeParDict when the Allrun
    # runs, not fixed when it is written: the review may change the case's
    # numberOfSubdomains but may not edit this Allrun, and two DAFoam cases
    # whose scripts required 2 ranks were started on 4, run after run, with
    # nothing the review could change (shelf_demo_dafoam_20261019_1057).
    return [
        "np=$(foamDictionary -entry numberOfSubdomains -value system/decomposeParDict 2>/dev/null)",
        f'[ -n "$np" ] || np={np}',
        'if [ "$np" -gt 1 ]; then rm -rf processor*; fi',
        f"mpirun -np $np {script_command(runtime['run'])} > {log} 2>&1 && echo End >> {log}",
        # Putting the fields back together is a convenience, not the run: a
        # script that writes its results itself leaves no time folder, and
        # reconstructPar's "No times selected" exit failed every such DAFoam
        # gradient check that had completed (shelf_demo_dafoam_20261019_1057,
        # case_004). A failed solve is still told by its log's missing "End".
        "if [ -d processor0 ]; then runApplication reconstructPar || "
        "echo 'reconstructPar had nothing to reconstruct (see log.reconstructPar)'; fi",
    ]


def build_mesh_and_solve_allrun(case_solver: str, *, max_procs: int = 16, cells_per_proc: int = 8000,
                                case_dir: Optional[Path] = None, mesh_given: bool = False) -> str:
    """Allrun for a case whose dictionaries are already complete: mesh it,
    check the mesh, then solve -- in parallel when the mesh is large enough to
    gain from it. The core count is decided after blockMesh, from the cell
    count, so a refined level gets more cores than its parent.

    Given ``case_dir``: a mesh shipped with the case (constant/polyMesh and no
    blockMeshDict, or ``mesh_given``) is used as it is, and a case on a solver's own OpenFOAM
    runs that solver's programs between meshing and solving, and is solved by
    its command when its program is not itself run
    (cfd_langgraph.solver_shelf.own_runtime)."""
    from cfd_langgraph.openfoam_version import get_profile, solver_command
    from cfd_langgraph.solver_shelf import own_runtime

    runtime = own_runtime(case_dir) if case_dir is not None else {}
    meshed = not mesh_given and (case_dir is None or (Path(case_dir) / "system" / "blockMeshDict").is_file())
    solver = solver_command(get_profile(), (case_solver or "").strip() or "simpleFoam")
    head = [
        "#!/bin/sh",
        'cd "${0%/*}" || exit 1',
        '. "$WM_PROJECT_DIR/bin/tools/RunFunctions"',
        "",
        *(["runApplication blockMesh"] if meshed else []),
        "runApplication checkMesh",
        # A case that sets its initial fields with setFields does so on every
        # new mesh (a refined level's fields are reset to their templates);
        # not a case started from a settled flow, which setFields would undo.
        *(["runApplication setFields"] if case_dir is not None and not mesh_given
          and (Path(case_dir) / "system" / "setFieldsDict").is_file() else []),
        *[f"runApplication {program}" for program in runtime.get("before", [])],
        "cells=$(sed -n 's/.*nCells:[ ]*\\([0-9]*\\).*/\\1/p' constant/polyMesh/owner | head -1)",
        f"np=$(( ${{cells:-0}} / {int(cells_per_proc)} ))",
        f'[ "$np" -gt {int(max_procs)} ] && np={int(max_procs)}',
    ]
    if runtime.get("run"):
        return "\n".join([*head, *_script_solve(runtime, script_ranks(case_dir)), ""])
    return "\n".join([
        *head,
        'if [ "$np" -ge 2 ]; then',
        "    cat > system/decomposeParDict <<EOF",
        "FoamFile { version 2.0; format ascii; class dictionary; object decomposeParDict; }",
        "numberOfSubdomains $np;",
        "method scotch;",
        "EOF",
        "    runApplication decomposePar -force",
        f"    runParallel {solver}",
        "    runApplication reconstructPar",
        "else",
        f"    runApplication {solver}",
        "fi",
        "",
    ])

