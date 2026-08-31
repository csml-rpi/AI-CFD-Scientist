from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from . import prompts as P

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
    system = P.COMMAND_SYSTEM_PROMPT
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
    raw = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)]).content
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


def build_allrun_script(command_list_text: str, case_solver: str = "") -> str:
    """Wrap the model's bare command list in the standard FoamAgent Allrun header.

    ``case_solver`` comes from the case's own controlDict ``application``
    entry, and is appended when the parsed list doesn't already run it. Which
    solver a case runs is a fact recorded in the case, not a judgement call —
    and a case whose Allrun omits it produces no solver log, so every retry
    fails for a reason no rewrite of the *dictionaries* can fix.
    """
    commands = parse_command_list(command_list_text)
    commands = _with_mesh_check(commands)
    solver = (case_solver or "").strip()
    if solver and not any(solver in cmd.split() for cmd in commands):
        commands.append(solver)

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
    lines.append("")
    return "\n".join(lines)
