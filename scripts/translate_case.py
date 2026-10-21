#!/usr/bin/env python3
"""Make one OpenFOAM case run on the installed OpenFOAM, changing nothing it simulates.

  python scripts/translate_case.py --case <case> --out <copy> --output <result.json>
  python scripts/translate_case.py --check --case <case>      (prints the verdict)

The case is first tried on the installation as it is (cfd_langgraph.case_translation
.check_case: dictionaries read, mesh built and checked, a few solver steps). If it
runs, it is copied unchanged and reported as aligned. If not, an agent works on the
copy with the installation's own tutorials for the same solver as the reference,
records every change in TRANSLATION.md, and may not finish until the same check
passes. The source case is never modified.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

_SCRIPTS = Path(__file__).resolve().parent
for _p in (str(_SCRIPTS), str(_SCRIPTS.parent / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cfd_langgraph.case_translation import check_case, copy_inputs  # noqa: E402
from cfd_langgraph.openfoam_version import (  # noqa: E402
    case_solver,
    case_version_evidence,
    describe,
    find_tutorials,
    get_profile,
)

DONE_FIELDS = {
    "summary": "One or two sentences: what had to change for this version, and that the check passes.",
    "changes_file": "Absolute path of TRANSLATION.md in the case folder.",
}

_TOOLS_SPEC = """You have four tools. Call exactly one per turn.

  read_file   {"tool": "read_file", "args": {"path": "<absolute path>", "start": <int, optional>, "max_bytes": <int, optional>}}
      Read a file in the case folder, the original case, or the OpenFOAM installation
      (its tutorials and sources). Long files come in pages; when truncated is true,
      call read_file again with start=next_start.

  write_file  {"tool": "write_file", "args": {"path": "<absolute path>", "content": "<file body>", "mode": "w"|"a"}}
      Write a file inside the case folder; missing folders are created.

  run_bash    {"tool": "run_bash", "args": {"cmd": "<command>", "cwd": "<absolute path inside the case folder>", "timeout": <seconds, optional>}}
      Run a shell command with the installed OpenFOAM loaded. The machine is read-only
      except the case folder and a private /tmp. Returns {rc, stdout, stderr, timeout,
      error_summary}; you see the last 6000 characters of each.

  done        {"tool": "done", "args": {"summary": "...", "changes_file": "..."}}
      Signal completion, only once the check command prints "ok": true.

If you can call these tools as functions, call them directly. Otherwise reply with
exactly one JSON object per turn in the format above: no prose, no markdown fences.
"""

SYSTEM_MESSAGE = (
    "You port OpenFOAM cases between OpenFOAM versions without changing what they simulate. "
    "You have four tools -- read_file, write_file, run_bash and done -- described in the "
    "brief. Call exactly one tool per turn.\n\n"
    "Each change you make is one the installed version requires: a file it reads under another "
    "name, a keyword or syntax it spells differently, a solver it runs another way, a setting it "
    "insists on. Find the installed version's form in its own tutorials or sources before "
    "writing it. Never change what the case simulates: the mesh, boundary conditions and their "
    "values, physical properties, the turbulence model, the schemes, the solver controls and "
    "the run length stay as they are.\n\n"
    "Turns are limited, so make every call do as much as it safely can."
)


def build_prompt(*, case: Path, source: Path, profile: Dict[str, Any], check: Dict[str, Any],
                 check_cmd: str, timeout_s: int) -> str:
    evidence = case_version_evidence(source)
    solver = case_solver(source)
    tutorials = find_tutorials(profile, solver.get("module") or solver.get("application"), limit=4)
    parts = [
        f"# Tools\n{_TOOLS_SPEC}",
        "# Task\n"
        f"Make the OpenFOAM case in {case} run on the installed OpenFOAM without changing what it\n"
        f"simulates. It is a copy of {source} (read-only), which was written for another version.\n"
        f"What its own files say about that version: {json.dumps(evidence)}\n\n"
        f"Tried on the installed version, it failed at the '{check.get('stage')}' stage:\n- "
        + "\n- ".join(check.get("problems") or ["(no detail)"]) + "\n",
        f"# The installed OpenFOAM\n{describe(profile)}\n"
        + ("Its tutorials that run the same solver; follow their conventions:\n- "
           + "\n- ".join(str(t) for t in tutorials) + "\n" if tutorials else ""),
        "# Rules\n"
        "- Change only what this version requires: file names, keywords and their syntax, how the\n"
        "  solver is selected, function-object syntax, settings it insists on. Keep the mesh, every\n"
        "  boundary condition and value, the physical properties, the turbulence model, the\n"
        "  schemes, the solver controls and the run length exactly as they are.\n"
        "- Where this version has no exact equivalent, use the closest one and say so.\n"
        "- Rewrite a function object this version cannot read in its own syntax rather than delete\n"
        "  it. Delete one only if this version has no way to compute it, and say so.\n"
        f"- Record every change in {case / 'TRANSLATION.md'}: the file, what changed, and why, one\n"
        "  line each.\n"
        f"- Check your work with:\n    {check_cmd}\n"
        "  It tries the case on the installed version in a scratch copy (every dictionary read,\n"
        "  mesh built and checked, a few solver steps) and prints a JSON verdict. Call done only\n"
        "  when it prints \"ok\": true.\n",
        f"# Time\nYou have about {timeout_s}s.\n",
    ]
    return "\n".join(parts)


def translate(*, source: Path, dest: Path, output: Path, openfoam_path: str, model: str,
              timeout_s: int, max_turns: int, extra_path: str = "", extra_lib: str = "") -> Dict[str, Any]:
    profile = get_profile(openfoam_path)
    if profile.get("ok"):
        # The agent's shell sources the installation named here.
        os.environ["WM_PROJECT_DIR"] = profile["project_dir"]
    # A third-party solver's programs and libraries, for the checks and for the
    # agent's shell alike.
    if extra_path:
        os.environ["PATH"] = f"{extra_path}:{os.environ.get('PATH', '')}"
    if extra_lib:
        os.environ["LD_LIBRARY_PATH"] = f"{extra_lib}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    target = profile.get("project_dir", openfoam_path)
    paths = {"extra_path": extra_path, "extra_lib": extra_lib}
    result: Dict[str, Any] = {"source": str(source), "case": str(dest), "version": profile.get("label")}
    first = check_case(source, openfoam_path=target, **paths)
    result["initial_check"] = first
    copy_inputs(source, dest)
    if first.get("ok"):
        result.update(status="aligned", check=first)
        _write(output, result)
        return result

    from code_mod_agentic import run_agent_loop  # type: ignore

    check_cmd = (f"{sys.executable} {Path(__file__).resolve()} --check --case {dest} --openfoam {target}"
                 + (f" --extra-path {extra_path}" if extra_path else "")
                 + (f" --extra-lib {extra_lib}" if extra_lib else ""))

    def done_check(_payload: Dict[str, Any]) -> Optional[str]:
        verdict = check_case(dest, openfoam_path=target, **paths)
        if verdict.get("ok"):
            return None
        return (f"The case still fails on the installed version at the '{verdict.get('stage')}' stage: "
                + "; ".join(verdict.get("problems") or []))

    loop = run_agent_loop(
        repo_root=_SCRIPTS.parent,
        hypothesis="Port this OpenFOAM case to the installed version without changing what it simulates.",
        plan="",
        variant_name=dest.name,
        run_dir=dest,
        starter_case=source,
        starter_root=source,
        topic="",
        model=model,
        max_turns=max_turns,
        timeout_s=timeout_s,
        system_message=SYSTEM_MESSAGE,
        initial_prompt=build_prompt(case=dest, source=source, profile=profile, check=first,
                                    check_cmd=check_cmd, timeout_s=timeout_s),
        done_fields=DONE_FIELDS,
        done_check=done_check,
    )
    final = check_case(dest, openfoam_path=target, **paths)
    changes = dest / "TRANSLATION.md"
    result.update(
        status="translated" if final.get("ok") else "failed",
        check=final,
        changes=changes.read_text(errors="replace")[:6000] if changes.is_file() else "",
        turns_used=loop.get("turns_used", 0),
        duration_s=loop.get("duration_s", 0),
        trajectory_log=loop.get("trajectory_log", ""),
        agent_summary=(loop.get("final_payload") or {}).get("summary", ""),
    )
    _write(output, result)
    return result


def _write(path: Path, doc: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, default=str))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--case", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--output", default="")
    ap.add_argument("--check", action="store_true", help="only try the case and print the verdict")
    ap.add_argument("--openfoam", default="", help="WM_PROJECT_DIR of the target installation")
    ap.add_argument("--model", default=os.environ.get("CFD_SCIENTIST_MODEL", ""))
    ap.add_argument("--timeout", type=int, default=2400)
    ap.add_argument("--max-turns", type=int, default=80)
    ap.add_argument("--extra-path", default="", help="folders to put first on PATH (a shelf solver's bin)")
    ap.add_argument("--extra-lib", default="", help="folders to put first on LD_LIBRARY_PATH")
    a = ap.parse_args()
    case = Path(a.case).resolve()
    if a.check:
        verdict = check_case(case, openfoam_path=a.openfoam, extra_path=a.extra_path, extra_lib=a.extra_lib)
        print(json.dumps(verdict, indent=2))
        return 0 if verdict.get("ok") else 1
    if not a.out or not a.output:
        ap.error("--out and --output are required unless --check")
    res = translate(source=case, dest=Path(a.out).resolve(), output=Path(a.output).resolve(),
                    openfoam_path=a.openfoam, model=a.model, timeout_s=a.timeout, max_turns=a.max_turns,
                    extra_path=a.extra_path, extra_lib=a.extra_lib)
    print(json.dumps({k: res.get(k) for k in ("status", "case", "version")}, indent=2))
    return 0 if res.get("status") in ("aligned", "translated") else 1


if __name__ == "__main__":
    raise SystemExit(main())
