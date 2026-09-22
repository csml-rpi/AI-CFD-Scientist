#!/usr/bin/env python3
"""Agentic implementation of a specified method -- the third kind of study.

A solver or surrogate study searches: many candidates, each scored against a
baseline, the archive deciding what to try next. An implementation study does
not. Its task fixes what is to be built -- a numerical scheme, a solver, a model
from a paper -- and the verification that decides whether it was built right:
reference cases, a scorer, acceptance criteria. There is one piece of work, and
the question is whether it passes the task's own checks.

So one agent does the whole job in one workspace: read the brief, write and
build the code, run the task's cases, run its scorer, write its report, and
record in verification.json how the scorer is to be run on what it produced.
The framework re-runs that scorer itself afterwards (tools.impl_verify); what
the agent reports about its own result is not the verdict.

The turn loop, tool protocol, sandbox, transcript handling and time limits are
code_mod_agentic's, imported rather than copied.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from code_mod_agentic import run_agent_loop  # noqa: E402

DONE_FIELDS = {
    "summary": "One or two sentences: what was implemented and what the task's scorer reported.",
    "report_file": "Absolute path of the report you wrote, inside your run directory.",
    "verification_file": "Absolute path of verification.json.",
}

_TOOLS_SPEC = """You have four tools. Call exactly one per turn.

  read_file   {"tool": "read_file", "args": {"path": "<absolute path>", "start": <int, optional>, "max_bytes": <int, optional>}}
      Read a file in your run directory or the task's starter folder. Long files
      come in pages of up to 6000 characters, split at line ends; when truncated
      is true, call read_file again with start=next_start for the next page.

  write_file  {"tool": "write_file", "args": {"path": "<absolute path>", "content": "<file body>", "mode": "w"|"a"}}
      Write a file inside your run directory; missing folders are created.
      Returns {ok, path, bytes_written}.

  run_bash    {"tool": "run_bash", "args": {"cmd": "<command>", "cwd": "<absolute path inside the run directory>", "timeout": <seconds, optional>}}
      Run a shell command. The whole machine is read-only except your run
      directory and a private /tmp. Simulation toolchains installed here (for
      example OpenFOAM) are loaded for every command. Returns {rc, stdout,
      stderr, timeout, error_summary, time_left_s}; you see the last 6000
      characters of stdout and of stderr, so redirect long output to a file in
      the run directory and page through it with read_file.
      A command is stopped when its allowance runs out -- ten minutes by
      default -- and everything it started is stopped with it. For long work,
      time a short slice first, then pass `timeout` big enough for the whole
      command. Never background work with `&` or `nohup`: it is stopped with
      the command that started it.

  done        {"tool": "done", "args": {"summary": "...", "report_file": "...", "verification_file": "..."}}
      Signal completion, only once the task's verification has been run,
      verification.json is written and the report is on disk.

If you can call these tools as functions, call them directly. Otherwise reply
with exactly one JSON object per turn in the format above: no prose, no
markdown fences.
"""

SYSTEM_MESSAGE = (
    "You are a research software engineer who implements a specified method and "
    "verifies it. You have four tools -- read_file, write_file, run_bash and done "
    "-- described in the brief. Call exactly one tool per turn.\n\n"
    "Work in verified steps: build the smallest piece, check it against the "
    "task's own references, and only then run the larger cases. When a check "
    "fails, find the cause before changing anything else.\n\n"
    "Turns are limited and each tool call costs one turn, so make every call do "
    "as much as it safely can: chain related commands in one run_bash, and split "
    "calls only when a later command depends on output you have not seen yet."
)


def _environment_block() -> str:
    try:
        from surrogate_agentic import _environment_block as probe  # type: ignore

        return probe()
    except Exception:
        return (f"# Environment\nInterpreter: {sys.executable}\n"
                "Use this interpreter for every script you run, by its full path.\n")


def build_prompt(*, topic: str, run_dir: Path, starter_root: Path, timeout_s: int,
                 prior_attempt: str = "", guidance: str = "") -> str:
    """The brief. It says nothing about any particular method: the task folder
    carries all of that, and what is stated here is only the workspace and the
    record the framework needs to check the result itself."""
    manifest = run_dir / "verification.json"
    parts = [
        _environment_block(),
        f"# Tools\n{_TOOLS_SPEC}",
        f"# Study\n{topic.strip()}\n",
        "# The task\n"
        f"The task's brief is in its starter folder, {starter_root}. Read it first, then\n"
        "every file it points you to. It fixes what to implement, the cases to run,\n"
        "the verification to perform, the acceptance criteria and the deliverables:\n"
        "do exactly that. Where the brief says to write outputs inside the starter\n"
        "folder, write them in your run directory instead.\n",
        "# Your workspace\n"
        f"- Everything you write goes under {run_dir}. The starter folder is READ-ONLY:\n"
        "  read it freely, and copy into your run directory anything you need to run\n"
        "  or change.\n"
        "- The rest of the machine is read-only too, including the user folders that\n"
        "  build tools write to by default (for OpenFOAM: $WM_PROJECT_USER_DIR,\n"
        "  $FOAM_USER_APPBIN and $FOAM_USER_LIBBIN). Build executables and libraries\n"
        "  into your run directory by pointing the build's output paths there, and put\n"
        "  those folders on PATH and LD_LIBRARY_PATH in the commands that use them.\n"
        "- Use the installed libraries and toolchains. Do not install anything.\n"
        "- Respect every rule in the task brief.\n",
        "# What you must leave behind\n"
        "1. Your code, and how to build it, inside your run directory.\n"
        "2. The cases you ran, with their logs and results, inside your run directory.\n"
        f"3. {manifest}, recording how the task's own scorer is run on your results:\n"
        "     {\"scorer\": \"<absolute path of the task's scorer script in the starter folder,\n"
        "                 or \\\"\\\" if the task has none>\",\n"
        "      \"args\": [\"<the arguments to pass it, with absolute paths to your result files>\"],\n"
        "      \"cwd\": \"<a directory inside your run directory to run it from>\",\n"
        "      \"outputs\": [\"<absolute paths of the files the scorer writes>\"],\n"
        "      \"evidence\": [\"<absolute paths of files showing any acceptance criterion the\n"
        "                   scorer does not check>\"]}\n"
        "   The framework runs that scorer itself, with the interpreter listed above, and\n"
        "   judges its output against the task's acceptance criteria. A result the scorer\n"
        "   does not reproduce from your files does not count.\n"
        "4. The report the task asks for, inside your run directory.\n",
    ]
    if prior_attempt.strip():
        parts.append(
            "# This continues an earlier attempt\n"
            f"{prior_attempt.strip()}\n\n"
            "Your run directory already holds what that attempt produced. Inspect it first\n"
            "and build on it; redo something only if you find it wrong or incomplete.\n"
        )
    if guidance.strip():
        parts.append(f"# Guidance for this attempt\n{guidance.strip()}\n")
    if timeout_s:
        parts.append(
            f"# Time\nYou have about {timeout_s}s. Each tool result shows time_left_s. Cost\n"
            "long runs before starting them, and leave time to run the scorer, write\n"
            "verification.json and write the report: work that is not verified and\n"
            "recorded when time runs out does not count.\n"
        )
    parts.append(
        "# Finish\n"
        "When the verification has been run, verification.json is written and the report\n"
        "is on disk, call done with a short summary, report_file and verification_file.\n"
    )
    return "\n".join(parts)


def _inside(path_text: Any, root: Path) -> Optional[Path]:
    text = str(path_text or "").strip()
    if not text:
        return None
    path = Path(text).expanduser()
    path = path if path.is_absolute() else root / path
    try:
        resolved = path.resolve()
        resolved.relative_to(root.resolve())
    except (OSError, ValueError):
        return None
    return resolved if resolved.is_file() else None


def run(*, run_dir: Path, starter_root: Path, topic: str, output_path: Path, model: str,
        timeout_s: int, max_turns: int, prior_attempt: str = "", guidance: str = "") -> Dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    if not starter_root.is_dir():
        out = {"status": "FAILED", "error": f"starter folder not found: {starter_root}"}
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        return out
    loop = run_agent_loop(
        repo_root=Path(__file__).resolve().parent.parent,
        hypothesis="Implement and verify what the task brief specifies.",
        plan="",
        variant_name="implementation",
        run_dir=run_dir,
        starter_case=starter_root,
        starter_root=starter_root,
        topic=topic,
        model=model,
        max_turns=max_turns,
        timeout_s=timeout_s,
        prior_attempt=prior_attempt,
        repair_goal="",
        system_message=SYSTEM_MESSAGE,
        initial_prompt=build_prompt(topic=topic, run_dir=run_dir, starter_root=starter_root,
                                    timeout_s=timeout_s, prior_attempt=prior_attempt,
                                    guidance=guidance),
        done_fields=DONE_FIELDS,
    )
    final = loop.get("final_payload") or {}
    manifest = run_dir / "verification.json"
    report = _inside(final.get("report_file"), run_dir) if isinstance(final, dict) else None
    result: Dict[str, Any] = {
        # OK means the agent says it finished and left the record the framework
        # checks; whether the work is right is impl_verify's question.
        "status": "OK" if final and manifest.is_file() else "FAILED",
        "finished_cleanly": bool(final) and not loop.get("aborted_reason") and not loop.get("provider_error"),
        "aborted_reason": loop.get("aborted_reason", ""),
        "provider_error": bool(loop.get("provider_error")),
        "duration_s": loop.get("duration_s", 0),
        "turns_used": loop.get("turns_used", 0),
        "verification_file": str(manifest) if manifest.is_file() else "",
        "report_file": str(report) if report else "",
        "trajectory_log": loop.get("trajectory_log", ""),
        "agent_final_payload": final,
    }
    if not final:
        result["error"] = "the agent did not call done"
    elif not manifest.is_file():
        result["error"] = f"the agent called done without writing {manifest}"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="Agentic implementation of a specified method")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--starter-root", required=True)
    ap.add_argument("--topic", default="")
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default=os.environ.get("CFD_SCIENTIST_MODEL", ""))
    ap.add_argument("--timeout", default=0, type=int)
    ap.add_argument("--max-turns", default=400, type=int)
    ap.add_argument("--prior-attempt", default="")
    ap.add_argument("--guidance", default="")
    a = ap.parse_args()
    res = run(
        run_dir=Path(a.run_dir).expanduser().resolve(),
        starter_root=Path(a.starter_root).expanduser().resolve(),
        topic=a.topic, output_path=Path(a.output).expanduser().resolve(), model=a.model,
        timeout_s=a.timeout, max_turns=a.max_turns, prior_attempt=a.prior_attempt,
        guidance=a.guidance,
    )
    print(json.dumps({k: res.get(k) for k in ("status", "error", "turns_used", "finished_cleanly")}, indent=2))
    return 0 if res.get("status") == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
