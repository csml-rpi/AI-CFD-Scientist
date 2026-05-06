#!/usr/bin/env python3
"""
Agentic code-mod runner — provider-agnostic, API-driven.

Runs a tool-using agent loop on top of cfd-scientist's existing langchain
factory (`create_langchain_llm`). Works with ANY provider already wired:
  claude-code, anthropic, openai, openai-codex (OAuth), bedrock, gemini.

The agent gets four tools and a bounded turn budget:

  read_file(path)        — read a file (sandboxed allow-list)
  write_file(path, ...)  — write a file (sandboxed to run_dir + add_dirs)
  run_bash(cmd, cwd)     — execute a shell command (timeout, cwd-restricted)
  done(case_dir, ...)    — signal completion with the artifacts produced

The agent can iterate on its own work: read OpenFOAM source, write a custom
turbulence-model class, run `wmake libso`, read the stderr, edit the source,
retry — all inside one tool-using session, the same shape ARIS uses but
without spawning the codex CLI.

This is generic across modification kinds — turbulence model derivation,
viscosity model, BC class, fvOption class, anything that ends in a
case-local `wmake libso`.

CLI:
  python scripts/code_mod_agentic.py \
      --hypothesis <text>            \
      --variant-name <slug>          \
      --run-dir <dir>                \
      --starter-case <readonly base case> \
      --topic <free text>            \
      --output <result.json>         \
      [--model gpt-5.4]              \
      [--timeout 1800]               \
      [--max-turns 80]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# tool implementations (sandboxed)
# ---------------------------------------------------------------------------


_ERR_PATTERNS = (
    "error:", "Error:", "ERROR:",
    "FOAM FATAL", "FATAL ERROR",
    "fatal error",
    "undefined reference",
    "cannot find", "No such file",
    "Failed wmake", "make: ***",
    "Segmentation fault",
    "Traceback (most recent call last)",
)


def _grep_error_lines(stdout: str, stderr: str, *, max_lines: int = 40) -> str:
    """Return a compact summary of error/failure lines from a build/run output.

    Looks across stdout+stderr for lines matching common error markers and
    returns them with a few lines of context above each match. Empty if no
    matches. This is shown to the agent at the top of the result so a
    truncated stdout never hides the real diagnostic.
    """
    combined: List[Tuple[str, str]] = []
    for label, blob in (("stdout", stdout or ""), ("stderr", stderr or "")):
        for ln in blob.splitlines():
            combined.append((label, ln))
    if not combined:
        return ""
    hits: List[int] = []
    for i, (_lbl, ln) in enumerate(combined):
        if any(p in ln for p in _ERR_PATTERNS):
            hits.append(i)
    if not hits:
        return ""
    # Collect each hit + 2 lines of context above; dedupe & cap.
    keep: set = set()
    for i in hits:
        for j in range(max(0, i - 2), i + 1):
            keep.add(j)
    sel = sorted(keep)[:max_lines]
    out_lines = [f"[{combined[i][0]}] {combined[i][1]}" for i in sel]
    if len(sorted(keep)) > max_lines:
        out_lines.append(f"… (+{len(sorted(keep)) - max_lines} more error/context lines elided)")
    return "\n".join(out_lines)


class Sandbox:
    """Bounded read / write / shell access tied to the run directory."""

    def __init__(
        self,
        *,
        run_dir: Path,
        starter_case: Path,
        wm_project_dir: Optional[Path],
        max_read_bytes: int = 200_000,
        max_write_bytes: int = 200_000,
        bash_timeout: int = 600,
    ) -> None:
        self.run_dir = Path(run_dir).resolve()
        self.starter_case = Path(starter_case).resolve()
        self.wm = Path(wm_project_dir).resolve() if wm_project_dir else None
        self.max_read_bytes = max_read_bytes
        self.max_write_bytes = max_write_bytes
        self.bash_timeout = bash_timeout
        # Per-path write counter for the perfectionism-stall circuit-breaker.
        # Generic across topics — any file written more than
        # MAX_WRITES_PER_PATH times in a single session triggers a reject
        # with progress-nudge guidance, forcing the agent to advance.
        self._writes_per_path: Dict[str, int] = {}
        self.MAX_WRITES_PER_PATH = 3

    # ---- path checks
    def _is_under(self, p: Path, root: Path) -> bool:
        try:
            p.resolve().relative_to(root)
            return True
        except Exception:
            return False

    def _read_allowed(self, p: Path) -> Tuple[bool, str]:
        rp = p.resolve()
        if self._is_under(rp, self.run_dir):
            return True, "run_dir"
        if self._is_under(rp, self.starter_case):
            return True, "starter_case"
        if self.wm and self._is_under(rp, self.wm):
            return True, "wm_project"
        # Allow common OpenFOAM-related read locations
        for system_root in ("/usr/include", "/etc"):
            if self._is_under(rp, Path(system_root)):
                return True, system_root
        return False, ""

    def _write_allowed(self, p: Path) -> Tuple[bool, str]:
        rp = p.resolve()
        # Block edits inside starter_case (read-only) and $WM_PROJECT_DIR.
        if self._is_under(rp, self.starter_case):
            return False, "starter_case is read-only"
        if self.wm and self._is_under(rp, self.wm):
            return False, "$WM_PROJECT_DIR is read-only"
        if self._is_under(rp, self.run_dir):
            return True, "run_dir"
        return False, "outside run_dir"

    # ---- tool: read_file
    def read_file(self, path: str, max_bytes: Optional[int] = None) -> Dict[str, Any]:
        p = Path(path).expanduser()
        ok, why = self._read_allowed(p)
        if not ok:
            return {"ok": False, "error": f"read denied: {p} (must be inside run_dir / starter_case / WM_PROJECT_DIR)"}
        if not p.is_file():
            return {"ok": False, "error": f"not a file: {p}"}
        cap = min(max_bytes or self.max_read_bytes, self.max_read_bytes)
        data = p.read_bytes()[:cap]
        try:
            content = data.decode("utf-8", errors="replace")
        except Exception as exc:
            return {"ok": False, "error": f"decode err: {exc}"}
        truncated = len(data) >= cap
        return {"ok": True, "path": str(p.resolve()), "content": content,
                "size": p.stat().st_size, "truncated": truncated}

    # ---- tool: write_file
    def write_file(self, path: str, content: str, *, mode: str = "w") -> Dict[str, Any]:
        p = Path(path).expanduser()
        ok, why = self._write_allowed(p)
        if not ok:
            return {"ok": False, "error": f"write denied: {p} ({why})"}
        if mode not in ("w", "a"):
            return {"ok": False, "error": f"unsupported mode: {mode}"}

        # Perfectionism-stall circuit-breaker. If this exact path has been
        # written `MAX_WRITES_PER_PATH` times already AND the new content
        # is byte-identical to what's on disk, refuse the write and tell
        # the agent to advance. Generic across any code-mod task; the
        # advance guidance is family-agnostic.
        rp = str(p.resolve())
        prior = self._writes_per_path.get(rp, 0)
        existing_content = ""
        if p.is_file():
            try:
                existing_content = p.read_text(encoding="utf-8", errors="replace")
            except Exception:
                existing_content = ""
        if mode == "w" and prior >= self.MAX_WRITES_PER_PATH:
            if existing_content == content:
                return {
                    "ok": False,
                    "error": (
                        f"refused: {rp} has already been written {prior} times in this "
                        f"session and the new content is identical to what is on disk. "
                        f"This file is FINAL — do not write to it again. "
                        f"ADVANCE: write the next required file (the implementation "
                        f"file if you've only written the header; the build descriptor "
                        f"files Make/files and Make/options if you've written .H and "
                        f".C; or invoke the build/run step if all source files are in "
                        f"place). Track which step of the workflow you are on."
                    ),
                }
            else:
                # Allow the write but warn the agent it's iterating too much.
                pass
        if len(content.encode("utf-8")) > self.max_write_bytes * 4:
            return {"ok": False, "error": f"content too large (>{self.max_write_bytes*4} bytes)"}
        if mode == "w":
            p.write_text(content, encoding="utf-8")
        else:
            with p.open("a", encoding="utf-8") as fh:
                fh.write(content)
        self._writes_per_path[rp] = prior + 1
        warning = ""
        if mode == "w" and self._writes_per_path[rp] >= self.MAX_WRITES_PER_PATH:
            warning = (
                f"WARNING: you have now written {self._writes_per_path[rp]} times to "
                f"this same path. Further byte-identical writes will be REFUSED. "
                f"Move on to the next workflow step instead of re-editing this file."
            )
        out = {"ok": True, "path": rp, "bytes_written": len(content),
               "writes_to_this_path_so_far": self._writes_per_path[rp]}
        if warning:
            out["warning"] = warning
        return out

    # ---- tool: run_bash
    def run_bash(self, cmd: str, cwd: Optional[str] = None, timeout: Optional[int] = None) -> Dict[str, Any]:
        # cwd must be inside run_dir
        cwd_p = Path(cwd).expanduser().resolve() if cwd else self.run_dir
        if not self._is_under(cwd_p, self.run_dir):
            return {"ok": False, "error": f"cwd must be inside run_dir: {cwd_p}"}
        if not cwd_p.is_dir():
            return {"ok": False, "error": f"cwd does not exist: {cwd_p}"}
        if not isinstance(cmd, str) or not cmd.strip():
            return {"ok": False, "error": "empty command"}
        # Ensure OpenFOAM env is sourced (so wmake works) — find a bashrc.
        bashrc_candidates = []
        if self.wm:
            bashrc_candidates.append(str(self.wm / "etc" / "bashrc"))
        bashrc_candidates += [
            "/mnt/sda1/openfoam10/etc/bashrc",
            "/opt/openfoam10/etc/bashrc",
        ]
        bashrc = next((b for b in bashrc_candidates if Path(b).is_file()), None)
        prefix = f". \"{bashrc}\" >/dev/null 2>&1 && " if bashrc else ""
        full = f"{prefix}{cmd}"
        t = min(timeout or self.bash_timeout, self.bash_timeout)
        try:
            proc = subprocess.run(
                ["bash", "-lc", full],
                cwd=str(cwd_p),
                text=True,
                capture_output=True,
                timeout=t,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            tout = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
            terr = (exc.stderr or "") if isinstance(exc.stderr, str) else ""
            return {
                "ok": True,
                "rc": -1,
                "error_summary": _grep_error_lines(tout, terr),
                "stdout": tout[-16000:],
                "stderr": terr[-16000:],
                "timeout": True,
                "cwd": str(cwd_p),
            }
        out = proc.stdout or ""
        err = proc.stderr or ""
        return {
            "ok": True,
            "rc": int(proc.returncode),
            "error_summary": _grep_error_lines(out, err),
            "stdout": out[-16000:],
            "stderr": err[-16000:],
            "timeout": False,
            "cwd": str(cwd_p),
        }


# ---------------------------------------------------------------------------
# prompt + tool-call protocol
# ---------------------------------------------------------------------------

_TOOLS_SPEC = """You have four tools. Each turn, output ONE JSON object naming a tool to call.
Do not output anything outside the JSON object. Do not use markdown fences.

TOOLS:

  {"tool": "read_file", "args": {"path": "<absolute path>", "max_bytes": <int, optional>}}
      Read a file. Allowed paths: anywhere inside the run directory, the
      starter case (read-only), and $WM_PROJECT_DIR/src or .../tutorials.
      Returns {ok, content, size, truncated}.

  {"tool": "write_file", "args": {"path": "<absolute path>", "content": "<file body>", "mode": "w"|"a"}}
      Write a file. Path must be inside the run directory.
      Returns {ok, path, bytes_written}.

  {"tool": "run_bash", "args": {"cmd": "<command>", "cwd": "<absolute path inside run_dir>", "timeout": <seconds, optional>}}
      Run a shell command. cwd must be inside the run directory. The OpenFOAM
      bashrc is auto-sourced before each command, so wmake / blockMesh /
      simpleFoam etc. are on PATH. Returns {rc, stdout, stderr, timeout,
      error_summary}. ALWAYS check `error_summary` first — it pre-extracts
      gcc/wmake/FOAM/runtime error lines (with 2-line context) so you don't
      need to scan a long stdout to find the failure. `stdout` and `stderr`
      contain the LAST 16K chars (truncation is tail-biased — recent output
      preserved). Plan your next edit from `error_summary`.

  {"tool": "done", "args": {"case_dir": "<abs path>", "class_name": "<name>", "compiled_so": "<abs .so path>", "summary": "<one-line summary>"}}
      Signal completion. Call this ONLY after wmake produced a .so AND
      simpleFoam reached End. The framework will validate.

PROTOCOL RULES:
  - Output EXACTLY ONE JSON object per turn. No prose. No code fences. No
    second JSON object — anything after the first object is silently dropped
    by the framework, so a second tool call in the same turn is wasted.
  - Do NOT read a file you just wrote. write_file already returns a confirmed
    bytes_written count; re-reading it consumes a turn for zero information.
  - Read parent / reference source files BEFORE writing — but only once per
    file. Subsequent edits should be guided by actual build/run stderr, not
    by re-reading your own writes.
  - When the build step fails, read `error_summary` first (pre-extracted
    error lines with context), then scan the tail of `stdout`/`stderr` (last
    16K chars are preserved). Edit the offending file based on the actual
    diagnostic — do not guess. If you need more than 16K of output, re-run
    the build piping through `2>&1 | tail -300 > /tmp/build.log` and
    read_file that log. Do not give up after one failure — iterate until
    it builds.
  - Activate the new component by editing whichever case dictionary is
    appropriate for THIS modification family (turbulence model → the
    momentum/turbulence transport dictionary; viscosity model →
    transportProperties; volumetric source → fvModels; field BC →
    0/<field>; numerical scheme → fvSchemes; thermo / radiation → the
    relevant constant/ dict; etc.) and add a libs (...) entry to
    system/controlDict if a compiled .so must be loaded.
  - Run the application named in system/controlDict (whatever it is —
    simpleFoam, pimpleFoam, foamRun, chtMultiRegionFoam, interFoam, etc.).
    Verify the log ends in `End` cleanly before calling done.

  ── PERFECTIONISM-STALL GUARD (HARD RULE) ─────────────────────────────────
  - You may write a given file at most 3 times in this session. Further
    byte-identical writes to the same path will be REFUSED by the framework
    with an error telling you to advance. Do NOT iterate on the "perfect"
    header before writing the implementation. Get the file to a *good
    enough* state, MOVE ON to the next required artifact, run the build,
    and let the build/run errors drive subsequent edits.

  ── GENERIC WORKFLOW CHECKLIST (track which step you are on) ─────────────
  When the modification is a CLASS DERIVATION (compiled custom library):
    [1] Write the class header  (declarations: members, virtual overrides)
    [2] Write the class implementation (definitions of overridden methods,
        and the runtime-selection-table macro registration if needed)
    [3] Write Make/files     (source list + LIB output basename)
    [4] Write Make/options   (use the verified $LIB_SRC paths and -l<name>
        from the OPENFOAM ENVIRONMENT FACTS block above; do NOT invent)
    [5] Run the build (`wmake libso` or equivalent). Iterate on errors —
        for each error message, read the file it references and fix the
        SPECIFIC issue. Do not rewrite the whole file from scratch.
    [6] Activate in the appropriate case dictionary for this modification
        family (see above) and add libs (...) to system/controlDict if a
        .so must be loaded.
    [7] Run the case to convergence using the application from
        system/controlDict (NOT a hardcoded one). Verify the log ends in
        `End` cleanly.
    [8] Call `done`.

  When the modification is a RUNTIME (coded* / dictionary edit only):
    [1] Edit the relevant case dictionary in-place (e.g. add a coded entry
        to constant/fvModels, or replace a BC in 0/<field>, or apply a
        dict patch).
    [2] Run the case (application from system/controlDict). Coded blocks
        are JIT-compiled by OpenFOAM at solver startup; no wmake needed.
    [3] If the JIT compile fails, fix the C++ in the SAME dictionary file
        and re-run.
    [4] Verify the log ends in `End`. Call `done`.

  At every turn, briefly self-state which step you are on (mental note —
  the JSON-only protocol means no prose; track it in your reasoning).
"""


_SAFETY = """SAFETY RULES (every tool call enforces these; violations get a denial):
1. $WM_PROJECT_DIR is READ-ONLY. You may read source and tutorials there but
   never write. Do not run wmake inside $WM_PROJECT_DIR.
2. The starter folder is READ-ONLY. Read it; copy files INTO the run dir
   before editing.
3. All writes go into the run directory. Custom OpenFOAM libraries must be
   built case-local at <run_dir>/<case_name>/customModels/<ClassName>/ via
   `wmake libso` (cwd inside that dir).
4. Activation by case dictionaries only: system/controlDict.libs and the
   relevant constant/ dictionary (constant/momentumTransport, etc.).
"""


def build_agent_prompt(*, topic: str, hypothesis: str, variant_name: str,
                       starter_case: Path, run_dir: Path,
                       wm_project_dir: Optional[Path]) -> str:
    """Generic deliverable template across CFD modification kinds: turbulence
    closure, viscosity / non-Newtonian model, thermophysical property model,
    custom boundary condition, custom fvOption / fvModel source term, custom
    discretization scheme, custom solver derivative, mass-transfer kernel,
    radiation closure, etc. The agent infers the right parent class and the
    right activation dictionary from the hypothesis + the OpenFOAM source it
    reads."""
    # Inject ground-truth facts about THIS OpenFOAM install (real src/ tree,
    # real Make/options, real .so basenames) so the LLM cannot hallucinate
    # include paths or -l<name> flags that don't exist on disk. Generic across
    # every modification family — purely a description of the install.
    try:
        from openfoam_grounding import build_grounding_block  # type: ignore
        grounding = build_grounding_block(wm_project_dir)
    except Exception:
        grounding = ""

    return (
        _SAFETY
        + "\n" + _TOOLS_SPEC
        + ("\n" + grounding if grounding else "")
        + "\n============================================================\n"
        + "TASK\n"
        + "============================================================\n\n"
        + f"Topic: {topic.strip()}\n\n"
        + f"Hypothesis to implement (variant_name={variant_name}):\n{hypothesis.strip()}\n\n"
        + f"WM_PROJECT_DIR (read-only OpenFOAM install): {wm_project_dir or '(unset)'}\n"
        + f"Starter case (read-only base): {starter_case}\n"
        + f"Run directory (your writable workspace): {run_dir}\n\n"
        + "DELIVERABLE (generic — applies to ANY OpenFOAM modification family):\n"
        + "  (a) Copy the starter case into the run dir under a case folder\n"
        + f"      named {variant_name}/ (use run_bash with `cp -a`).\n"
        + "  (b) Identify the appropriate OpenFOAM parent class for this\n"
        + "      hypothesis by reading $WM_PROJECT_DIR/src and matching\n"
        + "      against the hypothesis description. Examples by category:\n"
        + "        - turbulence/closure model     → derive from existing RAS/LES base class\n"
        + "        - viscosity / non-Newtonian    → derive from viscosityModel\n"
        + "        - thermophysical / transport   → derive from the relevant thermo base\n"
        + "        - boundary condition           → derive from the relevant fvPatchField\n"
        + "        - source term / forcing        → derive from fvModel / fvOption / coded base\n"
        + "        - discretization scheme        → derive from the relevant scheme base\n"
        + "      In ALL cases the C++ goes inside\n"
        + f"      <run_dir>/{variant_name}/customModels/<ClassName>/\n"
        + "      with {ClassName}.H, {ClassName}.C, Make/files, Make/options.\n"
        + "  (c) Compile with `wmake libso` from inside that customModels/<X>/\n"
        + "      directory. Iterate on errors as needed.\n"
        + "  (d) Activate the compiled library by editing the appropriate case\n"
        + "      dictionary. The right dictionary depends on the modification\n"
        + "      kind (constant/momentumTransport for turbulence, constant/\n"
        + "      transportProperties for viscosity, constant/fvModels for sources,\n"
        + "      0/<field> boundaryField for BCs, system/fvSchemes for schemes,\n"
        + "      etc.) and add a `libs (\"libYourName.so\");` entry to system/controlDict.\n"
        + "  (e) Run the application named in system/controlDict (whatever it\n"
        + "      is — simpleFoam, pimpleFoam, foamRun, chtMultiRegionFoam, etc.)\n"
        + "      to convergence. Verify log.<application> reaches `End` cleanly.\n"
        + "      DO NOT use the `-noFunctionObjects` flag. If system/controlDict\n"
        + "      defines functions { ... } (e.g. wallShearStress, yPlus, residuals,\n"
        + "      probes, sampling, force coefficients), those function objects\n"
        + "      compute the post-processed quantities the framework's comparator\n"
        + "      script needs to score the run. Skipping them produces a case\n"
        + "      that runs to End but has no usable comparison fields, and the\n"
        + "      comparator falls back to the wrong time directory yielding a\n"
        + "      bogus RMSE. Run the solver with default flags. If you must\n"
        + "      skip them for performance during debugging, run them as a\n"
        + "      post-processing pass afterwards via\n"
        + "      `<application> -postProcess -func wallShearStress -latestTime`\n"
        + "      (or the relevant function names from controlDict).\n"
        + "  (f) Call `done` with the case_dir, class_name, and .so path.\n\n"
        + "Begin. Output ONE tool call per turn as a single JSON object.\n"
    )


# ---------------------------------------------------------------------------
# JSON parsing tolerant of markdown fences / extra whitespace
# ---------------------------------------------------------------------------

def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    s = text.strip()
    # Strip code fences if any
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```\s*$", "", s)
    # Find the first {...} object
    start = s.find("{")
    if start < 0:
        return None
    depth = 0
    end = -1
    for i in range(start, len(s)):
        c = s[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end < 0:
        return None
    chunk = s[start:end + 1]
    try:
        return json.loads(chunk)
    except Exception:
        # try to repair common issues
        try:
            return json.loads(chunk.replace("\n", "\\n"))
        except Exception:
            return None


# ---------------------------------------------------------------------------
# agent loop using cfd-scientist's existing langchain factory
# ---------------------------------------------------------------------------

def _bootstrap_paths(repo_root: Path) -> None:
    for p in (repo_root / "src", repo_root / "Foam-Agent" / "src"):
        sp = str(p)
        if p.is_dir() and sp not in sys.path:
            sys.path.insert(0, sp)


def run_agent_loop(
    *,
    repo_root: Path,
    hypothesis: str,
    variant_name: str,
    run_dir: Path,
    starter_case: Path,
    topic: str,
    model: str,
    max_turns: int,
    timeout_s: int,
) -> Dict[str, Any]:
    _bootstrap_paths(repo_root)
    try:
        from langchain_core.messages import HumanMessage, SystemMessage  # type: ignore
        from cfd_langgraph.llm.factory import create_langchain_llm  # type: ignore
    except Exception as exc:
        return {"status": "FAILED", "error": f"langchain factory import failed: {exc}"}

    wm = os.environ.get("WM_PROJECT_DIR", "").strip()
    wm_path = Path(wm) if wm else None
    sandbox = Sandbox(
        run_dir=run_dir,
        starter_case=starter_case,
        wm_project_dir=wm_path,
    )
    trajectory_log = run_dir / "agentic_trajectory.log"
    trajectory_log.parent.mkdir(parents=True, exist_ok=True)
    log_fh = open(trajectory_log, "w", encoding="utf-8")

    def log(msg: str) -> None:
        log_fh.write(msg + "\n")
        log_fh.flush()

    log(f"# agentic loop start variant={variant_name} model={model}")
    log(f"# run_dir={run_dir}")
    log(f"# starter_case={starter_case}")
    log(f"# WM_PROJECT_DIR={wm_path}")

    sys_msg_text = (
        "You are an agentic OpenFOAM developer. You have read/write/bash tools "
        "(see protocol). Each turn, output ONE JSON object calling one tool. "
        "No prose, no code fences. Iterate on compile errors patiently."
    )
    initial_user_prompt = build_agent_prompt(
        topic=topic,
        hypothesis=hypothesis,
        variant_name=variant_name,
        starter_case=starter_case,
        run_dir=run_dir,
        wm_project_dir=wm_path,
    )

    # IMPORTANT: do NOT accumulate AIMessage / HumanMessage pairs across turns.
    # Different providers encode assistant turns differently (Codex Responses
    # API requires output_text content; the langchain wrapper encodes everything
    # as input_text → HTTP 400 on turn 2). Instead, maintain a single rolling
    # transcript inside ONE HumanMessage. Each turn, we send:
    #   [SystemMessage(sys_msg_text), HumanMessage(initial_prompt + transcript)]
    # The transcript records every prior tool call + result the agent made.
    # Provider-agnostic — the wrapper only ever sees user (input) content.
    transcript_chunks: List[str] = []
    TRANSCRIPT_CAP = 60_000  # chars; truncate oldest entries above this

    def render_transcript() -> str:
        if not transcript_chunks:
            return ""
        full = "\n\n=== CONVERSATION SO FAR ===\n" + "\n".join(transcript_chunks) + "\n=== END CONVERSATION ===\n\nNow output your next tool-call JSON object."
        if len(full) <= TRANSCRIPT_CAP:
            return full
        # Truncate from the front (keep most recent turns); always keep the
        # marker lines so the agent knows it's seeing a truncated view.
        keep = full[-TRANSCRIPT_CAP:]
        return ("\n\n=== CONVERSATION SO FAR (truncated; older turns omitted) ===\n"
                + keep)

    llm = create_langchain_llm(model=model, temperature=0.1)
    started = time.time()
    final_payload: Dict[str, Any] = {}
    aborted_reason = ""
    turn = 0
    parse_fail_streak = 0
    MAX_PARSE_FAILS = 3

    # Transient-error patterns from the Codex / generic streaming endpoints.
    # These are recoverable — usually a stream drop, brief 5xx, or rate-limit
    # backoff — and a fresh request normally works.
    _TRANSIENT_PATTERNS = (
        "response ended prematurely",
        "connection aborted",
        "connection reset",
        "remote disconnected",
        "incomplete read",
        "read timed out",
        "timed out",
        "503",
        "502",
        "504",
        "429",
        "rate limit",
        "throttl",
        "service unavailable",
        "temporarily unavailable",
    )
    MAX_TRANSIENT_RETRIES = 3

    # When timeout_s <= 0, the wall-clock cap is disabled and only the
    # `max_turns` budget bounds the agentic loop. This matters for slower
    # LLMs (e.g. claude-sonnet-4-6) that haven't finished class_derivation
    # within 30 min but are progressing turn-by-turn within budget.
    wall_clock_enabled = timeout_s > 0

    for turn in range(1, max_turns + 1):
        if wall_clock_enabled and time.time() - started > timeout_s:
            aborted_reason = f"timeout after {timeout_s}s"
            log(f"\n# {aborted_reason}")
            break
        log(f"\n--- turn {turn} ---")
        user_msg_text = initial_user_prompt + render_transcript()
        messages = [
            SystemMessage(content=sys_msg_text),
            HumanMessage(content=user_msg_text),
        ]
        ai_resp = None
        last_exc: Optional[Exception] = None
        for attempt in range(MAX_TRANSIENT_RETRIES + 1):
            if wall_clock_enabled and time.time() - started > timeout_s:
                break
            try:
                ai_resp = llm.invoke(messages)
                last_exc = None
                break
            except Exception as exc:
                last_exc = exc
                msg = str(exc).lower()
                is_transient = any(p in msg for p in _TRANSIENT_PATTERNS)
                if not is_transient or attempt >= MAX_TRANSIENT_RETRIES:
                    break
                backoff = 2.0 * (2 ** attempt)  # 2, 4, 8 s
                log(f"# transient llm error (attempt {attempt + 1}/{MAX_TRANSIENT_RETRIES + 1}): "
                    f"{type(exc).__name__}: {str(exc)[:200]}; sleeping {backoff:.1f}s and retrying.")
                time.sleep(backoff)
        if ai_resp is None:
            aborted_reason = (f"llm.invoke raised after {MAX_TRANSIENT_RETRIES + 1} attempts: "
                              f"{type(last_exc).__name__ if last_exc else 'Unknown'}: "
                              f"{str(last_exc)[:300] if last_exc else ''}")
            log(f"# {aborted_reason}")
            break
        ai_text = getattr(ai_resp, "content", None) or str(ai_resp)
        if isinstance(ai_text, list):
            ai_text = " ".join(getattr(p, "text", str(p)) for p in ai_text)
        ai_text = str(ai_text)
        log(f"AI: {ai_text[:1200]}{'…' if len(ai_text) > 1200 else ''}")

        tool_call = _extract_json_object(ai_text)
        if not tool_call:
            parse_fail_streak += 1
            if parse_fail_streak >= MAX_PARSE_FAILS:
                aborted_reason = f"parse-fail streak {parse_fail_streak} reached cap"
                log(f"# {aborted_reason}")
                break
            transcript_chunks.append(
                f"[turn {turn}] (your reply was not parseable as a tool-call JSON object; "
                f"reply contained: {ai_text[:300]!r}). NEXT TURN: output a single "
                "JSON object {\"tool\":..., \"args\":{...}} only — no prose."
            )
            continue
        parse_fail_streak = 0
        tool_name = str(tool_call.get("tool", "")).strip()
        tool_args = tool_call.get("args") or {}
        if tool_name == "done":
            final_payload = tool_args if isinstance(tool_args, dict) else {}
            log(f"# DONE: {json.dumps(final_payload)[:600]}")
            break

        # execute tool
        try:
            if tool_name == "read_file":
                tool_result = sandbox.read_file(**tool_args)
            elif tool_name == "write_file":
                tool_result = sandbox.write_file(**tool_args)
            elif tool_name == "run_bash":
                tool_result = sandbox.run_bash(**tool_args)
            else:
                tool_result = {"ok": False, "error": f"unknown tool: {tool_name!r}"}
        except TypeError as exc:
            tool_result = {"ok": False, "error": f"bad args for {tool_name}: {exc}"}
        except Exception as exc:
            tool_result = {"ok": False, "error": f"tool {tool_name} raised: {exc}"}

        # Truncate large fields before logging / before they enter the transcript.
        # For build/run output, prefer the TAIL (compile errors land at the end)
        # and never truncate `error_summary` — that's the focused diagnostic.
        compact = dict(tool_result)
        for k in ("content",):
            v = compact.get(k)
            if isinstance(v, str) and len(v) > 6000:
                compact[k] = v[:3000] + "\n…[truncated]…\n" + v[-3000:]
        for k in ("stdout", "stderr"):
            v = compact.get(k)
            if isinstance(v, str) and len(v) > 6000:
                # Keep the last 6000 chars — error/failure lines are at the tail.
                compact[k] = "…[earlier output elided]…\n" + v[-6000:]
        log(f"TOOL [{tool_name}] -> {json.dumps({k:(v if not isinstance(v,str) or len(v)<200 else v[:200]+'…') for k,v in compact.items()}, default=str)[:1200]}")

        # Append a single chunk for this turn into the rolling transcript.
        # Bumped from 8000 → 18000 so the agent actually sees the gcc errors
        # (a typical wmake stdout JSON-serializes to ~10–12K chars).
        transcript_chunks.append(
            f"[turn {turn}] you called {tool_name}({json.dumps(tool_args, default=str)[:600]}).\n"
            f"[turn {turn}] tool result: {json.dumps(compact, default=str)[:18000]}"
        )

    log_fh.flush()
    log_fh.close()
    duration = int(time.time() - started)
    return {
        "status": "OK" if final_payload else "FAILED",
        "aborted_reason": aborted_reason,
        "duration_s": duration,
        "turns_used": turn,
        "trajectory_log": str(trajectory_log),
        "final_payload": final_payload,
    }


# ---------------------------------------------------------------------------
# post-hoc artifact validation
# ---------------------------------------------------------------------------

def _find_compiled_artifacts(  # noqa: C901

    run_dir: Path,
    variant_name: str = "",
    agent_case_dir: str = "",
    agent_compiled_so: str = "",
    started_at: float = 0.0,
) -> Dict[str, Any]:
    """
    Locate the .so the agent just compiled, the case dir, and check
    convergence. wmake places the .so wherever Make/files's `LIB =` directive
    points — typically one of:
      (a) case-local: <case>/customModels/<X>/platforms/<arch>/lib<X>.so
      (b) user libbin: $FOAM_USER_LIBBIN/lib<X>.so  (wmake's default when
          the agent uses LIB = $(FOAM_USER_LIBBIN)/lib<X>)
    We scan both and filter by mtime to avoid picking up stale .so files left
    over from previous runs of the same study.

    Generic across modification kinds — turbulence, viscosity, BC class, source,
    or any other custom OpenFOAM derivative; only the file pattern matters.
    """
    candidates: List[Path] = []
    # (0) Trust the agent's claimed .so path when it actually exists, lives
    # inside the run directory, and was modified during this run. This avoids
    # false negatives when the agent picks an unconventional but valid LIB
    # output location (e.g. `../lib/libX.so` instead of `platforms/<arch>/`).
    if agent_compiled_so:
        ap = Path(agent_compiled_so)
        try:
            ap.resolve().relative_to(run_dir.resolve())
            if ap.is_file() and ap.name.startswith("lib") and ap.suffix == ".so":
                candidates.append(ap)
        except Exception:
            pass
    # (a) case-local — scan ANY .so under customModels/, regardless of whether
    # the agent used the conventional `platforms/<arch>/` layout or pointed
    # `LIB =` at a sibling directory like `../lib/lib<X>`. Both are valid
    # outputs of `wmake libso`.
    candidates.extend(run_dir.rglob("customModels/**/lib*.so"))
    # (b) FOAM_USER_LIBBIN
    user_libbin = os.environ.get("FOAM_USER_LIBBIN", "").strip()
    if not user_libbin:
        # heuristic: $HOME/OpenFOAM/<user>-<ver>/platforms/.../lib
        home = Path(os.environ.get("HOME", str(Path.home())))
        for ulb in home.glob("OpenFOAM/*-*/platforms/*/lib"):
            user_libbin = str(ulb)
            break
    if user_libbin and Path(user_libbin).is_dir():
        candidates.extend(Path(user_libbin).glob("lib*.so"))

    if started_at > 0:
        # Filter to .so files modified DURING this run only — avoid stale.
        fresh = [p for p in candidates if p.is_file() and p.stat().st_mtime >= started_at - 5]
        candidates = fresh

    # Variant-name preference: a .so whose basename mentions the variant_name
    # is much more likely to be ours than a random match.
    so: Optional[Path] = None
    if variant_name:
        slug = re.sub(r"[^A-Za-z0-9]+", "", variant_name.lower())
        scored = []
        for p in candidates:
            base = re.sub(r"[^A-Za-z0-9]+", "", p.stem.lower())
            score = 0
            if slug and slug in base:
                score = 2
            elif slug and any(slug[i:i+4] in base for i in range(0, max(1, len(slug) - 3))):
                score = 1
            scored.append((score, p.stat().st_mtime, p))
        scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
        if scored and scored[0][0] > 0:
            so = scored[0][2]
    if so is None and candidates:
        # Fall back to most-recent .so overall.
        so = max(candidates, key=lambda p: p.stat().st_mtime)

    class_name = ""
    if so:
        m = re.match(r"lib(.+)\.so$", so.name)
        if m:
            class_name = m.group(1)

    # Derive case_dir. Prefer case-local (.so under <case>/customModels/...).
    # Otherwise, trust agent_case_dir if it points at a valid OpenFOAM case
    # under run_dir.
    case_dir: Optional[Path] = None
    if so:
        cm = next((p for p in so.parents if p.name == "customModels"), None)
        if cm and cm.parent.is_dir():
            case_dir = cm.parent
    if case_dir is None and agent_case_dir:
        cand = Path(agent_case_dir)
        try:
            if cand.is_dir() and (cand / "constant").is_dir() and (cand / "system").is_dir():
                # only accept if inside the run_dir (sandbox)
                cand.resolve().relative_to(run_dir.resolve())
                case_dir = cand
        except Exception:
            pass
    if case_dir is None:
        # Last resort: scan run_dir for the first OpenFOAM case dir.
        for d in run_dir.rglob("*"):
            if not d.is_dir():
                continue
            if (d / "constant").is_dir() and (d / "system").is_dir() and (d / "system" / "controlDict").is_file():
                case_dir = d
                break

    converged = False
    if case_dir is not None:
        # The application name comes from controlDict — could be simpleFoam,
        # pimpleFoam, foamRun, chtMultiRegionFoam, etc. Look for any log.<app>.
        log_files = list(case_dir.glob("log.*"))
        # Prefer log.simpleFoam if present (most common), else any solver log.
        candidate_logs = [p for p in log_files if p.name in (
            "log.simpleFoam", "log.pimpleFoam", "log.foamRun",
            "log.icoFoam", "log.rhoSimpleFoam", "log.buoyantSimpleFoam",
        )]
        if not candidate_logs:
            # any non-mesh log
            candidate_logs = [p for p in log_files
                              if p.name not in ("log.blockMesh", "log.checkMesh", "log.snappyHexMesh")]
        for log in candidate_logs:
            try:
                tail = log.read_text(encoding="utf-8", errors="replace")[-4000:]
            except Exception:
                continue
            if re.search(r"(?:^|\n)End\s*$", tail) or ("ExecutionTime" in tail and "End" in tail):
                converged = True
                break
    return {
        "compiled_so": str(so) if so else "",
        "case_dir": str(case_dir) if case_dir else "",
        "class_name": class_name,
        "converged": converged,
    }


# ---------------------------------------------------------------------------
# top-level run
# ---------------------------------------------------------------------------

def run(
    *,
    hypothesis: str,
    variant_name: str,
    run_dir: Path,
    starter_case: Path,
    topic: str,
    output_path: Path,
    model: str,
    timeout_s: int,
    max_turns: int,
) -> Dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    if not starter_case.exists():
        out = {"status": "FAILED", "error": f"starter_case not found: {starter_case}"}
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        return out

    repo_root = Path(__file__).resolve().parent.parent
    started_at = time.time()
    loop = run_agent_loop(
        repo_root=repo_root,
        hypothesis=hypothesis,
        variant_name=variant_name,
        run_dir=run_dir,
        starter_case=starter_case,
        topic=topic,
        model=model,
        max_turns=max_turns,
        timeout_s=timeout_s,
    )
    # Use the agent's claimed case_dir (from done) and variant_name to find
    # the .so even if wmake placed it in $FOAM_USER_LIBBIN. Filter by mtime
    # so stale .so files from prior runs are ignored.
    final_payload = loop.get("final_payload") or {}
    agent_case = str(final_payload.get("case_dir", "") or "")
    agent_so = str(final_payload.get("compiled_so", "") or "")
    artifacts = _find_compiled_artifacts(
        run_dir,
        variant_name=variant_name,
        agent_case_dir=agent_case,
        agent_compiled_so=agent_so,
        started_at=started_at,
    )
    success = bool(artifacts.get("compiled_so")) and artifacts.get("converged")
    result = {
        "status": "OK" if success else "FAILED",
        "duration_s": loop.get("duration_s", 0),
        "turns_used": loop.get("turns_used", 0),
        "aborted_reason": loop.get("aborted_reason", ""),
        "case_dir": artifacts.get("case_dir") or "",
        "class_name": artifacts.get("class_name") or variant_name,
        "compile_ok": bool(artifacts.get("compiled_so")),
        "converged": bool(artifacts.get("converged")),
        "compiled_so": artifacts.get("compiled_so") or "",
        "compiled_model_name": artifacts.get("class_name") or variant_name,
        "compiled_model_description": hypothesis[:240],
        "compiled_case_dir": artifacts.get("case_dir") or "",
        "trajectory_log": loop.get("trajectory_log", ""),
        "agent_final_payload": loop.get("final_payload", {}),
    }
    if not success:
        if not artifacts.get("compiled_so"):
            result["error"] = "no .so produced under customModels/ or $FOAM_USER_LIBBIN"
        elif not artifacts.get("converged"):
            result["error"] = "compiled .so but simpleFoam did not reach End"
        else:
            result["error"] = "unknown agentic failure"
        result["compile_error_hint"] = result["error"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Agentic OpenFOAM code-mod runner (provider-agnostic).")
    parser.add_argument("--hypothesis", required=True, type=str)
    parser.add_argument("--variant-name", required=True, type=str)
    parser.add_argument("--run-dir", required=True, type=str)
    parser.add_argument("--starter-case", required=True, type=str)
    parser.add_argument("--topic", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--model", default="", type=str,
                        help="Model name; defaults to env CFD_SCIENTIST_MODEL or gpt-5.4")
    parser.add_argument("--timeout", default=0, type=int,
                        help="Wall-clock timeout in seconds for the agentic loop. "
                             "0 (default) disables the wall-clock cap and bounds the loop only by --max-turns. "
                             "Set a positive value if you want a hard wall-clock fence.")
    parser.add_argument("--max-turns", default=120, type=int)
    args = parser.parse_args()

    model = args.model.strip() or os.environ.get("CFD_SCIENTIST_MODEL", "").strip() \
        or os.environ.get("FOAMAGENT_MODEL_VERSION", "").strip() or "gpt-5.4"

    result = run(
        hypothesis=args.hypothesis,
        variant_name=args.variant_name,
        run_dir=Path(args.run_dir).expanduser().resolve(),
        starter_case=Path(args.starter_case).expanduser().resolve(),
        topic=args.topic,
        output_path=Path(args.output).expanduser().resolve(),
        model=model,
        timeout_s=args.timeout,
        max_turns=args.max_turns,
    )
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ("agent_final_payload",)}, indent=2, default=str))
    return 0 if result.get("status") == "OK" else 2


if __name__ == "__main__":
    raise SystemExit(main())
