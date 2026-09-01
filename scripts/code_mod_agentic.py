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
        # Real solver launches this candidate has made — the measured cost the
        # search charges it. See _note_solver_invocations.
        self.solver_invocations = 0
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

    # Solver executables whose invocation is what a candidate actually costs.
    # Counted from the command text rather than assumed from the action type:
    # measured on run oed_20260822_1626_codex_high, candidate
    # sa_sr_destruction ran 35 simpleFoam solves inside a single candidate
    # doing its own coefficient sweep, and the search recorded it as cost 2 —
    # the same as a candidate that solved once. With strategy a free choice
    # rather than a fixed enum, an unmeasured cost makes an expensive strategy
    # look free, and the archive cannot honestly compare strategies at all.
    _SOLVER_TOKENS = (
        "simpleFoam", "pimpleFoam", "pisoFoam", "rhoCentralFoam", "rhoPimpleFoam",
        "rhoSimpleFoam", "buoyantSimpleFoam", "buoyantPimpleFoam", "interFoam",
        "potentialFoam", "sonicFoam", "Allrun",
    )

    def _note_solver_invocations(self, cmd: str) -> None:
        """Count solver launches implied by one shell command.

        A loop body counts once per token occurrence, not once per iteration:
        the exact count inside a `for` loop is not knowable from the text. That
        under-counts a sweep, so the number is a floor on real cost, never an
        over-estimate — a bound in the safe direction for budget accounting.
        """
        text = str(cmd or "")
        for token in self._SOLVER_TOKENS:
            hits = text.count(token)
            if hits:
                self.solver_invocations += hits

    # ---- tool: run_bash
    def run_bash(self, cmd: str, cwd: Optional[str] = None, timeout: Optional[int] = None) -> Dict[str, Any]:
        # A cwd check alone is not a filesystem sandbox: a shell command can
        # still redirect to the OpenFOAM installation or point wmake's LIB at
        # $FOAM_USER_LIBBIN. Run inside a mount namespace where the host is
        # read-only and only this candidate's run_dir is writable.
        cwd_p = Path(cwd).expanduser().resolve() if cwd else self.run_dir
        if not self._is_under(cwd_p, self.run_dir):
            return {"ok": False, "error": f"cwd must be inside run_dir: {cwd_p}"}
        if not cwd_p.is_dir():
            return {"ok": False, "error": f"cwd does not exist: {cwd_p}"}
        if not isinstance(cmd, str) or not cmd.strip():
            return {"ok": False, "error": "empty command"}
        self._note_solver_invocations(cmd)
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
        bwrap = shutil.which("bwrap")
        if not bwrap:
            return {
                "ok": False,
                "error": (
                    "bubblewrap is required for code-mod shell isolation; "
                    "refusing to run an unrestricted shell"
                ),
            }
        sandbox_cmd = [
            bwrap,
            "--die-with-parent",
            "--new-session",
            # Own PID namespace, so nothing this command starts can outlive it.
            #
            # Without it --die-with-parent bounds only the bwrap process, and a
            # `nohup ... &` inside the shell escapes onto the host and keeps
            # running. Measured on run closure_20260826_codex:
            # crossgrad_ksource_solver_fit backgrounded its optimiser, the
            # agent was killed at the wall clock 0.9 hours in, and the orphan
            # ran a further 5.7 hours -- finishing 16 of the 32 objective
            # evaluations it needed and writing them to a ledger nobody was
            # left to read. Under a PID namespace, bwrap's init exiting takes
            # every descendant with it, so the work either completes inside the
            # call that started it or does not happen at all. That is a
            # lifecycle guarantee rather than a check on what the command says,
            # so it cannot be worked around by phrasing the command differently.
            "--unshare-pid",
            "--ro-bind", "/", "/",
        ]
        try:
            self.run_dir.relative_to(Path("/tmp"))
            run_is_under_tmp = True
        except ValueError:
            run_is_under_tmp = False
        if not run_is_under_tmp:
            sandbox_cmd += ["--tmpfs", "/tmp"]
        sandbox_cmd += [
            "--bind", str(self.run_dir), str(self.run_dir),
            "--proc", "/proc",
            "--dev", "/dev",
            "--chdir", str(cwd_p),
            "bash", "-lc", full,
        ]
        t = min(timeout or self.bash_timeout, self.bash_timeout)
        try:
            proc = subprocess.run(
                sandbox_cmd,
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
      Run a shell command in a filesystem sandbox: the host is read-only and
      only the run directory (plus an ephemeral /tmp) is writable. The OpenFOAM
      bashrc is auto-sourced before each command, so wmake / blockMesh /
      simpleFoam etc. are on PATH. Returns {rc, stdout, stderr, timeout,
      error_summary}. ALWAYS check `error_summary` first — it pre-extracts
      gcc/wmake/FOAM/runtime error lines (with 2-line context) so you don't
      need to scan a long stdout to find the failure. A Make/files entry that
      targets $FOAM_USER_LIBBIN will fail because it is read-only: LIB output
      must be an absolute path below the case's customModels directory.
      `stdout` and `stderr`
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
    the build while redirecting the log to a path inside the candidate run
    directory (not /tmp, which is private to each shell call), then read that
    file. Do not give up after one failure — iterate until
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
    [3] Write Make/files     (source list + an absolute LIB output path below
        this case's customModels tree; never $FOAM_USER_LIBBIN)
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
   `wmake libso` (cwd inside that dir), with Make/files LIB output also below
   that case-local customModels tree. $FOAM_USER_LIBBIN is not an allowed
   compilation target.
4. Activation by case dictionaries only: system/controlDict.libs and the
   relevant constant/ dictionary (constant/momentumTransport, etc.).
"""


def build_agent_prompt(*, topic: str, hypothesis: str, variant_name: str, plan: str = "",
                       starter_case: Path, run_dir: Path,
                       wm_project_dir: Optional[Path],
                       prior_attempt: str = "", repair_goal: str = "",
                       timeout_s: int = 0) -> str:
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
        + (
            "HOW TO RUN A FIT, A SWEEP, OR AN OPTIMISER LOOP\n\n"
            "  1. NEVER background it. No `nohup`, no trailing `&`, no detached\n"
            "     process. Run it in the foreground and wait. If the wall clock\n"
            "     kills you, a backgrounded job keeps burning compute with nobody\n"
            "     left to collect its answer, and the work is lost anyway.\n\n"
            "  2. Probe the ends of the range BEFORE optimising. Evaluate the\n"
            "     objective at the lowest and highest value you would consider. If\n"
            "     it barely moves between them, the parameter does not matter:\n"
            "     stop, report that finding, and do not spend the budget searching\n"
            "     a flat function.\n\n"
            "  3. Set bounds you are willing to be wrong about, and check them. If\n"
            "     the optimiser returns a value sitting on a bound, the optimum is\n"
            "     probably outside it and your answer is an artefact of the box.\n\n"
            "  4. A handful of evaluations is not a fit. If your optimiser reports\n"
            "     success after two or three objective calls, it has not searched\n"
            "     anything -- say so rather than reporting the number as fitted.\n\n"
            "  5. Write every objective evaluation to a ledger file on disk AS IT\n"
            "     HAPPENS -- one JSON line per evaluation with the parameter value,\n"
            "     the score, and the elapsed seconds. If you are killed, that\n"
            "     ledger is the only evidence the fit was working, and it is read\n"
            "     when deciding whether to give this candidate more time.\n\n"
            "  6. When the fit finishes, WRITE THE FITTED VALUE INTO THE FILE THE\n"
            "     RUN ACTUALLY READS, and verify it took effect. A value left in a\n"
            "     Python variable or a JSON file means the run uses its default --\n"
            "     which scores as the unmodified baseline while looking like a real\n"
            "     experiment, and is the single most expensive mistake available\n"
            "     here.\n\n"
            "STRATEGY PLAN — carry these steps out. They may require work before any\n"
            "model code is written: reading high-fidelity data, running a fit or an\n"
            "optimiser, and turning the fitted result into the model's coefficients or\n"
            "form. You have a shell and the study's Python libraries; use them. The\n"
            "deliverable below is unchanged either way — a compiled model and a case\n"
            "that runs — but HOW you arrive at the model is what this plan specifies.\n"
            f"{plan.strip()}\n\n"
            if str(plan or "").strip() else ""
        )
        + (
            # A continued attempt, not a fresh one. Without this the agent
            # restarts from an empty mental slate against a directory that
            # already holds its own compiled library, its fit artifacts and
            # its half-finished case, and spends its second budget redoing
            # the first one. Measured on run closure_20260826_codex: every
            # solver_fit candidate died at the wall clock mid-optimisation,
            # and a plain re-run would have repeated the compile and the
            # first optimiser iterations before reaching new ground.
            "============================================================\n"
            "THIS IS A CONTINUATION OF AN EARLIER ATTEMPT\n"
            "============================================================\n"
            f"{prior_attempt.strip()}\n\n"
            "Your run directory already contains whatever that attempt produced.\n"
            "INSPECT IT FIRST and build on it. Re-compiling a library that is\n"
            "already built, or re-running an optimiser iteration whose result is\n"
            "already on disk, spends the extra time you were granted on work that\n"
            "is already done. Only redo something if you find it is wrong or\n"
            "incomplete.\n\n"
            if str(prior_attempt or "").strip() else ""
        )
        + (
            # The agent could not previously see its own deadline, so it could
            # not plan against it. Measured on run closure_20260826_codex:
            # crossgrad_ksource_solver_fit launched a differential-evolution
            # fit needing 32 objective evaluations at ~23 minutes each -- 12.4
            # hours of solver time against a 52-minute fence. It was killed at
            # 16 evaluations having never had a chance. The arithmetic was
            # knowable before the first solve; the agent just was not told the
            # one number that makes it possible.
            "============================================================\n"
            "YOUR TIME BUDGET\n"
            "============================================================\n"
            f"You have {timeout_s} seconds ({timeout_s / 60:.0f} minutes) of wall clock.\n"
            "When it runs out you are killed where you stand, and anything not\n"
            "finished and written to disk is lost.\n\n"
            "BEFORE you start any fit, sweep, or optimiser loop, cost it out and say\n"
            "the arithmetic out loud in your reasoning:\n"
            "    (number of objective evaluations) x (cases per evaluation)\n"
            "        x (seconds per case) = total seconds\n"
            "Time ONE case first if you do not know the per-case cost -- guessing it\n"
            "is what makes the arithmetic worthless. If the total does not fit in the\n"
            "budget above with room to spare for compiling, deploying and verifying,\n"
            "DO NOT START IT. Shrink it until it fits: fewer fit cases, fewer\n"
            "optimiser evaluations, a coarser convergence criterion, a scalar\n"
            "bounded search instead of a population method. A smaller fit that\n"
            "FINISHES is worth more than a thorough one that is killed halfway,\n"
            "because a killed fit scores as the unmodified baseline and teaches the\n"
            "search that the whole approach fails.\n\n"
            "If the honest arithmetic says the work cannot fit even after shrinking,\n"
            "say so and call `done` with what you have, explaining the cost. That is a\n"
            "useful result. Being killed at the fence is not.\n\n"
            if timeout_s > 0 else ""
        )
        + f"WM_PROJECT_DIR (read-only OpenFOAM install): {wm_project_dir or '(unset)'}\n"
        + f"Starter case (read-only base): {starter_case}\n"
        + f"Run directory (your writable workspace): {run_dir}\n\n"
        + (
            # Repair mode: the model and the case already exist and something
            # about the plumbing around them is broken. The full deliverable
            # below would have the agent rebuild from scratch, which is both
            # wasteful and dangerous -- a "repair" that re-derives the closure
            # is a different experiment, not a repair.
            "============================================================\n"
            "REPAIR TASK — this overrides the deliverable below\n"
            "============================================================\n"
            "A previous attempt already produced this candidate. It is broken in a\n"
            "specific, diagnosed way. Carry out EXACTLY this repair and nothing\n"
            "else:\n\n"
            f"{repair_goal.strip()}\n\n"
            "Hard limits on what a repair may touch. You may fix our own plumbing:\n"
            "a library that is not being loaded, a missing or misspelled entry in a\n"
            "case dictionary, a coefficient the earlier attempt computed but never\n"
            "wrote into the case, a solver tolerance, a broken post-processing step.\n"
            "You may NOT change the mesh, the boundary conditions, the physics, the\n"
            "endTime, the numerics being graded, or the closure itself -- changing\n"
            "any of those makes this a different experiment and invalidates the\n"
            "comparison. If the repair you were given cannot be carried out without\n"
            "crossing that line, do NOT improvise a different one: call `done` and\n"
            "say so plainly in the summary.\n\n"
            "When the repair is made, re-run the case exactly as it was configured\n"
            "and confirm it reaches End, then call `done`.\n\n"
            if str(repair_goal or "").strip() else ""
        )
        + (
            # The transferable lesson, not the instance of it.
            #
            # An earlier version of this block spelled out five kOmegaSST
            # derivation rules -- include order, template arity, constructor
            # signature, include guards, registration. That fixed one real
            # candidate and was wrong to put here: this runner is generic
            # across viscosity models, boundary conditions, fvOptions and
            # schemes, and hardcoding one framework's turbulence-closure
            # boilerplate makes every other kind of study carry irrelevant
            # instructions. What generalises is the DEBUGGING HEURISTIC and
            # "copy a working example", which apply to extending any compiled
            # library in any framework.
            "============================================================\n"
            "WHEN A BUILD FAILS\n"
            "============================================================\n"
            "If a compiler error points at a line inside the READ-ONLY library\n"
            "tree rather than inside a file you wrote, the bug is almost always in\n"
            "your file, not theirs. A missing type, a template arity mismatch or a\n"
            "redefinition reported deep inside a framework header usually means\n"
            "your headers are included in the wrong order, your class does not\n"
            "match the base class's expected form, or a translation unit is being\n"
            "compiled twice. Re-read your own files against a working example\n"
            "before reading more library source.\n\n"
            "The fastest way to get the skeleton right is to copy it. Find an\n"
            "existing extension of the same kind -- in the framework's own source\n"
            "tree, in the starter case, or in a sibling candidate directory that\n"
            "already compiled -- and match its structure exactly: the same\n"
            "includes in the same order, the same class declaration and base-class\n"
            "form, the same constructor signature and argument order, the same\n"
            "registration macro, the same build-file layout. Change only the\n"
            "physics you are here to change. Deriving that skeleton from first\n"
            "principles is how a build budget gets spent on boilerplate.\n\n"
        )
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

_NO_FUNCTION_CALL_OVERRIDE = (
    "\n\nOVERRIDE (highest priority): no callable tool is registered for this "
    "turn. Ignore any impulse to emit a function call. The tools described "
    "above are invoked by writing a JSON object as ordinary text — reply with "
    "that JSON object and nothing else."
)


def _message_text(message: Any) -> str:
    """Text of an LLM reply, or "" when it carries none.

    Gemini answers a prompt that describes a tool protocol by attempting a
    real function call; when that call is malformed the API returns
    finish_reason=MALFORMED_FUNCTION_CALL and NO content parts. The previous
    `getattr(resp, "content", None) or str(resp)` then fell through to
    stringifying the whole AIMessage, producing "content=[] additional_kwargs=
    {...}" — text that can never parse as a tool call, so the run burned its
    parse-fail budget and aborted. Every code-mod candidate in a Gemini run
    died this way.
    """
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            else:
                text_attr = getattr(block, "text", None)
                if isinstance(text_attr, str):
                    parts.append(text_attr)
        return "".join(parts).strip()
    return ""


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
    plan: str = "",
    variant_name: str,
    run_dir: Path,
    starter_case: Path,
    topic: str,
    model: str,
    max_turns: int,
    timeout_s: int,
    prior_attempt: str = "",
    repair_goal: str = "",
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
    # Appended, never truncated, when this is a continuation or a repair: the
    # earlier attempt's trajectory is the evidence that justified granting more
    # time, and overwriting it would destroy the record of why. A fresh attempt
    # still starts clean.
    continuing = bool(str(prior_attempt or "").strip() or str(repair_goal or "").strip())
    log_fh = open(trajectory_log, "a" if continuing else "w", encoding="utf-8")

    def log(msg: str) -> None:
        log_fh.write(msg + "\n")
        log_fh.flush()

    if continuing:
        log("\n" + "=" * 60)
        log(f"# CONTINUATION at {time.strftime('%Y-%m-%d %H:%M:%S')} "
            f"(timeout_s={timeout_s}, max_turns={max_turns})")
        log("=" * 60)
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
        plan=plan,
        variant_name=variant_name,
        starter_case=starter_case,
        run_dir=run_dir,
        wm_project_dir=wm_path,
        prior_attempt=prior_attempt,
        repair_goal=repair_goal,
        timeout_s=timeout_s,
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

    def render_transcript(cap: Optional[int] = None) -> str:
        cap = TRANSCRIPT_CAP if cap is None else cap
        if not transcript_chunks:
            return ""
        full = "\n\n=== CONVERSATION SO FAR ===\n" + "\n".join(transcript_chunks) + "\n=== END CONVERSATION ===\n\nNow output your next tool-call JSON object."
        if len(full) <= cap:
            return full
        # Truncate from the front (keep most recent turns); always keep the
        # marker lines so the agent knows it's seeing a truncated view.
        keep = full[-cap:]
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
        # claude-agent-sdk subprocess transport dying mid-stream. The CLI exits
        # non-zero writing nothing to stderr, so the only signature is this
        # text. Measured on run oed_20260823_opus_low: all seven candidates
        # died here, and because the message matched none of the patterns above
        # the loop treated it as permanent and broke after ONE attempt while
        # reporting four. The provider wrapper now retries first; this stays as
        # the outer net for the case where the wrapper exhausts its own.
        "command failed with exit code",
        "fatal error in message reader",
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
            # attempt+1, not MAX+1: a non-transient error breaks out after the
            # first try, and reporting a fixed "4 attempts" there sent the
            # investigation of run oed_20260823_opus_low looking for a retry
            # exhaustion that had never happened.
            aborted_reason = (f"llm.invoke raised after {attempt + 1} attempt(s): "
                              f"{type(last_exc).__name__ if last_exc else 'Unknown'}: "
                              f"{str(last_exc)[:300] if last_exc else ''}")
            log(f"# {aborted_reason}")
            break
        ai_text = _message_text(ai_resp)
        if not ai_text:
            # Empty reply: retry with the tool-call impulse countermanded in
            # the SYSTEM message, which is where the tool protocol is
            # described. Measured: a user-turn nudge is not enough, the same
            # words in the system message are.
            #
            # Escalate rather than trying once. On a real run this override
            # recovered 6 of 9 empty replies, but the three it missed were
            # consecutive and ended the candidate at the parse-fail cap, so
            # the case ran with stock SpalartAllmaras and produced nothing.
            # Each retry also shrinks the transcript: MALFORMED_FUNCTION_CALL
            # gets likelier as context grows (these turns follow several 10k
            # source-file reads), so re-sending the identical prompt is the
            # one variation least likely to help.
            finish = (getattr(ai_resp, "response_metadata", {}) or {}).get("finish_reason", "")
            for override_attempt, cap in enumerate(
                (TRANSCRIPT_CAP, TRANSCRIPT_CAP // 3, TRANSCRIPT_CAP // 10), start=1
            ):
                log(f"# empty reply (finish_reason={finish!r}); tool-call override "
                    f"attempt {override_attempt}/3 (transcript cap {cap})")
                try:
                    ai_resp = llm.invoke([
                        SystemMessage(content=sys_msg_text + _NO_FUNCTION_CALL_OVERRIDE),
                        HumanMessage(content=initial_user_prompt + render_transcript(cap)),
                    ])
                    ai_text = _message_text(ai_resp)
                except Exception as exc:
                    log(f"# override retry raised: {type(exc).__name__}: {str(exc)[:200]}")
                    ai_text = ""
                if ai_text:
                    break
                finish = (getattr(ai_resp, "response_metadata", {}) or {}).get("finish_reason", "")
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
        "solver_invocations": getattr(sandbox, "solver_invocations", 0),
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
    convergence. Only artifacts below <case>/customModels/... are eligible.
    $FOAM_USER_LIBBIN and the OpenFOAM installation are excluded: all source
    and compilation outputs must remain local to the candidate case.

    Generic across modification kinds — turbulence, viscosity, BC class, source,
    or any other custom OpenFOAM derivative; only the file pattern matters.
    """
    candidates: List[Path] = []
    # Trust the claimed .so only when it is under run_dir/customModels.
    if agent_compiled_so:
        ap = Path(agent_compiled_so)
        try:
            rel = ap.resolve().relative_to(run_dir.resolve())
            if "customModels" in rel.parts and ap.is_file() and ap.name.startswith("lib") and ap.suffix == ".so":
                candidates.append(ap)
        except Exception:
            pass
    # Scan any case-local .so under customModels/, regardless of whether
    # the agent used the conventional `platforms/<arch>/` layout or pointed
    # `LIB =` at a sibling directory like `../lib/lib<X>`. Both are valid
    # outputs of `wmake libso`.
    for candidate in run_dir.rglob("customModels/**/lib*.so"):
        try:
            rel = candidate.resolve().relative_to(run_dir.resolve())
        except Exception:
            continue
        # Reject symlinks that resolve outside the case-local tree.
        if "customModels" in rel.parts:
            candidates.append(candidate)

    if started_at > 0:
        # Filter to .so files modified DURING this run only — avoid stale.
        fresh = [p for p in candidates if p.is_file() and p.stat().st_mtime >= started_at - 5]
        candidates = fresh
    # A text file named lib*.so is not compilation evidence. Read only the
    # header; real OpenFOAM libraries can be large.
    def _has_elf_header(path: Path) -> bool:
        try:
            with path.open("rb") as fh:
                return fh.read(4) == b"\x7fELF"
        except OSError:
            return False

    candidates = [p for p in candidates if p.is_file() and _has_elf_header(p)]

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
        if (
            cm
            and (cm.parent / "constant").is_dir()
            and (cm.parent / "system" / "controlDict").is_file()
        ):
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
        control_text = (case_dir / "system" / "controlDict").read_text(
            encoding="utf-8", errors="replace"
        )
        app_match = re.search(r"(?m)^\s*application\s+([^;\s]+)\s*;", control_text)
        application = app_match.group(1) if app_match else ""
        candidate_logs = [case_dir / f"log.{application}"] if application else []
        for log in candidate_logs:
            if not log.is_file():
                continue
            try:
                text = log.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            tail = text[-8000:]
            if (
                "OpenFOAM" in text[:8000]
                and "ExecutionTime" in tail
                and re.search(r"(?m)^\s*End\s*$", tail)
                and "FOAM FATAL" not in tail
            ):
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
    plan: str = "",
    variant_name: str,
    run_dir: Path,
    starter_case: Path,
    topic: str,
    output_path: Path,
    model: str,
    timeout_s: int,
    max_turns: int,
    prior_attempt: str = "",
    repair_goal: str = "",
) -> Dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    if not starter_case.exists():
        out = {"status": "FAILED", "error": f"starter_case not found: {starter_case}"}
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        return out

    repo_root = Path(__file__).resolve().parent.parent
    started_at = time.time()
    continuing = bool(str(prior_attempt or "").strip() or str(repair_goal or "").strip())
    loop = run_agent_loop(
        repo_root=repo_root,
        hypothesis=hypothesis,
        plan=plan,
        variant_name=variant_name,
        run_dir=run_dir,
        starter_case=starter_case,
        topic=topic,
        model=model,
        max_turns=max_turns,
        timeout_s=timeout_s,
        prior_attempt=prior_attempt,
        repair_goal=repair_goal,
    )
    # Use the claimed case_dir and variant_name to find a fresh case-local
    # .so. Outputs in $FOAM_USER_LIBBIN are never accepted.
    final_payload = loop.get("final_payload") or {}
    agent_case = str(final_payload.get("case_dir", "") or "")
    agent_so = str(final_payload.get("compiled_so", "") or "")
    artifacts = _find_compiled_artifacts(
        run_dir,
        variant_name=variant_name,
        agent_case_dir=agent_case,
        agent_compiled_so=agent_so,
        # The staleness filter exists to reject a library left behind by an
        # earlier, different build. On a continuation or a repair the earlier
        # build is this candidate's own work and is exactly what we want the
        # agent to reuse -- filtering it out would report FAILED for a
        # continuation that finished the fit without needing to recompile,
        # which is the common case. Scoping to this run_dir's customModels,
        # the ELF-header check, and the clean-solver-log requirement for
        # `converged` all still apply.
        started_at=0.0 if continuing else started_at,
    )
    success = bool(artifacts.get("compiled_so")) and artifacts.get("converged")
    result = {
        "status": "OK" if success else "FAILED",
        "duration_s": loop.get("duration_s", 0),
        "turns_used": loop.get("turns_used", 0),
        "solver_invocations": loop.get("solver_invocations", 0),
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
            result["error"] = "no fresh case-local .so produced under customModels/"
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
    parser.add_argument("--plan", default="", type=str,
                        help="Optional strategy steps: what to read, what to fit or optimise, "
                             "and what the fitted result becomes. When empty, the hypothesis "
                             "is the whole instruction.")
    parser.add_argument("--max-turns", default=120, type=int)
    parser.add_argument("--prior-attempt", default="", type=str,
                        help="What an earlier attempt at this same candidate did and why it "
                             "stopped. Turns this into a CONTINUATION: the agent is told to "
                             "inspect and build on the work already in the run dir rather than "
                             "start over, the trajectory log is appended instead of truncated, "
                             "and a library compiled by the earlier attempt is accepted.")
    parser.add_argument("--repair-goal", default="", type=str,
                        help="A specific, diagnosed defect to fix in an existing candidate. "
                             "Replaces the build deliverable with a bounded repair task that "
                             "may touch our own plumbing but never the mesh, physics, endTime "
                             "or the closure under test.")
    args = parser.parse_args()

    model = args.model.strip() or os.environ.get("CFD_SCIENTIST_MODEL", "").strip() \
        or os.environ.get("FOAMAGENT_MODEL_VERSION", "").strip() or "gpt-5.4"

    result = run(
        hypothesis=args.hypothesis,
        plan=args.plan,
        variant_name=args.variant_name,
        run_dir=Path(args.run_dir).expanduser().resolve(),
        starter_case=Path(args.starter_case).expanduser().resolve(),
        topic=args.topic,
        output_path=Path(args.output).expanduser().resolve(),
        model=model,
        timeout_s=args.timeout,
        max_turns=args.max_turns,
        prior_attempt=args.prior_attempt,
        repair_goal=args.repair_goal,
    )
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ("agent_final_payload",)}, indent=2, default=str))
    return 0 if result.get("status") == "OK" else 2


if __name__ == "__main__":
    raise SystemExit(main())
