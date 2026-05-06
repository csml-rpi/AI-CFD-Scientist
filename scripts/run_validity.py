#!/usr/bin/env python3
"""
Run-validity gate, time-pinned scoring helpers, LLM-driven Allrun pre-flight,
and LLM-driven runtime-investigator helper.

Owned end-to-end by this module to keep open_ended_discovery.py surgical:
- detect_max_time(case_dir): find largest numeric time directory.
- detect_baseline_final_time(...): resolve baseline_final_time from many sources.
- gate(...): run-validity gate after every runtime/agentic/code-mod run.
- preflight_allrun(...): LLM-driven Allrun pre-flight (with regex fallback).
- investigate_runtime(...): LLM classifies a RUN_INVALID iteration's root cause
  and (optionally) proposes a harness or model patch.

LLM call style mirrors `_llm_invoke` from oed_extensions.py and uses
`cfd_langgraph.config.get_settings()` to pick provider/model.
"""
from __future__ import annotations

import json
import multiprocessing as _mp
import os
import pickle as _pickle
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# small helpers (mirroring open_ended_discovery / oed_extensions style)
# ---------------------------------------------------------------------------

def _read_json(p: Path, default: Any) -> Any:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = "\n".join(s.split("\n")[1:])
        s = s.rsplit("```", 1)[0]
    return s.strip()


def _llm_worker(msgs_pickle: bytes, model: str, temp: float, queue: Any) -> None:
    """Child-process worker for `_llm_invoke`. Defined at module scope so the
    `fork` multiprocessing context can locate it cleanly. Pushes a
    (kind, payload) tuple onto `queue` where kind is 'ok' or 'err'.
    """
    try:
        from langchain_core.messages import HumanMessage, SystemMessage  # type: ignore
        from cfd_langgraph.llm.factory import create_langchain_llm  # type: ignore
        msgs_in = _pickle.loads(msgs_pickle)
        msgs = []
        for role, content in msgs_in:
            cls = SystemMessage if role == "system" else HumanMessage
            msgs.append(cls(content=content))
        llm = create_langchain_llm(model=model, temperature=temp)
        raw = llm.invoke(msgs)
        queue.put(("ok", str(getattr(raw, "content", raw))))
    except Exception as e:  # pragma: no cover — child-process error path
        import traceback
        queue.put(("err", f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=5)}"))


def _llm_invoke(messages: List[Tuple[str, str]], temperature: float = 0.0,
                timeout_s: int = 600) -> str:
    """Single-shot LLM call with hard timeout. messages = [(role, content), ...]
    with role in {'system','user'}. Returns content string.

    Runs ``llm.invoke`` in a forked child process. On timeout, the child is
    terminated (SIGTERM then SIGKILL) so any descendant SDK subprocess
    (e.g. the ``claude`` binary spawned by claude-code) is also reaped.
    Raises:
      - ``TimeoutError`` on timeout (default 600s).
      - ``RuntimeError`` if the child terminates without output or raised.
    """
    from cfd_langgraph.config import get_settings  # type: ignore
    model = get_settings().model
    ctx = _mp.get_context("fork")
    q = ctx.Queue()
    p = ctx.Process(
        target=_llm_worker,
        args=(_pickle.dumps(messages), model, temperature, q),
    )
    p.start()
    p.join(timeout=timeout_s)
    if p.is_alive():
        p.terminate()
        p.join(timeout=5)
        if p.is_alive():
            p.kill()
            p.join(timeout=2)
        raise TimeoutError(
            f"LLM call exceeded {timeout_s}s; SDK subprocess killed"
        )
    if q.empty():
        raise RuntimeError("LLM child terminated without producing output")
    kind, payload = q.get_nowait()
    if kind == "err":
        raise RuntimeError(f"LLM child raised: {payload}")
    return payload


def _load_prompts() -> Dict[str, Any]:
    """Best-effort load of prompts.yaml for RunValidityAgent block."""
    try:
        from cfd_langgraph.config import get_settings  # type: ignore
        from cfd_langgraph.ideation import load_prompts  # type: ignore
        return load_prompts(get_settings().prompts_path) or {}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# time-directory detection
# ---------------------------------------------------------------------------

_NUMERIC_RE = re.compile(r"^[0-9]+(\.[0-9]+)?(e[+-]?[0-9]+)?$", re.IGNORECASE)


def _is_numeric_dirname(name: str) -> bool:
    return bool(_NUMERIC_RE.match(name))


def _list_time_dirs(case_dir: Path) -> List[float]:
    out: List[float] = []
    if not case_dir.is_dir():
        return out
    for child in case_dir.iterdir():
        if not child.is_dir():
            continue
        if not _is_numeric_dirname(child.name):
            continue
        try:
            out.append(float(child.name))
        except Exception:
            continue
    return sorted(out)


def detect_max_time(case_dir: Path) -> float:
    """
    Largest numeric time directory in case_dir.

    Rules:
      - skip 0 / 0.orig only when other times exist
      - if only 0 exists -> 0.0
      - falls back to processor0/<latest> if decomposed and serial dirs absent
    """
    if not case_dir or not Path(case_dir).is_dir():
        return 0.0
    case_dir = Path(case_dir)
    times = _list_time_dirs(case_dir)
    non_zero = [t for t in times if t > 0]
    if non_zero:
        return float(max(non_zero))
    # decomposed fallback
    p0 = case_dir / "processor0"
    if p0.is_dir():
        ptimes = _list_time_dirs(p0)
        non_zero_p = [t for t in ptimes if t > 0]
        if non_zero_p:
            return float(max(non_zero_p))
    if times:
        return float(max(times))
    return 0.0


# ---------------------------------------------------------------------------
# baseline final-time resolution
# ---------------------------------------------------------------------------

_ENDTIME_RE = re.compile(r"^\s*endTime\s+([0-9.eE+\-]+)\s*;", re.MULTILINE)


def _grep_endtime_from_controldict(control_dict: Path) -> Optional[float]:
    try:
        text = control_dict.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None
    m = _ENDTIME_RE.search(text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def detect_baseline_final_time(
    *,
    baseline_metrics: Optional[Dict[str, Any]] = None,
    base_case: Optional[Path] = None,
) -> Optional[float]:
    """First-hit-wins resolution:
        1. baseline_metrics["baseline_final_time"]
        2. scan baseline_case_dir time dirs (from baseline_metrics)
        3. grep endTime from <base_case>/system/controlDict
        4. None (degraded gate, only enforces max_time>0)
    """
    if isinstance(baseline_metrics, dict):
        v = baseline_metrics.get("baseline_final_time")
        try:
            if v is not None:
                return float(v)
        except Exception:
            pass
        bc_str = baseline_metrics.get("baseline_case_dir")
        if bc_str:
            bc = Path(str(bc_str))
            if bc.is_dir():
                t = detect_max_time(bc)
                if t > 0:
                    return float(t)
    if base_case:
        cd = Path(base_case) / "system" / "controlDict"
        if cd.is_file():
            t = _grep_endtime_from_controldict(cd)
            if t is not None and t > 0:
                return float(t)
    return None


# ---------------------------------------------------------------------------
# diagnostic-bundle assembly
# ---------------------------------------------------------------------------

def _truncate(s: str, n: int) -> str:
    if s is None:
        return ""
    if len(s) <= n:
        return s
    return s[:n] + f"\n...<truncated {len(s)-n} bytes>"


def _list_dir_compact(p: Path, max_entries: int = 200) -> List[str]:
    if not p.is_dir():
        return []
    entries: List[str] = []
    try:
        for child in sorted(p.iterdir()):
            tag = child.name + ("/" if child.is_dir() else "")
            entries.append(tag)
            if len(entries) >= max_entries:
                entries.append("...<truncated>")
                break
    except Exception:
        pass
    return entries


def _read_log_tails(case_dir: Path, tail_bytes: int = 2000) -> Dict[str, str]:
    out: Dict[str, str] = {}
    try:
        for p in sorted(case_dir.glob("log.*")):
            if not p.is_file():
                continue
            try:
                txt = p.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            out[p.name] = txt[-tail_bytes:] if len(txt) > tail_bytes else txt
    except Exception:
        pass
    return out


def _last_residuals(case_dir: Path, n_lines: int = 50) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for p in sorted(case_dir.glob("log.*")):
        if not p.is_file():
            continue
        try:
            lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
            out[p.name] = "\n".join(lines[-n_lines:])
        except Exception:
            continue
    return out


def _build_diagnostic_bundle(
    *,
    case_dir: Path,
    max_time: float,
    baseline_final_time: Optional[float],
    min_required_time: float,
    runtime_run_result: Optional[Dict[str, Any]],
    reason: str,
) -> Dict[str, Any]:
    allrun = case_dir / "Allrun"
    allrun_text = ""
    if allrun.is_file():
        try:
            allrun_text = _truncate(allrun.read_text(encoding="utf-8", errors="replace"), 8 * 1024)
        except Exception:
            allrun_text = ""
    log_tails = _read_log_tails(case_dir)
    cd = case_dir / "system" / "controlDict"
    cd_endtime = _grep_endtime_from_controldict(cd) if cd.is_file() else None
    rrr = runtime_run_result or {}
    bundle = {
        "max_time": float(max_time),
        "baseline_final_time": baseline_final_time,
        "min_required_time": float(min_required_time),
        "allrun_path": str(allrun) if allrun.is_file() else "",
        "allrun_contents": allrun_text,
        "log_files_present": list(log_tails.keys()),
        "log_simpleFoam_present": ("log.simpleFoam" in log_tails),
        "log_tails": log_tails,
        "case_dir_listing": _list_dir_compact(case_dir),
        "controldict_endtime": cd_endtime,
        "stdout_tail": str(rrr.get("stdout_tail", ""))[-2000:],
        "stderr_tail": str(rrr.get("stderr_tail", ""))[-2000:],
        "last_residuals": _last_residuals(case_dir),
        "reason": reason,
    }
    return bundle


# ---------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------

def gate(
    *,
    case_dir: Path,
    baseline_metrics: Optional[Dict[str, Any]],
    runtime_run_result: Optional[Dict[str, Any]],
    base_case: Optional[Path],
    min_progress_fraction: float = 0.5,
) -> Dict[str, Any]:
    """
    Returns:
      {
        "valid": bool,
        "status": "RUN_INVALID" | "RUN_OK",
        "max_time": float,
        "baseline_final_time": float|None,
        "min_required_time": float,
        "reason": str,
        "diagnostic_bundle_path": "<case_dir>/run_validity_diagnostic.json"
      }
    """
    case_dir = Path(case_dir)
    max_time = detect_max_time(case_dir)
    baseline_final_time = detect_baseline_final_time(
        baseline_metrics=baseline_metrics, base_case=base_case,
    )

    # Honor case's own controlDict if it differs (smaller) from baseline.
    case_endtime = None
    cd = case_dir / "system" / "controlDict"
    if cd.is_file():
        case_endtime = _grep_endtime_from_controldict(cd)

    if baseline_final_time is None and case_endtime is None:
        # degraded: only enforce max_time > 0
        if max_time <= 0.0:
            reason = (
                "RUN_INVALID: no time directories advanced past 0 (no flow solver "
                "appears to have run); baseline_final_time and case controlDict "
                "endTime both unavailable."
            )
            valid = False
            status = "RUN_INVALID"
            min_required = 0.0
        else:
            reason = "RUN_OK (degraded gate: only enforced max_time>0; baseline_final_time unavailable)"
            valid = True
            status = "RUN_OK"
            min_required = 0.0
    else:
        # Pick the effective target. If case_endtime is markedly smaller than
        # baseline_final_time (e.g. agentic chose a smoke-test endTime), use it
        # to avoid false flags.
        if case_endtime is not None and baseline_final_time is not None:
            effective_target = min(float(case_endtime), float(baseline_final_time))
        elif baseline_final_time is not None:
            effective_target = float(baseline_final_time)
        else:
            effective_target = float(case_endtime or 0.0)
        min_required = float(min_progress_fraction) * float(effective_target)
        if max_time < min_required or max_time <= 0.0:
            reason = (
                f"RUN_INVALID: max_time={max_time} < {min_progress_fraction*100:.0f}% of "
                f"effective_target={effective_target} (baseline_final_time="
                f"{baseline_final_time}, case_endtime={case_endtime}); the flow solver "
                f"either did not run or stopped before producing meaningful results."
            )
            valid = False
            status = "RUN_INVALID"
        else:
            reason = (
                f"RUN_OK: max_time={max_time} >= {min_required} "
                f"(effective_target={effective_target})."
            )
            valid = True
            status = "RUN_OK"

    bundle = _build_diagnostic_bundle(
        case_dir=case_dir,
        max_time=max_time,
        baseline_final_time=baseline_final_time,
        min_required_time=min_required,
        runtime_run_result=runtime_run_result,
        reason=reason,
    )
    bundle_path = case_dir / "run_validity_diagnostic.json"
    try:
        _write_json(bundle_path, bundle)
    except Exception:
        pass

    return {
        "valid": valid,
        "status": status,
        "max_time": float(max_time),
        "baseline_final_time": baseline_final_time,
        "min_required_time": float(min_required),
        "reason": reason,
        "diagnostic_bundle_path": str(bundle_path),
    }


# ---------------------------------------------------------------------------
# LLM-driven Allrun pre-flight
# ---------------------------------------------------------------------------

# Heuristic: any non-comment line that runApplication/runParallel-invokes a
# *Foam binary OR matches the controlDict `application` field.
_HEURISTIC_FOAM_RE = re.compile(
    r"^[^#]*\b(?:runApplication|runParallel)\s+(?P<solver>\w+)",
    re.MULTILINE,
)


def _read_application(case_dir: Path) -> str:
    cd = case_dir / "system" / "controlDict"
    if not cd.is_file():
        return ""
    try:
        text = cd.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""
    m = re.search(r"^\s*application\s+(\w+)\s*;", text, re.MULTILINE)
    return (m.group(1) if m else "").strip()


def _heuristic_allrun_check(allrun_text: str, application: str) -> Tuple[bool, str]:
    """Return (looks_ok, why). looks_ok=True means we *think* a flow solver runs."""
    if not allrun_text:
        return False, "Allrun is empty or missing"
    found = []
    for m in _HEURISTIC_FOAM_RE.finditer(allrun_text):
        sol = m.group("solver")
        found.append(sol)
    if not found:
        return False, "no runApplication/runParallel invocation of any solver found"
    # Prefer matches against controlDict application or *Foam binaries.
    for sol in found:
        if application and sol == application:
            return True, f"runApplication/runParallel invokes controlDict application={sol}"
        if sol.endswith("Foam"):
            return True, f"runApplication/runParallel invokes a *Foam binary: {sol}"
    # Found something but not an obvious flow solver.
    return False, f"runApplication invocations did not match a *Foam binary or controlDict application: {found}"


def preflight_allrun(
    *,
    case_dir: Path,
    repo_root: Path,
    enable_llm: bool = True,
    llm_timeout_s: int = 30,
) -> Dict[str, Any]:
    """
    LLM-driven Allrun audit. Returns dict described in the plan.

    On verdict==BROKEN with corrected text, save original to
    `<case>/Allrun.preflight_original` and write the corrected Allrun (chmod +x);
    flip verdict to PATCHED.

    On LLM failure, fall back to a regex heuristic. In heuristic mode we do NOT
    auto-patch — let the run-validity gate catch a broken Allrun.
    """
    t0 = time.time()
    case_dir = Path(case_dir)
    allrun = case_dir / "Allrun"
    if not allrun.is_file():
        return {
            "checked": False, "verdict": "BROKEN", "method": "none",
            "rationale": f"Allrun missing at {allrun}",
            "original_allrun": "", "applied_allrun": None,
            "elapsed_s": time.time() - t0,
        }
    try:
        original = allrun.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return {
            "checked": False, "verdict": "BROKEN", "method": "none",
            "rationale": f"failed to read Allrun: {exc}",
            "original_allrun": "", "applied_allrun": None,
            "elapsed_s": time.time() - t0,
        }
    application = _read_application(case_dir)

    if enable_llm:
        try:
            prompts = _load_prompts() or {}
            block = prompts.get("RunValidityAgent", {}) or {}
            sys_msg = block.get(
                "allrun_preflight_system_prompt",
                _DEFAULT_ALLRUN_SYSTEM_PROMPT,
            )
            user_tmpl = block.get(
                "allrun_preflight_user_prompt",
                _DEFAULT_ALLRUN_USER_PROMPT,
            )
            ls_listing = "\n".join(_list_dir_compact(case_dir, 80))
            user_msg = user_tmpl.format(
                allrun_text=original[:8000],
                application=application or "(unknown)",
                case_listing=ls_listing,
            )
            raw = _llm_invoke(
                [("system", sys_msg), ("user", user_msg)],
                temperature=0.0,
            )
            body = _strip_code_fences(raw)
            s, e = body.find("{"), body.rfind("}")
            if s < 0 or e < 0:
                raise ValueError("no JSON object in Allrun preflight response")
            obj = json.loads(body[s:e + 1])
            verdict = str(obj.get("verdict", "")).upper().strip() or "OK"
            corrected = str(obj.get("corrected_allrun", "") or "")
            reason = str(obj.get("reason", ""))[:600]

            applied = None
            if verdict == "BROKEN" and corrected.strip():
                # Save original and write the patched Allrun.
                try:
                    (case_dir / "Allrun.preflight_original").write_text(
                        original, encoding="utf-8")
                    allrun.write_text(corrected, encoding="utf-8")
                    try:
                        os.chmod(allrun, 0o755)
                    except Exception:
                        pass
                    applied = corrected
                    verdict = "PATCHED"
                except Exception as exc:
                    reason += f" | failed to write patched Allrun: {exc}"
            return {
                "checked": True,
                "verdict": verdict if verdict in ("OK", "PATCHED", "BROKEN") else "OK",
                "method": "llm",
                "rationale": reason or "LLM audit complete.",
                "original_allrun": original,
                "applied_allrun": applied,
                "elapsed_s": time.time() - t0,
            }
        except TimeoutError as exc:
            # Hard 10-min timeout from _llm_invoke. Use heuristic fallback.
            ok, why = _heuristic_allrun_check(original, application)
            verdict = "UNCERTAIN_HEURISTIC_OK" if ok else "BROKEN"
            return {
                "checked": True, "verdict": verdict,
                "method": "heuristic_fallback",
                "failure_mode": "timeout",
                "rationale": f"LLM timed out ({exc}); heuristic: {why}",
                "original_allrun": original, "applied_allrun": None,
                "elapsed_s": time.time() - t0,
            }
        except Exception as exc:
            # Fall through to heuristic below.
            llm_err = f"{exc.__class__.__name__}: {exc}"
            ok, why = _heuristic_allrun_check(original, application)
            verdict = "UNCERTAIN_HEURISTIC_OK" if ok else "BROKEN"
            return {
                "checked": True, "verdict": verdict,
                "method": "heuristic_fallback",
                "rationale": f"LLM unavailable ({llm_err}); heuristic: {why}",
                "original_allrun": original, "applied_allrun": None,
                "elapsed_s": time.time() - t0,
            }

    # LLM disabled — heuristic only.
    ok, why = _heuristic_allrun_check(original, application)
    verdict = "UNCERTAIN_HEURISTIC_OK" if ok else "BROKEN"
    return {
        "checked": True, "verdict": verdict, "method": "heuristic_fallback",
        "rationale": why, "original_allrun": original, "applied_allrun": None,
        "elapsed_s": time.time() - t0,
    }


# Default prompts (used if prompts.yaml lacks RunValidityAgent block).
_DEFAULT_ALLRUN_SYSTEM_PROMPT = (
    "You are an OpenFOAM Allrun auditor. Given Allrun contents, decide if it "
    "executes a flow solver to completion. Return strict JSON: "
    "{verdict: 'OK'|'BROKEN', flow_solver_runs: bool, reason: '<short>', "
    "corrected_allrun: '<full Allrun text or empty>'}. "
    "Preserve all existing pre-solver setup commands; only fix solver-invocation "
    "lines. Accept multi-solver Allruns (potentialFoam + simpleFoam, etc.) as "
    "long as a flow solver runs. Accept both `runApplication simpleFoam` and "
    "`runParallel simpleFoam` forms. If the only solver invocation is commented "
    "out, that is BROKEN. Output JSON only, no prose."
)

_DEFAULT_ALLRUN_USER_PROMPT = (
    "Allrun contents:\n```\n{allrun_text}\n```\n\n"
    "controlDict.application = {application}\n\n"
    "case_dir listing (compact):\n{case_listing}\n"
)


# ---------------------------------------------------------------------------
# Investigator (used by INVESTIGATE_RUNTIME action)
# ---------------------------------------------------------------------------

_DEFAULT_INVESTIGATE_SYSTEM_PROMPT = (
    "You are an OpenFOAM run-validity investigator. A previous OED iteration "
    "was flagged RUN_INVALID by the run-validity gate (the flow solver did not "
    "advance the case to a meaningful time). Read the diagnostic bundle, the "
    "Allrun, the OpenFOAM log tails and the original action JSON. Classify "
    "the root cause and decide whether the fix is in the harness "
    "(Allrun / runner / preflight bug) or in the model (the proposed model "
    "modification is wrong, e.g. divergence, blow-up, BC bug). Output STRICT "
    "JSON only:\n"
    "{\n"
    "  \"root_cause_class\": \"code_mod_source_bug\"|\"allrun_bug\"|\"of_version\""
    "|\"oom\"|\"divergence\"|\"mesh\"|\"bc\"|\"other\",\n"
    "  \"explanation\": \"<short>\",\n"
    "  \"patch_target\": \"harness\"|\"model\",\n"
    "  \"patch\": {\n"
    "    \"files\": [{\"path\": \"Allrun\", \"new_content\": \"<full file>\", \"rationale\": \"...\"}],\n"
    "    \"rerun_strategy\": \"rerun_same_model\"|\"downgrade_to_revise\"\n"
    "  },\n"
    "  \"confidence\": <float 0..1>\n"
    "}\n"
    "If patch_target=='model', `patch.files` may be empty and rerun_strategy "
    "should be 'downgrade_to_revise'. If patch_target=='harness', provide the "
    "FULL replacement contents for each file (no diffs)."
)


_DEFAULT_INVESTIGATE_USER_PROMPT = (
    "DIAGNOSTIC BUNDLE (run_validity_diagnostic.json):\n{diag_json}\n\n"
    "ALLRUN CONTENTS:\n```\n{allrun_text}\n```\n\n"
    "ORIGINAL ACTION JSON:\n```\n{action_json}\n```\n\n"
    "LOG TAILS (last lines of any log.* file):\n{log_tails}\n"
)


def investigate_runtime_llm(
    *,
    diagnostic_bundle: Dict[str, Any],
    allrun_text: str,
    action_json: Dict[str, Any],
    log_tails: Dict[str, str],
) -> Dict[str, Any]:
    """LLM root-cause classification. Returns dict matching the plan's JSON shape.
    Falls back to a degraded {root_cause_class: 'other', patch_target: 'model',
    rerun_strategy: 'downgrade_to_revise'} on failure.
    """
    prompts = _load_prompts() or {}
    block = prompts.get("RunValidityAgent", {}) or {}
    sys_msg = block.get("investigate_runtime_system_prompt",
                       _DEFAULT_INVESTIGATE_SYSTEM_PROMPT)
    user_tmpl = block.get("investigate_runtime_user_prompt",
                         _DEFAULT_INVESTIGATE_USER_PROMPT)
    log_tails_text = ""
    for k, v in (log_tails or {}).items():
        log_tails_text += f"\n--- {k} (tail) ---\n{v[-2000:]}"
    try:
        user_msg = user_tmpl.format(
            diag_json=json.dumps(diagnostic_bundle, default=str)[:6000],
            allrun_text=(allrun_text or "")[:8000],
            action_json=json.dumps(action_json, default=str)[:6000],
            log_tails=log_tails_text[:6000],
        )
        raw = _llm_invoke([("system", sys_msg), ("user", user_msg)], temperature=0.0)
        body = _strip_code_fences(raw)
        s, e = body.find("{"), body.rfind("}")
        if s < 0 or e < 0:
            raise ValueError("no JSON object in investigation response")
        obj = json.loads(body[s:e + 1])
        if not isinstance(obj, dict):
            raise ValueError("investigation response not a dict")
        # Normalize.
        obj.setdefault("root_cause_class", "other")
        obj.setdefault("explanation", "")
        obj.setdefault("patch_target", "model")
        obj.setdefault("patch", {"files": [], "rerun_strategy": "downgrade_to_revise"})
        obj.setdefault("confidence", 0.0)
        return obj
    except TimeoutError as exc:
        return {
            "root_cause_class": "other",
            "explanation": f"investigator LLM timed out: {exc}",
            "patch_target": "model",
            "patch": {"files": [], "rerun_strategy": "downgrade_to_revise"},
            "confidence": 0.0,
            "failure_mode": "timeout",
        }
    except Exception as exc:
        return {
            "root_cause_class": "other",
            "explanation": f"investigator LLM failed: {exc}",
            "patch_target": "model",
            "patch": {"files": [], "rerun_strategy": "downgrade_to_revise"},
            "confidence": 0.0,
        }


__all__ = [
    "detect_max_time",
    "detect_baseline_final_time",
    "gate",
    "preflight_allrun",
    "investigate_runtime_llm",
]
