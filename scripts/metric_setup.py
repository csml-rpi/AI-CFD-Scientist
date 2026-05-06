"""Unified metric-setup stage.

Runs after baseline_setup and before open_ended_discovery. Decides metrics,
authors ONE multi-metric comparator script, validates the values via an
independent verifier agent, runs a recipe-sensitivity probe, and pins the
result for the OED loop. See CLAUDE.md and the design notes that introduced
this stage for the full contract.

CLI:
    python scripts/metric_setup.py
        --run-dir <run_dir>
        --topic "<topic>"
        --starter-dir <starter>
        --baseline-case-dir <baseline_case>
        --baseline-metrics <run_dir/baseline_metrics.json>
        --reference-data-manifest <run_dir/reference_data_manifest.json>
        --output <run_dir/metric_specs.json>
        --comparator-out <run_dir/comparators/>
        --timeline <run_dir/timeline.json>

Both the setup agent and the verifier agent share the python_script tool
sandbox in scripts.oed_extensions._run_verifier_script_sandboxed. LLM calls go
through scripts.oed_extensions._llm_invoke (10-min hard timeout).

Topic-agnostic. No flow-specific terms in the script body or prompts.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Make sibling scripts importable.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from oed_extensions import (  # type: ignore  # noqa: E402
    _enumerate_data_sources,
    _format_inventory_block,
    _llm_invoke,
    _parse_verifier_response,
    _run_verifier_script_sandboxed,
    _strip_code_fences,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_SETUP_MAX_TURNS = 15
_VERIFIER_MAX_TURNS = 8
_DISAGREEMENT_RETRIES = 2
_AGREE_REL_TOL = 0.20  # 20% relative tolerance for AGREE
_RECIPE_DRIFT_WARN = 0.15  # 15% drift => warn

_EXEMPLAR_BUDGET_BYTES = 6 * 1024


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def _read_json(p: Path, default: Any) -> Any:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _append_timeline(timeline_path: Optional[Path], event: Dict[str, Any]) -> None:
    if not timeline_path:
        return
    try:
        timeline_path.parent.mkdir(parents=True, exist_ok=True)
        existing = []
        if timeline_path.exists():
            try:
                existing = json.loads(timeline_path.read_text(encoding="utf-8"))
                if not isinstance(existing, list):
                    existing = []
            except Exception:
                existing = []
        event = {**event, "ts": time.time()}
        existing.append(event)
        timeline_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    except Exception as exc:
        print(f"[METRIC-SETUP] timeline append failed: {exc}")


def _load_prompts() -> Dict[str, Any]:
    """Load prompts.yaml. Returns {} on failure."""
    try:
        from cfd_langgraph.config import get_settings  # type: ignore
        from cfd_langgraph.ideation import load_prompts  # type: ignore
        return load_prompts(get_settings().prompts_path) or {}
    except Exception:
        # Fallback: read prompts.yaml directly.
        try:
            import yaml  # type: ignore
            cand = _THIS_DIR.parent / "prompts" / "prompts.yaml"
            if cand.is_file():
                return yaml.safe_load(cand.read_text(encoding="utf-8")) or {}
        except Exception:
            pass
    return {}


# ---------------------------------------------------------------------------
# Inventory / context blocks
# ---------------------------------------------------------------------------

def _detect_exemplar_scripts(starter_dir: Optional[Path]) -> List[Path]:
    """Find candidate exemplar comparator scripts in starter (compare_*.py /
    *_compare.py). Topic-agnostic by pattern match."""
    if not starter_dir or not Path(starter_dir).is_dir():
        return []
    out: List[Path] = []
    for p in Path(starter_dir).rglob("*.py"):
        n = p.name
        if n.startswith("compare_") or n.endswith("_compare.py") or "compare" in n:
            out.append(p)
    return out


def _read_exemplar_block(exemplars: List[Path]) -> str:
    if not exemplars:
        return "<none>"
    chosen = exemplars[0]
    try:
        text = chosen.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return f"<exemplar at {chosen} unreadable>"
    if len(text.encode("utf-8")) > _EXEMPLAR_BUDGET_BYTES:
        text = text.encode("utf-8")[:_EXEMPLAR_BUDGET_BYTES].decode("utf-8", errors="ignore")
        text += "\n... <truncated>\n"
    return f"# {chosen}\n{text}"


def _read_reference_block(ref_paths: List[Path], head_lines: int = 12) -> str:
    if not ref_paths:
        return "<no reference files>"
    out_parts: List[str] = []
    for rp in ref_paths[:6]:
        rp = Path(rp)
        if not rp.is_file():
            continue
        try:
            txt = rp.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        head = "\n".join(txt.splitlines()[:head_lines])
        out_parts.append(f"--- {rp} ---\n{head}")
    return "\n\n".join(out_parts) if out_parts else "<no readable reference files>"


def _resolve_reference_paths(manifest_path: Optional[Path], starter_dir: Optional[Path]) -> List[Path]:
    out: List[Path] = []
    if manifest_path and manifest_path.is_file():
        try:
            mf = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(mf, dict):
                cand = mf.get("reference_files") or mf.get("files") or []
                if isinstance(cand, list):
                    for c in cand:
                        if isinstance(c, str) and Path(c).is_file():
                            out.append(Path(c))
                        elif isinstance(c, dict):
                            p = c.get("path") or c.get("file")
                            if isinstance(p, str) and Path(p).is_file():
                                out.append(Path(p))
            elif isinstance(mf, list):
                for c in mf:
                    if isinstance(c, str) and Path(c).is_file():
                        out.append(Path(c))
        except Exception:
            pass
    if not out and starter_dir and Path(starter_dir).is_dir():
        # Best-effort: any csv/dat/txt at depth <= 2 in starter
        sd = Path(starter_dir)
        root_parts = len(sd.resolve().parts)
        for f in sorted(sd.rglob("*")):
            try:
                if not f.is_file():
                    continue
                if f.suffix.lower() not in (".csv", ".dat", ".txt"):
                    continue
                depth = len(f.resolve().parts) - root_parts
                if depth > 3:
                    continue
                out.append(f)
            except Exception:
                continue
    # Dedup
    seen = set()
    uniq: List[Path] = []
    for p in out:
        s = str(p.resolve())
        if s in seen:
            continue
        seen.add(s)
        uniq.append(p)
    return uniq


# ---------------------------------------------------------------------------
# Final-JSON parsing
# ---------------------------------------------------------------------------

def _parse_final_json(raw: str) -> Tuple[str, Dict[str, Any]]:
    """Parse one turn from the LLM. Returns (kind, payload).

    kind in {"tool", "final", "unknown"}.
      - "tool":   payload = {"code": str}
      - "final":  payload = full parsed JSON dict
      - "unknown": payload = {"raw": ...}
    """
    body = _strip_code_fences(raw or "")
    i = body.find("{")
    if i < 0:
        return "unknown", {"raw": body}
    depth = 0
    end = -1
    in_str = False
    esc = False
    for j in range(i, len(body)):
        ch = body[j]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = j + 1
                    break
    if end < 0:
        return "unknown", {"raw": body}
    snippet = body[i:end]
    # LLMs frequently emit literal control characters (newlines, tabs) inside
    # JSON string values when the value contains code. Strict json.loads
    # rejects those. Use the relaxed decoder first (strict=False allows
    # control chars in strings); if even that fails, escape stray control
    # chars inside string spans and retry. Only on the third failure do we
    # report unknown.
    obj = None
    try:
        obj = json.JSONDecoder(strict=False).decode(snippet)
    except Exception:
        try:
            # Walk the snippet, tracking whether we're inside a JSON string
            # span (respecting backslash escapes), and replace raw control
            # chars (LF, CR, TAB) inside strings with their JSON-escaped form.
            out_chars: List[str] = []
            in_str_local = False
            esc_local = False
            for ch in snippet:
                if in_str_local:
                    if esc_local:
                        out_chars.append(ch)
                        esc_local = False
                    elif ch == "\\":
                        out_chars.append(ch)
                        esc_local = True
                    elif ch == '"':
                        out_chars.append(ch)
                        in_str_local = False
                    elif ch == "\n":
                        out_chars.append("\\n")
                    elif ch == "\r":
                        out_chars.append("\\r")
                    elif ch == "\t":
                        out_chars.append("\\t")
                    elif ord(ch) < 0x20:
                        out_chars.append(f"\\u{ord(ch):04x}")
                    else:
                        out_chars.append(ch)
                else:
                    out_chars.append(ch)
                    if ch == '"':
                        in_str_local = True
            repaired = "".join(out_chars)
            obj = json.JSONDecoder(strict=False).decode(repaired)
        except Exception:
            return "unknown", {"raw": snippet}
    if isinstance(obj, dict) and obj.get("tool") == "python_script" and isinstance(obj.get("code"), str):
        return "tool", {"code": obj["code"]}
    if isinstance(obj, dict):
        return "final", obj
    return "unknown", {"raw": obj}


# ---------------------------------------------------------------------------
# Setup agent loop
# ---------------------------------------------------------------------------

def _format_turn_history(history: List[Dict[str, Any]]) -> str:
    if not history:
        return "<none>"
    lines: List[str] = []
    for i, t in enumerate(history, 1):
        code_snippet = (t.get("code", "") or "")[:600]
        out_snippet = (t.get("output", "") or "")[:1200]
        lines.append(f"--- turn {i} ---\nCODE:\n{code_snippet}\nOUTPUT:\n{out_snippet}")
    return "\n".join(lines)


def _run_setup_agent(
    *,
    topic: str,
    inventory_block: str,
    baseline_case_dir: str,
    reference_block: str,
    exemplar_block: str,
    baseline_metrics_path: str,
    baseline_metrics_block: str,
    comparator_out_path: str,
    extra_corrective: str = "",
    transcript_sink: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """Drive the setup agent. Returns (final_obj | None, turn_history)."""
    prompts = _load_prompts()
    ms = (prompts.get("MetricSetupAgent") or {})
    sys_tmpl = ms.get("metric_setup_system_prompt") or ""
    user_tmpl = ms.get("metric_setup_user_prompt") or ""
    history: List[Dict[str, Any]] = []
    final_obj: Optional[Dict[str, Any]] = None

    base_user_kwargs = {
        "topic": topic,
        "baseline_case_dir": baseline_case_dir,
        "comparator_out_path": comparator_out_path,
        "baseline_metrics_path": baseline_metrics_path,
        "inventory_block": inventory_block,
        "reference_block": reference_block,
        "exemplar_block": exemplar_block,
        "baseline_metrics_block": baseline_metrics_block,
    }

    print(f"[METRIC-SETUP][setup_agent] sys_tmpl_len={len(sys_tmpl)} user_tmpl_len={len(user_tmpl)} max_turns={_SETUP_MAX_TURNS}", file=sys.stderr, flush=True)
    for turn in range(1, _SETUP_MAX_TURNS + 1):
        try:
            user_msg = user_tmpl.format(
                turn_history=_format_turn_history(history),
                **base_user_kwargs,
            )
        except KeyError as kex:
            # Older prompts.yaml lacking some keys: progressively drop unknowns.
            print(f"[METRIC-SETUP][setup_agent] turn {turn} user_tmpl.format KeyError: {kex} -- using raw template", file=sys.stderr, flush=True)
            user_msg = user_tmpl
        except Exception as fex:
            print(f"[METRIC-SETUP][setup_agent] turn {turn} user_tmpl.format FAILED: {type(fex).__name__}: {fex}", file=sys.stderr, flush=True)
            import traceback; traceback.print_exc(file=sys.stderr)
            rec = {"turn": turn, "kind": "format_error", "error": f"{type(fex).__name__}: {fex}"}
            if transcript_sink is not None:
                transcript_sink.append(rec)
            break
        if extra_corrective and turn == 1:
            user_msg += "\n\n=== CORRECTIVE FEEDBACK FROM VERIFIER ===\n" + extra_corrective

        print(f"[METRIC-SETUP][setup_agent] turn {turn} -> _llm_invoke (user_msg_len={len(user_msg)})", file=sys.stderr, flush=True)
        t0 = time.time()
        try:
            raw = _llm_invoke([("system", sys_tmpl), ("user", user_msg)], temperature=0.0)
            print(f"[METRIC-SETUP][setup_agent] turn {turn} OK in {time.time()-t0:.1f}s, raw_len={len(raw)}, head={raw[:200]!r}", file=sys.stderr, flush=True)
        except TimeoutError as exc:
            dt = time.time() - t0
            print(f"[METRIC-SETUP][setup_agent] turn {turn} TIMEOUT in {dt:.1f}s: {exc}", file=sys.stderr, flush=True)
            rec = {"turn": turn, "kind": "timeout", "error": str(exc), "elapsed_s": dt}
            if transcript_sink is not None:
                transcript_sink.append(rec)
            break
        except Exception as exc:
            dt = time.time() - t0
            print(f"[METRIC-SETUP][setup_agent] turn {turn} EXCEPTION in {dt:.1f}s: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            import traceback; traceback.print_exc(file=sys.stderr)
            rec = {"turn": turn, "kind": "exception", "error": f"{type(exc).__name__}: {exc}", "elapsed_s": dt}
            if transcript_sink is not None:
                transcript_sink.append(rec)
            break

        kind, payload = _parse_final_json(raw)
        rec = {"turn": turn, "raw": raw[-4000:], "kind": kind}
        print(f"[METRIC-SETUP][setup_agent] turn {turn} parsed kind={kind}", file=sys.stderr, flush=True)
        if kind == "tool":
            code = payload.get("code", "")
            print(f"[METRIC-SETUP][setup_agent] turn {turn} tool call code_len={len(code)} -> running sandbox", file=sys.stderr, flush=True)
            stdout, stderr_out, rc = _run_verifier_script_sandboxed(code, timeout_s=60)
            print(f"[METRIC-SETUP][setup_agent] turn {turn} sandbox rc={rc} stdout_len={len(stdout or '')} stderr_len={len(stderr_out or '')}", file=sys.stderr, flush=True)
            output = (stdout or "") + (("\nSTDERR:\n" + stderr_out) if stderr_out else "")
            history.append({"code": code, "output": output, "rc": rc})
            rec.update({"code": code[:2000], "output": output[:4000], "rc": rc})
            if transcript_sink is not None:
                transcript_sink.append(rec)
            continue
        if kind == "final":
            print(f"[METRIC-SETUP][setup_agent] turn {turn} FINAL spec, n_metrics={len(payload.get('metrics', []) if isinstance(payload, dict) else [])}", file=sys.stderr, flush=True)
            final_obj = payload
            if transcript_sink is not None:
                transcript_sink.append(rec)
            break
        # unknown -> add empty turn and retry
        print(f"[METRIC-SETUP][setup_agent] turn {turn} UNPARSEABLE response head={str(payload)[:200]!r}", file=sys.stderr, flush=True)
        history.append({"code": "", "output": f"<unparseable response>: {str(payload)[:300]}", "rc": -1})
        rec.update({"output": "unparseable"})
        if transcript_sink is not None:
            transcript_sink.append(rec)

    print(f"[METRIC-SETUP][setup_agent] DONE final_obj={'<set>' if final_obj else 'None'} turns_used={len(transcript_sink) if transcript_sink is not None else len(history)}", file=sys.stderr, flush=True)
    return final_obj, history


# ---------------------------------------------------------------------------
# Verifier agent loop
# ---------------------------------------------------------------------------

def _run_verifier_agent(
    *,
    topic: str,
    inventory_block: str,
    reference_block: str,
    baseline_case_dir: str,
    metric_names_and_descriptions: str,
    setup_agent_values: str,
    transcript_sink: Optional[List[Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    """Drive the independent verifier. Returns the final verifier JSON or None."""
    prompts = _load_prompts()
    mv = (prompts.get("MetricSetupVerifier") or {})
    sys_tmpl = mv.get("metric_setup_verifier_system_prompt") or ""
    user_tmpl = mv.get("metric_setup_verifier_user_prompt") or ""
    history: List[Dict[str, Any]] = []
    final_obj: Optional[Dict[str, Any]] = None

    base_user_kwargs = {
        "topic": topic,
        "baseline_case_dir": baseline_case_dir,
        "inventory_block": inventory_block,
        "reference_block": reference_block,
        "metric_names_and_descriptions": metric_names_and_descriptions,
        "setup_agent_values": setup_agent_values,
    }

    print(f"[METRIC-SETUP][verifier_agent] sys_tmpl_len={len(sys_tmpl)} user_tmpl_len={len(user_tmpl)} max_turns={_VERIFIER_MAX_TURNS}", file=sys.stderr, flush=True)
    for turn in range(1, _VERIFIER_MAX_TURNS + 1):
        try:
            user_msg = user_tmpl.format(
                turn_history=_format_turn_history(history),
                **base_user_kwargs,
            )
        except KeyError as kex:
            print(f"[METRIC-SETUP][verifier_agent] turn {turn} user_tmpl.format KeyError: {kex}; using raw template", file=sys.stderr, flush=True)
            user_msg = user_tmpl

        print(f"[METRIC-SETUP][verifier_agent] turn {turn} -> _llm_invoke (user_msg_len={len(user_msg)})", file=sys.stderr, flush=True)
        t0 = time.time()
        try:
            raw = _llm_invoke([("system", sys_tmpl), ("user", user_msg)], temperature=0.0)
            print(f"[METRIC-SETUP][verifier_agent] turn {turn} OK in {time.time()-t0:.1f}s, raw_len={len(raw)}, head={raw[:200]!r}", file=sys.stderr, flush=True)
        except TimeoutError as exc:
            dt = time.time() - t0
            print(f"[METRIC-SETUP][verifier_agent] turn {turn} TIMEOUT in {dt:.1f}s: {exc}", file=sys.stderr, flush=True)
            rec = {"turn": turn, "kind": "timeout", "error": str(exc), "elapsed_s": dt}
            if transcript_sink is not None:
                transcript_sink.append(rec)
            break
        except Exception as exc:
            dt = time.time() - t0
            print(f"[METRIC-SETUP][verifier_agent] turn {turn} EXCEPTION in {dt:.1f}s: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            import traceback; traceback.print_exc(file=sys.stderr)
            rec = {"turn": turn, "kind": "exception", "error": f"{type(exc).__name__}: {exc}", "elapsed_s": dt}
            if transcript_sink is not None:
                transcript_sink.append(rec)
            break

        kind, payload = _parse_final_json(raw)
        rec = {"turn": turn, "raw": raw[-4000:], "kind": kind}
        print(f"[METRIC-SETUP][verifier_agent] turn {turn} parsed kind={kind}", file=sys.stderr, flush=True)
        if kind == "tool":
            code = payload.get("code", "")
            print(f"[METRIC-SETUP][verifier_agent] turn {turn} tool call code_len={len(code)} -> running sandbox", file=sys.stderr, flush=True)
            stdout, stderr_out, rc = _run_verifier_script_sandboxed(code, timeout_s=60)
            print(f"[METRIC-SETUP][verifier_agent] turn {turn} sandbox rc={rc} stdout_len={len(stdout or '')} stderr_len={len(stderr_out or '')}", file=sys.stderr, flush=True)
            output = (stdout or "") + (("\nSTDERR:\n" + stderr_out) if stderr_out else "")
            history.append({"code": code, "output": output, "rc": rc})
            rec.update({"code": code[:2000], "output": output[:4000], "rc": rc})
            if transcript_sink is not None:
                transcript_sink.append(rec)
            continue
        if kind == "final":
            print(f"[METRIC-SETUP][verifier_agent] turn {turn} FINAL verdict, has_keys={list(payload.keys()) if isinstance(payload, dict) else None}", file=sys.stderr, flush=True)
            final_obj = payload
            if transcript_sink is not None:
                transcript_sink.append(rec)
            break
        print(f"[METRIC-SETUP][verifier_agent] turn {turn} UNPARSEABLE response head={str(payload)[:200]!r}", file=sys.stderr, flush=True)
        history.append({"code": "", "output": f"<unparseable>: {str(payload)[:300]}", "rc": -1})
        rec.update({"output": "unparseable"})
        if transcript_sink is not None:
            transcript_sink.append(rec)

    print(f"[METRIC-SETUP][verifier_agent] DONE final_obj={'<set>' if final_obj else 'None'} turns_used={len(transcript_sink) if transcript_sink is not None else len(history)}", file=sys.stderr, flush=True)
    return final_obj


# ---------------------------------------------------------------------------
# Decision (AGREE / DISAGREE within tolerance)
# ---------------------------------------------------------------------------

def _values_agree(setup_val: Any, verifier_val: Any, rel_tol: float = _AGREE_REL_TOL) -> bool:
    try:
        a = float(setup_val)
        b = float(verifier_val)
    except (TypeError, ValueError):
        return False
    if a != a or b != b:  # NaN
        return False
    denom = max(abs(a), abs(b))
    if denom < 1e-12:
        return abs(a - b) < 1e-9
    return abs(a - b) / denom <= rel_tol


# ---------------------------------------------------------------------------
# Recipe-sensitivity probe (no LLM)
# ---------------------------------------------------------------------------

def _run_comparator(comparator_path: Path, case_dir: Path, reference_path: Optional[Path],
                    timeout_s: int = 90) -> Tuple[str, str, int]:
    cmd = [sys.executable, str(comparator_path), "--case", str(case_dir)]
    if reference_path is not None:
        cmd += ["--reference", str(reference_path)]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        return (res.stdout or ""), (res.stderr or ""), int(res.returncode)
    except Exception as exc:
        return "", f"comparator exec error: {exc}", -1


_METRIC_LINE_RE = re.compile(r"METRIC\s+([A-Za-z0-9_\-]+)\s*:\s*(-?\d+\.?\d*(?:[eE][+-]?\d+)?|nan)",
                             re.IGNORECASE)


def _extract_metric_values(blob: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for m in _METRIC_LINE_RE.finditer(blob):
        name = m.group(1)
        val_s = m.group(2).lower()
        if val_s == "nan":
            continue
        try:
            out[name] = float(val_s)
        except Exception:
            continue
    return out


# Topic-agnostic patterns we try to perturb in the comparator.
_SMOOTH_RE = re.compile(r"\b(smoothing[_\- ]?window|window[_\- ]?length|window[_\- ]?size|smooth[_\- ]?n)\s*=\s*(\d+)",
                        re.IGNORECASE)
_X_LO_RE = re.compile(r"\b(x[_\- ]?(?:lo|min|low|left|start))\s*=\s*(-?\d+\.?\d*)",
                      re.IGNORECASE)
_X_HI_RE = re.compile(r"\b(x[_\- ]?(?:hi|max|high|right|end))\s*=\s*(-?\d+\.?\d*)",
                      re.IGNORECASE)


def _perturb_source(src: str, kind: str) -> Optional[str]:
    """Topic-agnostic perturbations."""
    if kind == "smooth_plus2":
        def repl(m: re.Match) -> str:
            return f"{m.group(1)}={int(m.group(2)) + 2}"
        new, n = _SMOOTH_RE.subn(repl, src, count=1)
        return new if n else None
    if kind == "smooth_minus2":
        def repl(m: re.Match) -> str:
            return f"{m.group(1)}={max(1, int(m.group(2)) - 2)}"
        new, n = _SMOOTH_RE.subn(repl, src, count=1)
        return new if n else None
    if kind == "x_window_minus10":
        new = src
        n_total = 0
        def lo_repl(m: re.Match) -> str:
            return f"{m.group(1)}={float(m.group(2)) * 0.9:g}"
        def hi_repl(m: re.Match) -> str:
            return f"{m.group(1)}={float(m.group(2)) * 1.1:g}"
        new, n1 = _X_LO_RE.subn(lo_repl, new, count=1)
        new, n2 = _X_HI_RE.subn(hi_repl, new, count=1)
        n_total = n1 + n2
        return new if n_total else None
    if kind == "x_window_plus10":
        def lo_repl(m: re.Match) -> str:
            return f"{m.group(1)}={float(m.group(2)) * 1.1:g}"
        def hi_repl(m: re.Match) -> str:
            return f"{m.group(1)}={float(m.group(2)) * 0.9:g}"
        new = src
        new, n1 = _X_LO_RE.subn(lo_repl, new, count=1)
        new, n2 = _X_HI_RE.subn(hi_repl, new, count=1)
        return new if (n1 + n2) else None
    return None


def _recipe_sensitivity_probe(
    *,
    comparator_path: Path,
    case_dir: Path,
    reference_path: Optional[Path],
    metric_names: List[str],
    baseline_values: Dict[str, float],
) -> Dict[str, Dict[str, Any]]:
    """For each metric, perturb a few recipe parameters and rerun. Returns
    {metric_name: {"max_drift_pct": float, "stable": bool, "spread": [...]}}.
    """
    out: Dict[str, Dict[str, Any]] = {}
    if not comparator_path.is_file() or not case_dir.is_dir():
        for n in metric_names:
            out[n] = {"max_drift_pct": 0.0, "stable": True,
                      "note": "skipped: comparator or case missing"}
        return out
    try:
        src_orig = comparator_path.read_text(encoding="utf-8")
    except Exception as exc:
        for n in metric_names:
            out[n] = {"max_drift_pct": 0.0, "stable": True,
                      "note": f"skipped: cannot read comparator ({exc})"}
        return out

    spread: Dict[str, List[float]] = {n: [] for n in metric_names}
    perturb_kinds = ("smooth_plus2", "smooth_minus2", "x_window_plus10", "x_window_minus10")
    for kind in perturb_kinds:
        new_src = _perturb_source(src_orig, kind)
        if new_src is None or new_src == src_orig:
            continue
        with _temp_swap_file(comparator_path, new_src):
            stdout, _stderr, _rc = _run_comparator(comparator_path, case_dir, reference_path)
            vals = _extract_metric_values(stdout)
            for n in metric_names:
                if n in vals:
                    spread[n].append(vals[n])

    for n in metric_names:
        base = baseline_values.get(n)
        drifts: List[float] = []
        if base is not None:
            for v in spread[n]:
                denom = max(abs(base), abs(v))
                if denom > 1e-12:
                    drifts.append(abs(v - base) / denom * 100.0)
                else:
                    drifts.append(0.0)
        max_drift_pct = max(drifts) if drifts else 0.0
        out[n] = {
            "max_drift_pct": float(max_drift_pct),
            "stable": bool(max_drift_pct <= _RECIPE_DRIFT_WARN * 100.0),
            "spread": spread[n],
        }
    return out


class _temp_swap_file:
    """Context manager: swap a file's contents temporarily, restore on exit."""

    def __init__(self, path: Path, new_contents: str):
        self.path = path
        self.new_contents = new_contents
        self.original: Optional[str] = None

    def __enter__(self):
        try:
            self.original = self.path.read_text(encoding="utf-8")
        except Exception:
            self.original = None
        try:
            self.path.write_text(self.new_contents, encoding="utf-8")
        except Exception:
            pass
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.original is not None:
            try:
                self.path.write_text(self.original, encoding="utf-8")
            except Exception:
                pass
        return False


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _format_metric_names_descriptions(metrics: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for m in metrics:
        nm = str(m.get("name", "")).strip()
        ds = str(m.get("description", "")).strip()
        src = str(m.get("data_source", "")).strip()
        if not nm:
            continue
        lines.append(f"- {nm}: {ds} (data_source: {src})")
    return "\n".join(lines) if lines else "<none>"


def _format_setup_values(metrics: List[Dict[str, Any]]) -> str:
    out: Dict[str, Any] = {}
    for m in metrics:
        nm = str(m.get("name", "")).strip()
        if not nm:
            continue
        out[nm] = m.get("baseline_value")
    return json.dumps(out, indent=2)


def _format_baseline_metrics_block(baseline_metrics_path: Optional[Path]) -> str:
    if not baseline_metrics_path or not baseline_metrics_path.is_file():
        return "<no baseline_metrics.json>"
    try:
        obj = json.loads(baseline_metrics_path.read_text(encoding="utf-8"))
        return json.dumps(obj, indent=2)[:2000]
    except Exception as exc:
        return f"<unreadable: {exc}>"


def run_metric_setup(args: argparse.Namespace) -> int:
    print("="*80, file=sys.stderr, flush=True)
    print("[METRIC-SETUP] ENTERING run_metric_setup", file=sys.stderr, flush=True)
    print(f"[METRIC-SETUP] python={sys.executable}", file=sys.stderr, flush=True)
    print(f"[METRIC-SETUP] CFD_SCIENTIST_LLM_PROVIDER={os.environ.get('CFD_SCIENTIST_LLM_PROVIDER','(unset)')}", file=sys.stderr, flush=True)
    print(f"[METRIC-SETUP] CFD_SCIENTIST_MODEL={os.environ.get('CFD_SCIENTIST_MODEL','(unset)')}", file=sys.stderr, flush=True)
    print(f"[METRIC-SETUP] ANTHROPIC_API_KEY set? {bool(os.environ.get('ANTHROPIC_API_KEY'))}", file=sys.stderr, flush=True)
    print(f"[METRIC-SETUP] args={vars(args)}", file=sys.stderr, flush=True)
    # Verify cfd_langgraph is loading from the expected repo
    try:
        import cfd_langgraph.llm.factory as _fac_check
        print(f"[METRIC-SETUP] cfd_langgraph.llm.factory loaded from: {_fac_check.__file__}", file=sys.stderr, flush=True)
        print(f"[METRIC-SETUP] factory has _create_claude_code_chat_model: {hasattr(_fac_check, '_create_claude_code_chat_model')}", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"[METRIC-SETUP] FAILED to import cfd_langgraph.llm.factory: {e}", file=sys.stderr, flush=True)
    print("="*80, file=sys.stderr, flush=True)

    run_dir = Path(args.run_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    starter_dir = Path(args.starter_dir).expanduser().resolve() if args.starter_dir else None
    baseline_case_dir = Path(args.baseline_case_dir).expanduser().resolve() if args.baseline_case_dir else None
    baseline_metrics_path = Path(args.baseline_metrics).expanduser().resolve() if args.baseline_metrics else None
    reference_manifest_path = Path(args.reference_data_manifest).expanduser().resolve() if args.reference_data_manifest else None
    output_path = Path(args.output).expanduser().resolve()
    comparator_out_dir = Path(args.comparator_out).expanduser().resolve()
    comparator_out_dir.mkdir(parents=True, exist_ok=True)
    timeline_path = Path(args.timeline).expanduser().resolve() if args.timeline else None
    diagnostic_path = run_dir / "metric_setup_diagnostic.json"

    print(f"[METRIC-SETUP] run_dir={run_dir}", file=sys.stderr, flush=True)
    print(f"[METRIC-SETUP] baseline_case_dir={baseline_case_dir} exists={baseline_case_dir.is_dir() if baseline_case_dir else None}", file=sys.stderr, flush=True)
    print(f"[METRIC-SETUP] starter_dir={starter_dir} exists={starter_dir.is_dir() if starter_dir else None}", file=sys.stderr, flush=True)
    print(f"[METRIC-SETUP] reference_manifest={reference_manifest_path} exists={reference_manifest_path.is_file() if reference_manifest_path else None}", file=sys.stderr, flush=True)
    print(f"[METRIC-SETUP] diagnostic_path={diagnostic_path}", file=sys.stderr, flush=True)

    _append_timeline(timeline_path, {"stage": "metric_setup", "event": "starting"})

    # Stage A: inventory
    ref_paths = _resolve_reference_paths(reference_manifest_path, starter_dir)
    ref_search_paths: List[Path] = []
    if starter_dir and starter_dir.is_dir():
        ref_search_paths.append(starter_dir)
    for rp in ref_paths:
        parent = Path(rp).parent
        if parent.is_dir() and parent not in ref_search_paths:
            ref_search_paths.append(parent)

    print(f"[METRIC-SETUP] Stage A: building inventory ...", file=sys.stderr, flush=True)
    t_invA = time.time()
    inventory = _enumerate_data_sources(
        baseline_case_dir=baseline_case_dir or Path("/nonexistent"),
        reference_search_paths=ref_search_paths,
        head_lines=8,
    )
    print(f"[METRIC-SETUP] Stage A: inventory built in {time.time()-t_invA:.2f}s n_paths={len(inventory.get('all_paths', []))}", file=sys.stderr, flush=True)
    inventory_block = _format_inventory_block(inventory) or "<empty inventory>"
    print(f"[METRIC-SETUP] Stage A: inventory_block_len={len(inventory_block)}", file=sys.stderr, flush=True)
    reference_block = _read_reference_block(ref_paths)
    print(f"[METRIC-SETUP] Stage A: reference_block_len={len(reference_block)}", file=sys.stderr, flush=True)
    exemplars = _detect_exemplar_scripts(starter_dir)
    print(f"[METRIC-SETUP] Stage A: exemplars detected={[str(p) for p in exemplars]}", file=sys.stderr, flush=True)
    exemplar_block = _read_exemplar_block(exemplars)
    print(f"[METRIC-SETUP] Stage A: exemplar_block_len={len(exemplar_block)}", file=sys.stderr, flush=True)
    baseline_metrics_block = _format_baseline_metrics_block(baseline_metrics_path)
    print(f"[METRIC-SETUP] Stage A: baseline_metrics_block_len={len(baseline_metrics_block)}", file=sys.stderr, flush=True)

    # Verify prompts loaded
    _prompts_check = _load_prompts()
    _ms_block = _prompts_check.get('MetricSetupAgent') or {}
    print(f"[METRIC-SETUP] Prompts: MetricSetupAgent system_len={len(_ms_block.get('metric_setup_system_prompt') or '')} user_len={len(_ms_block.get('metric_setup_user_prompt') or '')}", file=sys.stderr, flush=True)
    _mv_block = _prompts_check.get('MetricSetupVerifier') or {}
    print(f"[METRIC-SETUP] Prompts: MetricSetupVerifier system_len={len(_mv_block.get('metric_setup_verifier_system_prompt') or '')} user_len={len(_mv_block.get('metric_setup_verifier_user_prompt') or '')}", file=sys.stderr, flush=True)

    comparator_script_path = comparator_out_dir / "compute_metrics.py"

    diagnostic: Dict[str, Any] = {
        "topic": args.topic,
        "starter_dir": str(starter_dir) if starter_dir else "",
        "baseline_case_dir": str(baseline_case_dir) if baseline_case_dir else "",
        "comparator_script_path": str(comparator_script_path),
        "exemplars_detected": [str(p) for p in exemplars],
        "reference_files": [str(p) for p in ref_paths],
        "rounds": [],
        "decisions": {},
        "recipe_sensitivity": {},
    }
    # Persist diagnostic incrementally so post-mortem visibility never depends
    # on completing the run.
    _write_json(diagnostic_path, diagnostic)
    print(f"[METRIC-SETUP] initial diagnostic written to {diagnostic_path}", file=sys.stderr, flush=True)

    # Stage B / C / D: setup -> verifier -> retry on disagreement
    final_setup: Optional[Dict[str, Any]] = None
    final_verifier: Optional[Dict[str, Any]] = None
    extra_corrective = ""
    for round_idx in range(_DISAGREEMENT_RETRIES + 1):
        round_rec: Dict[str, Any] = {"round": round_idx, "setup_turns": [], "verifier_turns": []}
        # Pre-pend the empty round_rec into diagnostic and persist, so even if
        # the setup agent crashes mid-loop, the post-mortem file shows we got
        # to this round.
        diagnostic["rounds"].append(round_rec)
        _write_json(diagnostic_path, diagnostic)
        print(f"[METRIC-SETUP] round {round_idx}: invoking setup agent", file=sys.stderr, flush=True)
        try:
            setup_obj, _setup_hist = _run_setup_agent(
                topic=args.topic,
                inventory_block=inventory_block,
                baseline_case_dir=str(baseline_case_dir) if baseline_case_dir else "<none>",
                reference_block=reference_block,
                exemplar_block=exemplar_block,
                baseline_metrics_path=str(baseline_metrics_path) if baseline_metrics_path else "<none>",
                baseline_metrics_block=baseline_metrics_block,
                comparator_out_path=str(comparator_script_path),
                extra_corrective=extra_corrective,
                transcript_sink=round_rec["setup_turns"],
            )
        except Exception as exc:
            print(f"[METRIC-SETUP] round {round_idx}: setup_agent raised UNCAUGHT {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            import traceback; traceback.print_exc(file=sys.stderr)
            round_rec["setup_uncaught_error"] = f"{type(exc).__name__}: {exc}"
            _write_json(diagnostic_path, diagnostic)
            _append_timeline(timeline_path, {"stage": "metric_setup", "event": "setup_agent_uncaught", "round": round_idx, "error": str(exc)[:500]})
            return 3
        # Persist after setup_agent call regardless of outcome.
        _write_json(diagnostic_path, diagnostic)
        if not (isinstance(setup_obj, dict) and isinstance(setup_obj.get("metrics"), list) and setup_obj["metrics"]):
            print(f"[METRIC-SETUP] round {round_idx}: setup agent did not return a usable spec; aborting. setup_obj={str(setup_obj)[:300]!r}", file=sys.stderr, flush=True)
            _write_json(diagnostic_path, diagnostic)
            _append_timeline(timeline_path, {"stage": "metric_setup", "event": "setup_agent_failed",
                                              "round": round_idx})
            return 2

        final_setup = setup_obj
        round_rec["setup_final"] = setup_obj

        # Stage C: independent verifier
        metrics = setup_obj.get("metrics", [])
        names_desc = _format_metric_names_descriptions(metrics)
        setup_vals_blob = _format_setup_values(metrics)
        print(f"[METRIC-SETUP] round {round_idx}: invoking verifier agent")
        verifier_obj = _run_verifier_agent(
            topic=args.topic,
            inventory_block=inventory_block,
            reference_block=reference_block,
            baseline_case_dir=str(baseline_case_dir) if baseline_case_dir else "<none>",
            metric_names_and_descriptions=names_desc,
            setup_agent_values=setup_vals_blob,
            transcript_sink=round_rec["verifier_turns"],
        )
        round_rec["verifier_final"] = verifier_obj
        # round_rec was already appended at start of iteration; persist update.
        _write_json(diagnostic_path, diagnostic)
        final_verifier = verifier_obj or {}

        # Stage D: decision
        verdicts = (verifier_obj or {}).get("verdicts", {}) if isinstance(verifier_obj, dict) else {}
        verifier_values = (verifier_obj or {}).get("verifier_values", {}) if isinstance(verifier_obj, dict) else {}
        notes = (verifier_obj or {}).get("discrepancy_notes", {}) if isinstance(verifier_obj, dict) else {}

        disagreements: List[str] = []
        for m in metrics:
            nm = str(m.get("name", "")).strip()
            if not nm:
                continue
            sv = m.get("baseline_value")
            vv = verifier_values.get(nm)
            verdict_text = str(verdicts.get(nm, "")).upper().strip()
            if vv is not None:
                if _values_agree(sv, vv):
                    final_verdict = "AGREE"
                else:
                    final_verdict = "DISAGREE"
            elif verdict_text in ("AGREE", "DISAGREE"):
                final_verdict = verdict_text
            else:
                final_verdict = "DISAGREE"
            if final_verdict == "DISAGREE":
                disagreements.append(nm)

        if not disagreements:
            print(f"[METRIC-SETUP] round {round_idx}: all metrics AGREE")
            break

        if round_idx >= _DISAGREEMENT_RETRIES:
            print(f"[METRIC-SETUP] round {round_idx}: still disagreeing on {disagreements} — "
                  "binding setup agent's values as best-effort")
            break

        # Build corrective feedback for next round.
        bullets: List[str] = []
        for nm in disagreements:
            sv = next((m.get("baseline_value") for m in metrics if m.get("name") == nm), None)
            vv = verifier_values.get(nm)
            note = notes.get(nm, "")
            bullets.append(f"- {nm}: setup_value={sv}, verifier_value={vv}, note={note}")
        extra_corrective = (
            "The independent verifier disagreed with these metric values. "
            "Re-derive the recipe for the listed metrics and emit a fresh SETUP "
            "JSON + comparator script.\n" + "\n".join(bullets)
        )

    # final_setup is guaranteed non-None at this point
    assert final_setup is not None
    metrics_out: List[Dict[str, Any]] = []
    verifier_values = (final_verifier or {}).get("verifier_values", {}) if isinstance(final_verifier, dict) else {}
    verdicts = (final_verifier or {}).get("verdicts", {}) if isinstance(final_verifier, dict) else {}
    notes = (final_verifier or {}).get("discrepancy_notes", {}) if isinstance(final_verifier, dict) else {}

    binding_kind = "metric_setup"

    for m in final_setup.get("metrics", []):
        if not isinstance(m, dict):
            continue
        nm = str(m.get("name", "")).strip()
        if not nm:
            continue
        sv = m.get("baseline_value")
        vv = verifier_values.get(nm)
        if vv is not None and _values_agree(sv, vv):
            v_verdict = "AGREE"
        elif vv is None:
            v_verdict = "DISAGREE_BIND_BEST_EFFORT" if str(verdicts.get(nm, "")).upper() != "AGREE" else "AGREE"
        else:
            v_verdict = "DISAGREE_BIND_BEST_EFFORT"
        spec = {
            **m,
            "verifier_value": vv,
            "verifier_verdict": v_verdict,
            "verifier_note": notes.get(nm, ""),
            "comparator_script": str(comparator_script_path),
            "binding_kind": binding_kind,
        }
        metrics_out.append(spec)

    # Stage E: recipe-sensitivity probe (only if comparator script + baseline case exist).
    metric_names = [s["name"] for s in metrics_out]
    baseline_values = {s["name"]: s.get("baseline_value") for s in metrics_out
                       if isinstance(s.get("baseline_value"), (int, float))}
    sensitivity: Dict[str, Dict[str, Any]] = {}
    if comparator_script_path.is_file() and baseline_case_dir and baseline_case_dir.is_dir():
        ref_for_run = ref_paths[0] if ref_paths else None
        sensitivity = _recipe_sensitivity_probe(
            comparator_path=comparator_script_path,
            case_dir=baseline_case_dir,
            reference_path=ref_for_run,
            metric_names=metric_names,
            baseline_values={k: float(v) for k, v in baseline_values.items()},
        )
    else:
        for n in metric_names:
            sensitivity[n] = {"max_drift_pct": 0.0, "stable": True,
                              "note": "skipped: missing comparator or baseline case"}
    diagnostic["recipe_sensitivity"] = sensitivity
    for s in metrics_out:
        nm = s["name"]
        s["recipe_sensitivity"] = sensitivity.get(nm, {"max_drift_pct": 0.0, "stable": True})
        if s["recipe_sensitivity"].get("max_drift_pct", 0.0) > _RECIPE_DRIFT_WARN * 100.0:
            print(f"[METRIC-SETUP] WARNING metric {nm} is recipe-sensitive: "
                  f"{s['recipe_sensitivity']['max_drift_pct']:.2f}% drift — "
                  "interpret comparisons cautiously.")

    all_agree = all(s.get("verifier_verdict") == "AGREE" for s in metrics_out)

    final_doc = {
        "metrics": metrics_out,
        "comparator_script": str(comparator_script_path),
        "comparator_uses_exemplar": bool(final_setup.get("comparator_uses_exemplar", False)),
        "recipe_summary": str(final_setup.get("recipe_summary", "")),
        "all_verifier_agree": bool(all_agree),
    }
    _write_json(output_path, final_doc)
    diagnostic["final_doc"] = final_doc
    _write_json(diagnostic_path, diagnostic)

    _append_timeline(timeline_path, {
        "stage": "metric_setup",
        "event": "complete",
        "n_metrics": len(metrics_out),
        "all_verifier_agree": bool(all_agree),
    })

    print(f"[METRIC-SETUP] wrote {output_path} with {len(metrics_out)} metrics "
          f"(all_verifier_agree={all_agree}); comparator at {comparator_script_path}")
    return 0


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Unified metric proposer + comparator author + verifier stage.",
    )
    p.add_argument("--run-dir", required=True, help="Run directory (where metric_specs.json lives).")
    p.add_argument("--topic", required=True, help="Study topic / user request text.")
    p.add_argument("--starter-dir", default="", help="Path to the starter case bundle (optional).")
    p.add_argument("--baseline-case-dir", default="", help="Baseline OpenFOAM case dir.")
    p.add_argument("--baseline-metrics", default="", help="Path to baseline_metrics.json.")
    p.add_argument("--reference-data-manifest", default="",
                   help="Path to reference_data_manifest.json.")
    p.add_argument("--output", required=True,
                   help="Output path for metric_specs.json (final spec list).")
    p.add_argument("--comparator-out", required=True,
                   help="Directory where compute_metrics.py is written.")
    p.add_argument("--timeline", default="", help="Path to timeline.json (optional).")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    try:
        return run_metric_setup(args)
    except SystemExit:
        raise
    except BaseException as exc:
        print(f"[METRIC-SETUP] UNCAUGHT TOP-LEVEL {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        import traceback; traceback.print_exc(file=sys.stderr)
        return 99


if __name__ == "__main__":
    sys.exit(main())
