#!/usr/bin/env python3
"""Scoring setup for a study whose candidates are fitted models (study_mode=surrogate).

The solver path's setup proposes metrics from OpenFOAM output and authors
comparators that read fields off a mesh. A study whose candidates are fitted
models has neither: its candidates write prediction files, and the starter
ships its own scorer that turns prediction files into numbers. So setup here
does three things, and nothing about any one study is written into them:

  1. Ask the model which numbers the search is judged on — given the task brief
     and the scorer's own source — with exactly one primary.
  2. Author ONE thin adapter that puts the study's scorer behind the harness's
     comparator contract (``--case <dir> --reference <file>`` ->
     ``METRIC <name>: <value>``), so every downstream reader — the baseline
     measurement, oed_score_candidate, the archive — is unchanged.
  3. Self-test that adapter on the supplied baseline result, feeding each
     failure back, until every metric returns a finite number.

The adapter wraps the scorer rather than re-deriving the metric. A re-derived
metric is a second, independent answer to a question the study has already
answered, and every time this repository let two derivations of one quantity
coexist they disagreed — by 1%, by 3x, by a factor of 1276.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent
for _p in (str(_SCRIPTS_DIR), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

ADAPTER_FILENAME = "surrogate_adapter.py"
MAX_ADAPTER_ATTEMPTS = 4
_SCORER_SOURCE_CHARS = 14000
# The self-test is the first time the study's scorer runs at all, so nothing
# about its speed is known yet. 900 s was below what the ml4cfd_airfrans scorer
# needs for its one-seed baseline (~15.5 minutes), which would have failed every
# attempt at a correct wrapper. A wrapper that is wrong usually fails in seconds.
_SELFTEST_TIMEOUT_S = 7200


# --------------------------------------------------------------------------- #
# Shared helpers                                                               #
# --------------------------------------------------------------------------- #

def _read_text(path: Path, limit: int) -> str:
    try:
        text = Path(path).read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:
        return f"(could not read {path}: {exc})"
    if len(text) > limit:
        return text[:limit] + f"\n# ... (truncated at {limit} characters)"
    return text


def _result_listing(result_dir: Path, *, max_entries: int = 60, head_chars: int = 1500) -> str:
    """What a supplied result directory holds: files, sizes, and a peek at two."""
    lines: List[str] = []
    peeks: List[str] = []
    try:
        entries = sorted(p for p in Path(result_dir).rglob("*") if p.is_file())
    except Exception as exc:
        return f"(could not list {result_dir}: {exc})"
    for p in entries[:max_entries]:
        try:
            size = p.stat().st_size
        except OSError:
            size = 0
        lines.append(f"  {p.relative_to(result_dir)} ({size} bytes)")
        if len(peeks) < 2 and p.suffix.lower() in {".csv", ".json", ".txt", ".tsv"} and size > 0:
            peeks.append(f"--- head of {p.relative_to(result_dir)} ---\n{_read_text(p, head_chars)}")
    if len(entries) > max_entries:
        lines.append(f"  ... and {len(entries) - max_entries} more files")
    return "\n".join(lines + peeks) if lines else "(empty)"


def _json_object(text: str) -> Optional[Dict[str, Any]]:
    """The one JSON object a reply was asked to be. Outermost braces, parsed.

    Not a pattern match over the content: the model is asked for bare JSON, and
    taking the outermost object is the least interpretation that recovers it
    from a reply that added a stray word.
    """
    text = str(text or "").strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(text[start:end + 1])
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def _task_brief(starter_dir: Path) -> str:
    try:
        from study_mode import _task_text  # type: ignore

        return _task_text(Path(starter_dir))
    except Exception:
        return "(task brief unavailable)"


# --------------------------------------------------------------------------- #
# 1. Which metrics                                                             #
# --------------------------------------------------------------------------- #

_PROPOSE_SYSTEM = (
    "You decide how a data-modelling study will be scored. "
    "Answer with one JSON object and nothing else."
)

_PROPOSE_USER = """A study's candidates are models fitted to data. Each candidate writes \
prediction files, and the study's own scorer (source below) turns prediction files into \
numbers. Decide which numbers the search is judged on.

Rules:
1. Every metric must be computable from what the scorer below produces for a set of
   prediction files. Do not invent a quantity it cannot produce.
2. Mark exactly ONE metric as primary. The search optimises the primary, so it must measure
   progress toward the success criterion the brief states.
3. If success requires several quantities to meet their targets at the same time, the
   primary must be a single combined quantity that improves only as the joint criterion gets
   closer to being met — for example the worst (largest) ratio of each quantity to its
   target, which is below 1 exactly when every target is met. Define it precisely in
   computation_hint, including the target values it uses. Add each individual quantity as
   a secondary metric.
4. If the brief requires statistics over repeated runs or seeds, state in computation_hint
   how each metric aggregates over them, and what it does when only a single run is
   present — a supplied baseline result may contain just one.
5. Names: letters, digits and underscores only, starting with a letter.
6. direction is "min" if lower is better, "max" if higher is better.

STUDY TOPIC:
{topic}

TASK BRIEF FROM THE STARTER FOLDER:
{brief}

THE STUDY'S SCORER ({scorer_path}):
{scorer_source}

THE SUPPLIED BASELINE RESULT ({baseline_dir}):
{baseline_listing}

Reply exactly:
{{"metrics": [{{"name": "...", "description": "...", "direction": "min", "primary": true, "computation_hint": "..."}}]}}
"""


def _clean_specs(payload: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    raw = (payload or {}).get("metrics")
    if not isinstance(raw, list):
        return []
    specs: List[Dict[str, Any]] = []
    seen = set()
    for m in raw:
        if not isinstance(m, dict):
            continue
        name = str(m.get("name", "") or "").strip()
        # str.isidentifier is the whole rule: a metric name becomes a key in
        # JSON files and a token in a `METRIC <name>:` line, and an identifier
        # is safe in both.
        if not name or not name.isidentifier() or name in seen:
            continue
        direction = str(m.get("direction", "min") or "min").strip().lower()
        if direction not in {"min", "max"}:
            direction = "min"
        seen.add(name)
        specs.append({
            "name": name,
            "description": str(m.get("description", "") or "").strip(),
            "direction": direction,
            "primary": bool(m.get("primary")),
            "computation_hint": str(m.get("computation_hint", "") or "").strip(),
            "data_source": "study scorer (via surrogate adapter)",
            "preferred_method": "surrogate_adapter",
        })
    if not specs:
        return []
    primaries = [s for s in specs if s["primary"]]
    if not primaries:
        specs[0]["primary"] = True
    elif len(primaries) > 1:
        for s in primaries[1:]:
            s["primary"] = False
    # Primary first. Every reader that picks "the" metric walks the spec list
    # in order and takes the first finite value, so a primary that is not
    # first can be silently replaced by a secondary.
    specs.sort(key=lambda s: not s["primary"])
    return specs


def propose_surrogate_metrics(
    *, topic: str, starter_dir: Path, scorer_path: Path, baseline_dir: Path,
    timeout_s: int = 600,
) -> List[Dict[str, Any]]:
    from oed_extensions import _llm_invoke  # type: ignore

    prompt = _PROPOSE_USER.format(
        topic=str(topic or "").strip()[:8000],
        brief=_task_brief(starter_dir),
        scorer_path=scorer_path,
        scorer_source=_read_text(scorer_path, _SCORER_SOURCE_CHARS),
        baseline_dir=baseline_dir,
        baseline_listing=_result_listing(baseline_dir),
    )
    for attempt in (1, 2):
        try:
            raw = _llm_invoke([("system", _PROPOSE_SYSTEM), ("user", prompt)],
                              timeout_s=timeout_s)
        except Exception as exc:
            print(f"[OED-SURROGATE] metric proposal attempt {attempt} failed: {exc}", flush=True)
            continue
        specs = _clean_specs(_json_object(raw))
        if specs:
            print(f"[OED-SURROGATE] metrics: "
                  f"{[(s['name'], s['direction'], 'primary' if s['primary'] else '') for s in specs]}",
                  flush=True)
            return specs
        print(f"[OED-SURROGATE] metric proposal attempt {attempt} returned no usable metrics",
              flush=True)
    return []


# --------------------------------------------------------------------------- #
# 2 + 3. The adapter, authored and self-tested                                 #
# --------------------------------------------------------------------------- #

_ADAPTER_SYSTEM = (
    "You write small, robust Python glue scripts. Output ONLY raw Python code. "
    "No markdown, no code fences, no commentary."
)

_ADAPTER_USER = """Write a Python script that runs a study's own scorer on one result and \
reports its metrics in a fixed format. It is glue: the scorer computes the numbers, the \
script only runs it and reports what it produced. Do not re-implement the metric.

EXACT CONTRACT
  python <this script> --case <DIR> --reference <FILE> [--metric-name <NAME>] [--baseline-time <T>]
- <DIR> holds one result to score. Its prediction files are under <DIR>/predictions/
  (search recursively). If <DIR>/predictions/ does not exist, the prediction files are
  directly inside <DIR> — a supplied baseline result has that layout. Use every
  prediction file found under <DIR>/predictions/: the framework places exactly the files to
  be scored there and nothing else. A supplied baseline directory may also hold other
  files (a summary, notes); from it take only the files the scorer's input contract names.
  If the scorer accepts several prediction files (for example one per seed), pass them all
  in one invocation.
- --reference, --metric-name and --baseline-time must be accepted. Ignore any you do not
  need; the scorer knows its own truth data.
- Run the scorer at {scorer_path} with the interpreter {python_exe}, using the working
  directory and command-line arguments its source requires. Read its source below to get
  them right; do not guess.
- Never write anything inside the starter folder {starter_dir}. If the scorer writes output
  files, point them at a fresh tempfile.mkdtemp() directory. If the environment variable
  CFD_SCIENTIST_SCORER_OUTPUT_DIR is set, copy every structured output file the scorer
  wrote (its JSON summary, for example) into that directory before removing anything.
- Take the numbers from the scorer's structured output if it writes one (a JSON summary,
  say). Fall back to parsing its printed text only if it has no structured output.
- Print exactly one line per metric listed below, always all of them, even when
  --metric-name is given:
      METRIC <name>: <value>
  If a metric cannot be computed, print `METRIC <name>: nan` and a line
  `ADAPTER_WARNING: <why>`.
- Exit 0 if every metric was computed, 2 otherwise. On any exception, print
  `ADAPTER_WARNING: <exception>` and exit 2 — never crash silently.
- Use only the Python standard library plus numpy and pandas.

METRICS (compute exactly these, following each computation_hint):
{metrics_json}

STUDY TOPIC:
{topic}

THE STUDY'S SCORER — {scorer_path}:
{scorer_source}

THE BASELINE RESULT THE SCRIPT WILL BE SELF-TESTED ON — {baseline_dir}:
{baseline_listing}
{feedback}"""


def _feedback_block(prev_source: str, output_tail: str, per_metric: Dict[str, Tuple[bool, str, Any]]) -> str:
    verdicts = "\n".join(
        f"  {name}: {'ok' if ok else 'FAILED'} — {reason} (value {value})"
        for name, (ok, reason, value) in per_metric.items()
    )
    return (
        "\n\nYOUR PREVIOUS ATTEMPT FAILED ITS SELF-TEST ON THE BASELINE RESULT.\n"
        f"Per-metric verdicts:\n{verdicts}\n\n"
        f"What the script printed (tail):\n{output_tail[-4000:]}\n\n"
        f"The script that produced this:\n{prev_source[:12000]}\n\n"
        "Fix the cause the output shows. Return the complete corrected script."
    )


def _author_adapter(
    *, topic: str, specs: List[Dict[str, Any]], scorer_path: Path, starter_dir: Path,
    baseline_dir: Path, out_path: Path, python_exe: str, feedback: str, timeout_s: int = 600,
) -> Optional[Path]:
    from oed_extensions import _llm_invoke, _strip_code_fences  # type: ignore

    prompt = _ADAPTER_USER.format(
        scorer_path=scorer_path,
        python_exe=python_exe,
        starter_dir=starter_dir,
        metrics_json=json.dumps(
            [{k: s.get(k) for k in ("name", "direction", "primary", "computation_hint")} for s in specs],
            indent=2,
        ),
        topic=str(topic or "").strip()[:6000],
        scorer_source=_read_text(scorer_path, _SCORER_SOURCE_CHARS),
        baseline_dir=baseline_dir,
        baseline_listing=_result_listing(baseline_dir),
        feedback=feedback,
    )
    raw = _llm_invoke([("system", _ADAPTER_SYSTEM), ("user", prompt)], timeout_s=timeout_s)
    code = _strip_code_fences(raw)
    if not code or "import" not in code:
        return None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(code, encoding="utf-8")
    return out_path


def bind_surrogate_comparators(
    *, topic: str, specs: List[Dict[str, Any]], scorer_path: Path, starter_dir: Path,
    baseline_dir: Path, out_dir: Path, reference_file: Path,
    max_attempts: int = MAX_ADAPTER_ATTEMPTS,
) -> Dict[str, Dict[str, Any]]:
    """Author the adapter and self-test it on the baseline. Returns the binding.

    Every metric is bound to the same adapter script. A metric whose self-test
    did not pass is bound with selftest_ok=False, which compute_metric_vector
    already skips — the caller decides whether that is fatal.
    """
    from oed_extensions import (  # type: ignore
        _run_comparator_with_optional_baseline_time,
        judge_comparator_output,
    )

    adapter = Path(out_dir) / ADAPTER_FILENAME
    # Created up front, not by the first successful authoring call: when every
    # call fails, the attempt log below still needs somewhere to go. It used to
    # vanish silently, and malmo_qwen38max_openrouter_20260912 failed setup
    # three times with no record of why on disk.
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    names = [s["name"] for s in specs]
    per_metric: Dict[str, Tuple[bool, str, Any]] = {n: (False, "not attempted", None) for n in names}
    attempt_log: List[Dict[str, Any]] = []
    feedback = ""
    attempts_used = 0
    scorer_seconds: Optional[float] = None

    for attempt in range(1, max_attempts + 1):
        attempts_used = attempt
        try:
            path = _author_adapter(
                topic=topic, specs=specs, scorer_path=scorer_path, starter_dir=starter_dir,
                baseline_dir=baseline_dir, out_path=adapter, python_exe=sys.executable,
                feedback=feedback,
            )
        except Exception as exc:
            attempt_log.append({"attempt": attempt, "authoring_ok": False, "error": str(exc)})
            print(f"[OED-SURROGATE] adapter attempt {attempt}: authoring failed ({exc})", flush=True)
            continue
        if path is None:
            attempt_log.append({"attempt": attempt, "authoring_ok": False,
                                "error": "authoring returned no code"})
            feedback = "\n\nYour previous reply contained no Python code. Return only the script."
            continue

        # One run, judged per metric. The adapter prints every metric on every
        # run, so self-testing each metric separately (plus a re-run to show
        # the output) ran the study's scorer len(names)+1 times per attempt.
        # The run is timed; compute_metric_vector sets its scoring time limit
        # from that measurement.
        started = time.monotonic()
        try:
            res = _run_comparator_with_optional_baseline_time(
                comparator=path, case_dir=baseline_dir, reference_file=reference_file,
                baseline_time=None, timeout_s=_SELFTEST_TIMEOUT_S, metric_name=names[0],
            )
            run_error = ""
        except subprocess.TimeoutExpired:
            res, run_error = None, f"timeout after {_SELFTEST_TIMEOUT_S}s"
        except Exception as exc:
            res, run_error = None, f"exec error: {exc}"
        scorer_seconds = round(time.monotonic() - started, 1)
        for name in names:
            per_metric[name] = (
                judge_comparator_output(res, metric_name=name, baseline_time=None)
                if res is not None else (False, run_error, None)
            )
        all_ok = all(ok for ok, _, _ in per_metric.values())
        attempt_log.append({
            "attempt": attempt,
            "authoring_ok": True,
            "scorer_seconds": scorer_seconds,
            "selftest": {n: {"ok": ok, "reason": r, "value": v} for n, (ok, r, v) in per_metric.items()},
        })
        print(f"[OED-SURROGATE] adapter attempt {attempt} ({scorer_seconds}s): "
              + ", ".join(f"{n}={'ok ' + str(v) if ok else 'FAIL (' + r + ')'}"
                          for n, (ok, r, v) in per_metric.items()),
              flush=True)
        if all_ok:
            break
        output_tail = ((res.stdout or "") + "\n" + (res.stderr or "")) if res is not None else run_error
        feedback = _feedback_block(path.read_text(encoding="utf-8", errors="ignore"),
                                   output_tail, per_metric)

    try:
        (Path(out_dir) / "surrogate_adapter_attempt_log.json").write_text(
            json.dumps({"scorer": str(scorer_path), "baseline_dir": str(baseline_dir),
                        "attempts": attempt_log}, indent=2, default=str),
            encoding="utf-8",
        )
    except Exception:
        pass

    return {
        name: {
            "path": str(adapter) if adapter.is_file() else "",
            "origin": "surrogate_adapter",
            "scorer": str(scorer_path),
            "selftest_ok": bool(per_metric[name][0]),
            "selftest_value": per_metric[name][2],
            "selftest_reason": per_metric[name][1],
            "attempts": attempts_used,
            "final_method": "surrogate_adapter",
            "scorer_seconds": scorer_seconds,
        }
        for name in names
    }
