#!/usr/bin/env python3
"""Which execution path this study's candidates take.

The search is the same either way — same archive, same niche selection, same
allocator, same budget accounting — so this decides one thing only: what a
candidate *is*.

    solver          simulation cases are run: a simulation study, or a search
                    whose candidates modify a simulation code, built and run
                    against a case (scripts/code_mod_agentic.py)
    surrogate       a candidate is a model fitted to data the starter ships,
                    which produces predictions (scripts/surrogate_agentic.py)
    implementation  no search: one method the task fixes is written, built and
                    verified (scripts/implementation_agentic.py)

Decided by reading the starter folder, not by matching names. A folder is not
a solver study because it contains the word "Foam", and it is not a surrogate
study because it contains a .csv — plenty of solver studies ship reference
CSVs, and the question is what a candidate has to DO, which only the task
description and the shape of the folder answer. So the starter's own
description of itself goes to the model, and the answer is cached next to the
study's other classification caches so it is decided once per study.

A classification that cannot be made is an error, not a guess. A failed or
unreadable reply is retried; if no attempt gives an answer, StudyModeUnavailable
is raised and nothing is cached, so the step fails visibly and is retried. It
used to return "solver" and cache that like a real answer: measured on
qwen_27b_nothink/slau_r2, where five studies shared one endpoint, the call got
no output within 300 s, the guess was cached, and an implementation task
started down the CFD-study pipeline.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent
for _p in (str(_SCRIPTS_DIR), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

VALID_MODES = ("solver", "surrogate", "implementation")
DEFAULT_MODE = "solver"


class StudyModeUnavailable(RuntimeError):
    """The kind of study could not be decided. Raised instead of guessing."""


def _attempts() -> int:
    try:
        return max(1, int(os.environ.get("CFD_SCIENTIST_STUDY_MODE_ATTEMPTS") or 3))
    except ValueError:
        return 3


def _retry_wait_s() -> float:
    """Base wait between attempts; the n-th retry waits n times this."""
    try:
        return max(0.0, float(os.environ.get("CFD_SCIENTIST_STUDY_MODE_RETRY_WAIT_S") or 30))
    except ValueError:
        return 30.0


def _is_defaulted(doc: Any) -> bool:
    """A cached answer that was a fallback, not a classification.

    Caches written before StudyModeUnavailable existed carry no flag; the
    fallback they stored always had confidence "low" and a reason ending in
    "defaulted", which this harness wrote itself.
    """
    if not isinstance(doc, dict):
        return False
    if doc.get("defaulted") is True:
        return True
    return (str(doc.get("confidence", "")).strip().lower() == "low"
            and str(doc.get("reason", "")).rstrip().endswith("defaulted"))

_SYSTEM = (
    "You classify a research task by what a candidate solution has to produce. "
    "Answer with one JSON object and nothing else."
)

_USER_TEMPLATE = """Below is a study's topic and a listing of its starter folder, \
including whatever task description the folder ships.

Decide which of these three kinds of study it is:

  "solver"         — the work is RUNNING SIMULATIONS. Either a simulation study
                     (set up and run cases with an existing solver, such as a
                     parameter study or a mesh-independence study, compared with
                     reference data), or a SEARCH over candidates where each
                     candidate modifies or extends a simulation/solver code, is
                     compiled or configured, and is then RUN on a case, and
                     candidates compete to beat a baseline.

  "surrogate"      — a SEARCH over candidates: each candidate is a model FITTED to
                     data that the starter already contains (a regression, a
                     surrogate, a reduced-order or data-driven model), trained
                     and then used to predict values for evaluation inputs.

  "implementation" — NO search: the task fixes one new piece of code to build (a
                     numerical scheme, a solver, a model or method specified in a
                     paper) and asks for it to be implemented, run on the
                     verification cases the task defines, and checked against the
                     task's own scorer or acceptance criteria.

Judge by what the task asks to be DONE, not by the subject matter and not by
which file extensions appear. A study can be about fluid dynamics and still be
a surrogate study; a study can ship CSV reference data and still be a solver
study. "implementation" means the task names new code to write. A task that
only asks for cases to be set up and run with an existing solver is a "solver"
study, even when it fixes the setup and ships a scorer with acceptance criteria.
Otherwise the difference is whether the task already says exactly what to build
(implementation) or asks for the best candidate among many (solver or
surrogate).

If the evidence genuinely does not distinguish them, answer "solver".

TOPIC:
{topic}

STARTER FOLDER: {starter_dir}
{listing}

TASK DESCRIPTION FOUND IN THE FOLDER:
{task_text}

Reply exactly:
{{"mode": "solver" | "surrogate" | "implementation", "confidence": "high" | "medium" | "low", "reason": "<one sentence>"}}
"""


def _folder_listing(starter_dir: Path, *, max_entries: int = 120) -> str:
    lines: List[str] = []
    try:
        for p in sorted(starter_dir.rglob("*")):
            if any(part.startswith(".") for part in p.parts):
                continue
            try:
                rel = p.relative_to(starter_dir)
            except Exception:
                continue
            if p.is_dir():
                lines.append(f"  {rel}/")
            else:
                try:
                    size = p.stat().st_size
                except OSError:
                    size = 0
                lines.append(f"  {rel} ({size} bytes)")
            if len(lines) >= max_entries:
                lines.append(f"  ... (listing truncated at {max_entries} entries)")
                break
    except Exception as exc:
        lines.append(f"  <could not list: {exc}>")
    return "\n".join(lines)


def _task_text(starter_dir: Path, *, max_chars: int = 12000) -> str:
    """Whatever the folder says about itself, in the order a person would read.

    Not a naming convention: these are the conventional names a task brief
    takes, tried in order, and the first one that exists is used. If none do,
    the listing alone has to carry the decision.
    """
    for name in ("TASK.md", "TASK.txt", "README.md", "README.txt", "task.md", "readme.md"):
        p = starter_dir / name
        if p.is_file():
            try:
                return f"--- {name} ---\n" + p.read_text(encoding="utf-8", errors="ignore")[:max_chars]
            except Exception:
                continue
    return "(no task description file found in the starter folder)"


def _coerce(payload: Any) -> Optional[Dict[str, str]]:
    if not isinstance(payload, dict):
        return None
    mode = str(payload.get("mode", "")).strip().lower()
    if mode not in VALID_MODES:
        return None
    return {
        "mode": mode,
        "confidence": str(payload.get("confidence", "") or "").strip().lower(),
        "reason": str(payload.get("reason", "") or "").strip()[:400],
    }


def _parse_decision(raw: Any) -> Optional[Dict[str, str]]:
    text = str(raw or "").strip()
    # The model is asked for one bare JSON object. Take the outermost braces
    # rather than pattern-matching the content: anything else is a parse of
    # free text, which is what asking for JSON is meant to avoid.
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        try:
            return _coerce(json.loads(text[start:end + 1]))
        except Exception:
            return None
    return None


def classify_study_mode(
    *, topic: str, starter_dir: Path, timeout_s: int = 300
) -> Dict[str, str]:
    """Ask the model, retrying a failed or unreadable reply.

    Raises StudyModeUnavailable when no attempt yields a decision.
    """
    try:
        from oed_extensions import _llm_invoke  # type: ignore
    except Exception as exc:
        raise StudyModeUnavailable(f"classifier unavailable ({exc})") from exc
    prompt = _USER_TEMPLATE.format(
        topic=str(topic or "(none given)").strip()[:6000],
        starter_dir=starter_dir,
        listing=_folder_listing(starter_dir),
        task_text=_task_text(starter_dir),
    )
    attempts = _attempts()
    problems: List[str] = []
    for attempt in range(1, attempts + 1):
        try:
            raw = _llm_invoke([("system", _SYSTEM), ("user", prompt)], timeout_s=timeout_s)
        except Exception as exc:
            problems.append(f"attempt {attempt}: call failed ({exc})")
        else:
            got = _parse_decision(raw)
            if got:
                return got
            problems.append(f"attempt {attempt}: no usable JSON in the reply")
        if attempt < attempts:
            time.sleep(_retry_wait_s() * attempt)
    raise StudyModeUnavailable(
        f"could not decide the kind of study after {attempts} attempts: "
        + "; ".join(problems)[-1500:]
    )


def resolve_study_mode(
    *, topic: str, starter_dir: Path, cache_path: Path, timeout_s: int = 300
) -> Dict[str, str]:
    """Cached classification. Decided once per study, then read back.

    A cached fallback from before StudyModeUnavailable is not trusted: the
    study is classified again. A classification that fails raises and leaves
    the cache untouched.
    """
    try:
        if cache_path.is_file():
            doc = json.loads(cache_path.read_text(encoding="utf-8"))
            cached = _coerce(doc)
            if cached and not _is_defaulted(doc):
                return cached
    except Exception:
        pass
    got = classify_study_mode(topic=topic, starter_dir=starter_dir, timeout_s=timeout_s)
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(got, indent=2), encoding="utf-8")
    except Exception:
        pass
    return got


def runner_script_for(mode: str) -> str:
    """Which agentic runner a candidate of this study uses."""
    mode = str(mode).strip().lower()
    if mode == "surrogate":
        return "scripts/surrogate_agentic.py"
    if mode == "implementation":
        return "scripts/implementation_agentic.py"
    return "scripts/code_mod_agentic.py"


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Classify a study's execution mode")
    ap.add_argument("--starter-dir", required=True)
    ap.add_argument("--topic", default="")
    ap.add_argument("--cache", default="")
    a = ap.parse_args()
    try:
        out = (resolve_study_mode(topic=a.topic, starter_dir=Path(a.starter_dir),
                                  cache_path=Path(a.cache))
               if a.cache else
               classify_study_mode(topic=a.topic, starter_dir=Path(a.starter_dir)))
    except StudyModeUnavailable as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(out, indent=2))
    print(f"runner: {runner_script_for(out['mode'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
