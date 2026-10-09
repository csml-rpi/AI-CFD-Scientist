#!/usr/bin/env python3
"""Agentic surrogate/data-model builder — the non-solver sibling of code_mod_agentic.

Some studies do not modify a solver. Their candidate is a model fitted to data
the starter folder ships: a surrogate for an expensive simulation, a closure
regressed from a dataset, a reduced-order model. The search that drives them is
the same one the solver studies use — the same archive, the same niche
selection, the same allocator, the same budget accounting — because none of
that is specific to how a candidate happens to be executed. Only the execution
and the "did it produce anything" gate differ:

    code_mod_agentic   candidate is a compiled library   gate: a fresh .so, solver reached End
    surrogate_agentic  candidate is a fitted model       gate: fresh predictions on the study's evaluation inputs

The turn loop, tool protocol, sandbox, transcript handling, repair and timeout
logic are imported from code_mod_agentic rather than copied, so the two paths
cannot drift apart.

Scoring is deliberately NOT done here. A candidate that scores itself is a
candidate that can report whatever it likes; the framework scores it afterwards
with the study's bound comparator, exactly as it does for a solver candidate.
This module's only job is to get a model built and its predictions written.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from code_mod_agentic import run_agent_loop  # noqa: E402

# Probed and handed to the agent rather than left for it to discover. A model
# that guesses at its toolchain writes an import for a library that is not
# there, reads the traceback, rewrites, and spends turns finding out what one
# call could have told it -- and the weaker the model, the more turns it
# spends. The list is a superset of what studies of this kind reach for; only
# what is actually importable is reported as available, and what is absent is
# named explicitly so it is not attempted.
_LIBRARY_PROBE_LIST = (
    "numpy", "pandas", "scipy", "sklearn", "torch", "tensorflow", "keras",
    "jax", "xgboost", "lightgbm", "catboost", "statsmodels", "matplotlib",
    "pyarrow", "optuna", "joblib", "numba", "h5py", "seaborn", "tqdm",
    "polars", "skorch", "gpytorch", "torch_geometric", "pyg_lib",
    "torch_scatter", "torch_sparse", "torch_cluster",
)


def probe_installed_libraries() -> Dict[str, Any]:
    """What this interpreter can actually import, with versions."""
    import importlib
    import importlib.metadata as _md

    present: Dict[str, str] = {}
    missing: List[str] = []
    for name in _LIBRARY_PROBE_LIST:
        try:
            mod = importlib.import_module(name)
        except Exception:
            missing.append(name)
            continue
        version = getattr(mod, "__version__", None)
        if version is None:
            try:
                version = _md.version(name)
            except Exception:
                version = "unknown"
        present[name] = str(version)
    accelerator = ""
    try:
        import torch  # noqa: F401

        if torch.cuda.is_available():
            accelerator = f"CUDA available, {torch.cuda.device_count()} device(s)"
        else:
            accelerator = "CPU only"
    except Exception:
        pass
    return {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "present": present,
        "missing": missing,
        "accelerator": accelerator,
    }


def _environment_block() -> str:
    env = probe_installed_libraries()
    present = ", ".join(f"{k} {v}" for k, v in sorted(env["present"].items()))
    missing = ", ".join(env["missing"])
    lines = [
        "# Environment (probed just now — this is what you actually have)",
        f"Interpreter: {env['executable']} (Python {env['python']})",
        f"Installed: {present}",
    ]
    if env["accelerator"]:
        lines.append(f"Compute: {env['accelerator']}")
    if missing:
        lines.append(
            f"NOT installed, do not import and do not try to install: {missing}"
        )
    lines.append(
        "Use the interpreter above for every script you run, by its full path, "
        "so you get this environment rather than whatever `python` resolves to."
    )
    return "\n".join(lines) + "\n"


# What the agent reports on completion. Every field is text so the same schema
# works for native function calling on every provider and for the JSON text
# protocol. Informational: the gate reads the prediction files on disk.
SURROGATE_DONE_FIELDS = {
    "summary": "One-line summary of the model you built.",
    "prediction_files": (
        "Absolute paths of your FINAL prediction files, comma-separated: exactly the "
        "set the study's scorer should receive, one per required seed or run. The "
        "framework scores these files."
    ),
    "seeds": "The seeds you trained, comma-separated.",
    "self_reported_score": "Your own measurement of the study's score, if you took one; empty otherwise.",
}

# The tool contract, stated in full. It was missing: the solver runner carries
# its tool specification inside its own brief, and this runner replaces that
# brief wholesale, so a surrogate agent was told it had tools "(see protocol)"
# with no protocol anywhere in its prompt.
_TOOLS_SPEC = """You have four tools. Call exactly one per turn.

  read_file   {"tool": "read_file", "args": {"path": "<absolute path>", "start": <int, optional>, "max_bytes": <int, optional>}}
      Read a file in the run directory or the starter folder (read-only).
      Long files come in pages of up to 6000 characters, split at line ends.
      Returns {ok, content, size, total_chars, start, end, truncated,
      next_start}. When truncated is true, call read_file again with
      start=next_start for the next page.

  write_file  {"tool": "write_file", "args": {"path": "<absolute path>", "content": "<file body>", "mode": "w"|"a"}}
      Write a file inside the run directory; missing folders are created.
      Returns {ok, path, bytes_written}.

  run_bash    {"tool": "run_bash", "args": {"cmd": "<command>", "cwd": "<absolute path inside run_dir>", "timeout": <seconds, optional>}}
      Run a shell command. The host is read-only; only the run directory and an
      ephemeral /tmp are writable. Returns {rc, stdout, stderr, timeout,
      error_summary}. You see the last 6000 characters of stdout and of
      stderr, so redirect long output to a file in the run directory and page
      through it with read_file. Quote heredocs (<<'EOF') so the shell does not expand $ in the
      file you are writing.
      A command that runs out of time is STOPPED, and the default allowance is
      ten minutes. Training usually needs more than that: time one epoch, or
      one seed, in a short call first, then pass `timeout` big enough for the
      whole command — a timed-out result tells you the allowance you used and
      the ceiling you may ask for. Train one seed per call rather than looping
      over every seed in one command, so a cut-off costs one seed, not all of
      them.

  done        {"tool": "done", "args": {"summary": "...", "prediction_files": "...", "seeds": "...", "self_reported_score": "..."}}
      Signal completion, only after your final prediction files are on disk.
      prediction_files lists exactly those files; they are what gets scored.

If you can call these tools as functions, call them directly. Otherwise reply
with exactly one JSON object per turn in the format above: no prose, no
markdown fences.
"""

SYSTEM_MESSAGE = (
    "You are an agentic machine-learning engineer. You have four tools — "
    "read_file, write_file, run_bash and done — described in the brief. Call "
    "exactly one tool per turn. Iterate on errors patiently.\n\n"
    "A command is stopped when its time allowance runs out — ten minutes by "
    "default — and everything it started is stopped with it. Training normally "
    "needs more: time one epoch, or one seed, in a short call first, then pass "
    "`timeout` big enough for the whole command (a timed-out result names the "
    "ceiling you may ask for). Train one seed per call rather than looping over "
    "every seed in one command, so a cut-off costs one seed and not all of "
    "them.\n\n"
    "Turns are limited and each tool call costs one turn, so make every call do "
    "as much as it safely can. Prefer a single run_bash that chains the whole "
    "step — write the script with a heredoc, run it, and print the result — "
    "over separate read/write/bash calls for the same work. Combine independent "
    "inspection commands into one call with `;` or `&&`. Split into separate "
    "calls only when a later command depends on output you have not seen yet."
)


def build_surrogate_prompt(
    *,
    topic: str,
    hypothesis: str,
    variant_name: str,
    run_dir: Path,
    starter_case: Path,
    plan: str = "",
    prior_attempt: str = "",
    repair_goal: str = "",
    timeout_s: int = 0,
) -> str:
    """The candidate brief. Deliberately says nothing about the science.

    Everything domain-specific reaches the agent through ``topic`` (the study
    brief, which names the data, the split and the metric) and ``hypothesis``
    (this candidate's idea). What is stated here is only the contract the
    framework needs in order to find and score the result afterwards, so the
    same text serves a surrogate for airfoil coefficients, a fitted closure
    term, or anything else a study puts in front of it.
    """
    pred_dir = run_dir / "predictions"
    model_dir = run_dir / "model"
    scorer_s = _baseline_scoring_seconds(run_dir)
    vspec = validation_spec(run_dir)
    parts: List[str] = []
    parts.append(_environment_block())
    parts.append(f"# Tools\n{_TOOLS_SPEC}")
    parts.append(f"# Study\n{topic.strip()}\n")
    parts.append(
        f"# Your candidate: {variant_name}\n{hypothesis.strip()}\n"
        + (f"\nPlan:\n{plan.strip()}\n" if plan.strip() else "")
    )
    if vspec:
        parts.append(_pipeline_block(run_dir, model_dir, vspec))
    else:
        parts.append(
        "# What you must produce\n"
        f"1. Your model code and environment notes under: {model_dir}\n"
        f"2. Your final predictions on the study's evaluation inputs under: {pred_dir}\n"
        "   Write them in whatever file format and column names the study brief\n"
        "   specifies. If the brief names a scorer, read that scorer first and\n"
        "   match its expected input exactly — it is the thing that will judge\n"
        "   you, and it is entitled to refuse a malformed file.\n"
        "   That folder is your submission: what it holds is handed to the scorer\n"
        "   as your result. Put only your final prediction files there, one\n"
        "   complete set per seed or run the study requires. Trial runs, probes,\n"
        "   blends and any other scratch output go elsewhere in your run directory\n"
        "   (a scratch/ folder, for example).\n"
        "3. If the study asks for multiple seeds, produce every seed it asks for.\n"
        "   A single-seed result does not satisfy a multi-seed requirement.\n"
        )
    parts.append(
        "# Rules\n"
        f"- Everything you write goes under {run_dir}. The starter folder\n"
        f"  ({starter_case}) is READ-ONLY input: read it freely, never write to it.\n"
        "- Respect every data-usage rule in the study brief. If the brief marks\n"
        "  data as held out or test-only, it must not touch training, validation,\n"
        "  early stopping, hyper-parameter choice, feature selection, or any\n"
        "  normalisation statistic. This is the one rule that invalidates a\n"
        "  result outright rather than merely worsening it.\n"
        "- Use the libraries listed as installed above. The environment is shared\n"
        "  and fixed: do not install, upgrade, or vendor anything.\n"
        "- Set and record explicit seeds so your result can be reproduced.\n"
        + (
            "- The data you see is a validation version of the study's data: the framework has\n"
            "  moved part of the training samples into the evaluation sets' place, and their\n"
            "  target values are withheld from you, as the test set's are. The real test set is\n"
            "  not used during the search at all. When the search ends, the framework re-runs the\n"
            "  best candidate's train.sh for every seed on the full training data and scores its\n"
            "  predict.sh output on the test set, once. So your scripts must take sample lists and\n"
            "  counts from the starter folder's files at run time and never hard-code them.\n"
            "- The study's scorer cannot score anything here (the targets are withheld): do not\n"
            "  try. Judge your model on a split you hold out from the training data you see, and\n"
            "  choose every setting on it. Read the scorer's code to match its file format.\n"
            "- After you finish, the framework scores your predictions on the validation sets and\n"
            f"  writes the aggregate result to {run_dir / 'validation_score.json'}. Other\n"
            "  candidates' folders beside yours show what they tried and their\n"
            "  validation_score.json.\n"
            if vspec else
            "- The test set's target values are withheld from you. In this workspace the\n"
            "  files that hold them show only their input fields, and files that cannot\n"
            "  be split are not readable, so the study's scorer cannot score your test\n"
            "  predictions here: do not try. Judge your model on a validation split you\n"
            "  hold out from the training data, and choose every setting on it. Read the\n"
            "  scorer's code to match the file format it expects. The framework scores\n"
            "  your final files after you finish, and that is the only score that counts.\n"
            if _targets_withheld(run_dir) else
            "- Judge your model while you build it on data the study lets you use for\n"
            "  that, normally a validation split you hold out from the training data.\n"
            "  Run the study's scorer on your files to check that it accepts them, and\n"
            "  use its numbers only as far as the study's data-usage rules allow. The\n"
            "  framework runs the scorer on your final files after you finish, and that\n"
            "  is the only score that counts, so you do not need to score the final set\n"
            "  yourself.\n"
        )
        + (
            f"- On this machine the study's scorer took about {scorer_s:.0f}s to score the\n"
            "  supplied baseline result. Allow for that whenever you run it.\n"
            if scorer_s and not _targets_withheld(run_dir) else ""
        )
    )
    if prior_attempt.strip():
        parts.append(f"# Previous attempt\n{prior_attempt.strip()}\n")
    if repair_goal.strip():
        parts.append(f"# Repair goal for this continuation\n{repair_goal.strip()}\n")
    if timeout_s:
        parts.append(
            f"# Time\nYou have about {timeout_s}s. Budget it: get one honest\n"
            "end-to-end result on disk early, then improve it. A finished\n"
            "mediocre model scores; an unfinished excellent one does not.\n"
        )
    parts.append(
        "# Finish\n"
        "When train.sh and predict.sh are in place and every seed is trained, call the done "
        "tool with a one-line summary, the seeds you trained, and your own validation score "
        "if you measured one (prediction_files may be empty). The framework runs predict.sh "
        "and scores what it writes; the score you report is informational.\n"
        if vspec else
        "# Finish\n"
        "When your final predictions are written, call the done tool with a one-line "
        "summary, prediction_files listing exactly your final prediction files, the "
        "seeds you ran, and your own score if you measured one. The framework scores "
        "the files you list (every file in the predictions folder if you list none); "
        "the score you report is informational.\n"
    )
    return "\n".join(parts)


def validation_spec(run_dir: Optional[Path]) -> Dict[str, Any]:
    """The per-seed file spec, when this study's search is scored on a
    validation split and candidates leave scripts the framework runs
    (cfd_langgraph.validation_split, cfd_langgraph.pipeline_contract)."""
    from cfd_langgraph import pipeline_contract, validation_split

    if run_dir is None:
        return {}
    for folder in Path(run_dir).resolve().parents:
        if validation_split.ready(folder):
            return pipeline_contract.load_spec(Path(run_dir).resolve().parent / pipeline_contract.SPEC_FILE)
    return {}


def _pipeline_block(run_dir: Path, model_dir: Path, spec: Dict[str, Any]) -> str:
    pipe = run_dir / "pipeline"
    sets = ", ".join(str(s["name"]) for s in spec["sets"])
    files = "; ".join(f"{s['name']}: " + ", ".join(s["files"]) for s in spec["sets"])
    seeds = ", ".join(str(k) for k in spec["seeds"])
    timing = spec.get("timing")
    return (
        "# What you must produce\n"
        f"1. Your model code under {model_dir}.\n"
        f"2. Two scripts in {pipe}, which the framework runs itself:\n"
        f"     bash {pipe}/train.sh SEED MODEL_DIR\n"
        "       trains seed SEED from scratch on the starter folder's training data and writes\n"
        "       everything prediction needs into MODEL_DIR. Each seed is its own training run,\n"
        "       with its random initialisation and data order drawn from SEED.\n"
        f"     bash {pipe}/predict.sh SEED MODEL_DIR OUT_DIR SET\n"
        "       loads MODEL_DIR and writes seed SEED's prediction file(s) for evaluation set SET\n"
        f"       into OUT_DIR, named as the scorer reads them ({files}; {{seed}} is the seed\n"
        f"       number). SET is one of: {sets}. It reads the set's inputs from the starter folder.\n"
        + (f"       Do not write {timing['file']}: the framework times predict.sh itself and\n"
           "       writes it.\n" if timing else "")
        + "   Both must work from any directory, find your code relative to their own location,\n"
        "   read data only from the starter folder, and never reuse predictions or arrays an\n"
        "   earlier run left behind. train.sh must not read the evaluation sets at all: when the\n"
        "   framework re-runs it, their files are hidden.\n"
        f"3. Every seed ({seeds}) trained with your train.sh, each into\n"
        f"   {pipe}/models/seed<SEED>/. Run predict.sh on one of them yourself to check it.\n"
        "   The framework scores this candidate by running predict.sh on those models for every\n"
        "   seed and set; prediction files you write yourself are not scored. Seeds whose\n"
        "   predictions are copies, or near-copies, of each other are rejected.\n"
    )


def _targets_withheld(run_dir: Optional[Path] = None) -> bool:
    """Whether this study withholds its test targets from candidates."""
    from cfd_langgraph.withheld_data import find_manifest, load_manifest

    return bool(load_manifest(find_manifest(run_dir)).get("entries"))


def _baseline_scoring_seconds(run_dir: Path) -> Optional[float]:
    """How long the study's scorer took on the baseline, from the binding setup
    wrote next to the candidates (<disc_dir>/bound_comparators.json)."""
    try:
        bound = json.loads((Path(run_dir).parent / "bound_comparators.json").read_text(encoding="utf-8"))
    except Exception:
        return None
    values = [info.get("scorer_seconds") for info in (bound or {}).values() if isinstance(info, dict)]
    values = [float(v) for v in values if isinstance(v, (int, float)) and v > 0]
    return max(values) if values else None


def declared_submission_files(payload: Any, run_dir: Path) -> Tuple[List[str], bool]:
    """The final prediction files an agent named in `done`, and whether it named any.

    Only files that exist inside <run_dir>/predictions/ count. The scoring
    wrapper used to receive every file in that folder, and agents leave trial
    runs and blends there: dlr_airfoil_codex_20260913b recorded 9.375 for a
    candidate whose ten real seeds give 9.312, and another candidate there
    would have been scored as thirteen "seeds" from two real ones.
    """
    raw = payload.get("prediction_files") if isinstance(payload, dict) else None
    items = raw if isinstance(raw, list) else str(raw or "").replace("\n", ",").split(",")
    run_dir = Path(run_dir)
    pred_dir = (run_dir / "predictions").resolve()
    files: List[str] = []
    for item in items:
        text = str(item or "").strip().strip("'\"")
        if not text:
            continue
        path = Path(text).expanduser()
        path = path if path.is_absolute() else run_dir / path
        try:
            resolved = path.resolve()
            resolved.relative_to(pred_dir)
        except (OSError, ValueError):
            continue
        if resolved.is_file() and str(resolved) not in files:
            files.append(str(resolved))
    return files, bool(files)


def find_surrogate_artifacts(
    *, run_dir: Path, started_at: float, stale_grace_s: float = 5.0
) -> Dict[str, Any]:
    """Locate what this candidate actually produced.

    Freshness is checked the same way the solver path checks a .so: an artifact
    left behind by an earlier attempt, or copied in from the starter, is not
    evidence that THIS candidate produced anything. Without that check a
    candidate that did nothing inherits its predecessor's files and is scored
    as if it had worked.
    """
    pred_dir = run_dir / "predictions"
    model_dir = run_dir / "model"
    fresh: List[str] = []
    stale: List[str] = []
    if pred_dir.is_dir():
        for p in sorted(pred_dir.rglob("*")):
            if not p.is_file() or p.stat().st_size == 0:
                continue
            if p.stat().st_mtime >= started_at - stale_grace_s:
                fresh.append(str(p))
            else:
                stale.append(str(p))
    model_files = (
        [str(p) for p in sorted(model_dir.rglob("*")) if p.is_file()]
        if model_dir.is_dir() else []
    )
    return {
        "prediction_files": fresh,
        "stale_prediction_files": stale,
        "model_files": model_files,
        "predictions_dir": str(pred_dir) if pred_dir.is_dir() else "",
        "model_dir": str(model_dir) if model_dir.is_dir() else "",
    }


def run(
    *,
    hypothesis: str,
    plan: str = "",
    variant_name: str,
    run_dir: Path,
    starter_case: Path,
    starter_root: Optional[Path] = None,
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
        starter_root=starter_root,
        topic=topic,
        model=model,
        max_turns=max_turns,
        timeout_s=timeout_s,
        prior_attempt=prior_attempt,
        repair_goal=repair_goal,
        system_message=SYSTEM_MESSAGE,
        initial_prompt=build_surrogate_prompt(
            topic=topic, hypothesis=hypothesis, variant_name=variant_name,
            run_dir=run_dir, starter_case=starter_case, plan=plan,
            prior_attempt=prior_attempt, repair_goal=repair_goal,
            timeout_s=timeout_s,
        ),
        done_fields=SURROGATE_DONE_FIELDS,
        # No cost tokens: a surrogate candidate is charged one budget unit by
        # oed_score_candidate, flat. Counting `python ` invocations would bill
        # every inspection and self-scoring command as a training run and
        # exhaust the budget on bookkeeping.
    )

    artifacts = find_surrogate_artifacts(
        run_dir=run_dir,
        started_at=0.0 if continuing else started_at,
    )
    produced = bool(artifacts.get("prediction_files"))
    vspec = validation_spec(run_dir)
    pipeline_missing: List[str] = []
    if vspec:
        # Scored on the scripts and trained seeds it leaves, not on files it
        # wrote itself (cfd_langgraph.pipeline_contract).
        from cfd_langgraph.pipeline_contract import missing_parts

        pipeline_missing = missing_parts(run_dir, vspec)
        produced = not pipeline_missing
    final_payload = loop.get("final_payload") or {}
    declared, has_declared = declared_submission_files(final_payload, run_dir)
    result: Dict[str, Any] = {
        # "OK" means prediction files exist, not that the work finished: an
        # agent stopped by the clock or the provider can leave some behind.
        # `finished_cleanly` says which, and the framework diagnoses any build
        # that did not finish before its files are scored.
        "status": "OK" if produced else "FAILED",
        "finished_cleanly": bool(final_payload) and not loop.get("aborted_reason")
        and not loop.get("provider_error"),
        "duration_s": loop.get("duration_s", 0),
        "turns_used": loop.get("turns_used", 0),
        "solver_invocations": loop.get("solver_invocations", 0),
        "aborted_reason": loop.get("aborted_reason", ""),
        "provider_error": bool(loop.get("provider_error")),
        "submission_files": declared if has_declared else artifacts.get("prediction_files", []),
        "submission_declared": has_declared,
        # `case_dir` is what the scoring layer scores, for either kind of
        # candidate. For a fitted model that is the candidate's own run dir,
        # which is where its predictions live.
        "case_dir": str(run_dir),
        "produced_predictions": produced,
        "prediction_files": artifacts.get("prediction_files", []),
        "model_files": artifacts.get("model_files", []),
        "predictions_dir": artifacts.get("predictions_dir", ""),
        "model_dir": artifacts.get("model_dir", ""),
        "compiled_model_name": variant_name,
        "compiled_model_description": hypothesis[:240],
        "trajectory_log": loop.get("trajectory_log", ""),
        "agent_final_payload": loop.get("final_payload", {}),
    }
    if not produced and vspec:
        result["error"] = "missing what the framework runs: " + ", ".join(pipeline_missing)
        result["compile_error_hint"] = result["error"]
    elif not produced:
        if artifacts.get("stale_prediction_files"):
            result["error"] = (
                "prediction files exist but none were written by this attempt "
                "(all older than its start) — the candidate produced nothing new"
            )
        else:
            result["error"] = (
                f"no prediction files produced under {run_dir / 'predictions'}"
            )
        result["compile_error_hint"] = result["error"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="Agentic surrogate/data-model builder")
    ap.add_argument("--hypothesis", required=True)
    ap.add_argument("--plan", default="")
    ap.add_argument("--variant-name", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--starter-case", required=True)
    # Sent by the shared candidate launcher, which also starts solver builds; a
    # fitted model spends no solver runs, so it is accepted and unused. Refusing
    # it killed every first launch of every candidate in a surrogate study.
    ap.add_argument("--solver-budget", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--starter-root", default="",
                    help="The study's starter folder, readable in full by the agent.")
    ap.add_argument("--topic", default="")
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default=os.environ.get("CFD_SCIENTIST_MODEL", ""))
    ap.add_argument("--timeout", default=0, type=int)
    ap.add_argument("--max-turns", default=120, type=int)
    ap.add_argument("--prior-attempt", default="")
    ap.add_argument("--repair-goal", default="")
    a = ap.parse_args()
    res = run(
        hypothesis=a.hypothesis, plan=a.plan, variant_name=a.variant_name,
        run_dir=Path(a.run_dir), starter_case=Path(a.starter_case), topic=a.topic,
        starter_root=(Path(a.starter_root) if str(a.starter_root).strip() else None),
        output_path=Path(a.output), model=a.model, timeout_s=a.timeout,
        max_turns=a.max_turns, prior_attempt=a.prior_attempt, repair_goal=a.repair_goal,
    )
    print(json.dumps({k: res.get(k) for k in ("status", "error", "turns_used",
                                              "produced_predictions")}, indent=2))
    return 0 if res.get("status") == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
