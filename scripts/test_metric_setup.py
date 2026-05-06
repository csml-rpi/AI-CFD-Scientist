"""Standalone driver to exercise metric_setup against the already-prepared run dir.
Adds verbose, immediate stderr printing so we can see exactly where the
agent loop fails or hangs.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))

import metric_setup  # noqa: E402

# --- patch _llm_invoke and _run_setup_agent to log every step --------------

import oed_extensions  # noqa: E402

_orig_llm_invoke = oed_extensions._llm_invoke

def _verbose_llm_invoke(messages, temperature=0.0, timeout_s=600):
    sys_preview = next((c for r,c in messages if r=="system"), "")[:80]
    user_preview = next((c for r,c in messages if r=="user"), "")[:80]
    t0 = time.time()
    print(f"[llm_invoke] CALL  sys={sys_preview!r} user={user_preview!r}", file=sys.stderr, flush=True)
    try:
        out = _orig_llm_invoke(messages, temperature=temperature, timeout_s=timeout_s)
        print(f"[llm_invoke] OK    {time.time()-t0:.1f}s out={out[:120]!r}", file=sys.stderr, flush=True)
        return out
    except Exception as e:
        print(f"[llm_invoke] EXC   {time.time()-t0:.1f}s {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        raise

oed_extensions._llm_invoke = _verbose_llm_invoke
metric_setup._llm_invoke = _verbose_llm_invoke  # in case it was imported into metric_setup's namespace

_orig_run_setup_agent = metric_setup._run_setup_agent

def _verbose_run_setup_agent(*a, **kw):
    print("[setup_agent] STARTING", file=sys.stderr, flush=True)
    sink = kw.get("transcript_sink")
    final_obj, hist = _orig_run_setup_agent(*a, **kw)
    print(f"[setup_agent] DONE  final_obj={'<set>' if final_obj else 'None'}  history_len={len(hist)}  sink_len={len(sink) if sink is not None else 'no-sink'}", file=sys.stderr, flush=True)
    return final_obj, hist

metric_setup._run_setup_agent = _verbose_run_setup_agent


if __name__ == "__main__":
    run_dir = "/home/somasn/Desktop/cfd-scientist-arch-change/runs/oed_turbulence_model_sonnet_46_multi_metric"
    topic = ("Open-ended discovery: find a novel SA model modification for "
             "periodic hill flow at Re=5600 that beats baseline SA and available "
             "literature. Evaluate candidates against DNS using THREE metrics: "
             "(1) Cf RMSE along the lower wall, (2) reattachment length "
             "x_reattach/h, and (3) separation onset x_separation/h. Base case "
             "and DNS reference data are in the starter folder. Propose, "
             "implement, and test new model terms not in the literature.")

    args = argparse.Namespace(
        run_dir=run_dir,
        topic=topic,
        starter_dir="/home/somasn/Desktop/cfd-scientist-arch-change/starter",
        baseline_case_dir=f"{run_dir}/baseline_case/case",
        baseline_metrics=f"{run_dir}/baseline_metrics.json",
        reference_data_manifest=f"{run_dir}/reference_data_manifest.json",
        output=f"{run_dir}/metric_specs.json",
        comparator_out=f"{run_dir}/comparators",
        timeline=f"{run_dir}/timeline.json",
    )

    print("[driver] launching run_metric_setup", file=sys.stderr, flush=True)
    try:
        rc = metric_setup.run_metric_setup(args)
        print(f"[driver] rc={rc}", file=sys.stderr, flush=True)
        sys.exit(rc)
    except Exception as e:
        print(f"[driver] UNCAUGHT {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        sys.exit(99)
