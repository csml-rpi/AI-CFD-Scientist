#!/usr/bin/env python3
"""pipeline_contract.run_training: the first seed trains alone, the rest side by
side, as many at once as the visible GPUs' free memory holds, each on the least
busy GPU."""
from __future__ import annotations

import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cfd_langgraph import pipeline_contract as pc  # noqa: E402

FAILURES = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + ("" if ok else f"  -- {detail}"))
    if not ok:
        FAILURES.append(name)


class FakeSandbox:
    """Runs no shell: writes the model file train.sh would, after a pause, and
    counts how many seeds train at once and on which GPU."""

    def __init__(self, pause: float):
        self.pause, self.running, self.most, self.gpus = pause, 0, 0, []
        self.lock = threading.Lock()

    def run_bash(self, cmd, cwd=None, timeout=None):
        with self.lock:
            self.running += 1
            self.most = max(self.most, self.running)
            self.gpus.append(cmd.split("CUDA_VISIBLE_DEVICES=")[1].split()[0]
                             if "CUDA_VISIBLE_DEVICES=" in cmd else None)
        time.sleep(self.pause)
        (Path(cmd.split()[-1]) / "model.bin").write_text("x")
        with self.lock:
            self.running -= 1
        return {"rc": 0}


def train(gpus, peak_gb, seeds=10, pause=0.3, procs=None):
    """run_training with nvidia-smi replaced: ``gpus`` {index: (total GB, used GB)}
    before training, the first GPU holding ``peak_gb`` more while seed 0 trains,
    and ``procs`` {index: processes on it}."""
    state = {k: (t * 1024, u * 1024) for k, (t, u) in gpus.items()}
    first = next(iter(state), None)
    calls = {"n": 0}

    def fake_memory():
        calls["n"] += 1
        out = dict(state)
        if first is not None and 1 < calls["n"] < 4:  # the samples taken while seed 0 trains
            t, u = out[first]
            out[first] = (t, u + peak_gb * 1024)
        return out

    pc.visible_gpu_memory = fake_memory
    pc.gpu_processes = lambda: dict(procs or {})
    sb = FakeSandbox(pause)
    with tempfile.TemporaryDirectory() as td:
        run = Path(td) / "cand"
        (run / "pipeline").mkdir(parents=True)
        res = pc.run_training(sandbox=sb, run_dir=run, spec={"seeds": list(range(seeds))},
                              models_root=Path(td) / "models", timeout_s=60)
    return res, sb


original, original_procs = pc.visible_gpu_memory, pc.gpu_processes
# The sampler wakes every 5 s: seed 0 lasts long enough for two samples.
res, sb = train({"1": (48, 4)}, peak_gb=2, pause=11)
check("every seed trains and is recorded in order",
      res["ok"] and [r["seed"] for r in res["runs"]] == list(range(10)), res)
check("seeds after the first train several at a time on a GPU with room", sb.most >= 4, sb.most)
check("each seed, the first too, is pinned to a visible GPU", set(sb.gpus) == {"1"}, sb.gpus)

res, sb = train({"0": (48, 47.5)}, peak_gb=2, pause=0.2)
check("a GPU with no room trains the seeds one after another", res["ok"] and sb.most == 1, sb.most)

res, sb = train({}, peak_gb=0, pause=0.1, seeds=3)
check("without a GPU the seeds train one after another, as before",
      res["ok"] and sb.most == 1 and all(g is None for g in sb.gpus), sb.gpus)

# The framework's extra seeds for a candidate (two) went to GPU 0, busy with
# five other runs, while GPUs 1 and 2 sat idle (airfrans_codex_20261009).
three = {"0": (48, 6.5), "1": (48, 0.1), "2": (48, 0.1)}
res, sb = train(three, peak_gb=1, pause=0.2, seeds=2, procs={"0": 5, "1": 0, "2": 0})
check("two seeds, trained one after another, avoid the busy GPU", res["ok"] and "0" not in sb.gpus, sb.gpus)
res, sb = train(three, peak_gb=1, pause=11, seeds=3, procs={"0": 5, "1": 0, "2": 0})
check("seeds trained side by side take the idle GPUs first",
      res["ok"] and set(sb.gpus) == {"1", "2"}, sb.gpus)

pc.visible_gpu_memory, pc.gpu_processes = original, original_procs
check("screening takes the first three of ten seeds", pc.screen_seeds({"seeds": list(range(10))}) == [0, 1, 2])
check("a study of three seeds or fewer is not screened", pc.screen_seeds({"seeds": [0, 1, 2]}) == [0, 1, 2])

print()
print("FAILURES:", FAILURES if FAILURES else "none")
sys.exit(1 if FAILURES else 0)
