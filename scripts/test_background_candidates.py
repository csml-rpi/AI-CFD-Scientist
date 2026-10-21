#!/usr/bin/env python3
"""Candidates run in the background and are collected as they finish.

On airfrans_codex_20261009 the manager launched 16 candidates as one message
of `task` calls; nothing came back until the slowest was done, and eight
finished candidates waited hours to be recorded. Checked here with a stand-in
runner: launch returns at once, each report comes back as soon as its
candidate finishes, errors come back as errors, nothing is returned twice.

Run: python scripts/test_background_candidates.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from langchain_core.messages import AIMessage  # noqa: E402

from cfd_langgraph.manager.background import build_background_candidate_tools  # noqa: E402

FAILURES: list[str] = []


def check(name: str, got: object, want: object = True) -> None:
    ok = got == want
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + ("" if ok else f": got {got!r}, want {want!r}"))
    if not ok:
        FAILURES.append(name)


class Runner:
    """Stands in for the oed-candidate-runner: 'sleep N' takes N seconds."""

    def invoke(self, state):
        task = state["messages"][0].content
        seconds = float(task.split()[1])
        time.sleep(seconds)
        if "boom" in task:
            raise RuntimeError("runner crashed")
        return {"messages": [AIMessage(content=f"done: {task}")]}


class Coordinator:
    def concurrency_for(self, group):
        return 8 if group == "oed_candidates" else None


with tempfile.TemporaryDirectory() as tmp:
    launch, wait = build_background_candidate_tools({"runnable": Runner()}, Path(tmp), Coordinator())
    t0 = time.time()
    out = launch(["sleep 1 fast", "sleep 4 slow", "sleep 2 boom", ""])
    check("launch returns at once", time.time() - t0 < 0.5)
    check("one job per non-empty task", out["launched"], ["bg_001", "bg_002", "bg_003"])
    check("and reports the build slots", out["build_slots"], 8)

    first = wait(30)
    check("the first to finish comes back first, without waiting for the rest",
          [f["id"] for f in first["finished"]], ["bg_001"])
    check("its report is the runner's final message", first["finished"][0]["report"], "done: sleep 1 fast")
    check("the others are still in flight", sorted(j["id"] for j in first["in_flight"]), ["bg_002", "bg_003"])
    check("returned well before the slowest", time.time() - t0 < 3)

    second = wait(30)
    check("a crashed runner comes back as an error", second["finished"][0].get("error"),
          "RuntimeError: runner crashed")
    third = wait(30)
    check("then the slow one", [f["id"] for f in third["finished"]], ["bg_002"])
    empty = wait(1)
    check("nothing is returned twice", empty["finished"], [])
    check("and nothing in flight is said plainly", "Nothing is in flight" in empty["next_step"])
    saved = json.loads((Path(tmp) / "open_ended_discovery" / "background_candidates.json").read_text())
    check("jobs are kept on disk for a restart", sorted(saved), ["bg_001", "bg_002", "bg_003"])

    launch(["sleep 3 later"])
    t1 = time.time()
    timed = wait(1)
    check("a wait with nothing finished ends at max_wait_s", (timed["finished"], time.time() - t1 < 2.5), ([], True))

print("== each build sees every GPU, its least-loaded one first, and is told to use them all")
from cfd_langgraph import pipeline_contract  # noqa: E402
from cfd_langgraph.manager import tools as T  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
import surrogate_agentic as SA  # noqa: E402

saved_gpus, saved_run, saved_use = pipeline_contract.visible_gpu_memory, T._run_script, SA._gpu_use
try:
    three = {"0": (49140.0, 9000.0), "1": (49140.0, 5.0), "2": (49140.0, 15.0)}
    pipeline_contract.visible_gpu_memory = lambda: three
    orders = [T._build_gpu_order() for _ in range(6)]
    check("six builds' defaults are spread two to a GPU", sorted(o[0] for o in orders), ["0", "0", "1", "1", "2", "2"])
    check("the first goes to the least-used GPU", orders[0][0], "1")
    check("and every build still sees all three", all(sorted(o) == ["0", "1", "2"] for o in orders), True)
    for o in orders:
        T._release_build_gpu(o[0])
    seen: dict = {}
    T._run_script = lambda args, timeout, env: seen.update(env) or "ran"
    T._run_build_script(["x.py"], timeout=1, env={"A": "1"})
    check("the build's environment lists every GPU, default first",
          (sorted(seen.get("CUDA_VISIBLE_DEVICES", "").split(",")), seen.get("CUDA_DEVICE_ORDER"), seen.get("A")),
          (["0", "1", "2"], "PCI_BUS_ID", "1"))
    check("and gives its default back when it ends", sum(T._BUILD_GPU_COUNT.values()), 0)
    SA._gpu_use = lambda: {"0": "97", "1": "0", "2": "3"}
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,0,2"
    block = SA._gpu_block(three, True)
    check("the agent is shown every GPU with its free memory and load",
          all(f"GPU {k}:" in block for k in three) and "97 % busy" in block, True)
    check("and told to spread its runs over them", ("Use all of them" in block, "CUDA_VISIBLE_DEVICES=<GPU number" in block),
          (True, True))
    check("and where an unpinned run goes", "goes to GPU 1" in block)
    pipeline_contract.visible_gpu_memory = lambda: {"0": (49140.0, 5.0)}
    check("with one GPU there is nothing to choose", T._build_gpu_order(), None)
finally:
    pipeline_contract.visible_gpu_memory, T._run_script, SA._gpu_use = saved_gpus, saved_run, saved_use
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)

print()
print("FAILURES:", FAILURES if FAILURES else "none")
sys.exit(1 if FAILURES else 0)
