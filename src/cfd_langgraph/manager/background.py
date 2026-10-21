"""Candidates run in the background, collected as they finish.

The manager used to launch a round of candidates as concurrent ``task`` calls
in one message. A message's tool calls return together, so nothing came back
until the slowest candidate of the round was done: on
airfrans_codex_20261009 a round of 16, built three at a time, kept eight
finished candidates unrecorded for three to five hours, and the search
recorded nothing from 09:24 into the evening.

Here the same oed-candidate-runner, with the same task text, runs in a
background thread. ``oed_launch_candidates`` returns at once;
``oed_wait_for_candidates`` hands back each candidate's report as soon as it
is done, so the manager records it and launches the next while the others
are still building. Builds still go through the study's CaseCoordinator,
which caps how many run at once.
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from langchain_core.messages import HumanMessage

from cfd_langgraph.llm.reply import reply_text

_MAX_THREADS = 64  # builds are capped by the coordinator, not by this pool
_WAIT_CAP_S = 3600


class BackgroundCandidates:
    def __init__(self, runnable: Any, out_dir: Path, slots: Callable[[], Optional[int]]):
        self._runnable = runnable
        self._out_dir = Path(out_dir)
        self._slots = slots
        self._pool = ThreadPoolExecutor(max_workers=_MAX_THREADS, thread_name_prefix="oed-candidate")
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._cv = threading.Condition()
        self._count = 0

    def _run(self, job_id: str, description: str) -> str:
        result = self._runnable.invoke({"messages": [HumanMessage(content=description)]})
        messages = (result or {}).get("messages") or []
        return reply_text(messages[-1]) if messages else "(the runner returned no message)"

    def _finished(self, job_id: str, future: Future) -> None:
        with self._cv:
            job = self._jobs[job_id]
            job["finished_at"] = time.time()
            try:
                job["report"] = future.result()
            except BaseException as exc:  # noqa: BLE001 - reported to the manager, not raised
                job["error"] = f"{type(exc).__name__}: {exc}"
            self._save()
            self._cv.notify_all()

    def _save(self) -> None:
        """The jobs on disk, so a restart shows what was in flight. Caller holds _cv."""
        path = self._out_dir / "open_ended_discovery" / "background_candidates.json"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({
                job_id: {k: v for k, v in job.items() if k != "future"}
                for job_id, job in self._jobs.items()
            }, indent=2))
        except OSError:
            pass

    def _in_flight(self) -> List[Dict[str, Any]]:
        now = time.time()
        return [{"id": job_id, "running_min": round((now - job["launched_at"]) / 60, 1),
                 "task": job["task"][:160]}
                for job_id, job in self._jobs.items() if "finished_at" not in job]

    def launch(self, task_descriptions: List[str]) -> dict:
        launched = []
        with self._cv:
            for description in task_descriptions:
                description = str(description or "").strip()
                if not description:
                    continue
                self._count += 1
                job_id = f"bg_{self._count:03d}"
                self._jobs[job_id] = {"task": description, "launched_at": time.time()}
                future = self._pool.submit(self._run, job_id, description)
                self._jobs[job_id]["future"] = future
                future.add_done_callback(lambda f, j=job_id: self._finished(j, f))
                launched.append(job_id)
            self._save()
            in_flight = len(self._in_flight())
        return {
            "launched": launched,
            "in_flight": in_flight,
            "build_slots": self._slots(),
            "next_step": ("These run in the background. Call oed_wait_for_candidates to collect "
                          "each report as it finishes, record those candidates, and launch the "
                          "next ones while the rest are still building."),
        }

    def wait(self, max_wait_s: int = 1800) -> dict:
        deadline = time.time() + max(0, min(int(max_wait_s or 0), _WAIT_CAP_S))
        with self._cv:
            def ready() -> List[str]:
                return [j for j, job in self._jobs.items() if "finished_at" in job and not job.get("delivered")]

            while not ready() and self._in_flight() and time.time() < deadline:
                self._cv.wait(timeout=max(1.0, deadline - time.time()))
            finished = []
            for job_id in ready():
                job = self._jobs[job_id]
                job["delivered"] = True
                finished.append({"id": job_id, "task": job["task"][:300],
                                 "minutes": round((job["finished_at"] - job["launched_at"]) / 60, 1),
                                 **({"error": job["error"]} if "error" in job else {"report": job.get("report", "")})})
            self._save()
            in_flight = self._in_flight()
        return {
            "finished": finished,
            "in_flight": in_flight,
            "build_slots": self._slots(),
            "next_step": (
                "Act on each finished report as step c1/c2 say, then record the finished "
                "candidates with oed_record_candidate_results. Keep about build_slots "
                "candidates in flight: propose and launch more as they finish."
                if finished else
                "Nothing finished within the wait. Call oed_wait_for_candidates again."
                if in_flight else
                "Nothing is in flight. Propose and launch candidates."),
        }


def build_background_candidate_tools(runner: Dict[str, Any], out_dir: Path, coordinator: Any) -> List[Any]:
    """The manager's two tools for running oed-candidate-runner in the background."""
    jobs = BackgroundCandidates(
        runner["runnable"], out_dir,
        slots=lambda: coordinator.concurrency_for("oed_candidates") if coordinator is not None else None,
    )

    def oed_launch_candidates(task_descriptions: List[str]) -> dict:
        """Start one oed-candidate-runner per task description, in the background,
        and return at once. Each description is the full task you would give the
        runner: a candidate spec to build and score (variant_name, action_type,
        hypothesis, plan, strategy, target_family, ...), or an existing
        candidate_dir to score. Builds run as many at a time as the study's
        build_slots allow; the rest wait their turn. Collect the reports with
        oed_wait_for_candidates."""
        return jobs.launch(task_descriptions)

    def oed_wait_for_candidates(max_wait_s: int = 1800) -> dict:
        """Wait until at least one launched candidate has finished (or
        max_wait_s, at most 3600, has passed) and return every finished
        candidate's report not returned before, with what is still in flight
        and the current build_slots."""
        return jobs.wait(max_wait_s)

    return [oed_launch_candidates, oed_wait_for_candidates]
