from __future__ import annotations

import threading
from typing import Any, Callable, Dict, Optional

from .resource_probe import ResourceProfile, benchmark_case
from .scheduler import ResourceAwareScheduler, compute_max_concurrency


class CaseCoordinator:
    """Per-study coordinator: the first case in a physics group calibrates the
    machine (runs alone, benchmarked), every later case in that group is
    scheduled against the concurrency limit that calibration produced.

    One instance per study (``out_dir``). The manager builds one and closes
    over it when it constructs the case-running tool, so every concurrent
    tool call the LLM issues shares the same semaphore instance — the actual
    concurrency cap is enforced here, not by trusting the model to self-limit.
    """

    def __init__(
        self,
        *,
        safety_margin: float = 0.85,
        max_concurrency_cap: Optional[int] = None,
        forced_max_concurrency: Optional[int] = None,
    ):
        self.safety_margin = safety_margin
        self.max_concurrency_cap = max_concurrency_cap
        self.forced_max_concurrency = forced_max_concurrency
        self._locks: Dict[str, threading.Lock] = {}
        self._schedulers: Dict[str, ResourceAwareScheduler] = {}
        self._profiles: Dict[str, ResourceProfile] = {}
        self._global_lock = threading.Lock()
        self._condition = threading.Condition()
        self._active_cases = 0
        self._calibrating = False
        self._calibration_waiters = 0

    def _lock_for(self, group: str) -> threading.Lock:
        with self._global_lock:
            if group not in self._locks:
                self._locks[group] = threading.Lock()
            return self._locks[group]

    def run_case(self, group: str, run_fn: Callable[[], Any]) -> Any:
        """Run ``run_fn`` (one case's blocking execution) under ``group``'s
        concurrency cap, calibrating on the first call for that group.

        Per-group semaphores alone are unsafe for mixed studies: group A and
        group B could each launch their full machine-sized allowance. Every
        case therefore also acquires a study-wide dynamic slot whose limit is
        the minimum of all calibrated group limits. New group calibration is
        exclusive and gets priority over ordinary launches, so its system-wide
        measurements are not contaminated by another case.
        """
        if group not in self._schedulers:
            lock = self._lock_for(group)
            with lock:
                if group not in self._schedulers:  # re-check: still true after acquiring the lock
                    if self.forced_max_concurrency is not None:
                        # User pinned it: no calibration, but the first case
                        # still has to acquire the same global/per-group slots
                        # as every later case.
                        with self._condition:
                            self._schedulers[group] = ResourceAwareScheduler(self.forced_max_concurrency)
                            self._condition.notify_all()
                    else:
                        result, profile = self._benchmark_exclusive(run_fn)
                        self._profiles[group] = profile
                        n = compute_max_concurrency(
                            profile,
                            safety_margin=self.safety_margin,
                            max_concurrency_cap=self.max_concurrency_cap,
                        )
                        with self._condition:
                            self._schedulers[group] = ResourceAwareScheduler(n)
                            self._condition.notify_all()
                        return result
        return self._run_limited(group, run_fn)

    def _effective_global_limit(self) -> int:
        if not self._schedulers:
            return 1
        return max(1, min(s.max_concurrency for s in self._schedulers.values()))

    def _benchmark_exclusive(self, run_fn: Callable[[], Any]) -> tuple[Any, ResourceProfile]:
        with self._condition:
            self._calibration_waiters += 1
            try:
                self._condition.wait_for(lambda: not self._calibrating and self._active_cases == 0)
                self._calibrating = True
            finally:
                self._calibration_waiters -= 1
                # Must notify even on the normal path's way out, but above all
                # on the abnormal one: if wait_for raises (KeyboardInterrupt is
                # a designed part of this system), the waiter count drops to
                # zero with nobody woken, and every thread parked in
                # _run_limited on `_calibration_waiters == 0` waits forever.
                self._condition.notify_all()
        try:
            return benchmark_case(run_fn)
        finally:
            with self._condition:
                self._calibrating = False
                self._condition.notify_all()

    def _run_limited(self, group: str, run_fn: Callable[[], Any]) -> Any:
        with self._condition:
            self._condition.wait_for(
                lambda: (
                    not self._calibrating
                    and self._calibration_waiters == 0
                    and self._active_cases < self._effective_global_limit()
                )
            )
            self._active_cases += 1
        try:
            return self._schedulers[group].run(run_fn)
        finally:
            with self._condition:
                self._active_cases -= 1
                self._condition.notify_all()

    def profile_for(self, group: str) -> Optional[ResourceProfile]:
        return self._profiles.get(group)

    def concurrency_for(self, group: str) -> Optional[int]:
        sched = self._schedulers.get(group)
        return sched.max_concurrency if sched else None
