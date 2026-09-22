from __future__ import annotations

import os
import threading
from typing import Any, Callable, Dict, Optional, Set

from .resource_probe import ResourceProfile, benchmark_case
from .scheduler import ResourceAwareScheduler, compute_max_concurrency


def _calibration_window_from_env() -> float:
    try:
        return float(os.environ.get("CFD_SCIENTIST_CALIBRATION_WINDOW_S", "") or 600.0)
    except ValueError:
        return 600.0


class CaseCoordinator:
    """Per-study coordinator: the first case in a physics group calibrates the
    machine (runs alone, benchmarked), every later case in that group is
    scheduled against the concurrency limit that calibration produced.

    Calibration holds everything else back for at most ``calibration_window_s``
    (CFD_SCIENTIST_CALIBRATION_WINDOW_S, default 600 s). A case still running
    then has shown what it costs; the limit is set from that measurement and
    the case carries on as an ordinary running case. It used to hold the
    whole batch until the case FINISHED, which for a build agent is hours: on
    2026-09-13 three studies each left three candidates idle for 2.5-3 hours
    behind their first one.

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
        calibration_window_s: Optional[float] = None,
    ):
        self.safety_margin = safety_margin
        self.max_concurrency_cap = max_concurrency_cap
        self.forced_max_concurrency = forced_max_concurrency
        self.calibration_window_s = (
            float(calibration_window_s) if calibration_window_s is not None
            else _calibration_window_from_env()
        )
        self._schedulers: Dict[str, ResourceAwareScheduler] = {}
        self._profiles: Dict[str, ResourceProfile] = {}
        self._condition = threading.Condition()
        self._active_cases = 0
        self._calibrating = False
        self._calibration_waiters = 0
        self._calibrating_groups: Set[str] = set()

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
        while True:
            calibrate = False
            with self._condition:
                if group in self._schedulers:
                    break
                if self.forced_max_concurrency is not None:
                    # User pinned it: no calibration, but the first case still
                    # acquires the same global/per-group slots as every later one.
                    self._schedulers[group] = ResourceAwareScheduler(self.forced_max_concurrency)
                    self._condition.notify_all()
                    break
                if group not in self._calibrating_groups:
                    self._calibrating_groups.add(group)
                    calibrate = True
                else:
                    # Another call is calibrating this group: wait for its limit,
                    # or for it to fail, in which case the next call calibrates.
                    self._condition.wait_for(
                        lambda: group in self._schedulers or group not in self._calibrating_groups
                    )
                    continue
            if calibrate:
                return self._calibrate(group, run_fn)
        return self._run_limited(group, run_fn)

    def _effective_global_limit(self) -> int:
        if not self._schedulers:
            return 1
        return max(1, min(s.max_concurrency for s in self._schedulers.values()))

    def _install(self, group: str, profile: ResourceProfile) -> None:
        """Set ``group``'s limit from ``profile``. Caller holds ``_condition``."""
        self._profiles[group] = profile
        n = compute_max_concurrency(
            profile,
            safety_margin=self.safety_margin,
            max_concurrency_cap=self.max_concurrency_cap,
        )
        self._schedulers[group] = ResourceAwareScheduler(n)

    def _calibrate(self, group: str, run_fn: Callable[[], Any]) -> Any:
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

        released = False

        def _release(profile: ResourceProfile) -> None:
            nonlocal released
            with self._condition:
                if released:
                    return
                self._install(group, profile)
                self._calibrating = False
                # Still running, now as an ordinary case that holds a slot.
                self._active_cases += 1
                released = True
                self._condition.notify_all()

        try:
            result, profile = benchmark_case(
                run_fn, window_s=self.calibration_window_s, on_window=_release,
            )
        except BaseException:
            with self._condition:
                if released:
                    self._active_cases -= 1
                else:
                    self._calibrating = False
                self._calibrating_groups.discard(group)
                self._condition.notify_all()
            raise
        with self._condition:
            if released:
                self._active_cases -= 1
            else:
                self._install(group, profile)
                self._calibrating = False
            self._calibrating_groups.discard(group)
            self._condition.notify_all()
        return result

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
