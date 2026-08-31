from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Optional, TypeVar

import psutil

from .resource_probe import ResourceProfile

logger = logging.getLogger(__name__)

T = TypeVar("T")


def compute_max_concurrency(
    profile: ResourceProfile,
    *,
    safety_margin: float = 0.85,
    min_concurrency: int = 1,
    max_concurrency_cap: Optional[int] = None,
) -> int:
    """How many cases like ``profile`` can safely run at once on this machine.

    ``min(cores available / cores per case, memory available / memory per
    case)``, then a safety margin — the same shape Snakemake uses for
    resource-sum scheduling and GNU Parallel's ``--memfree`` uses as a live
    gate, applied here to one benchmarked case instead of a user-declared
    resource request. See the design doc's "Running many things at once"
    section for the studies this follows.
    """
    cores_available = psutil.cpu_count(logical=True) or 1
    mem_available_mb = psutil.virtual_memory().available / (1024 * 1024)

    if not profile.measured:
        # The calibration case died before a single sample landed, so this
        # profile describes nothing. Deriving a number from it produced
        # absurd limits (435 concurrent cases on a 128-core box, from the
        # 0.1-core / 256 MB floors) that then persisted for the whole group.
        # Fall back to a structural guess from hardware alone, and let a
        # later real measurement replace it.
        fallback = max(min_concurrency, min(8, cores_available // 4))
        return min(fallback, max_concurrency_cap) if max_concurrency_cap is not None else fallback

    # A solver is at least one process on at least one core; a 0.25-core
    # floor permitted 4x oversubscription of every core on the machine.
    cores_per_case = max(profile.cores_used, 1.0)
    mem_per_case_mb = max(profile.peak_used_mem_mb, 256.0)  # floor: never assume ~0 footprint

    by_cores = cores_available / cores_per_case
    by_mem = mem_available_mb / mem_per_case_mb

    n = int(min(by_cores, by_mem) * safety_margin)
    n = max(min_concurrency, n)
    # Never more concurrent cases than the machine has logical cores, whatever
    # the memory headroom suggests.
    n = min(n, cores_available)
    if max_concurrency_cap is not None:
        n = min(n, max_concurrency_cap)
    return n


class ResourceAwareScheduler:
    """Caps how many cases run at once, and backs off new launches under pressure.

    Wrap per-case work with :meth:`run`. This never kills work already in
    flight — it only withholds the slot a *new* case needs to start, so a
    healthy running case is never interrupted by a pressure event (mirrors
    Dask adaptive's hysteresis: back off scaling up, don't tear down what's
    already running).
    """

    def __init__(
        self,
        max_concurrency: int,
        *,
        pressure_mem_available_mb_floor: float = 512.0,
        recheck_interval_s: float = 30.0,
        pressure_wait_timeout_s: float = 1800.0,
    ):
        self.max_concurrency = max(1, max_concurrency)
        self._sem = threading.Semaphore(self.max_concurrency)
        self._pressure_floor = pressure_mem_available_mb_floor
        self._recheck_interval_s = recheck_interval_s
        self._pressure_wait_timeout_s = pressure_wait_timeout_s
        self._last_check = 0.0
        self._paused = threading.Event()
        self._lock = threading.Lock()

    def _under_pressure(self) -> bool:
        with self._lock:
            now = time.monotonic()
            if now - self._last_check < self._recheck_interval_s:
                return self._paused.is_set()
            self._last_check = now
            available_mb = psutil.virtual_memory().available / (1024 * 1024)
            if available_mb < self._pressure_floor:
                if not self._paused.is_set():
                    logger.warning(
                        "Resource pressure: %.0fMB free < floor %.0fMB — holding new case launches.",
                        available_mb,
                        self._pressure_floor,
                    )
                self._paused.set()
            else:
                self._paused.clear()
            return self._paused.is_set()

    def run(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Run ``fn(*args, **kwargs)``, waiting for a free slot and for pressure to clear."""
        # Pressure is waited out BEFORE taking a slot. Holding the semaphore
        # while spinning on memory pressure deadlocked the study: the paused
        # case kept both its scheduler slot and its coordinator active-case
        # slot, and a new physics group's calibration waits for active cases
        # to reach zero while every other case waits for that calibration —
        # so with free memory stuck under the floor, nothing could ever
        # proceed and nothing timed out.
        waited_s = 0.0
        step_s = min(5.0, self._recheck_interval_s)
        while self._under_pressure():
            if waited_s >= self._pressure_wait_timeout_s:
                # Proceeding under pressure is worse than stalling forever
                # only if the wait is unbounded — which is exactly what this
                # avoids. Launch anyway and let the OS arbitrate, having said
                # so plainly.
                logger.warning(
                    "Still under memory pressure after %.0fs — launching anyway rather than "
                    "stalling the study indefinitely.",
                    waited_s,
                )
                break
            time.sleep(step_s)
            waited_s += step_s
        with self._sem:
            return fn(*args, **kwargs)
