from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, List, Tuple

import psutil


@dataclass
class ResourceProfile:
    """What one case actually costs, measured by watching it run alone.

    Produced by :func:`benchmark_case`. Feeds :func:`compute_max_concurrency`
    to decide how many more cases like this one can safely run at the same
    time on this machine.
    """

    wall_clock_s: float
    peak_used_mem_mb: float
    avg_cpu_percent: float  # system-wide psutil.cpu_percent() average, 0-100
    logical_cores: int = field(default_factory=lambda: psutil.cpu_count(logical=True) or 1)
    sample_count: int = 1  # how many live samples backed this profile; 0 = unmeasured

    @property
    def measured(self) -> bool:
        """False when the calibration case ended before a single sample landed."""
        return self.sample_count > 0

    @property
    def cores_used(self) -> float:
        """Rough core-equivalents consumed, from system-wide CPU% during the run.

        ``psutil.cpu_percent(percpu=False)`` is normalized to 0-100 for the
        whole machine, not 0-100 per core. Convert that utilization fraction
        back to logical-core equivalents. The previous division by 100 alone
        undercounted CPU demand by roughly ``logical_cores`` and could launch
        far too many solver processes.
        """
        return max(0.1, (self.avg_cpu_percent / 100.0) * self.logical_cores)


class _SystemSampler:
    """Samples system-wide memory/CPU on a background thread while calibration runs.

    Deliberately system-wide rather than per-process: the calibration case is
    run alone (no other case running concurrently), so the delta attributable
    to it is a reasonable proxy without needing to track the FoamAgent
    subprocess's PID tree. Known limitation: unrelated load on a shared
    machine during calibration will skew the measurement.
    """

    def __init__(self, interval_s: float = 2.0):
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.mem_used_samples_mb: List[float] = []
        self.cpu_percent_samples: List[float] = []

    def _loop(self) -> None:
        psutil.cpu_percent(interval=None)  # prime psutil's internal delta timer
        while not self._stop.is_set():
            time.sleep(self.interval_s)
            vm = psutil.virtual_memory()
            self.mem_used_samples_mb.append((vm.total - vm.available) / (1024 * 1024))
            self.cpu_percent_samples.append(psutil.cpu_percent(interval=None))

    def __enter__(self) -> "_SystemSampler":
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_s * 2)


def benchmark_case(
    run_fn: Callable[[], Any], *, sample_interval_s: float = 2.0
) -> Tuple[Any, ResourceProfile]:
    """Run one calibration case and measure what it actually cost.

    ``run_fn`` must be a blocking call that launches and waits for exactly one
    OpenFOAM case (e.g. ``lambda: FoamAgentRunner(...).run(..., execute=True)``).
    Call this only for the first case of a physics group / study — running it
    alone, with nothing else competing for the machine, is what makes the
    measurement meaningful.

    Returns ``(run_fn()'s return value, ResourceProfile)``.
    """
    baseline = psutil.virtual_memory()
    baseline_used_mb = (baseline.total - baseline.available) / (1024 * 1024)

    sampler = _SystemSampler(interval_s=sample_interval_s)
    t0 = time.monotonic()
    with sampler:
        result = run_fn()
    wall_clock_s = time.monotonic() - t0

    peak_used_mb = max(sampler.mem_used_samples_mb, default=baseline_used_mb)
    avg_cpu = (
        sum(sampler.cpu_percent_samples) / len(sampler.cpu_percent_samples)
        if sampler.cpu_percent_samples
        else 0.0
    )

    profile = ResourceProfile(
        wall_clock_s=wall_clock_s,
        peak_used_mem_mb=max(0.0, peak_used_mb - baseline_used_mb),
        avg_cpu_percent=avg_cpu,
        # Zero samples means the case finished inside one sampling interval —
        # in practice a dict/mesh error that died in seconds, which is the
        # most common first-case outcome. Without this flag it looks like a
        # case that needs ~0 CPU and ~0 memory, and the floors then produce
        # an enormous concurrency limit from what is really a failed
        # measurement. Recorded honestly instead, so the scheduler can decline
        # to derive a limit from it.
        sample_count=len(sampler.cpu_percent_samples),
    )
    return result, profile
