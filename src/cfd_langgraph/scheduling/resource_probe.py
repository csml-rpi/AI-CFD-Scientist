from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Tuple

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
    avg_cpu_percent: float  # the case's processes, as percent of the whole machine, 0-100
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
    """Samples the CPU and memory of this process's child processes on a
    background thread while calibration runs.

    The calibration case is the only case this study runs at the time, so its
    work is this process's children. It used to be measured system-wide,
    which counted every other study on a shared machine as this case's cost:
    on airfrans_codex_20261009, calibrated while three other studies were
    solving, one candidate build was taken to need a third of the machine and
    the study ran its builds three at a time on a machine 91% idle.
    """

    def __init__(self, interval_s: float = 2.0):
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.mem_used_samples_mb: List[float] = []
        self.cpu_percent_samples: List[float] = []

    @staticmethod
    def _tree() -> Tuple[dict, float]:
        """{pid: CPU seconds so far} and resident MB of this process's children."""
        cpu: dict = {}
        rss_mb = 0.0
        for proc in psutil.Process().children(recursive=True):
            try:
                with proc.oneshot():
                    times = proc.cpu_times()
                    cpu[proc.pid] = times.user + times.system
                    rss_mb += proc.memory_info().rss / (1024 * 1024)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
        return cpu, rss_mb

    def _loop(self) -> None:
        cores = psutil.cpu_count(logical=True) or 1
        last, _ = self._tree()
        t_last = time.monotonic()
        while not self._stop.is_set():
            time.sleep(self.interval_s)
            now_cpu, rss_mb = self._tree()
            now = time.monotonic()
            # A child first seen now started since the last sample: all its
            # CPU time falls in this interval.
            used_s = sum(max(0.0, t - last.get(pid, 0.0)) for pid, t in now_cpu.items())
            busy_cores = used_s / max(now - t_last, 1e-6)
            last, t_last = now_cpu, now
            self.mem_used_samples_mb.append(rss_mb)
            # Kept on the old scale, percent of the whole machine, which
            # ResourceProfile.cores_used converts back to cores.
            self.cpu_percent_samples.append(min(100.0, 100.0 * busy_cores / cores))

    def __enter__(self) -> "_SystemSampler":
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_s * 2)


def _profile_from(sampler: _SystemSampler, baseline_used_mb: float, wall_clock_s: float) -> ResourceProfile:
    mem = list(sampler.mem_used_samples_mb)
    cpu = list(sampler.cpu_percent_samples)
    peak_used_mb = max(mem, default=baseline_used_mb)
    return ResourceProfile(
        wall_clock_s=wall_clock_s,
        peak_used_mem_mb=max(0.0, peak_used_mb - baseline_used_mb),
        avg_cpu_percent=(sum(cpu) / len(cpu)) if cpu else 0.0,
        # Zero samples means the case finished inside one sampling interval —
        # in practice a dict/mesh error that died in seconds, which is the
        # most common first-case outcome. Without this flag it looks like a
        # case that needs ~0 CPU and ~0 memory, and the floors then produce
        # an enormous concurrency limit from what is really a failed
        # measurement. Recorded honestly instead, so the scheduler can decline
        # to derive a limit from it.
        sample_count=len(cpu),
    )


def benchmark_case(
    run_fn: Callable[[], Any], *, sample_interval_s: float = 2.0,
    window_s: Optional[float] = None,
    on_window: Optional[Callable[[ResourceProfile], None]] = None,
) -> Tuple[Any, ResourceProfile]:
    """Run one calibration case and measure what it actually cost.

    ``run_fn`` must be a blocking call that launches and waits for exactly one
    case (e.g. ``lambda: FoamAgentRunner(...).run(..., execute=True)``).
    Call this only for the first case of a physics group / study — running it
    alone, with nothing else competing for the machine, is what makes the
    measurement meaningful.

    With ``window_s`` and ``on_window``, ``on_window`` is called once, from a
    timer thread, with the profile measured over the first ``window_s``
    seconds if the case is still running then -- so a caller can stop holding
    everything else back without waiting for a case that may run for hours.
    It is never called after this function returns.

    Returns ``(run_fn()'s return value, ResourceProfile)``, the profile over
    the whole run.
    """
    # The sampler measures this process's children only, which start at nothing.
    baseline_used_mb = 0.0

    sampler = _SystemSampler(interval_s=sample_interval_s)
    t0 = time.monotonic()
    timer: Optional[threading.Timer] = None
    with sampler:
        if window_s and on_window is not None:
            def _fire() -> None:
                try:
                    on_window(_profile_from(sampler, baseline_used_mb, time.monotonic() - t0))
                except Exception:
                    pass

            timer = threading.Timer(float(window_s), _fire)
            timer.daemon = True
            timer.start()
        try:
            result = run_fn()
        finally:
            if timer is not None:
                timer.cancel()
                timer.join()
    return result, _profile_from(sampler, baseline_used_mb, time.monotonic() - t0)
