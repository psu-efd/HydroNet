"""
Wall-clock seconds per training step, reported at each progress log.

A long CPU run can degrade by one to two orders of magnitude while the model
itself is unchanged (OpenMP thread-pool contention; BIC_E/BIC_F, 2026-10-22).
Logging s/step makes that visible within one log interval, and a WARNING is
emitted once a phase runs slower than ``slow_factor`` x its best rate.
"""

import logging
import time

logger = logging.getLogger(__name__)


class StepTimer:
    def __init__(self, slow_factor: float = 3.0) -> None:
        self.slow_factor = slow_factor
        self.start(0)

    def start(self, step: int) -> None:
        """Begin a phase (Adam, L-BFGS) at ``step``; resets the best rate."""
        self._t = time.perf_counter()
        self._step = step
        self._best = float("inf")

    def tag(self, step: int) -> str:
        """Return ' (X.XXX s/step)' for the steps since the previous call."""
        now = time.perf_counter()
        n = step - self._step
        dt = now - self._t
        self._t, self._step = now, step
        if n <= 0:
            return ""
        sps = dt / n
        self._best = min(self._best, sps)
        if sps > self.slow_factor * self._best:
            logger.warning(
                f"Step time {sps:.3f} s is {sps / self._best:.1f}x the best in this "
                f"phase ({self._best:.3f} s) - check CPU thread contention / node load"
            )
        return f" ({sps:.3f} s/step)"
