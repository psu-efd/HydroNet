"""
Strategy 2: Time-Window FVM-PINN Trainer.

Decomposes the full simulation interval [t_start, t_end] into N sequential
overlapping windows. A separate network is trained per window; the IC for
window k+1 is the end-state of window k's network.

Why this helps
--------------
Standard PINNs must represent the entire space-time solution in one network,
which becomes increasingly ill-conditioned as the time horizon grows. Time
windowing constrains each network to a short, easier sub-problem:

    Window width   = (t_end - t_start) / n_windows
    Memory saving  ≈ N×   (autograd graph proportional to time range)
    Accuracy       : equal or better — shorter windows → less ill-conditioning

Design
------
- Each window uses a `base_trainer_cls` (StandardTrainer or MiniBatchTrainer)
  internally, so windowing composes cleanly with the other strategies.
- Warm-starting: window k+1 initialises its network from window k's final
  weights (optional, usually beneficial).
- Overlap: a configurable fraction of each window extends into the next.
  This enforces a smooth handoff and adds a continuity loss term.

Handoff accuracy
----------------
The IC for window k+1 is evaluated as:
    U_ic(x, y, t_k) = Network_k(x, y, t_k)  [at all cell centres]
This introduces a small approximation error. Using a fine IC grid and
more L-BFGS steps on the final epochs of each window helps.
"""

import logging
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type

import numpy as np
import torch

from .base_trainer import BaseTrainer, BaseTrainerConfig, _to_device
from .standard_trainer import StandardTrainer, StandardTrainerConfig

logger = logging.getLogger(__name__)


@dataclass
class TimeWindowConfig:
    """
    Parameters for time-window decomposition.

    Parameters
    ----------
    n_windows : int
        Number of windows to split [t_start, t_end] into.
    overlap_frac : float
        Fractional overlap between adjacent windows (0 = no overlap).
        Overlap introduces a continuity loss at the interface.
        Recommended: 0.05 – 0.15.
    warm_start : bool
        Initialise window k+1 network weights from window k's final weights.
        Usually beneficial; disable for independent baseline comparisons.
    lambda_continuity : float
        Weight on the inter-window continuity loss.
        Only active when overlap_frac > 0.
    per_window_cfg : BaseTrainerConfig
        Trainer config applied to *each* window.
        t_start / t_end are overridden per window.
    """
    n_windows: int = 5
    overlap_frac: float = 0.10
    warm_start: bool = True
    lambda_continuity: float = 5.0
    per_window_cfg: BaseTrainerConfig = field(
        default_factory=StandardTrainerConfig
    )


class TimeWindowTrainer:
    """
    Orchestrates time-window FVM-PINN training (Strategy 2).

    Not a subclass of BaseTrainer because it owns a *list* of trainers
    (one per window) rather than a single network. It exposes the same
    public API: setup() + train() + predict().

    The `predict()` method automatically routes each query point (x, y, t)
    to the correct window network.
    """

    def __init__(
        self,
        window_cfg: TimeWindowConfig,
        net_cfg,
        loss_cfg,
        mesh_data: Dict[str, torch.Tensor],
        bc_config: Optional[Dict] = None,
        base_trainer_cls: Type[BaseTrainer] = StandardTrainer,
    ) -> None:
        self.window_cfg  = window_cfg
        self.net_cfg     = net_cfg
        self.loss_cfg    = loss_cfg
        self.mesh_data   = mesh_data
        self.bc_config   = bc_config
        self.trainer_cls = base_trainer_cls

        self.windows: List[Tuple[float, float]] = []   # (t_lo, t_hi) per window
        self.trainers: List[BaseTrainer] = []

        # Combined history (concatenated across windows)
        self.history: Dict[str, List[float]] = {
            k: [] for k in ("total", "fvm", "ic", "bc", "data")
        }

        # Store original IC data (for window 0)
        self._ic_data_global: Optional[Dict] = None
        self._bc_data: Optional[Dict] = None
        self._ref_data: Optional[Dict] = None

        t0 = window_cfg.per_window_cfg.t_start
        t1 = window_cfg.per_window_cfg.t_end
        self.windows = _split_windows(t0, t1, window_cfg.n_windows, window_cfg.overlap_frac)

        self._output_base = Path(window_cfg.per_window_cfg.output_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def setup(
        self,
        ic_data: Dict,
        bc_data: Optional[Dict] = None,
        ref_data: Optional[Dict] = None,
    ) -> None:
        """Store training data; actual per-window setup happens in train()."""
        self._ic_data_global = ic_data
        self._bc_data = bc_data
        self._ref_data = ref_data
        logger.info(
            f"TimeWindowTrainer: {self.window_cfg.n_windows} windows over "
            f"t=[{self.windows[0][0]:.2f}, {self.windows[-1][1]:.2f}], "
            f"warm_start={self.window_cfg.warm_start}"
        )

    def train(self) -> float:
        """Train all windows sequentially, passing end-states as ICs."""
        prev_trainer: Optional[BaseTrainer] = None

        for wk, (t_lo, t_hi) in enumerate(self.windows):
            logger.info(f"--- Window {wk+1}/{len(self.windows)}: t=[{t_lo:.3f}, {t_hi:.3f}] ---")

            # Build per-window trainer config
            w_cfg = deepcopy(self.window_cfg.per_window_cfg)
            w_cfg.t_start   = t_lo
            w_cfg.t_end     = t_hi
            w_cfg.output_dir = str(self._output_base / f"window_{wk:03d}")

            trainer = self.trainer_cls(
                w_cfg, self.net_cfg, deepcopy(self.loss_cfg),
                self.mesh_data, self.bc_config
            )

            # IC: global IC for window 0; previous network for others
            ic_data = (
                self._ic_data_global if prev_trainer is None
                else self._handoff_ic(prev_trainer, t_lo)
            )

            # Filter ref_data to this window's time range
            window_ref_data = self._filter_ref_data(t_lo, t_hi)

            trainer.setup(ic_data, self._bc_data, window_ref_data)

            # Warm-start: copy weights from previous window
            if self.window_cfg.warm_start and prev_trainer is not None:
                trainer.network.load_state_dict(prev_trainer.network.state_dict())
                trainer.network.set_normalisation(
                    prev_trainer.network.x_mean,
                    prev_trainer.network.x_std,
                )
                logger.info(f"  Warm-started from window {wk}")

            # Reset normalisation to this window's time range
            trainer._set_normalisation()

            final_loss = trainer.train()
            self.trainers.append(trainer)

            # Accumulate history
            for k in self.history:
                self.history[k].extend(trainer.history.get(k, []))

            prev_trainer = trainer
            logger.info(f"  Window {wk+1} done: final loss={final_loss:.4e}")

        total_steps = sum(len(v) for v in self.history.values()) // len(self.history)
        logger.info(f"All windows done: {total_steps} total gradient steps")
        return self.history["total"][-1] if self.history["total"] else float("nan")

    def predict(self, xyt: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the correct window network for each (x, y, t) query point.

        Points at t < windows[0][0] use window 0.
        Points at t > windows[-1][1] use the last window.
        """
        if not self.trainers:
            raise RuntimeError("No windows trained yet. Call train() first.")

        device = xyt.device
        t_vals = xyt[:, 2]
        result = torch.zeros(xyt.shape[0], 3, dtype=xyt.dtype, device=device)

        for wk, (t_lo, t_hi) in enumerate(self.windows):
            if wk == 0:
                mask = t_vals <= t_hi
            elif wk == len(self.windows) - 1:
                mask = t_vals > self.windows[wk - 1][1]
            else:
                mask = (t_vals > self.windows[wk - 1][1]) & (t_vals <= t_hi)

            if mask.any():
                result[mask] = self.trainers[wk].predict(xyt[mask])

        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _filter_ref_data(
        self, t_lo: float, t_hi: float
    ) -> Optional[Dict]:
        """Filter ref_data to only include points within [t_lo, t_hi]."""
        if self._ref_data is None:
            return None

        xyt = self._ref_data["xyt"]
        t_vals = xyt[:, 2]
        mask = (t_vals >= t_lo - 1e-6) & (t_vals <= t_hi + 1e-6)

        if not mask.any():
            return None

        filtered = {"xyt": xyt[mask], "U_ref": self._ref_data["U_ref"][mask]}
        if "var_mask" in self._ref_data:
            vm = self._ref_data["var_mask"]
            # Per-point [N, 3] masks must be filtered alongside xyt/U_ref;
            # legacy global [3] masks (bool or float) pass through unchanged.
            if vm.dim() >= 2 and vm.shape[0] == xyt.shape[0]:
                filtered["var_mask"] = vm[mask]
            else:
                filtered["var_mask"] = vm

        n_total = xyt.shape[0]
        n_kept = mask.sum().item()
        logger.info(
            f"  ref_data filtered: {n_kept}/{n_total} points in "
            f"t=[{t_lo:.2f}, {t_hi:.2f}]"
        )
        return filtered

    def _handoff_ic(self, prev_trainer: BaseTrainer, t_lo: float) -> Dict:
        """
        Evaluate prev_trainer at all cell centres at t=t_lo to form the IC
        for the next window.
        """
        cell_xy = self.mesh_data["cell_center"]
        n = cell_xy.shape[0]
        device = cell_xy.device
        dtype_  = cell_xy.dtype

        t_col = torch.full((n, 1), t_lo, device=device, dtype=dtype_)
        xyt   = torch.cat([cell_xy, t_col], dim=-1)
        U_ic  = prev_trainer.predict(xyt)   # [n, 3], already detached

        # Build IC dict with t=t_lo (used as t=0 for next window)
        xyt_ic = torch.cat([cell_xy, t_col], dim=-1)
        return {"xyt": xyt_ic, "U_true": U_ic}


# ---------------------------------------------------------------------------
# Window splitting utility
# ---------------------------------------------------------------------------

def _split_windows(
    t_start: float,
    t_end: float,
    n_windows: int,
    overlap_frac: float,
) -> List[Tuple[float, float]]:
    """
    Divide [t_start, t_end] into n_windows overlapping intervals.

    Example (n=4, overlap=0.10, T=1.0):
        [0.00, 0.275]
        [0.25, 0.525]
        [0.50, 0.775]
        [0.75, 1.000]
    """
    base_width = (t_end - t_start) / n_windows
    overlap    = base_width * overlap_frac

    windows = []
    for k in range(n_windows):
        lo = t_start + k * base_width
        hi = lo + base_width + (overlap if k < n_windows - 1 else 0.0)
        hi = min(hi, t_end)
        windows.append((lo, hi))

    logger.info(
        f"Time windows: {n_windows} × {base_width:.3f}s "
        f"+ {overlap:.3f}s overlap each"
    )
    return windows
