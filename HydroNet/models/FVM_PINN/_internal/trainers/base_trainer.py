"""
Base trainer for FVM-PINN.

Provides the shared training loop (Adam + L-BFGS), logging, checkpointing,
and input-normalisation setup. Subclasses override `_make_cell_mask()` to
implement different cell-sampling strategies.

All trainers share the same public API:
    trainer = XxxTrainer(trainer_cfg, net_cfg, loss_cfg, mesh_data)
    trainer.setup(ic_data, bc_data, ref_data)
    trainer.train()
    U = trainer.predict(xyt)
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim

from ..pinn.network import SWENet, NetworkConfig
from ..pinn.loss import FVMPINNLoss, LossConfig
from .memory_tracker import MemoryTracker

logger = logging.getLogger(__name__)


@dataclass
class BaseTrainerConfig:
    """
    Shared hyperparameters for all trainer variants.

    Strategy flags
    --------------
    use_grad_checkpoint : bool
        Strategy 3 — recompute activations in backward to save memory.
        Applies to all trainers; costs ~20% extra compute.
    """
    # Adam phase
    adam_epochs: int = 5000
    adam_lr: float = 1e-3
    adam_lr_decay: float = 0.9
    adam_decay_every: int = 1000

    # L-BFGS refinement phase
    lbfgs_epochs: int = 500
    lbfgs_max_iter: int = 20
    lbfgs_lr: float = 1.0

    # Time sampling (per step)
    n_time_samples: int = 8
    t_start: float = 0.0
    t_end: float = 1.0

    # Strategy 3: gradient checkpointing
    use_grad_checkpoint: bool = False

    # Logging / checkpointing
    log_every: int = 200
    checkpoint_every: int = 1000
    output_dir: str = "outputs"

    # Hardware
    device: str = "cpu"
    dtype: str = "float64"


class BaseTrainer(ABC):
    """
    Abstract FVM-PINN trainer.

    Subclasses must implement `_make_cell_mask()` to control which cells
    contribute to the FVM physics loss at each training step.

    Full-batch trainers return None (all cells).
    Mini-batch trainers return a boolean mask over cells.
    """

    def __init__(
        self,
        trainer_cfg: BaseTrainerConfig,
        net_cfg: NetworkConfig,
        loss_cfg: LossConfig,
        mesh_data: Dict[str, torch.Tensor],
        bc_config: Optional[Dict] = None,
    ) -> None:
        self.cfg = trainer_cfg
        self.device = torch.device(trainer_cfg.device)
        self.dtype = torch.float64 if trainer_cfg.dtype == "float64" else torch.float32

        # Move mesh tensors to device/dtype
        self.mesh_data = _to_device(mesh_data, self.device, self.dtype)

        # Propagate gradient checkpointing flag to loss config
        loss_cfg.use_grad_checkpoint = trainer_cfg.use_grad_checkpoint

        # Build network and loss
        # h_still: still water reference for perturbation form (default: zeros)
        h_still = mesh_data.get("h_still", None)
        self.network = SWENet(net_cfg).to(self.device).to(self.dtype)
        self.loss_fn = FVMPINNLoss(loss_cfg, self.mesh_data, h_still, bc_config)

        # Training data (set by setup())
        self.ic_data: Optional[Dict] = None
        self.bc_data: Optional[Dict] = None
        self.ref_data: Optional[Dict] = None

        # History — list of scalar values, one per gradient step.
        # Core keys always present; per-BC keys added dynamically as they
        # appear in bc_data. Per-SWE-component keys (``*_xi`` / ``*_hu`` /
        # ``*_hv``) are tracked for each elementwise loss (FVM residual, IC,
        # data) so the user can inspect scale imbalances across the
        # conserved variables.
        core_keys = (
            "total",
            "fvm", "fvm_xi", "fvm_hu", "fvm_hv",
            "ic",  "ic_xi",  "ic_hu",  "ic_hv",
            "bc",
            "data", "data_xi", "data_hu", "data_hv",
        )
        self.history: Dict[str, List[float]] = {k: [] for k in core_keys}

        self.output_dir = Path(trainer_cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Optional GPU memory tracker; attached externally by the
        # FVM_PINNTrainer wrapper or by TimeWindowTrainer (shared across
        # per-window trainers). None by default so non-instrumented callers
        # see no behavioural change.
        self.memory_tracker: Optional[MemoryTracker] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def setup(
        self,
        ic_data: Dict,
        bc_data: Optional[Dict] = None,
        ref_data: Optional[Dict] = None,
    ) -> None:
        """
        Set training data and configure input normalisation.

        Must be called before train().

        Parameters
        ----------
        ic_data  : {xyt: [n, 3], U_true: [n, 3]}
        bc_data  : {bc_id: {type, xyt, value, nx, ny}}
        ref_data : {xyt: [n, 3], U_ref: [n, 3]}
        """
        self.ic_data = _to_device(ic_data, self.device, self.dtype)

        if bc_data is not None:
            self.bc_data = {
                k: _to_device(v, self.device, self.dtype) if isinstance(v, dict) else v
                for k, v in bc_data.items()
            }

        if ref_data is not None:
            self.ref_data = _to_device(ref_data, self.device, self.dtype)

        self._set_normalisation()
        logger.info(
            f"{self.__class__.__name__} ready: "
            f"{self.mesh_data['n_cells']} cells, "
            f"t=[{self.cfg.t_start:.2f}, {self.cfg.t_end:.2f}]"
        )

    def train(self) -> float:
        """
        Run full training (Adam + L-BFGS).

        Returns
        -------
        float : final total loss value
        """
        logger.info(f"=== {self.__class__.__name__} training start ===")
        t0 = time.perf_counter()

        self._adam_phase()
        self._lbfgs_phase()

        elapsed = time.perf_counter() - t0
        final_loss = self.history["total"][-1] if self.history["total"] else float("nan")
        logger.info(f"Training complete: {elapsed:.1f}s, final loss={final_loss:.4e}")
        self._save_checkpoint("final")
        self.plot_loss_history()
        return final_loss

    def predict(self, xyt: torch.Tensor) -> torch.Tensor:
        """Evaluate network at arbitrary (x, y, t) points — no grad."""
        self.network.eval()
        with torch.no_grad():
            xyt = xyt.to(self.device, dtype=self.dtype)
            return self.network(xyt).detach()

    def load_checkpoint(self, path: str) -> None:
        ck = torch.load(path, map_location=self.device)
        self.network.load_state_dict(ck["network_state"])
        self.network.set_normalisation(ck["x_mean"], ck["x_std"])
        if "history" in ck:
            self.history = ck["history"]
        logger.info(f"Checkpoint loaded: {path}")

    # ------------------------------------------------------------------
    # Abstract hook: subclasses define the sampling strategy
    # ------------------------------------------------------------------

    @abstractmethod
    def _make_cell_mask(self) -> Optional[torch.Tensor]:
        """
        Return a boolean tensor of shape [n_cells] selecting which cells
        contribute to the FVM residual this step.

        Return None to use all cells (full-batch mode).
        """

    # ------------------------------------------------------------------
    # Shared training loop
    # ------------------------------------------------------------------

    def _adam_phase(self) -> None:
        if self.cfg.adam_epochs <= 0:
            return
        logger.info(f"Adam: {self.cfg.adam_epochs} epochs, lr={self.cfg.adam_lr}")
        if self.memory_tracker is not None:
            self.memory_tracker.reset_peak("adam-start")
            self.memory_tracker.sample("adam", 0)
        optimizer = optim.Adam(self.network.parameters(), lr=self.cfg.adam_lr)
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, self.cfg.adam_decay_every, self.cfg.adam_lr_decay
        )
        for epoch in range(self.cfg.adam_epochs):
            optimizer.zero_grad()
            losses = self._step()
            losses["total"].backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            self._record(losses)
            if (epoch + 1) % self.cfg.log_every == 0:
                lr = optimizer.param_groups[0]["lr"]
                self._log(f"Adam [{epoch+1}/{self.cfg.adam_epochs}]", losses, lr)
                if self.memory_tracker is not None:
                    self.memory_tracker.sample("adam", epoch + 1)
            if (epoch + 1) % self.cfg.checkpoint_every == 0:
                self._save_checkpoint(f"adam_{epoch+1}")

    def _lbfgs_phase(self) -> None:
        if self.cfg.lbfgs_epochs <= 0:
            return
        logger.info(f"L-BFGS: {self.cfg.lbfgs_epochs} steps")
        if self.memory_tracker is not None:
            self.memory_tracker.reset_peak("lbfgs-start")
            self.memory_tracker.sample("lbfgs", 0)
        optimizer = optim.LBFGS(
            self.network.parameters(),
            lr=self.cfg.lbfgs_lr,
            max_iter=self.cfg.lbfgs_max_iter,
            history_size=50,
            line_search_fn="strong_wolfe",
        )
        counter = [0]

        def closure():
            optimizer.zero_grad()
            losses = self._step()
            losses["total"].backward()
            self._record(losses)
            counter[0] += 1
            if counter[0] % self.cfg.log_every == 0:
                self._log(f"L-BFGS [{counter[0]}]", losses)
                if self.memory_tracker is not None:
                    self.memory_tracker.sample("lbfgs", counter[0])
            return losses["total"]

        for _ in range(self.cfg.lbfgs_epochs):
            optimizer.step(closure)

    def _step(self) -> Dict[str, torch.Tensor]:
        """Single training step: sample times + cell mask, compute losses."""
        self.network.train()
        t = self._sample_times()
        cell_mask = self._make_cell_mask()
        return self.loss_fn(
            self.network, t,
            ic_data=self.ic_data,
            bc_data=self.bc_data,
            ref_data=self.ref_data,
            cell_mask=cell_mask,
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _sample_times(self) -> torch.Tensor:
        t = torch.rand(self.cfg.n_time_samples, device=self.device, dtype=self.dtype)
        t = self.cfg.t_start + t * (self.cfg.t_end - self.cfg.t_start)
        return t.requires_grad_(True)

    def _set_normalisation(self) -> None:
        cell_xy = self.mesh_data["cell_center"]
        ic_xy   = self.ic_data["xyt"][:, :2]
        all_xy  = torch.cat([cell_xy, ic_xy], dim=0)

        x_mean_2d = all_xy.mean(0)
        x_std_2d  = all_xy.std(0).clamp(min=1e-6)

        t_range = max(self.cfg.t_end - self.cfg.t_start, 1e-6)
        t_mid   = 0.5 * (self.cfg.t_start + self.cfg.t_end)
        t_mean  = torch.tensor(t_mid,   device=self.device, dtype=self.dtype)
        t_std   = torch.tensor(t_range, device=self.device, dtype=self.dtype)

        x_mean = torch.cat([x_mean_2d, t_mean.unsqueeze(0)])
        x_std  = torch.cat([x_std_2d,  t_std.unsqueeze(0)])
        self.network.set_normalisation(x_mean, x_std)

    def _record(self, losses: Dict[str, torch.Tensor]) -> None:
        for k, v in losses.items():
            if isinstance(v, torch.Tensor):
                if k not in self.history:
                    self.history[k] = []
                self.history[k].append(v.item())

    def _log(
        self,
        prefix: str,
        losses: Dict[str, torch.Tensor],
        lr: Optional[float] = None,
    ) -> None:
        parts = [
            f"total={losses['total'].item():.3e}",
            f"fvm={losses['fvm'].item():.3e}",
            f"ic={losses['ic'].item():.3e}",
            f"bc={losses['bc'].item():.3e}",
        ]
        if losses.get("data", None) is not None and losses["data"].item() > 0:
            parts.append(f"data={losses['data'].item():.3e}")
        if lr is not None:
            parts.append(f"lr={lr:.2e}")
        logger.info(f"{prefix}  " + "  ".join(parts))

        # Detailed log at lower frequency (every 5x log_every)
        step_count = len(self.history.get("total", []))
        if step_count % (self.cfg.log_every * 5) == 0:
            detail_parts = []
            for k in ("fvm_xi", "fvm_hu", "fvm_hv",
                      "ic_xi", "ic_hu", "ic_hv",
                      "data_xi", "data_hu", "data_hv"):
                if k in losses and losses[k].item() > 0:
                    detail_parts.append(f"{k}={losses[k].item():.3e}")
            for k, v in losses.items():
                if k.startswith("bc_") and isinstance(v, torch.Tensor):
                    detail_parts.append(f"{k}={v.item():.3e}")
            if detail_parts:
                logger.info(f"  detail: " + "  ".join(detail_parts))

    def _save_checkpoint(self, tag: str) -> None:
        path = self.output_dir / f"ckpt_{tag}.pt"
        torch.save({
            "network_state": self.network.state_dict(),
            "history": self.history,
            "x_mean": self.network.x_mean,
            "x_std": self.network.x_std,
        }, path)
        logger.debug(f"Checkpoint: {path}")

    def plot_loss_history(self, save_path: Optional[str] = None) -> None:
        """Plot detailed loss history with per-equation and per-BC breakdowns."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available; skipping loss plots")
            return

        if not self.history.get("total"):
            return

        save_path = save_path or str(self.output_dir / "loss_history_detail.png")

        # Determine which keys have data
        fvm_eq_keys = [k for k in ("fvm_xi", "fvm_hu", "fvm_hv") if self.history.get(k)]
        bc_keys = [k for k in self.history if k.startswith("bc_") and self.history[k]]

        n_plots = 2 + (1 if fvm_eq_keys else 0) + (1 if bc_keys else 0)
        fig, axes = plt.subplots(n_plots, 1, figsize=(12, 4 * n_plots), sharex=True)
        if n_plots == 1:
            axes = [axes]

        ax_idx = 0

        # Panel 1: Total loss and weighted components
        ax = axes[ax_idx]
        steps = range(len(self.history["total"]))
        ax.semilogy(steps, self.history["total"], "k-", linewidth=2, label="total", alpha=0.8)
        ax.set_ylabel("Loss")
        ax.set_title("Total Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax_idx += 1

        # Panel 2: Raw (unweighted) loss components
        ax = axes[ax_idx]
        for key, color, label in [
            ("fvm", "C0", "FVM (physics)"),
            ("ic", "C1", "IC"),
            ("bc", "C2", "BC"),
            ("data", "C3", "Data"),
        ]:
            vals = self.history.get(key, [])
            if vals and any(v > 0 for v in vals):
                ax.semilogy(range(len(vals)), vals, color=color, label=label, alpha=0.8)
        ax.set_ylabel("Raw Loss (unweighted)")
        ax.set_title("Loss Components (raw, before weighting)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax_idx += 1

        # Panel 3: FVM per-equation breakdown
        if fvm_eq_keys:
            ax = axes[ax_idx]
            labels = {"fvm_xi": "Continuity (xi)", "fvm_hu": "x-Momentum (hu)", "fvm_hv": "y-Momentum (hv)"}
            for key, color in zip(["fvm_xi", "fvm_hu", "fvm_hv"], ["C0", "C1", "C2"]):
                vals = self.history.get(key, [])
                if vals:
                    ax.semilogy(range(len(vals)), vals, color=color, label=labels.get(key, key), alpha=0.8)
            ax.set_ylabel("FVM Loss per Equation")
            ax.set_title("FVM Residual: Per-Equation Breakdown")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax_idx += 1

        # Panel 4: BC per-boundary breakdown
        if bc_keys:
            ax = axes[ax_idx]
            for i, key in enumerate(sorted(bc_keys)):
                vals = self.history[key]
                if vals and any(v > 0 for v in vals):
                    ax.semilogy(range(len(vals)), vals, f"C{i}", label=key, alpha=0.8)
            ax.set_ylabel("BC Loss per Boundary")
            ax.set_title("BC Loss: Per-Boundary Breakdown")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax_idx += 1

        axes[-1].set_xlabel("Training Step")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        logger.info(f"Loss history plot: {save_path}")


# ---------------------------------------------------------------------------
# Shared utility
# ---------------------------------------------------------------------------

def _to_device(
    d: Dict, device: torch.device, dtype: torch.dtype
) -> Dict:
    """Move all float tensors in dict to device/dtype; int tensors to device only."""
    result = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            result[k] = v.to(device, dtype=dtype) if v.is_floating_point() else v.to(device)
        else:
            result[k] = v
    return result
