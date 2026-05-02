"""
Strategy 1: Cell Mini-Batch FVM-PINN Trainer.

At each training step, a random subset of cells is selected. The FVM residual
is computed only for those cells (using their 1-ring stencil for flux accuracy).
This reduces:
  - Network forward-pass size: O(n_sample + n_neighbors)  vs  O(n_cells)
  - Autograd graph size:        proportional to forward-pass size
  - Effective memory per step:  ~sample_frac × full-batch memory

Trade-off: gradient estimates are noisier (stochastic), which may slow
convergence slightly but allows training on much larger meshes.

Can be combined with gradient checkpointing (Strategy 3) by setting
use_grad_checkpoint=True in the config for multiplicative memory savings.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import torch

from .base_trainer import BaseTrainer, BaseTrainerConfig

logger = logging.getLogger(__name__)


@dataclass
class MiniBatchTrainerConfig(BaseTrainerConfig):
    """
    Config for the cell mini-batch trainer.

    Parameters
    ----------
    cell_sample_frac : float
        Fraction of cells to sample per step (0 < frac <= 1).
        Typical values: 0.10 – 0.25.
        Lower → less memory, noisier gradients.
    sampling : str
        "uniform"    — random uniform sampling (default, simplest)
        "importance" — sample proportional to |residual| from previous step
                       (more accurate, requires residual cache)
    """
    cell_sample_frac: float = 0.20
    sampling: str = "uniform"


class MiniBatchTrainer(BaseTrainer):
    """
    Cell mini-batch FVM-PINN trainer (Strategy 1).

    Overrides `_make_cell_mask` to return a random boolean mask selecting
    `cell_sample_frac × n_cells` cells per step.

    The loss module (_build_stencil in loss.py) automatically expands the
    mask to include 1-ring face-neighbors so that Roe fluxes are computed
    exactly — the stencil approach ensures no approximation is introduced
    at stencil boundaries.

    Memory scaling
    --------------
    Full-batch memory per step: M_full
    Mini-batch memory per step: ≈ M_full × (frac + avg_neighbors/n_cells)
                                ≈ M_full × (frac + 3*frac)   [triangular mesh]
                                ≈ 4 × frac × M_full

    For frac=0.20 and a triangular mesh:  ~0.80 × M_full  (modest on small meshes)
    For frac=0.05 on a 50k-cell mesh:     ~0.20 × M_full  (significant saving)
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        cfg = self.cfg
        if not isinstance(cfg, MiniBatchTrainerConfig):
            raise TypeError("MiniBatchTrainer requires MiniBatchTrainerConfig")
        self._n_sample = max(
            1,
            int(cfg.cell_sample_frac * self.mesh_data["n_cells"])
        )
        logger.info(
            f"MiniBatchTrainer: sampling {self._n_sample}/{self.mesh_data['n_cells']} "
            f"cells/step ({cfg.cell_sample_frac:.0%}), "
            f"grad_checkpoint={cfg.use_grad_checkpoint}"
        )

        # For importance sampling: cache of per-cell residual magnitudes
        self._residual_cache: Optional[torch.Tensor] = None

    def _make_cell_mask(self) -> torch.Tensor:
        """
        Return a boolean mask with `_n_sample` True entries.

        Sampling modes
        --------------
        uniform    : each cell has equal probability 1/n_cells
        importance : cells with larger cached residuals are sampled more often
        """
        n = self.mesh_data["n_cells"]
        cfg: MiniBatchTrainerConfig = self.cfg

        if cfg.sampling == "importance" and self._residual_cache is not None:
            weights = self._residual_cache.clamp(min=1e-8)
            idx = torch.multinomial(weights, self._n_sample, replacement=False)
        else:
            idx = torch.randperm(n, device=self.device)[: self._n_sample]

        mask = torch.zeros(n, dtype=torch.bool, device=self.device)
        mask[idx] = True
        return mask
