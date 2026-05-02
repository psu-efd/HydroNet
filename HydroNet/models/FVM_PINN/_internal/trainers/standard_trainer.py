"""
Standard (full-batch) FVM-PINN trainer.

Baseline strategy: all cells are used at every training step.
Gradient checkpointing is off by default but can be enabled via
BaseTrainerConfig.use_grad_checkpoint = True.
"""

from dataclasses import dataclass
from typing import Optional

import torch

from .base_trainer import BaseTrainer, BaseTrainerConfig


@dataclass
class StandardTrainerConfig(BaseTrainerConfig):
    """
    Config for the standard full-batch trainer.

    No additional parameters beyond the base config.
    Set use_grad_checkpoint=True (inherited) to enable Strategy 3.
    """
    pass


class StandardTrainer(BaseTrainer):
    """
    Full-batch FVM-PINN trainer (baseline).

    All n_cells are evaluated at every step. This gives the most accurate
    gradient estimate but has the highest memory footprint and per-step cost.

    Gradient checkpointing (Strategy 3) can be activated by setting
    `use_grad_checkpoint=True` in the config.
    """

    def _make_cell_mask(self) -> Optional[torch.Tensor]:
        """Return None — use all cells (full batch)."""
        return None
