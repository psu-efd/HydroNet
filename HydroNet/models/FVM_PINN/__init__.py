"""Public API for the FVM-PINN model family."""

from .model import FVM_SWE_PINN
from .trainer import FVM_PINNTrainer
from .data import FVM_PINNDataset

__all__ = ["FVM_SWE_PINN", "FVM_PINNTrainer", "FVM_PINNDataset"]
