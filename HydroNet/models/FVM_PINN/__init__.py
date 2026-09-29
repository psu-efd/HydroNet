"""Public API for the FVM-PINN model family."""

from .model import FVM_SWE_PINN
from .trainer import FVM_PINNTrainer
from .data import FVM_PINNDataset
from .conservation import mass_balance_report, plot_mass_balance

__all__ = ["FVM_SWE_PINN", "FVM_PINNTrainer", "FVM_PINNDataset",
           "mass_balance_report", "plot_mass_balance"]
