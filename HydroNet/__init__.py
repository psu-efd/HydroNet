"""
HydroNet: Physics-informed (PI) machine learning for hydrodynamics and hydraulics modeling. DeepONet, PINN, and PI-DeepONet are supported.
"""

# Version
from .__about__ import __version__

# Models
from .models.DeepONet.model import SWE_DeepONetModel
from .models.PINN.model import SWE_PINN
from .models.PI_DeepONet.model import PI_DeepONetModel

# Trainers
from .models.DeepONet.trainer import SWE_DeepONetTrainer
from .models.PINN.trainer import PINNTrainer
from .models.PI_DeepONet.trainer import PI_DeepONetTrainer

# Datasets
from .data.PINN_dataset import PINNDataset
from .data.DeepONet_dataset import SWE_DeepONetDataset

# Data loaders
from .data.PINN_dataset import get_pinn_dataloader
from .data.DeepONet_dataset import create_deeponet_dataloader

# Utilities
from .utils.config import Config
from .utils.gmsh_to_points import gmsh2D_to_points
from .utils.predict_on_gmsh import predict_on_gmsh2d_mesh

# Define public API
__all__ = [
    # Version
    "__version__",
    
    # Models
    "SWE_DeepONetModel",
    "SWE_PINN",
    "PI_DeepONetModel",
    
    # Trainers
    "SWE_DeepONetTrainer",
    "PINNTrainer",
    "PI_DeepONetTrainer",
    
    # Datasets
    "PINNDataset",
    "SWE_DeepONetDataset",
    
    # Data loaders
    "get_pinn_dataloader",
    "create_deeponet_dataloader",
    
    # Utilities
    "Config",
    "gmsh2D_to_points",
    "predict_on_gmsh2d_mesh",
    "plot_solution",
    "plot_loss_history",
]
