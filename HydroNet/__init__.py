"""
HydroNet: A Physics-Informed DeepONet Framework for Computational Hydraulics

This package implements a physics-informed DeepONet framework for computational hydraulics,
focusing on learning the operator of shallow water equations (SWEs).

The framework has three main components:
1. DeepONet: Implementation of the unstacked DeepONet architecture
2. PINN: Physics-Informed Neural Network for 2D shallow water equations
3. PI-DeepONet: Physics-Informed DeepONet combining DeepONet with physics constraints
"""

# Import main components
from .models.DeepONet.model import *
from .models.PINN.model import *
from .models.PI_DeepONet.model import *

# Import trainers
from .models.DeepONet.trainer import *
from .models.PINN.trainer import *
from .models.PI_DeepONet.trainer import *

# Import datasets
from .data import *

# Import utilities
from .utils.config import *
from .utils.visualization import * 
from .utils.gmsh_to_points import *
from .utils.predict_on_gmsh import *

from .__about__ import __version__

__all__ = [
    "DeepONetModel",
    "SWE_PINN",
    "PI_DeepONetModel",
    "DeepONetTrainer",
    "PINNTrainer",
    "HydraulicDataset",
    "PINNDataset",
    "get_hydraulic_dataloader",
    "get_pinn_dataloader",
    "Config",
    "gmsh2D_to_points",
    "predict_on_gmsh2d_mesh",
]
