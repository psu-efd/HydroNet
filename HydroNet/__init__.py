"""
HydroNet: Physics-informed (PI) machine learning for hydrodynamics and hydraulics modeling. PINN and PI-DeepONet are supported.
"""

# Version
from .__about__ import __version__

# Models
from .models.PINN.model import SWE_PINN
from .models.PI_DeepONet.model import PI_SWE_DeepONetModel

# Trainers
from .models.PINN.trainer import PINNTrainer
from .models.PI_DeepONet.trainer import PI_SWE_DeepONetTrainer

# Datasets
from .models.PINN.data import PINNDataset
from .models.PI_DeepONet.data import PI_SWE_DeepONetDataset

# Data loaders
from .models.PINN.data import get_pinn_dataloader
from .models.PI_DeepONet.data import create_pi_deeponet_dataloader

# Utilities
from .utils.config import Config
from .utils.gmsh_to_points import gmsh2D_to_points
from .utils.predict_on_gmsh import predict_on_gmsh2d_mesh
from .utils.pi_deeponet_utils import (
    pi_deeponet_train,
    pi_deeponet_test,
    pi_deeponet_plot_training_history,
    pi_deeponet_convert_test_results_to_vtk,
    pi_deeponet_application_create_model_application_dataset,
    pi_deeponet_application_run_model_application,
    pi_deeponet_application_compare_model_application_results_with_simulation_results
)

# Define public API
__all__ = [
    # Version
    "__version__",
    
    # Models
    "SWE_PINN",
    "PI_SWE_DeepONetModel",
    
    # Trainers
    "PINNTrainer",
    "PI_SWE_DeepONetTrainer",
    
    # Datasets
    "PINNDataset",
    "PI_SWE_DeepONetDataset",
    
    # Data loaders
    "get_pinn_dataloader",
    "create_pi_deeponet_dataloader",
    
    # Utilities
    "Config",
    "gmsh2D_to_points",
    "predict_on_gmsh2d_mesh",
    "plot_solution",
    "plot_loss_history",
    "pi_deeponet_train",
    "pi_deeponet_test",
    "pi_deeponet_plot_training_history",
    "pi_deeponet_convert_test_results_to_vtk",
    "pi_deeponet_application_create_model_application_dataset",
    "pi_deeponet_application_run_model_application",
    "pi_deeponet_application_compare_model_application_results_with_simulation_results",
]
