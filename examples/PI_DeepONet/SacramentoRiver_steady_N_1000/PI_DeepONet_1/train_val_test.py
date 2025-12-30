"""
Example usage of the HydroNet framework.

This example demonstrates the usage of the DeepONet component of HydroNet for learning 
the operator of the shallow water equations without physics-informed constraints.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
import torch
import time  # Add back time import for the main timer
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
import yaml  # Add yaml import
import vtk
from vtk.util.numpy_support import numpy_to_vtk
import h5py

# Set the random seed before generating any random numbers
np.random.seed(123456)  # You can use any integer as the seed

# Set the random seed for PyTorch
torch.manual_seed(123456)
# If using CUDA (GPUs)
if torch.cuda.is_available():
    torch.cuda.manual_seed(123456)

# Because of the using PyTorch DataLoader with multiple workers, we also need to set the hash seed
#os.environ['PYTHONHASHSEED'] = str(123456)

plt.rc('text', usetex=False)  #allow the use of Latex for math expressions and equations
plt.rc('font', family='serif') #specify the default font family to be "serif"

# Get the project root directory (assumes this script is in examples/PI_DeepONet/SacramentoRiver_steady/SacramentoRiver_steady_DeepONet directory)
script_path = os.path.abspath(__file__)
# Go up 4 levels: script -> SacramentoRiver_steady_DeepONet -> SacramentoRiver_steady -> PI_DeepONet -> examples
examples_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(script_path))))
# Go up one more level from examples to get project root (where HydroNet package is)
project_root = os.path.dirname(examples_dir)

# Add the project root to the Python path if it's not already there
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"Added {project_root} to Python path")

from HydroNet import (
    PI_SWE_DeepONetModel, 
    PI_SWE_DeepONetTrainer, 
    PI_SWE_DeepONetDataset, 
    Config,
    pi_deeponet_train,
    pi_deeponet_test,
    pi_deeponet_plot_training_history,
    pi_deeponet_convert_test_results_to_vtk
)


if __name__ == "__main__":
    # Start the main timer
    main_start_time = time.time()

    # Load configuration
    config_file = './pi_deeponet_config.yaml'
    config = Config(config_file)
    
    # Set the multiprocessing start method
    mp.set_start_method('spawn', force=True)
    
    # Train the model
    model, history, history_file_name = pi_deeponet_train(config)

    # Plot training history
    pi_deeponet_plot_training_history(history_file_name)

    # Test the model with the best model and save the test results to vtk files
    pi_deeponet_test('./checkpoints/pi_deeponet_epoch_best.pt', config)
    
    # Calculate and print the total execution time
    main_execution_time = time.time() - main_start_time
    print(f"\n⏱️ Total execution time: {main_execution_time:.2f} seconds")
    
    print("\nExample completed successfully!") 
