"""
Example usage of the HydroNet framework.

This script is used to apply the trained model to the evaluation cases.
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

from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
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

# Get the project root directory (it depends how deep the directory structure is; revise accordingly)
script_path = os.path.abspath(__file__)
examples_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(script_path))))
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
    pi_deeponet_convert_test_results_to_vtk,
    pi_deeponet_application_run_model_application,
    pi_deeponet_application_compare_model_application_results_with_simulation_results
)

# Import data processing functions
from HydroNet.utils.data_processing_common import get_cells_in_domain
import vtk

def plot_difference_lists(config):
    """
    Plot the difference lists from the diff_lists.json file.
    """

    # Get the application directory
    application_dir = config['application']['application_dir']

    # Load the difference lists from the json file
    with open(os.path.join(application_dir, "diff_lists.json"), "r") as f:
        diff_lists = json.load(f)

    case_indices = list(range(1, len(diff_lists["diff_h_mse_list"]) + 1))
    
    # Extract the data
    diff_h_mse_list = diff_lists["diff_h_mse_list"]
    diff_velocity_magnitude_mse_list = diff_lists["diff_velocity_magnitude_mse_list"]

    # Plot the difference lists (only h and velocity magnitude in two subplots)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot water depth MSE
    ax1.plot(case_indices, diff_h_mse_list, 'o-', linewidth=2, markersize=8, color='#2E86AB', label='Water Depth MSE')
    ax1.set_xlabel('Case Index', fontsize=12, fontweight='bold')
    ax1.set_ylabel('MSE (m²)', fontsize=12, fontweight='bold')
    ax1.set_title('Water Depth Prediction Error (MSE)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=11)
    ax1.set_xticks(case_indices)
    ax1.tick_params(axis='both', labelsize=10)
    
    # Plot velocity magnitude MSE
    ax2.plot(case_indices, diff_velocity_magnitude_mse_list, 's-', linewidth=2, markersize=8, color='#A23B72', label='Velocity Magnitude MSE')
    ax2.set_xlabel('Case Index', fontsize=12, fontweight='bold')
    ax2.set_ylabel('MSE (m²/s²)', fontsize=12, fontweight='bold')
    ax2.set_title('Velocity Magnitude Prediction Error (MSE)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(fontsize=11)
    ax2.set_xticks(case_indices)
    ax2.tick_params(axis='both', labelsize=10)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Save the figure
    output_file = os.path.join(application_dir, 'prediction_errors.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    # Start the main timer
    main_start_time = time.time()

    # Load configuration
    config_file = './pi_deeponet_config.yaml'
    config = Config(config_file)

    #name of the system
    system_name = "Windows"
    
    # Set the multiprocessing start method
    mp.set_start_method('spawn', force=True)
   
    # Run the model with the best model and save the predictions results to vtk files
    #pi_deeponet_application_run_model_application(config)

    # Compare the model application results with the simulation results
    #pi_deeponet_application_compare_model_application_results_with_simulation_results(config, system_name)

    # Plot the difference lists
    plot_difference_lists(config)

    # Calculate and print the total execution time
    main_execution_time = time.time() - main_start_time
    print(f"\n⏱️ Total execution time: {main_execution_time:.2f} seconds")
    
    print("\nModel application completed successfully!") 