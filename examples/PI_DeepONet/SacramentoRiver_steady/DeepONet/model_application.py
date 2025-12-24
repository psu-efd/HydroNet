"""
Example usage of the HydroNet framework.

This script is used to apply the trained model to the evaluation cases.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import time  # Add back time import for the main timer
import torch.multiprocessing as mp

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

from HydroNet import Config
from HydroNet.utils.pi_deeponet_utils import (
    pi_deeponet_application_create_model_application_dataset,
    pi_deeponet_application_compute_distance_between_application_and_training_data,
    pi_deeponet_application_run_model_application,
    pi_deeponet_application_compare_model_application_results_with_simulation_results,
    pi_deeponet_application_plot_difference_metrics_against_parameter_distance
)

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

    # Compute the distance between the application data and the training data
    pi_deeponet_application_compute_distance_between_application_and_training_data(config)
   
    # Run the model with the best model and save the predictions results to vtk files
    pi_deeponet_application_run_model_application(config)

    # Compare the model application results with the simulation results
    pi_deeponet_application_compare_model_application_results_with_simulation_results(config, system_name)

    # Plot the difference metrics against the parameter distance from the training data
    pi_deeponet_application_plot_difference_metrics_against_parameter_distance(config)

    # Calculate and print the total execution time
    main_execution_time = time.time() - main_start_time
    print(f"\n⏱️ Total execution time: {main_execution_time:.2f} seconds")
    
    print("\nModel application completed successfully!") 