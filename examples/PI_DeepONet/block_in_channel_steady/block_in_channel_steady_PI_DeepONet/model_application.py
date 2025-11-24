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
    pi_deeponet_convert_test_results_to_vtk
)

# Import data processing functions
from HydroNet.utils.data_processing_common import get_cells_in_domain
import vtk

def create_model_application_dataset(config):
    """

    Create the model application dataset (PI_SWE_DeepONetDataset). The dataset is created from the application parameters and the application vtk file. In fact, only the DeepONet data is relevant for the model application. The PINN data is not needed.

    Args:
        config: The configuration object.

    Returns:
        dataset: The model application dataset (PI_SWE_DeepONetDataset).
    """

    # Check if config is an instance of Config
    if not isinstance(config, Config):
        raise ValueError("config must be an instance of Config")

    # Application parameters file: The path to the application parameters file
    application_dir = config['application']['application_dir']
    application_parameters_file = config['application']['application_parameters_file']
    application_vtk_file = config['application']['application_vtk_file']

    # Load the application parameters
    application_parameters = np.loadtxt(os.path.join(application_dir, application_parameters_file), skiprows=1)
    
    # Ensure application_parameters is 2D (handle case where it's 1D with only one feature)
    if application_parameters.ndim == 1:
        application_parameters = application_parameters.reshape(-1, 1)

    # Make sure the dimension of the application parameters is compatible with the DeepONet branch inputs
    if application_parameters.shape[1] != config['model']['branch_net']['branch_input_dim']:
        print(f"The dimension of the application parameters is {application_parameters.shape[1]}")
        print(f"The dimension of the DeepONet branch inputs is {config['model']['branch_net']['branch_input_dim']}")
        raise ValueError("The dimension of the application parameters is not compatible with the DeepONet branch inputs")

    # Read VTK file and extract cell centers only
    print(f"Reading VTK file and extracting cell centers from {application_vtk_file}...")
    vtk_reader = vtk.vtkUnstructuredGridReader()
    vtk_reader.SetFileName(os.path.join(application_dir, application_vtk_file))
    vtk_reader.Update()
    
    # Get cell centers from the VTK file
    cell_ids, cell_centers, cell_Sx, cell_Sy = get_cells_in_domain(vtk_reader)
    
    # Get the number of cells
    n_cells = len(cell_centers)
    print(f"Number of cells: {n_cells}")
    
    # Determine unit system for metadata (default to SI if not specified)
    unit_system = config['physics']['unit']
    if unit_system not in ['SI', 'EN']:
        print(f"Invalid unit system: {unit_system}")
        print("Valid unit systems are: SI, EN")
        raise ValueError("Invalid unit system")
    
    # Prepare data for this application case
    # Branch inputs: tile the application parameters for each cell
    # application_parameters should have shape (n_cases, n_features). n_cases is the number of application cases. n_features is the number of application parameters. Need to consider the case where n_features is 1.
    # For each case, we tile the parameters for all cells
    n_cases = application_parameters.shape[0]
    if application_parameters.ndim == 1:
        n_features = 1
    else:
        n_features = application_parameters.shape[1]

    # Check if the number of cases, cells, and features are valid
    if n_cases < 1:
        print(f"Number of application cases is {n_cases}")
        raise ValueError("Number of application cases is less than 1")
    if n_cells < 1:
        print(f"Number of cells is {n_cells}")
        raise ValueError("Number of cells is less than 1")
    if n_features < 1:
        print(f"Number of features is {n_features}")
        raise ValueError("Number of features is less than 1")    
    
    # Create branch inputs: for each case, tile the parameters for all cells
    branch_inputs_list = []
    trunk_inputs_list = []
    
    for case_idx in range(n_cases):
        # Tile the parameters for this case to match the number of cells
        case_branch_inputs = np.tile(application_parameters[case_idx, :], (n_cells, 1))
        branch_inputs_list.append(case_branch_inputs)
        
        # Trunk inputs are the same for all cases (cell centers)
        case_trunk_inputs = cell_centers[:, :2]  # Only need x, y coordinates for steady case. For unsteady case, we need to add the time variable.
        trunk_inputs_list.append(case_trunk_inputs)
    
    # Concatenate all cases
    branch_inputs = np.concatenate(branch_inputs_list, axis=0).astype(np.float32)
    trunk_inputs = np.concatenate(trunk_inputs_list, axis=0).astype(np.float32)
    
    # Load normalization stats from the training data
    stats_path = config['application']['train_val_test_stats_path']
    if not os.path.exists(os.path.join(application_dir, stats_path)):
        raise FileNotFoundError(f"Normalization stats file not found: {os.path.join(application_dir, stats_path)}")
    
    with open(os.path.join(application_dir, stats_path), 'r') as f:
        all_DeepONet_stats = json.load(f)
    
    # Get normalization methods and stats
    normalization_specs = all_DeepONet_stats['all_DeepONet_branch_trunk_outputs_stats']
    branch_normalization_method = normalization_specs['normalization_method']['branch_inputs']
    trunk_normalization_method = normalization_specs['normalization_method']['trunk_inputs']
    
    branch_stats = normalization_specs['branch_inputs']
    trunk_stats = normalization_specs['trunk_inputs']
    
    # Normalize branch inputs
    if branch_normalization_method == 'z-score':
        branch_mean = np.array(branch_stats['mean'])
        branch_std = np.array(branch_stats['std'])
        branch_inputs = (branch_inputs - branch_mean) / branch_std
    elif branch_normalization_method == 'min-max':
        branch_min = np.array(branch_stats['min'])
        branch_max = np.array(branch_stats['max'])
        branch_inputs = (branch_inputs - branch_min) / (branch_max - branch_min)
    else:
        raise ValueError(f"Invalid branch inputs normalization method: {branch_normalization_method}")
    
    # Normalize trunk inputs
    if trunk_normalization_method == 'z-score':
        trunk_mean = np.array(trunk_stats['mean'])
        trunk_std = np.array(trunk_stats['std'])
        trunk_inputs = (trunk_inputs - trunk_mean) / trunk_std
    elif trunk_normalization_method == 'min-max':
        trunk_min = np.array(trunk_stats['min'])
        trunk_max = np.array(trunk_stats['max'])
        trunk_inputs = (trunk_inputs - trunk_min) / (trunk_max - trunk_min)
    else:
        raise ValueError(f"Invalid trunk inputs normalization method: {trunk_normalization_method}")
    
    # Create output directory for application data
    # Put it under application/DeepONet/ so it can find the stats file correctly
    # The PI_SWE_DeepONetDataset looks for all_DeepONet_stats.json in the parent directory
    application_data_dir = os.path.join(application_dir, 'DeepONet')
    os.makedirs(application_data_dir, exist_ok=True)
    
    # Create dummy outputs (zeros) since we don't have ground truth for application
    # The shape should match: (n_cases * n_cells, n_outputs) where n_outputs = 3 (h, u, v)
    n_outputs = config['model']['output_dim']
    outputs = np.zeros((n_cases * n_cells, n_outputs), dtype=np.float32)
    
    # Save to HDF5 file
    h5_file_path = os.path.join(application_data_dir, 'data.h5')
    print(f"Saving application dataset to {h5_file_path}...")
    with h5py.File(h5_file_path, 'w') as f:
        f.create_dataset('branch_inputs', data=branch_inputs, dtype='float32')
        f.create_dataset('trunk_inputs', data=trunk_inputs, dtype='float32')
        f.create_dataset('outputs', data=outputs, dtype='float32')
        
        # Add metadata
        f.attrs['n_cases'] = n_cases
        f.attrs['n_cells'] = n_cells
        f.attrs['n_features'] = n_features
        f.attrs['n_outputs'] = n_outputs
        f.attrs['unit'] = unit_system
    
    print(f"Application dataset created successfully!")
    print(f"  - Branch inputs shape: {branch_inputs.shape}")
    print(f"  - Trunk inputs shape: {trunk_inputs.shape}")
    print(f"  - Outputs shape: {outputs.shape}")
    
    # Create and return the dataset
    # First set model.use_physics_loss to False because we don't need the PINN data for model application
    config['model']['use_physics_loss'] = False
    # Then create the dataset
    dataset = PI_SWE_DeepONetDataset(data_path=application_data_dir, config=config)
    
    return dataset, application_parameters


def run_model_application(config):
    """
    Apply the trained model to the evaluation cases.

    Args:
        config: The configuration object.
    """

    # Check if config is an instance of Config
    if not isinstance(config, Config):
        raise ValueError("config must be an instance of Config")

    # Get application directory
    application_dir = config['application']['application_dir']
    
    # Model checkpoint file: The path to the best model checkpoint file
    model_checkpoint_file = config['application']['model_checkpoint_file']

    # Create the application dataset
    dataset, application_parameters = create_model_application_dataset(config)

    # Get a sample to determine dimensions
    sample_branch, sample_trunk, sample_output = dataset[0]
    branch_dim = sample_branch.shape[0]
    trunk_dim = sample_trunk.shape[0]
    output_dim = sample_output.shape[0]
    print(f"Detected branch_dim={branch_dim}, trunk_dim={trunk_dim}, output_dim={output_dim}")

    # Create model
    print("\n\nCreating the trained model...")
    model = PI_SWE_DeepONetModel(config=config)
    model.check_model_input_output_dimensions(branch_dim, trunk_dim, output_dim)

    # Create trainer
    trainer = PI_SWE_DeepONetTrainer(model, config=config)
    
    # Load the trained model checkpoint
    try:
        trainer.load_checkpoint(model_checkpoint_file)
        print(f"Successfully loaded model checkpoint from {model_checkpoint_file}")
    except (FileNotFoundError, RuntimeError) as e:
        print(f"Error loading model checkpoint: {e}, exiting...")        
        exit()

    # Load application dataset
    print("\n\nLoading application dataset...")
    test_loader = DataLoader(
            dataset,
            batch_size=1280,   # Batch size for the application dataset (probably does not matter)
            shuffle=False, # Do not shuffle the application dataset because the order of the samples is important
            num_workers=0,
            pin_memory=False
        )

     # Get the all_DeepONet_stats
    all_DeepONet_stats = dataset.get_deeponet_stats()

    # Make sure the normalization method for outputs is "z-score" (currently only "z-score" is supported)
    if all_DeepONet_stats['all_DeepONet_branch_trunk_outputs_stats']['normalization_method']['outputs'] != "z-score":
        raise ValueError("Output normalization method is not supported. Currently only 'z-score' is supported.")

    output_mean = all_DeepONet_stats['all_DeepONet_branch_trunk_outputs_stats']['outputs']['mean']
    output_std = all_DeepONet_stats['all_DeepONet_branch_trunk_outputs_stats']['outputs']['std']

    # Evaluate model on application dataset
    model.eval()  # Set model to evaluation mode
    all_predictions = []
    
    print("Running predictions on application dataset...")
    sample_count = 0

    with torch.no_grad():  # Disable gradient calculation for inference
        for branch_input, trunk_input, target in test_loader:
            # Move data to device
            branch_input = branch_input.to(trainer.device)
            trunk_input = trunk_input.to(trainer.device)
            target = target.to(trainer.device)
            
            # Forward pass
            output = model(branch_input, trunk_input)           
            
            # Store predictions 
            all_predictions.append(output.cpu().numpy())
            
            # Count samples
            sample_count += len(branch_input)

    # Concatenate all predictions
    all_predictions = np.vstack(all_predictions)

    # Denormalize the predictions
    # Assuming the outputs are normalized as (x - mean) / std
    denorm_predictions = all_predictions * output_std + output_mean
    
    # Save the denormalized predictions to a npz file
    np.savez(f'{application_dir}/application_predictions.npz', denorm_predictions=denorm_predictions)

    print(f"Application predictions saved to {application_dir}/application_predictions.npz")

    # Convert the denormalized predictions to vtk files for visualization
    print("\n\nConverting application predictions to vtk files...")
    print("================================================")
    
    # Get application parameters and VTK file paths
    application_vtk_file = config['application']['application_vtk_file']
    application_vtk_path = os.path.join(application_dir, application_vtk_file)
    
    # application_parameters is already loaded from create_model_application_dataset
    # Ensure application_parameters is 2D (handle case where it's 1D with only one feature)
    if application_parameters.ndim == 1:
        application_parameters = application_parameters.reshape(-1, 1)
    n_features = application_parameters.shape[1]
    
    # Get the number of cells from the dataset HDF5 file metadata
    application_data_dir = os.path.join(application_dir, 'DeepONet')
    h5_file_path = os.path.join(application_data_dir, 'data.h5')
    with h5py.File(h5_file_path, 'r') as f:
        n_cells = f.attrs['n_cells']
        n_cases = f.attrs['n_cases']
    
    print(f"Number of cells: {n_cells}")
    print(f"Number of cases: {n_cases}")
    print(f"Number of parameters: {n_features}")

    # make the directory for the vtk files
    vtk_dir = os.path.join(application_dir, 'predictions_vtks')
    os.makedirs(vtk_dir, exist_ok=True)
    
    # Process each case
    for case_idx in range(n_cases):
        # Get the current case's predictions: h, u, v
        h_pred = denorm_predictions[case_idx * n_cells:(case_idx + 1) * n_cells, 0]
        u_pred = denorm_predictions[case_idx * n_cells:(case_idx + 1) * n_cells, 1]
        v_pred = denorm_predictions[case_idx * n_cells:(case_idx + 1) * n_cells, 2]
        
        # Get the current case's parameters
        case_parameters = application_parameters[case_idx, :]  # Shape: (n_features,)
        
        # Read the VTK file which has the mesh information
        reader = vtk.vtkUnstructuredGridReader()
        reader.SetFileName(application_vtk_path)
        reader.Update()
        mesh = reader.GetOutput()
        
        # Remove any existing point data and cell data from the template file
        # We only want the mesh structure (points and cells), not any existing data arrays
        mesh.GetPointData().Initialize()
        mesh.GetCellData().Initialize()
        mesh.GetFieldData().Initialize()  # Also clear field data
        
        # Create cell data arrays for the predicted results
        h_pred_array = numpy_to_vtk(h_pred, deep=True, array_type=vtk.VTK_FLOAT)
        h_pred_array.SetName("Water_Depth_Pred")
        
        u_pred_array = numpy_to_vtk(u_pred, deep=True, array_type=vtk.VTK_FLOAT)
        u_pred_array.SetName("X_Velocity_Pred")
        
        v_pred_array = numpy_to_vtk(v_pred, deep=True, array_type=vtk.VTK_FLOAT)
        v_pred_array.SetName("Y_Velocity_Pred")
        
        # Create velocity vectors for predicted values
        velocity_pred = np.column_stack((u_pred, v_pred, np.zeros_like(u_pred)))  # Add z-component as 0
        velocity_pred_array = numpy_to_vtk(velocity_pred, deep=True, array_type=vtk.VTK_FLOAT)
        velocity_pred_array.SetNumberOfComponents(3)
        velocity_pred_array.SetName("Velocity_Pred")

        # Create velocity magnitude for predicted values
        velocity_magnitude_pred = np.sqrt(u_pred**2 + v_pred**2)
        velocity_magnitude_pred_array = numpy_to_vtk(velocity_magnitude_pred, deep=True, array_type=vtk.VTK_FLOAT)
        velocity_magnitude_pred_array.SetName("Velocity_Magnitude_Pred")
        
        # Add the arrays to the mesh as cell data
        mesh.GetCellData().AddArray(h_pred_array)
        mesh.GetCellData().AddArray(u_pred_array)
        mesh.GetCellData().AddArray(v_pred_array)
        mesh.GetCellData().AddArray(velocity_pred_array)
        mesh.GetCellData().AddArray(velocity_magnitude_pred_array)
        
        # Add parameter values as Field Data (metadata for the entire dataset)
        # Field Data is appropriate for case-level information like parameters
        for param_idx in range(n_features):
            # Create a single-element array with the parameter value
            param_value = np.array([case_parameters[param_idx]], dtype=np.float32)
            param_array = numpy_to_vtk(param_value, deep=True, array_type=vtk.VTK_FLOAT)
            param_array.SetName(f"Parameter_{param_idx + 1}")
            mesh.GetFieldData().AddArray(param_array)
        
        # Create a writer
        writer = vtk.vtkUnstructuredGridWriter()
        output_file = os.path.join(vtk_dir, f'case_{case_idx + 1}_application_results.vtk')
        writer.SetFileName(output_file)
        writer.SetInputData(mesh)
        writer.Write()
        
        # Print parameter values for this case
        param_str = ", ".join([f"Param_{i+1}={case_parameters[i]:.6f}" for i in range(n_features)])
        print(f"Saved results for case {case_idx + 1} to {output_file} ({param_str})")
    
    print(f"\nFinished converting {n_cases} application cases to VTK files.")


def compare_model_application_results_with_simulation_results(config, system_name):
    """
    Compare the model application results with the simulation results. 
    The simulation results are stored in the "application/simulated_vtks" directory.
    The model application results are stored in the "application/predictions_vtks" directory.

    The results (both simulation and model application) are in vtk unstructured grid format. The function will read the results and compute the difference.

    Args:
        config: The configuration object.
        system_name: The name of the system (e.g. "Windows", "Linux", "Mac").
    """

    # Check if config is an instance of Config
    if not isinstance(config, Config):
        raise ValueError("config must be an instance of Config")

    # Get the unit system
    unit_system = config['physics']['unit']
    if unit_system not in ['SI', 'EN']:
        raise ValueError("Invalid unit system. Valid unit systems are: SI, EN")

    # Get the application directory
    application_dir = config['application']['application_dir']

    # Set the variable names for the results of model application and simulation
    if unit_system == 'SI':
        # Model application variables
        model_application_h_name = 'Water_Depth_Pred'
        model_application_u_name = 'X_Velocity_Pred'
        model_application_v_name = 'Y_Velocity_Pred'
        model_application_velocity_name = 'Velocity_Pred'  # Vector (3 components)
        # Simulation variables
        simulation_h_name = 'Water_Depth_m'
        simulation_velocity_name = 'Velocity_m_p_s'  # Vector (2 or 3 components)
        simulation_velocity_magnitude_name = 'Vel_Mag_m_p_s'
    elif unit_system == 'EN':
        # Model application variables
        model_application_h_name = 'Water_Depth_Pred'
        model_application_u_name = 'X_Velocity_Pred'
        model_application_v_name = 'Y_Velocity_Pred'
        model_application_velocity_name = 'Velocity_Pred'  # Vector (3 components)
        # Simulation variables
        simulation_h_name = 'Water_Depth_ft'
        simulation_velocity_name = 'Velocity_ft_p_s'  # Vector (2 or 3 components)
        simulation_velocity_magnitude_name = 'Vel_Mag_ft_p_s'
    else:
        raise ValueError("Invalid unit system. Valid unit systems are: SI, EN")

    n_cases = 10

    diff_h_mse_list = []
    diff_u_mse_list = []
    diff_v_mse_list = []
    diff_velocity_magnitude_mse_list = []

    # Get the simulation results directory
    simulation_results_dir = os.path.join(config['application']['application_dir'], 'simulated_vtks')
    # Get the model application results directory
    predictions_vtks_dir = os.path.join(config['application']['application_dir'], 'predictions_vtks')

    # Loop over all cases 
    for case_idx in range(n_cases):
        # Get the simulation results
        simulation_results = os.path.join(simulation_results_dir, f'case_{case_idx + 1}.vtk')

        # Get the model application results
        predictions_results = os.path.join(predictions_vtks_dir, f'case_{case_idx + 1}_application_results.vtk')

        # Read the two vtk files, check the consistency of the variables
        simulation_reader = vtk.vtkUnstructuredGridReader()
        simulation_reader.SetFileName(simulation_results)
        simulation_reader.Update()
        simulation_mesh = simulation_reader.GetOutput()
        predictions_reader = vtk.vtkUnstructuredGridReader()
        predictions_reader.SetFileName(predictions_results)
        predictions_reader.Update()
        predictions_mesh = predictions_reader.GetOutput()

        # Check the consistency of the variables in simulation results
        simulation_cell_data = simulation_mesh.GetCellData()
        if not simulation_cell_data.HasArray(simulation_h_name):
            raise ValueError(f"Variable {simulation_h_name} not found in simulation results")
        if not simulation_cell_data.HasArray(simulation_velocity_name):
            raise ValueError(f"Variable {simulation_velocity_name} not found in simulation results")
        if not simulation_cell_data.HasArray(simulation_velocity_magnitude_name):
            raise ValueError(f"Variable {simulation_velocity_magnitude_name} not found in simulation results")
        
        # Check the consistency of the variables in predictions results
        predictions_cell_data = predictions_mesh.GetCellData()
        if not predictions_cell_data.HasArray(model_application_h_name):
            raise ValueError(f"Variable {model_application_h_name} not found in predictions results")
        if not predictions_cell_data.HasArray(model_application_u_name):
            raise ValueError(f"Variable {model_application_u_name} not found in predictions results")
        if not predictions_cell_data.HasArray(model_application_v_name):
            raise ValueError(f"Variable {model_application_v_name} not found in predictions results")
        if not predictions_cell_data.HasArray(model_application_velocity_name):
            raise ValueError(f"Variable {model_application_velocity_name} not found in predictions results")

        # Get the variables from the simulation results and convert to numpy arrays
        simulation_h_array = simulation_mesh.GetCellData().GetArray(simulation_h_name)
        simulation_velocity_array = simulation_mesh.GetCellData().GetArray(simulation_velocity_name)
        simulation_velocity_magnitude_array = simulation_mesh.GetCellData().GetArray(simulation_velocity_magnitude_name)
        
        # Convert VTK arrays to numpy arrays
        simulation_h = vtk_to_numpy(simulation_h_array)
        simulation_velocity = vtk_to_numpy(simulation_velocity_array)  # Shape: (n_cells, n_components)
        simulation_velocity_magnitude = vtk_to_numpy(simulation_velocity_magnitude_array)

        #make the z component of the simulation velocity array to be 0
        simulation_velocity[:, 2] = 0
        
        # Extract u and v components from simulation velocity vector
        # Simulation velocity may have 2 or 3 components (x, y, [z])
        if simulation_velocity.shape[1] >= 2:
            simulation_u = simulation_velocity[:, 0]
            simulation_v = simulation_velocity[:, 1]
        else:
            raise ValueError(f"Simulation velocity array has unexpected shape: {simulation_velocity.shape}")

        # Get the variables from the predictions results and convert to numpy arrays
        predictions_h_array = predictions_mesh.GetCellData().GetArray(model_application_h_name)
        predictions_u_array = predictions_mesh.GetCellData().GetArray(model_application_u_name)
        predictions_v_array = predictions_mesh.GetCellData().GetArray(model_application_v_name)
        predictions_velocity_array = predictions_mesh.GetCellData().GetArray(model_application_velocity_name)

        # Convert VTK arrays to numpy arrays
        predictions_h = vtk_to_numpy(predictions_h_array)
        predictions_u = vtk_to_numpy(predictions_u_array)
        predictions_v = vtk_to_numpy(predictions_v_array)
        predictions_velocity = vtk_to_numpy(predictions_velocity_array)  # Shape: (n_cells, 3)

        #make the z component of the predictions velocity array to be 0
        predictions_velocity[:, 2] = 0

        print(f"predictions_velocity.shape: {predictions_velocity.shape}")
        print(f"simulation_velocity.shape: {simulation_velocity.shape}")
        
        # Compute velocity magnitude from predictions if not available
        # predictions_velocity has shape (n_cells, 3) with components [u, v, 0]
        predictions_velocity_magnitude = np.sqrt(predictions_u**2 + predictions_v**2)

        # Compute the difference between the simulation and predictions
        diff_h = predictions_h - simulation_h
        diff_u = predictions_u - simulation_u
        diff_v = predictions_v - simulation_v
        diff_velocity_magnitude = predictions_velocity_magnitude - simulation_velocity_magnitude

        # Compute the MSE of the difference
        diff_h_mse = np.mean(diff_h**2)
        diff_u_mse = np.mean(diff_u**2)
        diff_v_mse = np.mean(diff_v**2)
        diff_velocity_magnitude_mse = np.mean(diff_velocity_magnitude**2)

        diff_h_mse_list.append(float(diff_h_mse))
        diff_u_mse_list.append(float(diff_u_mse))
        diff_v_mse_list.append(float(diff_v_mse))
        diff_velocity_magnitude_mse_list.append(float(diff_velocity_magnitude_mse))

        # Create a new vtk file which contains the difference between the simulation and predictions
        # Create a new unstructured grid and deep copy the mesh structure from simulation
        difference_mesh = vtk.vtkUnstructuredGrid()
        difference_mesh.DeepCopy(simulation_mesh)
        
        # Clear the cell data (we'll add difference arrays instead)
        difference_mesh.GetCellData().Initialize()
        
        # Compute velocity difference vector
        diff_velocity = np.column_stack((diff_u, diff_v, np.zeros_like(diff_u)))  # Add z-component as 0
        
        # Add the difference arrays to the cell data
        diff_h_array = numpy_to_vtk(diff_h, deep=True, array_type=vtk.VTK_FLOAT)
        diff_h_array.SetName("Diff_Water_Depth")
        difference_mesh.GetCellData().AddArray(diff_h_array)
        
        diff_u_array = numpy_to_vtk(diff_u, deep=True, array_type=vtk.VTK_FLOAT)
        diff_u_array.SetName("Diff_X_Velocity")
        difference_mesh.GetCellData().AddArray(diff_u_array)
        
        diff_v_array = numpy_to_vtk(diff_v, deep=True, array_type=vtk.VTK_FLOAT)
        diff_v_array.SetName("Diff_Y_Velocity")
        difference_mesh.GetCellData().AddArray(diff_v_array)

        diff_velocity_array = numpy_to_vtk(diff_velocity, deep=True, array_type=vtk.VTK_FLOAT)
        diff_velocity_array.SetNumberOfComponents(3)
        diff_velocity_array.SetName("Diff_Velocity")
        difference_mesh.GetCellData().AddArray(diff_velocity_array)

        diff_velocity_magnitude_array = numpy_to_vtk(diff_velocity_magnitude, deep=True, array_type=vtk.VTK_FLOAT)
        diff_velocity_magnitude_array.SetName("Diff_Velocity_Magnitude")
        difference_mesh.GetCellData().AddArray(diff_velocity_magnitude_array)
        
        #create a writer
        difference_writer = vtk.vtkUnstructuredGridWriter()
        difference_writer.SetFileName(os.path.join(application_dir, "diff_vtks", f'case_{case_idx + 1}_diff.vtk'))
        difference_writer.SetInputData(difference_mesh)
        difference_writer.Write()   

    print(f"Diff_h_mse_list: {diff_h_mse_list}")
    print(f"Diff_u_mse_list: {diff_u_mse_list}")
    print(f"Diff_v_mse_list: {diff_v_mse_list}")
    print(f"Diff_velocity_magnitude_mse_list: {diff_velocity_magnitude_mse_list}")

    # Save the difference lists to a json file
    with open(os.path.join(application_dir, "diff_lists.json"), "w") as f:
        json.dump({"diff_h_mse_list": diff_h_mse_list, "diff_u_mse_list": diff_u_mse_list, "diff_v_mse_list": diff_v_mse_list, "diff_velocity_magnitude_mse_list": diff_velocity_magnitude_mse_list}, f, indent=4)
    print(f"Diff lists saved to {os.path.join(application_dir, 'diff_lists.json')}")

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
    #run_model_application(config)

    # Compare the model application results with the simulation results
    #compare_model_application_results_with_simulation_results(config, system_name)

    # Plot the difference lists
    plot_difference_lists(config)

    # Calculate and print the total execution time
    main_execution_time = time.time() - main_start_time
    print(f"\n⏱️ Total execution time: {main_execution_time:.2f} seconds")
    
    print("\nModel application completed successfully!") 