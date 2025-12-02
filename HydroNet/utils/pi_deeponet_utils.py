"""
Common utilities for PI-DeepONet training, testing, and visualization.

This module provides reusable functions for training, testing, and visualizing
PI-DeepONet models, reducing code duplication across example scripts.
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from datetime import datetime
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from scipy.spatial.distance import cdist
import h5py

from ..models.PI_DeepONet.model import PI_SWE_DeepONetModel
from ..models.PI_DeepONet.trainer import PI_SWE_DeepONetTrainer
from ..models.PI_DeepONet.data import PI_SWE_DeepONetDataset
from .config import Config
from .data_processing_common import get_cells_in_domain


def pi_deeponet_train(config):
    """
    Train DeepONet with optional physics-informed constraints.

    Args:
        config: Configuration object (Config instance)
    
    Returns:
        tuple: (model, history) - The trained model and training history dictionary
    """    

    # Check if config is an instance of Config
    if not isinstance(config, Config):
        raise ValueError("config must be an instance of Config")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    def print_gpu_memory(label):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
            print(f"{label}:")
            print(f"  Allocated: {allocated:.2f} MB")
            print(f"  Reserved: {reserved:.2f} MB")
    
    # Monitor initial GPU memory
    print_gpu_memory("Initial GPU memory")
    
    try:
        # Create PI-DeepONet dataset and dataloader for training and validation
        print("Loading training dataset...")
        train_dataset = PI_SWE_DeepONetDataset(
            data_path=config.get('data.deeponet.train_data_path'),
            config=config
        )
        
        print("Loading validation dataset...")
        val_dataset = PI_SWE_DeepONetDataset(
            data_path=config.get('data.deeponet.val_data_path'),
            config=config
        )
        
        # Monitor memory after dataset loading
        print_gpu_memory("GPU memory after dataset loading")
        
        # Create dataloaders
        # Note: On Windows, num_workers > 0 and pin_memory=True can cause issues with buffer variables
        # Adjust these settings based on your platform and needs
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.get('training.batch_size'),
            shuffle=True,
            num_workers=config.get('training.num_workers', 0),
            pin_memory=config.get('training.pin_memory', False)
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.get('training.batch_size'),
            shuffle=False,
            num_workers=config.get('training.num_workers', 0),
            pin_memory=config.get('training.pin_memory', False)
        )
        
        # Initialize model 
        print("Initializing model...")
        model = PI_SWE_DeepONetModel(config)        

        # Check whether the model and data are compatible
        model.check_model_input_output_dimensions(train_dataset.branch_dim, train_dataset.trunk_dim, train_dataset.output_dim)
        
        # Initialize trainer
        trainer = PI_SWE_DeepONetTrainer(
            model,
            config
        )

        # Load the initial model if specified in the config
        bLoad_initial_model = config.get('training.bLoad_initial_model', False)
        if bLoad_initial_model:
            initial_model_checkpoint_file = config.get('training.initial_model_checkpoint_file')
            if initial_model_checkpoint_file is None:
                raise ValueError("initial_model_checkpoint_file is not specified in the config")
            trainer.load_checkpoint(initial_model_checkpoint_file)
            print(f"Successfully loaded initial model from {initial_model_checkpoint_file}")
        else:
            print("No initial model to load; using random initial weights")
        
        # Monitor memory after first batch. Note the GPU memory footprint only involves the DeepONet model, not the physics equations constrains.
        print("\nMonitoring memory during first batch...")
        for branch_input, trunk_input, target in train_loader:
            branch_input = branch_input.to(device)
            trunk_input = trunk_input.to(device)
            target = target.to(device)
            print_gpu_memory("GPU memory after data transfer")
            
            # Forward pass
            output = model(branch_input, trunk_input)
            print_gpu_memory("GPU memory after forward pass")
            
            # Backward pass
            loss = trainer.loss_fn(output, target)
            loss.backward()
            print_gpu_memory("GPU memory after backward pass")
            
            # Optimizer step
            trainer.optimizer.step()
            trainer.optimizer.zero_grad()
            print_gpu_memory("GPU memory after optimizer step")
            break
        
        # Train the model
        print("\nStarting training...")
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            physics_dataset=train_dataset.physics_dataset
        )

        # Save the history to a json file with a time stamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f'history_{timestamp}.json', 'w') as f:
            json.dump(history, f, indent=4)
        print(f"History saved to history_{timestamp}.json")
        
        return model, history
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("\n⚠️ CUDA out of memory error detected!")
            print_gpu_memory("Current GPU memory")
            
            print("\nSuggestions to resolve out of memory:")
            print("1. Reduce batch size")
            print("2. Use gradient accumulation")
            print("3. Use mixed precision training")
            print("4. Use data streaming instead")
            
            # Clean up GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            raise RuntimeError("Out of memory error. Please try using data streaming or reducing batch size.")
        else:
            raise e
            
    finally:
        # Clean up
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print_gpu_memory("Final GPU memory")


def pi_deeponet_test(best_model_path, config, case_indices=None, num_samples_to_plot=1E8):
    """
    Using the trained DeepONet model for testing on the test dataset.

    Note: The test dataset is not shuffled, so the first nCells (e.g., 100) samples are from the first case, 
    the next nCells samples are from the second case, etc. The total number of samples in the test dataset is nCases * nCells. We assume each case has the same number of cells (nCells).
    
    Args:
        best_model_path: Path to the best model checkpoint
        config: Configuration object (Config instance)
        case_indices: Optional list of case indices (1-based) to plot. If None, will randomly select.
        num_samples_to_plot: Number of samples to plot (default: 1E8). Only used if case_indices is None. The default value is 1E8, which means to plot all samples.
    Returns:
        None
    """

    # Check if config is an instance of Config
    if not isinstance(config, Config):
        raise ValueError("config must be an instance of Config")

    #print("\n=== PI-DeepONet Test ===")
    
    # Configuration and paths
    checkpoint_path = best_model_path 

    # Get the number of cells in the SRH-2D cases from the config file
    nCells = config.get('data.nCells')
    if nCells is None:
        print("Warning: data.nCells not found in config, exiting...")
        exit()

    print("\n")
    print("Starting PI-DeepONet test...")
    print("================================================")
    print("Best model path: ", best_model_path)
    print("Test dataset path: ", config.get('data.deeponet.test_data_path'))
    print("Number of cells: ", nCells)
    print("================================================")
    print("\n")
    
    print("Performing some sanity checks on the test dataset...")

    # First, peek at the test dataset to get input dimensions
    print("Determining input dimensions from test data...")
    test_data_path = config.get('data.deeponet.test_data_path')

    try:
        test_dataset = PI_SWE_DeepONetDataset(
            data_path=test_data_path,
            config=config)            

        # Get a sample to determine dimensions
        sample_branch, sample_trunk, sample_output = test_dataset[0]
        branch_dim = sample_branch.shape[0]
        trunk_dim = sample_trunk.shape[0]
        output_dim = sample_output.shape[0]
        print(f"Detected branch_dim={branch_dim}, trunk_dim={trunk_dim}, output_dim={output_dim}")
    except (IndexError, FileNotFoundError) as e:
        # Default to dimensions from training (assuming 10 for branch and 3 for trunk)
        print(f"Could not determine dimensions from test data: {e}")
        exit()
    
    # Create model with default configuration and check the compatibility of the model with the test data
    print("\n\nCreating the trained model...")
    model = PI_SWE_DeepONetModel(config=config)
    model.check_model_input_output_dimensions(branch_dim, trunk_dim, output_dim)
    
    # Create trainer
    trainer = PI_SWE_DeepONetTrainer(model, config=config)
    
    # Load the trained model checkpoint
    try:
        trainer.load_checkpoint(checkpoint_path)
        print(f"Successfully loaded checkpoint from {checkpoint_path}")
    except (FileNotFoundError, RuntimeError) as e:
        print(f"Error loading checkpoint: {e}, exiting...")        
        exit()
    
    # Load test dataset
    print("\n\nLoading test dataset...")
    test_loader = DataLoader(
            test_dataset,
            batch_size=config.get('training.batch_size'),
            shuffle=False, # Do not shuffle the test dataset because the order of the samples is important
            num_workers=config.get('training.num_workers', 0),
            pin_memory=config.get('training.pin_memory', False)
        )

    # Get the all_DeepONet_stats
    all_DeepONet_stats = test_dataset.get_deeponet_stats()

    # Make sure the normalization method for outputs is "z-score" (currently only "z-score" is supported)
    if all_DeepONet_stats['all_DeepONet_branch_trunk_outputs_stats']['normalization_method']['outputs'] != "z-score":
        raise ValueError("Output normalization method is not supported. Currently only 'z-score' is supported.")

    output_mean = all_DeepONet_stats['all_DeepONet_branch_trunk_outputs_stats']['outputs']['mean']
    output_std = all_DeepONet_stats['all_DeepONet_branch_trunk_outputs_stats']['outputs']['std']
    
    # Evaluate model on test dataset
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0    
    all_predictions = []
    all_targets = []
    
    print("Running predictions on test dataset...")
    sample_count = 0
    
    with torch.no_grad():  # Disable gradient calculation for inference
        for branch_input, trunk_input, target in test_loader:
            # Move data to device
            branch_input = branch_input.to(trainer.device)
            trunk_input = trunk_input.to(trainer.device)
            target = target.to(trainer.device)
            
            # Forward pass
            output = model(branch_input, trunk_input)
            
            # Calculate loss
            loss = trainer.loss_fn(output, target)
            total_loss += loss.item()

            # Store predictions and targets for analysis
            all_predictions.append(output.cpu().numpy())
            all_targets.append(target.cpu().numpy())
            
            # Count samples
            sample_count += len(branch_input)

    # Calculate average test loss
    avg_test_loss = total_loss / len(test_loader)
    print(f"\nTest Loss (average) on test dataset (normalized): {avg_test_loss:.6f}")
    
    # Concatenate all predictions and targets
    all_predictions = np.vstack(all_predictions)
    all_targets = np.vstack(all_targets)
    
    # Calculate additional metrics (e.g., MSE for each output dimension)
    mse_per_dim = np.mean((all_predictions - all_targets) ** 2, axis=0)
    print(f"\nMSE per output dimension (normalized): {mse_per_dim}")

    # Denormalize the predictions and targets
    # Assuming the outputs are normalized as (x - mean) / std
    denorm_predictions = all_predictions * output_std + output_mean
    denorm_targets = all_targets * output_std + output_mean
    
    # Calculate metrics on denormalized values
    denorm_mse_per_dim = np.mean((denorm_predictions - denorm_targets) ** 2, axis=0)
    print(f"MSE per output dimension (denormalized): {denorm_mse_per_dim}\n")

    # Save the denormalized predictions and targets to a npz file
    np.savez(f'{test_data_path}/test_results.npz', denorm_predictions=denorm_predictions, denorm_targets=denorm_targets)

    results_summary = {
        'test_loss': avg_test_loss,
        'normalized': {
           'mse_per_dim': mse_per_dim.tolist(),
       },
       'denormalized': {
           'mse_per_dim': denorm_mse_per_dim.tolist(),
       },
        'samples': {
            'num_samples': sample_count,
            'batch_size': trainer.batch_size
        }
    }
    
    # Save test results
    with open(f'{test_data_path}/test_results_summary.json', 'w') as f:
        json.dump(results_summary, f, indent=4)

    # Convert the test results to vtk files
    pi_deeponet_convert_test_results_to_vtk(config, case_indices=case_indices, num_samples_to_plot=num_samples_to_plot)
    
    print("\nFinished testing the model.")


def pi_deeponet_convert_test_results_to_vtk(config, case_indices=None, num_samples_to_plot=1E8):
    """Save the test results to vtk files.

    Args:
        config: Configuration object (Config instance)
        case_indices: Optional list of case indices (1-based) to plot. If None, will randomly select.
        num_samples_to_plot: Number of samples to plot (default: 1E8). Only used if case_indices is None. The default value is 1E8, which means to plot all samples.

    Returns:
        None
    """

    print("\n\nConverting test results to vtk files...")
    print("================================================")
    
    # Check if config is an instance of Config
    if not isinstance(config, Config):
        raise ValueError("config must be an instance of Config")

    # Get the test data path from the config file
    test_data_path = config.get('data.deeponet.test_data_path')

    # Get the number of cells in the SRH-2D cases from the config file
    nCells = config.get('data.nCells')
    if nCells is None:
        print("Warning: data.nCells not found in config, exiting...")
        exit()

    # Get the template vtk file from the config file
    template_vtk_file = config.get('data.template_vtk_file')
    if template_vtk_file is None:
        print("Warning: data.template_vtk_file not found in config, exiting...")
        exit()

    # Load predictions and targets from file {test_data_path}/test_results.npz
    test_results = np.load(f'{test_data_path}/test_results.npz')
    denorm_predictions = test_results["denorm_predictions"]
    denorm_targets = test_results["denorm_targets"]

    #print("denorm_predictions.shape: ", denorm_predictions.shape)
    #print("denorm_targets.shape: ", denorm_targets.shape)

    # Load the split indices (1-based) from file {test_data_path}/../split_indices.json
    split_indices = json.load(open(f'{test_data_path}/../split_indices.json')) 
    test_indices = split_indices["test_indices"]

    nCases_in_test_dataset = len(test_indices)
    
    # Select cases to plot
    if case_indices is None:
        num_samples_to_plot = min(num_samples_to_plot, nCases_in_test_dataset)
        # Randomly select some samples from the test indices (1-based)
        random_indices = np.random.choice(test_indices, size=num_samples_to_plot, replace=False).tolist()
    else:
        # Use provided case indices
        random_indices = case_indices
        num_samples_to_plot = len(case_indices)

    print("selected random indices for plotting: ", random_indices)

    # Find the corresponding indices for each random index
    corresponding_indices = []
    for rand_idx in random_indices:
        # Find the position of this index in the test_indices array
        for i, test_idx in enumerate(test_indices):
            if test_idx == rand_idx:
                corresponding_indices.append(i)
                break
    
    #print("corresponding indices: ", corresponding_indices)

    for i in range(num_samples_to_plot):
        # Get the current case's results: h, u, v
        h_pred = denorm_predictions[corresponding_indices[i]*nCells:(corresponding_indices[i]+1)*nCells,0]
        u_pred = denorm_predictions[corresponding_indices[i]*nCells:(corresponding_indices[i]+1)*nCells,1]
        v_pred = denorm_predictions[corresponding_indices[i]*nCells:(corresponding_indices[i]+1)*nCells,2]

        h_target = denorm_targets[corresponding_indices[i]*nCells:(corresponding_indices[i]+1)*nCells,0]
        u_target = denorm_targets[corresponding_indices[i]*nCells:(corresponding_indices[i]+1)*nCells,1]
        v_target = denorm_targets[corresponding_indices[i]*nCells:(corresponding_indices[i]+1)*nCells,2]

        # Compute the difference between the predicted and target results
        h_diff = h_pred - h_target
        u_diff = u_pred - u_target
        v_diff = v_pred - v_target

        # Read the empty vtk file which only has the mesh information, then add the results to the vtk file and save it
        # Read the VTK file
        reader = vtk.vtkUnstructuredGridReader()
        reader.SetFileName(template_vtk_file)
        reader.Update()
        mesh = reader.GetOutput()
        
        # Remove any existing point data and cell data from the template file
        # We only want the mesh structure (points and cells), not any existing data arrays
        mesh.GetPointData().Initialize()
        mesh.GetCellData().Initialize()

        # Create cell data arrays for the predicted results
        h_pred_array = numpy_to_vtk(h_pred, deep=True, array_type=vtk.VTK_FLOAT)
        h_pred_array.SetName("Water_Depth_Pred")
        
        u_pred_array = numpy_to_vtk(u_pred, deep=True, array_type=vtk.VTK_FLOAT)
        u_pred_array.SetName("X_Velocity_Pred")
        
        v_pred_array = numpy_to_vtk(v_pred, deep=True, array_type=vtk.VTK_FLOAT)
        v_pred_array.SetName("Y_Velocity_Pred")

        # Create cell data arrays for the target results
        h_target_array = numpy_to_vtk(h_target, deep=True, array_type=vtk.VTK_FLOAT)
        h_target_array.SetName("Water_Depth_Target")
        
        u_target_array = numpy_to_vtk(u_target, deep=True, array_type=vtk.VTK_FLOAT)
        u_target_array.SetName("X_Velocity_Target")
        
        v_target_array = numpy_to_vtk(v_target, deep=True, array_type=vtk.VTK_FLOAT)
        v_target_array.SetName("Y_Velocity_Target")

        # Create velocity vectors for predicted values
        velocity_pred = np.column_stack((u_pred, v_pred, np.zeros_like(u_pred)))  # Add z-component as 0
        velocity_pred_array = numpy_to_vtk(velocity_pred, deep=True, array_type=vtk.VTK_FLOAT)
        velocity_pred_array.SetNumberOfComponents(3)
        velocity_pred_array.SetName("Velocity_Pred")

        # Create velocity vectors for target values
        velocity_target = np.column_stack((u_target, v_target, np.zeros_like(u_target)))  # Add z-component as 0
        velocity_target_array = numpy_to_vtk(velocity_target, deep=True, array_type=vtk.VTK_FLOAT)
        velocity_target_array.SetNumberOfComponents(3)
        velocity_target_array.SetName("Velocity_Target")

        # Create velocity difference vectors
        velocity_diff = np.column_stack((u_diff, v_diff, np.zeros_like(u_diff)))  # Add z-component as 0
        velocity_diff_array = numpy_to_vtk(velocity_diff, deep=True, array_type=vtk.VTK_FLOAT)
        velocity_diff_array.SetNumberOfComponents(3)
        velocity_diff_array.SetName("Velocity_Diff")

        # Create water depth difference array
        h_diff_array = numpy_to_vtk(h_diff, deep=True, array_type=vtk.VTK_FLOAT)
        h_diff_array.SetName("Water_Depth_Diff")

        # Create x-velocity difference array
        u_diff_array = numpy_to_vtk(u_diff, deep=True, array_type=vtk.VTK_FLOAT)
        u_diff_array.SetName("X_Velocity_Diff")

        # Create y-velocity difference array
        v_diff_array = numpy_to_vtk(v_diff, deep=True, array_type=vtk.VTK_FLOAT)
        v_diff_array.SetName("Y_Velocity_Diff") 

        # Add the arrays to the mesh as cell data
        mesh.GetCellData().AddArray(h_pred_array)
        mesh.GetCellData().AddArray(u_pred_array)
        mesh.GetCellData().AddArray(v_pred_array)
        mesh.GetCellData().AddArray(h_target_array)
        mesh.GetCellData().AddArray(u_target_array)
        mesh.GetCellData().AddArray(v_target_array)
        mesh.GetCellData().AddArray(velocity_pred_array)
        mesh.GetCellData().AddArray(velocity_target_array)

        # Add the difference arrays to the mesh as cell data
        mesh.GetCellData().AddArray(velocity_diff_array)
        mesh.GetCellData().AddArray(h_diff_array)
        mesh.GetCellData().AddArray(u_diff_array)
        mesh.GetCellData().AddArray(v_diff_array)

        # Create a writer
        writer = vtk.vtkUnstructuredGridWriter()
        writer.SetFileName(f"{test_data_path}/case_{random_indices[i]}_test_results.vtk")
        writer.SetInputData(mesh)
        writer.Write()

        print(f"Saved results for case {random_indices[i]} to {test_data_path}/case_{random_indices[i]}_test_results.vtk")


def pi_deeponet_plot_training_history(history_file_name, save_path=None):
    """Plot the training history from the history json file.
    
    Args:
        history_file_name: Path to the history JSON file
        save_path: Optional path to save the plot. If None, saved to the current directory.
    """

    # Load the history json file
    with open(history_file_name, 'r') as f:
        history = json.load(f)

    # Read the training loss and validation loss from the history
    if 'training_loss_history' in history:
        train_loss = history['training_loss_history']        
    else:
        raise KeyError("History file must contain 'training_loss_history'")

    if 'validation_loss_history' in history:
        val_loss = history['validation_loss_history']
    else:
        raise KeyError("History file must contain 'validation_loss_history'")

    if 'training_component_loss_history' in history:
        training_component_loss_history = history['training_component_loss_history']

        #get the component loss values from the training_component_loss_history
        deeponet_data_loss = training_component_loss_history['deeponet_data_loss']
        pinn_pde_loss = training_component_loss_history['pinn_pde_loss']
        pinn_pde_loss_cty = training_component_loss_history['pinn_pde_loss_cty']
        pinn_pde_loss_mom_x = training_component_loss_history['pinn_pde_loss_mom_x']
        pinn_pde_loss_mom_y = training_component_loss_history['pinn_pde_loss_mom_y']
        pinn_initial_loss = training_component_loss_history['pinn_initial_loss']
        pinn_boundary_loss = training_component_loss_history['pinn_boundary_loss']
    else:
        raise KeyError("History file must contain 'training_component_loss_history'")

    if 'adaptive_weight_history' in history:
        adaptive_weight_history = history['adaptive_weight_history']

        #get the adaptive weight values from the adaptive_weight_history
        adaptive_weight_deeponet_data_loss = adaptive_weight_history['deeponet_data_loss_weight']
        adaptive_weight_pinn_pde_loss = adaptive_weight_history['deeponet_pinn_loss_weight']
        adaptive_weight_pde_continuity = adaptive_weight_history['pde_continuity_weight']
        adaptive_weight_pde_momentum_x = adaptive_weight_history['pde_momentum_x_weight']
        adaptive_weight_pde_momentum_y = adaptive_weight_history['pde_momentum_y_weight']
    else:
        raise KeyError("History file must contain 'adaptive_weight_history'")

    # Plot the training history (instead of train_loss, we pass in deeponet_data_loss because train_loss is the total loss=data_loss+pinn_loss)
    #_pi_deeponet_plot_training_history_training_loss_and_validation_loss(train_loss, val_loss, save_path)
    _pi_deeponet_plot_training_history_training_loss_and_validation_loss(deeponet_data_loss, val_loss, save_path)

    _pi_deeponet_plot_training_component_loss_history_data_loss_and_pinn_loss(deeponet_data_loss, pinn_pde_loss, adaptive_weight_deeponet_data_loss, adaptive_weight_pinn_pde_loss, save_path)

    _pi_deeponet_plot_training_component_loss_history_pde_loss_continuity_and_momentum(pinn_pde_loss_cty, pinn_pde_loss_mom_x, pinn_pde_loss_mom_y, adaptive_weight_pde_continuity, adaptive_weight_pde_momentum_x, adaptive_weight_pde_momentum_y, save_path)
    

def _pi_deeponet_plot_training_history_training_loss_and_validation_loss(train_loss, val_loss, save_path=None):
    """Plot the training history from the training loss and validation loss lists.
    
    Args:
        train_loss: List of training loss values
        val_loss: List of validation loss values
        save_path: Optional path to save the plot. If None, saved to the current directory.
    """

    # Length of the training history
    num_epochs = len(train_loss)

    epochs = np.arange(1, num_epochs + 1)

    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')

    # Add axis labels
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)

    # Make y axis in log scale (optional, uncomment if needed)
    plt.yscale('log')

    # Axis ticks format
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Add legend
    plt.legend(fontsize=14)
    plt.tight_layout()

    # Save or show the plot
    if save_path is None:
        save_path = 'training_history.png'
    else:
        save_path = os.path.join(save_path, 'training_history.png')

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def _pi_deeponet_plot_training_component_loss_history_data_loss_and_pinn_loss(deeponet_data_loss, pinn_pde_loss, adaptive_weight_deeponet_data_loss, adaptive_weight_pinn_pde_loss, save_path=None):
    """Plot the training component loss history from the training component loss lists.
    Creates a figure with two subplots (two rows):
    - Top: Losses without weights
    - Bottom: Losses with adaptive weights applied

    Args:
        deeponet_data_loss: List of deeponet data loss values
        pinn_pde_loss: List of pinn pde loss values
        adaptive_weight_deeponet_data_loss: List of adaptive weight deeponet data loss values
        adaptive_weight_pinn_pde_loss: List of adaptive weight pinn pde loss values
        save_path: Optional path to save the plot. If None, saved to the current directory.
    """

    # Length of the training component loss history
    num_epochs = len(deeponet_data_loss)
    epochs = np.arange(1, num_epochs + 1)

    # Convert lists to numpy arrays
    deeponet_data_loss = np.array(deeponet_data_loss)
    pinn_pde_loss = np.array(pinn_pde_loss)
    
    # Weight history may have one extra entry (recorded before training starts)
    # Align weights with losses by taking the last num_epochs entries
    adaptive_weight_deeponet_data_loss = np.array(adaptive_weight_deeponet_data_loss[-num_epochs:])
    adaptive_weight_pinn_pde_loss = np.array(adaptive_weight_pinn_pde_loss[-num_epochs:])

    # Create figure with two subplots (two rows)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # Top subplot: Losses without weights
    ax1.plot(epochs, deeponet_data_loss, label='DeepONet Data Loss', linewidth=2)
    ax1.plot(epochs, pinn_pde_loss, label='PINN PDE Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=14)
    ax1.set_ylabel('Loss', fontsize=14)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Component Losses (Unweighted)', fontsize=14, fontweight='bold')

    # Bottom subplot: Losses with adaptive weights
    # Need to deal with the case of resuming training from a checkpoint. In this case, the length of the adaptive weights list is shorter than the length of the losses list. In this case, we need to pad the adaptive weights list on the front with zero values.
    if len(adaptive_weight_deeponet_data_loss) < len(deeponet_data_loss):
        adaptive_weight_deeponet_data_loss = np.concatenate([np.zeros(len(deeponet_data_loss) - len(adaptive_weight_deeponet_data_loss)), adaptive_weight_deeponet_data_loss])
    if len(adaptive_weight_pinn_pde_loss) < len(pinn_pde_loss):
        adaptive_weight_pinn_pde_loss = np.concatenate([np.zeros(len(pinn_pde_loss) - len(adaptive_weight_pinn_pde_loss)), adaptive_weight_pinn_pde_loss])

    ax2.plot(epochs, deeponet_data_loss * adaptive_weight_deeponet_data_loss, label='DeepONet Data Loss (Weighted)', linewidth=2)
    ax2.plot(epochs, pinn_pde_loss * adaptive_weight_pinn_pde_loss, label='PINN PDE Loss (Weighted)', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=14)
    ax2.set_ylabel('Weighted Loss', fontsize=14)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Component Losses (Weighted by Adaptive Weights)', fontsize=14, fontweight='bold')

    plt.tight_layout()

    # Save or show the plot
    if save_path is None:
        save_path = 'data_loss_and_pinn_loss.png'
    else:
        save_path = os.path.join(save_path, 'data_loss_and_pinn_loss.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    

def _pi_deeponet_plot_training_component_loss_history_pde_loss_continuity_and_momentum(pinn_pde_loss_cty, pinn_pde_loss_mom_x, pinn_pde_loss_mom_y, adaptive_weight_pde_continuity, adaptive_weight_pde_momentum_x, adaptive_weight_pde_momentum_y, save_path=None):
    """Plot the training component loss history for PDE continuity and momentum losses.
    Creates a figure with two subplots (two rows):
    - Top: PDE component losses without weights
    - Bottom: PDE component losses with adaptive weights applied

    Args:
        pinn_pde_loss_cty: List of pinn pde loss continuity values
        pinn_pde_loss_mom_x: List of pinn pde loss momentum x values
        pinn_pde_loss_mom_y: List of pinn pde loss momentum y values
        adaptive_weight_pde_continuity: List of adaptive weight pinn pde loss continuity values
        adaptive_weight_pde_momentum_x: List of adaptive weight pinn pde loss momentum x values
        adaptive_weight_pde_momentum_y: List of adaptive weight pinn pde loss momentum y values
        save_path: Optional path to save the plot. If None, saved to the current directory.
    """

    # Length of the training component loss history
    num_epochs = len(pinn_pde_loss_cty)
    epochs = np.arange(1, num_epochs + 1)

    # Convert lists to numpy arrays
    pinn_pde_loss_cty = np.array(pinn_pde_loss_cty)
    pinn_pde_loss_mom_x = np.array(pinn_pde_loss_mom_x)
    pinn_pde_loss_mom_y = np.array(pinn_pde_loss_mom_y)
    
    # Weight history may have one extra entry (recorded before training starts)
    # Align weights with losses by taking the last num_epochs entries
    adaptive_weight_pde_continuity = np.array(adaptive_weight_pde_continuity[-num_epochs:])
    adaptive_weight_pde_momentum_x = np.array(adaptive_weight_pde_momentum_x[-num_epochs:])
    adaptive_weight_pde_momentum_y = np.array(adaptive_weight_pde_momentum_y[-num_epochs:])

    # Create figure with two subplots (two rows)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # Top subplot: PDE component losses without weights
    ax1.plot(epochs, pinn_pde_loss_cty, label='PINN PDE Loss Continuity', linewidth=2)
    ax1.plot(epochs, pinn_pde_loss_mom_x, label='PINN PDE Loss Momentum X', linewidth=2)
    ax1.plot(epochs, pinn_pde_loss_mom_y, label='PINN PDE Loss Momentum Y', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=14)
    ax1.set_ylabel('Loss', fontsize=14)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_title('PDE Component Losses (Unweighted)', fontsize=14, fontweight='bold')

    # Bottom subplot: PDE component losses with adaptive weights
    # Need to deal with the case of resuming training from a checkpoint. In this case, the length of the adaptive weights list is shorter than the length of the losses list. In this case, we need to pad the adaptive weights list on the front with zero values.
    if len(adaptive_weight_pde_continuity) < len(pinn_pde_loss_cty):
        adaptive_weight_pde_continuity = np.concatenate([np.zeros(len(pinn_pde_loss_cty) - len(adaptive_weight_pde_continuity)), adaptive_weight_pde_continuity])
    if len(adaptive_weight_pde_momentum_x) < len(pinn_pde_loss_mom_x):
        adaptive_weight_pde_momentum_x = np.concatenate([np.zeros(len(pinn_pde_loss_mom_x) - len(adaptive_weight_pde_momentum_x)), adaptive_weight_pde_momentum_x])
    if len(adaptive_weight_pde_momentum_y) < len(pinn_pde_loss_mom_y):
        adaptive_weight_pde_momentum_y = np.concatenate([np.zeros(len(pinn_pde_loss_mom_y) - len(adaptive_weight_pde_momentum_y)), adaptive_weight_pde_momentum_y])
        
    ax2.plot(epochs, pinn_pde_loss_cty * adaptive_weight_pde_continuity, label='PINN PDE Loss Continuity (Weighted)', linewidth=2)
    ax2.plot(epochs, pinn_pde_loss_mom_x * adaptive_weight_pde_momentum_x, label='PINN PDE Loss Momentum X (Weighted)', linewidth=2)
    ax2.plot(epochs, pinn_pde_loss_mom_y * adaptive_weight_pde_momentum_y, label='PINN PDE Loss Momentum Y (Weighted)', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=14)
    ax2.set_ylabel('Weighted Loss', fontsize=14)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_title('PDE Component Losses (Weighted by Adaptive Weights)', fontsize=14, fontweight='bold')

    plt.tight_layout()

    # Save or show the plot
    if save_path is None:
        save_path = 'pde_loss_continuity_and_momentum.png'
    else:
        save_path = os.path.join(save_path, 'pde_loss_continuity_and_momentum.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()



def pi_deeponet_application_create_model_application_dataset(config):
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

    # Read the header row to determine the number of parameters
    with open(os.path.join(application_dir, application_parameters_file), 'r') as f:
        header_line = f.readline().strip()
        n_params = len(header_line.split())  # Count the number of column headers
    
    print(f"Number of parameters (from header) for application: {n_params}")
    
    # Load the application parameters (skip the header row)
    application_parameters = np.loadtxt(os.path.join(application_dir, application_parameters_file), skiprows=1)
    print(f"Application parameters: {application_parameters}")
    
    # Ensure application_parameters is 2D
    # If 1D, it could be:
    # - One case with multiple parameters: reshape to (1, n_params) if length == n_params
    # - Multiple cases with one parameter: reshape to (n_cases, 1) if length != n_params
    if application_parameters.ndim == 1:
        if len(application_parameters) == n_params:
            # One case with multiple parameters: reshape to (1, n_params)
            print(f"Application parameters is 1D with {len(application_parameters)} values (matches {n_params} parameters). Reshaping to (1, {n_params})...")
            application_parameters = application_parameters.reshape(1, -1)
        else:
            # Multiple cases with one parameter: reshape to (n_cases, 1)
            print(f"Application parameters is 1D with {len(application_parameters)} values (does not match {n_params} parameters). Reshaping to ({len(application_parameters)}, 1)...")
            application_parameters = application_parameters.reshape(-1, 1)
    # If already 2D, it means multiple rows (multiple cases), so keep as is
        
    print(f"Application parameters after reshaping: {application_parameters}")
    print(f"Application parameters shape: {application_parameters.shape}")

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
    # application_parameters should have shape (n_cases, n_features). n_cases is the number of application cases. n_features is the number of application parameters.
    # For each case, we tile the parameters for all cells
    # Note: application_parameters is already 2D at this point (reshaped earlier if needed)
    n_cases = application_parameters.shape[0]
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

    print(f"All DeepONet stats: {all_DeepONet_stats}")
    
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

def pi_deeponet_application_compute_distance_between_application_and_training_data(config):
    """
    Compute the distance between the application data and the training data (the branch inputs).

    Currently, we compute the Wasserstein Distance between the application data and training distribution.

    $$D_W(\mathbf{q}) = |\mathbf{q} - \mathbf{q}_{\mathrm{closest;train}}|_1.$$

    Equivalent to "distance to nearest training example", but grounded in optimal transport theory.

    The application parameter file and the training parameter file are in the dat format. Their file names should be specified in the config file.

    Args:
        config: The configuration object.

    """

    # Check if config is an instance of Config
    if not isinstance(config, Config):
        raise ValueError("config must be an instance of Config")

   
    # Get the application directory
    application_dir = config['application']['application_dir']

    # Get the application parameters file
    application_parameters_file = config['application']['application_parameters_file']
    
    # Read the header row to determine the number of parameters
    with open(os.path.join(application_dir, application_parameters_file), 'r') as f:
        header_line = f.readline().strip()
        n_params = len(header_line.split())  # Count the number of column headers
    
    # Load the application parameters (skip the header row)
    application_parameters = np.loadtxt(os.path.join(application_dir, application_parameters_file), skiprows=1)
    
    # Ensure application_parameters is 2D
    if application_parameters.ndim == 1:
        if len(application_parameters) == n_params:
            # One case with multiple parameters: reshape to (1, n_params)
            application_parameters = application_parameters.reshape(1, -1)
        else:
            # Multiple cases with one parameter: reshape to (n_cases, 1)
            application_parameters = application_parameters.reshape(-1, 1)
    
    print(f"Application parameters shape: {application_parameters.shape}")
    
    # Get the training parameters file
    training_parameters_file = config['application']['training_parameters_file']
    
    # Read the header row to determine the number of parameters (should match application)
    with open(os.path.join(application_dir, training_parameters_file), 'r') as f:
        header_line = f.readline().strip()
        n_params_train = len(header_line.split())
    
    if n_params_train != n_params:
        raise ValueError(f"Number of parameters in training file ({n_params_train}) does not match application file ({n_params})")
    
    # Load the training parameters (skip the header row)
    training_parameters = np.loadtxt(os.path.join(application_dir, training_parameters_file), skiprows=1)
    
    # Ensure training_parameters is 2D
    if training_parameters.ndim == 1:
        if len(training_parameters) == n_params:
            training_parameters = training_parameters.reshape(1, -1)
        else:
            training_parameters = training_parameters.reshape(-1, 1)
    
    print(f"Training parameters shape: {training_parameters.shape}")
    
    # Load normalization stats from the training data
    stats_path = config['application']['train_val_test_stats_path']
    if not os.path.exists(os.path.join(application_dir, stats_path)):
        raise FileNotFoundError(f"Normalization stats file not found: {os.path.join(application_dir, stats_path)}")
    
    with open(os.path.join(application_dir, stats_path), 'r') as f:
        all_DeepONet_stats = json.load(f)
    
    # Get normalization methods and stats for branch inputs
    normalization_specs = all_DeepONet_stats['all_DeepONet_branch_trunk_outputs_stats']
    branch_normalization_method = normalization_specs['normalization_method']['branch_inputs']
    branch_stats = normalization_specs['branch_inputs']
    
    # Normalize application and training parameters (currently only z-score is supported)
    if branch_normalization_method == 'z-score':
        branch_mean = np.array(branch_stats['mean'])
        branch_std = np.array(branch_stats['std'])
        application_parameters_norm = (application_parameters - branch_mean) / branch_std
        training_parameters_norm = (training_parameters - branch_mean) / branch_std
    else:
        raise ValueError(f"Invalid branch inputs normalization method: {branch_normalization_method}")
    
    print(f"Normalized application parameters shape: {application_parameters_norm.shape}")
    print(f"Normalized training parameters shape: {training_parameters_norm.shape}")
    
    # Compute the Wasserstein Distance for each application parameter row
    # Based on the formula: D_W(q) = |q - q_closest_train|_1
    # This computes the L1 distance to the nearest training example for each application sample
    
    n_application_cases = application_parameters_norm.shape[0]
    wasserstein_distances = []
    
    for i in range(n_application_cases):
        print(f"Computing Wasserstein Distance for case {i+1} of {n_application_cases}...")

        # Get the current application parameter row
        app_param = application_parameters_norm[i, :].reshape(1, -1)  # Shape: (1, n_params)
        print(f"Application parameter row {i+1}: {app_param}")
        
        # Compute pairwise L1 distances between this application sample and all training samples
        # cdist with metric='cityblock' computes L1 (Manhattan) distance
        distances = cdist(app_param, training_parameters_norm, metric='cityblock')
        
        # Find the minimum distance to any training sample (this is the Wasserstein distance)
        min_distance = np.min(distances)
        wasserstein_distances.append(float(min_distance))
        
        print(f"Case {i+1}: Wasserstein Distance = {min_distance:.6f}")
    
    # Compute statistics across all application cases
    wasserstein_distances_array = np.array(wasserstein_distances)
    mean_distance = np.mean(wasserstein_distances_array)
    min_distance = np.min(wasserstein_distances_array)
    max_distance = np.max(wasserstein_distances_array)
    median_distance = np.median(wasserstein_distances_array)
    std_distance = np.std(wasserstein_distances_array)
    
    print(f"\nWasserstein Distance Statistics (across all application cases):")
    print(f"  Mean: {mean_distance:.6f}")
    print(f"  Min: {min_distance:.6f}")
    print(f"  Max: {max_distance:.6f}")
    print(f"  Median: {median_distance:.6f}")
    print(f"  Std: {std_distance:.6f}")
    
    # Save the results
    distance_results = {
        'wasserstein_distance_per_case': wasserstein_distances,
        'statistics': {
            'mean': float(mean_distance),
            'min': float(min_distance),
            'max': float(max_distance),
            'median': float(median_distance),
            'std': float(std_distance)
        },
        'normalization_method': branch_normalization_method
    }
    
    results_file = os.path.join(application_dir, 'wasserstein_distance_results.json')
    with open(results_file, 'w') as f:
        json.dump(distance_results, f, indent=4)
    print(f"\nDistance results saved to {results_file}")

    return wasserstein_distances_array

def pi_deeponet_application_run_model_application(config):
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
    dataset, application_parameters = pi_deeponet_application_create_model_application_dataset(config)

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


def pi_deeponet_application_compare_model_application_results_with_simulation_results(config, system_name):
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
    
    # Make the diff_vtks directory
    diff_vtks_dir = os.path.join(application_dir, "diff_vtks")
    os.makedirs(diff_vtks_dir, exist_ok=True)

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

    # Get the number of cases from the dataset HDF5 file metadata
    application_data_dir = os.path.join(application_dir, 'DeepONet')
    h5_file_path = os.path.join(application_data_dir, 'data.h5')
    with h5py.File(h5_file_path, 'r') as f:
        n_cases = f.attrs['n_cases']
    
    print(f"Number of cases: {n_cases}")

    # Get the normalization statistics from the json file
    with open(os.path.join(application_dir, config['application']['train_val_test_stats_path']), "r") as f:
        normalization_stats = json.load(f)
    normalization_method = normalization_stats["all_DeepONet_branch_trunk_outputs_stats"]["normalization_method"]
    output_normalization_method = normalization_method["outputs"]
    h_mean = normalization_stats["all_data_stats"]["h_mean"]
    h_std = normalization_stats["all_data_stats"]["h_std"]
    u_mean = normalization_stats["all_data_stats"]["u_mean"]
    u_std = normalization_stats["all_data_stats"]["u_std"]
    v_mean = normalization_stats["all_data_stats"]["v_mean"]
    v_std = normalization_stats["all_data_stats"]["v_std"]
    velocity_magnitude_mean = normalization_stats["all_data_stats"]["Umag_mean"]
    velocity_magnitude_std = normalization_stats["all_data_stats"]["Umag_std"]

    diff_h_mse_list = []
    diff_u_mse_list = []
    diff_v_mse_list = []
    diff_velocity_magnitude_mse_list = []

    diff_h_normalized_mse_list = []
    diff_u_normalized_mse_list = []
    diff_v_normalized_mse_list = []
    diff_velocity_magnitude_normalized_mse_list = []

    # Get the simulation results directory
    simulation_results_dir = os.path.join(config['application']['application_dir'], 'simulated_vtks')
    # Get the model application results directory
    predictions_vtks_dir = os.path.join(config['application']['application_dir'], 'predictions_vtks')

    # Loop over all cases 
    for case_idx in range(n_cases):
        # Get the simulation results
        simulation_results = os.path.join(simulation_results_dir, f'case_{str(case_idx + 1).zfill(6)}.vtk')

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

        #print(f"predictions_velocity.shape: {predictions_velocity.shape}")
        #print(f"simulation_velocity.shape: {simulation_velocity.shape}")
        
        # Compute velocity magnitude from predictions if not available
        # predictions_velocity has shape (n_cells, 3) with components [u, v, 0]
        predictions_velocity_magnitude = np.sqrt(predictions_u**2 + predictions_v**2)

        # compute the normalized variables for both simulation and predictions
        if output_normalization_method == 'z-score':
            predictions_h_normalized = (predictions_h - h_mean) / h_std
            predictions_u_normalized = (predictions_u - u_mean) / u_std
            predictions_v_normalized = (predictions_v - v_mean) / v_std
            predictions_velocity_magnitude_normalized = (predictions_velocity_magnitude - velocity_magnitude_mean) / velocity_magnitude_std

            simulation_h_normalized = (simulation_h - h_mean) / h_std
            simulation_u_normalized = (simulation_u - u_mean) / u_std
            simulation_v_normalized = (simulation_v - v_mean) / v_std
            simulation_velocity_magnitude_normalized = (simulation_velocity_magnitude - velocity_magnitude_mean) / velocity_magnitude_std
        else:
            raise ValueError(f"Invalid output normalization method: {output_normalization_method}")

        # Compute the difference between the simulation and predictions
        diff_h = predictions_h - simulation_h
        diff_u = predictions_u - simulation_u
        diff_v = predictions_v - simulation_v
        diff_velocity_magnitude = predictions_velocity_magnitude - simulation_velocity_magnitude

        # Compute the normalized difference between the simulation and predictions
        diff_h_normalized = predictions_h_normalized - simulation_h_normalized
        diff_u_normalized = predictions_u_normalized - simulation_u_normalized
        diff_v_normalized = predictions_v_normalized - simulation_v_normalized
        diff_velocity_magnitude_normalized = predictions_velocity_magnitude_normalized - simulation_velocity_magnitude_normalized

        # Compute the MSE of the difference
        diff_h_mse = np.mean(diff_h**2)
        diff_u_mse = np.mean(diff_u**2)
        diff_v_mse = np.mean(diff_v**2)
        diff_velocity_magnitude_mse = np.mean(diff_velocity_magnitude**2)

        diff_h_normalized_mse = np.mean(diff_h_normalized**2)
        diff_u_normalized_mse = np.mean(diff_u_normalized**2)
        diff_v_normalized_mse = np.mean(diff_v_normalized**2)
        diff_velocity_magnitude_normalized_mse = np.mean(diff_velocity_magnitude_normalized**2)

        diff_h_mse_list.append(float(diff_h_mse))
        diff_u_mse_list.append(float(diff_u_mse))
        diff_v_mse_list.append(float(diff_v_mse))
        diff_velocity_magnitude_mse_list.append(float(diff_velocity_magnitude_mse))

        diff_h_normalized_mse_list.append(float(diff_h_normalized_mse))
        diff_u_normalized_mse_list.append(float(diff_u_normalized_mse))
        diff_v_normalized_mse_list.append(float(diff_v_normalized_mse))
        diff_velocity_magnitude_normalized_mse_list.append(float(diff_velocity_magnitude_normalized_mse))

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

        # add the normalized difference arrays to the difference mesh
        diff_h_normalized_array = numpy_to_vtk(diff_h_normalized, deep=True, array_type=vtk.VTK_FLOAT)
        diff_h_normalized_array.SetName("Diff_Water_Depth_Normalized")
        difference_mesh.GetCellData().AddArray(diff_h_normalized_array)
        
        diff_u_normalized_array = numpy_to_vtk(diff_u_normalized, deep=True, array_type=vtk.VTK_FLOAT)
        diff_u_normalized_array.SetName("Diff_X_Velocity_Normalized")
        difference_mesh.GetCellData().AddArray(diff_u_normalized_array)
        
        diff_v_normalized_array = numpy_to_vtk(diff_v_normalized, deep=True, array_type=vtk.VTK_FLOAT)
        diff_v_normalized_array.SetName("Diff_Y_Velocity_Normalized")
        difference_mesh.GetCellData().AddArray(diff_v_normalized_array)
        
        diff_velocity_magnitude_normalized_array = numpy_to_vtk(diff_velocity_magnitude_normalized, deep=True, array_type=vtk.VTK_FLOAT)
        diff_velocity_magnitude_normalized_array.SetName("Diff_Velocity_Magnitude_Normalized")
        difference_mesh.GetCellData().AddArray(diff_velocity_magnitude_normalized_array)

        # Add the prediction arrays to the difference mesh
        predictions_h_array = numpy_to_vtk(predictions_h, deep=True, array_type=vtk.VTK_FLOAT)
        predictions_h_array.SetName("Pred_Water_Depth")
        difference_mesh.GetCellData().AddArray(predictions_h_array)
        
        predictions_u_array = numpy_to_vtk(predictions_u, deep=True, array_type=vtk.VTK_FLOAT)
        predictions_u_array.SetName("Pred_X_Velocity")
        difference_mesh.GetCellData().AddArray(predictions_u_array)
        
        predictions_v_array = numpy_to_vtk(predictions_v, deep=True, array_type=vtk.VTK_FLOAT)
        predictions_v_array.SetName("Pred_Y_Velocity")
        difference_mesh.GetCellData().AddArray(predictions_v_array)
        
        predictions_velocity_array = numpy_to_vtk(predictions_velocity, deep=True, array_type=vtk.VTK_FLOAT)
        predictions_velocity_array.SetNumberOfComponents(3)
        predictions_velocity_array.SetName("Pred_Velocity")
        difference_mesh.GetCellData().AddArray(predictions_velocity_array)
        
        predictions_velocity_magnitude_array = numpy_to_vtk(predictions_velocity_magnitude, deep=True, array_type=vtk.VTK_FLOAT)
        predictions_velocity_magnitude_array.SetName("Pred_Velocity_Magnitude")
        difference_mesh.GetCellData().AddArray(predictions_velocity_magnitude_array)

        # Add the simulation arrays to the difference mesh
        simulation_h_array = numpy_to_vtk(simulation_h, deep=True, array_type=vtk.VTK_FLOAT)
        simulation_h_array.SetName("Sim_Water_Depth")
        difference_mesh.GetCellData().AddArray(simulation_h_array)
        
        simulation_velocity_array = numpy_to_vtk(simulation_velocity, deep=True, array_type=vtk.VTK_FLOAT)
        simulation_velocity_array.SetName("Sim_Velocity")
        difference_mesh.GetCellData().AddArray(simulation_velocity_array)
        
        simulation_velocity_magnitude_array = numpy_to_vtk(simulation_velocity_magnitude, deep=True, array_type=vtk.VTK_FLOAT)
        simulation_velocity_magnitude_array.SetName("Sim_Velocity_Magnitude")
        difference_mesh.GetCellData().AddArray(simulation_velocity_magnitude_array)
        
        #create a writer
        difference_writer = vtk.vtkUnstructuredGridWriter()
        difference_writer.SetFileName(os.path.join(application_dir, "diff_vtks", f'case_{case_idx + 1}_diff.vtk'))
        difference_writer.SetInputData(difference_mesh)
        difference_writer.Write()   

    print(f"Diff_h_mse_list: {diff_h_mse_list}")
    print(f"Diff_u_mse_list: {diff_u_mse_list}")
    print(f"Diff_v_mse_list: {diff_v_mse_list}")
    print(f"Diff_velocity_magnitude_mse_list: {diff_velocity_magnitude_mse_list}")

    print(f"Diff_h_normalized_mse_list: {diff_h_normalized_mse_list}")
    print(f"Diff_u_normalized_mse_list: {diff_u_normalized_mse_list}")
    print(f"Diff_v_normalized_mse_list: {diff_v_normalized_mse_list}")
    print(f"Diff_velocity_magnitude_normalized_mse_list: {diff_velocity_magnitude_normalized_mse_list}")
 
    # Save both the original and normalized difference lists to a json file
    with open(os.path.join(application_dir, "diff_lists.json"), "w") as f:
        json.dump({"diff_h_mse_list": diff_h_mse_list, "diff_h_normalized_mse_list": diff_h_normalized_mse_list, "diff_u_mse_list": diff_u_mse_list, "diff_u_normalized_mse_list": diff_u_normalized_mse_list, "diff_v_mse_list": diff_v_mse_list, "diff_v_normalized_mse_list": diff_v_normalized_mse_list, "diff_velocity_magnitude_mse_list": diff_velocity_magnitude_mse_list, "diff_velocity_magnitude_normalized_mse_list": diff_velocity_magnitude_normalized_mse_list}, f, indent=4)
    print(f"Both original and normalized diff lists saved to {os.path.join(application_dir, 'diff_lists.json')}")

def pi_deeponet_application_plot_difference_metrics_against_parameter_distance(config):
    """
    Plot the difference metrics against the parameter distance from the training data.

    The distance is in file wasserstein_distance_results.json.
    The difference metrics are in file diff_lists.json.
    """

    # Get the application directory
    application_dir = config['application']['application_dir']

    # Load the difference lists from the json file
    with open(os.path.join(application_dir, "diff_lists.json"), "r") as f:
        diff_lists = json.load(f)

    case_indices = list(range(1, len(diff_lists["diff_h_mse_list"]) + 1))
    
    # Extract the data (only the normalized difference lists)
    diff_h_normalized_mse_list = diff_lists["diff_h_normalized_mse_list"]
    diff_u_normalized_mse_list = diff_lists["diff_u_normalized_mse_list"]
    diff_v_normalized_mse_list = diff_lists["diff_v_normalized_mse_list"]
    diff_velocity_magnitude_normalized_mse_list = diff_lists["diff_velocity_magnitude_normalized_mse_list"]

    # Load the distance from the json file
    with open(os.path.join(application_dir, "wasserstein_distance_results.json"), "r") as f:
        wasserstein_distance_results = json.load(f)
    distance_list = wasserstein_distance_results["wasserstein_distance_per_case"]

    #diff value lists and distance list should have the same length
    if len(diff_h_normalized_mse_list) != len(distance_list):
        raise ValueError("The length of diff_h_normalized_mse_list and distance_list should be the same")
    if len(diff_velocity_magnitude_normalized_mse_list) != len(distance_list):
        raise ValueError("The length of diff_velocity_magnitude_normalized_mse_list and distance_list should be the same")

    # Order the diff value lists and distance list by the distance
    diff_h_normalized_mse_list = [x for _, x in sorted(zip(distance_list, diff_h_normalized_mse_list))]
    diff_u_normalized_mse_list = [x for _, x in sorted(zip(distance_list, diff_u_normalized_mse_list))]
    diff_v_normalized_mse_list = [x for _, x in sorted(zip(distance_list, diff_v_normalized_mse_list))]
    diff_velocity_magnitude_normalized_mse_list = [x for _, x in sorted(zip(distance_list, diff_velocity_magnitude_normalized_mse_list))]
    distance_list = sorted(distance_list)

    # Plot (scatter) the difference lists (h, u, v and velocity magnitude in four subplots; the top three subplots share the same x axis as the bottom subplot)
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 16), sharex=True)

    scatter_size = 40
    
    # Plot water depth MSE
    ax1.scatter(distance_list, diff_h_normalized_mse_list, color='blue', marker='o', s=scatter_size, label='Water Depth MSE')
    #ax1.set_xlabel('Wasserstein Distance', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Normalized MSE', fontsize=18)
    #ax1.set_title('Water Depth Prediction Error (MSE)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=16)
    #ax1.set_xlim(0, max(distance_list))
    ax1.tick_params(axis='both', labelsize=16)

    # Plot x velocity MSE
    ax2.scatter(distance_list, diff_u_normalized_mse_list, color='green', marker='o', s=scatter_size, label='X Velocity MSE')
    #ax2.set_xlabel('Wasserstein Distance', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Normalized MSE', fontsize=18)
    #ax2.set_title('X Velocity Prediction Error (MSE)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(fontsize=16)
    #ax2.set_xlim(0, max(distance_list))
    ax2.tick_params(axis='both', labelsize=16)

    # Plot y velocity MSE
    ax3.scatter(distance_list, diff_v_normalized_mse_list, color='red', marker='o', s=scatter_size, label='Y Velocity MSE')
    #ax3.set_xlabel('Wasserstein Distance', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Normalized MSE', fontsize=18)
    #ax3.set_title('Y Velocity Prediction Error (MSE)', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.legend(fontsize=16)
    #ax3.set_xlim(0, max(distance_list))
    ax3.tick_params(axis='both', labelsize=16)
    
    # Plot velocity magnitude MSE
    ax4.scatter(distance_list, diff_velocity_magnitude_normalized_mse_list, color='purple', marker='o', s=scatter_size, label='Velocity Magnitude MSE')
    ax4.set_xlabel('Wasserstein Distance', fontsize=18)
    ax4.set_ylabel('Normalized MSE', fontsize=18)
    #ax4.set_title('Velocity Magnitude Prediction Error (MSE)', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.legend(fontsize=16)
    ax4.set_xlim(0, max(distance_list))
    ax4.tick_params(axis='both', labelsize=16)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Save the figure
    output_file = os.path.join(application_dir, 'difference_metrics_against_parameter_distance.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    
    # Show the plot
    #plt.show()

