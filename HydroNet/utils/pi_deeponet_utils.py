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
from vtk.util.numpy_support import numpy_to_vtk

from ..models.PI_DeepONet.model import PI_SWE_DeepONetModel
from ..models.PI_DeepONet.trainer import PI_SWE_DeepONetTrainer
from ..models.PI_DeepONet.data import PI_SWE_DeepONetDataset
from .config import Config


def pi_deeponet_train(config):
    """
    Train DeepONet with optional physics-informed constraints.

    Args:
        config: Configuration object (Config instance)
    
    Returns:
        tuple: (model, history) - The trained model and training history dictionary
    """    
    
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
        model = PI_SWE_DeepONetModel(config).to(device)
        print_gpu_memory("GPU memory after model initialization")

        # Check whether the model and data are compatible
        model.check_model_input_output_dimensions(train_dataset.branch_dim, train_dataset.trunk_dim, train_dataset.output_dim)
        
        # Initialize trainer
        trainer = PI_SWE_DeepONetTrainer(
            model,
            config
        )
        
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
    # plt.yscale('log')

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

