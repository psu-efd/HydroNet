"""
Example usage of the HydroNet framework.

This script demonstrates how to use the PI-DeepONet component of HydroNet for learning 
the operator of shallow water equations with physics-informed constraints.
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

# Get the project root directory (assumes this script is in examples/DeepONet directory)
script_path = os.path.abspath(__file__)
examples_dir = os.path.dirname(os.path.dirname(os.path.dirname(script_path)))
project_root = os.path.dirname(examples_dir)

# Add the project root to the Python path if it's not already there
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"Added {project_root} to Python path")

from HydroNet import PI_SWE_DeepONetModel, PI_SWE_DeepONetTrainer, PI_SWE_DeepONetDataset, Config


def pi_deeponet_train_with_full_data(config):
    """
    Train PI-DeepONet.

    Args:
        config: Configuration dictionary
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
            deeponet_data_dir=config.get('data.deeponet.train_data_path'),
            config=config
        )
        
        print("Loading validation dataset...")
        val_dataset = PI_SWE_DeepONetDataset(
            deeponet_data_dir=config.get('data.deeponet.val_data_path'),
            config=config
        )
        
        # Monitor memory after dataset loading
        print_gpu_memory("GPU memory after dataset loading")
        
        # Create dataloaders with multiple workers for faster data loading
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.get('training.batch_size'),
            shuffle=True,
            num_workers=1,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.get('training.batch_size'),
            shuffle=False,
            num_workers=1,
            pin_memory=True
        )
        
        # Initialize model
        print("Initializing model...")
        model = PI_SWE_DeepONetModel(config).to(device)
        print_gpu_memory("GPU memory after model initialization")
        
        # Initialize trainer
        trainer = PI_SWE_DeepONetTrainer(
            model,
            config
        )
        
        # Monitor memory after first batch
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

        #save the history to a json file with a time stamp
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

def pi_deeponet_test(best_model_path, config):
    """
    Using the trained DeepONet model for testing on the test dataset.

    Note: The test dataset is not shuffled, so the first N (e.g., 100) samples are from the first case, the next N samples are from the second case, etc.
    
    Args:
        best_model_path: Path to the best model checkpoint
        config: Configuration object
    
    Returns:
        None
    """

    # Check if config is an instance of Config
    if not isinstance(config, Config):
        raise ValueError("config must be an instance of Config")

    print("\n=== DeepONet Test Example ===")
    
    # Configuration and paths
    checkpoint_path = best_model_path 

    # Get the number of cells in the SRH-2D cases from the config file
    nCells = config.get('data.nCells')
    if nCells is None:
        print("Warning: data.nCells not found in config, exiting...")
        exit()
    
    # First, peek at the test dataset to get input dimensions
    print("Determining input dimensions from test data...")
    try:
        test_dataset = SWE_DeepONetDataset(
            data_dir=config.get('data.test_data_path'),
            normalize=not config.get('data.bNormalized'),   #if bNormalized is true, the data is already normalized, thus no need to normalize it again
            transform=None)            

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
    print("Creating the trained model...")
    model = SWE_DeepONetModel(config)
    model.check_model_input_output_dimensions(branch_dim, trunk_dim, output_dim)
    
    # Create trainer
    trainer = SWE_DeepONetTrainer(model, config)
    
    # Load the trained model checkpoint
    try:
        trainer.load_checkpoint(checkpoint_path)
        print(f"Successfully loaded checkpoint from {checkpoint_path}")
    except (FileNotFoundError, RuntimeError) as e:
        print(f"Error loading checkpoint: {e}, exiting...")        
        exit()
    
    # Load test dataset
    print("Loading test dataset...")
    test_loader = DataLoader(
            test_dataset,
            batch_size=config.get('training.batch_size'),
            shuffle=False, # Do not shuffle the test dataset because the order of the samples is important (the first nCells samples are from the first case, the next nCells samples are from the second case, etc.)
            num_workers=1,
            pin_memory=True
        )

    # read the mean and std of the dataset from file data_mean_std_path
    dataset_mean_std = h5py.File(config.get('data.train_val_test_stats_path'), 'r')
    output_mean = dataset_mean_std['train_val_test_mean_outputs'][()]
    output_std = dataset_mean_std['train_val_test_std_outputs'][()]
    dataset_mean_std.close()
    
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

    #print("len(all_predictions): ", len(all_predictions))
    #print("all_predictions[0].shape: ", all_predictions[0].shape)

    #print("len(all_targets): ", len(all_targets))
    #print("all_targets[0].shape: ", all_targets[0].shape)
    
    # Calculate average test loss
    avg_test_loss = total_loss / len(test_loader)
    print(f"Test Loss: {avg_test_loss:.6f}")
    
    # Concatenate all predictions and targets
    all_predictions = np.vstack(all_predictions)
    all_targets = np.vstack(all_targets)
    
    # Calculate additional metrics (e.g., MSE for each output dimension)
    mse_per_dim = np.mean((all_predictions - all_targets) ** 2, axis=0)
    print(f"MSE per output dimension: {mse_per_dim}")

    #print("after vstack")
    #print("all_predictions.shape: ", all_predictions.shape)
    #print("all_targets.shape: ", all_targets.shape)

    #print("all_predictions[0].shape: ", all_predictions[0].shape)
    #print("all_targets[0].shape: ", all_targets[0].shape)
    
    # Denormalize the predictions and targets
    # Assuming the outputs are normalized as (x - mean) / std
    denorm_predictions = all_predictions * output_std + output_mean
    denorm_targets = all_targets * output_std + output_mean
    
    # Calculate metrics on denormalized values
    denorm_mse_per_dim = np.mean((denorm_predictions - denorm_targets) ** 2, axis=0)
    print(f"Denormalized MSE per output dimension: {denorm_mse_per_dim}")

    #print("denorm_predictions[0].shape: ", denorm_predictions[0].shape)
    #print("denorm_targets[0].shape: ", denorm_targets[0].shape)

    #save the denormalized predictions and targets to a npz file
    np.savez(f'data/test/test_results.npz', denorm_predictions=denorm_predictions, denorm_targets=denorm_targets)

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
    with open(f'data/test/test_results_summary.json', 'w') as f:
        json.dump(results_summary, f, indent=4)

    # convert the test results to vtk files
    convert_test_results_to_vtk(config)
    
    print("Finished testing the model.")


def convert_test_results_to_vtk(config):
    """Save the test results to vtk files.

    Args:
        config: Configuration object

    Returns:
        None
    """
    
    # Check if config is an instance of Config
    if not isinstance(config, Config):
        raise ValueError("config must be an instance of Config")

    # Get the number of cells in the SRH-2D cases from the config file
    nCells = config.get('data.nCells')
    if nCells is None:
        print("Warning: data.nCells not found in config, exiting...")
        exit()

    # Load predictions and targets from file ./data/test/test_results.npz
    test_results = np.load('./data/test/test_results.npz')
    denorm_predictions = test_results["denorm_predictions"]
    denorm_targets = test_results["denorm_targets"]

    print("denorm_predictions.shape: ", denorm_predictions.shape)
    print("denorm_targets.shape: ", denorm_targets.shape)

    # Load the split indices (1-based) from file ./data/split_indices.json
    split_indices = json.load(open('./data/split_indices.json')) 
    test_indices = split_indices["test_indices"]

    nCases_in_test_dataset = len(test_indices)
    num_samples_to_plot = min(4, nCases_in_test_dataset)

    #randomly select some samples from the test indices (1-based)
    #random_indices = np.random.choice(test_indices, size=num_samples_to_plot, replace=False)
    random_indices = [522, 738, 741, 661]

    print("selected random indices for plotting: ", random_indices)

    # Find the corresponding indices for each random index
    corresponding_indices = []
    for rand_idx in random_indices:
        # Find the position of this index in the test_indices array
        for i, test_idx in enumerate(test_indices):
            if test_idx == rand_idx:
                corresponding_indices.append(i)
                break
    
    print("corresponding indices: ", corresponding_indices)

    for i in range(num_samples_to_plot):
        #get the corresponding parameter value (Q)
        #Q = successful_samples[random_indices[i]-1, 0]

        #get the current case's results: h, u, v
        h_pred = denorm_predictions[corresponding_indices[i]*nCells:(corresponding_indices[i]+1)*nCells,0]
        u_pred = denorm_predictions[corresponding_indices[i]*nCells:(corresponding_indices[i]+1)*nCells,1]
        v_pred = denorm_predictions[corresponding_indices[i]*nCells:(corresponding_indices[i]+1)*nCells,2]

        h_target = denorm_targets[corresponding_indices[i]*nCells:(corresponding_indices[i]+1)*nCells,0]
        u_target = denorm_targets[corresponding_indices[i]*nCells:(corresponding_indices[i]+1)*nCells,1]
        v_target = denorm_targets[corresponding_indices[i]*nCells:(corresponding_indices[i]+1)*nCells,2]

        #read the empty vtk file which only has the mesh information, then add the results to the vtk file and save it
        # Read the VTK file
        reader = vtk.vtkUnstructuredGridReader()
        reader.SetFileName("data/case_mesh.vtk")
        reader.Update()
        mesh = reader.GetOutput()

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

        # Add the arrays to the mesh as cell data
        mesh.GetCellData().AddArray(h_pred_array)
        mesh.GetCellData().AddArray(u_pred_array)
        mesh.GetCellData().AddArray(v_pred_array)
        mesh.GetCellData().AddArray(h_target_array)
        mesh.GetCellData().AddArray(u_target_array)
        mesh.GetCellData().AddArray(v_target_array)
        mesh.GetCellData().AddArray(velocity_pred_array)
        mesh.GetCellData().AddArray(velocity_target_array)

        # Create a writer
        writer = vtk.vtkUnstructuredGridWriter()
        writer.SetFileName(f"data/test/case_{random_indices[i]}_test_results.vtk")
        writer.SetInputData(mesh)
        writer.Write()

        print(f"Saved results for case {random_indices[i]} to data/test/case_{random_indices[i]}_test_results.vtk")

def plot_training_history(history_file_name):
    """Plot the training history from the history json file."""

    # Load the history json file
    with open(history_file_name, 'r') as f:
        history = json.load(f)

    train_loss = history['train_loss']
    val_loss = history['val_loss']

    #length of the training history
    num_epochs = len(train_loss)

    epochs = np.arange(1, num_epochs + 1)

    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')

    #add axis labels
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)

    #make y axis in log scale
    #plt.yscale('log')

    #axis ticks format
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    #add legend
    plt.legend(fontsize=14)
    plt.tight_layout()

    #save the plot
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')

    #plt.show()
    

if __name__ == "__main__":
    # Start the main timer
    main_start_time = time.time()

    # Load configuration
    config_file = './pi_deeponet_config.yaml'
    config = Config(config_file)
    
    # Set the multiprocessing start method
    mp.set_start_method('spawn', force=True)
    
    # Train the model
    pi_deeponet_train_with_full_data(config)

    # Plot training history
    #plot_training_history('./history_20250314_154327.json')

    # Test the model with the best model and save the test results to vtk files
    #deeponet_test('./checkpoints/deeponet_epoch_96.pt', config)
    
    # Calculate and print the total execution time
    main_execution_time = time.time() - main_start_time
    print(f"\n⏱️ Total execution time: {main_execution_time:.2f} seconds")
    
    print("\nExample completed successfully!") 