"""
Example usage of the HydroNet framework.

This script demonstrates how to use the DeepONet component of HydroNet for learning 
the operator of shallow water equations.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
import torch
import time  # Add back time import for the main timer

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

from HydroNet import SWE_DeepONetModel, SWE_DeepONetTrainer, SWE_DeepONetDataset, Config, create_deeponet_dataloader

def check_data_normalization(train_dataset, val_dataset):
    """
    Check normalization consistency between training and validation datasets.
    
    Args:
        train_dataset: SWE_DeepONetDataset for training data
        val_dataset: SWE_DeepONetDataset for validation data
    
    Returns:
        dict: Dictionary containing statistics for both datasets
    """
    print("\n=== Data Normalization Check ===")
    
    # Function to compute statistics for a dataset
    def compute_dataset_stats(dataset, name):
        branch_inputs = []
        trunk_inputs = []
        outputs = []
        
        for branch, trunk, target in dataset:
            branch_inputs.append(branch.numpy())
            trunk_inputs.append(trunk.numpy())
            outputs.append(target.numpy())
        
        branch_inputs = np.concatenate(branch_inputs)
        trunk_inputs = np.concatenate(trunk_inputs)
        outputs = np.concatenate(outputs)
        
        print(f"\n{name} Dataset Statistics:")
        
        # Branch inputs statistics (for each feature)
        print(f"\nBranch Inputs (shape: {branch_inputs.shape}):")
        for i in range(branch_inputs.shape[1]):
            print(f"  Feature {i}:")
            print(f"    Mean: {np.mean(branch_inputs[:, i]):.6f}")
            print(f"    Std: {np.std(branch_inputs[:, i]):.6f}")
            print(f"    Min: {np.min(branch_inputs[:, i]):.6f}")
            print(f"    Max: {np.max(branch_inputs[:, i]):.6f}")
        
        # Trunk inputs statistics (for each spatial/temporal coordinate)
        print(f"\nTrunk Inputs (shape: {trunk_inputs.shape}):")
        for i in range(trunk_inputs.shape[1]):
            print(f"  Coordinate {i}:")
            print(f"    Mean: {np.mean(trunk_inputs[:, i]):.6f}")
            print(f"    Std: {np.std(trunk_inputs[:, i]):.6f}")
            print(f"    Min: {np.min(trunk_inputs[:, i]):.6f}")
            print(f"    Max: {np.max(trunk_inputs[:, i]):.6f}")
        
        # Outputs statistics (for each output variable)
        print(f"\nOutputs (shape: {outputs.shape}):")
        output_names = ['X-Velocity (u)', 'Y-Velocity (v)', 'Water Depth (h)']
        for i in range(outputs.shape[1]):
            print(f"  {output_names[i]}:")
            print(f"    Mean: {np.mean(outputs[:, i]):.6f}")
            print(f"    Std: {np.std(outputs[:, i]):.6f}")
            print(f"    Min: {np.min(outputs[:, i]):.6f}")
            print(f"    Max: {np.max(outputs[:, i]):.6f}")
        
        return {
            'branch': {
                'mean': np.mean(branch_inputs, axis=0),
                'std': np.std(branch_inputs, axis=0),
                'min': np.min(branch_inputs, axis=0),
                'max': np.max(branch_inputs, axis=0)
            },
            'trunk': {
                'mean': np.mean(trunk_inputs, axis=0),
                'std': np.std(trunk_inputs, axis=0),
                'min': np.min(trunk_inputs, axis=0),
                'max': np.max(trunk_inputs, axis=0)
            },
            'output': {
                'mean': np.mean(outputs, axis=0),
                'std': np.std(outputs, axis=0),
                'min': np.min(outputs, axis=0),
                'max': np.max(outputs, axis=0)
            }
        }
    
    # Compute statistics for both datasets
    train_stats = compute_dataset_stats(train_dataset, "Training")
    val_stats = compute_dataset_stats(val_dataset, "Validation")
    
    # Compare statistics between datasets
    print("\n=== Normalization Comparison ===")
    
    def compare_stats(train_stat, val_stat, name, dim_names=None):
        if dim_names is None:
            dim_names = [f"Dimension {i}" for i in range(len(train_stat['mean']))]
            
        for i, (train_mean, val_mean, train_std, val_std) in enumerate(zip(
            train_stat['mean'], val_stat['mean'],
            train_stat['std'], val_stat['std']
        )):
            mean_diff = abs(train_mean - val_mean)
            std_diff = abs(train_std - val_std)
            print(f"\n{name} - {dim_names[i]}:")
            print(f"  Mean difference: {mean_diff:.6f}")
            print(f"  Std difference: {std_diff:.6f}")
            if mean_diff > 0.1 or std_diff > 0.1:
                print(f"  ⚠️ WARNING: Large difference detected in {name} - {dim_names[i]} statistics!")
    
    # Compare branch inputs (25 features)
    branch_dim_names = [f"Feature {i}" for i in range(25)]
    compare_stats(train_stats['branch'], val_stats['branch'], "Branch Inputs", branch_dim_names)
    
    # Compare trunk inputs (3 spatial coordinates)
    trunk_dim_names = [f"Coordinate {i}" for i in range(3)]
    compare_stats(train_stats['trunk'], val_stats['trunk'], "Trunk Inputs", trunk_dim_names)
    
    # Compare outputs (3 variables)
    output_dim_names = ['Water Depth (h)', 'X-Velocity (u)', 'Y-Velocity (v)']
    compare_stats(train_stats['output'], val_stats['output'], "Outputs", output_dim_names)
    
    return {
        'train_stats': train_stats,
        'val_stats': val_stats
    }

def deeponet_train():
    """Example of using DeepONet for learning the operator of shallow water equations."""

    print("\n=== DeepONet: 1D channel steady state ===")
    
    # Create a configuration or load from file
    config_file = './deeponet_config.yaml'
    
    # Create model
    model = SWE_DeepONetModel(config_file=config_file)        
    
    # Create trainer
    trainer = SWE_DeepONetTrainer(model, config_file=config_file)
    print(f"Model device: {next(model.parameters()).device}")
    
    # Train model
    print("\nStarting training...")
    history = trainer.train()

    # Save the history to a JSON file whose name has the current date and time
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f'history_{timestamp}.json', 'w') as f:
        json.dump(history, f, indent=4)
    
    print("DeepONet training completed.")


def deeponet_test(best_model_path):
    """
    Using the trained DeepONet model for testing on the test dataset.

    Note: The test dataset is not shuffled, so the first N (e.g., 100) samples are from the first case, the next N samples are from the second case, etc.
    
    """

    print("\n=== DeepONet Test Example ===")
    
    # Configuration and paths
    config_file = './deeponet_config.yaml'
    checkpoint_path = best_model_path 
    test_data_dir = "./data/test"  # Path to test dataset
    config = Config(config_file)

    # Get the number of cells in the SRH-2D cases from the config file
    nCells = config.get('data.nCells')
    if nCells is None:
        print("Warning: data.nCells not found in config, exiting...")
        exit()
    
    # First, peek at the test dataset to get input dimensions
    print("Determining input dimensions from test data...")
    try:
        test_dataset = SWE_DeepONetDataset(test_data_dir, normalize=True)            

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
    print("Creating model...")
    model = SWE_DeepONetModel(config_file=config_file)
    model.check_model_input_output_dimensions(branch_dim, trunk_dim, output_dim)
    
    # Create trainer
    trainer = SWE_DeepONetTrainer(model, config_file=config_file)
    
    # Load the trained model checkpoint
    try:
        trainer.load_checkpoint(checkpoint_path)
        print(f"Successfully loaded checkpoint from {checkpoint_path}")
    except (FileNotFoundError, RuntimeError) as e:
        print(f"Error loading checkpoint: {e}, exiting...")        
        exit()
    
    # Load test dataset
    print("Loading test dataset...")
    test_loader = get_hydraulic_dataloader(
        test_dataset, 
        batch_size=trainer.batch_size, 
        shuffle=False,           # Do not shuffle the test dataset because the order of the samples is important (the first nCells samples are from the first case, the next nCells samples are from the second case, etc.)
        num_workers=4
    )

    # Store the normalization parameters for later denormalization
    output_mean = test_dataset.output_mean
    output_std = test_dataset.output_std
    
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
    np.savez(f'data/test_results.npz', denorm_predictions=denorm_predictions, denorm_targets=denorm_targets)

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
    with open(f'data/test_results_summary.json', 'w') as f:
        json.dump(results_summary, f, indent=4)
    
    print("Finished testing the model.")


def plot_test_results():
    """Plot the test results from the test results json file."""

    # Configuration and paths
    config_file = './deeponet_config.yaml'    
    config = Config(config_file)

    # Get the number of cells in the SRH-2D cases from the config file
    nCells = config.get('data.nCells')
    if nCells is None:
        print("Warning: data.nCells not found in config, exiting...")
        exit()

    # Load the postprocessed sampling results (truth) from file ./data/postprocessed_sampling_results.npz
    postprocessed_sampling_results = np.load('./data/postprocessed_sampling_results.npz') 
    results_array = postprocessed_sampling_results["results_array"]
    cell_centers = postprocessed_sampling_results["cell_centers"]
    successful_samples = postprocessed_sampling_results["successful_samples"]

    print("successful_samples.shape: ", successful_samples.shape)

    # Load the test results from file ./data/test_results.npz
    test_results = np.load('./data/test_results.npz')
    denorm_predictions = test_results["denorm_predictions"]
    denorm_targets = test_results["denorm_targets"]

    # Load the split indices (1-based) from file ./data/split_indices.json
    split_indices = json.load(open('./data/split_indices.json')) 
    test_indices = split_indices["test_indices"]

    nCases_in_test_dataset = len(test_indices)
    num_samples_to_plot = min(4, nCases_in_test_dataset)

    #randomly select 4 samples from the test indices (1-based)
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

    #create four different colors for the plots
    colors = ['black', 'red', 'blue', 'green']
    
    plt.figure(figsize=(10, 8))

    for i in range(num_samples_to_plot):
        #get the corresponding parameter value (Q)
        Q = successful_samples[random_indices[i]-1, 0]

        print(f"Sample {random_indices[i]}, Q={Q:.3f} m$^3$/s")

        plt.plot(cell_centers[:, 0], denorm_targets[corresponding_indices[i]*nCells:(corresponding_indices[i]+1)*nCells,2] + cell_centers[:, 2], color=colors[i], label=f'Sample {random_indices[i]}, Q={Q:.3f} m$^3$/s')
        plt.plot(cell_centers[:, 0], denorm_predictions[corresponding_indices[i]*nCells:(corresponding_indices[i]+1)*nCells,2] + cell_centers[:, 2], color=colors[i], linestyle='--')
        
    #plot the bed elevation
    plt.plot(cell_centers[:, 0], cell_centers[:, 2], color='k', label='Bed Elevation')
    
    plt.legend()

    #add axis labels
    plt.xlabel('x (m)', fontsize=14)
    plt.ylabel('Elevation (m)', fontsize=14)        
    
    plt.tight_layout()
    plt.savefig('./test_prediction_samples.png')
    plt.show()

    print(f"Saved prediction samples plot to './test_prediction_samples.png'")

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
    
    # Train the model
    #deeponet_train()

    # Plot training history
    #plot_training_history('./history_20250314_154327.json')

    # Test the model
    #deeponet_test(best_model_path='./checkpoints/deeponet_epoch_78.pt')

    # Plot test results
    plot_test_results()

    # Calculate and print the total execution time
    main_execution_time = time.time() - main_start_time
    print(f"\n⏱️ Total execution time: {main_execution_time:.2f} seconds")
    
    print("\nExample completed successfully!") 