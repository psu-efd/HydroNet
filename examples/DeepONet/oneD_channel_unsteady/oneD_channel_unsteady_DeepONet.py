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
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

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

from HydroNet import DeepONetModel, DeepONetTrainer, HydraulicDataset, Config, get_hydraulic_dataloader

def check_data_normalization(train_dataset, val_dataset, test_dataset=None):
    """
    Check normalization consistency between datasets.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Optional test dataset
    """
    print("\n=== Checking Data Normalization ===")
    
    def get_dataset_stats(dataset, name):
        branch_inputs = []
        trunk_inputs = []
        outputs = []
        
        for i in range(len(dataset)):
            branch, trunk, target = dataset[i]
            branch_inputs.append(branch)
            trunk_inputs.append(trunk)
            outputs.append(target)
        
        branch_inputs = torch.stack(branch_inputs)
        trunk_inputs = torch.stack(trunk_inputs)
        outputs = torch.stack(outputs)
        
        print(f"\n{name} Dataset Statistics:")
        print(f"Branch inputs - Mean: {branch_inputs.mean():.6f}, Std: {branch_inputs.std():.6f}")
        print(f"Trunk inputs - Mean: {trunk_inputs.mean():.6f}, Std: {trunk_inputs.std():.6f}")
        print(f"Outputs - Mean: {outputs.mean():.6f}, Std: {outputs.std():.6f}")
        
        return {
            'branch': {'mean': branch_inputs.mean(), 'std': branch_inputs.std()},
            'trunk': {'mean': trunk_inputs.mean(), 'std': trunk_inputs.std()},
            'output': {'mean': outputs.mean(), 'std': outputs.std()}
        }
    
    train_stats = get_dataset_stats(train_dataset, "Training")
    val_stats = get_dataset_stats(val_dataset, "Validation")
    
    # Compare statistics between training and validation
    print("\n=== Normalization Comparison ===")
    for key in ['branch', 'trunk', 'output']:
        print(f"\n{key.capitalize()} Statistics:")
        print(f"Mean difference (train vs val): {abs(train_stats[key]['mean'] - val_stats[key]['mean']):.6f}")
        print(f"Std difference (train vs val): {abs(train_stats[key]['std'] - val_stats[key]['std']):.6f}")
    
    # If test dataset is provided, include it in the comparison
    if test_dataset is not None:
        test_stats = get_dataset_stats(test_dataset, "Test")
        for key in ['branch', 'trunk', 'output']:
            print(f"\n{key.capitalize()} Statistics (Test):")
            print(f"Mean difference (train vs test): {abs(train_stats[key]['mean'] - test_stats[key]['mean']):.6f}")
            print(f"Std difference (train vs test): {abs(train_stats[key]['std'] - test_stats[key]['std']):.6f}")

def compare_dataset_distribution():
    """
        Example of comparing dataset distribution.

        It is important to ensure that the distribution of the training, validation, and test datasets are similar.
        This is important for the generalizability of the model.
    """
    try:
        print("\n=== Compare dataset distribution ===")
        
        # Load datasets
        print("Loading datasets...")
        train_dataset = HydraulicDataset('./data/train')
        val_dataset = HydraulicDataset('./data/val')
        test_dataset = HydraulicDataset('./data/test')

        print("train_dataset.normalization: ", train_dataset.normalization)
        print("\n")
        print("val_dataset.normalization: ", val_dataset.normalization)
        print("\n")
        print("test_dataset.normalization: ", test_dataset.normalization)          
        
        # Run normalization check
        print("\nChecking data normalization...")
        bPassed = check_data_normalization(train_dataset, val_dataset, test_dataset)

        if bPassed:
            print("\nData normalization consistency check passed")
        else:
            raise ValueError("Data normalization consistency check failed")
        
    except Exception as e:
        print(f"\n❌ Error during comparing dataset distribution: {str(e)}")
        raise

def deeponet_train(train_dataset, val_dataset):
    """
    Train the DeepONet model with improved loss calculation verification.
    """
    # Check data normalization first
    #check_data_normalization(train_dataset, val_dataset)  # Removed None parameter
    
    # Create a configuration or load from file
    config_file = './deeponet_config.yaml'
    
    # Create model
    model = DeepONetModel(config_file=config_file)        
    
    # Create trainer
    trainer = DeepONetTrainer(model, config_file=config_file)
    print(f"Model device: {next(model.parameters()).device}")
    
    # Train model
    print("\nStarting training...")
    history = trainer.train()

    # Save the history to a JSON file whose name has the current date and time
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f'history_{timestamp}.json', 'w') as f:
        json.dump(history, f, indent=4)

def deeponet_test(best_model_path):
    """
    Using the trained DeepONet model for testing on the test dataset.
    
    Args:
        best_model_path (str): Path to the best model checkpoint
        
    Note: The test dataset is not shuffled, so the first N samples are from the first case,
          the next N samples are from the second case, etc.
    """
    try:
        print("\n=== DeepONet Test Example ===")
        
        # Configuration and paths
        config_file = './deeponet_config.yaml'
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
            
        if not os.path.exists(best_model_path):
            raise FileNotFoundError(f"Model checkpoint not found: {best_model_path}")
            
        test_data_dir = "./data/test"
        if not os.path.exists(test_data_dir):
            raise FileNotFoundError(f"Test data directory not found: {test_data_dir}")
            
        config = Config(config_file)

        # Get the number of cells in the SRH-2D cases from the config file
        nCells = config.get('data.nCells')
        if nCells is None:
            raise ValueError("data.nCells not found in config file")
        
        # First, peek at the test dataset to get input dimensions
        print("\nDetermining input dimensions from test data...")
        test_dataset = HydraulicDataset(test_data_dir, normalize=True)            
        sample_branch, sample_trunk, sample_output = test_dataset[0]
        branch_dim = sample_branch.shape[0]
        trunk_dim = sample_trunk.shape[0]
        output_dim = sample_output.shape[0]
        print(f"Detected dimensions: branch={branch_dim}, trunk={trunk_dim}, output={output_dim}")
        
        # Create model and check dimensions
        print("\nCreating model...")
        model = DeepONetModel(config_file=config_file)
        model.check_model_input_output_dimensions(branch_dim, trunk_dim, output_dim)
        
        # Create trainer
        print("\nInitializing trainer...")
        trainer = DeepONetTrainer(model, config_file=config_file)
        
        # Load the trained model checkpoint
        print(f"\nLoading checkpoint from {best_model_path}...")
        trainer.load_checkpoint(best_model_path)
        print("Checkpoint loaded successfully")
        
        # Load test dataset
        print("\nLoading test dataset...")
        test_loader = get_hydraulic_dataloader(
            test_dataset, 
            batch_size=trainer.batch_size, 
            shuffle=False,    #no shuffling because the data is ordered by case, cell, and time
            num_workers=4
        )

        # Store normalization parameters
        output_mean = test_dataset.output_mean
        output_std = test_dataset.output_std
        
        # Evaluate model
        print("\nRunning predictions on test dataset...")
        model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        sample_count = 0
        
        with torch.no_grad():
            for batch_idx, (branch_input, trunk_input, target) in enumerate(test_loader):
                # Move data to device
                branch_input = branch_input.to(trainer.device)
                trunk_input = trunk_input.to(trainer.device)
                target = target.to(trainer.device)
                
                # Forward pass
                output = model(branch_input, trunk_input)
                
                # Calculate loss
                loss = trainer.loss_fn(output, target)
                total_loss += loss.item()
                
                # Store predictions and targets
                all_predictions.append(output.cpu().numpy())
                all_targets.append(target.cpu().numpy())
                
                # Count samples
                sample_count += len(branch_input)
                
                # Print progress
                if (batch_idx + 1) % 10 == 0:
                    print(f"Processed {batch_idx + 1} batches...")
        
        # Calculate metrics
        avg_test_loss = total_loss / len(test_loader)
        print(f"\nTest Loss: {avg_test_loss:.6f}")
        
        # Concatenate results
        all_predictions = np.vstack(all_predictions)
        all_targets = np.vstack(all_targets)
        
        # Calculate MSE per dimension
        mse_per_dim = np.mean((all_predictions - all_targets) ** 2, axis=0)
        print(f"MSE per output dimension: {mse_per_dim}")
        
        # Denormalize results
        denorm_predictions = all_predictions * output_std + output_mean
        denorm_targets = all_targets * output_std + output_mean
        
        # Calculate denormalized metrics
        denorm_mse_per_dim = np.mean((denorm_predictions - denorm_targets) ** 2, axis=0)
        print(f"Denormalized MSE per output dimension: {denorm_mse_per_dim}")

        # Save results
        results_file = 'data/test_results.npz'
        np.savez(results_file, 
                denorm_predictions=denorm_predictions, 
                denorm_targets=denorm_targets)
        print(f"\nResults saved to {results_file}")

        # Save summary
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
        
        summary_file = 'data/test_results_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(results_summary, f, indent=4)
        print(f"Summary saved to {summary_file}")
        
        print("\nTesting completed successfully.")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {str(e)}")
        raise

def plot_example_profiles_test_results(variable_name, iTime_to_plot, case_list_to_plot_profiles):
    """Plot the example profiles from the test results json file.
    
    Args:
        variable_name (str): The variable to plot.
        iTime_to_plot (int): The time step to plot.
    """

    print(f"\n=== Plotting Example Profiles of Test Results for {variable_name} at time step {iTime_to_plot} ===")
    
    # Configuration and paths
    config_file = './deeponet_config.yaml'    
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
        
    config = Config(config_file)
    nCells = config.get('data.nCells')
    if nCells is None:
        raise ValueError("data.nCells not found in config file")

    nSaveTimes = config.get('data.nSaveTimes')
    if nSaveTimes is None:
        raise ValueError("data.nSaveTimes not found in config file")

    # Load data files
    print("\nLoading data files...")
    if not os.path.exists('./data/postprocessed_sampling_results.npz'):
        raise FileNotFoundError("Postprocessed sampling results not found")
        
    postprocessed_sampling_results = np.load('./data/postprocessed_sampling_results.npz') 
    cell_centers = postprocessed_sampling_results["cell_centers"]

    if not os.path.exists('./data/test_results.npz'):
        raise FileNotFoundError("Test results not found")
        
    test_results = np.load('./data/test_results.npz')
    denorm_predictions = test_results["denorm_predictions"]
    denorm_targets = test_results["denorm_targets"]

    if not os.path.exists('./data/split_indices.json'):
        raise FileNotFoundError("Split indices file not found")
        
    split_indices = json.load(open('./data/split_indices.json'))  #1-based index
    test_indices = split_indices["test_indices"]   #1-based index

    # Select samples to plot
    nCases_in_test_dataset = len(test_indices)
        
    #random_indices = np.random.choice(test_indices, num_samples_to_plot, replace=False)
    random_indices = sorted(case_list_to_plot_profiles)  # Fixed indices for reproducibility, sorted in ascending order, 1-based index

    num_samples_to_plot = len(random_indices)

    print(f"\nSelected samples for plotting: {random_indices}")

    # Find corresponding indices
    corresponding_indices = []
    for rand_idx in random_indices:
        for i, test_idx in enumerate(test_indices):
            if test_idx == rand_idx:
                corresponding_indices.append(i)   #0-based index
                break
    print(f"Corresponding indices: {corresponding_indices}")

    # collect the (u,v,h) profiles for each case and time step
    u_profiles_pred = []
    v_profiles_pred = []
    h_profiles_pred = []

    u_profiles_true = []
    v_profiles_true = []
    h_profiles_true = []

    for i in range(num_samples_to_plot):
        u_profile_pred = np.zeros((nSaveTimes, nCells))
        v_profile_pred = np.zeros((nSaveTimes, nCells))
        h_profile_pred = np.zeros((nSaveTimes, nCells))

        u_profile_true = np.zeros((nSaveTimes, nCells))
        v_profile_true = np.zeros((nSaveTimes, nCells))
        h_profile_true = np.zeros((nSaveTimes, nCells))

        for j in range(nCells):
            for k in range(nSaveTimes):
                u_profile_pred[k, j] = denorm_predictions[corresponding_indices[i]*nCells*nSaveTimes + j*nSaveTimes + k, 0]
                v_profile_pred[k, j] = denorm_predictions[corresponding_indices[i]*nCells*nSaveTimes + j*nSaveTimes + k, 1]
                h_profile_pred[k, j] = denorm_predictions[corresponding_indices[i]*nCells*nSaveTimes + j*nSaveTimes + k, 2]

                u_profile_true[k, j] = denorm_targets[corresponding_indices[i]*nCells*nSaveTimes + j*nSaveTimes + k, 0]
                v_profile_true[k, j] = denorm_targets[corresponding_indices[i]*nCells*nSaveTimes + j*nSaveTimes + k, 1]
                h_profile_true[k, j] = denorm_targets[corresponding_indices[i]*nCells*nSaveTimes + j*nSaveTimes + k, 2]

        u_profiles_pred.append(u_profile_pred)
        v_profiles_pred.append(v_profile_pred)
        h_profiles_pred.append(h_profile_pred)

        u_profiles_true.append(u_profile_true)
        v_profiles_true.append(v_profile_true)
        h_profiles_true.append(h_profile_true)

    # Create plot
    print("\nGenerating plot...")
    plt.figure(figsize=(8, 6))
    colors = ['black', 'red', 'blue', 'green']
    
    #loop over selected cases
    for i in range(num_samples_to_plot):
        # Plot water surface elevation
        if variable_name == "wse":
            plt.plot(cell_centers[:, 0], 
                    h_profiles_pred[i][iTime_to_plot, :] + cell_centers[:, 2], 
                    color=colors[i], 
                    label=f'Case {random_indices[i]}',
                    linewidth=2)
            plt.plot(cell_centers[:, 0], 
                    h_profiles_true[i][iTime_to_plot, :] + cell_centers[:, 2], 
                    color=colors[i], 
                    linestyle='--',
                    linewidth=2)
            
            # Plot bed elevation
            if i == num_samples_to_plot - 1:
                plt.plot(cell_centers[:, 0], cell_centers[:, 2], color='k', label='Bed Elevation', linewidth=1.5)
        elif variable_name == "h":
            plt.plot(cell_centers[:, 0], 
                    h_profiles_pred[i][iTime_to_plot, :], 
                    color=colors[i], 
                    label=f'Case {random_indices[i]}',
                    linewidth=2)
            plt.plot(cell_centers[:, 0], 
                    h_profiles_true[i][iTime_to_plot, :], 
                    color=colors[i], 
                    linestyle='--',
                    linewidth=2)
        elif variable_name == "u":
            plt.plot(cell_centers[:, 0], 
                    u_profiles_pred[i][iTime_to_plot, :], 
                    color=colors[i], 
                    label=f'Case {random_indices[i]}',
                    linewidth=2)
            plt.plot(cell_centers[:, 0], 
                    u_profiles_true[i][iTime_to_plot, :], 
                    color=colors[i], 
                    linestyle='--',
                    linewidth=2)
       
    
    # Customize plot    
    plt.xlabel('x (m)', fontsize=14)
    if variable_name == "wse":
        plt.ylabel('WSE (m)', fontsize=14)
    elif variable_name == "h":
        plt.ylabel('Water Depth (m)', fontsize=14)
    elif variable_name == "u":
        plt.ylabel('Velocity (m/s)', fontsize=14)

    #tick style and fontsize for both axis
    plt.tick_params(axis='both', which='major', labelsize=12)

    #axis limits
    plt.xlim(cell_centers[0, 0], cell_centers[-1, 0])

    if variable_name == "wse" or variable_name == "h":
        plt.ylim(-0.1, 1.0)
    elif variable_name == "u":
        plt.ylim(0, 0.8)

    plt.grid(True, linestyle='--', alpha=0.7)

    plt.legend(fontsize=12, loc='upper right')

    plt.title(f'{variable_name} at saved time step {iTime_to_plot}')

    plt.tight_layout()
    
    # Save plot
    # Create directory if it doesn't exist
    output_dir = './plots'
    os.makedirs(output_dir, exist_ok=True)

    # Save plot
    output_file = f'./plots/{variable_name}_profiles_{iTime_to_plot:04d}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {output_file}")
    plt.close()
    
    print("\nPlotting completed successfully.")


def plot_example_time_history_test_results(variable_name, iCell_to_plot, case_list_to_plot):
    """Plot the example profiles from the test results json file.
    
    Args:
        variable_name (str): The variable to plot.
        iCell_to_plot (int): The cell index to plot. 1-based index
        case_list_to_plot (list): The list of case indices to plot. 1-based index
    """

    print(f"\n=== Plotting Example Time History of Test Results for {variable_name} ===")
    
    # Configuration and paths
    config_file = './deeponet_config.yaml'    
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
        
    config = Config(config_file)
    nCells = config.get('data.nCells')
    if nCells is None:
        raise ValueError("data.nCells not found in config file")
    
    if iCell_to_plot > nCells:
        raise ValueError(f"iCell_to_plot {iCell_to_plot} is greater than nCells {nCells}")

    start_time = config.get('data.start_time')
    if start_time is None:
        raise ValueError("data.start_time not found in config file")

    end_time = config.get('data.end_time')
    if end_time is None:
        raise ValueError("data.end_time not found in config file")

    nSaveTimes = config.get('data.nSaveTimes')
    if nSaveTimes is None:
        raise ValueError("data.nSaveTimes not found in config file")
    
    save_delta_time = (end_time - start_time) / nSaveTimes

    time_array = [start_time + i*save_delta_time for i in range(1, nSaveTimes+1)]

    # Load data files
    print("\nLoading data files...")
    if not os.path.exists('./data/postprocessed_sampling_results.npz'):
        raise FileNotFoundError("Postprocessed sampling results not found")
        
    postprocessed_sampling_results = np.load('./data/postprocessed_sampling_results.npz') 
    cell_centers = postprocessed_sampling_results["cell_centers"]

    if not os.path.exists('./data/test_results.npz'):
        raise FileNotFoundError("Test results not found")
        
    test_results = np.load('./data/test_results.npz')
    denorm_predictions = test_results["denorm_predictions"]
    denorm_targets = test_results["denorm_targets"]

    if not os.path.exists('./data/split_indices.json'):
        raise FileNotFoundError("Split indices file not found")
        
    split_indices = json.load(open('./data/split_indices.json'))  #1-based index
    test_indices = split_indices["test_indices"]   #1-based index

    # Select samples to plot
    nCases_in_test_dataset = len(test_indices)
        
    #random_indices = np.random.choice(test_indices, num_samples_to_plot, replace=False)
    random_indices = sorted(case_list_to_plot)  # Fixed indices for reproducibility, sorted in ascending order, 1-based index

    num_cases_to_plot = len(random_indices)

    #generate different colors for each case
    colors = ['black', 'red', 'blue', 'green']

    print(f"\nSelected samples for plotting: {random_indices}")

    # Find corresponding indices
    corresponding_indices = []
    for rand_idx in random_indices:
        for i, test_idx in enumerate(test_indices):
            if test_idx == rand_idx:
                corresponding_indices.append(i)   #0-based index
                break
    print(f"Corresponding indices: {corresponding_indices}")

    # collect the (u,v,h) histories for each case and time step
    u_histories_pred = []
    v_histories_pred = []
    h_histories_pred = []

    u_histories_true = []
    v_histories_true = []
    h_histories_true = []

    for i in range(num_cases_to_plot):
        u_history_pred = np.zeros((nSaveTimes))
        v_history_pred = np.zeros((nSaveTimes))
        h_history_pred = np.zeros((nSaveTimes))

        u_history_true = np.zeros((nSaveTimes))
        v_history_true = np.zeros((nSaveTimes))
        h_history_true = np.zeros((nSaveTimes))

        for k in range(nSaveTimes):
            u_history_pred[k] = denorm_predictions[corresponding_indices[i]*nCells*nSaveTimes + (iCell_to_plot-1)*nSaveTimes + k, 0]    #iCell_to_plot is 1-based index
            v_history_pred[k] = denorm_predictions[corresponding_indices[i]*nCells*nSaveTimes + (iCell_to_plot-1)*nSaveTimes + k, 1]
            h_history_pred[k] = denorm_predictions[corresponding_indices[i]*nCells*nSaveTimes + (iCell_to_plot-1)*nSaveTimes + k, 2]

            u_history_true[k] = denorm_targets[corresponding_indices[i]*nCells*nSaveTimes + (iCell_to_plot-1)*nSaveTimes + k, 0]    #iCell_to_plot is 1-based index
            v_history_true[k] = denorm_targets[corresponding_indices[i]*nCells*nSaveTimes + (iCell_to_plot-1)*nSaveTimes + k, 1]
            h_history_true[k] = denorm_targets[corresponding_indices[i]*nCells*nSaveTimes + (iCell_to_plot-1)*nSaveTimes + k, 2]

        u_histories_pred.append(u_history_pred)
        v_histories_pred.append(v_history_pred)
        h_histories_pred.append(h_history_pred)

        u_histories_true.append(u_history_true)
        v_histories_true.append(v_history_true)
        h_histories_true.append(h_history_true)

    # Create plot
    print("\nGenerating plot...")

    # Create a figure (all cases in one plot)
    plt.figure(figsize=(8, 6))
    
    #loop over selected cases 
    for i in range(num_cases_to_plot):        

        # Plot water surface elevation time history
        if variable_name == "wse":
            plt.plot(time_array, 
                    h_histories_pred[i] + cell_centers[iCell_to_plot-1, 2], 
                    color=colors[i], 
                    label=f'Case {random_indices[i]}',
                    linewidth=2)
            plt.plot(time_array, 
                    h_histories_true[i] + cell_centers[iCell_to_plot-1, 2], 
                    color=colors[i], 
                    #label=f'Truth',
                    linestyle='--',
                    linewidth=2)           
          
        elif variable_name == "h":
            plt.plot(time_array, 
                    h_histories_pred[i], 
                    color=colors[i], 
                    label=f'Case {random_indices[i]}',
                    linewidth=2)
            plt.plot(time_array, 
                    h_histories_true[i], 
                    color=colors[i], 
                    #label=f'Truth',
                    linestyle='--',
                    linewidth=2)
        elif variable_name == "u":
            plt.plot(time_array, 
                    u_histories_pred[i], 
                    color=colors[i], 
                    label=f'Case {random_indices[i]}',
                    linewidth=2)
            plt.plot(time_array, 
                    u_histories_true[i], 
                    color=colors[i], 
                    #label=f'Truth',
                    linestyle='--',
                    linewidth=2)
       
    
    # Customize plot    
    plt.xlabel('Time (hrs)', fontsize=14)
    if variable_name == "wse":
        plt.ylabel('WSE (m)', fontsize=14)
    elif variable_name == "h":
        plt.ylabel('Water Depth (m)', fontsize=14)
    elif variable_name == "u":
        plt.ylabel('Velocity (m/s)', fontsize=14)

    #tick style and fontsize for both axis
    plt.tick_params(axis='both', which='major', labelsize=12)

    #axis limits
    plt.xlim(start_time, end_time)

    if variable_name == "wse" or variable_name == "h":
        plt.ylim(-0.1, 1.0)
    elif variable_name == "u":
        plt.ylim(0, 0.6)

    plt.grid(True, linestyle='--', alpha=0.7)

    plt.legend(fontsize=12, loc='upper right')

    #add text below the legend: dashed lines are truth
    plt.text(0.05, 0.95, 'Solid lines are ML predictions \nDashed lines are truth', ha='left', va='top', fontsize=12,  transform=plt.gca().transAxes)

    if variable_name == "wse":
        plt.title(f'Water Surface Elevation at cell {iCell_to_plot}')
    elif variable_name == "h":
        plt.title(f'Water Depth at cell {iCell_to_plot}')
    elif variable_name == "u":
        plt.title(f'x-Velocity at cell {iCell_to_plot}')

    plt.tight_layout()
        
    # Save plot
    # Create directory if it doesn't exist
    output_dir = './plots/history'
    os.makedirs(output_dir, exist_ok=True)

    # Save plot
    output_file = f'./plots/history/{variable_name}_history_cellID_{iCell_to_plot:04d}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {output_file}")
    plt.close()
    
    print("\nPlotting completed successfully.")

def plot_training_history(history_file_name):
    """Plot the training history from the history json file."""
    try:
        print("\n=== Plotting Training History ===")
        
        if not os.path.exists(history_file_name):
            raise FileNotFoundError(f"History file not found: {history_file_name}")

        # Load the history json file
        print("\nLoading training history...")
        with open(history_file_name, 'r') as f:
            history = json.load(f)

        train_loss = history['train_loss']
        val_loss = history['val_loss']
        num_epochs = len(train_loss)
        epochs = np.arange(1, num_epochs + 1)

        # Create plot
        print("\nGenerating plot...")
        plt.figure(figsize=(12, 6))
        
        # Plot losses
        plt.plot(epochs, train_loss, label='Training Loss', linewidth=2)
        plt.plot(epochs, val_loss, label='Validation Loss', linewidth=2)

        # Customize plot
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()

        # Save plot
        output_file = 'training_history.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to {output_file}")
        plt.close()
        
        print("\nPlotting completed successfully.")
        
    except Exception as e:
        print(f"\n❌ Error during plotting: {str(e)}")
        raise

def compute_dataset_stats(train_dataset, val_dataset, test_dataset):
    """
    Check normalization consistency between training, validation, and test datasets
    using pre-computed normalization parameters.
    
    Args:
        train_dataset: HydraulicDataset for training data
        val_dataset: HydraulicDataset for validation data
        test_dataset: HydraulicDataset for test data
    
    Returns:
        dict: Dictionary containing normalization parameters for all datasets
    """
    print("\n=== Data Normalization Check ===")
    
    def print_dataset_stats(dataset, name):
        """Print normalization statistics for a dataset."""
        print(f"\n{name} Dataset Normalization Parameters:")
        
        # Branch inputs statistics
        print(f"\nBranch Inputs:")
        print(f"  Mean shape: {dataset.branch_mean.shape}")
        print(f"  Std shape: {dataset.branch_std.shape}")
        for i in range(dataset.branch_mean.shape[1]):
            print(f"  Feature {i}:")
            print(f"    Mean: {dataset.branch_mean[0, i]:.6f}")
            print(f"    Std: {dataset.branch_std[0, i]:.6f}")
        
        # Trunk inputs statistics
        print(f"\nTrunk Inputs:")
        print(f"  Mean shape: {dataset.trunk_mean.shape}")
        print(f"  Std shape: {dataset.trunk_std.shape}")
        for i in range(dataset.trunk_mean.shape[1]):
            print(f"  Coordinate {i}:")
            print(f"    Mean: {dataset.trunk_mean[0, i]:.6f}")
            print(f"    Std: {dataset.trunk_std[0, i]:.6f}")
        
        # Outputs statistics
        print(f"\nOutputs:")
        print(f"  Mean shape: {dataset.output_mean.shape}")
        print(f"  Std shape: {dataset.output_std.shape}")
        output_names = ['X-Velocity (u)', 'Y-Velocity (v)', 'Water Depth (h)']
        for i in range(dataset.output_mean.shape[1]):
            print(f"  {output_names[i]}:")
            print(f"    Mean: {dataset.output_mean[0, i]:.6f}")
            print(f"    Std: {dataset.output_std[0, i]:.6f}")
        
        return {
            'branch': {
                'mean': dataset.branch_mean,
                'std': dataset.branch_std
            },
            'trunk': {
                'mean': dataset.trunk_mean,
                'std': dataset.trunk_std
            },
            'output': {
                'mean': dataset.output_mean,
                'std': dataset.output_std
            }
        }
    
    # Print and collect statistics for each dataset
    train_stats = print_dataset_stats(train_dataset, "Training")
    val_stats = print_dataset_stats(val_dataset, "Validation")
    test_stats = print_dataset_stats(test_dataset, "Test")
    
    # Compare statistics between datasets
    print("\n=== Normalization Comparison ===")
    
    def compare_stats(dataset1_stats, dataset2_stats, dataset1_name, dataset2_name, name, dim_names=None):
        if dim_names is None:
            dim_names = [f"Dimension {i}" for i in range(dataset1_stats['mean'].shape[1])]
            
        for i, (mean1, mean2, std1, std2) in enumerate(zip(
            dataset1_stats['mean'][0], dataset2_stats['mean'][0],
            dataset1_stats['std'][0], dataset2_stats['std'][0]
        )):
            mean_diff = abs(mean1 - mean2)
            std_diff = abs(std1 - std2)
            print(f"\n{name} - {dim_names[i]}:")
            print(f"  Mean difference between {dataset1_name} and {dataset2_name}: {mean_diff:.6f}")
            print(f"  Std difference between {dataset1_name} and {dataset2_name}: {std_diff:.6f}")
            if mean_diff > 0.1 or std_diff > 0.1:
                print(f"  ⚠️ WARNING: Large difference detected in {name} - {dim_names[i]} statistics!")
    
    # Compare branch inputs (25 features)
    branch_dim_names = [f"Feature {i}" for i in range(25)]
    compare_stats(train_stats['branch'], val_stats['branch'], "Training", "Validation", "Branch Inputs", branch_dim_names)
    compare_stats(train_stats['branch'], test_stats['branch'], "Training", "Test", "Branch Inputs", branch_dim_names)
    
    # Compare trunk inputs (3 spatial coordinates)
    trunk_dim_names = [f"Coordinate {i}" for i in range(3)]
    compare_stats(train_stats['trunk'], val_stats['trunk'], "Training", "Validation", "Trunk Inputs", trunk_dim_names)
    compare_stats(train_stats['trunk'], test_stats['trunk'], "Training", "Test", "Trunk Inputs", trunk_dim_names)
    
    # Compare outputs (3 variables)
    output_dim_names = ['Water Depth (h)', 'X-Velocity (u)', 'Y-Velocity (v)']
    compare_stats(train_stats['output'], val_stats['output'], "Training", "Validation", "Outputs", output_dim_names)
    compare_stats(train_stats['output'], test_stats['output'], "Training", "Test", "Outputs", output_dim_names)
    
    return {
        'train_stats': train_stats,
        'val_stats': val_stats,
        'test_stats': test_stats
    }

if __name__ == "__main__":
    # Start the main timer
    main_start_time = time.time()
    print("\n=== Starting DeepONet Example ===")
    
    # Load datasets
    print("\nLoading datasets...")
    train_dataset = HydraulicDataset('./data/train')
    val_dataset = HydraulicDataset('./data/val')
    test_dataset = HydraulicDataset('./data/test')
    
    # Train the model
    #deeponet_train(train_dataset, val_dataset)

    #exit()
    
    # Plot training history
    #plot_training_history()
    
    # Find best model checkpoint
    best_model_path = os.path.join('./checkpoints', 'deeponet_epoch_28.pt')
    if os.path.exists(best_model_path):
        print(f"\nFound best model checkpoint: {best_model_path}")
        # Test the model
        #deeponet_test(best_model_path)

        # Plot test results
        #iTimes steps to plot
        iTimes_to_plot = range(0, 24)
        #case list to plot profiles
        case_list_to_plot = [637, 77, 821, 522]
        for iTime_to_plot in iTimes_to_plot:
            print(f"\nPlotting profiles of test results for time step {iTime_to_plot}")
            plot_example_profiles_test_results("wse", iTime_to_plot, case_list_to_plot)
            plot_example_profiles_test_results("h", iTime_to_plot, case_list_to_plot)
            plot_example_profiles_test_results("u", iTime_to_plot, case_list_to_plot)
            

        #plot time history of solution variables for selected cases
        cell_list_to_plot = [20, 40, 60]
        for iCell_to_plot in cell_list_to_plot:
            print(f"\nPlotting time history of test results for cell {iCell_to_plot}")
            #plot_example_time_history_test_results("wse", iCell_to_plot, case_list_to_plot)
            #plot_example_time_history_test_results("h", iCell_to_plot, case_list_to_plot)
            #plot_example_time_history_test_results("u", iCell_to_plot, case_list_to_plot)
    
    # Calculate and print the total execution time
    main_execution_time = time.time() - main_start_time
    print(f"\n⏱️ Total execution time: {main_execution_time:.2f} seconds")
    
    print("\nExample completed successfully!") 