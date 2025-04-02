"""
Example usage of the HydroNet framework.

This script demonstrates how to use the PINN component of HydroNet for solving 
shallow water equations with physics-informed constraints.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
import torch
import time
import torch.optim as optim
import yaml

# Set random seeds for reproducibility
np.random.seed(123456)
torch.manual_seed(123456)
if torch.cuda.is_available():
    torch.cuda.manual_seed(123456)

plt.rc('text', usetex=False)
plt.rc('font', family='serif')

# Add project root to Python path
script_path = os.path.abspath(__file__)
examples_dir = os.path.dirname(os.path.dirname(os.path.dirname(script_path)))
project_root = os.path.dirname(examples_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"Added {project_root} to Python path")

from HydroNet import SWE_PINN, PINNTrainer, PINNDataset, Config
from HydroNet.utils.predict_on_gmsh import predict_on_gmsh2d_mesh

def initial_conditions(points):
    """
    Define initial conditions for the shallow water equations. The order of points should be fixed (no random shuffling) and no minibatching. All initial condition points should be included in the loss calculation here.
    
    Args:
        points (torch.Tensor): Points at t=0, shape (N, 2) for (x, y)
        
    Returns:
        torch.Tensor: Initial values for (h, u, v), shape (N, 3)
    """
    device = points.device
    IC_points_size = points.shape[0]
    
    # Get x coordinates
    x = points[:, 0]
    
    # Initial velocities (zero everywhere)
    u_init = torch.zeros(IC_points_size, device=device)  # Initial x-velocity
    v_init = torch.zeros(IC_points_size, device=device)  # Initial y-velocity
    
    # Initial water depth (discontinuous)
    h_init = torch.ones(IC_points_size, device=device)   # Initialize to ones
    
    # Find the middle of x domain
    x_min = x.min()
    x_max = x.max()
    x_middle = (x_min + x_max) / 2
    
    # Set water depth based on x position
    left_half = (x <= x_middle)
    right_half = (x > x_middle)
    
    h_init[left_half] = 1.0   # Left half: h = 1.0 m
    h_init[right_half] = 0.5  # Right half: h = 0.5 m
    
    # Print some information about the initial condition
    #print("\nInitial Condition Summary:")
    #print(f"Left half (x ≤ {x_middle:.2f}): h = 1.0 m")
    #print(f"Right half (x > {x_middle:.2f}): h = 0.5 m")
    #print(f"Number of points in left half: {torch.sum(left_half).item()}")
    #print(f"Number of points in right half: {torch.sum(right_half).item()}")
    
    return torch.stack([h_init, u_init, v_init], dim=1)

def boundary_conditions(boundary_info):
    """
    Define boundary conditions for the shallow water equations.
    
    Args:
        boundary_info (tuple): Tuple of (boundary_points, boundary_ids, boundary_normals)
        
    Returns:
        torch.Tensor: Boundary values for (h, u, v), shape (N, 3)
    """
    device = boundary_info[0].device
    batch_size = boundary_info[0].shape[0]
    
    # Unpack boundary info
    boundary_points, boundary_ids, boundary_normals = boundary_info
    
    # Initialize outputs
    u_bc = torch.zeros(batch_size, device=device)
    v_bc = torch.zeros(batch_size, device=device)
    h_bc = torch.zeros(batch_size, device=device)
    
    # Get boundary identifiers (no need for [:, 0] as it's already 1D)
    boundary_ids = boundary_ids  # boundary_ids is already 1D

    #print for debugging
    #print(f"Boundary IDs: {boundary_ids}")
    #print(f"Batch size: {batch_size}")
    
    # Inlet (ID: 1)
    inlet_mask = (boundary_ids == 1)
    u_bc[inlet_mask] = 0.0
    v_bc[inlet_mask] = 0.0
    #h_bc[inlet_mask] = 1.0

    #print for debugging
    #print(f"Inlet mask: {inlet_mask}")
    #print(f"u_bc: {u_bc}")
    #print(f"v_bc: {v_bc}")
    #print(f"h_bc: {h_bc}")
    
    # Outlet (ID: 2)
    outlet_mask = (boundary_ids == 2)
    u_bc[outlet_mask] = 0.0
    v_bc[outlet_mask] = 0.0    
    # Free outflow - Neumann boundary condition
    # This will be handled in the PINN loss function
    
    # Walls (ID: 3)
    walls_mask = (boundary_ids == 3)
    u_bc[walls_mask] = 0.0  # No-slip condition
    v_bc[walls_mask] = 0.0
    
    # Obstacle (ID: 4)
    obstacle_mask = (boundary_ids == 4)
    u_bc[obstacle_mask] = 0.0  # No-slip condition
    v_bc[obstacle_mask] = 0.0
    
    return torch.stack([h_bc, u_bc, v_bc], dim=1)

def pinn_train(dataset, config_file):
    """
    Train the PINN model.
    
    Args:
        dataset (PINNDataset): Dataset containing collocation points
        config_file (str): Path to configuration file
    """
    print("\n=== Starting PINN Training ===")
    
    # Create model and trainer
    model = SWE_PINN(config_file=config_file)
    trainer = PINNTrainer(model, config_file=config_file)
    print(f"Model device: {next(model.parameters()).device}")
    
    # Get points from dataset
    pde_points = dataset.get_pde_points()
    initial_points = dataset.get_initial_points()
    boundary_points, boundary_ids, boundary_normals = dataset.get_boundary_points()
    
    # Train model with boundary and initial conditions
    print("\nStarting training...")
    history = trainer.train(
        dataset,
        initial_conditions=initial_conditions,
        boundary_conditions=boundary_conditions
    )

    # Save training history
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    history_file = f'history_pinn_{timestamp}.json'
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=4)
    print(f"\nTraining history saved to {history_file}")

def plot_training_history(history_file):
    """Plot the training history."""
    print("\n=== Plotting Training History ===")
    
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    # Plot total loss
    plt.figure(figsize=(10, 6))
    plt.plot(history['loss_history'], label='Total Loss')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('PINN Training History - Total Loss')
    plt.grid(True)
    plt.legend()
    plt.savefig('plots/total_loss.png')
    plt.close()
    
    # Plot component losses
    plt.figure(figsize=(12, 8))
    for key in history['component_loss_history'].keys():
        plt.plot(history['component_loss_history'][key], label=key)
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('PINN Training History - Component Losses')
    plt.grid(True)
    plt.legend()
    plt.savefig('plots/component_losses.png')
    plt.close()

def predict_solution(config, model, gmsh2d_fileName, prediction_variable_names_list, bNodal, output_dir='plots/solutions'):    
    """
    Plot the PINN solution.
    
    Args:
        config (Config): Configuration object
        model (SWE_PINN): PINN model
        dataset (PINNDataset): Dataset containing collocation points
        output_dir (str): Directory to save the plots
    """


    print("\n=== Plotting PINN Solution ===")
    
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    
    # Get device from model
    device = next(model.parameters()).device

    # Get time information from config
    t_start = config.sampling.domain.t_min
    t_end = config.sampling.domain.t_max
    num_time_steps = config.sampling.num_time_steps

    # Create time steps
    time_points = np.linspace(t_start, t_end, num_time_steps, dtype=np.float32)
    
    with torch.no_grad():
        # Loop over time steps
        for itime, t in enumerate(time_points):
            #predict on the gmsh mesh for the current time step
            vtkFileSaveName = f'{output_dir}/solution_{str(itime).zfill(6)}.vtk'
            predict_on_gmsh2d_mesh(model, gmsh2d_fileName, prediction_variable_names_list, vtkFileSaveName, bNodal, time_pred=t, device=device)
              
        

if __name__ == "__main__":
    main_start_time = time.time()
    print("\n=== Starting PINN Example ===")
    
    # Load configuration
    config_file = 'pinn_config.yaml'
    print(f"\nLoading configuration from {config_file}")
    config = Config(config_file)
    
    # Set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load dataset
    print("\nLoading dataset...")
    data_dir = "pinn_points"
    dataset = PINNDataset(
        domain={
            'x_min': config.sampling.domain.x_min,
            'x_max': config.sampling.domain.x_max,
            'y_min': config.sampling.domain.y_min,
            'y_max': config.sampling.domain.y_max,
            't_min': config.sampling.domain.t_min,
            't_max': config.sampling.domain.t_max
        },
        data_dir=data_dir
    )
    
    # Create and train model
    model = SWE_PINN(config_file=config_file)
    model = model.to(device)  # Move model to device
    
    # Train the model
    pinn_train(dataset, config_file)
    
    # Find best model checkpoint
    best_model_path = os.path.join('./checkpoints', 'pinn_epoch_best.pt')
    if os.path.exists(best_model_path):
        print(f"\nFound best model checkpoint: {best_model_path}")
        
        # Plot training history
        history_files = [f for f in os.listdir('.') if f.startswith('history_pinn_')]
        if history_files:
            latest_history = max(history_files)
            plot_training_history(latest_history)
        
        # Load best model and plot solutions
        model = SWE_PINN(config_file=config_file)
        checkpoint = torch.load(best_model_path, map_location=device)  # Add map_location
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)  # Move model to device

        gmsh2d_fileName = "case_preparation/generate_PINN_points/block_in_channel.msh"
        prediction_variable_names_list = ["h", "u", "v"]
        bNodal = True

        predict_solution(config, model, gmsh2d_fileName, prediction_variable_names_list, bNodal, output_dir='plots/solutions')
    
    # Print execution time
    main_execution_time = time.time() - main_start_time
    print(f"\n⏱️ Total execution time: {main_execution_time:.2f} seconds")
    print("\nPINN example completed successfully!") 