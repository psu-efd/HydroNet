"""
Example usage of the HydroNet framework.

This script demonstrates how to use the PINN component of HydroNet for solving 
shallow water equations with physics and data-driven constraints.
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
from vtk import vtkPoints, vtkPolyData, vtkFloatArray, vtkPolyDataWriter

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
from HydroNet.utils.predict_on_vtk import predict_on_vtk2d_mesh

def save_predictions_and_true_values_to_vtk(predictions_and_true_values_file, output_dir='plots'):
    """
    Save predictions and true values to separate point VTK files for each type of point.
    
    Args:
        predictions_and_true_values_file (str): File containing predictions and true values
        output_dir (str): Directory to save the VTK files
    """

    # Load predictions and true values
    with open(predictions_and_true_values_file, 'r') as f:
        predictions_and_true_values = json.load(f)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    def save_points_to_vtk(coords, h_pred, u_pred, v_pred, h_true=None, u_true=None, v_true=None, filename=None):
        """Helper function to save points and their values to a VTK file."""
        if coords is None or len(coords) == 0:
            return

        # Create VTK points object
        points = vtkPoints()
        
        # Create arrays for each variable
        h_pred_array = vtkFloatArray()
        h_pred_array.SetName("h_pred")
        velocity_pred_array = vtkFloatArray()
        velocity_pred_array.SetNumberOfComponents(3)  # 3D velocity vector
        velocity_pred_array.SetName("velocity_pred")
        
        h_true_array = vtkFloatArray()
        h_true_array.SetName("h_true")
        velocity_true_array = vtkFloatArray()
        velocity_true_array.SetNumberOfComponents(3)  # 3D velocity vector
        velocity_true_array.SetName("velocity_true")
        
        # Add points and values
        for i in range(len(coords)):
            # Add z=0 for 2D points
            points.InsertNextPoint(coords[i][0], coords[i][1], 0.0)
            # Extract scalar values from arrays
            h_pred_array.InsertNextValue(float(h_pred[i].item()))
            velocity_pred_array.InsertNextTuple3(float(u_pred[i].item()), float(v_pred[i].item()), 0.0)
            
            if h_true is not None:
                h_true_array.InsertNextValue(float(h_true[i].item()))
                velocity_true_array.InsertNextTuple3(float(u_true[i].item()), float(v_true[i].item()), 0.0)
            else:
                h_true_array.InsertNextValue(0.0)
                velocity_true_array.InsertNextTuple3(0.0, 0.0, 0.0)
        
        # Create polydata and add points and arrays
        polydata = vtkPolyData()
        polydata.SetPoints(points)
        polydata.GetPointData().AddArray(h_pred_array)
        polydata.GetPointData().AddArray(velocity_pred_array)
        polydata.GetPointData().AddArray(h_true_array)
        polydata.GetPointData().AddArray(velocity_true_array)
        
        # Write to file
        writer = vtkPolyDataWriter()
        writer.SetFileName(filename)
        writer.SetInputData(polydata)
        writer.Write()

    # Save PDE points
    if predictions_and_true_values['bPDE_loss']:
        pde_points = np.array(predictions_and_true_values['pde_points'])
        h_pred_pde = np.array(predictions_and_true_values['h_pred_pde_points'])
        u_pred_pde = np.array(predictions_and_true_values['u_pred_pde_points'])
        v_pred_pde = np.array(predictions_and_true_values['v_pred_pde_points'])
        save_points_to_vtk(pde_points, h_pred_pde, u_pred_pde, v_pred_pde,
                          filename=os.path.join(output_dir, 'pde_points.vtk'))
    
    # Save initial points
    #check if bInitial_loss is in the predictions_and_true_values dictionary
    if 'bInitial_loss' in predictions_and_true_values:
        if predictions_and_true_values['bInitial_loss']:
            initial_points = np.array(predictions_and_true_values['initial_points'])
            h_pred_initial = np.array(predictions_and_true_values['h_pred_initial_points'])
            u_pred_initial = np.array(predictions_and_true_values['u_pred_initial_points'])
            v_pred_initial = np.array(predictions_and_true_values['v_pred_initial_points'])
            h_true_initial = np.array(predictions_and_true_values['h_true_initial_points'])
            u_true_initial = np.array(predictions_and_true_values['u_true_initial_points'])
            v_true_initial = np.array(predictions_and_true_values['v_true_initial_points'])
            save_points_to_vtk(initial_points, h_pred_initial, u_pred_initial, v_pred_initial,
                            h_true_initial, u_true_initial, v_true_initial,
                            filename=os.path.join(output_dir, 'initial_points.vtk'))
    
    # Save boundary points
    if predictions_and_true_values['bBoundary_loss']:
        boundary_points = np.array(predictions_and_true_values['boundary_points'])
        h_pred_boundary = np.array(predictions_and_true_values['h_pred_boundary_points'])
        u_pred_boundary = np.array(predictions_and_true_values['u_pred_boundary_points'])
        v_pred_boundary = np.array(predictions_and_true_values['v_pred_boundary_points'])
        save_points_to_vtk(boundary_points, h_pred_boundary, u_pred_boundary, v_pred_boundary,
                          filename=os.path.join(output_dir, 'boundary_points.vtk'))
    
    # Save data points
    if predictions_and_true_values['bData_loss']:
        data_points = np.array(predictions_and_true_values['data_points'])
        h_pred_data = np.array(predictions_and_true_values['h_pred_data_points'])
        u_pred_data = np.array(predictions_and_true_values['u_pred_data_points'])
        v_pred_data = np.array(predictions_and_true_values['v_pred_data_points'])
        h_true_data = np.array(predictions_and_true_values['h_true_data_points'])
        u_true_data = np.array(predictions_and_true_values['u_true_data_points'])
        v_true_data = np.array(predictions_and_true_values['v_true_data_points'])
        save_points_to_vtk(data_points, h_pred_data, u_pred_data, v_pred_data,
                          h_true_data, u_true_data, v_true_data,
                          filename=os.path.join(output_dir, 'data_points.vtk'))

def pinn_train(trainer):
    """
    Train the PINN model.
    
    Args:
        trainer (PINNTrainer): PINN trainer
    """
    print("\n=== Starting PINN Training ===")
    
    # Train model with boundary and initial conditions
    print("\nStarting training...")
    history, predictions_and_true_values = trainer.train()

    # Save training history
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    history_file = f'history_pinn_{timestamp}.json'
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=4)
    print(f"\nTraining history saved to {history_file}")

    # Convert tensors to lists before saving
    predictions_and_true_values_save = {}
    for key, value in predictions_and_true_values.items():
        if isinstance(value, torch.Tensor):
            predictions_and_true_values_save[key] = value.detach().cpu().numpy().tolist()
        else:
            predictions_and_true_values_save[key] = value

    # Save predictions and true values
    os.makedirs('plots', exist_ok=True)
    predictions_and_true_values_file = 'plots/predictions_and_true_values.json'
    with open(predictions_and_true_values_file, 'w') as f:
        json.dump(predictions_and_true_values_save, f, indent=4)
    print(f"\nPredictions and true values saved to {predictions_and_true_values_file}")

def plot_training_history(history_file):
    """Plot the training history."""
    print("\n=== Plotting Training History ===")
    
    with open(history_file, 'r') as f:
        history = json.load(f)

    # Dictionary history keys: loss_history, component_loss_history, and sub-keys of component_loss_history
    # print("\n=== History Dictionary Structure ===")
    # def print_dict_structure(d, indent=0):
    #     for key, value in d.items():
    #         print("  " * indent + f"Key: {key}")
    #         if isinstance(value, dict):
    #             print_dict_structure(value, indent + 1)
    #         elif isinstance(value, list):
    #             print("  " * (indent + 1) + f"Type: list with {len(value)} elements")
    #             if len(value) > 0:
    #                 print("  " * (indent + 1) + f"First element type: {type(value[0])}")
    #         else:
    #             print("  " * (indent + 1) + f"Type: {type(value)}")

    # print_dict_structure(history)
    # print("\n=== End of Dictionary Structure ===\n")
    
    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    # Plot total loss
    plt.figure(figsize=(10, 6))
    plt.plot(history['loss_history'], label='Total Loss')
    plt.plot(history['component_loss_history']['weighted_total_loss'], label='weighted_total_loss', linewidth=2)
    plt.plot(history['component_loss_history']['unweighted_total_loss'], label='unweighted_total_loss', linewidth=2)
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('PINN Training History - Total Loss')
    plt.grid(True)
    plt.legend()
    plt.savefig('plots/total_loss.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot total loss components: pde, boundary, data
    plt.figure(figsize=(10, 6))
    plt.plot(history['component_loss_history']['unweighted_total_loss'], label='unweighted_total_loss', linewidth=2)
    plt.plot(history['component_loss_history']['loss_components']['pde_loss'], label='pde_loss', linewidth=2)
    plt.plot(history['component_loss_history']['loss_components']['boundary_loss'], label='boundary_loss', linewidth=2)
    plt.plot(history['component_loss_history']['loss_components']['data_loss'], label='data_loss', linewidth=2)
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('PINN Training History - Total Loss Components')
    plt.grid(True)
    plt.legend()
    plt.savefig('plots/total_loss_components.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot PDE/BC/IC/Data losses
    plt.figure(figsize=(12, 8))
    plt.plot(history['component_loss_history']['pde_loss_components']['continuity_loss'], label='PDE (continuity) Loss', linewidth=2)
    plt.plot(history['component_loss_history']['pde_loss_components']['momentum_x_loss'], label='PDE (x-momentum) Loss', linewidth=2)
    plt.plot(history['component_loss_history']['pde_loss_components']['momentum_y_loss'], label='PDE (y-momentum) Loss', linewidth=2)
    #plt.plot(history['initial_condition_loss'], label='IC Loss', linewidth=2)
    plt.plot(history['component_loss_history']['loss_components']['boundary_loss'], label='BC Loss', linewidth=2)
    plt.plot(history['component_loss_history']['loss_components']['data_loss'], label='Data Loss', linewidth=2)
    plt.yscale('log')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('PINN Training History - PDE/BC/IC/Data Losses', fontsize=14)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig('plots/component_losses_detailed.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot data loss components
    plt.figure(figsize=(12, 8))
    plt.plot(history['component_loss_history']['data_loss_components']['total_data_loss'], label='Total Data Loss', linewidth=2)
    plt.plot(history['component_loss_history']['data_loss_components']['data_h_loss'], label='Data Loss (h)', linewidth=2)
    plt.plot(history['component_loss_history']['data_loss_components']['data_u_loss'], label='Data Loss (u)', linewidth=2)
    plt.plot(history['component_loss_history']['data_loss_components']['data_v_loss'], label='Data Loss (v)', linewidth=2)
    plt.yscale('log')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('PINN Training History - Data Loss Components', fontsize=14)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig('plots/data_loss_components.png', dpi=300, bbox_inches='tight')
    plt.close()

    # plot boundary loss components
    plt.figure(figsize=(12, 8))
    
    # Define BC loss prefixes to look for
    bc_prefixes = ['wall', 'inlet-q', 'exit-h']
    
    # Store losses and their averages for ranking
    bc_losses = {}
    
    # Filter and plot BC losses
    for key in history['component_loss_history']['boundary_loss_components'].keys():
        if any(key.startswith(prefix) for prefix in bc_prefixes):
            # Create a more readable label by replacing underscores with spaces and capitalizing
            label = key.replace('_', ' ').title()
            losses = history['component_loss_history']['boundary_loss_components'][key]
            bc_losses[label] = np.mean(losses)  # Store average loss
            plt.plot(losses, label=label, linewidth=2)
    
    # Find top 3 losses
    top_3_losses = sorted(bc_losses.items(), key=lambda x: x[1], reverse=True)[:3]
    
    # Add text box with top 3 losses
    textstr = '\n'.join([f'{i+1}. {name}: {value:.2e}' for i, (name, value) in enumerate(top_3_losses)])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    plt.yscale('log')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('PINN Training History - Boundary Condition Losses', fontsize=14)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('plots/boundary_loss_components.png', dpi=300, bbox_inches='tight')
    plt.close()

    # plot the loss weights history 
    plt.figure(figsize=(12, 8))
    
    # Define weight keys to plot
    weight_keys = ['pde_loss', 'boundary_loss', 'data_loss']

    #print(history['component_loss_history']['loss_weights'])
    
    # Plot each weight history
    for key in weight_keys:
        if f'{key}' in history['component_loss_history']['loss_weights']:
            
            plt.plot(history['component_loss_history']['loss_weights'][f'{key}'], 
                    label=f'{key} Weight', linewidth=2)
    #make y axis log scale
    plt.yscale('log')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Weight Value', fontsize=12)
    plt.title('PINN Training History - Loss Weights', fontsize=14)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend(fontsize=10, loc='upper right')
    plt.tight_layout()
    plt.savefig('plots/loss_weights_history.png', dpi=300, bbox_inches='tight')
    plt.close()

def predict_solution(config, model, mesh_stats, data_stats, vtk2d_fileName, prediction_variable_names_list, bNodal, output_dir='plots/solutions'):    
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
    
    # Get device 
    device = model.get_device()
    print(f"\nUsing device: {device}")

    # Get time information from config
    #t_start = config.sampling.domain.t_min
    #t_end = config.sampling.domain.t_max
    #num_time_steps = config.sampling.num_time_steps

    # Create time steps
    #time_points = np.linspace(t_start, t_end, num_time_steps, dtype=np.float32)

    # Get normalization parameters
    bNormalize = model.get_bNormalize()
    normalization_method = model.get_normalization_method()
    
    with torch.no_grad():
        # Loop over time steps
        #for itime, t in enumerate(time_points):
        #predict on the gmsh mesh for the current time step
        if bNodal:
            vtkFileSaveName = f'{output_dir}/solution_nodal.vtk'
        else:
            vtkFileSaveName = f'{output_dir}/solution_cell_centered.vtk'

        predict_on_vtk2d_mesh(model, vtk2d_fileName, prediction_variable_names_list, vtkFileSaveName, bNodal, 
                              bNormalize=bNormalize, normalization_method=normalization_method,
                              mesh_stats=mesh_stats, data_stats=data_stats,
                              device=device)
              
        

if __name__ == "__main__":
    main_start_time = time.time()
    print("\n=== Starting PINN case ===")
    
    # Load configuration
    config_file = 'pinn_config.yaml'
    print(f"\nLoading configuration from {config_file}")
    config = Config(config_file)

    # Create model
    model = SWE_PINN(config)    
    print(f"Model device: {next(model.parameters()).device}")
    
    # Load dataset
    print("\nLoading dataset...")    
    dataset = PINNDataset(config, model)

    dataset.print_stats()

    #exit()

    # Create trainer
    trainer = PINNTrainer(model, dataset, config)    
    
    # Train the model
    pinn_train(trainer)
    
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
        checkpoint = torch.load(best_model_path, map_location=model.get_device())  # Add map_location
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(model.get_device())  # Move model to device

        # Get stats
        mesh_stats = dataset.get_mesh_stats()
        data_stats = dataset.get_data_stats()

        # Predict on the vtk file (only the points in the vtk file are used for prediction)
        vtk2d_fileName = "SRH2D_oneD_channel_with_bump_C_0010.vtk"  
        prediction_variable_names_list = ["h", "u", "v"]

        bNodal = True
        predict_solution(config, model, mesh_stats, data_stats, vtk2d_fileName, prediction_variable_names_list, bNodal, output_dir='plots/solutions')

        bNodal = False
        predict_solution(config, model, mesh_stats, data_stats, vtk2d_fileName, prediction_variable_names_list, bNodal, output_dir='plots/solutions')

        # Save predictions and true values to VTK files
        save_predictions_and_true_values_to_vtk("plots/predictions_and_true_values.json", output_dir='plots')
    
    # Print execution time
    main_execution_time = time.time() - main_start_time
    print(f"\n⏱️ Total execution time: {main_execution_time:.2f} seconds")
    print("\nPINN example completed successfully!") 