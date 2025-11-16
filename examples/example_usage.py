"""
Example usage of the HydroNet framework.

This script demonstrates how to use the three main components of HydroNet:
1. DeepONet for learning the operator of shallow water equations
2. PINN for solving 2D shallow water equations
3. Physics-Informed DeepONet for combining data-driven and physics-informed approaches
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import Dataset

# Add parent directory to path to import HydroNet modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from HydroNet.src.models.DeepONet.model import SWE_DeepONetModel
from HydroNet.src.models.DeepONet.trainer import SWE_DeepONetTrainer
from HydroNet.src.models.PINN.model import SWE_PINN
from HydroNet.src.models.PINN.trainer import PINNTrainer
from HydroNet.src.models.PI_DeepONet.model import PI_DeepONetModel
from HydroNet.src.models.PI_DeepONet.trainer import PI_DeepONetTrainer
from HydroNet.src.utils.visualization import plot_comparison, plot_vector_field


def create_dummy_data(num_samples=100, grid_size=20, save_dir='../src/data'):
    """
    Create dummy data for demonstration purposes.
    
    Args:
        num_samples (int): Number of samples to generate.
        grid_size (int): Size of the spatial grid.
        save_dir (str): Directory to save the data.
    """
    print("Creating dummy data for demonstration...")
    
    # Create directories
    os.makedirs(os.path.join(save_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'val'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'test'), exist_ok=True)
    
    # Generate grid
    x = np.linspace(0, 1, grid_size)
    y = np.linspace(0, 1, grid_size)
    X, Y = np.meshgrid(x, y)
    coords = np.stack([X.flatten(), Y.flatten(), np.zeros_like(X.flatten())], axis=1)  # (x, y, t=0)
    
    # Generate dummy branch inputs (e.g., initial conditions, boundary conditions)
    branch_inputs = np.random.rand(num_samples, 10)  # 10 parameters per sample
    
    # Generate dummy outputs
    outputs = []
    for i in range(num_samples):
        # Generate water depth (h)
        h = 0.1 + 0.5 * np.exp(-10 * ((X - 0.5)**2 + (Y - 0.5)**2))
        
        # Generate velocities (u, v)
        u = 0.2 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
        v = -0.2 * np.cos(2 * np.pi * X) * np.sin(2 * np.pi * Y)
        
        # Add some noise and parameter dependence
        params = branch_inputs[i]
        h = h + 0.05 * params[0] * np.sin(4 * np.pi * X * params[1])
        u = u + 0.05 * params[2] * np.cos(4 * np.pi * Y * params[3])
        v = v + 0.05 * params[4] * np.sin(4 * np.pi * X * params[5])
        
        # Stack h, u, v
        output = np.stack([h.flatten(), u.flatten(), v.flatten()], axis=1)
        outputs.append(output)
    
    # Repeat coordinates for each sample
    trunk_inputs = np.tile(coords, (num_samples, 1, 1))
    
    # Stack outputs
    outputs = np.stack(outputs, axis=0)
    
    # Split data
    train_size = int(0.7 * num_samples)
    val_size = int(0.15 * num_samples)
    test_size = num_samples - train_size - val_size
    
    # Save training data
    np.save(os.path.join(save_dir, 'train', 'branch_inputs.npy'), branch_inputs[:train_size])
    np.save(os.path.join(save_dir, 'train', 'trunk_inputs.npy'), trunk_inputs[:train_size])
    np.save(os.path.join(save_dir, 'train', 'outputs.npy'), outputs[:train_size])
    
    # Save validation data
    np.save(os.path.join(save_dir, 'val', 'branch_inputs.npy'), branch_inputs[train_size:train_size+val_size])
    np.save(os.path.join(save_dir, 'val', 'trunk_inputs.npy'), trunk_inputs[train_size:train_size+val_size])
    np.save(os.path.join(save_dir, 'val', 'outputs.npy'), outputs[train_size:train_size+val_size])
    
    # Save test data
    np.save(os.path.join(save_dir, 'test', 'branch_inputs.npy'), branch_inputs[train_size+val_size:])
    np.save(os.path.join(save_dir, 'test', 'trunk_inputs.npy'), trunk_inputs[train_size+val_size:])
    np.save(os.path.join(save_dir, 'test', 'outputs.npy'), outputs[train_size+val_size:])
    
    print(f"Dummy data created and saved to {save_dir}")


def example_deeponet():
    """Example of using DeepONet for learning the operator of shallow water equations."""
    print("\n=== DeepONet Example ===")
    
    # Create model
    model = SWE_DeepONetModel(config_file='../src/config/deeponet_config.yaml')
    
    # Create trainer
    trainer = SWE_DeepONetTrainer(model, config_file='../src/config/deeponet_config.yaml')
    
    # Train model (with very few epochs for demonstration)
    trainer.config.epochs = 5  # Override epochs for quick demonstration
    history = trainer.train()
    
    print("DeepONet training completed.")


def example_pinn():
    """Example of using PINN for solving 2D shallow water equations."""
    print("\n=== PINN Example ===")
    
    # Create model
    model = SWE_PINN(config_file='../src/config/pinn_config.yaml')
    
    # Create trainer
    trainer = PINNTrainer(model, config_file='../src/config/pinn_config.yaml')
    
    # Define initial conditions
    def initial_conditions(points):
        """
        Define initial conditions for the shallow water equations.
        
        Args:
            points (torch.Tensor): Points with shape [batch_size, 2] containing (x, y) coordinates.
            
        Returns:
            torch.Tensor: Initial conditions with shape [batch_size, 3] containing (h, u, v).
        """
        x, y = points[:, 0], points[:, 1]
        
        # Water depth (h) - Gaussian bump
        h = 0.1 + 0.5 * torch.exp(-10 * ((x - 0.5)**2 + (y - 0.5)**2))
        
        # Velocities (u, v) - initially zero
        u = torch.zeros_like(h)
        v = torch.zeros_like(h)
        
        return torch.stack([h, u, v], dim=1)
    
    # Define boundary conditions
    def boundary_conditions(points):
        """
        Define boundary conditions for the shallow water equations.
        
        Args:
            points (torch.Tensor): Points with shape [batch_size, 3] containing (x, y, t) coordinates.
            
        Returns:
            torch.Tensor: Boundary conditions with shape [batch_size, 3] containing (h, u, v).
        """
        x, y, t = points[:, 0], points[:, 1], points[:, 2]
        
        # Simple reflective boundary conditions
        h = 0.1 * torch.ones_like(x)
        u = torch.zeros_like(x)
        v = torch.zeros_like(x)
        
        return torch.stack([h, u, v], dim=1)
    
    # Train model (with very few epochs for demonstration)
    trainer.config.epochs = 5  # Override epochs for quick demonstration
    history = trainer.train(initial_conditions=initial_conditions, boundary_conditions=boundary_conditions)
    
    print("PINN training completed.")


def example_pi_deeponet():
    """Example of using Physics-Informed DeepONet."""
    print("\n=== Physics-Informed DeepONet Example ===")
    
    # Create model
    model = PI_DeepONetModel(config_file='../src/config/pi_deeponet_config.yaml')
    
    # Create trainer
    trainer = PI_DeepONetTrainer(model, config_file='../src/config/pi_deeponet_config.yaml')
    
    # Define initial conditions
    def initial_conditions(branch_input, points):
        """
        Define initial conditions for the shallow water equations.
        
        Args:
            branch_input (torch.Tensor): Branch input with shape [batch_size, branch_dim].
            points (torch.Tensor): Points with shape [batch_size, 2] containing (x, y) coordinates.
            
        Returns:
            torch.Tensor: Initial conditions with shape [batch_size, 3] containing (h, u, v).
        """
        x, y = points[:, 0], points[:, 1]
        
        # Water depth (h) - Gaussian bump
        h = 0.1 + 0.5 * torch.exp(-10 * ((x - 0.5)**2 + (y - 0.5)**2))
        
        # Velocities (u, v) - initially zero
        u = torch.zeros_like(h)
        v = torch.zeros_like(h)
        
        return torch.stack([h, u, v], dim=1)
    
    # Define boundary conditions
    def boundary_conditions(branch_input, points):
        """
        Define boundary conditions for the shallow water equations.
        
        Args:
            branch_input (torch.Tensor): Branch input with shape [batch_size, branch_dim].
            points (torch.Tensor): Points with shape [batch_size, 3] containing (x, y, t) coordinates.
            
        Returns:
            torch.Tensor: Boundary conditions with shape [batch_size, 3] containing (h, u, v).
        """
        x, y, t = points[:, 0], points[:, 1], points[:, 2]
        
        # Simple reflective boundary conditions
        h = 0.1 * torch.ones_like(x)
        u = torch.zeros_like(x)
        v = torch.zeros_like(x)
        
        return torch.stack([h, u, v], dim=1)
    
    # Train model (with very few epochs for demonstration)
    trainer.config.epochs = 5  # Override epochs for quick demonstration
    history = trainer.train(initial_conditions=initial_conditions, boundary_conditions=boundary_conditions)
    
    print("Physics-Informed DeepONet training completed.")


class PINNDataset(Dataset):
    def _generate_collocation_points(self):
        # ... existing code ...
        
        # Add boundary identifiers
        self.boundary_identifiers = torch.zeros(len(self.boundary_points))
        n_per_edge = self.n_boundary // 4
        
        # Mark different boundaries
        self.boundary_identifiers[:n_per_edge] = 1  # bottom
        self.boundary_identifiers[n_per_edge:2*n_per_edge] = 2  # top
        self.boundary_identifiers[2*n_per_edge:3*n_per_edge] = 3  # left
        self.boundary_identifiers[3*n_per_edge:] = 4  # right


if __name__ == "__main__":
    # Create dummy data for demonstration
    create_dummy_data(num_samples=100, grid_size=20, save_dir='../src/data')
    
    # Run examples
    example_deeponet()
    #example_pinn()
    #example_pi_deeponet()
    
    print("\nAll examples completed successfully!") 