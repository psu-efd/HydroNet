"""
Data loading utilities for Physics-Informed DeepONet training data.
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional

from ..utils.config import Config

from .DeepONet_dataset import SWE_DeepONetDataset
from .PINN_dataset import PINNDataset


class PI_SWE_DeepONetDataset(SWE_DeepONetDataset):
    """
    Dataset for Physics-Informed DeepONet training data.
    
    This class combines:
    - DeepONet training data (branch_inputs, trunk_inputs, outputs) from SWE_DeepONetDataset
    - Physics constraint data (PDE points, boundary points, initial points) from PINNDataset
    
    It inherits from SWE_DeepONetDataset for DeepONet data functionality and
    uses composition to include PINNDataset for physics constraint data.
    """
    def __init__(self, deeponet_data_dir, config):
        """
        Initialize the Physics-Informed DeepONet dataset.
        
        Args:
            deeponet_data_dir (str): Directory containing DeepONet training data (HDF5 file).
            config (Config): Configuration object.
        """
        if not isinstance(config, Config):
            raise ValueError("config must be a Config object")
        
        # Initialize parent class (SWE_DeepONetDataset) for DeepONet training data
        super(PI_SWE_DeepONetDataset, self).__init__(
            data_path=deeponet_data_dir,
            config=config
        )

        # Initialize physics dataset
        self.physics_dataset = PINNDataset(config)
        
    
    def get_pinn_pde_points(self):
        """
        Get PINN PDE collocation points and associated data.
        
        Returns:
            tuple or None: (pinn_pde_points, pinn_pde_data) if available, None otherwise.
        """
        return self.physics_dataset.get_pde_points()
    
    def get_pinn_initial_points(self):
        """
        Get initial condition points and values.
        
        Returns:
            tuple or None: (initial_points, initial_values) if available, None otherwise.
        """
        return self.physics_dataset.get_initial_points()
    
    def get_pinn_boundary_points(self):
        """
        Get boundary condition points and associated information.
        
        Returns:
            tuple or None: (boundary_points, boundary_ids, boundary_z, boundary_normals, 
                          boundary_lengths, boundary_ManningN) if available, None otherwise.
        """
        return self.physics_dataset.get_boundary_points()
    
    def get_pinn_data_points(self):
        """
        Get data points, values, and flags.
        
        Returns:
            tuple or None: (data_points, data_values, data_flags) if available, None otherwise.
        """
        return self.physics_dataset.get_data_points()
    
    def get_pinn_mesh_stats(self):
        """
        Get statistics of the mesh points for normalization.
        
        Returns:
            dict: Dictionary containing mesh statistics.
        """
        return self.physics_dataset.get_mesh_stats()
    
    def get_pinn_data_stats(self):
        """
        Get statistics of the data points for normalization.
        
        Returns:
            dict: Dictionary containing data statistics.
        """
        return self.physics_dataset.get_data_stats()
    
    def get_pinn_collocation_points(self):
        """
        Get all collocation points from the physics dataset.
        
        Returns:
            tuple: (interior_points, boundary_points, initial_points, data_points)
        """
        return self.physics_dataset.get_collocation_points()
    
    # Inherit __len__ and __getitem__ from SWE_DeepONetDataset for DeepONet training data
    # The physics dataset is accessed via the methods above, not through __getitem__


def create_pi_deeponet_dataloader(dataset, batch_size=32, shuffle=True, num_workers=4):
    """
    Create a DataLoader for a Physics-Informed DeepONet dataset.
    
    This creates a DataLoader for the DeepONet training data portion of the dataset.
    Physics constraint data should be accessed via dataset methods (get_pde_points(), etc.).
    
    Args:
        dataset (PI_SWE_DeepONetDataset): The dataset to load.
        batch_size (int, optional): Batch size.
        shuffle (bool, optional): Whether to shuffle the data.
        num_workers (int, optional): Number of worker processes.
        
    Returns:
        DataLoader: DataLoader for the Physics-Informed DeepONet training data.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

