"""
Data loading utilities for DeepONet training data (with optional physics-informed constraints).
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import h5py
from typing import Tuple, Optional
import json

from ...utils.config import Config
from ...models.PINN.data import PINNDataset


class PI_SWE_DeepONetDataset(Dataset):
    """
    Unified dataset for DeepONet training data with optional physics-informed constraints.
    
    This class handles:
    - DeepONet training data (branch_inputs, trunk_inputs, outputs) from HDF5 files
    - Optional physics constraint data (PDE points, boundary points, initial points) from PINNDataset
    
    The DeepONet data is expected to be normalized already.
    
    The DeepONet data file is expected to be called "data.h5" and to be in the following format:
    - branch_inputs: Input functions for the branch net.
    - trunk_inputs: Coordinates for the trunk net.
    - outputs: Corresponding output values.
    """
    
    def __init__(self, data_path, config):
        """
        Initialize the DeepONet dataset.
        
        Args:
            data_path (str): Path to the data directory containing the HDF5 file.
            config (Config): Configuration object.
                The config should contain model.use_physics_loss to determine whether
                to initialize physics dataset.
        """
        if not isinstance(config, Config):
            raise ValueError("config must be a Config object")
        
        self.data_path = data_path
        self.config = config
        
        # Determine if physics dataset should be used from config
        self.use_physics_dataset = config.get_required_config("model.use_physics_loss")

        # Dimension of the branch input, trunk input, and output of the data
        self.branch_dim = 0
        self.trunk_dim = 0
        self.output_dim = 0
        
        # Load the DeepONet data
        self._load_data()
        
        # Get the normalization stats
        # get the upper directory of the data_path
        self.upper_data_path = os.path.dirname(self.data_path)
        
        if not os.path.exists(os.path.join(self.upper_data_path, 'all_DeepONet_stats.json')):
            raise FileNotFoundError(f"Required file all_DeepONet_stats.json not found in {self.upper_data_path}")
        
        with open(os.path.join(self.upper_data_path, 'all_DeepONet_stats.json'), 'r') as f:
            self.all_DeepONet_stats = json.load(f)
        
        #print("all_DeepONet_stats: ", self.all_DeepONet_stats)
        
        # Initialize physics dataset if needed
        self.physics_dataset = None
        if self.use_physics_dataset:
            self.physics_dataset = PINNDataset(config)
    
    def _load_data(self):
        """
        Load the hydraulic simulation data from HDF5 files.
        """
        
        # Check if the data directory exists
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data directory {self.data_path} does not exist")
                
        try:
            # Check if required HDF5 file exists
            h5_file = os.path.join(self.data_path, 'data.h5')
            
            if not os.path.exists(h5_file):
                raise FileNotFoundError(f"Required file data.h5 not found in {self.data_path}")
                
            # Load data from HDF5 file
            with h5py.File(h5_file, 'r') as f:
                # Load input functions (for branch net)
                self.branch_inputs = f['branch_inputs'][:].astype(np.float64)
                self.branch_dim = self.branch_inputs.shape[1]

                # Load coordinates (for trunk net)
                self.trunk_inputs = f['trunk_inputs'][:].astype(np.float64)
                self.trunk_dim = self.trunk_inputs.shape[1]
                
                # Load output values (h, u, v)
                self.outputs = f['outputs'][:].astype(np.float64)
                self.output_dim = self.outputs.shape[1]
            
            print("branch_inputs.shape: ", self.branch_inputs.shape)
            print("trunk_inputs.shape: ", self.trunk_inputs.shape)
            print("outputs.shape: ", self.outputs.shape)
            
            print(f"Loaded {len(self.branch_inputs)} samples from {self.data_path}")
            
        except Exception as e:
            print(f"Error loading data: {e}")            
            raise RuntimeError("Error loading data. Please check the data directory and file structure.")
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.branch_inputs)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample.
            
        Returns:
            tuple: (branch_input, trunk_input, output) where:
                - branch_input is the input function for the branch net
                - trunk_input is the coordinate for the trunk net
                - output is the corresponding output value
        """
        branch_input = self.branch_inputs[idx]
        trunk_input = self.trunk_inputs[idx]
        output = self.outputs[idx]
        
        # Convert to torch tensors
        branch_input = torch.tensor(branch_input, dtype=torch.float32)
        trunk_input = torch.tensor(trunk_input, dtype=torch.float32)
        output = torch.tensor(output, dtype=torch.float32)
            
        return branch_input, trunk_input, output
    
    def get_deeponet_stats(self):
        """
        Get the normalization stats for the DeepONet dataset.
        
        Returns:
            dict: Dictionary containing DeepONet statistics.
        """
        return self.all_DeepONet_stats
    
    # Physics dataset methods (only available if physics_dataset is initialized)
    
    def get_pinn_pde_points(self):
        """
        Get PINN PDE collocation points and associated data.
        
        Returns:
            tuple or None: (pinn_pde_points, pinn_pde_data) if available, None otherwise.
        """
        if self.physics_dataset is None:
            print("Physics dataset is not initialized. Returning None.")
            return None
        return self.physics_dataset.get_pde_points()
    
    def get_pinn_initial_points(self):
        """
        Get initial condition points and values.
        
        Returns:
            tuple or None: (initial_points, initial_values) if available, None otherwise.
        """
        if self.physics_dataset is None:
            print("Physics dataset is not initialized. Returning None.")
            return None
        return self.physics_dataset.get_initial_points()
    
    def get_pinn_boundary_points(self):
        """
        Get boundary condition points and associated information.
        
        Returns:
            tuple or None: (boundary_points, boundary_ids, boundary_z, boundary_normals, 
                          boundary_lengths, boundary_ManningN) if available, None otherwise.
        """
        if self.physics_dataset is None:
            print("Physics dataset is not initialized. Returning None.")
            return None
        return self.physics_dataset.get_boundary_points()
    
    def get_pinn_data_points(self):
        """
        Get data points, values, and flags.
        
        Returns:
            tuple or None: (data_points, data_values, data_flags) if available, None otherwise.
        """
        if self.physics_dataset is None:
            print("Physics dataset is not initialized. Returning None.")
            return None
        return self.physics_dataset.get_data_points()
    
    def get_pinn_mesh_stats(self):
        """
        Get statistics of the mesh points for normalization.
        
        Returns:
            dict or None: Dictionary containing mesh statistics, or None if physics dataset not initialized.
        """
        if self.physics_dataset is None:
            print("Physics dataset is not initialized. Returning None.")
            return None
        return self.physics_dataset.get_mesh_stats()
    
    def get_pinn_data_stats(self):
        """
        Get statistics of the data points for normalization.
        
        Returns:
            dict or None: Dictionary containing data statistics, or None if physics dataset not initialized.
        """
        if self.physics_dataset is None:
            print("Physics dataset is not initialized. Returning None.")
            return None
        return self.physics_dataset.get_data_stats()
    
    def get_pinn_collocation_points(self):
        """
        Get all collocation points from the physics dataset.
        
        Returns:
            tuple or None: (interior_points, boundary_points, initial_points, data_points) if available, None otherwise.
        """
        if self.physics_dataset is None:
            print("Physics dataset is not initialized. Returning None.")
            return None
        return self.physics_dataset.get_collocation_points()


def create_pi_deeponet_dataloader(dataset, batch_size=32, shuffle=True, num_workers=4):
    """
    Create a DataLoader for a DeepONet dataset (with or without physics-informed constraints).
    
    This creates a DataLoader for the DeepONet training data portion of the dataset.
    If physics constraints are enabled, physics constraint data should be accessed via 
    dataset methods (get_pde_points(), etc.).
    
    Args:
        dataset (PI_SWE_DeepONetDataset): The dataset to load.
        batch_size (int, optional): Batch size.
        shuffle (bool, optional): Whether to shuffle the data.
        num_workers (int, optional): Number of worker processes.
        
    Returns:
        DataLoader: DataLoader for the DeepONet training data.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

