"""
Data loading utilities for DeepONet training data.
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import h5py
from typing import Tuple, Optional
import json


class SWE_DeepONetDataset(Dataset):
    """
    Dataset for DeepONet training data.
    
    This dataset handles input functions (branch net inputs) and coordinates (trunk net inputs),
    along with corresponding output values. It is designed to work with any type of operator learning.
    
    The data is expected to be normalized.

    The data is expected to be called "data.h5" and to be in the following format:
    - branch_inputs: Input functions for the branch net.
    - trunk_inputs: Coordinates for the trunk net.
    - outputs: Corresponding output values.
    """

    def __init__(self, data_path, config):
        """
        Initialize the DeepONet dataset.
        
        Args:
            data_path (str): Path to the data directory.
            config (dict): Configuration dictionary.
        """
        self.data_path = data_path
        self.config = config

        # Load the data
        self._load_data()

        # Get the normalization stats
        # get the upper directory of the data_path
        self.upper_data_path = os.path.dirname(self.data_path)

        if not os.path.exists(os.path.join(self.upper_data_path, 'all_DeepONet_stats.json')):
            raise FileNotFoundError(f"Required file all_DeepONet_stats.json not found in {self.upper_data_path}")
        
        with open(os.path.join(self.upper_data_path, 'all_DeepONet_stats.json'), 'r') as f:
            self.all_DeepONet_stats = json.load(f)
        
        print("all_DeepONet_stats: ", self.all_DeepONet_stats)
        
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
                
                # Load coordinates (for trunk net)
                self.trunk_inputs = f['trunk_inputs'][:].astype(np.float64)
                
                # Load output values (h, u, v)
                self.outputs = f['outputs'][:].astype(np.float64)

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
        """
        return self.all_DeepONet_stats


def create_deeponet_dataloader(dataset, batch_size=32, shuffle=True, num_workers=4):
    """
    Create a DataLoader for a DeepONet dataset.
    
    Args:
        dataset (Dataset): The dataset to load.
        batch_size (int, optional): Batch size.
        shuffle (bool, optional): Whether to shuffle the data.
        num_workers (int, optional): Number of worker processes.
        
    Returns:
        DataLoader: DataLoader for the dataset.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )


