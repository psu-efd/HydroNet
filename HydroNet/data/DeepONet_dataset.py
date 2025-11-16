"""
Data loading utilities for DeepONet training data.
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import h5py
from typing import Tuple, Optional


class SWE_DeepONetDataset(Dataset):
    """
    Dataset for DeepONet training data.
    
    This dataset handles input functions (branch net inputs) and coordinates (trunk net inputs),
    along with corresponding output values. It is designed to work with any type of operator learning
    problem that can be formulated in the DeepONet framework.
    """
    def __init__(self, data_dir, transform=None, normalize=False):
        """
        Initialize the hydraulic dataset.
        
        Args:
            data_dir (str): Directory containing the simulation data.
            transform (callable, optional): Optional transform to be applied to samples.
            normalize (bool, optional): Whether to normalize the data.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.normalize = normalize
        
        # Load the data
        self._load_data()
        
        # Normalize if required
        if self.normalize:
            self._normalize_data()
            
    def _load_data(self):
        """
        Load the hydraulic simulation data from HDF5 files.
        """
        # Path to data files
        data_path = self.data_dir
        
        # Check if the data directory exists
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data directory {data_path} does not exist")
                
        try:
            # Check if required HDF5 file exists
            h5_file = os.path.join(data_path, 'data.h5')
            
            if not os.path.exists(h5_file):
                raise FileNotFoundError(f"Required file data.h5 not found in {data_path}")
                
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
            
            print(f"Loaded {len(self.branch_inputs)} samples from {data_path}")
            
        except Exception as e:
            print(f"Error loading data: {e}")            
            raise RuntimeError("Error loading data. Please check the data directory and file structure.")
            
    def _normalize_data(self):
        """
        Normalize the data to improve training stability.
        """

        print("SWE_DeepONetDataset: Normalizing the data ...")

        # Branch input normalization
        self.branch_mean = np.mean(self.branch_inputs, axis=0, keepdims=True)
        self.branch_std = np.std(self.branch_inputs, axis=0, keepdims=True) + 1e-8
        self.branch_inputs = (self.branch_inputs - self.branch_mean) / self.branch_std

        # Trunk input normalization
        self.trunk_mean = np.mean(self.trunk_inputs, axis=0, keepdims=True)
        self.trunk_std = np.std(self.trunk_inputs, axis=0, keepdims=True) + 1e-8
        self.trunk_inputs = (self.trunk_inputs - self.trunk_mean) / self.trunk_std
        
        # Output normalization
        self.output_mean = np.mean(self.outputs, axis=0, keepdims=True)
        self.output_std = np.std(self.outputs, axis=0, keepdims=True) + 1e-8
        self.outputs = (self.outputs - self.output_mean) / self.output_std
        
        # Save normalization params for later use
        self.normalization = {
            'branch_mean': self.branch_mean,
            'branch_std': self.branch_std,
            'trunk_mean': self.trunk_mean,
            'trunk_std': self.trunk_std,
            'output_mean': self.output_mean,
            'output_std': self.output_std
        }

        #print the normalization parameters
        print("Normalization parameters:")
        print(self.normalization)
        
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
        
        # Apply any transformations
        if self.transform:
            branch_input, trunk_input, output = self.transform(branch_input, trunk_input, output)
            
        return branch_input, trunk_input, output


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


