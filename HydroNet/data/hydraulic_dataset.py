"""
Data loading utilities for hydraulic simulation data used by DeepONet.
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import h5py
from typing import Tuple, Optional


class HydraulicDataset(Dataset):
    """
    Dataset for hydraulic simulation data to be used with DeepONet.
    
    This dataset handles SRH-2D simulation data for shallow water equations.
    It consists of input functions (branch net inputs) and coordinates (trunk net inputs),
    along with corresponding output values.
    """
    def __init__(self, data_dir, transform=None, normalize=True):
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
        Load the hydraulic simulation data from files.
        
        """
        # Path to data files
        data_path = self.data_dir
        
        # Check if the data directory exists
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data directory {data_path} does not exist")
                
        try:
            # Check if required files exist
            branch_file = os.path.join(data_path, 'branch_inputs.npy')
            trunk_file = os.path.join(data_path, 'trunk_inputs.npy')
            outputs_file = os.path.join(data_path, 'outputs.npy')
            
            if not (os.path.exists(branch_file) and 
                   os.path.exists(trunk_file) and 
                   os.path.exists(outputs_file)):
                raise FileNotFoundError(f"Required files (branch_inputs.npy, trunk_inputs.npy, outputs.npy) not found in {data_path}")
                
            # Load input functions (for branch net)
            self.branch_inputs = np.load(branch_file)
            
            # Load coordinates (for trunk net)
            self.trunk_inputs = np.load(trunk_file)
            
            # Load output values (h, u, v)
            self.outputs = np.load(outputs_file)

            print("branch_inputs.shape: ", self.branch_inputs.shape)
            print("trunk_inputs.shape: ", self.trunk_inputs.shape)
            print("outputs.shape: ", self.outputs.shape)
            
            print(f"Loaded {len(self.branch_inputs)} samples from {data_path}")
            
        except Exception as e:
            print(f"Error loading data: {e}")
            print("Creating dummy data for demonstration purposes")
            self._create_dummy_data(data_path)
            
    def _create_dummy_data(self, data_path):
        """
        Create dummy data for demonstration purposes.
        """
        # Create some simple dummy data
        # Branch inputs: 100 samples, 10 features each (input function samples)
        self.branch_inputs = np.random.rand(100, 10)
        
        # Trunk inputs: 100 samples, 3 coordinates (x, y, t) coordinates
        # The model expects 3D coordinates, not 2D
        self.trunk_inputs = np.random.rand(100, 3)
        
        # Outputs: 100 samples, 3 variables (h, u, v)
        self.outputs = np.random.rand(100, 3)
        
        # Save the dummy data
        os.makedirs(data_path, exist_ok=True)
        np.save(os.path.join(data_path, 'branch_inputs.npy'), self.branch_inputs)
        np.save(os.path.join(data_path, 'trunk_inputs.npy'), self.trunk_inputs)
        np.save(os.path.join(data_path, 'outputs.npy'), self.outputs)
        
        print(f"Created and saved dummy data in {data_path}")
            
    def _normalize_data(self):
        """
        Normalize the data to improve training stability.
        """
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


def get_hydraulic_dataloader(dataset, batch_size=32, shuffle=True, num_workers=4):
    """
    Create a DataLoader for a hydraulic dataset.
    
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


class StreamingHydraulicDataset(Dataset):
    """
    A dataset class that streams data from HDF5 files in chunks to handle large datasets.
    """
    def __init__(self, 
                 data_dir: str, 
                 chunk_size: int = 10000,
                 normalize: bool = True,
                 transform: Optional[callable] = None,
                 shuffle: bool = True):
        """
        Initialize the streaming dataset.
        
        Args:
            data_dir (str): Directory containing the HDF5 files
            chunk_size (int): Number of samples to load at once
            normalize (bool): Whether to normalize the data
            transform (callable, optional): Optional transform to be applied
            shuffle (bool): Whether to shuffle the data
        """
        self.data_dir = data_dir
        self.chunk_size = chunk_size
        self.normalize = normalize
        self.transform = transform
        self.shuffle = shuffle
        
        # Get list of HDF5 files
        self.h5_files = [f for f in os.listdir(data_dir) if f.endswith('.h5')]
        self.h5_files.sort()  # Ensure consistent ordering
        
        # Initialize statistics for normalization
        self.branch_mean = None
        self.branch_std = None
        self.trunk_mean = None
        self.trunk_std = None
        self.output_mean = None
        self.output_std = None
        
        # Calculate total number of samples
        self.total_samples = 0
        for h5_file in self.h5_files:
            with h5py.File(os.path.join(data_dir, h5_file), 'r') as f:
                self.total_samples += f['branch_inputs'].shape[0]
        
        # Initialize current chunk
        self.current_chunk = None
        self.current_chunk_start = 0
        self.current_file_index = 0
        
        # Initialize shuffle indices
        self.shuffle_indices = None
        if self.shuffle:
            self._initialize_shuffle()
        
        # Load first chunk
        self._load_next_chunk()
        
        # Calculate normalization parameters if needed
        if normalize:
            self._calculate_normalization_params()
            
    def _initialize_shuffle(self):
        """Initialize the shuffle indices."""
        self.shuffle_indices = np.random.permutation(self.total_samples)
        
    def _get_shuffled_index(self, idx: int) -> int:
        """Get the shuffled index for a given index."""
        if self.shuffle:
            if self.shuffle_indices is None:
                self._initialize_shuffle()
            return self.shuffle_indices[idx]
        return idx
        
    def _load_next_chunk(self):
        """Load the next chunk of data from the current file."""
        if self.current_file_index >= len(self.h5_files):
            self.current_file_index = 0  # Reset to first file
            self.current_chunk_start = 0
        
        h5_file = self.h5_files[self.current_file_index]
        with h5py.File(os.path.join(self.data_dir, h5_file), 'r') as f:
            # Get the remaining samples in current file
            remaining_samples = f['branch_inputs'].shape[0] - self.current_chunk_start
            samples_to_load = min(self.chunk_size, remaining_samples)
            
            # Load the chunk
            self.current_chunk = {
                'branch_inputs': f['branch_inputs'][self.current_chunk_start:self.current_chunk_start + samples_to_load],
                'trunk_inputs': f['trunk_inputs'][self.current_chunk_start:self.current_chunk_start + samples_to_load],
                'outputs': f['outputs'][self.current_chunk_start:self.current_chunk_start + samples_to_load]
            }
            
            # Update indices
            self.current_chunk_start += samples_to_load
            if self.current_chunk_start >= f['branch_inputs'].shape[0]:
                self.current_file_index += 1
                self.current_chunk_start = 0
                
            # If we've loaded all files, reset to the beginning
            if self.current_file_index >= len(self.h5_files):
                self.current_file_index = 0
                self.current_chunk_start = 0
                self._load_next_chunk()  # Load the first chunk of the first file
    
    def _calculate_normalization_params(self):
        """Calculate normalization parameters for the dataset."""
        branch_sum = np.zeros(self.current_chunk['branch_inputs'].shape[1])
        branch_sq_sum = np.zeros(self.current_chunk['branch_inputs'].shape[1])
        trunk_sum = np.zeros(self.current_chunk['trunk_inputs'].shape[1])
        trunk_sq_sum = np.zeros(self.current_chunk['trunk_inputs'].shape[1])
        output_sum = np.zeros(self.current_chunk['outputs'].shape[1])
        output_sq_sum = np.zeros(self.current_chunk['outputs'].shape[1])
        n_samples = 0

        # Process each chunk
        for i in range(0, self.current_chunk['branch_inputs'].shape[0], self.chunk_size):
            end_idx = min(i + self.chunk_size, self.current_chunk['branch_inputs'].shape[0])
            indices = np.arange(i, end_idx)  # Create indices in ascending order
            
            with h5py.File(os.path.join(self.data_dir, self.h5_files[self.current_file_index]), 'r') as f:
                # Read data for this chunk
                branch_data = f['branch_inputs'][indices]
                trunk_data = f['trunk_inputs'][indices]
                output_data = f['outputs'][indices]
            
            # Update sums
            branch_sum += np.sum(branch_data, axis=0)
            branch_sq_sum += np.sum(branch_data ** 2, axis=0)
            trunk_sum += np.sum(trunk_data, axis=0)
            trunk_sq_sum += np.sum(trunk_data ** 2, axis=0)
            output_sum += np.sum(output_data, axis=0)
            output_sq_sum += np.sum(output_data ** 2, axis=0)
            n_samples += len(indices)

        # Calculate means and standard deviations
        self.branch_mean = branch_sum / n_samples
        self.branch_std = np.sqrt(branch_sq_sum / n_samples - self.branch_mean ** 2)
        self.trunk_mean = trunk_sum / n_samples
        self.trunk_std = np.sqrt(trunk_sq_sum / n_samples - self.trunk_mean ** 2)
        self.output_mean = output_sum / n_samples
        self.output_std = np.sqrt(output_sq_sum / n_samples - self.output_mean ** 2)

        # Handle zero standard deviations
        self.branch_std[self.branch_std == 0] = 1.0
        self.trunk_std[self.trunk_std == 0] = 1.0
        self.output_std[self.output_std == 0] = 1.0
    
    def __len__(self) -> int:
        return self.total_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (branch_input, trunk_input, output)
        """
        # Get the shuffled index if shuffling is enabled
        idx = self._get_shuffled_index(idx)
        
        # Wrap the index around if it's beyond the total number of samples
        idx = idx % self.total_samples
        
        # Calculate which file and position this index corresponds to
        current_pos = 0
        file_idx = 0
        for i, h5_file in enumerate(self.h5_files):
            with h5py.File(os.path.join(self.data_dir, h5_file), 'r') as f:
                file_size = f['branch_inputs'].shape[0]
                if idx < current_pos + file_size:
                    file_idx = i
                    local_idx = idx - current_pos
                    break
                current_pos += file_size
        
        # Load the appropriate chunk if needed
        if (self.current_file_index != file_idx or 
            local_idx < self.current_chunk_start or 
            local_idx >= self.current_chunk_start + len(self.current_chunk['branch_inputs'])):
            self.current_file_index = file_idx
            self.current_chunk_start = (local_idx // self.chunk_size) * self.chunk_size
            self._load_next_chunk()
        
        # Get the sample from the current chunk
        local_idx = idx - self.current_chunk_start
        if local_idx >= len(self.current_chunk['branch_inputs']):
            # If we're beyond the current chunk, load the next one
            self.current_chunk_start = (local_idx // self.chunk_size) * self.chunk_size
            self._load_next_chunk()
            local_idx = idx - self.current_chunk_start
        
        branch_input = self.current_chunk['branch_inputs'][local_idx]
        trunk_input = self.current_chunk['trunk_inputs'][local_idx]
        output = self.current_chunk['outputs'][local_idx]
        
        # Convert to tensors
        branch_input = torch.from_numpy(branch_input).float()
        trunk_input = torch.from_numpy(trunk_input).float()
        output = torch.from_numpy(output).float()
        
        # Normalize if required
        if self.normalize:
            branch_input = (branch_input - torch.from_numpy(self.branch_mean).float()) / torch.from_numpy(self.branch_std).float()
            trunk_input = (trunk_input - torch.from_numpy(self.trunk_mean).float()) / torch.from_numpy(self.trunk_std).float()
            output = (output - torch.from_numpy(self.output_mean).float()) / torch.from_numpy(self.output_std).float()
        
        # Apply transform if provided
        if self.transform:
            branch_input, trunk_input, output = self.transform(branch_input, trunk_input, output)
        
        return branch_input, trunk_input, output 