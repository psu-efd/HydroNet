"""
Data loading utilities for physics-informed neural networks (PINNs).
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os

class PINNDataset:
    """Dataset for physics-informed neural networks using full batch training."""
    
    def __init__(self, config, model):
        """
        Initialize the PINN dataset. The dataset is initialized based on the model, which has to be created before the dataset is initialized.
        
        Args:
            config (dict): Configuration dictionary.
            model (PINNModel): PINN model.
        """

        self.config = config

        # Get domain from config
        try:
            self.domain = {
                'x_min': self.config.get('sampling.domain.x_min'),
                'x_max': self.config.get('sampling.domain.x_max'),
                'y_min': self.config.get('sampling.domain.y_min'),
                'y_max': self.config.get('sampling.domain.y_max'),
                't_min': self.config.get('sampling.domain.t_min'),
                't_max': self.config.get('sampling.domain.t_max')
            }
            if any(v is None for v in self.domain.values()):
                raise ValueError("All domain parameters must be specified in config file")
        except KeyError:
            raise ValueError("Domain parameters must be specified in config file")

        # Get model config
        self.bPDE_loss, self.bInitial_loss, self.bBoundary_loss, self.bData_loss = model.get_loss_flags()
        self.bSteady = not self.bInitial_loss

        # Get device from config
        self.device = model.get_device()

        # Load points (and optionally data) from files
        self._load_points_from_files()

    def _load_points_from_files(self):
        """
        Load points and associated information from files.
        
        Expected file format:
        - All .npy files contain numpy arrays
        - Coordinates are in columns: x, y, t for unsteady problems and x, y for steady problems
        - Boundary info contains identifiers, normal vectors, and represented lengths
        """
        
        data_dir = self.config.get('data.points_dir')
        
        # Load PDE points if bPDE_loss is true
        if self.bPDE_loss:
            pde_file = os.path.join(data_dir, 'pde_points.npy')
            if os.path.exists(pde_file):
                self.interior_points = np.load(pde_file)
            else:
                raise FileNotFoundError(f"PDE points file not found: {pde_file}")
        else:
            self.interior_points = None

        # Load boundary points and info if bBoundary_loss is true
        if self.bBoundary_loss:
            boundary_file = os.path.join(data_dir, 'boundary_points.npy')
            boundary_info_file = os.path.join(data_dir, 'boundary_info.npy')
            if os.path.exists(boundary_file) and os.path.exists(boundary_info_file):
                self.boundary_points = np.load(boundary_file)
                boundary_info = np.load(boundary_info_file)
                self.boundary_identifiers = boundary_info[:, 0].astype(np.int32)  # First column: identifiers should be converted to int32
                self.boundary_normals = boundary_info[:, 1:3]    # normal vectors
                self.boundary_lengths = boundary_info[:, 3]      # represented lengths
            else:
                raise FileNotFoundError("Boundary points or info file not found")
        else:
            self.boundary_points = None
            self.boundary_identifiers = None
            self.boundary_normals = None
            self.boundary_lengths = None

        # Load initial points if bInitial_loss is true
        if self.bInitial_loss:
            initial_file = os.path.join(data_dir, 'initial_points.npy')             #initial points: x, y
            initial_values_file = os.path.join(data_dir, 'initial_values.npy')     #initial values: h, u, v
            if os.path.exists(initial_file) and os.path.exists(initial_values_file):
                self.initial_points = np.load(initial_file)
                self.initial_values = np.load(initial_values_file)
            else:
                raise FileNotFoundError(f"Initial points file not found: {initial_file} or {initial_values_file}")
        else:
            self.initial_points = None
            self.initial_values = None

        # Load data points and values if bData_loss is true 
        if self.bData_loss:
            data_points_file = os.path.join(data_dir, 'data_points.npy')      #data points: x, y, t for unsteady problems and x, y for steady problems
            data_values_file = os.path.join(data_dir, 'data_values.npy')      #data values: h, u, v
            if os.path.exists(data_points_file) and os.path.exists(data_values_file):
                self.data_points = np.load(data_points_file)
                self.data_values = np.load(data_values_file)                
            else:
                raise FileNotFoundError(f"Data points or values file not found: {data_points_file} or {data_values_file}")
        else:
            self.data_points = None
            self.data_values = None

        # Convert to PyTorch tensors and move to device
        if self.bPDE_loss and self.interior_points is not None:
            self.interior_points = torch.tensor(self.interior_points, dtype=torch.float32, device=self.device)
            self.interior_points.requires_grad_(True)

        if self.bBoundary_loss and self.boundary_points is not None:
            self.boundary_points = torch.tensor(self.boundary_points, dtype=torch.float32, device=self.device)
            self.boundary_points.requires_grad_(True)
            self.boundary_identifiers = torch.tensor(self.boundary_identifiers, dtype=torch.int32, device=self.device)
            self.boundary_normals = torch.tensor(self.boundary_normals, dtype=torch.float32, device=self.device)
            self.boundary_lengths = torch.tensor(self.boundary_lengths, dtype=torch.float32, device=self.device)

            #print the first 5 boundary points
            #print(f"First 5 boundary points: {self.boundary_points}")
            #print(f"First 5 boundary identifiers: {self.boundary_identifiers}")
            #print(f"First 5 boundary normals: {self.boundary_normals}")
            #print(f"First 5 boundary lengths: {self.boundary_lengths}")
        
        if self.bInitial_loss and self.initial_points is not None:
            self.initial_points = torch.tensor(self.initial_points, dtype=torch.float32, device=self.device)
            self.initial_values = torch.tensor(self.initial_values, dtype=torch.float32, device=self.device)
                
        if self.bData_loss and self.data_points is not None:
            self.data_points = torch.tensor(self.data_points, dtype=torch.float32, device=self.device)
            self.data_values = torch.tensor(self.data_values, dtype=torch.float32, device=self.device)

    def __len__(self):
        """Return the total number of collocation points."""
        total = 0
        
        if self.bPDE_loss and self.interior_points is not None:
            total += len(self.interior_points)
            
        if self.bBoundary_loss and self.boundary_points is not None:
            total += len(self.boundary_points)
            
        if self.bInitial_loss and self.initial_points is not None:
            total += len(self.initial_points)

        if self.bData_loss and self.data_points is not None:
            total += len(self.data_points)
            
        return total
        
    def __getitem__(self, idx):
        """
        Get a collocation point.
        
        Note: For physics-informed neural networks, we typically don't use
        the __getitem__ method as we pass the entire batch of points to the model.
        This implementation is just to comply with the Dataset interface.
        
        Args:
            idx (int): Index of the point.
            
        Returns:
            tensor: x, y, t coordinates
        """

        #report an error of not implemented
        raise NotImplementedError("__getitem__ method is not implemented for PINN dataset")
            
    def get_collocation_points(self):
        """
        Get all collocation points.
        
        Returns:
            tuple: (interior_points, boundary_points, initial_points, data_points)
                 where points are included based on their respective loss flags
        """
        points = []
        
        if self.bPDE_loss and self.interior_points is not None:
            points.append(self.interior_points)
            
        if self.bBoundary_loss and self.boundary_points is not None:
            points.append(self.boundary_points)
            
        if self.bInitial_loss and self.initial_points is not None:
            points.append(self.initial_points)
            
        if self.bData_loss and self.data_points is not None:
            points.append(self.data_points)
            
        return tuple(points)
    
    def get_pde_points(self):
        """Get all points for enforcing PDE residuals."""
        if self.bPDE_loss:
            return self.interior_points
        else:
            print("No PDE points for this problem")
            return None

    def get_initial_points(self):
        """Get all points for enforcing initial conditions."""
        if self.bInitial_loss:
            print("No initial points for steady case")
            return None 
        else:
            return self.initial_points, self.initial_values

    def get_boundary_points(self):
        """Get all points for enforcing boundary conditions."""
        return self.boundary_points, self.boundary_identifiers, self.boundary_normals, self.boundary_lengths

    def get_data_points(self):
        """Get all points for data points."""
        if self.bData_loss:
            return self.data_points, self.data_values
        else:
            print("No data points for this problem")
            return None



def get_pinn_dataloader(dataset, batch_size, shuffle, num_workers):
    """
    Create a DataLoader for a PINN dataset.
    
    Args:
        dataset (Dataset): The dataset to load.
        batch_size (int): Batch size.
        shuffle (bool): Whether to shuffle the data.
        num_workers (int): Number of worker processes.
        
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