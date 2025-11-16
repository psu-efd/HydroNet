"""
Data loading utilities for physics-informed neural networks (PINNs). 

PINNDataset is a subclass of torch.utils.data.Dataset: this ensures compatibility with PyTorch's data loading ecosystem
and allows you to use the dataset with PyTorch's DataLoader
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os

class PINNDataset(Dataset):
    """Dataset for physics-informed neural networks using full batch training."""
    
    def __init__(self, config, model):
        """
        Initialize the PINN dataset. The dataset is initialized based on the model, which has to be created before the dataset is initialized.
        
        Args:
            config (dict): Configuration dictionary.
            model (PINNModel): PINN model.
        """

        self.config = config       

        # Get model config
        self.bPDE_loss, self.bInitial_loss, self.bBoundary_loss, self.bData_loss = model.get_loss_flags()
        self.bSteady = model.get_bSteady()
        self.bNormalize = model.get_bNormalize()
        self.normalization_method = model.get_normalization_method()

        # Get device from config
        self.device = model.get_device()

        # Load points (and optionally data) from files
        self._load_points_from_files()

    def _load_points_from_files(self):
        """
        Load points and associated information from files. Normalize the data if bNormalize is true.
        
        Expected files and format:
        - All .npy files contain numpy arrays:
            - pde_points.npy: x, y, t for unsteady problems and x, y for steady problems
            - pde_data.npy: zb, Sx, Sy, ManningN
            - boundary_points.npy: x, y
            - boundary_info.npy: bc_ID, normal vectors, represented lengths, ManningN
            - all_mesh_points_stats.npy: x_min, x_max, x_std, y_min, y_max, y_std, t_min, t_max, t_std, zb_min, zb_max, zb_std, Sx_min, Sx_max, Sx_std, Sy_min, Sy_max, Sy_std, ManningN_min, ManningN_max, ManningN_std
            - data_points.npy: x, y, t for unsteady problems and x, y for steady problems
            - data_values.npy: h, u, v
            - data_flags.npy: h_flag, u_flag, v_flag
            - all_data_points_stats.npy: x_min, x_max, x_std, y_min, y_max, y_std, t_min, t_max, t_std, h_min, h_max, h_std, u_min, u_max, u_std, v_min, v_max, v_std, Umag_min, Umag_max, Umag_std
         
        - Data flags indicate which variables (h, u, v) are available for each point
        """
        
        data_dir = self.config.get('data.points_dir')
        
        # Load PDE points if bPDE_loss is true
        if self.bPDE_loss:
            pde_file = os.path.join(data_dir, 'pde_points.npy')
            pde_data_file = os.path.join(data_dir, 'pde_data.npy')
            if os.path.exists(pde_file) and os.path.exists(pde_data_file):
                self.interior_points = np.load(pde_file)
                self.pde_data = np.load(pde_data_file)

                #check the shape of the pde points: for steady problems, the shape should be (num_points, 2) and for unsteady problems, the shape should be (num_points, 3)
                if self.bSteady:
                    if self.interior_points.shape[1] != 2:
                        raise ValueError("For steady problems, the number of columns in pde_points should be 2")
                else:
                    if self.interior_points.shape[1] != 3:
                        raise ValueError("For unsteady problems, the number of columns in pde_points should be 3")
                    
                #check the consistency of the pde points and pde data
                if len(self.interior_points) != len(self.pde_data):
                    raise ValueError("The number of points in pde_points and pde_data must be the same")
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
                self.boundary_z = boundary_info[:, 1]
                self.boundary_normals = boundary_info[:, 2:4]    # normal vectors
                self.boundary_lengths = boundary_info[:, 4]      # represented lengths
                self.boundary_ManningN = boundary_info[:, 5]    # ManningN
            else:
                raise FileNotFoundError("Boundary points or info file not found")
        else:
            self.boundary_points = None
            self.boundary_identifiers = None
            self.boundary_z = None
            self.boundary_normals = None
            self.boundary_lengths = None
            self.boundary_ManningN = None

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
            data_flags_file = os.path.join(data_dir, 'data_flags.npy')        #data flags: h_flag, u_flag, v_flag

            if os.path.exists(data_points_file) and os.path.exists(data_values_file) and os.path.exists(data_flags_file):
                self.data_points = np.load(data_points_file)
                self.data_values = np.load(data_values_file)
                self.data_flags = np.load(data_flags_file)

                #check the shape of the data points: for steady problems, the shape should be (num_points, 2) and for unsteady problems, the shape should be (num_points, 3)
                if self.bSteady:
                    if self.data_points.shape[1] != 2:
                        raise ValueError("For steady problems, the number of columns in data_points should be 2")   
                else:
                    if self.data_points.shape[1] != 3:
                        raise ValueError("For unsteady problems, the number of columns in data_points should be 3")
                    
                #check the length of data points and data values and data flags
                if len(self.data_points) != len(self.data_values) or len(self.data_points) != len(self.data_flags):
                    raise ValueError("The length of data points, data values, and data flags must be the same") 
            else:
                raise FileNotFoundError(f"Data points, values, or flags file not found: {data_points_file}, {data_values_file}, or {data_flags_file}")
        else:
            self.data_points = None
            self.data_values = None
            self.data_flags = None

        # read the statistics of all the mesh points (pde_points and boundary_points)
        all_points_stats_file = os.path.join(data_dir, 'all_mesh_points_stats.npy')
        if os.path.exists(all_points_stats_file):
            self.all_points_stats = np.load(all_points_stats_file)

            #unpack them into variables
            self.mesh_x_min, self.mesh_x_max, self.mesh_x_mean, self.mesh_x_std, self.mesh_y_min, self.mesh_y_max, self.mesh_y_mean, self.mesh_y_std, self.mesh_t_min, self.mesh_t_max, self.mesh_t_mean, self.mesh_t_std, self.mesh_zb_min, self.mesh_zb_max, self.mesh_zb_mean, self.mesh_zb_std, self.mesh_Sx_min, self.mesh_Sx_max, self.mesh_Sx_mean, self.mesh_Sx_std, self.mesh_Sy_min, self.mesh_Sy_max, self.mesh_Sy_mean, self.mesh_Sy_std, self.ManningN_min, self.ManningN_max, self.ManningN_mean, self.ManningN_std = self.all_points_stats
            
        else:
            raise FileNotFoundError(f"All points stats file not found: {all_points_stats_file}")

        # read the statistics of all the data points
        all_data_points_stats_file = os.path.join(data_dir, 'all_data_points_stats.npy')
        if os.path.exists(all_data_points_stats_file):
            self.all_data_points_stats = np.load(all_data_points_stats_file)

            #unpack them into variables
            self.data_x_min, self.data_x_max, self.data_x_mean, self.data_x_std, self.data_y_min, self.data_y_max, self.data_y_mean, self.data_y_std, self.data_t_min, self.data_t_max, self.data_t_mean, self.data_t_std, self.data_h_min, self.data_h_max, self.data_h_mean, self.data_h_std, self.data_u_min, self.data_u_max, self.data_u_mean, self.data_u_std, self.data_v_min, self.data_v_max, self.data_v_mean, self.data_v_std, self.data_Umag_min, self.data_Umag_max, self.data_Umag_mean, self.data_Umag_std = self.all_data_points_stats

        else:
            raise FileNotFoundError(f"All data points stats file not found: {all_data_points_stats_file}")

        #if bNormalize is true, normalize the data
        if self.bNormalize:
            self.normalize_data()

        # Convert to PyTorch tensors and move to device
        if self.bPDE_loss and self.interior_points is not None:
            self.interior_points = torch.tensor(self.interior_points, dtype=torch.float32, device=self.device)
            self.interior_points.requires_grad_(True)

            self.pde_data = torch.tensor(self.pde_data, dtype=torch.float32, device=self.device)

        if self.bBoundary_loss and self.boundary_points is not None:
            self.boundary_points = torch.tensor(self.boundary_points, dtype=torch.float32, device=self.device)
            self.boundary_points.requires_grad_(True)
            self.boundary_identifiers = torch.tensor(self.boundary_identifiers, dtype=torch.int32, device=self.device)
            self.boundary_z = torch.tensor(self.boundary_z, dtype=torch.float32, device=self.device)
            self.boundary_normals = torch.tensor(self.boundary_normals, dtype=torch.float32, device=self.device)
            self.boundary_lengths = torch.tensor(self.boundary_lengths, dtype=torch.float32, device=self.device)
            self.boundary_ManningN = torch.tensor(self.boundary_ManningN, dtype=torch.float32, device=self.device)

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
            self.data_flags = torch.tensor(self.data_flags, dtype=torch.float32, device=self.device)

        # move the statistics to device
        self.mesh_x_min = torch.tensor(self.mesh_x_min, dtype=torch.float32, device=self.device)
        self.mesh_x_max = torch.tensor(self.mesh_x_max, dtype=torch.float32, device=self.device)
        self.mesh_x_mean = torch.tensor(self.mesh_x_mean, dtype=torch.float32, device=self.device)
        self.mesh_x_std = torch.tensor(self.mesh_x_std, dtype=torch.float32, device=self.device)
        self.mesh_y_min = torch.tensor(self.mesh_y_min, dtype=torch.float32, device=self.device)
        self.mesh_y_max = torch.tensor(self.mesh_y_max, dtype=torch.float32, device=self.device)
        self.mesh_y_mean = torch.tensor(self.mesh_y_mean, dtype=torch.float32, device=self.device)
        self.mesh_y_std = torch.tensor(self.mesh_y_std, dtype=torch.float32, device=self.device)
        self.mesh_t_min = torch.tensor(self.mesh_t_min, dtype=torch.float32, device=self.device)
        self.mesh_t_max = torch.tensor(self.mesh_t_max, dtype=torch.float32, device=self.device)
        self.mesh_t_mean = torch.tensor(self.mesh_t_mean, dtype=torch.float32, device=self.device)
        self.mesh_t_std = torch.tensor(self.mesh_t_std, dtype=torch.float32, device=self.device)    
        self.mesh_zb_min = torch.tensor(self.mesh_zb_min, dtype=torch.float32, device=self.device)
        self.mesh_zb_max = torch.tensor(self.mesh_zb_max, dtype=torch.float32, device=self.device)
        self.mesh_zb_mean = torch.tensor(self.mesh_zb_mean, dtype=torch.float32, device=self.device)
        self.mesh_zb_std = torch.tensor(self.mesh_zb_std, dtype=torch.float32, device=self.device)
        self.mesh_Sx_min = torch.tensor(self.mesh_Sx_min, dtype=torch.float32, device=self.device)
        self.mesh_Sx_max = torch.tensor(self.mesh_Sx_max, dtype=torch.float32, device=self.device)
        self.mesh_Sx_mean = torch.tensor(self.mesh_Sx_mean, dtype=torch.float32, device=self.device)
        self.mesh_Sx_std = torch.tensor(self.mesh_Sx_std, dtype=torch.float32, device=self.device)
        self.mesh_Sy_min = torch.tensor(self.mesh_Sy_min, dtype=torch.float32, device=self.device)
        self.mesh_Sy_max = torch.tensor(self.mesh_Sy_max, dtype=torch.float32, device=self.device)
        self.mesh_Sy_mean = torch.tensor(self.mesh_Sy_mean, dtype=torch.float32, device=self.device)
        self.mesh_Sy_std = torch.tensor(self.mesh_Sy_std, dtype=torch.float32, device=self.device)
        self.ManningN_min = torch.tensor(self.ManningN_min, dtype=torch.float32, device=self.device)
        self.ManningN_max = torch.tensor(self.ManningN_max, dtype=torch.float32, device=self.device)
        self.ManningN_mean = torch.tensor(self.ManningN_mean, dtype=torch.float32, device=self.device)
        self.ManningN_std = torch.tensor(self.ManningN_std, dtype=torch.float32, device=self.device)

        #pack all mesh stats into a dictionary
        self.mesh_stats = {
            'x_min': self.mesh_x_min,
            'x_max': self.mesh_x_max,
            'x_mean': self.mesh_x_mean,
            'x_std': self.mesh_x_std,
            'y_min': self.mesh_y_min,
            'y_max': self.mesh_y_max,
            'y_mean': self.mesh_y_mean,
            'y_std': self.mesh_y_std,
            't_min': self.mesh_t_min,
            't_max': self.mesh_t_max,
            't_mean': self.mesh_t_mean, 
            't_std': self.mesh_t_std,
            'zb_min': self.mesh_zb_min,
            'zb_max': self.mesh_zb_max,
            'zb_mean': self.mesh_zb_mean,
            'zb_std': self.mesh_zb_std, 
            'Sx_min': self.mesh_Sx_min,
            'Sx_max': self.mesh_Sx_max,
            'Sx_mean': self.mesh_Sx_mean,
            'Sx_std': self.mesh_Sx_std,
            'Sy_min': self.mesh_Sy_min,
            'Sy_max': self.mesh_Sy_max,
            'Sy_mean': self.mesh_Sy_mean,
            'Sy_std': self.mesh_Sy_std, 
            'ManningN_min': self.ManningN_min,
            'ManningN_max': self.ManningN_max,
            'ManningN_mean': self.ManningN_mean,
            'ManningN_std': self.ManningN_std
        }

        # move the data points statistics to device
        self.data_x_min = torch.tensor(self.data_x_min, dtype=torch.float32, device=self.device)
        self.data_x_max = torch.tensor(self.data_x_max, dtype=torch.float32, device=self.device)
        self.data_x_mean = torch.tensor(self.data_x_mean, dtype=torch.float32, device=self.device)
        self.data_x_std = torch.tensor(self.data_x_std, dtype=torch.float32, device=self.device)
        self.data_y_min = torch.tensor(self.data_y_min, dtype=torch.float32, device=self.device)
        self.data_y_max = torch.tensor(self.data_y_max, dtype=torch.float32, device=self.device)
        self.data_y_mean = torch.tensor(self.data_y_mean, dtype=torch.float32, device=self.device)
        self.data_y_std = torch.tensor(self.data_y_std, dtype=torch.float32, device=self.device)
        self.data_t_min = torch.tensor(self.data_t_min, dtype=torch.float32, device=self.device)
        self.data_t_max = torch.tensor(self.data_t_max, dtype=torch.float32, device=self.device)
        self.data_t_mean = torch.tensor(self.data_t_mean, dtype=torch.float32, device=self.device)
        self.data_t_std = torch.tensor(self.data_t_std, dtype=torch.float32, device=self.device)
        self.data_h_min = torch.tensor(self.data_h_min, dtype=torch.float32, device=self.device)
        self.data_h_max = torch.tensor(self.data_h_max, dtype=torch.float32, device=self.device)    
        self.data_h_mean = torch.tensor(self.data_h_mean, dtype=torch.float32, device=self.device)
        self.data_h_std = torch.tensor(self.data_h_std, dtype=torch.float32, device=self.device)
        self.data_u_min = torch.tensor(self.data_u_min, dtype=torch.float32, device=self.device)
        self.data_u_max = torch.tensor(self.data_u_max, dtype=torch.float32, device=self.device)
        self.data_u_mean = torch.tensor(self.data_u_mean, dtype=torch.float32, device=self.device)
        self.data_u_std = torch.tensor(self.data_u_std, dtype=torch.float32, device=self.device)
        self.data_v_min = torch.tensor(self.data_v_min, dtype=torch.float32, device=self.device)
        self.data_v_max = torch.tensor(self.data_v_max, dtype=torch.float32, device=self.device)    
        self.data_v_mean = torch.tensor(self.data_v_mean, dtype=torch.float32, device=self.device)
        self.data_v_std = torch.tensor(self.data_v_std, dtype=torch.float32, device=self.device)
        self.data_Umag_min = torch.tensor(self.data_Umag_min, dtype=torch.float32, device=self.device)
        self.data_Umag_max = torch.tensor(self.data_Umag_max, dtype=torch.float32, device=self.device)
        self.data_Umag_mean = torch.tensor(self.data_Umag_mean, dtype=torch.float32, device=self.device)
        self.data_Umag_std = torch.tensor(self.data_Umag_std, dtype=torch.float32, device=self.device) 

        #pack all data points stats into a dictionary
        self.data_stats = {
            'x_min': self.data_x_min,
            'x_max': self.data_x_max,
            'x_mean': self.data_x_mean, 
            'x_std': self.data_x_std,
            'y_min': self.data_y_min,
            'y_max': self.data_y_max,
            'y_mean': self.data_y_mean,
            'y_std': self.data_y_std,
            't_min': self.data_t_min,
            't_max': self.data_t_max,
            't_mean': self.data_t_mean,
            't_std': self.data_t_std,
            'h_min': self.data_h_min,
            'h_max': self.data_h_max,
            'h_mean': self.data_h_mean,
            'h_std': self.data_h_std,
            'u_min': self.data_u_min,
            'u_max': self.data_u_max,
            'u_mean': self.data_u_mean,
            'u_std': self.data_u_std,
            'v_min': self.data_v_min,
            'v_max': self.data_v_max,
            'v_mean': self.data_v_mean,
            'v_std': self.data_v_std,
            'Umag_min': self.data_Umag_min,
            'Umag_max': self.data_Umag_max,
            'Umag_mean': self.data_Umag_mean,
            'Umag_std': self.data_Umag_std
        }
            
       
        

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
            

    def normalize_data(self):
        """Normalize the data based on the normalization method."""
        if self.normalization_method == "min-max":
            self.normalize_min_max()
        elif self.normalization_method == "z-score":
            self.normalize_z_score()


    def normalize_min_max(self):
        """Normalize the data using min-max normalization."""
        # Add small epsilon to avoid division by zero
        eps = 1e-8

        #normalize the interior points if we have them
        if self.interior_points is not None:
            # Normalize x coordinates (first column)
            self.interior_points[:, 0] = (self.interior_points[:, 0] - self.mesh_x_min) / (self.mesh_x_max - self.mesh_x_min + eps)
            # Normalize y coordinates (second column)
            self.interior_points[:, 1] = (self.interior_points[:, 1] - self.mesh_y_min) / (self.mesh_y_max - self.mesh_y_min + eps)
            # If unsteady, normalize t coordinates (third column)
            if self.interior_points.shape[1] == 3:
                self.interior_points[:, 2] = (self.interior_points[:, 2] - self.mesh_t_min) / (self.mesh_t_max - self.mesh_t_min + eps)

        #normalize the boundary points if we have them
        if self.boundary_points is not None:
            # Normalize x coordinates (first column)    
            self.boundary_points[:, 0] = (self.boundary_points[:, 0] - self.mesh_x_min) / (self.mesh_x_max - self.mesh_x_min + eps)
            # Normalize y coordinates (second column)
            self.boundary_points[:, 1] = (self.boundary_points[:, 1] - self.mesh_y_min) / (self.mesh_y_max - self.mesh_y_min + eps)

        #normalize the initial points if unsteady
        if self.initial_points is not None and self.initial_points.shape[1] == 3:
            self.initial_points[:, 2] = (self.initial_points[:, 2] - self.mesh_t_min) / (self.mesh_t_max - self.mesh_t_min + eps)
        
        #normalize the data points if we have them
        if self.data_points is not None:
            # Normalize x coordinates (first column)
            self.data_points[:, 0] = (self.data_points[:, 0] - self.data_x_min) / (self.data_x_max - self.data_x_min + eps)
            # Normalize y coordinates (second column)
            self.data_points[:, 1] = (self.data_points[:, 1] - self.data_y_min) / (self.data_y_max - self.data_y_min + eps)
            # If unsteady, normalize t coordinates (third column)
            if self.data_points.shape[1] == 3:
                self.data_points[:, 2] = (self.data_points[:, 2] - self.data_t_min) / (self.data_t_max - self.data_t_min + eps)

        #normalize the data values if we have them
        if self.data_values is not None:
            self.data_values[:, 0] = (self.data_values[:, 0] - self.data_h_min) / (self.data_h_max - self.data_h_min + eps)
            self.data_values[:, 1] = (self.data_values[:, 1] - self.data_u_min) / (self.data_u_max - self.data_u_min + eps)
            self.data_values[:, 2] = (self.data_values[:, 2] - self.data_v_min) / (self.data_v_max - self.data_v_min + eps) 

    def normalize_z_score(self):
        """Normalize the data using z-score normalization."""
        # Add small epsilon to avoid division by zero
        eps = 1e-8

        #normalize the interior points if we have them
        if self.interior_points is not None:
            # Normalize x coordinates (first column)
            self.interior_points[:, 0] = (self.interior_points[:, 0] - self.mesh_x_mean) / (self.mesh_x_std + eps)
            # Normalize y coordinates (second column)
            self.interior_points[:, 1] = (self.interior_points[:, 1] - self.mesh_y_mean) / (self.mesh_y_std + eps)
            # If unsteady, normalize t coordinates (third column)
            if self.interior_points.shape[1] == 3:
                self.interior_points[:, 2] = (self.interior_points[:, 2] - self.mesh_t_mean) / (self.mesh_t_std + eps)

        #normalize the boundary points if we have them
        if self.boundary_points is not None:
            # Normalize x coordinates (first column)    
            self.boundary_points[:, 0] = (self.boundary_points[:, 0] - self.mesh_x_mean) / (self.mesh_x_std + eps)
            # Normalize y coordinates (second column)
            self.boundary_points[:, 1] = (self.boundary_points[:, 1] - self.mesh_y_mean) / (self.mesh_y_std + eps)

        #normalize the initial points if unsteady
        if self.initial_points is not None and self.initial_points.shape[1] == 3:
            self.initial_points[:, 2] = (self.initial_points[:, 2] - self.mesh_t_mean) / (self.mesh_t_std + eps)
        
        #normalize the data points if we have them
        if self.data_points is not None:
            # Normalize x coordinates (first column)
            self.data_points[:, 0] = (self.data_points[:, 0] - self.data_x_mean) / (self.data_x_std + eps)
            # Normalize y coordinates (second column)
            self.data_points[:, 1] = (self.data_points[:, 1] - self.data_y_mean) / (self.data_y_std + eps)
            # If unsteady, normalize t coordinates (third column)
            if self.data_points.shape[1] == 3:
                self.data_points[:, 2] = (self.data_points[:, 2] - self.data_t_mean) / (self.data_t_std + eps)

        #normalize the data values if we have them
        if self.data_values is not None:
            self.data_values[:, 0] = (self.data_values[:, 0] - self.data_h_mean) / (self.data_h_std + eps)
            self.data_values[:, 1] = (self.data_values[:, 1] - self.data_u_mean) / (self.data_u_std + eps)
            self.data_values[:, 2] = (self.data_values[:, 2] - self.data_v_mean) / (self.data_v_std + eps)

    def print_stats(self):
        """Print the statistics of the data computed from the data in the class (not the stats read in from the file)."""

        print(f"Interior points: {self.interior_points}")
        print(f"Boundary points: {self.boundary_points}")
        print(f"Boundary identifiers: {self.boundary_identifiers}")
        print(f"Boundary z: {self.boundary_z}")
        print(f"Boundary normals: {self.boundary_normals}")
        print(f"Boundary lengths: {self.boundary_lengths}")
        print(f"Boundary Manning N: {self.boundary_ManningN}")
        print(f"Initial points: {self.initial_points}")
        print(f"Data points: {self.data_points}")
        print(f"Data values: {self.data_values}")
        print(f"Data flags: {self.data_flags}")

        print("Mesh stats:")
        print(self.mesh_stats)

        print("Data stats:")
        print(self.data_stats)

        #interior points
        if self.interior_points is not None:
            print("Interior points:")
            print(f"x_min: {self.interior_points[:, 0].min()}, x_max: {self.interior_points[:, 0].max()}, x_mean: {self.interior_points[:, 0].mean()}, x_std: {self.interior_points[:, 0].std()}")
            
            print(f"y_min: {self.interior_points[:, 1].min()}, y_max: {self.interior_points[:, 1].max()}, y_mean: {self.interior_points[:, 1].mean()}, y_std: {self.interior_points[:, 1].std()}")

            if self.interior_points.shape[1] == 3:
                print(f"t_min: {self.interior_points[:, 2].min()}, t_max: {self.interior_points[:, 2].max()}, t_mean: {self.interior_points[:, 2].mean()}, t_std: {self.interior_points[:, 2].std()}")

        #boundary points
        print("Boundary points:")
        if self.boundary_points is not None:
            print(f"x_min: {self.boundary_points[:, 0].min()}, x_max: {self.boundary_points[:, 0].max()}, x_mean: {self.boundary_points[:, 0].mean()}, x_std: {self.boundary_points[:, 0].std()}")
            print(f"y_min: {self.boundary_points[:, 1].min()}, y_max: {self.boundary_points[:, 1].max()}, y_mean: {self.boundary_points[:, 1].mean()}, y_std: {self.boundary_points[:, 1].std()}")  

        #initial points
        if self.initial_points is not None:
            print("Initial points:")
            print(f"x_min: {self.initial_points[:, 0].min()}, x_max: {self.initial_points[:, 0].max()}, x_mean: {self.initial_points[:, 0].mean()}, x_std: {self.initial_points[:, 0].std()}")
            print(f"y_min: {self.initial_points[:, 1].min()}, y_max: {self.initial_points[:, 1].max()}, y_mean: {self.initial_points[:, 1].mean()}, y_std: {self.initial_points[:, 1].std()}")
            if self.initial_points.shape[1] == 3:
                print(f"t_min: {self.initial_points[:, 2].min()}, t_max: {self.initial_points[:, 2].max()}, t_mean: {self.initial_points[:, 2].mean()}, t_std: {self.initial_points[:, 2].std()}")
        
        #data points
        if self.data_points is not None:
            print("Data points:")
            print(f"x_min: {self.data_points[:, 0].min()}, x_max: {self.data_points[:, 0].max()}, x_mean: {self.data_points[:, 0].mean()}, x_std: {self.data_points[:, 0].std()}")
            print(f"y_min: {self.data_points[:, 1].min()}, y_max: {self.data_points[:, 1].max()}, y_mean: {self.data_points[:, 1].mean()}, y_std: {self.data_points[:, 1].std()}")
            if self.data_points.shape[1] == 3:
                print(f"t_min: {self.data_points[:, 2].min()}, t_max: {self.data_points[:, 2].max()}, t_mean: {self.data_points[:, 2].mean()}, t_std: {self.data_points[:, 2].std()}")

        #data values
        if self.data_values is not None:
            print("Data values:")
            print(f"h_min: {self.data_values[:, 0].min()}, h_max: {self.data_values[:, 0].max()}, h_mean: {self.data_values[:, 0].mean()}, h_std: {self.data_values[:, 0].std()}")
            print(f"u_min: {self.data_values[:, 1].min()}, u_max: {self.data_values[:, 1].max()}, u_mean: {self.data_values[:, 1].mean()}, u_std: {self.data_values[:, 1].std()}")
            print(f"v_min: {self.data_values[:, 2].min()}, v_max: {self.data_values[:, 2].max()}, v_mean: {self.data_values[:, 2].mean()}, v_std: {self.data_values[:, 2].std()}")


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
        """Get all points and data for enforcing PDE residuals."""
        if self.bPDE_loss:
            return self.interior_points, self.pde_data
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
        return self.boundary_points, self.boundary_identifiers, self.boundary_z, self.boundary_normals, self.boundary_lengths, self.boundary_ManningN

    def get_data_points(self):
        """Get all points for data points."""
        if self.bData_loss:
            return self.data_points, self.data_values, self.data_flags
        else:
            print("No data points for this problem")
            return None
    
    def get_mesh_stats(self):
        """Get the statistics of the mesh."""
        return self.mesh_stats

    def get_data_stats(self):
        """Get the statistics of the data."""
        return self.data_stats


def get_pinn_dataloader(dataset, batch_size, shuffle, num_workers):
    """
    (Currently not used. Not sure if it's needed. For PINN, we pass the entire dataset to the model.)

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