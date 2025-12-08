"""
Physics-Informed Neural Network (PINN) model for 2D shallow water equations.
"""

import torch
import torch.nn as nn
import numpy as np
from ...utils.config import Config
from typing import Dict, Tuple, Optional, Union

from .loss_weight_scheduler import (
    LossWeightScheduler,
    ConstantWeightScheduler,
    ManualWeightScheduler,
    GradNormScheduler,
    SoftAdaptScheduler
)


class SWE_PINN(nn.Module):
    """
    Physics-Informed Neural Network for 2D Shallow Water Equations.
    
    This model solves the 2D shallow water equations using a physics-informed approach,
    enforcing the governing equations, boundary conditions, initial conditions (if unsteady), and data points (from simulation or measurement).

    The model supports both steady and unsteady problems (bSteady = True or False in the config file).

    Boundary conditions are specified in the config file and the boundary points are loaded from the data file. Currently, supported boundary conditions are:
    - "inlet-q": Inlet discharge. The summation of h*u_normal*length is the specified discharge.
    - "exit-h": Outlet water surface elevation. The specified water surface elevation is the specified value. Water depth is computed from the water surface elevation and bed elevation. For velocities, it is zero gradient along the normal direction (zero Neumann BC).
    - "wall": Wall boundary. The velocity (u,v) is set to zero. For the water depth h, it is zero gradient along the normal direction (zero Neumann BC).

    Data points are loaded from the data files (by class PINN_dataset). The data points are typically from:
    - physics-based simulation: Data points from physics-based simulation (e.g., SRH-2D, HEC-RAS, etc).
    - measurement: Data points from measurement (e.g., field measurements, laboratory experiments, etc).

    The model is trained by minimizing the total loss, which is the sum of the PDE loss, initial condition loss (if unsteady), boundary condition loss, and/or data loss.
    """

    def __init__(self, config):
        """
        Initialize the PINN model.
        
        Args:
            config (Config): Configuration object.
        """
        super().__init__()

        if not isinstance(config, Config):
            raise ValueError("config must be a Config object.")
        
        # Load configuration
        self.config = config
        
        #get device from config
        device_type = self.config.get_required_config("device.type")
        if device_type is not None:
            device_index = self.config.get_required_config("device.index")
            if device_type == "cuda" and torch.cuda.is_available():
                self.device = torch.device(f"cuda:{device_index}")
            else:
                self.device = torch.device("cpu")
        else:  #by default, the model will be on the CPU
            self.device = torch.device("cpu")
            
        # Get model parameters from config
        # The model name has to be SWE_PINN; otherwise error will be raised
        self.model_name = self.config.get_required_config('model.name')
        if self.model_name != 'SWE_PINN':
            raise ValueError("model.name must be 'SWE_PINN' for SWE_PINN model")

        # Get loss flags from config
        try:
            self.bPDE_loss = bool(self.config.get_required_config('model.loss_flags.bPDE_loss'))
            if self.bPDE_loss is None:
                raise ValueError("model.loss_flags.bPDE_loss must be specified in config file")
        except (TypeError, ValueError):
            raise ValueError("model.loss_flags.bPDE_loss must be a boolean value in config file")
            
        try:
            self.bBoundary_loss = bool(self.config.get_required_config('model.loss_flags.bBoundary_loss'))
            if self.bBoundary_loss is None:
                raise ValueError("model.loss_flags.bBoundary_loss must be specified in config file")
        except (TypeError, ValueError):
            raise ValueError("model.loss_flags.bBoundary_loss must be a boolean value in config file")
            
        try:
            self.bSteady = bool(self.config.get_required_config('physics.bSteady'))
            #print(f"bSteady: {self.bSteady}")
            self.bInitial_loss = not self.bSteady    #for unsteady problems, bSteady is false and the initial conditions mismatch is included in the loss function
            #print(f"bInitial_loss: {self.bInitial_loss}")
            if self.bSteady is None:
                raise ValueError("physics.bSteady must be specified in config file")
        except (TypeError, ValueError):
            raise ValueError("physics.bSteady must be a boolean value in config file")
            
        try:
            self.bData_loss = bool(self.config.get_required_config('model.loss_flags.bData_loss'))
            if self.bData_loss is None:
                raise ValueError("model.loss_flags.bData_loss must be specified in config file")
        except (TypeError, ValueError):
            raise ValueError("model.loss_flags.bData_loss must be a boolean value in config file")

        hidden_layers = self.config.get_required_config('model.hidden_layers')
        activation = self.config.get_required_config('model.activation')
        self.output_dim = self.config.get_required_config('model.output_dim')
        self.initialization = self.config.get_required_config('model.initialization')
       
        # Initialize physics buffers (g, length_scale, velocity_scale)
        self._init_physics_buffers()
        
        # Initialize loss weight buffers
        self._init_loss_weight_buffers()

        # Read in the boundary conditions
        try:
            self.BCs_from_config = self.config.get_required_config('boundary_conditions')
            if self.BCs_from_config is None:
                raise ValueError("boundary_conditions must be specified in config file")
            if not isinstance(self.BCs_from_config, dict):
                raise ValueError("boundary_conditions must be a dictionary in config file")
            if not self.BCs_from_config:
                raise ValueError("boundary_conditions dictionary cannot be empty")
        except (TypeError, ValueError) as e:
            raise ValueError(f"Error loading boundary conditions: {str(e)}")
        
        #print(f"BCs_from_config: {self.BCs_from_config}")
        #exit()
        
        # Input dimension: (x, y, t) for transient problems, (x, y) for steady problems
        self.input_dim = 3 if not self.bSteady else 2
        
        # Build the network
        layers = []
        layer_dims = [self.input_dim] + hidden_layers + [self.output_dim]
        
        for i in range(len(layer_dims) - 1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
            
            # Add activation to all but the last layer
            if i < len(layer_dims) - 2:
                if activation == 'tanh':
                    layers.append(nn.Tanh())
                elif activation == 'relu':
                    layers.append(nn.ReLU())
                elif activation == 'sigmoid':
                    layers.append(nn.Sigmoid())
                elif activation == 'leaky_relu':
                    layers.append(nn.LeakyReLU(0.2))
                    
        self.net = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()

        # Read rating curve file if any boundary condition is a rating curve
        for bc_id, bc_config in self.BCs_from_config.items():
            if bc_config['type'] == "exit-h" and bc_config['exit_h_option'] == "rating-curve":
                rating_curve_file = bc_config['rating_curve_file']
                # Read rating curve file
                discharges = []
                wses = []
                with open(rating_curve_file, 'r') as f:
                    # Skip header line
                    next(f)
                    for line in f:
                        # Skip comment lines
                        if line.strip().startswith('//'):
                            continue
                        # Parse discharge and WSE values
                        try:
                            q, wse = map(float, line.strip().split())
                            discharges.append(q)
                            wses.append(wse)
                        except ValueError:
                            continue
                
                # Add rating curve data to bc_config
                bc_config['discharges'] = torch.tensor(discharges, device=self.device)
                bc_config['wses'] = torch.tensor(wses, device=self.device)
        
        # Move the model to the device
        self.to(self.device)
    
    def _init_physics_buffers(self):
        """
        Initialize the physics buffers (g, length_scale, velocity_scale).
        These are stored as buffers so they move with the model to the correct device
        and are saved/loaded with checkpoints.
        """
        g_value = float(self.config.get_required_config("physics.gravity"))
        length_scale_value = self.config.get_required_config("physics.scales.length")
        velocity_scale_value = self.config.get_required_config("physics.scales.velocity")
        
        # Create buffers without specifying device - they will be moved by self.to(device) later
        self.register_buffer("g", torch.tensor(g_value, dtype=torch.float32))
        self.register_buffer("length_scale", torch.tensor(float(length_scale_value), dtype=torch.float32))
        self.register_buffer("velocity_scale", torch.tensor(float(velocity_scale_value), dtype=torch.float32))
    
    def _init_loss_weight_buffers(self):
        """
        Initialize the loss weight buffers.
        These are stored as buffers so they move with the model to the correct device
        and are saved/loaded with checkpoints.
        """
        def to_tensor(val):
            return torch.tensor(float(val), dtype=torch.float32)
        
        if self.bSteady:
            self.register_buffer(
                "loss_weight_pde_loss",
                to_tensor(self.config.get_required_config("training.loss_weights.pinn.pde_loss")),
            )
            self.register_buffer(
                "loss_weight_boundary_loss",
                to_tensor(self.config.get_required_config("training.loss_weights.pinn.boundary_loss")),
            )
            self.register_buffer(
                "loss_weight_data_loss",
                to_tensor(self.config.get_required_config("training.loss_weights.pinn.data_loss")),
            )
        else:
            self.register_buffer(
                "loss_weight_pde_loss",
                to_tensor(self.config.get_required_config("training.loss_weights.pinn.pde_loss")),
            )
            self.register_buffer(
                "loss_weight_initial_loss",
                to_tensor(self.config.get_required_config("training.loss_weights.pinn.initial_loss")),
            )
            self.register_buffer(
                "loss_weight_boundary_loss",
                to_tensor(self.config.get_required_config("training.loss_weights.pinn.boundary_loss")),
            )
            self.register_buffer(
                "loss_weight_data_loss",
                to_tensor(self.config.get_required_config("training.loss_weights.pinn.data_loss")),
            )
        # Weights for individual PDE component losses (continuity, momentum_x, momentum_y)
        self.register_buffer(
            "loss_weight_pde_continuity",
            to_tensor(self.config.get_required_config("training.loss_weights.pde.continuity")),
        )
        self.register_buffer(
            "loss_weight_pde_momentum_x",
            to_tensor(self.config.get_required_config("training.loss_weights.pde.momentum_x")),
        )
        self.register_buffer(
            "loss_weight_pde_momentum_y",
            to_tensor(self.config.get_required_config("training.loss_weights.pde.momentum_y")),
        )        

    @property
    def loss_weights(self):
        """
        Property that returns loss weights as a dictionary for convenient access.
        The actual weights are stored as registered buffers.
        """
        base_dict = {
                'pde_loss': self.loss_weight_pde_loss,
                'boundary_loss': self.loss_weight_boundary_loss,
                'data_loss': self.loss_weight_data_loss,
                "pde_continuity": self.loss_weight_pde_continuity,
                "pde_momentum_x": self.loss_weight_pde_momentum_x,
                "pde_momentum_y": self.loss_weight_pde_momentum_y,
            }
        
        if self.bSteady:
            return base_dict
        else:
            base_dict.update({
                'initial_loss': self.loss_weight_initial_loss,                
            })
            return base_dict
        
    def _initialize_weights(self):
        """Initialize the weights of the network."""
        
        
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                if self.initialization == 'xavier_normal':
                    nn.init.xavier_normal_(m.weight)
                elif self.initialization == 'xavier_uniform':
                    nn.init.xavier_uniform_(m.weight)
                elif self.initialization == 'kaiming_normal':
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                elif self.initialization == 'kaiming_uniform':
                    nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                else:
                    raise ValueError(f"Unsupported initialization method: {self.initialization}")
                    
                nn.init.zeros_(m.bias)
                
    def forward(self, x):
        """
        Forward pass of the PINN model.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, 2/3] containing coordinates (x, y, t) for transient problems and (x, y) for steady problems. 
            
        Returns:
            tuple: (h, u, v) where each is a tensor of shape [batch_size, 1]
        """

        #output = self.net(x)  # Shape: [batch_size, 3]
        #print(f"Output shape: {output.shape}")
        #print(f"Output type: {type(output)}")

        return self.net(x)
    
    def get_loss_flags(self):
        """
        Get the loss flags.
        """
        
        return self.bPDE_loss, self.bInitial_loss, self.bBoundary_loss, self.bData_loss
    
    def get_bSteady(self):
        """
        Get the steady flag.
        """
        return self.bSteady
    
    def get_bNormalize(self):
        """
        Get the normalize flag.
        """
        return self.bNormalize
    
    def get_normalization_method(self):
        """
        Get the normalization method.
        """
        return self.normalization_method
    
    def get_device(self):
        """
        Get the device.
        """
        return self.device
        
    def compute_pde_residuals(self, x, pde_data, mesh_stats, data_stats):
        """
        Compute PDE residuals for the shallow water equations.

        "x" might be normalized or not normalized. If normalized, the output of PINN is also normalized. 

        The residuals are computed in the unnormalized space.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, 2/3] containing coordinates (x, y, t) for transient problems and (x, y) for steady problems. Normalized.
            pde_data (torch.Tensor): Data for enforcing PDE residuals (zb, nx, ny, ManningN). It is not normalized.
            mesh_stats (dict): Statistics of the mesh points.                            

        Returns:
            tuple: (mass_residual, momentum_x_residual, momentum_y_residual)
        """       
        
        # Compute derivatives
        x.requires_grad_(True)
        predictions = self.forward(x)
        h_hat, u_hat, v_hat = predictions[:, 0:1], predictions[:, 1:2], predictions[:, 2:3]      

        # Compute all gradients at once
        h_hat_grad = torch.autograd.grad(h_hat, x, grad_outputs=torch.ones_like(h_hat),
                                   create_graph=True, retain_graph=True)[0]
        u_hat_grad = torch.autograd.grad(u_hat, x, grad_outputs=torch.ones_like(u_hat),
                                   create_graph=True, retain_graph=True)[0]
        v_hat_grad = torch.autograd.grad(v_hat, x, grad_outputs=torch.ones_like(v_hat),
                                   create_graph=True, retain_graph=True)[0]
        
        # Extract specific derivatives
        dh_hat_dx_hat = h_hat_grad[:, 0:1]
        dh_hat_dy_hat = h_hat_grad[:, 1:2]
        du_hat_dx_hat = u_hat_grad[:, 0:1]
        du_hat_dy_hat = u_hat_grad[:, 1:2]
        dv_hat_dx_hat = v_hat_grad[:, 0:1]
        dv_hat_dy_hat = v_hat_grad[:, 1:2]
        
        if not self.bSteady:
            dh_hat_dt_hat = h_hat_grad[:, 2:3]
            du_hat_dt_hat = u_hat_grad[:, 2:3]
            dv_hat_dt_hat = v_hat_grad[:, 2:3]        
        
        # Get the stats
        x_min = mesh_stats['x_min']
        x_max = mesh_stats['x_max']
        y_min = mesh_stats['y_min']
        y_max = mesh_stats['y_max']
        if not self.bSteady:
            t_min = mesh_stats['t_min']
            t_max = mesh_stats['t_max']

        Lx = x_max - x_min
        Ly = y_max - y_min
        if not self.bSteady:
            Lt = t_max - t_min

        mu_h = data_stats['h_mean']
        sigma_h = data_stats['h_std']
        mu_u = data_stats['u_mean']
        sigma_u = data_stats['u_std']
        mu_v = data_stats['v_mean']
        sigma_v = data_stats['v_std']

        # Extract PDE data (zb, Sx, Sy, ManningN on PDE points)
        zb = pde_data[:, 0:1]
        Sx = pde_data[:, 1:2]
        Sy = pde_data[:, 2:3]
        ManningN = pde_data[:, 3:4]

        # Extract coordinates
        x_coord = x[:, 0:1]  # Keep dimension
        y_coord = x[:, 1:2]  # Keep dimension
        if not self.bSteady:
            t_coord = x[:, 2:3]  # Keep dimension

        #denormalize the outputs (which are normalized with z-score)
        h = h_hat * sigma_h + mu_h
        u = u_hat * sigma_u + mu_u
        v = v_hat * sigma_v + mu_v        

        #clip water depth to be positive
        h = torch.clamp(h, min=1e-3)

        # Compute velocity magnitude
        u_mag = torch.sqrt(u*u + v*v + 1e-8)
                
        # Compute the derivatives in dimensional space
        dh_dx = dh_hat_dx_hat * sigma_h / Lx
        dh_dy = dh_hat_dy_hat * sigma_h / Ly
        du_dx = du_hat_dx_hat * sigma_u / Lx
        du_dy = du_hat_dy_hat * sigma_u / Ly
        dv_dx = dv_hat_dx_hat * sigma_v / Lx
        dv_dy = dv_hat_dy_hat * sigma_v / Ly

        if not self.bSteady:
            dh_dt = dh_hat_dt_hat * sigma_h / Lt
            du_dt = du_hat_dt_hat * sigma_u / Lt
            dv_dt = dv_hat_dt_hat * sigma_v / Lt

        # Mass conservation equation in dimensional space
        mass_residual = h*du_dx + u*dh_dx + h*dv_dy + v*dh_dy
        if not self.bSteady:
            mass_residual = dh_dt + mass_residual

        # Momentum conservation equations
        momentum_x_residual = u*du_dx + v*du_dy + self.g * dh_dx - self.g * Sx + self.g * ManningN**2 * u * u_mag / (h**(4.0/3.0) + 1e-8)
        if not self.bSteady:
            momentum_x_residual = du_dt + momentum_x_residual

        momentum_y_residual = u*dv_dx + v*dv_dy + self.g * dh_dy - self.g * Sy + self.g * ManningN**2 * v * u_mag / (h**(4.0/3.0) + 1e-8)
        if not self.bSteady:
            momentum_y_residual = dv_dt + momentum_y_residual

        # Scale the residuals based on physics scales
        mass_residual = mass_residual / self.velocity_scale
        momentum_x_residual = momentum_x_residual / self.velocity_scale**2 * self.length_scale
        momentum_y_residual = momentum_y_residual / self.velocity_scale**2 * self.length_scale
        
        return mass_residual, momentum_x_residual, momentum_y_residual, h, u, v
        
    def compute_pde_loss(self, pde_points, pde_data, mesh_stats, data_stats):
        """
        Compute the PDE loss for the shallow water equations.
        
        Args:
            pde_points (torch.Tensor): Points inside the domain (x, y, t for unsteady problems and x, y for steady problems). Normalized.
            pde_data (torch.Tensor): Data for enforcing PDE residuals (zb, nx, ny, ManningN). It is not normalized.
            mesh_stats (dict): Statistics of the mesh points.

        Returns:
            torch.Tensor: PDE loss.
        """
        continuity_residual, momentum_x_residual, momentum_y_residual, h, u, v = self.compute_pde_residuals(pde_points, pde_data, mesh_stats, data_stats)
        
        # Compute the loss for each equation with stability
        continuity_loss = torch.mean(continuity_residual**2)
        momentum_x_loss = torch.mean(momentum_x_residual**2)
        momentum_y_loss = torch.mean(momentum_y_residual**2)
        
        # Add small epsilon to prevent complete zero
        eps = torch.tensor(1e-8, device=self.device)

        # PDE loss is the weighted sum of the losses for each equation
        pde_loss = (self.loss_weight_pde_continuity * continuity_loss + 
                   self.loss_weight_pde_momentum_x * momentum_x_loss + 
                   self.loss_weight_pde_momentum_y * momentum_y_loss + eps)
        
        pde_loss_components = {
            'continuity_loss': continuity_loss.item(),
            'momentum_x_loss': momentum_x_loss.item(),
            'momentum_y_loss': momentum_y_loss.item()
        }
        
        return pde_loss, pde_loss_components, h, u, v
        
    def compute_initial_loss(self, initial_points, initial_values):
        """
        Compute the loss for initial conditions.
        
        Args:
            initial_points (torch.Tensor): Points at initial time.
            initial_values (torch.Tensor): True values at initial points.
            
        Returns:
            torch.Tensor: Initial condition loss.
        """
        # Get model predictions at initial points
        predictions = self.forward(initial_points)

        h, u, v = predictions[:, 0:1], predictions[:, 1:2], predictions[:, 2:3]

        #clip h to be positive
        h = torch.clip(h, min=1e-3)
        
        # Compute MSE loss
        initial_loss = torch.mean((predictions - initial_values)**2)
        
        return initial_loss, h, u, v
        
    def compute_boundary_loss(self, boundary_info, mesh_stats, data_stats):
        """
        Compute the loss for boundary conditions. 

        The boundary conditions are specified in the config file.

        - For "inlet-q" BC, the summation of h*u_normal is the specified discharge.
        - For "exit-h" BC, the specified water surface elevation is the specified value. Water depth is computed from the water surface elevation and bed elevation. For velocities, it is zero gradient along the normal direction (zero Neumann BC).
        - For "wall" BC, the velocity (u,v) is set to zero. For the water depth h, it is zero gradient along the normal direction (zero Neumann BC).
        
        Args:
            boundary_info (tuple): Tuple of (boundary_points, boundary_ids, boundary_normals, boundary_lengths).
            mesh_stats (dict): Statistics of the mesh points.
            data_stats (dict): Statistics of the data points.                

        Returns:
            torch.Tensor: Combined boundary condition loss.
        """

        # Get the stats
        x_min = mesh_stats['x_mean']
        x_max = mesh_stats['x_max']
        y_min = mesh_stats['y_mean']
        y_max = mesh_stats['y_max']
        if not self.bSteady:
            t_min = mesh_stats['t_min']
            t_max = mesh_stats['t_max']

        Lx = x_max - x_min
        Ly = y_max - y_min
        if not self.bSteady:
            Lt = t_max - t_min

        mu_h = data_stats['h_mean']
        sigma_h = data_stats['h_std']
        mu_u = data_stats['u_mean']
        sigma_u = data_stats['u_std']
        mu_v = data_stats['v_mean']
        sigma_v = data_stats['v_std']

        boundary_points, boundary_ids, boundary_z, boundary_normals, boundary_lengths, boundary_ManningN = boundary_info

        device = boundary_points.device
        
        # Get model predictions
        predictions = self.forward(boundary_points)
        h_hat, u_hat, v_hat = predictions[:, 0:1], predictions[:, 1:2], predictions[:, 2:3]

        # Compute all gradients at once (more efficient than separate calls)
        h_hat_grad = torch.autograd.grad(h_hat, boundary_points, grad_outputs=torch.ones_like(h_hat),
                                   create_graph=True, retain_graph=True)[0]
        u_hat_grad = torch.autograd.grad(u_hat, boundary_points, grad_outputs=torch.ones_like(u_hat),
                                   create_graph=True, retain_graph=True)[0]
        v_hat_grad = torch.autograd.grad(v_hat, boundary_points, grad_outputs=torch.ones_like(v_hat),
                                   create_graph=True, retain_graph=True)[0]
        
        # Extract specific derivatives
        dh_hat_dx_hat = h_hat_grad[:, 0:1]
        dh_hat_dy_hat = h_hat_grad[:, 1:2]
        du_hat_dx_hat = u_hat_grad[:, 0:1]
        du_hat_dy_hat = u_hat_grad[:, 1:2]
        dv_hat_dx_hat = v_hat_grad[:, 0:1]
        dv_hat_dy_hat = v_hat_grad[:, 1:2]

        if not self.bSteady:
            dh_hat_dt_hat = h_hat_grad[:, 2:3]
            du_hat_dt_hat = u_hat_grad[:, 2:3]
            dv_hat_dt_hat = v_hat_grad[:, 2:3]

        # Compute the derivatives in dimensional space
        dh_dx = dh_hat_dx_hat * sigma_h / Lx
        dh_dy = dh_hat_dy_hat * sigma_h / Ly
        du_dx = du_hat_dx_hat * sigma_u / Lx
        du_dy = du_hat_dy_hat * sigma_u / Ly
        dv_dx = dv_hat_dx_hat * sigma_v / Lx
        dv_dy = dv_hat_dy_hat * sigma_v / Ly

        if not self.bSteady:
            dh_dt = dh_hat_dt_hat * sigma_h / Lt
            du_dt = du_hat_dt_hat * sigma_u / Lt
            dv_dt = dv_hat_dt_hat * sigma_v / Lt

        #denormalize the outputs (which are normalized with z-score)
        h = h_hat * sigma_h + mu_h
        u = u_hat * sigma_u + mu_u
        v = v_hat * sigma_v + mu_v        

        #clip water depth to be positive
        h = torch.clamp(h, min=1e-3)
        
        # Initialize losses
        boundary_loss = torch.tensor(0.0, device=device)
        
        # Add small epsilon for numerical stability
        eps = torch.tensor(1e-8, device=self.device)
                
        # Project gradients onto normal direction
        dh_dn = dh_dx * boundary_normals[:, 0:1] + dh_dy * boundary_normals[:, 1:2]
        du_dn = du_dx * boundary_normals[:, 0:1] + du_dy * boundary_normals[:, 1:2]
        dv_dn = dv_dx * boundary_normals[:, 0:1] + dv_dy * boundary_normals[:, 1:2]
        
        # Compute normal velocity for inlet discharge
        u_normal = u * boundary_normals[:, 0:1] + v * boundary_normals[:, 1:2]

        #boundary loss components
        boundary_loss_components = {}
        
        # Loop over all boundaries specified in config.boundary_conditions
        for bc_id, bc_config in self.BCs_from_config.items():
            bc_type = bc_config['type']
            
            # Get mask for current boundary
            bc_mask = (boundary_ids == bc_id)
            if not torch.any(bc_mask):
                continue
                
            if bc_type == "inlet-q":
                # Get number of points on the boundary
                n_inlet_q_points = torch.sum(bc_mask)

                # Get specified discharge from config
                q_specified = bc_config['q_value']

                # Convert specified discharge to tensor and distribute evenly across points
                q_specified_tensor = torch.tensor(q_specified, device=self.device)
                q_specified_per_point = torch.full((n_inlet_q_points,), q_specified_tensor / n_inlet_q_points, device=self.device)
                
                # Compute discharge per point
                q_computed_per_point = -h[bc_mask] * u_normal[bc_mask] * boundary_lengths[bc_mask]
                
                # Loss is the squared difference between computed and specified discharge per point
                discharge_loss = torch.mean((q_computed_per_point - q_specified_per_point)**2 + eps)

                boundary_loss += discharge_loss

                # Zero normal gradient for h at inlet
                h_grad_loss = torch.mean(dh_dn[bc_mask]**2 + eps)
                boundary_loss += h_grad_loss

                #store loss components
                boundary_loss_components['inlet-q_discharge_loss_bc_id_'+str(bc_id)] = discharge_loss
                boundary_loss_components['inlet-q_h_grad_loss_bc_id_'+str(bc_id)] = h_grad_loss
                
            elif bc_type == "exit-h":
                #get the exit_h_option from config
                exit_h_option = bc_config['exit_h_option']

                # Get bed elevation 
                bed_elevation = boundary_z[bc_mask]
                bed_tensor = bed_elevation.clone().detach().to(device)

                if exit_h_option == "constant":
                    # Get specified water surface elevation from config
                    wse_specified_value = bc_config['wse_value']
                    # Convert to tensor
                    wse_specified = torch.tensor(wse_specified_value, dtype=torch.float32, device=device)
                    
                elif exit_h_option == "rating-curve":
                    # Compute the total discharge from the predicted velocities and water depth 
                    q_computed = torch.sum(h[bc_mask] * u_normal[bc_mask] * boundary_lengths[bc_mask])
                    q_computed_tensor = torch.clip(q_computed.clone().detach(), min=0.0)                    
                    
                    # Find the WSE that corresponds to the computed discharge using linear interpolation
                    discharges = bc_config['discharges']
                    wses = bc_config['wses']
                    
                    # Find the indices for interpolation
                    idx = torch.searchsorted(discharges, q_computed_tensor)
                    idx = torch.clamp(idx, 1, len(discharges) - 1)
                    
                    # Get surrounding points
                    q0 = discharges[idx - 1]
                    q1 = discharges[idx]
                    wse0 = wses[idx - 1]
                    wse1 = wses[idx]
                    
                    # Linear interpolation
                    wse_specified = wse0 + (wse1 - wse0) * (q_computed_tensor - q0) / (q1 - q0)

                else: 
                    raise ValueError(f"Unsupported exit_h_option: {exit_h_option}")
                    
                # Convert to tensors and compute water depth from WSE and bed elevation
                # wse_specified is already a tensor in both cases
                wse_tensor = wse_specified.clone().detach()
                    
                h_specified = torch.clip(wse_tensor - bed_tensor, min=1e-3)
                
                # Loss is the difference between computed and specified water depth
                h_loss = torch.mean((h[bc_mask] - h_specified)**2 + eps)
                boundary_loss += h_loss

                # Zero normal gradient for velocities at outlet
                u_grad_loss = torch.mean(du_dn[bc_mask]**2 + eps)
                v_grad_loss = torch.mean(dv_dn[bc_mask]**2 + eps)
                boundary_loss += u_grad_loss + v_grad_loss
                
                #store loss components
                boundary_loss_components['exit-h_h_loss_bc_id_'+str(bc_id)] = h_loss
                boundary_loss_components['exit-h_u_grad_loss_bc_id_'+str(bc_id)] = u_grad_loss
                boundary_loss_components['exit-h_v_grad_loss_bc_id_'+str(bc_id)] = v_grad_loss
                
            elif bc_type == "wall":
                # No-slip condition: zero velocity
                u_loss = torch.mean((u[bc_mask]**2 + eps))
                v_loss = torch.mean((v[bc_mask]**2 + eps))
                boundary_loss += u_loss + v_loss
                
                # Zero gradient for h along normal direction
                h_grad_loss = torch.mean(dh_dn[bc_mask]**2 + eps)
                boundary_loss += h_grad_loss
                
                #store loss components
                boundary_loss_components['wall_h_grad_loss_bc_id_'+str(bc_id)] = h_grad_loss
                boundary_loss_components['wall_u_loss_bc_id_'+str(bc_id)] = u_loss
                boundary_loss_components['wall_v_loss_bc_id_'+str(bc_id)] = v_loss      

            elif bc_type == "symm":
                # Symmetry condition: zero normal velocity and zero normal gradient for h
                velocity_normal_loss = torch.mean((u_normal[bc_mask]**2 + eps))                

                h_grad_loss = torch.mean(dh_dn[bc_mask]**2 + eps)
                boundary_loss += velocity_normal_loss + h_grad_loss
                
                #store loss components
                boundary_loss_components['symm_velocity_normal_loss_bc_id_'+str(bc_id)] = velocity_normal_loss
                boundary_loss_components['symm_h_grad_loss_bc_id_'+str(bc_id)] = h_grad_loss                

            else:
                raise ValueError(f"Unsupported boundary condition type: {bc_type}")
        
        return boundary_loss, boundary_loss_components, h, u, v
        
    def compute_data_loss(self, data_points, data_values, data_flags, mesh_stats, data_stats):
        """
        Compute the loss for data points.
        
        Args:
            data_points (torch.Tensor): Data points. Normalized.
            data_values (torch.Tensor): h, u, v true values at data points. Normalized.
            data_flags (torch.Tensor): Flags indicating which variables are available (h, u, v).
            mesh_stats (dict): Statistics of the mesh points.
            data_stats (dict): Statistics of the data points.

        Returns:
            torch.Tensor: Total data loss.
            h_pred, u_pred, v_pred: Predicted values at data points.
        """
        
        # Ensure tensors are on the same device
        data_points = data_points.to(self.device)
        data_values = data_values.to(self.device)
        data_flags = data_flags.to(self.device)
        
        # Get model predictions at data points
        predictions = self.forward(data_points)
        
        # Split predictions and values into components
        h_pred, u_pred, v_pred = predictions[:, 0:1], predictions[:, 1:2], predictions[:, 2:3]
        h_true, u_true, v_true = data_values[:, 0:1], data_values[:, 1:2], data_values[:, 2:3]
        h_flag, u_flag, v_flag = data_flags[:, 0:1], data_flags[:, 1:2], data_flags[:, 2:3]
        
        # Compute component-wise losses with numerical stability
        eps = torch.tensor(1e-8, device=self.device)
        
        # Only compute loss for points where the variable is available (flag = 1)
        h_loss = torch.mean(h_flag * (h_pred - h_true)**2) + eps
        u_loss = torch.mean(u_flag * (u_pred - u_true)**2) + eps
        v_loss = torch.mean(v_flag * (v_pred - v_true)**2) + eps
        
        # Combine losses with weights (currently equal weights)
        data_loss = h_loss + u_loss + v_loss

        # store loss components
        loss_components = {
            'data_h_loss': h_loss.item(),
            'data_u_loss': u_loss.item(),
            'data_v_loss': v_loss.item(),
            'total_data_loss': data_loss.item()
        }

        return data_loss, loss_components, h_pred, u_pred, v_pred
        
    def compute_total_loss(self, pde_points, pde_data, initial_points, initial_values, boundary_info, data_points, data_values, data_flags, mesh_stats, data_stats):
        """
        Compute the total loss for the PINN model.
        
        Args:
            pde_points (torch.Tensor): Points for enforcing PDE residuals (x, y, t for unsteady problems and x, y for steady problems). Normalized if bNormalize is True.
            pde_data (torch.Tensor): Data for enforcing PDE residuals (zb, nx, ny, ManningN). It is not normalized.
            initial_points (torch.Tensor): Points for enforcing initial conditions (x, y, t for unsteady problems and x, y for steady problems). Normalized if bNormalize is True.
            initial_values (torch.Tensor): True values at initial points (h, u, v).
            boundary_info (tuple): Tuple containing boundary ID, normals, lengths, ManningN.
            data_points (torch.Tensor): Points for data loss (x, y, t for unsteady problems and x, y for steady problems). Normalized if bNormalize is True.
            data_values (torch.Tensor): True values at data points (h, u, v). Normalized if bNormalize is True. 
            data_flags (torch.Tensor): Flags indicating which variables are available at data points (h, u, v).
            mesh_stats (dict): Statistics of the mesh points.
            data_stats (dict): Statistics of the data points.           
            
        Returns:
            tuple: (total_loss, loss_components, predictions_and_true_values)
        """
        # Initialize losses with requires_grad=True
        pde_loss = torch.zeros(1, device=self.device, requires_grad=True)
        initial_loss = torch.zeros(1, device=self.device, requires_grad=True)
        boundary_loss = torch.zeros(1, device=self.device, requires_grad=True)
        data_loss = torch.zeros(1, device=self.device, requires_grad=True)
        
        # Initialize loss components
        pde_loss_components = {}
        initial_loss_components = {}
        boundary_loss_components = {}
        data_loss_components = {}
        
        # Initialize predictions and true values
        if self.bSteady:    
            predictions_and_true_values = {
                "bPDE_loss": self.bPDE_loss,
                "bBoundary_loss": self.bBoundary_loss,
                "bData_loss": self.bData_loss
            }
        else:
            predictions_and_true_values = {
                "bPDE_loss": self.bPDE_loss,
                "bInitial_loss": self.bInitial_loss,
                "bBoundary_loss": self.bBoundary_loss,
                "bData_loss": self.bData_loss
            }
        
        # Store loss tensors for GradNorm
        loss_tensors = {}
        
        # Compute PDE loss
        if self.bPDE_loss:
            if pde_points is not None and pde_data is not None:
                pde_loss, pde_loss_components, h_pred_pde_points, u_pred_pde_points, v_pred_pde_points = self.compute_pde_loss(pde_points, pde_data, mesh_stats, data_stats)
                predictions_and_true_values.update({
                    'pde_points': pde_points,
                    'h_pred_pde_points': h_pred_pde_points,
                    'u_pred_pde_points': u_pred_pde_points,
                    'v_pred_pde_points': v_pred_pde_points
                })
                # Ensure pde_loss is connected to the graph
                if not pde_loss.requires_grad:
                    pde_loss = pde_loss.detach().requires_grad_(True)
                loss_tensors['pde_loss'] = pde_loss
        
        # Compute initial loss
        if self.bInitial_loss:
            if initial_points is not None and initial_values is not None:
                initial_loss, initial_loss_components, h_pred_initial_points, u_pred_initial_points, v_pred_initial_points = self.compute_initial_loss(initial_points, initial_values)
                predictions_and_true_values.update({
                    'initial_points': initial_points,
                    'h_pred_initial_points': h_pred_initial_points,
                    'u_pred_initial_points': u_pred_initial_points,
                    'v_pred_initial_points': v_pred_initial_points,
                    'h_true_initial_points': initial_values[:, 0:1],
                    'u_true_initial_points': initial_values[:, 1:2],
                    'v_true_initial_points': initial_values[:, 2:3]
                })
                # Ensure initial_loss is connected to the graph
                if not initial_loss.requires_grad:
                    initial_loss = initial_loss.detach().requires_grad_(True)
                loss_tensors['initial_loss'] = initial_loss
        
        # Compute boundary loss
        if self.bBoundary_loss:
            if boundary_info is not None:
                boundary_loss, boundary_loss_components, h_pred_boundary_points, u_pred_boundary_points, v_pred_boundary_points = self.compute_boundary_loss(boundary_info, mesh_stats, data_stats)
                predictions_and_true_values.update({
                    'boundary_points': boundary_info[0],
                    'h_pred_boundary_points': h_pred_boundary_points,
                    'u_pred_boundary_points': u_pred_boundary_points,
                    'v_pred_boundary_points': v_pred_boundary_points
                })
                # Ensure boundary_loss is connected to the graph
                if not boundary_loss.requires_grad:
                    boundary_loss = boundary_loss.detach().requires_grad_(True)
                loss_tensors['boundary_loss'] = boundary_loss
        
        # Compute data loss
        if self.bData_loss:
            if data_points is not None and data_values is not None and data_flags is not None:
                data_loss, data_loss_components, h_pred_data_points, u_pred_data_points, v_pred_data_points = self.compute_data_loss(data_points, data_values, data_flags, mesh_stats, data_stats)
                predictions_and_true_values.update({
                    'data_points': data_points,
                    'h_pred_data_points': h_pred_data_points,
                    'u_pred_data_points': u_pred_data_points,
                    'v_pred_data_points': v_pred_data_points,
                    'h_true_data_points': data_values[:, 0:1],
                    'u_true_data_points': data_values[:, 1:2],
                    'v_true_data_points': data_values[:, 2:3],
                    'data_flags': data_flags
                })
                # Ensure data_loss is connected to the graph
                if not data_loss.requires_grad:
                    data_loss = data_loss.detach().requires_grad_(True)
                loss_tensors['data_loss'] = data_loss
        
        # Collect loss components for weight scheduling
        if self.bSteady:
            loss_components = {
                'pde_loss': pde_loss.item(),
                'boundary_loss': boundary_loss.item(),
                'data_loss': data_loss.item()
            }
        else:
            loss_components = {
                'pde_loss': pde_loss.item(),
                'initial_loss': initial_loss.item(),
                'boundary_loss': boundary_loss.item(),
                'data_loss': data_loss.item()
            }
               
        
        # Compute total loss with current weights
        if self.bSteady:
            weighted_total_loss = (
                self.loss_weights['pde_loss'] * pde_loss +
                self.loss_weights['boundary_loss'] * boundary_loss +
                self.loss_weights['data_loss'] * data_loss
            )
        else:
            weighted_total_loss = (
                self.loss_weights['pde_loss'] * pde_loss +
                self.loss_weights['initial_loss'] * initial_loss +
                self.loss_weights['boundary_loss'] * boundary_loss +
                self.loss_weights['data_loss'] * data_loss
            )
        
        # Compute unweighted total loss
        if self.bSteady:
            unweighted_total_loss = pde_loss + boundary_loss + data_loss
        else:
            unweighted_total_loss = pde_loss + initial_loss + boundary_loss + data_loss
        
        # create a new dictionary with the loss components
        if self.bSteady:
            loss_components_for_return = {
                'loss_components': loss_components,
                'pde_loss_components': pde_loss_components,                                
                'boundary_loss_components': boundary_loss_components,
                'data_loss_components': data_loss_components,
                'weighted_total_loss': weighted_total_loss.item(),
                'unweighted_total_loss': unweighted_total_loss.item(),
                'loss_weights': self.loss_weights.copy()
            }
        else:
            loss_components_for_return = {
                'pde_loss_components': pde_loss_components,
                'initial_loss_components': initial_loss_components,
                'boundary_loss_components': boundary_loss_components,
                'data_loss_components': data_loss_components,
                'weighted_total_loss': weighted_total_loss.item(),
                'unweighted_total_loss': unweighted_total_loss.item(),
                'loss_weights': self.loss_weights.copy()
            }
        
        return weighted_total_loss, loss_components_for_return, predictions_and_true_values
