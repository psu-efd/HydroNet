"""
Physics-Informed Neural Network (PINN) model for 2D shallow water equations.
"""
import torch
import torch.nn as nn
import numpy as np
from ...utils.config import Config


class SWE_PINN(nn.Module):
    """
    Physics-Informed Neural Network for 2D Shallow Water Equations.
    
    This model solves the 2D shallow water equations using a physics-informed approach,
    enforcing the governing equations, boundary conditions, initial conditions, and data points (from simulation or measurement).

    The model supports both steady and unsteady problems (bSteady = True or False in the config file).

    Boundary conditions are specified in the config file and the boundary points are loaded from the data file. Currently, supported boundary conditions are:
    - "inlet-q": Inlet discharge. The summation of h*u_normal is the specified discharge.
    - "exit-h": Outlet water surface elevation. The specified water surface elevation is the specified value. Water depth is computed from the water surface elevation and bed elevation. For velocities, it is zero gradient along the normal direction (zero Neumann BC).
    - "wall": Wall boundary. The velocity (u,v) is set to zero. For the water depth h, it is zero gradient along the normal direction (zero Neumann BC).

    Data points are specified in the config file and the data points are loaded from the data file. The data points are typically from:
    - physics-based simulation: Data points from physics-based simulation.
    - measurement: Data points from measurement.

    The model is trained by minimizing the total loss, which is the sum of the PDE loss, initial condition loss (if unsteady), boundary condition loss, and/or data loss.
    """

    def __init__(self, config):
        """
        Initialize the PINN model.
        
        Args:
            config (Config): Configuration object.
        """
        super(SWE_PINN, self).__init__()
        
        # Load configuration
        if config is not None:
            self.config = config
        else:
            raise ValueError("config must be provided and not None")
        
        #get device from config
        device_type = self.config.get('device.type', 'cuda')
        device_index = self.config.get('device.index', 0)
        
        if device_type == 'cuda' and torch.cuda.is_available():
            self.device = torch.device(f'cuda:{device_index}')
        else:
            self.device = torch.device('cpu')
            
        # Get model parameters from config
        try:
            self.bPDE_loss = bool(self.config.get('model.bPDE_loss'))
            if self.bPDE_loss is None:
                raise ValueError("model.bPDE_loss must be specified in config file")
        except (TypeError, ValueError):
            raise ValueError("model.bPDE_loss must be a boolean value in config file")
            
        try:
            self.bBoundary_loss = bool(self.config.get('model.bBoundary_loss'))
            if self.bBoundary_loss is None:
                raise ValueError("model.bBoundary_loss must be specified in config file")
        except (TypeError, ValueError):
            raise ValueError("model.bBoundary_loss must be a boolean value in config file")
            
        try:
            self.bSteady = bool(self.config.get('model.bSteady'))
            #print(f"bSteady: {self.bSteady}")
            self.bInitial_loss = not self.bSteady    #for unsteady problems, bSteady is false and the initial conditions mismatch is included in the loss function
            #print(f"bInitial_loss: {self.bInitial_loss}")
            if self.bSteady is None:
                raise ValueError("model.bSteady must be specified in config file")
        except (TypeError, ValueError):
            raise ValueError("model.bSteady must be a boolean value in config file")
            
        try:
            self.bData_loss = bool(self.config.get('model.bData_loss'))
            if self.bData_loss is None:
                raise ValueError("model.bData_loss must be specified in config file")
        except (TypeError, ValueError):
            raise ValueError("model.bData_loss must be a boolean value in config file")

        hidden_layers = self.config.get('model.hidden_layers', [64, 128, 128, 64])
        activation = self.config.get('model.activation', 'tanh')
        self.output_dim = self.config.get('model.output_dim', 3)  # h, u, v
        
        # Physics parameters
        self.g = self.config.get('physics.gravity', 9.81)  # Gravitational acceleration
        self.n = self.config.get('physics.manning_coefficient', 0.03)  # Manning's coefficient (needs to be specified in a file for non-uniform roughness)
        
        # Loss weights
        self.loss_weights = {
            'continuity': self.config.get('physics.loss_weights.continuity_eq', 1.0),
            'momentum_x': self.config.get('physics.loss_weights.momentum_x_eq', 1.0),
            'momentum_y': self.config.get('physics.loss_weights.momentum_y_eq', 1.0),
            'initial': self.config.get('physics.loss_weights.initial_condition', 1.0),
            'boundary': self.config.get('physics.loss_weights.boundary_condition', 1.0),
            'data': self.config.get('physics.loss_weights.data_points', 1.0)
        }

        # Read in the boundary conditions
        try:
            self.BCs_from_config = self.config.get('boundary_conditions')
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

        # Move the model to the device
        self.to(self.device)
        
    def _initialize_weights(self):
        """Initialize the weights of the network."""
        initialization = self.config.get('model.initialization', 'xavier_normal')
        
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                if initialization == 'xavier_normal':
                    nn.init.xavier_normal_(m.weight)
                elif initialization == 'xavier_uniform':
                    nn.init.xavier_uniform_(m.weight)
                elif initialization == 'kaiming_normal':
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                elif initialization == 'kaiming_uniform':
                    nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                    
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
    
    def get_device(self):
        """
        Get the device.
        """
        return self.device
        
    def compute_pde_residuals(self, x):
        """
        Compute PDE residuals for the shallow water equations.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, 2/3] containing coordinates (x, y, t) for transient problems and (x, y) for steady problems. 
                            
        Returns:
            tuple: (mass_residual, momentum_x_residual, momentum_y_residual)
        """

        # Extract coordinates
        x_coord = x[:, 0:1]  # Keep dimension
        y_coord = x[:, 1:2]  # Keep dimension
        if not self.bSteady:
            t_coord = x[:, 2:3]  # Keep dimension
        
        # Compute derivatives
        x.requires_grad_(True)
        predictions = self.forward(x)
        h, u, v = predictions[:, 0:1], predictions[:, 1:2], predictions[:, 2:3]
        
        #clip h to be positive
        h = torch.clip(h, min=1e-3)
        
        # Compute all gradients at once
        h_grad = torch.autograd.grad(h, x, grad_outputs=torch.ones_like(h),
                                   create_graph=True)[0]
        u_grad = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                   create_graph=True)[0]
        v_grad = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v),
                                   create_graph=True)[0]
        
        # Extract specific derivatives
        dh_dx = h_grad[:, 0:1]
        dh_dy = h_grad[:, 1:2]
        du_dx = u_grad[:, 0:1]
        du_dy = u_grad[:, 1:2]
        dv_dx = v_grad[:, 0:1]
        dv_dy = v_grad[:, 1:2]
        
        if not self.bSteady:
            dh_dt = h_grad[:, 2:3]
            du_dt = u_grad[:, 2:3]
            dv_dt = v_grad[:, 2:3]
        
        # Compute velocity magnitude
        u_mag = torch.sqrt(u*u + v*v)

        # Make sure h is positive (clipping)
        h = torch.clip(h, min=1e-3)
        
        # Mass conservation equation
        if self.bSteady:            
            mass_residual = h*du_dx + u*dh_dx + h*dv_dy + v*dh_dy
        else:
            mass_residual = dh_dt + h*du_dx + u*dh_dx + h*dv_dy + v*dh_dy
        
        # Momentum conservation equations
        if self.bSteady:
            momentum_x_residual = (u*du_dx + v*du_dy + 
                                 self.g * dh_dx + 
                                 self.g * self.n**2 * u * u_mag / (h**(4/3)))
            momentum_y_residual = (u*dv_dx + v*dv_dy + 
                                 self.g * dh_dy + 
                                 self.g * self.n**2 * v * u_mag / (h**(4/3)))
        else:
            momentum_x_residual = (du_dt + u*du_dx + v*du_dy + 
                                 self.g * dh_dx + 
                                 self.g * self.n**2 * u * u_mag / (h**(4/3)))
            momentum_y_residual = (dv_dt + u*dv_dx + v*dv_dy + 
                                 self.g * dh_dy + 
                                 self.g * self.n**2 * v * u_mag / (h**(4/3)))
        
        return mass_residual, momentum_x_residual, momentum_y_residual, h, u, v
        
    def compute_pde_loss(self, pde_points):
        """
        Compute the PDE loss for the shallow water equations.
        
        Args:
            pde_points (torch.Tensor): Points inside the domain.
            
        Returns:
            torch.Tensor: PDE loss.
        """
        continuity_residual, momentum_x_residual, momentum_y_residual, h, u, v = self.compute_pde_residuals(pde_points)
        
        # Compute the loss for each equation with stability
        continuity_loss = torch.mean(continuity_residual**2)
        momentum_x_loss = torch.mean(momentum_x_residual**2)
        momentum_y_loss = torch.mean(momentum_y_residual**2)
        
        # Add small epsilon to prevent complete zero
        eps = torch.tensor(1e-8, device=self.device)

        pde_loss_components = {
            'continuity_loss': continuity_loss.item(),
            'momentum_x_loss': momentum_x_loss.item(),
            'momentum_y_loss': momentum_y_loss.item()
        }
        
        pde_loss = (
            self.loss_weights['continuity'] * (continuity_loss + eps) +
            self.loss_weights['momentum_x'] * (momentum_x_loss + eps) +
            self.loss_weights['momentum_y'] * (momentum_y_loss + eps)
        )
        
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
        
    def compute_boundary_loss(self, boundary_info):
        """
        Compute the loss for boundary conditions. 

        The boundary conditions are specified in the config file.

        - For "inlet-q" BC, the summation of h*u_normal is the specified discharge.
        - For "exit-h" BC, the specified water surface elevation is the specified value. Water depth is computed from the water surface elevation and bed elevation. For velocities, it is zero gradient along the normal direction (zero Neumann BC).
        - For "wall" BC, the velocity (u,v) is set to zero. For the water depth h, it is zero gradient along the normal direction (zero Neumann BC).
        
        Args:
            boundary_info (tuple): Tuple of (boundary_points, boundary_ids, boundary_normals, boundary_lengths).
            
        Returns:
            torch.Tensor: Combined boundary condition loss.
        """
        boundary_points, boundary_ids, boundary_normals, boundary_lengths = boundary_info
        device = boundary_points.device
        
        # Get model predictions
        predictions = self.forward(boundary_points)
        h, u, v = predictions[:, 0:1], predictions[:, 1:2], predictions[:, 2:3]

        # Make sure h is positive (clipping)
        h = torch.clip(h, min=1e-3)
        
        # Initialize losses
        boundary_loss = torch.tensor(0.0, device=device)
        
        # Add small epsilon for numerical stability
        eps = torch.tensor(1e-8, device=self.device)
        
        # Compute gradients for Neumann conditions
        h_grad = torch.autograd.grad(
            h,  
            boundary_points,
            grad_outputs=torch.ones_like(h),
            create_graph=True,
            retain_graph=True
        )[0]      #grad returns a tuple of gradients; this is why we need [0]
        u_grad = torch.autograd.grad(
            u,
            boundary_points,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True
        )[0]
        v_grad = torch.autograd.grad(
            v,
            boundary_points,
            grad_outputs=torch.ones_like(v),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Project gradients onto normal direction
        dh_dn = h_grad[:, 0:1] * boundary_normals[:, 0:1] + h_grad[:, 1:2] * boundary_normals[:, 1:2]
        du_dn = u_grad[:, 0:1] * boundary_normals[:, 0:1] + u_grad[:, 1:2] * boundary_normals[:, 1:2]
        dv_dn = v_grad[:, 0:1] * boundary_normals[:, 0:1] + v_grad[:, 1:2] * boundary_normals[:, 1:2]
        
        # Compute normal velocity for inlet discharge
        u_normal = u * boundary_normals[:, 0:1] + v * boundary_normals[:, 1:2]

        #boundary loss components
        boundary_loss_components = {}
        
        # Loop over all boundaries specified in config.boundary_conditions
        for bc_name, bc_config in self.BCs_from_config.items():
            bc_id = bc_config['id']
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
                # Get specified water surface elevation from config
                wse_specified = bc_config['wse_value']
                # Get bed elevation (assuming it's 0 for now)
                bed_elevation = 0.0
                # Convert to tensors and compute water depth from WSE and bed elevation
                wse_tensor = torch.tensor(wse_specified, device=device)
                bed_tensor = torch.tensor(bed_elevation, device=device)
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

            else:
                raise ValueError(f"Unsupported boundary condition type: {bc_type}")
        
        return boundary_loss, boundary_loss_components, h, u, v
        
    def compute_data_loss(self, data_points, data_values, data_flags):
        """
        Compute the loss for data points.
        
        Args:
            data_points (torch.Tensor): Data points.
            data_values (torch.Tensor): True values at data points.
            data_flags (torch.Tensor): Flags indicating which variables are available (h, u, v).
            
        Returns:
            torch.Tensor: Total data loss.
            h_pred, u_pred, v_pred: Predicted values at data points.
        """
        # Input validation
        if data_points is None or data_values is None or data_flags is None:
            raise ValueError("data_points, data_values, and data_flags cannot be None")
        if not isinstance(data_points, torch.Tensor) or not isinstance(data_values, torch.Tensor) or not isinstance(data_flags, torch.Tensor):
            raise TypeError("data_points, data_values, and data_flags must be torch.Tensors")
        
        # Check shapes
        if data_values.shape[1] != self.output_dim:
            raise ValueError(f"data_values must have shape [batch_size, {self.output_dim}], got {data_values.shape}")
        if data_flags.shape[1] != self.output_dim:
            raise ValueError(f"data_flags must have shape [batch_size, {self.output_dim}], got {data_flags.shape}")
        
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
        
        # Combine losses with weights
        data_loss = (
            self.loss_weights.get('data_h', 1.0) * h_loss +
            self.loss_weights.get('data_u', 1.0) * u_loss +
            self.loss_weights.get('data_v', 1.0) * v_loss
        )

        # store loss components
        loss_components = {
            'data_h_loss': h_loss.item(),
            'data_u_loss': u_loss.item(),
            'data_v_loss': v_loss.item(),
            'total_data_loss': data_loss.item()
        }

        return data_loss, loss_components, h_pred, u_pred, v_pred
        
    def compute_total_loss(self, pde_points, initial_points, initial_values, boundary_info, data_points, data_values, data_flags):
        """
        Compute the total loss for the PINN model.
        
        Args:
            pde_points (torch.Tensor): Points for enforcing PDE residuals.
            initial_points (torch.Tensor): Points for enforcing initial conditions.
            initial_values (torch.Tensor): True values at initial points.
            boundary_info (tuple): Tuple containing boundary points, identifiers, normals, and lengths.
            data_points (torch.Tensor): Points for data loss.
            data_values (torch.Tensor): True values at data points.
            data_flags (torch.Tensor): Flags indicating which variables are available at data points.
            
        Returns:
            tuple: (total_loss, loss_components, predictions_and_true_values)
        """
        # Initialize losses
        pde_loss = torch.tensor(0.0, device=self.device)
        initial_loss = torch.tensor(0.0, device=self.device)
        boundary_loss = torch.tensor(0.0, device=self.device)
        data_loss = torch.tensor(0.0, device=self.device)
        
        # Initialize loss components
        pde_loss_components = {}
        initial_loss_components = {}
        boundary_loss_components = {}
        data_loss_components = {}
        
        # Initialize predictions and true values
        predictions_and_true_values = {
            "bPDE_loss": self.bPDE_loss,
            "bInitial_loss": self.bInitial_loss,
            "bBoundary_loss": self.bBoundary_loss,
            "bData_loss": self.bData_loss
        }
        
        # Compute PDE loss
        if self.bPDE_loss:
            if pde_points is not None:
                pde_loss, pde_loss_components, h_pred_pde_points, u_pred_pde_points, v_pred_pde_points = self.compute_pde_loss(pde_points)
                predictions_and_true_values.update({
                    'pde_points': pde_points,
                    'h_pred_pde_points': h_pred_pde_points,
                    'u_pred_pde_points': u_pred_pde_points,
                    'v_pred_pde_points': v_pred_pde_points
                })
        
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
        
        # Compute boundary loss
        if self.bBoundary_loss:
            if boundary_info is not None:
                boundary_loss, boundary_loss_components, h_pred_boundary_points, u_pred_boundary_points, v_pred_boundary_points = self.compute_boundary_loss(boundary_info)
                predictions_and_true_values.update({
                    'boundary_points': boundary_info[0],
                    'h_pred_boundary_points': h_pred_boundary_points,
                    'u_pred_boundary_points': u_pred_boundary_points,
                    'v_pred_boundary_points': v_pred_boundary_points
                })
        
        # Compute data loss
        if self.bData_loss:
            if data_points is not None and data_values is not None and data_flags is not None:
                data_loss, data_loss_components, h_pred_data_points, u_pred_data_points, v_pred_data_points = self.compute_data_loss(data_points, data_values, data_flags)
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
        
        # Compute total loss
        total_loss = (
            self.loss_weights.get('pde', 1.0) * pde_loss +
            self.loss_weights.get('initial', 1.0) * initial_loss +
            self.loss_weights.get('boundary', 1.0) * boundary_loss +
            self.loss_weights.get('data', 1.0) * data_loss
        )
        
        # Return total loss and individual components
        loss_components = {
            'pde_loss': pde_loss.item(),
            'pde_loss_components': pde_loss_components,
            'initial_loss': initial_loss.item(),
            'initial_loss_components': initial_loss_components,
            'boundary_loss': boundary_loss.item(),
            'boundary_loss_components': boundary_loss_components,
            'data_loss': data_loss.item(),
            'data_loss_components': data_loss_components,
            'total_loss': total_loss.item()
        }
        
        return total_loss, loss_components, predictions_and_true_values