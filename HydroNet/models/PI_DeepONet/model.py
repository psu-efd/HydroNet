"""
Physics-Informed DeepONet model for HydroNet.
"""
import torch
import torch.nn as nn
import numpy as np
from ...utils.config import Config
from ..DeepONet.model import BranchNet, TrunkNet


class PI_DeepONetModel(nn.Module):
    """
    Physics-Informed DeepONet model for learning the operator of shallow water equations.
    
    This model combines the DeepONet architecture with physics-informed constraints
    to enforce the governing equations of shallow water flow.
    """
    def __init__(self, config_file=None, config=None):
        """
        Initialize the Physics-Informed DeepONet model.
        
        Args:
            config_file (str, optional): Path to configuration file.
            config (Config, optional): Configuration object.
        """
        super(PI_DeepONetModel, self).__init__()
        
        # Load configuration
        if config is not None:
            self.config = config
        elif config_file is not None:
            self.config = Config(config_file)
        else:
            raise ValueError("Either config_file or config must be provided")
            
        # Get model parameters from config
        branch_layers = self.config.get('model.branch_net.hidden_layers', [128, 128, 128])
        branch_activation = self.config.get('model.branch_net.activation', 'relu')
        branch_dropout = self.config.get('model.branch_net.dropout_rate', 0.0)
        
        trunk_layers = self.config.get('model.trunk_net.hidden_layers', [128, 128, 128])
        trunk_activation = self.config.get('model.trunk_net.activation', 'tanh')
        trunk_dropout = self.config.get('model.trunk_net.dropout_rate', 0.0)
        
        # Input dimensions will be determined at training time if not specified
        self.branch_dim = self.config.get('data.input_dim', 0)
        self.trunk_dim = 3  # (x, y, t) coordinates
        
        # Output dimension - 3 for (h, u, v) in shallow water equations
        self.output_dim = self.config.get('model.output_dim', 3)
        
        # Hidden dimension for the DeepONet architecture
        self.hidden_dim = branch_layers[-1]
        
        # Create branch and trunk networks
        self.branch_net = BranchNet(
            self.branch_dim, 
            self.hidden_dim, 
            branch_layers, 
            branch_activation, 
            branch_dropout
        )
        
        self.trunk_net = TrunkNet(
            self.trunk_dim, 
            self.hidden_dim, 
            trunk_layers, 
            trunk_activation, 
            trunk_dropout
        )
        
        # Output bias for each output component
        self.bias = nn.Parameter(torch.zeros(self.output_dim))
        
        # Physics parameters
        self.g = self.config.get('physics.gravity', 9.81)  # Gravitational acceleration
        self.n = self.config.get('physics.manning_coefficient', 0.03)  # Manning's coefficient
        
        # Loss weights
        self.loss_weights = {
            'data_loss': self.config.get('physics.loss_weights.data_loss', 1.0),
            'continuity': self.config.get('physics.loss_weights.continuity_eq', 0.1),
            'momentum_x': self.config.get('physics.loss_weights.momentum_x_eq', 0.1),
            'momentum_y': self.config.get('physics.loss_weights.momentum_y_eq', 0.1),
            'initial': self.config.get('physics.loss_weights.initial_condition', 1.0),
            'boundary': self.config.get('physics.loss_weights.boundary_condition', 1.0)
        }
        
    def forward(self, branch_input, trunk_input):
        """
        Forward pass of Physics-Informed DeepONet.
        
        Args:
            branch_input (torch.Tensor): Input function for branch net.
            trunk_input (torch.Tensor): Coordinates for trunk net.
            
        Returns:
            torch.Tensor: Model output.
        """
        # Process branch and trunk inputs
        branch_output = self.branch_net(branch_input)  # [batch_size, hidden_dim]
        trunk_output = self.trunk_net(trunk_input)     # [batch_size, hidden_dim]
        
        # Compute inner product for each output component
        outputs = []
        for i in range(self.output_dim):
            # Extract the i-th component from each network output
            branch_i = branch_output[:, i::self.output_dim]  # [batch_size, hidden_dim // output_dim]
            trunk_i = trunk_output[:, i::self.output_dim]    # [batch_size, hidden_dim // output_dim]
            
            # Compute dot product
            output_i = torch.sum(branch_i * trunk_i, dim=1, keepdim=True) + self.bias[i]
            outputs.append(output_i)
            
        # Combine outputs
        output = torch.cat(outputs, dim=1)  # [batch_size, output_dim]
        
        return output
        
    def compute_pde_residuals(self, branch_input, trunk_input):
        """
        Compute the residuals of the PDEs (shallow water equations).
        
        Args:
            branch_input (torch.Tensor): Input function for branch net.
            trunk_input (torch.Tensor): Coordinates for trunk net.
            
        Returns:
            tuple: (continuity_residual, momentum_x_residual, momentum_y_residual)
        """
        # Ensure that gradients are computed
        trunk_input.requires_grad_(True)
        
        # Forward pass to get h, u, v
        output = self.forward(branch_input, trunk_input)
        h, u, v = output[:, 0:1], output[:, 1:2], output[:, 2:3]
        
        # Extract coordinates
        x_coord = trunk_input[:, 0:1]
        y_coord = trunk_input[:, 1:2]
        t = trunk_input[:, 2:3]
        
        # Compute gradients with respect to input coordinates
        h_x = torch.autograd.grad(h, trunk_input, torch.ones_like(h), create_graph=True)[0][:, 0:1]
        h_y = torch.autograd.grad(h, trunk_input, torch.ones_like(h), create_graph=True)[0][:, 1:2]
        h_t = torch.autograd.grad(h, trunk_input, torch.ones_like(h), create_graph=True)[0][:, 2:3]
        
        u_x = torch.autograd.grad(u, trunk_input, torch.ones_like(u), create_graph=True)[0][:, 0:1]
        u_y = torch.autograd.grad(u, trunk_input, torch.ones_like(u), create_graph=True)[0][:, 1:2]
        u_t = torch.autograd.grad(u, trunk_input, torch.ones_like(u), create_graph=True)[0][:, 2:3]
        
        v_x = torch.autograd.grad(v, trunk_input, torch.ones_like(v), create_graph=True)[0][:, 0:1]
        v_y = torch.autograd.grad(v, trunk_input, torch.ones_like(v), create_graph=True)[0][:, 1:2]
        v_t = torch.autograd.grad(v, trunk_input, torch.ones_like(v), create_graph=True)[0][:, 2:3]
        
        # Compute residuals for the shallow water equations
        
        # Continuity equation: h_t + (hu)_x + (hv)_y = 0
        continuity_residual = h_t + (h * u_x + u * h_x) + (h * v_y + v * h_y)
        
        # Momentum equation in x-direction:
        # u_t + u*u_x + v*u_y + g*h_x + g*n^2*u*sqrt(u^2 + v^2)/h^(4/3) = 0
        friction_x = self.g * self.n**2 * u * torch.sqrt(u**2 + v**2) / (h**(4/3) + 1e-8)
        momentum_x_residual = u_t + u * u_x + v * u_y + self.g * h_x + friction_x
        
        # Momentum equation in y-direction:
        # v_t + u*v_x + v*v_y + g*h_y + g*n^2*v*sqrt(u^2 + v^2)/h^(4/3) = 0
        friction_y = self.g * self.n**2 * v * torch.sqrt(u**2 + v**2) / (h**(4/3) + 1e-8)
        momentum_y_residual = v_t + u * v_x + v * v_y + self.g * h_y + friction_y
        
        return continuity_residual, momentum_x_residual, momentum_y_residual
        
    def compute_pde_loss(self, branch_input, trunk_input):
        """
        Compute the PDE loss for the shallow water equations.
        
        Args:
            branch_input (torch.Tensor): Input function for branch net.
            trunk_input (torch.Tensor): Coordinates for trunk net.
            
        Returns:
            torch.Tensor: PDE loss.
        """
        continuity_residual, momentum_x_residual, momentum_y_residual = self.compute_pde_residuals(branch_input, trunk_input)
        
        continuity_loss = torch.mean(continuity_residual**2)
        momentum_x_loss = torch.mean(momentum_x_residual**2)
        momentum_y_loss = torch.mean(momentum_y_residual**2)
        
        pde_loss = (
            self.loss_weights['continuity'] * continuity_loss +
            self.loss_weights['momentum_x'] * momentum_x_loss +
            self.loss_weights['momentum_y'] * momentum_y_loss
        )
        
        return pde_loss
        
    def compute_data_loss(self, branch_input, trunk_input, target):
        """
        Compute the data loss.
        
        Args:
            branch_input (torch.Tensor): Input function for branch net.
            trunk_input (torch.Tensor): Coordinates for trunk net.
            target (torch.Tensor): Target values.
            
        Returns:
            torch.Tensor: Data loss.
        """
        # Get model predictions
        predictions = self.forward(branch_input, trunk_input)
        
        # Compute MSE loss
        data_loss = torch.mean((predictions - target)**2)
        
        return data_loss
        
    def compute_initial_loss(self, branch_input, initial_trunk_input, initial_conditions):
        """
        Compute the loss for initial conditions.
        
        Args:
            branch_input (torch.Tensor): Input function for branch net.
            initial_trunk_input (torch.Tensor): Coordinates at initial time.
            initial_conditions (callable): Function that returns initial conditions.
            
        Returns:
            torch.Tensor: Initial condition loss.
        """
        # Get model predictions at initial points
        predictions = self.forward(branch_input, initial_trunk_input)
        
        # Get true initial conditions
        true_values = initial_conditions(branch_input, initial_trunk_input[:, 0:2])
        
        # Compute MSE loss
        initial_loss = torch.mean((predictions - true_values)**2)
        
        return initial_loss
        
    def compute_boundary_loss(self, branch_input, boundary_trunk_input, boundary_conditions):
        """
        Compute the loss for boundary conditions.
        
        Args:
            branch_input (torch.Tensor): Input function for branch net.
            boundary_trunk_input (torch.Tensor): Coordinates at boundary.
            boundary_conditions (callable): Function that returns boundary conditions.
            
        Returns:
            torch.Tensor: Boundary condition loss.
        """
        # Get model predictions at boundary points
        predictions = self.forward(branch_input, boundary_trunk_input)
        
        # Get true boundary conditions
        true_values = boundary_conditions(branch_input, boundary_trunk_input)
        
        # Compute MSE loss
        boundary_loss = torch.mean((predictions - true_values)**2)
        
        return boundary_loss
        
    def compute_total_loss(self, branch_input, trunk_input, target=None, 
                          physics_branch_input=None, physics_trunk_input=None,
                          initial_trunk_input=None, boundary_trunk_input=None,
                          initial_conditions=None, boundary_conditions=None):
        """
        Compute the total loss for the Physics-Informed DeepONet model.
        
        Args:
            branch_input (torch.Tensor): Input function for branch net.
            trunk_input (torch.Tensor): Coordinates for trunk net.
            target (torch.Tensor, optional): Target values for data loss.
            physics_branch_input (torch.Tensor, optional): Input function for physics constraints.
            physics_trunk_input (torch.Tensor, optional): Coordinates for physics constraints.
            initial_trunk_input (torch.Tensor, optional): Coordinates at initial time.
            boundary_trunk_input (torch.Tensor, optional): Coordinates at boundary.
            initial_conditions (callable, optional): Function that returns initial conditions.
            boundary_conditions (callable, optional): Function that returns boundary conditions.
            
        Returns:
            tuple: (total_loss, loss_components)
        """
        # Initialize losses
        data_loss = torch.tensor(0.0, device=trunk_input.device)
        pde_loss = torch.tensor(0.0, device=trunk_input.device)
        initial_loss = torch.tensor(0.0, device=trunk_input.device)
        boundary_loss = torch.tensor(0.0, device=trunk_input.device)
        
        # Compute data loss if target is provided
        if target is not None:
            data_loss = self.compute_data_loss(branch_input, trunk_input, target)
            
        # Compute PDE loss if physics inputs are provided
        if physics_branch_input is not None and physics_trunk_input is not None:
            pde_loss = self.compute_pde_loss(physics_branch_input, physics_trunk_input)
            
        # Compute initial condition loss if provided
        if initial_trunk_input is not None and initial_conditions is not None:
            initial_loss = self.compute_initial_loss(branch_input, initial_trunk_input, initial_conditions)
            
        # Compute boundary condition loss if provided
        if boundary_trunk_input is not None and boundary_conditions is not None:
            boundary_loss = self.compute_boundary_loss(branch_input, boundary_trunk_input, boundary_conditions)
            
        # Compute total loss
        total_loss = (
            self.loss_weights['data_loss'] * data_loss +
            pde_loss +
            self.loss_weights['initial'] * initial_loss +
            self.loss_weights['boundary'] * boundary_loss
        )
        
        # Return total loss and individual components
        loss_components = {
            'data_loss': data_loss.item(),
            'pde_loss': pde_loss.item(),
            'initial_loss': initial_loss.item(),
            'boundary_loss': boundary_loss.item(),
            'total_loss': total_loss.item()
        }
        
        return total_loss, loss_components
        
    def set_branch_dim(self, dim):
        """
        Set the input dimension for the branch network.
        
        Args:
            dim (int): New input dimension.
        """
        self.branch_dim = dim
        # Recreate branch network with new input dimension
        branch_layers = self.config.get('model.branch_net.hidden_layers', [128, 128, 128])
        branch_activation = self.config.get('model.branch_net.activation', 'relu')
        branch_dropout = self.config.get('model.branch_net.dropout_rate', 0.0)
        
        self.branch_net = BranchNet(
            self.branch_dim, 
            self.hidden_dim, 
            branch_layers, 
            branch_activation, 
            branch_dropout
        ) 