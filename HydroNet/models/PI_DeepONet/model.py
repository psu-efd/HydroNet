"""
Physics-Informed SWE_DeepONet model for HydroNet.
"""
import torch
import torch.nn as nn
import numpy as np
from ...utils.config import Config
from ..DeepONet.model import SWE_DeepONetModel, BranchNet
from typing import Dict, Tuple, Optional


class PI_SWE_DeepONetModel(SWE_DeepONetModel):
    """
    Physics-Informed SWE_DeepONet model for learning the operator of shallow water equations.
    
    This model extends SWE_DeepONetModel with physics-informed constraints
    to enforce the governing equations of shallow water flow.
    
    The model supports both steady and unsteady problems, similar to PINN.
    """
    def __init__(self, config):
        """
        Initialize the Physics-Informed DeepONet model.
        
        Args:            
            config (Config): Configuration object.
        """

        if not isinstance(config, Config):
            raise ValueError("config must be a Config object")

        self.config = config
        
        # Initialize parent class (SWE_DeepONetModel)
        # Note: We need to handle the case where branch_input_dim might be 0
        # The parent class expects a valid branch_input_dim, so we'll set a default
        # and update it later if needed
        #original_branch_dim = self.config.get('model.branch_net.branch_input_dim', 0)
        #if original_branch_dim == 0:
            # Temporarily set a default value for parent initialization
        #    self.config.set('model.branch_net.branch_input_dim', 10)  # Temporary default
        
        # Check if steady or unsteady and set trunk_input_dim accordingly
        #bSteady = self.config.get('physics.bSteady', False)
        #if bSteady:
        #    self.config.set('data.deeponet.trunk_input_dim', 2)  # (x, y) for steady
        #else:
        #    self.config.set('data.deeponet.trunk_input_dim', 3)  # (x, y, t) for unsteady
        
        # Call parent constructor
        super(PI_SWE_DeepONetModel, self).__init__(self.config)
        
        # Restore original branch_input_dim if it was 0
        #if original_branch_dim == 0:
        #    self.branch_input_dim = 0
        #    # Set branch_net to None so it can be created later
        #    self.branch_net = None
        
        # Get device from config
        device_type = self.config.get('device.type', 'cuda')
        device_index = self.config.get('device.index', 0)
        
        if device_type == 'cuda' and torch.cuda.is_available():
            self.device = torch.device(f'cuda:{device_index}')
        else:
            self.device = torch.device('cpu')
        
        # Check if steady or unsteady (no need because it is already set in the parent class)
        #self.bSteady = bSteady
        
        # Physics parameters
        g_value = self.config.get('physics.gravity', 9.81)
        self.register_buffer('g', torch.tensor(g_value, dtype=torch.float32))
        
        # Get physics scales       
        length_scale_value = self.config.get('physics.scales.length')
        print(f"DEBUG: length_scale_value from config: {length_scale_value}, type: {type(length_scale_value)}")
        #make sure length scale is positive 
        if length_scale_value <= 0:
            raise ValueError("Length scale must be positive")
        length_scale_tensor = torch.tensor(length_scale_value, dtype=torch.float32)
        print(f"DEBUG: length_scale_tensor before register_buffer: {length_scale_tensor}")
        self.register_buffer('length_scale', length_scale_tensor)
        print(f"DEBUG: self.length_scale after register_buffer: {self.length_scale}")
       
        velocity_scale_value = self.config.get('physics.scales.velocity')
        print(f"DEBUG: velocity_scale_value from config: {velocity_scale_value}, type: {type(velocity_scale_value)}")
        if velocity_scale_value <= 0:
            raise ValueError("Velocity scale must be positive")
        velocity_scale_tensor = torch.tensor(velocity_scale_value, dtype=torch.float32)
        print(f"DEBUG: velocity_scale_tensor before register_buffer: {velocity_scale_tensor}")
        self.register_buffer('velocity_scale', velocity_scale_tensor)
        print(f"DEBUG: self.velocity_scale after register_buffer: {self.velocity_scale}")

        #debug: print length_scale and velocity_scale
        print(f"length_scale: {self.length_scale.item() if self.length_scale.numel() == 1 else self.length_scale}")
        print(f"velocity_scale: {self.velocity_scale.item() if self.velocity_scale.numel() == 1 else self.velocity_scale}")
        
        # Loss weights - register as buffers so they're properly moved with self.to(device)
        if self.bSteady:
            deeponet_data_loss_val = self.config.get('training.loss_weights.deeponet.data_loss', 1.0)
            deeponet_pinn_loss_val = self.config.get('training.loss_weights.deeponet.pinn_loss', 1.0)
            print(f"DEBUG: Loss weights from config - deeponet_data_loss: {deeponet_data_loss_val}, deeponet_pinn_loss: {deeponet_pinn_loss_val}")
            self.register_buffer('loss_weight_pinn_data_loss', torch.tensor(self.config.get('training.loss_weights.pinn.data_loss', 1.0), dtype=torch.float32))
            self.register_buffer('loss_weight_pinn_pde_loss', torch.tensor(self.config.get('training.loss_weights.pinn.pde_loss', 1.0), dtype=torch.float32))
            self.register_buffer('loss_weight_pinn_boundary_loss', torch.tensor(self.config.get('training.loss_weights.pinn.boundary_loss', 1.0), dtype=torch.float32))
            self.register_buffer('loss_weight_deeponet_data_loss', torch.tensor(deeponet_data_loss_val, dtype=torch.float32))
            self.register_buffer('loss_weight_deeponet_pinn_loss', torch.tensor(deeponet_pinn_loss_val, dtype=torch.float32))
            self.register_buffer('loss_weight_deeponet_boundary_loss', torch.tensor(self.config.get('training.loss_weights.deeponet.boundary_loss', 1.0), dtype=torch.float32))
            print(f"DEBUG: After register_buffer - loss_weight_deeponet_data_loss: {self.loss_weight_deeponet_data_loss}, loss_weight_deeponet_pinn_loss: {self.loss_weight_deeponet_pinn_loss}")
        else:
            deeponet_data_loss_val = self.config.get('training.loss_weights.deeponet.data_loss', 1.0)
            deeponet_pinn_loss_val = self.config.get('training.loss_weights.deeponet.pinn_loss', 1.0)
            print(f"DEBUG: Loss weights from config - deeponet_data_loss: {deeponet_data_loss_val}, deeponet_pinn_loss: {deeponet_pinn_loss_val}")
            self.register_buffer('loss_weight_pinn_data_loss', torch.tensor(self.config.get('training.loss_weights.pinn.data_loss', 1.0), dtype=torch.float32))
            self.register_buffer('loss_weight_pinn_pde_loss', torch.tensor(self.config.get('training.loss_weights.pinn.pde_loss', 1.0), dtype=torch.float32))
            self.register_buffer('loss_weight_pinn_boundary_loss', torch.tensor(self.config.get('training.loss_weights.pinn.boundary_loss', 1.0), dtype=torch.float32))
            self.register_buffer('loss_weight_pinn_initial_loss', torch.tensor(self.config.get('training.loss_weights.pinn.initial_loss', 1.0), dtype=torch.float32))
            self.register_buffer('loss_weight_deeponet_data_loss', torch.tensor(deeponet_data_loss_val, dtype=torch.float32))
            self.register_buffer('loss_weight_deeponet_pinn_loss', torch.tensor(deeponet_pinn_loss_val, dtype=torch.float32))
            self.register_buffer('loss_weight_deeponet_boundary_loss', torch.tensor(self.config.get('training.loss_weights.deeponet.boundary_loss', 1.0), dtype=torch.float32))
            self.register_buffer('loss_weight_deeponet_initial_loss', torch.tensor(self.config.get('training.loss_weights.deeponet.initial_loss', 1.0), dtype=torch.float32))
            print(f"DEBUG: After register_buffer - loss_weight_deeponet_data_loss: {self.loss_weight_deeponet_data_loss}, loss_weight_deeponet_pinn_loss: {self.loss_weight_deeponet_pinn_loss}")
        
        # Move model to device
        print(f"DEBUG: Before self.to(device), length_scale: {self.length_scale}, velocity_scale: {self.velocity_scale}")
        print(f"DEBUG: Before self.to(device), loss_weights: {self.loss_weights}")
        print(f"DEBUG: Device: {self.device}")
        self.to(self.device)
        print(f"DEBUG: After self.to(device), length_scale: {self.length_scale}, velocity_scale: {self.velocity_scale}")
        print(f"DEBUG: After self.to(device), loss_weights: {self.loss_weights}")
        print(f"DEBUG: After self.to(device), length_scale device: {self.length_scale.device}, velocity_scale device: {self.velocity_scale.device}")
    
    def load_state_dict(self, state_dict, strict=True):
        """
        Override load_state_dict to preserve buffers if they're missing from the loaded state_dict.
        This ensures that physics scales and loss weights are not reset to 0 when loading checkpoints
        that were saved before these buffers were added.
        """
        # Store current buffer values before loading
        buffer_backup = {}
        for name, buffer in self.named_buffers():
            if name in ['length_scale', 'velocity_scale', 'g'] or name.startswith('loss_weight_'):
                buffer_backup[name] = buffer.clone()
        
        # Load the state_dict
        missing_keys, unexpected_keys = super().load_state_dict(state_dict, strict=strict)
        
        # Restore buffers if they were missing from the loaded state_dict
        for name, backup_value in buffer_backup.items():
            if name not in state_dict:
                # Buffer was missing, restore from backup
                self.register_buffer(name, backup_value)
                print(f"Warning: Buffer '{name}' was missing from state_dict, preserving current value: {backup_value.item() if backup_value.numel() == 1 else backup_value}")
        
        return missing_keys, unexpected_keys
    
    @property
    def loss_weights(self):
        """
        Property that returns loss weights as a dictionary for convenient access.
        The actual weights are stored as registered buffers.
        """
        if self.bSteady:
            return {
                'pinn_data_loss': self.loss_weight_pinn_data_loss,
                'pinn_pde_loss': self.loss_weight_pinn_pde_loss,
                'pinn_boundary_loss': self.loss_weight_pinn_boundary_loss,
                'deeponet_data_loss': self.loss_weight_deeponet_data_loss,
                'deeponet_pinn_loss': self.loss_weight_deeponet_pinn_loss,
                'deeponet_boundary_loss': self.loss_weight_deeponet_boundary_loss
            }
        else:
            return {
                'pinn_data_loss': self.loss_weight_pinn_data_loss,
                'pinn_pde_loss': self.loss_weight_pinn_pde_loss,
                'pinn_boundary_loss': self.loss_weight_pinn_boundary_loss,
                'pinn_initial_loss': self.loss_weight_pinn_initial_loss,
                'deeponet_data_loss': self.loss_weight_deeponet_data_loss,
                'deeponet_pinn_loss': self.loss_weight_deeponet_pinn_loss,
                'deeponet_boundary_loss': self.loss_weight_deeponet_boundary_loss,
                'deeponet_initial_loss': self.loss_weight_deeponet_initial_loss
            }
    
    def forward(self, branch_input, trunk_input):
        """
        Forward pass of Physics-Informed DeepONet.
        
        This method extends the parent's forward method to add a check for branch_net initialization.
        
        Args:
            branch_input (torch.Tensor): Input function for branch net [batch_size, branch_input_dim].
            trunk_input (torch.Tensor): Coordinates for trunk net [batch_size, trunk_input_dim].
            
        Returns:
            torch.Tensor: Model output [batch_size, output_dim] containing (h, u, v).
        """
        if self.branch_net is None:
            raise ValueError("Branch network not initialized. Set branch_input_dim first using set_branch_input_dim().")
        
        # Use parent's forward method
        return super().forward(branch_input, trunk_input)
    
    def set_branch_input_dim(self, dim):
        """
        Set the input dimension for the branch network and recreate it.
        
        Args:
            dim (int): New input dimension.
        """
        self.branch_input_dim = dim
        
        # Get branch network parameters from config
        branch_layers = self.config.get('model.branch_net.hidden_layers', [128, 128, 128])
        branch_activation = self.config.get('model.branch_net.activation', 'relu')
        branch_dropout = self.config.get('model.branch_net.dropout_rate', 0.0)
        
        # Recreate branch network with new input dimension
        self.branch_net = BranchNet(
            self.branch_input_dim, 
            self.hidden_dim, 
            branch_layers, 
            branch_activation, 
            branch_dropout
        ).to(self.device)
        
    def compute_pde_residuals(self, branch_input, trunk_input, pde_data, deeponet_points_stats, pinn_points_stats):
        """
        Compute the residuals of the PDEs (shallow water equations).
        
        This method follows the same approach as PINN for computing PDE residuals,
        including bed elevation, bed slopes, and Manning's coefficient from pde_data.
        
        Args:
            branch_input (torch.Tensor): Input function for branch net [batch_size, branch_input_dim] (normalized)
            trunk_input (torch.Tensor): Coordinates for trunk net [batch_size, trunk_input_dim] (normalized)
            pde_data (torch.Tensor): PDE data containing (zb, Sx, Sy, ManningN) [batch_size, 4] (not normalized).
            deeponet_points_stats (dict): Statistics of the DeepONet points for normalization.
            pinn_points_stats (dict): Statistics of the PINN points for normalization.
            
        Returns:
            tuple: (continuity_residual, momentum_x_residual, momentum_y_residual, h, u, v)
        """
        # Safety check: Ensure buffers are not zero (they might have been reset by load_state_dict)
        if self.velocity_scale.item() == 0.0 or self.length_scale.item() == 0.0:
            print(f"WARNING: Physics scales are zero! Re-initializing from config...")
            print(f"  velocity_scale was: {self.velocity_scale.item()}, length_scale was: {self.length_scale.item()}")
            velocity_scale_value = self.config.get('physics.scales.velocity')
            length_scale_value = self.config.get('physics.scales.length')
            if velocity_scale_value > 0:
                self.register_buffer('velocity_scale', torch.tensor(velocity_scale_value, dtype=torch.float32, device=self.device))
            if length_scale_value > 0:
                self.register_buffer('length_scale', torch.tensor(length_scale_value, dtype=torch.float32, device=self.device))
            print(f"  Re-initialized: velocity_scale={self.velocity_scale.item()}, length_scale={self.length_scale.item()}")
        
        # Ensure that gradients are computed
        trunk_input = trunk_input.clone().detach().requires_grad_(True)
        
        # Forward pass to get h, u, v
        output = self.forward(branch_input, trunk_input)
        h_hat, u_hat, v_hat = output[:, 0:1], output[:, 1:2], output[:, 2:3]

        # Compute all gradients at once (more efficient than separate calls)
        h_hat_grad = torch.autograd.grad(h_hat, trunk_input, grad_outputs=torch.ones_like(h_hat),
                                   create_graph=True, retain_graph=True)[0]
        u_hat_grad = torch.autograd.grad(u_hat, trunk_input, grad_outputs=torch.ones_like(u_hat),
                                   create_graph=True, retain_graph=True)[0]
        v_hat_grad = torch.autograd.grad(v_hat, trunk_input, grad_outputs=torch.ones_like(v_hat),
                                   create_graph=True, retain_graph=True)[0]
        
        # Extract specific derivatives
        dh_hat_dx_hat = h_hat_grad[:, 0:1]
        dh_hat_dy_hat = h_hat_grad[:, 1:2]
        du_hat_dx_hat = u_hat_grad[:, 0:1]
        du_hat_dy_hat = u_hat_grad[:, 1:2]
        dv_hat_dx_hat = v_hat_grad[:, 0:1]
        dv_hat_dy_hat = v_hat_grad[:, 1:2]

        #debug: print h_hat, u_hat, v_hat and their gradients
        #print(f"h_hat: {h_hat}")
        #print(f"u_hat: {u_hat}")
        #print(f"v_hat: {v_hat}")
        #print(f"dh_hat_dx_hat: {dh_hat_dx_hat}")
        #print(f"dh_hat_dy_hat: {dh_hat_dy_hat}")
        #print(f"du_hat_dx_hat: {du_hat_dx_hat}")
        #print(f"du_hat_dy_hat: {du_hat_dy_hat}")
        #print(f"dv_hat_dx_hat: {dv_hat_dx_hat}")
        #print(f"dv_hat_dy_hat: {dv_hat_dy_hat}")

        
        if not self.bSteady:
            dh_hat_dt_hat = h_hat_grad[:, 2:3]
            du_hat_dt_hat = u_hat_grad[:, 2:3]
            dv_hat_dt_hat = v_hat_grad[:, 2:3]
        
        #debug: print all keys of deeponet_points_stats
        #print(f"All keys of deeponet_points_stats: {deeponet_points_stats.keys()}")
        #print(f"All keys of pinn_points_stats: {pinn_points_stats.keys()}")

        # Get normalization stats (note: the inputs and outputs are always normalized)
        # Normalization stats for the output (h, u, v) of the DeepONet.
        mu_h = deeponet_points_stats['all_data_stats']['h_mean']
        sigma_h = deeponet_points_stats['all_data_stats']['h_std']
        mu_u = deeponet_points_stats['all_data_stats']['u_mean']
        sigma_u = deeponet_points_stats['all_data_stats']['u_std']
        mu_v = deeponet_points_stats['all_data_stats']['v_mean']
        sigma_v = deeponet_points_stats['all_data_stats']['v_std']        
        
        # Normalization stats for the coordinates (x, y, t) of PINN.
        x_min = pinn_points_stats['all_points_stats']['x_mean']
        x_max = pinn_points_stats['all_points_stats']['x_max']
        y_min = pinn_points_stats['all_points_stats']['y_mean']
        y_max = pinn_points_stats['all_points_stats']['y_max']
        if not self.bSteady:
            t_min = pinn_points_stats['all_points_stats']['t_min']
            t_max = pinn_points_stats['all_points_stats']['t_max']

        Lx = x_max - x_min
        Ly = y_max - y_min
        if not self.bSteady:
            Lt = t_max - t_min

        #denormalize the outputs (which are normalized with z-score)
        h = h_hat * sigma_h + mu_h
        u = u_hat * sigma_u + mu_u
        v = v_hat * sigma_v + mu_v        

        #clip water depth to be positive
        h = torch.clamp(h, min=1e-3)
        
        #denormalize the coordinates (which are normalized with min-max) (NOT USEFUL FOR NOW)
        #x = x_hat * Lx + x_min
        #y = y_hat * Ly + y_min
        #if not self.bSteady:
        #    t = t_hat * Lt + t_min
        
        # Extract PDE data (zb, Sx, Sy, ManningN, which are not normalized)
        zb = pde_data[:, 0:1]
        Sx = pde_data[:, 1:2]
        Sy = pde_data[:, 2:3]
        ManningN = pde_data[:, 3:4]
        
        # Compute velocity magnitude
        u_mag = torch.sqrt(u*u + v*v + 1e-8)
                
        # Compute the derivatives in dimensional space
        dh_dx = dh_hat_dx_hat * sigma_h / Lx
        dh_dy = dh_hat_dy_hat * sigma_h / Ly
        du_dx = du_hat_dx_hat * sigma_u / Lx
        du_dy = du_hat_dy_hat * sigma_u / Ly
        dv_dx = dv_hat_dx_hat * sigma_v / Lx
        dv_dy = dv_hat_dy_hat * sigma_v / Ly

        #debug: print dh_dx, dh_dy, du_dx, du_dy, dv_dx, dv_dy
        #print(f"dh_dx: {dh_dx}")
        #print(f"dh_dy: {dh_dy}")
        #print(f"du_dx: {du_dx}")
        #print(f"du_dy: {du_dy}")
        #print(f"dv_dx: {dv_dx}")
        #print(f"dv_dy: {dv_dy}")
        
        if not self.bSteady:
            dh_dt = dh_hat_dt_hat * sigma_h / Lt
            du_dt = du_hat_dt_hat * sigma_u / Lt
            dv_dt = dv_hat_dt_hat * sigma_v / Lt
        
        # Mass conservation equation in dimensional space
        mass_residual = h*du_dx + u*dh_dx + h*dv_dy + v*dh_dy
        if not self.bSteady:
            mass_residual = dh_dt + mass_residual

        # Momentum conservation equations
        momentum_x_residual = u*du_dx + v*du_dy + self.g * dh_dx - self.g * Sx + self.g * ManningN**2 * u * u_mag / (h**(4/3) + 1e-8)
        if not self.bSteady:
            momentum_x_residual = du_dt + momentum_x_residual

        momentum_y_residual = u*dv_dx + v*dv_dy + self.g * dh_dy - self.g * Sy + self.g * ManningN**2 * v * u_mag / (h**(4/3) + 1e-8)
        if not self.bSteady:
            momentum_y_residual = dv_dt + momentum_y_residual
        
        # Scale the residuals based on physics scales
        mass_residual = mass_residual / self.velocity_scale
        momentum_x_residual = momentum_x_residual / self.velocity_scale**2 * self.length_scale
        momentum_y_residual = momentum_y_residual / self.velocity_scale**2 * self.length_scale

        #debug: print mass_residual, momentum_x_residual, momentum_y_residual
        print(f"DEBUG at line 329: velocity_scale type: {type(self.velocity_scale)}, value: {self.velocity_scale}, device: {self.velocity_scale.device if hasattr(self.velocity_scale, 'device') else 'N/A'}")
        print(f"DEBUG at line 329: length_scale type: {type(self.length_scale)}, value: {self.length_scale}, device: {self.length_scale.device if hasattr(self.length_scale, 'device') else 'N/A'}")
        print(f"velocity_scale: {self.velocity_scale.item() if self.velocity_scale.numel() == 1 else self.velocity_scale}")
        print(f"length_scale: {self.length_scale.item() if self.length_scale.numel() == 1 else self.length_scale}")
        print(f"mass_residual: {mass_residual}")
        print(f"momentum_x_residual: {momentum_x_residual}")
        print(f"momentum_y_residual: {momentum_y_residual}")

        #exit()
        
        return mass_residual, momentum_x_residual, momentum_y_residual, h, u, v
        
    def compute_pde_loss(self, branch_input, trunk_input, pde_data, deeponet_points_stats, pinn_points_stats):
        """
        Compute the PDE loss for the shallow water equations.
        
        Args:
            branch_input (torch.Tensor): Input function for branch net [batch_size, branch_input_dim] (normalized)
            trunk_input (torch.Tensor): Coordinates for trunk net [batch_size, trunk_input_dim] (normalized)
            pde_data (torch.Tensor): PDE data containing (zb, Sx, Sy, ManningN) [batch_size, 4] (not normalized).
            deeponet_points_stats (dict): Statistics of the DeepONet points for normalization.
            pinn_points_stats (dict): Statistics of the PINN points for normalization.
            
        Returns:
            tuple: (pde_loss, pde_loss_components, h, u, v)
        """
        mass_residual, momentum_x_residual, momentum_y_residual, h, u, v = self.compute_pde_residuals(
            branch_input, trunk_input, pde_data, deeponet_points_stats, pinn_points_stats
        )
        
        # Compute the loss for each equation
        continuity_loss = torch.mean(mass_residual**2)
        momentum_x_loss = torch.mean(momentum_x_residual**2)
        momentum_y_loss = torch.mean(momentum_y_residual**2)
        
        # Add small epsilon to prevent complete zero
        eps = torch.tensor(1e-8, device=self.device)
        
        # PDE loss is just the sum of the losses for each equation (no weighting implemented yet)
        pde_loss = continuity_loss + momentum_x_loss + momentum_y_loss + eps
        
        pde_loss_components = {
            'continuity_loss': continuity_loss.item(),
            'momentum_x_loss': momentum_x_loss.item(),
            'momentum_y_loss': momentum_y_loss.item()
        }
        
        return pde_loss, pde_loss_components, h, u, v
        
    def compute_deeponet_data_loss(self, branch_input, trunk_input, target):
        """
        Compute the DeepONet data loss. The loss is the MSE loss of the predictions and the target.

        Note: all data (branch_input, trunk_input, target) are already normalized. So the loss is normalized.
        
        Args:
            branch_input (torch.Tensor): Input function for branch net [batch_size, branch_input_dim].
            trunk_input (torch.Tensor): Coordinates for trunk net [batch_size, trunk_input_dim].
            target (torch.Tensor): Target values [batch_size, output_dim].
            
        Returns:
            tuple: (data_loss, loss_components)
        """
        # Get model predictions
        predictions = self.forward(branch_input, trunk_input)
        
        # Compute MSE loss
        eps = torch.tensor(1e-8, device=self.device)
        data_loss = torch.mean((predictions - target)**2) + eps
        
        loss_components = {
            'data_loss': data_loss.item()
        }
        
        return data_loss, loss_components
        
    def compute_deeponet_initial_loss(self, branch_input, initial_trunk_input, initial_values):
        """
        Compute the DeepONet initial condition loss.
        
        Args:
            branch_input (torch.Tensor): Input function for branch net [batch_size, branch_input_dim].
            initial_trunk_input (torch.Tensor): Coordinates at initial time [batch_size, trunk_input_dim].
            initial_values (torch.Tensor): True initial values [batch_size, output_dim].
            
        Returns:
            tuple: (initial_loss, h, u, v)
        """
        # Get model predictions at initial points
        predictions = self.forward(branch_input, initial_trunk_input)
        h, u, v = predictions[:, 0:1], predictions[:, 1:2], predictions[:, 2:3]
        
        # Clip h to be positive
        h = torch.clamp(h, min=1e-3)
        
        # Compute MSE loss
        eps = torch.tensor(1e-8, device=self.device)
        initial_loss = torch.mean((predictions - initial_values)**2) + eps
        
        return initial_loss, h, u, v
        
    def compute_deeponet_boundary_loss(self, branch_input, boundary_trunk_input, boundary_values):
        """
        Compute the DeepONet boundary condition loss.
        
        Note: This is a simplified boundary loss. For more complex boundary conditions
        (like inlet-q, exit-h, wall), a more sophisticated implementation similar to PINN
        would be needed. This can be implemented later when needed.
        
        Args:
            branch_input (torch.Tensor): Input function for branch net [batch_size, branch_input_dim].
            boundary_trunk_input (torch.Tensor): Coordinates at boundary [batch_size, trunk_input_dim].
            boundary_values (torch.Tensor): True boundary values [batch_size, output_dim].
            
        Returns:
            tuple: (boundary_loss, h, u, v)
        """
        # Get model predictions at boundary points
        predictions = self.forward(branch_input, boundary_trunk_input)
        h, u, v = predictions[:, 0:1], predictions[:, 1:2], predictions[:, 2:3]
        
        # Clip h to be positive
        h = torch.clamp(h, min=1e-3)
        
        # Compute MSE loss
        eps = torch.tensor(1e-8, device=self.device)
        boundary_loss = torch.mean((predictions - boundary_values)**2) + eps
        
        return boundary_loss, h, u, v
        
    def compute_total_loss(self, branch_input, trunk_input, target=None,
                          physics_branch_input=None, physics_trunk_input=None, pde_data=None,
                          all_deeponet_points_stats=None, all_pinn_points_stats=None):
        """
        Compute the total loss for the Physics-Informed DeepONet model.

        Currently, only DeepONet data loss and PINN PDE loss are supported. Other losses, such as boundary conditions, are not supported yet.
        
        Args:
            branch_input (torch.Tensor): Input function for branch net [batch_size, branch_input_dim].
            trunk_input (torch.Tensor): Coordinates for trunk net [batch_size, trunk_input_dim].
            target (torch.Tensor, optional): Target values for data loss [batch_size, output_dim].

            physics_branch_input (torch.Tensor, optional): Input function for PINN physics constraints. (normalized)
            physics_trunk_input (torch.Tensor, optional): Coordinates for PINN physics constraints. (normalized)
            pde_data (torch.Tensor, optional): PINN PDE data (zb, Sx, Sy, ManningN) [batch_size, 4]. (not normalized)           

            all_deeponet_points_stats (dict, optional): Statistics of DeepONet points for normalization.
            all_pinn_points_stats (dict, optional): Statistics of PINN points for normalization.

        Returns:
            tuple: (total_loss, loss_components)
        """
        # Safety check: Ensure loss weight buffers are not zero (they might have been reset)
        loss_weights_dict = self.loss_weights
        any_zero = any(w.item() == 0.0 for w in loss_weights_dict.values() if hasattr(w, 'item'))
        if any_zero:
            print(f"WARNING: Some loss weights are zero! Re-initializing from config...")
            # Re-initialize all loss weights from config
            if self.bSteady:
                self.register_buffer('loss_weight_pinn_data_loss', torch.tensor(self.config.get('training.loss_weights.pinn.data_loss', 1.0), dtype=torch.float32, device=self.device))
                self.register_buffer('loss_weight_pinn_pde_loss', torch.tensor(self.config.get('training.loss_weights.pinn.pde_loss', 1.0), dtype=torch.float32, device=self.device))
                self.register_buffer('loss_weight_pinn_boundary_loss', torch.tensor(self.config.get('training.loss_weights.pinn.boundary_loss', 1.0), dtype=torch.float32, device=self.device))
                self.register_buffer('loss_weight_deeponet_data_loss', torch.tensor(self.config.get('training.loss_weights.deeponet.data_loss', 1.0), dtype=torch.float32, device=self.device))
                self.register_buffer('loss_weight_deeponet_pinn_loss', torch.tensor(self.config.get('training.loss_weights.deeponet.pinn_loss', 1.0), dtype=torch.float32, device=self.device))
                self.register_buffer('loss_weight_deeponet_boundary_loss', torch.tensor(self.config.get('training.loss_weights.deeponet.boundary_loss', 1.0), dtype=torch.float32, device=self.device))
            else:
                self.register_buffer('loss_weight_pinn_data_loss', torch.tensor(self.config.get('training.loss_weights.pinn.data_loss', 1.0), dtype=torch.float32, device=self.device))
                self.register_buffer('loss_weight_pinn_pde_loss', torch.tensor(self.config.get('training.loss_weights.pinn.pde_loss', 1.0), dtype=torch.float32, device=self.device))
                self.register_buffer('loss_weight_pinn_boundary_loss', torch.tensor(self.config.get('training.loss_weights.pinn.boundary_loss', 1.0), dtype=torch.float32, device=self.device))
                self.register_buffer('loss_weight_pinn_initial_loss', torch.tensor(self.config.get('training.loss_weights.pinn.initial_loss', 1.0), dtype=torch.float32, device=self.device))
                self.register_buffer('loss_weight_deeponet_data_loss', torch.tensor(self.config.get('training.loss_weights.deeponet.data_loss', 1.0), dtype=torch.float32, device=self.device))
                self.register_buffer('loss_weight_deeponet_pinn_loss', torch.tensor(self.config.get('training.loss_weights.deeponet.pinn_loss', 1.0), dtype=torch.float32, device=self.device))
                self.register_buffer('loss_weight_deeponet_boundary_loss', torch.tensor(self.config.get('training.loss_weights.deeponet.boundary_loss', 1.0), dtype=torch.float32, device=self.device))
                self.register_buffer('loss_weight_deeponet_initial_loss', torch.tensor(self.config.get('training.loss_weights.deeponet.initial_loss', 1.0), dtype=torch.float32, device=self.device))
            print(f"  Re-initialized loss weights: {self.loss_weights}")
        
        # Initialize losses with requires_grad=True

        # Initialize losses for PINN        
        pinn_pde_loss = torch.zeros(1, device=self.device, requires_grad=True)       

        # Initialize losses for DeepONet
        deeponet_data_loss = torch.zeros(1, device=self.device, requires_grad=True)        

        # Initialize loss components
        loss_components = {
            'deeponet_data_loss': 0.0,
            'pinn_pde_loss': 0.0,
            'pinn_pde_loss_cty': 0.0,
            'pinn_pde_loss_mom_x': 0.0,
            'pinn_pde_loss_mom_y': 0.0,
            'pinn_initial_loss': 0.0,
            'pinn_boundary_loss': 0.0,
            'total_loss': 0.0
        }
        
        # Compute DeepONet data loss if target is provided
        if target is not None:
            deeponet_data_loss, deeponet_data_loss_components = self.compute_deeponet_data_loss(branch_input, trunk_input, target)
            loss_components['deeponet_data_loss'] = deeponet_data_loss.item()
            
        # Compute PDE loss if physics inputs are provided
        # if (physics_branch_input is not None and physics_trunk_input is not None and 
        #     pde_data is not None and all_pinn_points_stats is not None and all_deeponet_points_stats is not None):
        #     pinn_pde_loss, pinn_pde_loss_components, _, _, _ = self.compute_pde_loss(
        #         physics_branch_input, physics_trunk_input, pde_data, all_deeponet_points_stats, all_pinn_points_stats
        #     )

        #     loss_components['pinn_pde_loss'] = pinn_pde_loss.item()
        #     loss_components['pinn_pde_loss_cty'] = pinn_pde_loss_components['continuity_loss']
        #     loss_components['pinn_pde_loss_mom_x'] = pinn_pde_loss_components['momentum_x_loss']
        #     loss_components['pinn_pde_loss_mom_y'] = pinn_pde_loss_components['momentum_y_loss']
            
        # Compute total loss
        if self.bSteady:
            total_loss = (
                self.loss_weights['deeponet_data_loss'] * deeponet_data_loss +
                self.loss_weights['deeponet_pinn_loss'] * pinn_pde_loss
            )
        else:
            total_loss = (
                self.loss_weights['deeponet_data_loss'] * deeponet_data_loss +
                self.loss_weights['deeponet_pinn_loss'] * pinn_pde_loss
            )

        #debug: print loss weights
        #print(f"Loss weights: {self.loss_weights}")
        #exit()
        
        # Update loss components with total loss
        loss_components['total_loss'] = total_loss.item()        

        #debug: print loss components
        print(f"Loss components: {loss_components}")
        
        return total_loss, loss_components 