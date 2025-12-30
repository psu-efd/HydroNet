"""
Physics-Informed SWE_DeepONet (PI-SWE-DeepONet) model (with the option of turning on/off the physics-informed loss) for HydroNet.
"""
import torch
import torch.nn as nn

from ...utils.config import Config


class FCLayer(nn.Module):
    """
    Fully connected layer with activation and optional dropout.
    """

    def __init__(self, in_dim, out_dim, activation="relu", dropout_rate=0.0):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU(0.2)
        else:
            self.activation = nn.Identity()

        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, x):
        x = self.fc(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class BranchNet(nn.Module):
    """
    Branch network of DeepONet for encoding input functions.
    """

    def __init__(
        self,
        in_dim,
        out_dim,
        hidden_layers=None,
        activation="relu",
        dropout_rate=0.0,
    ):
        super().__init__()
        hidden_layers = hidden_layers or [128, 128, 128]
        layers = []
        layer_dims = [in_dim] + hidden_layers

        for i in range(len(layer_dims) - 1):
            layers.append(
                FCLayer(
                    layer_dims[i], layer_dims[i + 1], activation=activation, dropout_rate=dropout_rate
                )
            )

        # Add final layer with Xavier initialization (same as FCLayer)
        final_layer = nn.Linear(layer_dims[-1], out_dim)
        nn.init.xavier_normal_(final_layer.weight)
        nn.init.zeros_(final_layer.bias)
        layers.append(final_layer)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class TrunkNet(nn.Module):
    """
    Trunk network of DeepONet for encoding coordinates.
    """

    def __init__(
        self,
        in_dim,
        out_dim,
        hidden_layers=None,
        activation="relu",
        dropout_rate=0.0,
    ):
        super().__init__()
        hidden_layers = hidden_layers or [128, 128, 128]
        layers = []
        layer_dims = [in_dim] + hidden_layers

        for i in range(len(layer_dims) - 1):
            layers.append(
                FCLayer(
                    layer_dims[i], layer_dims[i + 1], activation=activation, dropout_rate=dropout_rate
                )
            )

        # Add final layer with Xavier initialization (same as FCLayer)
        final_layer = nn.Linear(layer_dims[-1], out_dim)
        nn.init.xavier_normal_(final_layer.weight)
        nn.init.zeros_(final_layer.bias)
        layers.append(final_layer)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class PI_SWE_DeepONetModel(nn.Module):
    """
    Unified DeepONet model with optional physics-informed loss computation.
    """

    def __init__(
        self,
        config: Config,
    ):
        super().__init__()
        if not isinstance(config, Config):
            raise ValueError("config must be a Config object.")

        self.config = config
        self.bSteady = self.config.get_required_config("physics.bSteady")
        self.use_physics_loss = self.config.get_required_config("model.use_physics_loss")
        self.SWE_form = self.config.get_required_config("model.SWE_form")
        self.bDryWet = self.config.get_required_config("model.bDryWet")
        # Ensure dry_wet_threshold is a float (YAML might read scientific notation as string)
        self.dry_wet_threshold = float(self.config.get_required_config("model.dry_wet_threshold"))

        if self.bDryWet and self.dry_wet_threshold < 0:
            raise ValueError("Dry/wet threshold must be positive if dry/wet points are being dealt with.")

        print(f"Use physics loss: {self.use_physics_loss}")
        print(f"SWE form: {self.SWE_form}")
        print(f"Whether deal with dry/wet points in pde loss: {self.bDryWet}")
        print(f"Dry/wet threshold (m): {self.dry_wet_threshold}")

        device_type = self.config.get_required_config("device.type")
        if device_type is not None:
            device_index = self.config.get_required_config("device.index")
            if device_type == "cuda" and torch.cuda.is_available():
                self.device = torch.device(f"cuda:{device_index}")
            else:
                self.device = torch.device("cpu")
        else:  #by default, the model will be on the CPU
            self.device = torch.device("cpu")

        # Required model configuration parameters
        branch_layers = self.config.get_required_config("model.branch_net.hidden_layers")
        branch_activation = self.config.get_required_config("model.branch_net.activation")
        branch_dropout = self.config.get_required_config("model.branch_net.dropout_rate")

        trunk_layers = self.config.get_required_config("model.trunk_net.hidden_layers")
        trunk_activation = self.config.get_required_config("model.trunk_net.activation")
        trunk_dropout = self.config.get_required_config("model.trunk_net.dropout_rate")

        self.branch_input_dim = self.config.get_required_config("model.branch_net.branch_input_dim")
        self.trunk_input_dim = self.config.get_required_config("model.trunk_net.trunk_input_dim")
        self.output_dim = self.config.get_required_config("model.output_dim")

        if branch_layers[-1] != trunk_layers[-1]:
            raise ValueError(
                f"Branch output dimension mismatch: {branch_layers[-1]} != {trunk_layers[-1]}"
            )

        self.hidden_dim = branch_layers[-1]

        self.branch_net = None
        if self.branch_input_dim > 0:
            self.branch_net = BranchNet(
                self.branch_input_dim,
                self.hidden_dim,
                branch_layers,
                branch_activation,
                branch_dropout,
            )

        self.trunk_net = TrunkNet(
            self.trunk_input_dim,
            self.hidden_dim,
            trunk_layers,
            trunk_activation,
            trunk_dropout,
        )

        # Check if hidden_dim is divisible by output_dim
        if self.hidden_dim % self.output_dim != 0:
            raise ValueError(f"hidden_dim {self.hidden_dim} must be divisible by output_dim {self.output_dim}")

        self.bias = nn.Parameter(torch.zeros(self.output_dim))

        self._init_physics_buffers()
        self._init_loss_weight_buffers()

        self.to(self.device)

    def set_use_physics_loss(self, flag: bool):
        """
        Enable or disable physics-informed loss terms after initialization.
        """
        self.use_physics_loss = bool(flag)


    def _init_physics_buffers(self):
        """
        Initialize the physics buffers.

        Explaining PyTorch buffers and why they're used here:

        ## PyTorch Buffers Explained

        **Buffers** are non-trainable tensors that are part of the model's state. They differ from **Parameters** (trainable weights/biases) in that they don't receive gradients.

        ### Why Use Buffers?

        1. Automatic device management: When you call `model.to(device)`, buffers move with the model (CPU ↔ GPU).
        2. State persistence: Buffers are included in `state_dict()`, so they're saved/loaded with checkpoints.
        3. No gradients: They're excluded from backpropagation and optimizer updates.
        4. Model state: They're part of the model's persistent state, not just temporary variables.

        ### In Our Code

        In `_init_physics_buffers()`, you're storing:

        ```python
        self.register_buffer("g", ...)              # Gravity constant (9.81 m/s²)
        self.register_buffer("length_scale", ...)   # Physical length scale for normalization
        self.register_buffer("velocity_scale", ...) # Physical velocity scale for normalization
        ```

        These are physics constants that should:
        - Move to GPU when the model moves to GPU
        - Be saved in checkpoints (so you can reload the model with the same physics parameters)
        - Not be updated during training (they're constants, not learnable parameters)
        - Be accessible as `self.g`, `self.length_scale`, etc., throughout the model

        ### Alternative (Without Buffers)

        If we used regular attributes like `self.g = torch.tensor(9.81)`:
        - They wouldn't automatically move to GPU with `model.to(device)`
        - They wouldn't be saved in `state_dict()` (checkpoint loading would lose them)
        - You'd have to manually manage device placement

        Buffers handle this automatically, which is why they're perfect for constants and configuration values that need to persist with the model.
        """
        g_value = float(self.config.get_required_config("physics.gravity"))
        length_scale_value = self.config.require_positive("physics.scales.length", 1.0, self.use_physics_loss)
        velocity_scale_value = self.config.require_positive("physics.scales.velocity", 1.0, self.use_physics_loss)
        
        # Create buffers without specifying device - they will be moved by self.to(device) later
        self.register_buffer("g", torch.tensor(g_value, dtype=torch.float32))
        self.register_buffer("length_scale", torch.tensor(length_scale_value, dtype=torch.float32))
        self.register_buffer("velocity_scale", torch.tensor(velocity_scale_value, dtype=torch.float32))

    def _init_loss_weight_buffers(self):
        """
        Initialize the loss weight buffers.
        """
        def to_tensor(val):
            return torch.tensor(float(val), dtype=torch.float32)

        deeponet_data_loss_val = self.config.get_required_config("training.loss_weights.deeponet.data_loss")
        deeponet_pinn_loss_val = self.config.get_required_config("training.loss_weights.deeponet.pinn_loss")

        self.register_buffer(
            "loss_weight_pinn_data_loss",
            to_tensor(self.config.get_required_config("training.loss_weights.pinn.data_loss")),
        )
        self.register_buffer(
            "loss_weight_pinn_pde_loss",
            to_tensor(self.config.get_required_config("training.loss_weights.pinn.pde_loss")),
        )
        self.register_buffer(
            "loss_weight_pinn_boundary_loss",
            to_tensor(self.config.get_required_config("training.loss_weights.pinn.boundary_loss")),
        )

        self.register_buffer("loss_weight_deeponet_data_loss", to_tensor(deeponet_data_loss_val))
        self.register_buffer("loss_weight_deeponet_pinn_loss", to_tensor(deeponet_pinn_loss_val))
        self.register_buffer(
            "loss_weight_deeponet_boundary_loss",
            to_tensor(self.config.get_required_config("training.loss_weights.deeponet.boundary_loss")),
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

        if not self.bSteady:
            self.register_buffer(
                "loss_weight_pinn_initial_loss",
                to_tensor(self.config.get_required_config("training.loss_weights.pinn.initial_loss")),
            )
            self.register_buffer(
                "loss_weight_deeponet_initial_loss",
                to_tensor(self.config.get_required_config("training.loss_weights.deeponet.initial_loss")),
            )

    def load_state_dict(self, state_dict, strict=True):
        """
        Override load_state_dict to always use loss weights from config file, regardless of checkpoint.
        Physics buffers (g, length_scale, velocity_scale) are still loaded from checkpoint if present.
        """
        # Store current buffer values before loading (these come from config during initialization)
        loss_weight_backup = {}
        physics_buffer_backup = {}
        for name, buffer in self.named_buffers():
            if name.startswith('loss_weight_'):
                # Always preserve loss weights from config
                loss_weight_backup[name] = buffer.clone()
            elif name in ['length_scale', 'velocity_scale', 'g']:
                # Preserve physics buffers as backup (in case they're missing from checkpoint)
                physics_buffer_backup[name] = buffer.clone()
        
        # Remove loss weight buffers from state_dict so they won't be loaded from checkpoint
        state_dict_filtered = {k: v for k, v in state_dict.items() if not k.startswith('loss_weight_')}
        
        # Load the state_dict (without loss weight buffers) - use strict=False to allow missing loss_weight buffers
        missing_keys, unexpected_keys = super().load_state_dict(state_dict_filtered, strict=False)
        
        # Filter out loss_weight buffers from missing_keys (they're intentionally excluded and will be restored from config)
        missing_keys_filtered = [k for k in missing_keys if not k.startswith('loss_weight_')]
        
        # Always restore loss weight buffers from config (backup)
        for name, backup_value in loss_weight_backup.items():
            self.register_buffer(name, backup_value)
        
        # Restore physics buffers if they were missing from the loaded state_dict
        for name, backup_value in physics_buffer_backup.items():
            if name not in state_dict:
                # Buffer was missing, restore from backup
                self.register_buffer(name, backup_value)
                print(f"Warning: Buffer '{name}' was missing from state_dict, preserving current value: {backup_value.item() if backup_value.numel() == 1 else backup_value}")
        
        # If strict mode and there are missing keys (excluding loss_weight buffers), raise error
        if strict and missing_keys_filtered:
            raise RuntimeError(
                f"Missing key(s) in state_dict: {', '.join(sorted(missing_keys_filtered))}"
            )
        
        return missing_keys_filtered, unexpected_keys

    @property
    def loss_weights(self):
        """
        Property that returns loss weights as a dictionary for convenient access.
        The actual weights are stored as registered buffers.
        """
        base_dict = {
            "pinn_data_loss": self.loss_weight_pinn_data_loss,
            "pinn_pde_loss": self.loss_weight_pinn_pde_loss,
            "pinn_boundary_loss": self.loss_weight_pinn_boundary_loss,
            "deeponet_data_loss": self.loss_weight_deeponet_data_loss,
            "deeponet_pinn_loss": self.loss_weight_deeponet_pinn_loss,
            "deeponet_boundary_loss": self.loss_weight_deeponet_boundary_loss,
            "pde_continuity": self.loss_weight_pde_continuity,
            "pde_momentum_x": self.loss_weight_pde_momentum_x,
            "pde_momentum_y": self.loss_weight_pde_momentum_y,
        }
        
        if self.bSteady:
            return base_dict
        else:
            base_dict.update({
                "pinn_initial_loss": self.loss_weight_pinn_initial_loss,
                "deeponet_initial_loss": self.loss_weight_deeponet_initial_loss,
            })
            return base_dict

    def check_model_input_output_dimensions(self, branch_input_dim, trunk_input_dim, output_dim):
        if self.branch_input_dim > 0 and branch_input_dim != self.branch_input_dim:
            raise ValueError(
                f"Branch input dimension mismatch: {branch_input_dim} != {self.branch_input_dim}"
            )
        if trunk_input_dim != self.trunk_input_dim:
            raise ValueError(
                f"Trunk input dimension mismatch: {trunk_input_dim} != {self.trunk_input_dim}"
            )
        if output_dim != self.output_dim:
            raise ValueError(f"Output dimension mismatch: {output_dim} != {self.output_dim}")
        #print("Model input and output dimensions are consistent with the specified configuration.")

    def forward(self, branch_input, trunk_input, deeponet_points_stats):
        """
        Forward pass of the model. 

        Tried wet/dry points correction here, but the training stops earlier than no correction. More work is needed here.
        """
        if self.branch_net is None:
            raise ValueError(
                "Branch network not initialized. Set branch_input_dim first using set_branch_input_dim()."
            )

        branch_output = self.branch_net(branch_input)
        trunk_output = self.trunk_net(trunk_input)
        
        # Debug: Check shapes
        if branch_output.shape != trunk_output.shape:
            raise ValueError(
                f"Branch and trunk outputs must have the same shape. "
                f"Got branch_output.shape={branch_output.shape}, trunk_output.shape={trunk_output.shape}. "
                f"branch_input.shape={branch_input.shape}, trunk_input.shape={trunk_input.shape}"
            )

        outputs = []
        for i in range(self.output_dim):
            branch_i = branch_output[:, i :: self.output_dim]
            trunk_i = trunk_output[:, i :: self.output_dim]
            if branch_i.shape != trunk_i.shape:
                raise ValueError(
                    f"After slicing, branch_i.shape={branch_i.shape} != trunk_i.shape={trunk_i.shape}. "
                    f"i={i}, output_dim={self.output_dim}, branch_output.shape={branch_output.shape}, trunk_output.shape={trunk_output.shape}"
                )
            output_i = torch.sum(branch_i * trunk_i, dim=1, keepdim=True) + self.bias[i]
            outputs.append(output_i)

        return torch.cat(outputs, dim=1)


        # branch_output = self.branch_net(branch_input)
        # trunk_output = self.trunk_net(trunk_input)

        # #get the stats of the output
        # mu_h = deeponet_points_stats['all_data_stats']['h_mean']
        # sigma_h = deeponet_points_stats['all_data_stats']['h_std']
        # mu_u = deeponet_points_stats['all_data_stats']['u_mean']
        # sigma_u = deeponet_points_stats['all_data_stats']['u_std']
        # mu_v = deeponet_points_stats['all_data_stats']['v_mean']
        # sigma_v = deeponet_points_stats['all_data_stats']['v_std']

        # outputs = []
        
        # # Process all outputs first to get denormalized values
        # denormalized_outputs = []
        # for i in range(self.output_dim):
        #     branch_i = branch_output[:, i :: self.output_dim]
        #     trunk_i = trunk_output[:, i :: self.output_dim]
        #     output_i = torch.sum(branch_i * trunk_i, dim=1, keepdim=True) + self.bias[i]

        #     #denormalize the output_i by the stats
        #     if i == 0: #h
        #         output_i_denormalized = output_i * sigma_h + mu_h
        #     elif i == 1: #u
        #         output_i_denormalized = output_i * sigma_u + mu_u
        #     elif i == 2: #v
        #         output_i_denormalized = output_i * sigma_v + mu_v
        #     else:
        #         raise ValueError(f"Invalid output index: {i}")
            
        #     denormalized_outputs.append(output_i_denormalized)
        
        # # Check dry/wet points based on water depth (h, index 0)
        # # flag for water depth above a small threshold:
        # # if below, set the water depth to be the small threshold and the velocity to be zero
        # h_denormalized = denormalized_outputs[0]
        # b_h_positive = h_denormalized > 1e-3
        
        # # Apply dry/wet corrections
        # for i in range(self.output_dim):
        #     output_i_denormalized = denormalized_outputs[i]
            
        #     if i == 0: #h
        #         # For dry points, set water depth to minimum threshold
        #         output_i_denormalized[~b_h_positive] = 1e-3
        #     elif i == 1: #u
        #         # For dry points, set velocity to zero
        #         output_i_denormalized[~b_h_positive] = 0.0
        #     elif i == 2: #v
        #         # For dry points, set velocity to zero
        #         output_i_denormalized[~b_h_positive] = 0.0
        #     else:
        #         raise ValueError(f"Invalid output index: {i}")

        #     #now normalize the output_i_denormalized by the stats
        #     if i == 0: #h
        #         output_i_normalized = (output_i_denormalized - mu_h) / sigma_h
        #     elif i == 1: #u
        #         output_i_normalized = (output_i_denormalized - mu_u) / sigma_u
        #     elif i == 2: #v
        #         output_i_normalized = (output_i_denormalized - mu_v) / sigma_v
        #     else:
        #         raise ValueError(f"Invalid output index: {i}")

        #     outputs.append(output_i_normalized)


        # return torch.cat(outputs, dim=1)

    def set_branch_input_dim(self, dim):
        """
        Set the input dimension for the branch network and recreate it.
        
        Args:
            dim (int): New input dimension.
        """
        self.branch_input_dim = dim
        
        # Get branch network parameters from config (required)
        branch_layers = self.config.get_required_config('model.branch_net.hidden_layers')
        branch_activation = self.config.get_required_config('model.branch_net.activation')
        branch_dropout = self.config.get_required_config('model.branch_net.dropout_rate')
        
        # Recreate branch network with new input dimension
        self.branch_net = BranchNet(
            self.branch_input_dim,
            self.hidden_dim,
            branch_layers,
            branch_activation,
            branch_dropout,
        ).to(self.device)
        
    def compute_pde_residuals_non_conservative(self, branch_input, trunk_input, pde_data, deeponet_points_stats, pinn_points_stats):
        """
        Compute the residuals of the non-conservative form of the PDEs (shallow water equations).
        
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
        
        
        # Ensure that gradients are computed
        trunk_input = trunk_input.clone().detach().requires_grad_(True)
        
        # Forward pass to get h, u, v
        output = self.forward(branch_input, trunk_input, deeponet_points_stats)
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

        if not self.bSteady:
            dh_hat_dt_hat = h_hat_grad[:, 2:3]
            du_hat_dt_hat = u_hat_grad[:, 2:3]
            dv_hat_dt_hat = v_hat_grad[:, 2:3]

        # Get normalization stats (note: the inputs and outputs are always normalized)
        # Normalization stats for the output (h, u, v) of the DeepONet.
        mu_h = deeponet_points_stats['all_data_stats']['h_mean']
        sigma_h = deeponet_points_stats['all_data_stats']['h_std']
        mu_u = deeponet_points_stats['all_data_stats']['u_mean']
        sigma_u = deeponet_points_stats['all_data_stats']['u_std']
        mu_v = deeponet_points_stats['all_data_stats']['v_mean']
        sigma_v = deeponet_points_stats['all_data_stats']['v_std']        
        
        # Normalization stats for the coordinates (x, y, t) of PINN.
        x_min = pinn_points_stats['all_points_stats']['x_min']
        x_max = pinn_points_stats['all_points_stats']['x_max']
        y_min = pinn_points_stats['all_points_stats']['y_min']
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

        #flag dry points based on water depth: threshold 
        #flag = 1.0 for wet points, 0.0 for dry points
        #flag value is used to exclude the pde residuals at dry points from the loss calculation
        flag = torch.where(h > self.dry_wet_threshold, torch.ones_like(h), torch.zeros_like(h))

        #clip water depth to be positive
        h = torch.clamp(h, min=self.dry_wet_threshold)
        
        #denormalize the coordinates (which are normalized with min-max) (NOT USED FOR NOW)
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

        #exclude the pde residuals at dry points from the loss calculation
        if self.bDryWet:
            mass_residual = mass_residual * flag
            momentum_x_residual = momentum_x_residual * flag
            momentum_y_residual = momentum_y_residual * flag

        #debug print (all tensors are batched, so print first element and statistics)
        bDebug = False
        if bDebug:
            #print(f"zb shape: {zb.shape}, first: {zb[0].item():.6f}, mean: {zb.mean().item():.6f}, min: {zb.min().item():.6f}, max: {zb.max().item():.6f}")
            #print(f"Sx shape: {Sx.shape}, first: {Sx[0].item():.6f}, mean: {Sx.mean().item():.6f}, min: {Sx.min().item():.6f}, max: {Sx.max().item():.6f}")
            #print(f"Sy shape: {Sy.shape}, first: {Sy[0].item():.6f}, mean: {Sy.mean().item():.6f}, min: {Sy.min().item():.6f}, max: {Sy.max().item():.6f}")
            #print(f"ManningN shape: {ManningN.shape}, first: {ManningN[0].item():.6f}, mean: {ManningN.mean().item():.6f}, min: {ManningN.min().item():.6f}, max: {ManningN.max().item():.6f}")
            print(f"h shape: {h.shape}, first: {h[0].item():.6f}, mean: {h.mean().item():.6f}, min: {h.min().item():.6f}, max: {h.max().item():.6f}")
            print(f"u shape: {u.shape}, first: {u[0].item():.6f}, mean: {u.mean().item():.6f}, min: {u.min().item():.6f}, max: {u.max().item():.6f}")
            print(f"v shape: {v.shape}, first: {v[0].item():.6f}, mean: {v.mean().item():.6f}, min: {v.min().item():.6f}, max: {v.max().item():.6f}")
            print(f"mass_residual shape: {mass_residual.shape}, first: {mass_residual[0].item():.6f}, mean: {mass_residual.mean().item():.6f}, min: {mass_residual.min().item():.6f}, max: {mass_residual.max().item():.6f}")
            print(f"momentum_x_residual shape: {momentum_x_residual.shape}, first: {momentum_x_residual[0].item():.6f}, mean: {momentum_x_residual.mean().item():.6f}, min: {momentum_x_residual.min().item():.6f}, max: {momentum_x_residual.max().item():.6f}")
            print(f"momentum_y_residual shape: {momentum_y_residual.shape}, first: {momentum_y_residual[0].item():.6f}, mean: {momentum_y_residual.mean().item():.6f}, min: {momentum_y_residual.min().item():.6f}, max: {momentum_y_residual.max().item():.6f}")
            print(f"dh_dx shape: {dh_dx.shape}, first: {dh_dx[0].item():.6f}, mean: {dh_dx.mean().item():.6f}")
            print(f"dh_dy shape: {dh_dy.shape}, first: {dh_dy[0].item():.6f}, mean: {dh_dy.mean().item():.6f}")
            print(f"du_dx shape: {du_dx.shape}, first: {du_dx[0].item():.6f}, mean: {du_dx.mean().item():.6f}")
            print(f"du_dy shape: {du_dy.shape}, first: {du_dy[0].item():.6f}, mean: {du_dy.mean().item():.6f}")
            print(f"dv_dx shape: {dv_dx.shape}, first: {dv_dx[0].item():.6f}, mean: {dv_dx.mean().item():.6f}")
            print(f"dv_dy shape: {dv_dy.shape}, first: {dv_dy[0].item():.6f}, mean: {dv_dy.mean().item():.6f}")
            if not self.bSteady:
                print(f"dh_dt shape: {dh_dt.shape}, first: {dh_dt[0].item():.6f}, mean: {dh_dt.mean().item():.6f}")
                print(f"du_dt shape: {du_dt.shape}, first: {du_dt[0].item():.6f}, mean: {du_dt.mean().item():.6f}")
                print(f"dv_dt shape: {dv_dt.shape}, first: {dv_dt[0].item():.6f}, mean: {dv_dt.mean().item():.6f}")

            #print the components of the momentum conservation equations: first element and mean, min, max, std
            u_du_dx = u * du_dx
            v_du_dy = v * du_dy
            g_dh_dx = self.g * dh_dx
            g_sx = self.g * Sx
            friction_term_x = self.g * ManningN**2 * u * u_mag / (h**(4.0/3.0) + 1e-8)
            u_dv_dx = u * dv_dx
            v_dv_dy = v * dv_dy
            g_dh_dy = self.g * dh_dy
            g_sy = self.g * Sy
            friction_term_y = self.g * ManningN**2 * v * u_mag / (h**(4.0/3.0) + 1e-8)
            
            print(f"u*du_dx: {u_du_dx[0].item():.6f}, mean: {u_du_dx.mean().item():.6f}, min: {u_du_dx.min().item():.6f}, max: {u_du_dx.max().item():.6f}, std: {u_du_dx.std().item():.6f}")
            print(f"v*du_dy: {v_du_dy[0].item():.6f}, mean: {v_du_dy.mean().item():.6f}, min: {v_du_dy.min().item():.6f}, max: {v_du_dy.max().item():.6f}, std: {v_du_dy.std().item():.6f}")
            print(f"self.g * dh_dx: {g_dh_dx[0].item():.6f}, mean: {g_dh_dx.mean().item():.6f}, min: {g_dh_dx.min().item():.6f}, max: {g_dh_dx.max().item():.6f}, std: {g_dh_dx.std().item():.6f}")
            print(f"self.g * Sx: {g_sx[0].item():.6f}, mean: {g_sx.mean().item():.6f}, min: {g_sx.min().item():.6f}, max: {g_sx.max().item():.6f}, std: {g_sx.std().item():.6f}")
            print(f"self.g * ManningN**2 * u * u_mag / (h**(4.0/3.0) + 1e-8): {friction_term_x[0].item():.6f}, mean: {friction_term_x.mean().item():.6f}, min: {friction_term_x.min().item():.6f}, max: {friction_term_x.max().item():.6f}, std: {friction_term_x.std().item():.6f}")
            print(f"u*dv_dx: {u_dv_dx[0].item():.6f}, mean: {u_dv_dx.mean().item():.6f}, min: {u_dv_dx.min().item():.6f}, max: {u_dv_dx.max().item():.6f}, std: {u_dv_dx.std().item():.6f}")
            print(f"v*dv_dy: {v_dv_dy[0].item():.6f}, mean: {v_dv_dy.mean().item():.6f}, min: {v_dv_dy.min().item():.6f}, max: {v_dv_dy.max().item():.6f}, std: {v_dv_dy.std().item():.6f}")
            print(f"self.g * dh_dy: {g_dh_dy[0].item():.6f}, mean: {g_dh_dy.mean().item():.6f}, min: {g_dh_dy.min().item():.6f}, max: {g_dh_dy.max().item():.6f}, std: {g_dh_dy.std().item():.6f}")
            print(f"self.g * Sy: {g_sy[0].item():.6f}, mean: {g_sy.mean().item():.6f}, min: {g_sy.min().item():.6f}, max: {g_sy.max().item():.6f}, std: {g_sy.std().item():.6f}")
            print(f"u*dv_dx: {u_dv_dx[0].item():.6f}, v*dv_dy: {v_dv_dy[0].item():.6f}, self.g * dh_dy: {g_dh_dy[0].item():.6f}, self.g * Sy: {g_sy[0].item():.6f}, self.g * ManningN**2 * v * u_mag / (h**(4.0/3.0) + 1e-8): {friction_term_y[0].item():.6f}")

            # Lx, Ly, Lt, mu_*, sigma_* are Python floats from stats dict, not tensors
            print(f"Lx: {Lx}, Ly: {Ly}" + (f", Lt: {Lt}" if not self.bSteady else ""))
            print(f"sigma_h: {sigma_h}, sigma_u: {sigma_u}, sigma_v: {sigma_v}")
            print(f"mu_h: {mu_h}, mu_u: {mu_u}, mu_v: {mu_v}")
            exit()


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
        # Compute the PDE residuals based on the SWE form
        if self.SWE_form == "conservative":  #Not correct yet; the formulas are not correct
            raise NotImplementedError("Conservative SWE form is not correctly implemented yet")

            #mass_residual, momentum_x_residual, momentum_y_residual, h, u, v = self.compute_pde_residuals_conservative(
            #    branch_input, trunk_input, pde_data, deeponet_points_stats, pinn_points_stats
            #)
        elif self.SWE_form == "non-conservative":
            mass_residual, momentum_x_residual, momentum_y_residual, h, u, v = self.compute_pde_residuals_non_conservative(
                branch_input, trunk_input, pde_data, deeponet_points_stats, pinn_points_stats
            )
        else:
            raise ValueError(f"Invalid SWE form: {self.SWE_form}")
        
        # Compute the loss for each equation
        continuity_loss = torch.mean(mass_residual**2)
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
        
    def compute_deeponet_data_loss(self, branch_input, trunk_input, target, deeponet_points_stats):
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
        predictions = self.forward(branch_input, trunk_input, deeponet_points_stats)
        
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
        
        # Initialize losses with requires_grad=True
        # Initialize loss for DeepONet
        deeponet_data_loss = torch.zeros(1, device=self.device, requires_grad=True)

        # Initialize loss for PINN
        pinn_pde_loss = torch.zeros(1, device=self.device, requires_grad=True)

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
            deeponet_data_loss, deeponet_data_loss_components = self.compute_deeponet_data_loss(branch_input, trunk_input, target, all_deeponet_points_stats)
            loss_components['deeponet_data_loss'] = deeponet_data_loss.item()
            

        # Compute PINN PDE loss if physics-informed loss is enabled and all required inputs are provided
        # If physics loss is enabled but inputs are missing (e.g., no PDE points configured), skip physics loss
        if self.use_physics_loss:
            if physics_branch_input is not None and physics_trunk_input is not None and pde_data is not None and all_pinn_points_stats is not None and all_deeponet_points_stats is not None:
                pinn_pde_loss, pinn_pde_loss_components, _, _, _ = self.compute_pde_loss(
                    physics_branch_input, physics_trunk_input, pde_data, all_deeponet_points_stats, all_pinn_points_stats
                )

                loss_components['pinn_pde_loss'] = pinn_pde_loss.item()
                loss_components['pinn_pde_loss_cty'] = pinn_pde_loss_components['continuity_loss']
                loss_components['pinn_pde_loss_mom_x'] = pinn_pde_loss_components['momentum_x_loss']
                loss_components['pinn_pde_loss_mom_y'] = pinn_pde_loss_components['momentum_y_loss']
            else:
                # Physics loss is enabled but inputs are missing - skip physics loss
                # This can happen if PDE points are not configured for the problem
                pinn_pde_loss = torch.zeros(1, device=self.device, requires_grad=True)        
            
        # Compute total loss
        total_loss = self.loss_weights['deeponet_data_loss'] * deeponet_data_loss
        if self.use_physics_loss and physics_branch_input is not None and physics_trunk_input is not None and pde_data is not None:
            total_loss = total_loss + self.loss_weights['deeponet_pinn_loss'] * pinn_pde_loss

        # Update loss components with total loss
        loss_components['total_loss'] = total_loss.item()

        return total_loss, loss_components 