"""
DeepONet model architecture for HydroNet.
"""
import torch
import torch.nn as nn
import numpy as np
from ...utils.config import Config


class FCLayer(nn.Module):
    """
    Fully connected layer with activation and optional dropout.
    """
    def __init__(self, in_dim, out_dim, activation='relu', dropout_rate=0.0):
        """
        Initialize the fully connected layer.
        
        Args:
            in_dim (int): Input dimension.
            out_dim (int): Output dimension.
            activation (str, optional): Activation function.
            dropout_rate (float, optional): Dropout rate.
        """
        super(FCLayer, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        
        # Initialize weights
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        
        # Set activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2)
        else:
            self.activation = nn.Identity()
            
        # Set dropout
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        
    def forward(self, x):
        """Forward pass."""
        x = self.fc(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class BranchNet(nn.Module):
    """
    Branch network of DeepONet for encoding input functions.
    """
    def __init__(self, in_dim, out_dim, hidden_layers=[128, 128, 128], activation='relu', dropout_rate=0.0):
        """
        Initialize the branch network.
        
        Args:
            in_dim (int): Input dimension.
            out_dim (int): Output dimension.
            hidden_layers (list, optional): Hidden layer dimensions.
            activation (str, optional): Activation function.
            dropout_rate (float, optional): Dropout rate.
        """
        super(BranchNet, self).__init__()
        
        layers = []
        layer_dims = [in_dim] + hidden_layers
        
        # Construct hidden layers
        for i in range(len(layer_dims) - 1):
            layers.append(FCLayer(layer_dims[i], layer_dims[i+1], activation, dropout_rate))
            
        # Output layer
        layers.append(nn.Linear(layer_dims[-1], out_dim))
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        """Forward pass."""
        return self.net(x)


class TrunkNet(nn.Module):
    """
    Trunk network of DeepONet for encoding coordinates.
    """
    def __init__(self, in_dim, out_dim, hidden_layers=[128, 128, 128], activation='relu', dropout_rate=0.0):
        """
        Initialize the trunk network.
        
        Args:
            in_dim (int): Input dimension (typically coordinate dimension).
            out_dim (int): Output dimension.
            hidden_layers (list, optional): Hidden layer dimensions.
            activation (str, optional): Activation function.
            dropout_rate (float, optional): Dropout rate.
        """
        super(TrunkNet, self).__init__()
        
        layers = []
        layer_dims = [in_dim] + hidden_layers
        
        # Construct hidden layers
        for i in range(len(layer_dims) - 1):
            layers.append(FCLayer(layer_dims[i], layer_dims[i+1], activation, dropout_rate))
            
        # Output layer
        layers.append(nn.Linear(layer_dims[-1], out_dim))
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        """Forward pass."""
        return self.net(x)


class DeepONetModel(nn.Module):
    """
    DeepONet model for learning the operator of shallow water equations.
    """
    def __init__(self, config_file=None, config=None):
        """
        Initialize the DeepONet model.
        
        Args:
            config_file (str, optional): Path to configuration file.
            config (Config, optional): Configuration object.
        """
        super(DeepONetModel, self).__init__()
        
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
        trunk_activation = self.config.get('model.trunk_net.activation', 'relu')
        trunk_dropout = self.config.get('model.trunk_net.dropout_rate', 0.0)
        
        # Input dimensions - use defaults if not specified
        self.branch_input_dim = self.config.get('data.branch_input_dim', 10)  # Default to 10 for dummy data
        self.trunk_input_dim = self.config.get('data.trunk_input_dim', 2)  # 2 for (x, y) coordinates, 3 for (x, y, t) coordinates
        
        # Output dimension - 3 for (h, u, v) in shallow water equations
        self.output_dim = self.config.get('model.output_dim', 3)
        
        # Hidden dimension for the DeepONet architecture
        self.hidden_dim = branch_layers[-1]
        
        # Create branch and trunk networks
        self.branch_net = BranchNet(
            self.branch_input_dim, 
            self.hidden_dim, 
            branch_layers, 
            branch_activation, 
            branch_dropout
        )
        
        self.trunk_net = TrunkNet(
            self.trunk_input_dim, 
            self.hidden_dim, 
            trunk_layers, 
            trunk_activation, 
            trunk_dropout
        )
        
        # Output bias for each output component
        self.bias = nn.Parameter(torch.zeros(self.output_dim))

    def check_model_input_output_dimensions(self, branch_input_dim, trunk_input_dim, output_dim):
        """
        Check if the model input and output dimensions are consistent with the configuration.
        """
        if branch_input_dim != self.branch_input_dim:
            raise ValueError(f"Branch input dimension mismatch: {branch_input_dim} != {self.branch_input_dim}")
        
        if trunk_input_dim != self.trunk_input_dim:
            raise ValueError(f"Trunk input dimension mismatch: {trunk_input_dim} != {self.trunk_input_dim}")
        
        if output_dim != self.output_dim:
            raise ValueError(f"Output dimension mismatch: {output_dim} != {self.output_dim}")
        
        print("Model input and output dimensions are consistent with the specified configuration.")
            
        
    def forward(self, branch_input, trunk_input):
        """
        Forward pass of DeepONet.
        
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
        
