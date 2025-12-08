"""
Configuration utilities for HydroNet.
"""
import os
import yaml
import torch


class Config:
    """
    Configuration class for loading and accessing configuration parameters.
    """
    def __init__(self, config_path=None):
        """
        Initialize the Config object.
        
        Args:
            config_path (str, optional): Path to the YAML configuration file.
                If None, an empty config is created.
        """
        self.config = {}
        if config_path is not None:
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Config file not found: {config_path}")
                
            with open(config_path, 'r') as file:
                self.config = yaml.safe_load(file)
                
    def get(self, key, default=None):
        """
        Get a configuration value.
        
        Args:
            key (str): The configuration key, can be nested using dots (e.g., 'model.hidden_layers').
            default: Default value to return if the key is not found.
            
        Returns:
            The configuration value or the default if not found.
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
                
        return value
        
    def set(self, key, value):
        """
        Set a configuration value.
        
        Args:
            key (str): The configuration key, can be nested using dots (e.g., 'model.hidden_layers').
            value: The value to set.
        """
        keys = key.split('.')
        d = self.config
        
        for k in keys[:-1]:
            if k not in d:
                d[k] = {}
            d = d[k]
            
        d[keys[-1]] = value
        
    def get_device(self):
        """
        Get the device to use (CPU or GPU).
        
        Returns:
            torch.device: The device to use.
        """
        device_type = self.get('device.type', 'cpu')
        device_index = self.get('device.index', 0)
        
        if device_type == 'cuda' and torch.cuda.is_available():
            return torch.device(f'cuda:{device_index}')
        else:
            return torch.device('cpu')
    
    def get_required_config(self, key: str):
        """
        Get a required configuration value, raising an error if it's missing.
        
        Args:
            key: Configuration key (e.g., "model.branch_net.hidden_layers")
            
        Returns:
            The configuration value
            
        Raises:
            ValueError: If the configuration key is missing
        """
        value = self.get(key, None)
        if value is None:
            raise ValueError(f"Required configuration parameter '{key}' is missing from config.")
        return value
    
    def require_positive(self, key: str, default: float, use_physics_loss: bool = False) -> float:
        """
        Get a configuration value that must be positive, with optional validation.
        
        Args:
            key: Configuration key (e.g., "physics.scales.length")
            default: Default value to return if the key is missing or invalid
            use_physics_loss: If True, raises an error when physics loss is enabled and value is missing or non-positive
            
        Returns:
            The configuration value (as float), or default if missing/invalid and validation is not enforced
            
        Raises:
            ValueError: If use_physics_loss is True and the value is missing or non-positive
        """
        value = self.get(key, None)
        if value is None:
            if use_physics_loss:
                raise ValueError(f"{key} must be provided when physics-informed loss is enabled.")
            return default
        if value <= 0:
            if use_physics_loss:
                raise ValueError(f"{key} must be positive.")
            return default
        return float(value)
            
    def __getitem__(self, key):
        """
        Allow dictionary-like access to configuration.
        
        Args:
            key (str): The configuration key.
            
        Returns:
            The configuration value.
        """
        return self.get(key)
        
    def __contains__(self, key):
        """
        Check if a key exists in the configuration.
        
        Args:
            key (str): The configuration key.
            
        Returns:
            bool: True if the key exists, False otherwise.
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return False
                
        return True 
    
    def __getattr__(self, name):
        # Avoid recursion by checking for special attributes first
        # These are used by Python's pickle and other internal mechanisms
        if name.startswith('__') and name.endswith('__'):
            raise AttributeError(f"'Config' object has no attribute '{name}'")
        
        # Use object.__getattribute__ to directly access __dict__ without triggering __getattr__
        try:
            instance_dict = object.__getattribute__(self, '__dict__')
            if name in instance_dict:
                return instance_dict[name]
        except AttributeError:
            pass
        
        # Now safely check config dictionary
        try:
            config_dict = object.__getattribute__(self, 'config')
            if name in config_dict:
                item = config_dict[name]
                if isinstance(item, dict):
                    nested_config = Config.__new__(Config)
                    nested_config.config = item
                    return nested_config
                return item
        except AttributeError:
            pass
            
        raise AttributeError(f"'Config' object has no attribute '{name}'")