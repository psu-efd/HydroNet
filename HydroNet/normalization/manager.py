"""
Normalization manager for handling multiple variables with different normalization methods.

This module provides a high-level interface for managing normalization across multiple
variables in 2D computational hydraulics models.
"""
import numpy as np
import torch
from typing import Union, Dict, Optional, List, Any
from pathlib import Path

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    yaml = None

from .normalizers import (
    BaseNormalizer, MinMaxNormalizer, ZScoreNormalizer, 
    PhysicsBasedNormalizer, IdentityNormalizer
)


class NormalizationManager:
    """
    Manager for handling normalization of multiple variables.
    
    This class allows you to:
    - Assign different normalization methods to different variables
    - Fit normalizers on training data
    - Transform and inverse transform data
    - Save/load normalization parameters to/from YAML files
    """
    
    def __init__(self):
        """Initialize the normalization manager."""
        self.normalizers: Dict[str, BaseNormalizer] = {}
        self.metadata: Dict[str, Any] = {
            'version': '1.0',
            'description': 'Normalization parameters for 2D computational hydraulics data'
        }
    
    def add_normalizer(self, name: str, normalizer: BaseNormalizer, overwrite: bool = False) -> 'NormalizationManager':
        """
        Add a normalizer for a variable.
        
        Args:
            name: Name of the variable (e.g., 'x', 'h', 'zb').
            normalizer: Normalizer instance.
            overwrite: If True, allow overwriting an existing normalizer. Default: False.
            
        Returns:
            self: Returns self for method chaining.
            
        Raises:
            ValueError: If a normalizer with the same name already exists and overwrite=False.
        """
        if name in self.normalizers and not overwrite:
            raise ValueError(
                f"Normalizer for '{name}' already exists. "
                f"Use overwrite=True to replace it, or remove it first."
            )
        
        normalizer.name = name
        self.normalizers[name] = normalizer
        return self
    
    def create_normalizer(self, name: str, method: str, overwrite: bool = False, **kwargs) -> 'NormalizationManager':
        """
        Create and add a normalizer for a variable.
        
        Args:
            name: Name of the variable.
            method: Normalization method ('min-max', 'z-score', 'physics-based', 'identity').
            overwrite: If True, allow overwriting an existing normalizer. Default: False.
            **kwargs: Additional arguments for the normalizer.
            
        Returns:
            self: Returns self for method chaining.
            
        Raises:
            ValueError: If a normalizer with the same name already exists and overwrite=False.
        """
        if name in self.normalizers and not overwrite:
            raise ValueError(
                f"Normalizer for '{name}' already exists. "
                f"Use overwrite=True to replace it, or remove it first."
            )
        
        if method == 'min-max':
            normalizer = MinMaxNormalizer(name=name, **kwargs)
        elif method == 'z-score':
            normalizer = ZScoreNormalizer(name=name, **kwargs)
        elif method == 'physics-based':
            normalizer = PhysicsBasedNormalizer(name=name, **kwargs)
        elif method == 'identity':
            normalizer = IdentityNormalizer(name=name)
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        self.normalizers[name] = normalizer
        return self
    
    def fit(self, name: str, data: Union[np.ndarray, torch.Tensor]) -> 'NormalizationManager':
        """
        Fit a normalizer to data.
        
        Args:
            name: Name of the variable.
            data: Training data.
            
        Returns:
            self: Returns self for method chaining.
        """
        if name not in self.normalizers:
            raise ValueError(f"Normalizer for '{name}' not found. Use add_normalizer or create_normalizer first.")
        
        self.normalizers[name].fit(data)
        return self
    
    def fit_all(self, data_dict: Dict[str, Union[np.ndarray, torch.Tensor]]) -> 'NormalizationManager':
        """
        Fit all normalizers to their corresponding data.
        
        Args:
            data_dict: Dictionary mapping variable names to their data.
            
        Returns:
            self: Returns self for method chaining.
        """
        for name, data in data_dict.items():
            if name in self.normalizers:
                self.fit(name, data)
            else:
                print(f"Warning: No normalizer found for '{name}', skipping.")
        return self
    
    def transform(self, name: str, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Transform data for a variable.
        
        Args:
            name: Name of the variable.
            data: Data to transform.
            
        Returns:
            Normalized data.
        """
        if name not in self.normalizers:
            raise ValueError(f"Normalizer for '{name}' not found.")
        
        return self.normalizers[name].transform(data)
    
    def transform_dict(self, data_dict: Dict[str, Union[np.ndarray, torch.Tensor]]) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        """
        Transform multiple variables at once.
        
        Args:
            data_dict: Dictionary mapping variable names to their data.
            
        Returns:
            Dictionary of normalized data.
        """
        result = {}
        for name, data in data_dict.items():
            if name in self.normalizers:
                result[name] = self.transform(name, data)
            else:
                print(f"Warning: No normalizer found for '{name}', returning original data.")
                result[name] = data
        return result
    
    def inverse_transform(self, name: str, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Inverse transform normalized data for a variable.
        
        Args:
            name: Name of the variable.
            data: Normalized data.
            
        Returns:
            Denormalized data.
        """
        if name not in self.normalizers:
            raise ValueError(f"Normalizer for '{name}' not found.")
        
        return self.normalizers[name].inverse_transform(data)
    
    def inverse_transform_dict(self, data_dict: Dict[str, Union[np.ndarray, torch.Tensor]]) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        """
        Inverse transform multiple variables at once.
        
        Args:
            data_dict: Dictionary mapping variable names to their normalized data.
            
        Returns:
            Dictionary of denormalized data.
        """
        result = {}
        for name, data in data_dict.items():
            if name in self.normalizers:
                result[name] = self.inverse_transform(name, data)
            else:
                print(f"Warning: No normalizer found for '{name}', returning original data.")
                result[name] = data
        return result
    
    def get_normalizer(self, name: str) -> Optional[BaseNormalizer]:
        """
        Get a normalizer by name.
        
        Args:
            name: Name of the variable.
            
        Returns:
            Normalizer instance or None if not found.
        """
        return self.normalizers.get(name)
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get all normalization parameters.
        
        Returns:
            Dictionary containing all normalization parameters.
        """
        params = {
            'metadata': self.metadata,
            'normalizers': {}
        }
        
        for name, normalizer in self.normalizers.items():
            params['normalizers'][name] = normalizer.get_params()
        
        return params
    
    def set_params(self, params: Dict[str, Any]) -> 'NormalizationManager':
        """
        Set normalization parameters (useful for loading from file).
        
        Args:
            params: Dictionary containing normalization parameters.
            
        Returns:
            self: Returns self for method chaining.
        """
        if 'metadata' in params:
            self.metadata.update(params['metadata'])
        
        if 'normalizers' in params:
            for name, normalizer_params in params['normalizers'].items():
                method = normalizer_params.get('method', 'identity')
                
                # Create normalizer based on method
                if method == 'min-max':
                    normalizer = MinMaxNormalizer(name=name)
                elif method == 'z-score':
                    normalizer = ZScoreNormalizer(name=name)
                elif method == 'physics-based':
                    normalizer = PhysicsBasedNormalizer(name=name)
                elif method == 'identity':
                    normalizer = IdentityNormalizer(name=name)
                else:
                    raise ValueError(f"Unknown normalization method: {method}")
                
                # Set parameters
                normalizer.set_params(normalizer_params)
                self.normalizers[name] = normalizer
        
        return self
    
    def save_yaml(self, filepath: Union[str, Path]) -> 'NormalizationManager':
        """
        Save normalization parameters to a YAML file.
        
        Args:
            filepath: Path to save the YAML file.
            
        Returns:
            self: Returns self for method chaining.
        """
        if not HAS_YAML:
            raise ImportError("PyYAML is required for YAML support. Install it with: pip install pyyaml")
        
        filepath = Path(filepath)
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        params = self.get_params()
        
        with open(filepath, 'w') as f:
            yaml.dump(params, f, default_flow_style=False, sort_keys=False)
        
        print(f"Normalization parameters saved to {filepath}")
        return self
    
    def load_yaml(self, filepath: Union[str, Path]) -> 'NormalizationManager':
        """
        Load normalization parameters from a YAML file.
        
        Args:
            filepath: Path to the YAML file.
            
        Returns:
            self: Returns self for method chaining.
        """
        if not HAS_YAML:
            raise ImportError("PyYAML is required for YAML support. Install it with: pip install pyyaml")
        
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Normalization file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            params = yaml.safe_load(f)
        
        self.set_params(params)
        print(f"Normalization parameters loaded from {filepath}")
        return self
    
    def remove_normalizer(self, name: str) -> 'NormalizationManager':
        """
        Remove a normalizer by name.
        
        Args:
            name: Name of the variable.
            
        Returns:
            self: Returns self for method chaining.
            
        Raises:
            KeyError: If the normalizer doesn't exist.
        """
        if name not in self.normalizers:
            raise KeyError(f"Normalizer for '{name}' not found.")
        del self.normalizers[name]
        return self
    
    def __repr__(self) -> str:
        """String representation of the manager."""
        normalizer_names = list(self.normalizers.keys())
        return f"NormalizationManager(normalizers={normalizer_names})"
    
    def __len__(self) -> int:
        """Number of normalizers."""
        return len(self.normalizers)
    
    def __contains__(self, name: str) -> bool:
        """Check if a normalizer exists."""
        return name in self.normalizers

