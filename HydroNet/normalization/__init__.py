"""
Normalization module for 2D computational hydraulics data.

This module provides flexible, consistent normalization methods for DeepONet,
PINN, and Physics-Informed DeepONet models.

Main components:
- BaseNormalizer: Abstract base class for all normalizers
- MinMaxNormalizer: Min-max normalization to a specified range
- ZScoreNormalizer: Z-score (standardization) normalization
- PhysicsBasedNormalizer: Physics-based scaling using characteristic scales
- IdentityNormalizer: No transformation (for variables that don't need normalization)
- NormalizationManager: High-level interface for managing multiple variables

Example usage:
    >>> from HydroNet.normalization import NormalizationManager, create_default_normalization_manager
    >>> 
    >>> # Create a manager with default settings
    >>> manager = create_default_normalization_manager()
    >>> 
    >>> # Fit normalizers on training data
    >>> training_data = {
    ...     'x': x_train,
    ...     'y': y_train,
    ...     'h': h_train,
    ...     'u': u_train,
    ...     'v': v_train
    ... }
    >>> manager.fit_all(training_data)
    >>> 
    >>> # Transform data
    >>> normalized_data = manager.transform_dict(training_data)
    >>> 
    >>> # Save normalization parameters
    >>> manager.save_yaml('normalization_params.yaml')
    >>> 
    >>> # Load normalization parameters (e.g., during inference)
    >>> manager2 = NormalizationManager()
    >>> manager2.load_yaml('normalization_params.yaml')
    >>> 
    >>> # Inverse transform predictions
    >>> predictions_denorm = manager2.inverse_transform('h', predictions_norm)
"""

from .normalizers import (
    BaseNormalizer,
    MinMaxNormalizer,
    ZScoreNormalizer,
    PhysicsBasedNormalizer,
    IdentityNormalizer
)

from .manager import NormalizationManager

from .helpers import (
    create_default_normalization_manager,
    create_coordinate_normalizers,
    create_solution_normalizers,
    create_static_field_normalizers,
    fit_from_data_dict
)

__all__ = [
    # Base classes
    'BaseNormalizer',
    
    # Normalizer implementations
    'MinMaxNormalizer',
    'ZScoreNormalizer',
    'PhysicsBasedNormalizer',
    'IdentityNormalizer',
    
    # Manager
    'NormalizationManager',
    
    # Helper functions
    'create_default_normalization_manager',
    'create_coordinate_normalizers',
    'create_solution_normalizers',
    'create_static_field_normalizers',
    'fit_from_data_dict',
]

__version__ = '1.0.0'

