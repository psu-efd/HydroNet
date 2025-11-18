"""
Helper functions for common normalization setups in 2D computational hydraulics.

This module provides convenience functions for setting up normalization
according to best practices for different variable types.
"""
from typing import Dict, Optional, Union
import numpy as np
import torch

from .manager import NormalizationManager
from .normalizers import MinMaxNormalizer, ZScoreNormalizer, PhysicsBasedNormalizer


def create_default_normalization_manager() -> NormalizationManager:
    """
    Create a normalization manager with default settings for 2D shallow water equations.
    
    Based on recommendations in the README:
    - Coordinates (x, y, t): min-max to [-1, 1]
    - Solution variables (h, u, v): z-score
    - Static fields (zb, n, sx, sy): z-score
    
    Returns:
        NormalizationManager with default normalizers configured.
    """
    manager = NormalizationManager()
    
    # Coordinates: min-max to [-1, 1]
    manager.create_normalizer('x', 'min-max', min_val=-1.0, max_val=1.0)
    manager.create_normalizer('y', 'min-max', min_val=-1.0, max_val=1.0)
    manager.create_normalizer('t', 'min-max', min_val=-1.0, max_val=1.0)
    
    # Solution variables: z-score
    manager.create_normalizer('h', 'z-score')
    manager.create_normalizer('u', 'z-score')
    manager.create_normalizer('v', 'z-score')
    
    # Static fields: z-score
    manager.create_normalizer('zb', 'z-score')
    manager.create_normalizer('n', 'z-score')  # Manning's n
    manager.create_normalizer('sx', 'z-score')  # Bed slope x
    manager.create_normalizer('sy', 'z-score')  # Bed slope y
    
    return manager


def create_coordinate_normalizers(
    manager: NormalizationManager,
    method: str = 'min-max',
    min_val: float = -1.0,
    max_val: float = 1.0
) -> NormalizationManager:
    """
    Create normalizers for coordinates (x, y, t).
    
    Args:
        manager: NormalizationManager instance.
        method: Normalization method ('min-max' or 'physics-based').
        min_val: Minimum value for min-max (default: -1.0).
        max_val: Maximum value for min-max (default: 1.0).
        
    Returns:
        Updated NormalizationManager.
    """
    if method == 'min-max':
        manager.create_normalizer('x', 'min-max', min_val=min_val, max_val=max_val)
        manager.create_normalizer('y', 'min-max', min_val=min_val, max_val=max_val)
        manager.create_normalizer('t', 'min-max', min_val=min_val, max_val=max_val)
    elif method == 'physics-based':
        # Physics-based would need reference and scale values
        # This is a placeholder - user should set these appropriately
        manager.create_normalizer('x', 'physics-based', reference=0.0, scale=1.0)
        manager.create_normalizer('y', 'physics-based', reference=0.0, scale=1.0)
        manager.create_normalizer('t', 'physics-based', reference=0.0, scale=1.0)
    else:
        raise ValueError(f"Unknown method for coordinates: {method}")
    
    return manager


def create_solution_normalizers(
    manager: NormalizationManager,
    method: str = 'z-score',
    robust: bool = False
) -> NormalizationManager:
    """
    Create normalizers for solution variables (h, u, v).
    
    Args:
        manager: NormalizationManager instance.
        method: Normalization method ('z-score' or 'min-max').
        robust: If True, use robust scaling (median/IQR) for z-score.
        
    Returns:
        Updated NormalizationManager.
    """
    if method == 'z-score':
        manager.create_normalizer('h', 'z-score', robust=robust)
        manager.create_normalizer('u', 'z-score', robust=robust)
        manager.create_normalizer('v', 'z-score', robust=robust)
    elif method == 'min-max':
        manager.create_normalizer('h', 'min-max', min_val=-1.0, max_val=1.0)
        manager.create_normalizer('u', 'min-max', min_val=-1.0, max_val=1.0)
        manager.create_normalizer('v', 'min-max', min_val=-1.0, max_val=1.0)
    else:
        raise ValueError(f"Unknown method for solution variables: {method}")
    
    return manager


def create_static_field_normalizers(
    manager: NormalizationManager,
    method: str = 'z-score',
    zb_method: Optional[str] = None,
    n_method: Optional[str] = None,
    slope_method: Optional[str] = None
) -> NormalizationManager:
    """
    Create normalizers for static fields (zb, n, sx, sy).
    
    Args:
        manager: NormalizationManager instance.
        method: Default normalization method for all static fields.
        zb_method: Override method for bed elevation (optional).
        n_method: Override method for Manning's n (optional).
        slope_method: Override method for slopes (optional).
        
    Returns:
        Updated NormalizationManager.
    """
    # Bed elevation
    zb_m = zb_method if zb_method is not None else method
    if zb_m == 'z-score':
        manager.create_normalizer('zb', 'z-score')
    elif zb_m == 'min-max':
        manager.create_normalizer('zb', 'min-max', min_val=-1.0, max_val=1.0)
    else:
        raise ValueError(f"Unknown method for zb: {zb_m}")
    
    # Manning's n
    n_m = n_method if n_method is not None else method
    if n_m == 'z-score':
        manager.create_normalizer('n', 'z-score')
    elif n_m == 'min-max':
        manager.create_normalizer('n', 'min-max', min_val=0.0, max_val=1.0)
    else:
        raise ValueError(f"Unknown method for n: {n_m}")
    
    # Slopes
    slope_m = slope_method if slope_method is not None else method
    if slope_m == 'z-score':
        manager.create_normalizer('sx', 'z-score')
        manager.create_normalizer('sy', 'z-score')
    elif slope_m == 'min-max':
        manager.create_normalizer('sx', 'min-max', min_val=-1.0, max_val=1.0)
        manager.create_normalizer('sy', 'min-max', min_val=-1.0, max_val=1.0)
    else:
        raise ValueError(f"Unknown method for slopes: {slope_m}")
    
    return manager


def fit_from_data_dict(
    manager: NormalizationManager,
    data_dict: Dict[str, Union[np.ndarray, torch.Tensor]],
    variable_groups: Optional[Dict[str, list]] = None
) -> NormalizationManager:
    """
    Fit normalizers from a dictionary of data.
    
    This is a convenience function that fits all normalizers that exist
    in the manager and have corresponding data in the dictionary.
    
    Args:
        manager: NormalizationManager instance.
        data_dict: Dictionary mapping variable names to their data.
        variable_groups: Optional dictionary mapping group names to variable lists.
                        If provided, computes statistics across groups.
        
    Returns:
        Updated NormalizationManager with fitted normalizers.
    """
    if variable_groups is None:
        # Simple case: fit each variable independently
        manager.fit_all(data_dict)
    else:
        # Fit variables that are part of groups
        for group_name, var_list in variable_groups.items():
            # Collect all data for variables in this group
            group_data = []
            for var_name in var_list:
                if var_name in data_dict:
                    data = data_dict[var_name]
                    # Convert to numpy and flatten
                    if isinstance(data, torch.Tensor):
                        data = data.detach().cpu().numpy()
                    group_data.append(data.flatten())
            
            if group_data:
                # Concatenate all data in the group
                combined_data = np.concatenate(group_data)
                
                # Fit each variable in the group using the combined statistics
                # This ensures consistent normalization across variables in the group
                for var_name in var_list:
                    if var_name in manager.normalizers:
                        manager.fit(var_name, combined_data)
        
        # Fit variables not in any group
        all_grouped_vars = set()
        for var_list in variable_groups.values():
            all_grouped_vars.update(var_list)
        
        for var_name, data in data_dict.items():
            if var_name not in all_grouped_vars and var_name in manager.normalizers:
                manager.fit(var_name, data)
    
    return manager

