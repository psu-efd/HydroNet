"""
Base normalizer classes and specific normalization method implementations.

This module provides flexible normalization methods for 2D computational hydraulics data,
including coordinates, solution variables, and static fields.
"""
import numpy as np
import torch
from abc import ABC, abstractmethod
from typing import Union, Dict, Optional, Tuple
import warnings


class BaseNormalizer(ABC):
    """
    Abstract base class for all normalizers.
    
    All normalizers must implement fit, transform, inverse_transform, and get_params methods.
    """
    
    def __init__(self, name: str = None):
        """
        Initialize the normalizer.
        
        Args:
            name (str, optional): Name of the normalizer (e.g., 'x', 'h', 'zb').
        """
        self.name = name
        self.fitted = False
        self.params = {}
    
    @abstractmethod
    def fit(self, data: Union[np.ndarray, torch.Tensor]) -> 'BaseNormalizer':
        """
        Fit the normalizer to the data (compute statistics).
        
        Args:
            data: Input data array (numpy or torch tensor).
            
        Returns:
            self: Returns self for method chaining.
        """
        pass
    
    @abstractmethod
    def transform(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Transform data using the fitted parameters.
        
        Args:
            data: Input data array (numpy or torch tensor).
            
        Returns:
            Normalized data (same type as input).
        """
        pass
    
    @abstractmethod
    def inverse_transform(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Inverse transform normalized data back to original scale.
        
        Args:
            data: Normalized data array (numpy or torch tensor).
            
        Returns:
            Denormalized data (same type as input).
        """
        pass
    
    @abstractmethod
    def get_params(self) -> Dict:
        """
        Get normalization parameters.
        
        Returns:
            Dictionary containing normalization parameters.
        """
        pass
    
    def set_params(self, params: Dict) -> 'BaseNormalizer':
        """
        Set normalization parameters (useful for loading from file).
        
        Args:
            params: Dictionary containing normalization parameters.
            
        Returns:
            self: Returns self for method chaining.
        """
        self.params = params
        self.fitted = True
        return self
    
    def _to_numpy(self, data: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Convert input to numpy array."""
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        return np.asarray(data)
    
    def _preserve_type(self, data: Union[np.ndarray, torch.Tensor], 
                      result: np.ndarray) -> Union[np.ndarray, torch.Tensor]:
        """Convert result back to original type."""
        if isinstance(data, torch.Tensor):
            return torch.from_numpy(result).to(data.device).to(data.dtype)
        return result


class MinMaxNormalizer(BaseNormalizer):
    """
    Min-max normalization: maps data to [min_val, max_val] range.
    
    Formula: normalized = (data - min) / (max - min) * (max_val - min_val) + min_val
    
    Default range is [-1, 1]. Can be changed to [0, 1] or custom range.
    
    Pros:
    - Bounded output range
    - Preserves relative relationships
    - Good for coordinates when mapping to fixed domain
    
    Cons:
    - Sensitive to outliers
    - Requires min/max to be known a priori
    - Can compress most values if range is dominated by extremes
    """
    
    def __init__(self, name: str = None, min_val: float = -1.0, max_val: float = 1.0, 
                 clip: bool = False, eps: float = 1e-8):
        """
        Initialize min-max normalizer.
        
        Args:
            name (str, optional): Name of the normalizer.
            min_val (float): Minimum value of output range (default: -1.0).
            max_val (float): Maximum value of output range (default: 1.0).
            clip (bool): Whether to clip values outside [min, max] during transform.
            eps (float): Small epsilon to avoid division by zero.
        """
        super().__init__(name)
        self.min_val = min_val
        self.max_val = max_val
        self.clip = clip
        self.eps = eps
    
    def fit(self, data: Union[np.ndarray, torch.Tensor]) -> 'MinMaxNormalizer':
        """Fit the normalizer to compute min and max."""
        data_np = self._to_numpy(data)
        
        # Handle multi-dimensional arrays
        if data_np.ndim > 1:
            # Compute min/max along all dimensions except the last (if it's a feature dimension)
            # For simplicity, flatten and compute global min/max
            data_flat = data_np.flatten()
        else:
            data_flat = data_np
        
        self.params = {
            'min': float(np.min(data_flat)),
            'max': float(np.max(data_flat)),
            'min_val': self.min_val,
            'max_val': self.max_val,
            'clip': self.clip,
            'eps': self.eps
        }
        
        self.fitted = True
        return self
    
    def transform(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Transform data to [min_val, max_val] range."""
        if not self.fitted:
            raise ValueError(f"Normalizer {self.name} must be fitted before transform")
        
        data_np = self._to_numpy(data)
        is_torch = isinstance(data, torch.Tensor)
        
        min_data = self.params['min']
        max_data = self.params['max']
        min_val = self.params['min_val']
        max_val = self.params['max_val']
        eps = self.params['eps']
        
        # Avoid division by zero
        range_data = max_data - min_data
        if range_data < eps:
            # Constant data: center at middle of range
            result = np.full_like(data_np, (min_val + max_val) / 2.0)
        else:
            # Normalize
            result = (data_np - min_data) / range_data * (max_val - min_val) + min_val
        
        # Clip if requested
        if self.clip:
            result = np.clip(result, min_val, max_val)
        
        return self._preserve_type(data, result)
    
    def inverse_transform(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Inverse transform from [min_val, max_val] back to original scale."""
        if not self.fitted:
            raise ValueError(f"Normalizer {self.name} must be fitted before inverse_transform")
        
        data_np = self._to_numpy(data)
        
        min_data = self.params['min']
        max_data = self.params['max']
        min_val = self.params['min_val']
        max_val = self.params['max_val']
        eps = self.params['eps']
        
        # Avoid division by zero
        range_val = max_val - min_val
        if range_val < eps:
            # Should not happen with valid min_val/max_val, but handle it
            result = np.full_like(data_np, (min_data + max_data) / 2.0)
        else:
            # Denormalize
            result = (data_np - min_val) / range_val * (max_data - min_data) + min_data
        
        return self._preserve_type(data, result)
    
    def get_params(self) -> Dict:
        """Get normalization parameters."""
        return {
            'method': 'min-max',
            'name': self.name,
            **self.params
        }


class ZScoreNormalizer(BaseNormalizer):
    """
    Z-score normalization: standardizes data to have mean 0 and std 1.
    
    Formula: normalized = (data - mean) / std
    
    Pros:
    - Handles outliers better than min-max
    - Works well for approximately normal distributions
    - Good for solution variables (h, u, v)
    - Preserves relative differences
    
    Cons:
    - Output is unbounded (can have large values)
    - Sensitive to outliers in mean/std calculation
    - Assumes roughly symmetric distribution
    """
    
    def __init__(self, name: str = None, robust: bool = False, eps: float = 1e-8):
        """
        Initialize z-score normalizer.
        
        Args:
            name (str, optional): Name of the normalizer.
            robust (bool): If True, use median and IQR instead of mean and std.
            eps (float): Small epsilon to avoid division by zero.
        """
        super().__init__(name)
        self.robust = robust
        self.eps = eps
    
    def fit(self, data: Union[np.ndarray, torch.Tensor]) -> 'ZScoreNormalizer':
        """Fit the normalizer to compute mean and std (or median and IQR)."""
        data_np = self._to_numpy(data)
        
        # Handle multi-dimensional arrays
        if data_np.ndim > 1:
            data_flat = data_np.flatten()
        else:
            data_flat = data_np
        
        if self.robust:
            # Robust scaling using median and IQR
            median = np.median(data_flat)
            q75, q25 = np.percentile(data_flat, [75, 25])
            iqr = q75 - q25
            if iqr < self.eps:
                iqr = self.eps
            self.params = {
                'mean': float(median),
                'std': float(iqr),
                'robust': True
            }
        else:
            # Standard z-score
            mean = np.mean(data_flat)
            std = np.std(data_flat)
            if std < self.eps:
                std = 1.0  # Avoid division by zero for constant data
                warnings.warn(f"Standard deviation is near zero for {self.name}, using std=1.0")
            self.params = {
                'mean': float(mean),
                'std': float(std),
                'robust': False
            }
        
        self.params['eps'] = self.eps
        self.fitted = True
        return self
    
    def transform(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Transform data to z-score normalized."""
        if not self.fitted:
            raise ValueError(f"Normalizer {self.name} must be fitted before transform")
        
        data_np = self._to_numpy(data)
        
        mean = self.params['mean']
        std = self.params['std']
        
        result = (data_np - mean) / (std + self.eps)
        
        return self._preserve_type(data, result)
    
    def inverse_transform(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Inverse transform from z-score back to original scale."""
        if not self.fitted:
            raise ValueError(f"Normalizer {self.name} must be fitted before inverse_transform")
        
        data_np = self._to_numpy(data)
        
        mean = self.params['mean']
        std = self.params['std']
        
        result = data_np * (std + self.eps) + mean
        
        return self._preserve_type(data, result)
    
    def get_params(self) -> Dict:
        """Get normalization parameters."""
        return {
            'method': 'z-score',
            'name': self.name,
            **self.params
        }


class PhysicsBasedNormalizer(BaseNormalizer):
    """
    Physics-based normalization using characteristic scales.
    
    Formula: normalized = (data - reference) / scale
    
    This is useful for:
    - Coordinates: normalize by characteristic length/time
    - Solution variables: normalize by characteristic depth/velocity
    - Making problems dimensionless for better generalization
    
    Pros:
    - Physically meaningful
    - Can improve cross-scenario generalization
    - Makes learning problem closer to dimensionless relationships
    
    Cons:
    - Requires domain knowledge to choose appropriate scales
    - May need different scales for different cases
    """
    
    def __init__(self, name: str = None, reference: float = 0.0, scale: float = 1.0):
        """
        Initialize physics-based normalizer.
        
        Args:
            name (str, optional): Name of the normalizer.
            reference (float): Reference value (e.g., domain center, initial time).
            scale (float): Characteristic scale (e.g., length scale, time scale).
        """
        super().__init__(name)
        self.reference = reference
        self.scale = scale
        self.params = {
            'reference': reference,
            'scale': scale
        }
        self.fitted = True  # Physics-based normalizer doesn't need fitting from data
    
    def fit(self, data: Union[np.ndarray, torch.Tensor]) -> 'PhysicsBasedNormalizer':
        """
        Fit the normalizer. For physics-based, this can optionally compute
        reference and scale from data, or use provided values.
        """
        data_np = self._to_numpy(data)
        
        if data_np.ndim > 1:
            data_flat = data_np.flatten()
        else:
            data_flat = data_np
        
        # Optionally compute reference from data (e.g., center of domain)
        # For now, use provided reference and scale
        # Could add logic to compute from data if needed
        
        self.params = {
            'reference': self.reference,
            'scale': self.scale
        }
        self.fitted = True
        return self
    
    def transform(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Transform data using physics-based scaling."""
        if not self.fitted:
            raise ValueError(f"Normalizer {self.name} must be fitted before transform")
        
        data_np = self._to_numpy(data)
        
        reference = self.params['reference']
        scale = self.params['scale']
        
        if abs(scale) < 1e-8:
            raise ValueError(f"Scale cannot be zero for physics-based normalizer {self.name}")
        
        result = (data_np - reference) / scale
        
        return self._preserve_type(data, result)
    
    def inverse_transform(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Inverse transform from physics-based scaling back to original scale."""
        if not self.fitted:
            raise ValueError(f"Normalizer {self.name} must be fitted before inverse_transform")
        
        data_np = self._to_numpy(data)
        
        reference = self.params['reference']
        scale = self.params['scale']
        
        result = data_np * scale + reference
        
        return self._preserve_type(data, result)
    
    def get_params(self) -> Dict:
        """Get normalization parameters."""
        return {
            'method': 'physics-based',
            'name': self.name,
            **self.params
        }
    
    def set_reference_scale(self, reference: float, scale: float) -> 'PhysicsBasedNormalizer':
        """
        Set reference and scale values.
        
        Args:
            reference: Reference value.
            scale: Characteristic scale.
            
        Returns:
            self: Returns self for method chaining.
        """
        self.reference = reference
        self.scale = scale
        self.params = {
            'reference': reference,
            'scale': scale
        }
        return self


class IdentityNormalizer(BaseNormalizer):
    """
    Identity normalizer: no transformation (useful for variables that don't need normalization).
    """
    
    def __init__(self, name: str = None):
        """Initialize identity normalizer."""
        super().__init__(name)
        self.fitted = True
        self.params = {}
    
    def fit(self, data: Union[np.ndarray, torch.Tensor]) -> 'IdentityNormalizer':
        """Fit does nothing for identity normalizer."""
        self.fitted = True
        return self
    
    def transform(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Return data unchanged."""
        return data
    
    def inverse_transform(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Return data unchanged."""
        return data
    
    def get_params(self) -> Dict:
        """Get normalization parameters."""
        return {
            'method': 'identity',
            'name': self.name
        }

