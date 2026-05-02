"""
Smooth differentiable operations for FVM computations.

Standard operations like abs, sqrt, and pow have discontinuous or
undefined gradients at zero. These smooth approximations ensure
stable automatic differentiation through the FVM solver.

Follows the approach in Hydrograd.jl/src/utilities/smooth_functions.jl.
"""

import torch


def smooth_abs(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Smooth approximation of |x|: sqrt(x^2 + eps)."""
    return torch.sqrt(x * x + eps)


def smooth_sqrt(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Smooth sqrt that avoids zero: sqrt(max(x, eps))."""
    return torch.sqrt(torch.clamp(x, min=eps))


def smooth_pow(x: torch.Tensor, p: float, eps: float = 1e-6) -> torch.Tensor:
    """Smooth power that avoids zero base: max(x, eps)^p."""
    return torch.pow(torch.clamp(x, min=eps), p)
