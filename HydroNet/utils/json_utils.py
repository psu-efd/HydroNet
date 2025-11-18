"""
JSON utility functions for data serialization.

This module provides utilities for converting Python objects to JSON-serializable formats,
particularly for handling numpy arrays and other non-standard types.
"""
import numpy as np
from typing import Any, Union, Dict, List


def convert_to_json_serializable(obj: Any) -> Union[Dict, List, int, float, str, bool, None]:
    """
    Recursively convert numpy arrays and other non-serializable types to JSON-compatible types.
    
    This function handles:
    - numpy arrays -> Python lists
    - numpy integers -> Python int
    - numpy floats -> Python float
    - Nested dictionaries and lists
    
    Args:
        obj: Object to convert (can be dict, list, numpy array, numpy scalar, or any other type).
        
    Returns:
        JSON-serializable version of the input object.
        
    Examples:
        >>> import numpy as np
        >>> data = {
        ...     'array': np.array([1, 2, 3]),
        ...     'nested': {
        ...         'value': np.float64(3.14),
        ...         'list': [np.int64(1), np.int64(2)]
        ...     }
        ... }
        >>> result = convert_to_json_serializable(data)
        >>> result['array']  # [1, 2, 3]
        >>> result['nested']['value']  # 3.14
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_json_serializable(item) for item in obj)
    else:
        return obj

