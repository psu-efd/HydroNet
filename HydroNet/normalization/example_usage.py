"""
Example usage of the normalization module.

This script demonstrates how to use the normalization module for 2D computational
hydraulics data preprocessing and postprocessing.
"""
import numpy as np
import torch
from pathlib import Path
import sys
import os

# Handle both direct execution and import as module
if __name__ == '__main__':
    # When running as script, set up package structure for relative imports
    script_dir = os.path.dirname(os.path.abspath(__file__))
    hydronet_dir = os.path.dirname(os.path.dirname(script_dir))
    
    # Add to path
    if hydronet_dir not in sys.path:
        sys.path.insert(0, hydronet_dir)
    
    # Set up package structure in sys.modules for relative imports to work
    import types
    import importlib.util
    
    # Create package modules
    normalization_pkg = types.ModuleType('normalization')
    normalization_pkg.__path__ = [script_dir]
    sys.modules['normalization'] = normalization_pkg
    
    # Load normalizers module
    normalizers_path = os.path.join(script_dir, 'normalizers.py')
    spec_normalizers = importlib.util.spec_from_file_location("normalization.normalizers", normalizers_path)
    normalizers_module = importlib.util.module_from_spec(spec_normalizers)
    sys.modules['normalization.normalizers'] = normalizers_module
    spec_normalizers.loader.exec_module(normalizers_module)
    
    # Load manager module (it will use the normalizers module from sys.modules)
    manager_path = os.path.join(script_dir, 'manager.py')
    spec_manager = importlib.util.spec_from_file_location("normalization.manager", manager_path)
    manager_module = importlib.util.module_from_spec(spec_manager)
    sys.modules['normalization.manager'] = manager_module
    spec_manager.loader.exec_module(manager_module)
    
    # Load helpers module
    helpers_path = os.path.join(script_dir, 'helpers.py')
    spec_helpers = importlib.util.spec_from_file_location("normalization.helpers", helpers_path)
    helpers_module = importlib.util.module_from_spec(spec_helpers)
    sys.modules['normalization.helpers'] = helpers_module
    spec_helpers.loader.exec_module(helpers_module)
    
    # Import the classes we need
    NormalizationManager = manager_module.NormalizationManager
    create_default_normalization_manager = helpers_module.create_default_normalization_manager
else:
    # Relative imports when used as module
    from .manager import NormalizationManager
    from .helpers import create_default_normalization_manager


def example_basic_usage():
    """Basic usage example."""
    print("=" * 60)
    print("Example 1: Basic Usage")
    print("=" * 60)
    
    # Create a normalization manager
    manager = NormalizationManager()
    
    # Create normalizers for different variables
    manager.create_normalizer('x', 'min-max', min_val=-1.0, max_val=1.0)
    manager.create_normalizer('y', 'min-max', min_val=-1.0, max_val=1.0)
    manager.create_normalizer('h', 'z-score')
    manager.create_normalizer('u', 'z-score')
    manager.create_normalizer('v', 'z-score')
    
    # Generate some synthetic training data
    np.random.seed(42)
    n_samples = 1000
    
    x_train = np.random.uniform(0, 1000, n_samples)  # x coordinates in meters
    y_train = np.random.uniform(0, 500, n_samples)   # y coordinates in meters
    h_train = np.random.uniform(0.5, 5.0, n_samples)  # water depth in meters
    u_train = np.random.uniform(0, 2.0, n_samples)    # velocity x in m/s
    v_train = np.random.uniform(-1.0, 1.0, n_samples) # velocity y in m/s
    
    # Fit normalizers on training data
    print("\nFitting normalizers...")
    manager.fit('x', x_train)
    manager.fit('y', y_train)
    manager.fit('h', h_train)
    manager.fit('u', u_train)
    manager.fit('v', v_train)
    
    # Transform data
    print("\nTransforming data...")
    x_norm = manager.transform('x', x_train)
    h_norm = manager.transform('h', h_train)
    
    print(f"Original x range: [{x_train.min():.2f}, {x_train.max():.2f}]")
    print(f"Normalized x range: [{x_norm.min():.2f}, {x_norm.max():.2f}]")
    print(f"Original h mean: {h_train.mean():.2f}, std: {h_train.std():.2f}")
    print(f"Normalized h mean: {h_norm.mean():.4f}, std: {h_norm.std():.4f}")
    
    # Inverse transform
    print("\nInverse transforming...")
    x_denorm = manager.inverse_transform('x', x_norm)
    h_denorm = manager.inverse_transform('h', h_norm)
    
    print(f"Reconstruction error x: {np.abs(x_train - x_denorm).max():.2e}")
    print(f"Reconstruction error h: {np.abs(h_train - h_denorm).max():.2e}")


def example_default_manager():
    """Example using the default normalization manager."""
    print("\n" + "=" * 60)
    print("Example 2: Using Default Normalization Manager")
    print("=" * 60)
    
    # Create manager with default settings
    manager = create_default_normalization_manager()
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'x': np.random.uniform(0, 1000, n_samples),
        'y': np.random.uniform(0, 500, n_samples),
        't': np.random.uniform(0, 3600, n_samples),
        'h': np.random.uniform(0.5, 5.0, n_samples),
        'u': np.random.uniform(0, 2.0, n_samples),
        'v': np.random.uniform(-1.0, 1.0, n_samples),
        'zb': np.random.uniform(100, 150, n_samples),
        'n': np.random.uniform(0.02, 0.05, n_samples),
        'sx': np.random.uniform(-0.01, 0.01, n_samples),
        'sy': np.random.uniform(-0.01, 0.01, n_samples),
    }
    
    # Fit all normalizers
    print("\nFitting all normalizers...")
    manager.fit_all(data)
    
    # Transform all data
    print("\nTransforming all data...")
    normalized_data = manager.transform_dict(data)
    
    # Print some statistics
    print("\nNormalization results:")
    for var_name in ['x', 'h', 'u', 'zb']:
        orig = data[var_name]
        norm = normalized_data[var_name]
        print(f"{var_name}:")
        print(f"  Original: min={orig.min():.4f}, max={orig.max():.4f}, "
              f"mean={orig.mean():.4f}, std={orig.std():.4f}")
        print(f"  Normalized: min={norm.min():.4f}, max={norm.max():.4f}, "
              f"mean={norm.mean():.4f}, std={norm.std():.4f}")


def example_save_load():
    """Example of saving and loading normalization parameters."""
    print("\n" + "=" * 60)
    print("Example 3: Save and Load Normalization Parameters")
    print("=" * 60)
    
    # Create and fit a manager
    manager = create_default_normalization_manager()
    
    np.random.seed(42)
    data = {
        'x': np.random.uniform(0, 1000, 100),
        'y': np.random.uniform(0, 500, 100),
        'h': np.random.uniform(0.5, 5.0, 100),
    }
    
    manager.fit_all(data)
    
    # Save to YAML
    save_path = Path('normalization_params_example.yaml').resolve()
    manager.save_yaml(save_path)
    print(f"YAML file saved to: {save_path}")
    
    # Load from YAML
    print("\nLoading from YAML...")
    manager2 = NormalizationManager()
    manager2.load_yaml(save_path)
    
    # Test that loaded parameters work
    test_data = np.array([500.0, 250.0, 2.5])
    norm1 = manager.transform('x', test_data[0])
    norm2 = manager2.transform('x', test_data[0])
    
    print(f"Original manager transform: {norm1:.4f}")
    print(f"Loaded manager transform: {norm2:.4f}")
    print(f"Match: {np.isclose(norm1, norm2)}")
    
    # Clean up (comment out to keep the files)
    print("\nNote: Files are saved in the current working directory.")
    print("To keep the files, comment out the cleanup section below.")
    cleanup_files = False  # Set to True to delete files after example
    if cleanup_files:
        if save_path.exists():
            save_path.unlink()
            print(f"Deleted: {save_path}")


def example_torch_tensors():
    """Example using PyTorch tensors."""
    print("\n" + "=" * 60)
    print("Example 4: Using PyTorch Tensors")
    print("=" * 60)
    
    manager = NormalizationManager()
    manager.create_normalizer('h', 'z-score')
    
    # Create PyTorch tensor
    h_train = torch.randn(100, 1) * 2.0 + 3.0  # mean=3, std=2
    
    # Fit and transform
    manager.fit('h', h_train)
    h_norm = manager.transform('h', h_train)
    
    print(f"Original tensor: mean={h_train.mean():.4f}, std={h_train.std():.4f}")
    print(f"Normalized tensor: mean={h_norm.mean():.4f}, std={h_norm.std():.4f}")
    print(f"Tensor type preserved: {isinstance(h_norm, torch.Tensor)}")
    print(f"Device preserved: {h_norm.device == h_train.device}")


if __name__ == '__main__':
    # Run all examples
    example_basic_usage()
    example_default_manager()
    example_save_load()
    example_torch_tensors()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)

