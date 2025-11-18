# Normalization Module Usage Guide

This guide provides practical examples of using the normalization module for 2D computational hydraulics data.

## Quick Start

```python
from HydroNet.normalization import NormalizationManager, create_default_normalization_manager
import numpy as np

# Create a manager with default settings
manager = create_default_normalization_manager()

# Prepare your training data
training_data = {
    'x': x_coordinates,      # Spatial x coordinates
    'y': y_coordinates,      # Spatial y coordinates
    't': time_values,        # Time values (if unsteady)
    'h': water_depth,        # Water depth
    'u': velocity_x,         # Velocity in x direction
    'v': velocity_y,         # Velocity in y direction
    'zb': bed_elevation,     # Bed elevation
    'n': mannings_n,         # Manning's roughness coefficient
    'sx': bed_slope_x,       # Bed slope in x direction
    'sy': bed_slope_y,       # Bed slope in y direction
}

# Fit normalizers on training data
manager.fit_all(training_data)

# Transform data for training
normalized_data = manager.transform_dict(training_data)

# Save normalization parameters for later use
manager.save_yaml('normalization_params.yaml')

# During inference, load the parameters
manager_inference = NormalizationManager()
manager_inference.load_yaml('normalization_params.yaml')

# Normalize input data
input_normalized = manager_inference.transform_dict(input_data)

# After model prediction, denormalize output
predictions_denorm = manager_inference.inverse_transform('h', predictions_normalized)
```

## Custom Normalization Setup

If you want to customize the normalization methods for different variables:

```python
from HydroNet.normalization import NormalizationManager
from HydroNet.normalization.helpers import (
    create_coordinate_normalizers,
    create_solution_normalizers,
    create_static_field_normalizers
)

manager = NormalizationManager()

# Coordinates: min-max to [-1, 1]
create_coordinate_normalizers(manager, method='min-max', min_val=-1.0, max_val=1.0)

# Solution variables: z-score
create_solution_normalizers(manager, method='z-score', robust=False)

# Static fields: z-score (with option to override individual fields)
create_static_field_normalizers(
    manager,
    method='z-score',
    zb_method='z-score',      # Bed elevation: z-score
    n_method='min-max',       # Manning's n: min-max to [0, 1]
    slope_method='z-score'    # Slopes: z-score
)

# Fit on data
manager.fit_all(training_data)
```

## Individual Variable Normalization

You can also create normalizers for individual variables:

```python
manager = NormalizationManager()

# Min-max normalization for coordinates
manager.create_normalizer('x', 'min-max', min_val=-1.0, max_val=1.0)
manager.create_normalizer('y', 'min-max', min_val=-1.0, max_val=1.0)

# Z-score normalization for solution variables
manager.create_normalizer('h', 'z-score')
manager.create_normalizer('u', 'z-score', robust=True)  # Robust scaling

# Physics-based normalization
manager.create_normalizer('t', 'physics-based', reference=0.0, scale=3600.0)

# Fit and transform
manager.fit('x', x_data)
x_normalized = manager.transform('x', x_data)
x_denormalized = manager.inverse_transform('x', x_normalized)
```

## Working with PyTorch Tensors

The module automatically handles both NumPy arrays and PyTorch tensors:

```python
import torch

# Create PyTorch tensors
h_train = torch.randn(1000, 1) * 2.0 + 3.0

# Fit and transform (automatically preserves tensor type and device)
manager = NormalizationManager()
manager.create_normalizer('h', 'z-score')
manager.fit('h', h_train)
h_norm = manager.transform('h', h_train)  # Returns torch.Tensor

# Works with GPU tensors too
h_train_gpu = h_train.cuda()
h_norm_gpu = manager.transform('h', h_train_gpu)  # Preserves device
```

## Saving and Loading Parameters

### Save to YAML

```python
manager.save_yaml('normalization_params.yaml')
```

### Load parameters

```python
manager.load_yaml('normalization_params.yaml')
```

## Integration with Data Processing Pipeline

### During Data Preprocessing

```python
# 1. Load raw simulation data
raw_data = load_simulation_data('case1.h5')

# 2. Create and fit normalizer
manager = NormalizationManager()
manager.create_normalizer('x', 'min-max', min_val=-1.0, max_val=1.0)
manager.create_normalizer('h', 'z-score')
# ... add more normalizers

# 3. Fit on training data
manager.fit_all(raw_data)

# 4. Transform and save normalized data
normalized_data = manager.transform_dict(raw_data)
save_normalized_data(normalized_data, 'case1_normalized.h5')

# 5. Save normalization parameters
manager.save_yaml('case1_normalization.yaml')
```

### During Training

```python
# Load normalization parameters
manager = NormalizationManager()
manager.load_yaml('case1_normalization.yaml')

# Normalize batch data
for batch in dataloader:
    batch_normalized = manager.transform_dict(batch)
    # ... training code
```

### During Inference

```python
# Load normalization parameters
manager = NormalizationManager()
manager.load_yaml('case1_normalization.yaml')

# Normalize input
input_normalized = manager.transform_dict(input_data)

# Model prediction
predictions_norm = model(input_normalized)

# Denormalize predictions
predictions = manager.inverse_transform_dict(predictions_norm)
```

## Best Practices

1. **Fit once on training data**: Compute normalization statistics on the entire training dataset, not per-batch.

2. **Save parameters**: Always save normalization parameters alongside your trained model for consistent inference.

3. **Use consistent methods**: Use the same normalization method for the same variable type across all cases.

4. **Handle missing variables**: If a variable doesn't need normalization, use `IdentityNormalizer` or simply don't add it to the manager.

5. **Test reconstruction**: After fitting, test that `inverse_transform(transform(data))` reconstructs the original data accurately.

6. **Document your choices**: Record why you chose specific normalization methods for each variable type in your project documentation.

## Common Patterns

### Pattern 1: Multi-case Training

```python
# Collect data from multiple cases
all_data = {}
for case in cases:
    case_data = load_case_data(case)
    for key, value in case_data.items():
        if key not in all_data:
            all_data[key] = []
        all_data[key].append(value)

# Concatenate all cases
for key in all_data:
    all_data[key] = np.concatenate(all_data[key])

# Fit on combined data
manager.fit_all(all_data)
```

### Pattern 2: Per-variable Statistics

```python
# Get statistics for each variable
for var_name in manager.normalizers:
    normalizer = manager.get_normalizer(var_name)
    params = normalizer.get_params()
    print(f"{var_name}: {params}")
```

### Pattern 3: Selective Normalization

```python
# Only normalize specific variables
variables_to_normalize = ['x', 'y', 'h', 'u', 'v']
normalized_data = {}
for var in variables_to_normalize:
    if var in manager.normalizers:
        normalized_data[var] = manager.transform(var, data[var])
    else:
        normalized_data[var] = data[var]  # Keep original
```

## Troubleshooting

### Issue: "Normalizer must be fitted before transform"

**Solution**: Make sure to call `fit()` or `fit_all()` before using `transform()`.

### Issue: "Division by zero" warnings

**Solution**: This happens when data has zero variance (constant values). The normalizer handles this automatically, but you may want to check your data.

### Issue: Reconstruction errors are large

**Solution**: Check for numerical precision issues. For min-max normalization, ensure the range is not too small. For z-score, check if std is very small.

### Issue: YAML save/load fails

**Solution**: Make sure PyYAML is installed: `pip install pyyaml`.

