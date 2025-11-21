# Loss Balancing in Physics-Informed DeepONet

## Problem: Loss Scale Mismatch

When training Physics-Informed DeepONet models, you may observe that:
- **Data loss** decreases well (e.g., from 0.8 to 0.17)
- **PDE loss** remains large (e.g., from 13.2 to 6.2)
- Training is slow and the PDE loss dominates the gradient

### Root Cause

The issue stems from a **scale mismatch** between different loss components:

1. **Data Loss**: Computed on normalized data (MSE on normalized h, u, v)
   - Typically ranges from 0.1 to 1.0
   - Values are small because data is normalized

2. **PDE Loss**: Computed on dimensional PDE residuals
   - Even after physics scaling, residuals can be large (5-15)
   - Involves squared residuals of differential equations
   - Derivatives and physical terms contribute to larger magnitudes

3. **Equal Weights**: With both losses weighted equally (1.0), the larger PDE loss dominates the gradient, causing:
   - Optimizer focuses primarily on reducing PDE loss
   - Data fitting suffers
   - Slow convergence

## Solutions

### Solution 1: Manual Weight Adjustment (Quick Fix)

Reduce the PDE loss weight in your config file:

```yaml
training:
  loss_weights:
    deeponet:
      data_loss: 1.0
      pinn_loss: 0.01  # Much smaller weight for PDE loss
```

**Recommended starting values:**
- If data loss ~0.1-1.0 and PDE loss ~5-15: use `pinn_loss: 0.01-0.1`
- Adjust based on your specific loss magnitudes

### Solution 2: Adaptive Loss Balancing (Recommended)

Enable automatic loss balancing in your config:

```yaml
training:
  loss_weights:
    use_adaptive_balancing: true  # Enable automatic balancing
    adaptive_balancing_epochs: 10  # Update weights for first 10 epochs
    deeponet:
      data_loss: 1.0
      pinn_loss: 0.01  # Starting weight (will be adjusted)
```

**How it works:**
1. Before training starts, computes initial loss magnitudes from a few sample batches
2. Calculates balancing factors to normalize losses to similar scales
3. Automatically adjusts loss weights based on these factors
4. Optionally continues updating weights during initial epochs

**Benefits:**
- No manual tuning required
- Automatically adapts to your specific problem
- More robust across different datasets

### Solution 3: Curriculum Learning

Gradually introduce physics constraints:

1. **Phase 1**: Train with data loss only (`use_physics_loss: false`)
2. **Phase 2**: After convergence, enable physics loss with small weight
3. **Phase 3**: Gradually increase physics loss weight

This can be implemented by:
- Training for N epochs without physics
- Loading checkpoint and continuing with physics enabled
- Using a learning rate schedule that reduces physics weight over time

## Implementation Details

### Adaptive Loss Balancing Algorithm

1. **Initial Magnitude Estimation**:
   ```python
   # Sample a few batches and compute losses
   data_loss_magnitude = median(data_losses)
   pde_loss_magnitude = median(pde_losses)
   ```

2. **Balancing Factor Calculation**:
   ```python
   reference = data_loss_magnitude  # Use data loss as reference
   balancing_factor_pde = reference / pde_loss_magnitude
   ```

3. **Weight Update**:
   ```python
   new_pde_weight = base_pde_weight * balancing_factor_pde
   ```

4. **Optional: Continuous Updates**:
   - During first N epochs, update balancing factors using exponential moving average
   - Recompute weights periodically

### Loss Weight Configuration

The loss weights are stored as PyTorch buffers in the model:

```python
model.loss_weight_deeponet_data_loss  # Weight for data loss
model.loss_weight_deeponet_pinn_loss  # Weight for PDE loss
```

These can be accessed and modified during training if needed.

## Best Practices

1. **Monitor Loss Components**: Always track individual loss components separately
   - Use TensorBoard or logging to visualize loss trends
   - Ensure both losses are decreasing, not just total loss

2. **Start with Adaptive Balancing**: Enable `use_adaptive_balancing: true` for new problems
   - Let the system automatically find good weights
   - Fine-tune manually if needed

3. **Check Loss Scales**: Before training, run a few forward passes to check loss magnitudes
   - If scales differ by >10x, use adaptive balancing or manual adjustment

4. **Validation Loss**: Monitor validation loss (data loss only)
   - If validation loss increases while training loss decreases, physics loss may be too strong

5. **Physics Scales**: Ensure physics scales (`length_scale`, `velocity_scale`) are appropriate
   - These affect PDE residual magnitudes
   - Should match typical scales in your problem

## Example: Before and After

### Before (Equal Weights)
```
Epoch 1: Data Loss: 0.83, PDE Loss: 13.2, Total: 14.0
Epoch 50: Data Loss: 0.18, PDE Loss: 6.4, Total: 6.6
```
- PDE loss dominates
- Slow convergence
- Data fitting may suffer

### After (Adaptive Balancing)
```
Initial magnitudes: Data: 0.83, PDE: 13.2
Balancing factors: Data: 1.0, PDE: 0.063
Updated weights: Data: 1.0, PDE: 0.063

Epoch 1: Data Loss: 0.83, PDE Loss: 13.2, Total: 1.66
Epoch 50: Data Loss: 0.15, PDE Loss: 5.2, Total: 0.48
```
- Balanced contributions
- Faster convergence
- Both losses decrease together

## References

- Wang, S., et al. "Understanding and mitigating gradient pathologies in physics-informed neural networks." (2021)
- Wang, S., et al. "Learning the solution operator of parametric partial differential equations with physics-informed DeepONets." (2021)

