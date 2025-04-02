# HydroNet Examples

This directory contains example scripts demonstrating how to use the HydroNet framework.

## Available Examples

- `example_usage.py`: Demonstrates the basic usage of all three main components of HydroNet:
  - DeepONet for learning the operator of shallow water equations
  - PINN for solving 2D shallow water equations
  - Physics-Informed DeepONet for combining data-driven and physics-informed approaches

## Running the Examples

To run the examples, make sure you have installed the HydroNet package:

```bash
# From the root directory of the repository
pip install -e .
```

Then, you can run the examples:

```bash
# From the examples directory
python example_usage.py
```

## Creating Your Own Examples

When creating your own examples, follow these steps:

1. Import the necessary modules from HydroNet
2. Create and configure the model using a configuration file or by directly setting parameters
3. Create a trainer for the model
4. Define any necessary functions (e.g., initial conditions, boundary conditions)
5. Train the model
6. Use the trained model for predictions

See the existing examples for reference implementations. 