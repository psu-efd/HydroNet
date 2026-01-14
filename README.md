# HydroNet

Physics-informed machine learning framework for solving 2D shallow water equations using deep learning.

## Overview

HydroNet is a PyTorch-based framework that implements physics-informed neural networks (PINN) and physics-informed DeepONet (PI-DeepONet) for solving 2D shallow water equations. The framework enables operator learning and provides tools for training, testing, and applying models to hydrodynamics and hydraulics problems.

## Features

- **Physics-Informed Neural Networks (PINN)**: Direct solution of PDEs using neural networks with physics constraints
- **Physics-Informed DeepONet (PI-DeepONet)**: Operator learning for parameterized PDE solutions
- **2D Shallow Water Equations**: Specialized implementation for hydrodynamics modeling
- **Flexible Configuration**: YAML-based configuration system for easy model customization
- **Data Processing**: Utilities for handling GMSH meshes, VTK files, and SRH-2D simulation data
- **Visualization**: Tools for plotting solutions and training history

## Installation

### Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- NumPy, SciPy, Pandas
- VTK >= 9.2.0 (for mesh processing)
- pyHMT2D (for SRH-2D simulation data; https://github.com/psu-efd/pyHMT2D)
- Additional dependencies listed in `requirements.txt`

### Install from Source

```bash
git clone https://github.com/psu-efd/HydroNet.git
cd HydroNet
pip install -e .
```

For development dependencies:

```bash
pip install -e ".[dev]"
```

## Quick Start

See the `examples/` directory for complete working examples:
- `examples/PINN/`: PINN model examples
- `examples/PI_DeepONet/`: PI-DeepONet model examples

## Project Structure

```
HydroNet/
├── HydroNet/
│   ├── models/
│   │   ├── PINN/          # PINN model implementation
│   │   └── PI_DeepONet/   # PI-DeepONet model implementation
│   ├── utils/             # Data processing and utility functions
│   └── configs/           # Configuration files
├── examples/              # Example scripts and data
├── requirements.txt       # Python dependencies
└── setup.py              # Package setup
```

## Examples

See the `examples/` directory for complete working examples:
- `examples/PINN/`: PINN model examples
- `examples/PI_DeepONet/`: PI-DeepONet model examples

## Configuration

Models are configured using YAML files. Key configuration sections include:
- **Model architecture**: Network layers, activation functions, dropout
- **Physics parameters**: Gravity, scales, steady/transient settings
- **Training parameters**: Learning rate, batch size, loss weights, schedulers
- **Data paths**: Training, validation, and test data locations

See `examples` for example configurations.

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Author

Xiaofeng Liu (xiaofengliu19@gmail.com)

## Citation

If you use HydroNet in your research, please cite:

```bibtex
@misc{liu2026,
      title={Physics-Informed Deep Operator Learning for Computational Hydraulics Modeling}, 
      author={Xiaofeng Liu and Yong G. Lai},
      year={2026},
      eprint={2601.08086},
      archivePrefix={arXiv},
      primaryClass={physics.flu-dyn},
      url={https://arxiv.org/abs/2601.08086}, 
}

@software{hydronet2025,
  title={HydroNet: Physics-informed machine learning for hydrodynamics modeling},
  author={Liu, Xiaofeng},
  year={2026},
  url={https://github.com/psu-efd/HydroNet}
}
```
