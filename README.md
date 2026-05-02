# HydroNet

Physics-informed machine learning framework for solving 2D shallow water equations using deep learning.

## Overview

HydroNet is a PyTorch-based framework that implements three physics-informed neural network families for 2D shallow water equations:

- **PINN** — classical physics-informed neural network for a single case (PDE residual at cell centres).
- **PI-DeepONet** — operator-learning variant that generalises across parameterised cases (discharge, geometry).
- **FVM-PINN** — hybrid that replaces the strong-form PDE residual with a **differentiable well-balanced Roe finite-volume residual** on unstructured SRH-2D meshes; supports standard / mini-batch / time-window / FVM-trajectory-teacher training strategies.

All three share a common `Config`/`Dataset`/`Trainer`/`Model` pattern and output `[h, u, v]` in SI units.

## Features

- **PINN** (`HydroNet.SWE_PINN`): direct SWE solve with PDE + BC + IC + data losses.
- **PI-DeepONet** (`HydroNet.PI_SWE_DeepONetModel`): branch+trunk operator learning with toggleable physics loss.
- **FVM-PINN** (`HydroNet.FVM_SWE_PINN`): well-balanced Roe solver on unstructured meshes, Manning friction, bed-slope source, perturbation-form conserved vars, four training strategies (`standard` / `minibatch` / `window` / `teacher`).
- **Flexible YAML configuration** for model architecture, physics, BCs, training budget.
- **Mesh I/O**: GMSH, VTK, and SRH-2D (`.srhhydro`/`.srhgeom`/`.srhmat` + XMDFC h5).
- **Visualization utilities** for solution fields and training history.

## Installation

### Requirements

- Python ≥ 3.10 (tested on 3.12)
- Git (for cloning)
- Optional: NVIDIA GPU with CUDA 12.x for GPU training

External dependency (installed separately, see below):

- **pyHMT2D** — SRH-2D I/O helpers (<https://github.com/psu-efd/pyHMT2D>)

### Recommended setup: local virtual environment

A project-local `.venv` is the recommended setup and is what the example scripts assume.

**1. Clone and create the virtual environment:**

```bash
git clone https://github.com/psu-efd/HydroNet.git
cd HydroNet

# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python -m venv .venv
source .venv/bin/activate
```

**2. Upgrade pip / setuptools / wheel:**

```bash
python -m pip install --upgrade pip setuptools wheel
```

**3. Install HydroNet in editable mode:**

```bash
pip install -e .
```

This pulls in all runtime dependencies (`torch`, `numpy`, `scipy`, `pandas`, `matplotlib`, `plotly`, `scikit-learn`, `vtk`, `xarray`, `netCDF4`, `meshio`, `h5py`, `geopandas`, `shapely`, `pyogrio`, `pyproj`, `tqdm`, `pyyaml`, `tensorboard`, `gmsh`).

**4. Install `pyHMT2D` (editable from a local clone):**

```bash
git clone https://github.com/psu-efd/pyHMT2D.git ../pyHMT2D
pip install -e ../pyHMT2D --no-deps
```

### Optional: CUDA-enabled PyTorch

`pip install -e .` installs the CPU-only `torch` wheel by default on Windows. For GPU training, replace it with the CUDA build that matches your driver:

```bash
pip uninstall -y torch
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

(Swap `cu128` for `cu121` or `cu118` if your driver is older. Check available wheels at <https://pytorch.org/get-started/locally/>.)

Verify:

```bash
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

### Optional: development extras

```bash
pip install -e ".[dev]"
```

Adds `pytest`, `black`, `flake8`, `mypy`, `sphinx` — note that no test / lint targets are wired up yet.

### IDE setup

In VS Code / PyCharm / similar: open the HydroNet folder as the project root, then select `./.venv/Scripts/python.exe` (Windows) or `./.venv/bin/python` (macOS / Linux) as the Python interpreter. Most IDEs auto-detect the local `.venv`.

## Quick Start

Examples are self-contained scripts. Each example owns its own YAML config alongside the entry script; paths inside the YAML are resolved relative to the script's directory.

**PINN:**

```bash
cd examples/PINN/block_in_channel_steady
python block_in_channel_PINN.py
```

**PI-DeepONet:**

```bash
cd examples/PI_DeepONet/block_in_channel_steady/block_in_channel_steady_PI_DeepONet
python train_val_test.py
```

**FVM-PINN:**

```bash
# Dam-break 1D (analytical Stoker solution; no SRH-2D data needed)
cd examples/FVM_PINN/dam_break_1d
python dam_break_1d_FVM_PINN.py

# 1D transcritical flow over a bump (FullSWOF reference)
cd examples/FVM_PINN/channel_with_bump
python channel_with_bump_fvm_only.py              # pure-FVM baseline
python channel_with_bump_FVM_PINN.py              # teacher PINN

# 2D block-in-channel wake (SRH-2D reference)
cd examples/FVM_PINN/block_in_channel
python block_in_channel_fvm_only.py --device cuda

# Savannah River reach (~1 km real-world case)
cd examples/FVM_PINN/savannah_river
python savannah_river_fvm_only.py --device cuda
```

Each FVM-PINN case provides both an `<case>_fvm_only.py` driver (pure Heun RK2 on the Roe solver) and an `<case>_FVM_PINN.py` driver (PINN training + evaluation). The two scripts share a single `fvm_pinn_config.yaml`. Every PINN script supports `--post-only` to regenerate plots from a saved checkpoint without retraining.

## Project Structure

```
HydroNet/
├── HydroNet/
│   ├── __init__.py               # public API (SWE_PINN, PI_SWE_DeepONetModel,
│   │                             #             FVM_SWE_PINN, *Trainer, *Dataset, Config)
│   ├── models/
│   │   ├── PINN/                 # SWE_PINN + PINNTrainer + PINNDataset
│   │   ├── PI_DeepONet/          # PI_SWE_DeepONetModel + trainer + dataset
│   │   └── FVM_PINN/             # FVM_SWE_PINN + FVM_PINNTrainer + FVM_PINNDataset
│   │       └── _internal/        #   ported well-balanced Roe solver, mesh
│   │                             #   topology, SRH-2D reader, trainer family
│   ├── utils/                    # config, mesh I/O, visualization, prediction
│   └── config/ configs/          # reference YAMLs (per model family)
├── examples/
│   ├── PINN/                     # PINN case runners
│   ├── PI_DeepONet/              # PI-DeepONet case runners
│   └── FVM_PINN/                 # FVM-PINN case runners
│       ├── dam_break_1d/
│       ├── channel_with_bump/
│       ├── block_in_channel/
│       └── savannah_river/
├── requirements.txt              # dep snapshot (not used by pip)
└── setup.py                      # package setup + runtime install_requires
```

## Configuration

Each model family has its own YAML schema. Common sections:

- **`device`**: `type` (`cuda` / `cpu`) and `index` (GPU ID).
- **`model`**: architecture (name, hidden_dim, n_layers, activation, etc.).
- **`physics`**: gravity, Manning coefficient, domain scales, steady/unsteady flag.
- **`training`**: optimizer (Adam + optional L-BFGS polish), epoch budget, loss weights, strategy.
- **`boundary_conditions`**: per-BC-id `type` (`inlet-q` / `exit-h` / `wall` / `symmetry`) and `value`.

Reference YAMLs in `HydroNet/config/` document every available knob:

- `pinn_config.yaml`
- `pi_deeponet_config.yaml`
- `fvm_pinn_config.yaml`

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

@software{hydronet2026,
  title={HydroNet: Physics-informed machine learning for hydrodynamics modeling},
  author={Liu, Xiaofeng},
  year={2026},
  url={https://github.com/psu-efd/HydroNet}
}
```
