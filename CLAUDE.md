# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

HydroNet is a PyTorch framework for solving the 2D shallow water equations (SWE) with physics-informed deep learning. It ships three model families that share a common Config/Dataset/Trainer pattern:

- **PINN** (`HydroNet/models/PINN/`) — `SWE_PINN` solves a single SWE problem directly. Loss = PDE residual + boundary + (initial) + data. Boundary types `inlet-q`, `exit-h`, `wall` are wired in the model itself, configured under `boundary_conditions:` in the YAML.
- **PI-DeepONet** (`HydroNet/models/PI_DeepONet/`) — `PI_SWE_DeepONetModel` learns the SWE *operator* across many cases. Branch net encodes input functions (e.g. discharge sensors), trunk net encodes `(x, y)` coordinates. Physics loss is toggled via `model.use_physics_loss` in the YAML.
- **FVM-PINN** (`HydroNet/models/FVM_PINN/`) — `FVM_SWE_PINN` replaces the strong-form PDE residual with a **differentiable well-balanced Roe finite-volume residual** on unstructured SRH-2D meshes (see `_internal/fvm/`). The internal network predicts perturbation-form conserved variables `[xi, hu, hv]` where `xi = h - h_still`; the public `forward` wraps to `[h, u, v]`. Four training strategies via `training.strategy`: `standard` / `minibatch` / `window` / `teacher` (FVM trajectory distillation). Dataset reads raw SRH-2D files directly (no preprocessing step).

All three models output `[h, u, v]` (depth and velocity components) in SI units.

## Environment

HydroNet is developed on **Python 3.12 in a project-local `.venv`** (not conda). Standard flow:

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows (or: source .venv/bin/activate on macOS/Linux)
pip install --upgrade pip setuptools wheel
pip install -e .                 # installs all runtime deps
pip install -e ../pyHMT2D --no-deps    # external SRH-2D helper (editable from sibling clone)
```

For GPU: `pip uninstall -y torch && pip install torch --index-url https://download.pytorch.org/whl/cu128`.

## Common commands

```bash
pip install -e .              # install package in editable mode
pip install -e ".[dev]"       # + pytest, black, flake8, mypy, sphinx
```

Examples are self-contained scripts run from their own directory (paths inside the YAML config are resolved relative to CWD):

```bash
# PINN
cd examples/PINN/block_in_channel_steady && python block_in_channel_PINN.py

# PI-DeepONet
cd examples/PI_DeepONet/block_in_channel_steady/block_in_channel_steady_PI_DeepONet && python train_val_test.py

# FVM-PINN (each case provides an *_fvm_only.py baseline and a *_FVM_PINN.py trainer;
# they share a single fvm_pinn_config.yaml)
cd examples/FVM_PINN/channel_with_bump && python channel_with_bump_fvm_only.py
cd examples/FVM_PINN/savannah_river && python savannah_river_FVM_PINN.py --device cuda
python savannah_river_FVM_PINN.py --post-only    # reload checkpoint, skip training
```

There is no test suite or lint config wired up — `pytest`/`black`/`flake8`/`mypy` are listed as dev extras but no targets are defined.

## Architecture notes that span files

- **Public API is curated in `HydroNet/__init__.py`.** Examples import top-level (`from HydroNet import SWE_PINN, PINNTrainer, PINNDataset, Config, ...`). When adding a new user-facing class or helper, re-export it there or examples will need deep imports.
- **Config is a dotted-key wrapper around YAML** (`HydroNet/utils/config.py`). Use `config.get_required_config('model.name')` for required keys (raises) and `config.get('foo.bar', default)` for optional. Models validate `model.name` matches the expected string (e.g. `SWE_PINN`, `PI_SWE_DeepONet`) and will raise otherwise.
- **Device selection** is config-driven: `device.type` (`cuda`/`cpu`) and `device.index` are read inside each model's `__init__`. Don't add ad-hoc `.cuda()` calls — let the model's `self.device` propagate.
- **PI-DeepONet workflow is wrapped in `HydroNet/utils/pi_deeponet_utils.py`.** The example scripts mostly call `pi_deeponet_train(config)`, `pi_deeponet_test(checkpoint, config, case_indices=...)`, `pi_deeponet_plot_training_history(history_json)`, plus the `pi_deeponet_application_*` trio for inference on a new mesh. PINN examples have no equivalent wrappers — they instantiate `SWE_PINN` + `PINNDataset` + `PINNTrainer` directly and call `trainer.train()`.
- **Two config dirs exist** (`HydroNet/config/` and `HydroNet/configs/`) but neither is loaded by the package — each example carries its own YAML alongside the script. Treat the in-package YAMLs as references only; don't assume they're the source of truth.
- **Loss weighting** has shared scaffolding: PINN uses `loss_weight_scheduler.py` (`ConstantWeightScheduler`, `ManualWeightScheduler`, `GradNormScheduler`, `SoftAdaptScheduler`); both models support `use_adaptive_balancing` / `use_adaptive_pde_component_balancing` flags under `training.loss_weights`. When tuning loss behavior, check whether the change belongs in the scheduler or the trainer's weight-update step.
- **Training data comes from SRH-2D** (external solver via `pyHMT2D`). The data-generation pipeline lives under `examples/.../data_generation/` and `examples/data_case_preparation/` — these prepare cases, run SRH-2D in parallel, and post-process VTK outputs into the JSON/HDF5 inputs the datasets expect. Treat dataset files as opaque; regenerating them is a separate (and slow) workflow.
- **Mesh I/O uses VTK + GMSH.** Predictions for visualization go through `predict_on_vtk2d_mesh` / `predict_on_gmsh2d_mesh` (in `HydroNet/utils/`) which handle nodal vs cell-centered output and denormalization via the dataset's `mesh_stats` / `data_stats`.
- **FVM-PINN internals live under `models/FVM_PINN/_internal/`.** This is a verbatim port of a standalone research codebase, ring-fenced so it can evolve separately from the PINN / PI-DeepONet pieces. Key pieces:
  - `_internal/fvm/riemann_solver.py` — Roe Riemann solver + bed-slope source + Manning friction, well-balanced perturbation form.
  - `_internal/fvm/time_stepping.py` — shared Heun RK2 stepper (`run_fvm_rk2`) used by both the `*_fvm_only.py` drivers and the `TeacherTrainer`'s trajectory generator. Single source of truth — don't inline a separate RK2 loop.
  - `_internal/mesh/srh2d_reader.py` + `mesh_topology.py` — parses `.srhhydro`/`.srhgeom`/`.srhmat` and builds an `UnstructuredMesh` with face-based topology.
  - `_internal/pinn/network.py` + `loss.py` — `SWENet` (Fourier-feature MLP) + `FVMPINNLoss` (physics + IC + BC + data, elementwise per conserved variable with per-component weights `lambda_xi` / `lambda_hu` / `lambda_hv`).
  - `_internal/trainers/` — `BaseTrainer`, `StandardTrainer`, `MiniBatchTrainer`, `TimeWindowTrainer`, `TeacherTrainer`, plus the `TrainerFactory` dispatch.
  Public wrappers (`model.py` / `trainer.py` / `data.py` in `FVM_PINN/`) translate HydroNet-style `Config` objects into the internal dataclass configs and convert `[xi, hu, hv] → [h, u, v]` at the public boundary.
- **FVM-PINN uses z-score input normalization** (not min-max like PI-DeepONet) because unstructured meshes have non-uniform sample distributions. **No output normalization** — the Roe solver consumes physical SI units directly; scale mismatches between `xi` and `hu/hv` are handled via per-component loss weights (`training.component_weights.*` in the YAML).
- **FVM-PINN checkpoints** include `network_state`, `x_mean`, `x_std`, `history`. Loading into a fresh model requires setting `x_mean`/`x_std` before inference (see the `--post-only` paths in the example scripts for the reference pattern). The `FVM_SWE_PINN.set_mesh_context(h_still, cell_xy)` method must also be called (normally done by `FVM_PINNTrainer.__init__`, explicit under `--post-only`) so the public `forward` can recover physical `h` from the internal `xi`.

## Project conventions (from `.cursorrules`)

- PyTorch-first, OOP for models / functional for data pipelines.
- Use Xavier init + explicit activation choice in new layers (matches `FCLayer` in `PI_DeepONet/model.py`).
- Hyperparameters belong in YAML, never hardcoded in scripts.
