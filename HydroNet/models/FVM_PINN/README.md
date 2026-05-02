# FVM-PINN

Finite-Volume-augmented Physics-Informed Neural Network for 2D Shallow Water
Equations on unstructured SRH-2D meshes.

FVM-PINN replaces the strong-form PDE residual used by a classical PINN
with a **differentiable well-balanced Roe finite-volume residual** evaluated
on each cell. The residual is the same operator that a finite-volume solver
would apply: it gathers left/right states across faces, applies the Roe
Riemann solver, scatters flux contributions back, and adds bed-slope + Manning
source terms. Wrapping it in autograd lets the PINN inherit FVM's discrete
conservation, shock-capture, and well-balancing properties while remaining
a mesh-free neural representation.

Reference: Liu, X. & Song, Y. (WRR) — well-balanced Roe solver formulation,
Rogers et al. (2001, 2003) / Liu et al. (2008) — source-term splitting.

## Public API

```python
from HydroNet import Config, FVM_SWE_PINN, FVM_PINNTrainer, FVM_PINNDataset

config  = Config("fvm_pinn_config.yaml")
model   = FVM_SWE_PINN(config)
dataset = FVM_PINNDataset(config)                  # reads SRH-2D files, builds mesh
trainer = FVM_PINNTrainer(model, dataset, config)  # dispatches to the chosen strategy
history, preds = trainer.train()

# Inference (matches HydroNet convention — returns [h, u, v] in SI units)
Q_phys = model(xyt)                                # public forward
```

All three public classes are re-exported from `HydroNet` top-level. The
internal network predicts perturbation-form conserved variables
`[xi, hu, hv]` where `xi = h - h_still`; the public `forward` converts to
`[h, u, v]` using the per-cell still-water depth `h_still`.

## Directory layout

```
HydroNet/models/FVM_PINN/
├── README.md                # this file
├── __init__.py              # re-exports FVM_SWE_PINN / FVM_PINNTrainer / FVM_PINNDataset
├── model.py                 # FVM_SWE_PINN — wraps _internal/pinn/network.SWENet
├── trainer.py               # FVM_PINNTrainer — wraps _internal/trainers/TrainerFactory
├── data.py                  # FVM_PINNDataset — reads SRH-2D OR takes a pre-built mesh
└── _internal/               # ported research code, shared across public wrappers
    ├── fvm/
    │   ├── riemann_solver.py    # Roe flux + well-balanced bed-slope source + Manning friction
    │   ├── geometry.py          # Green–Gauss S0, face / cell geometry tensors
    │   ├── smooth_ops.py        # AD-friendly smooth_abs / smooth_sqrt / smooth_pow
    │   └── time_stepping.py     # shared Heun RK2 (``run_fvm_rk2``) + CFL dt
    ├── mesh/
    │   ├── srh2d_reader.py      # .srhhydro / .srhgeom / .srhmat parser
    │   └── mesh_topology.py     # face-based UnstructuredMesh builder
    ├── pinn/
    │   ├── network.py           # SWENet (Fourier-feature MLP) + SirenSWENet
    │   └── loss.py              # FVMPINNLoss: FVM + IC + BC + data, per-component
    ├── trainers/
    │   ├── base_trainer.py      # shared Adam + L-BFGS loop, checkpointing
    │   ├── standard_trainer.py  # full-batch (classical PINN)
    │   ├── minibatch_trainer.py # cell mini-batching
    │   ├── window_trainer.py    # time-window decomposition
    │   └── teacher_trainer.py   # FVM-trajectory distillation
    └── utils/
        ├── vtk_writer.py        # VTK export (depth, velocity, bed, Manning, S0)
        └── run_logger.py
```

The ring fence around `_internal/` is intentional: it holds a verbatim port of a
standalone research codebase so it can evolve separately from the PINN /
PI-DeepONet modules. Users should only import from the three public files
(`model.py`, `trainer.py`, `data.py`) unless they need raw access (e.g., the
`*_fvm_only.py` drivers use `_internal/fvm/time_stepping.run_fvm_rk2` and
`_internal/utils/vtk_writer` directly).

## Training strategies

Selected via `training.strategy` in the YAML. All four slot into the same
`FVM_PINNTrainer(model, dataset, config)` constructor; only the underlying
loop changes.

| Strategy | What it does | When to use |
|---|---|---|
| `standard` | Full-batch classical PINN. Loss = `λ_fvm·FVM + λ_ic·IC + λ_bc·BC + λ_data·data`. | Small meshes, no time-horizon issues, no multi-steady-state ambiguity. |
| `minibatch` | Same loss, subset of cells per step (configurable fraction, auto-expands the face stencil). | Larger meshes where the autograd graph blows up peak memory. |
| `window` | Splits `[t_start, t_end]` into N overlapping sub-intervals; warm-starts each window's network from the previous one. | Long time horizons where a single network can't represent the full space-time solution. |
| `teacher` | Runs the FVM solver once from the IC to `t_end` (cached), then distils the network onto each snapshot + optional physics regulariser + optional anchor data. | Transcritical / multi-steady-state problems where a pure-residual PINN can land on a spurious branch. Recommended for most FVM-PINN use cases. |

`use_grad_checkpoint: true` is an orthogonal flag that activates PyTorch's
autograd activation checkpointing (trades ~30% compute for ~2–4× lower peak
memory); composable with any strategy.

## Normalization (differs from PI-DeepONet)

- **Input** `(x, y, t)` — **z-score** via the `SWENet.x_mean` / `x_std` buffers,
  set by the trainer after the mesh and snapshot times are known. Z-score is
  a better fit for unstructured mesh sample distributions (cells cluster where
  the mesh is refined) than the min-max that PI-DeepONet uses on its regular
  input grids.
- **Output** `[xi, hu, hv]` — **no normalization**. The internal network
  predicts physical SI units so the Roe solver can consume the output
  directly without per-call denormalization. Scale mismatches between `xi`
  (often ≪ 1 under the well-balanced form) and `hu / hv` are handled via
  explicit per-component loss weights:

  ```yaml
  training:
    component_weights:
      lambda_xi: 1.0
      lambda_hu: 1.0
      lambda_hv: 1.0
  ```

  These multiply the elementwise MSE inside the FVM residual, IC, data (and
  for teacher mode: distill, physics-residual) losses. Defaults 1.0 are
  uniform; raise `lambda_xi` if your case has `xi ≪ hu` (typical for
  shallow perturbations around a well-chosen `wse_still`).

## Well-balanced perturbation form

The model trains on the well-balanced perturbation form

```
Q = [xi, hu, hv]     with     xi = h - h_still
h_still = max(0, wse_still - bed_elev)   (per cell)
```

instead of the raw conserved variables `[h, hu, hv]`. At the rest state
`WSE = wse_still`, `xi ≡ 0` makes the Roe pressure flux
`0.5·g·(xi² + 2·xi·h_still)` and the bed-slope source `g·xi·∇z_b` both
vanish and cancel exactly on the Green-Gauss grid — i.e., the well-balanced
property is exact, and static water over arbitrary bathymetry stays static
to machine precision.

Setting `physics.wse_still` close to the expected operating WSE is what
makes this useful. Good defaults:
- **Dam-break / flat bed problems**: `wse_still = 0` (standard SWE form).
- **Steady river reaches**: `wse_still = exit_wse` (downstream stage).
- **Bump / transcritical**: `wse_still = downstream_subcritical_depth`.

See [`HydroNet/config/fvm_pinn_config.yaml`](../../config/fvm_pinn_config.yaml)
for the full schema with comments.

## Example cases

Four reference cases are bundled under `examples/FVM_PINN/`. Each has a
**shared `fvm_pinn_config.yaml`** and **two entry scripts**: a pure-FVM baseline
and a PINN trainer.

| Case | Domain | Reference | PINN script |
|---|---|---|---|
| [`dam_break_1d`](../../../examples/FVM_PINN/dam_break_1d/) | 1D strip, Riemann IC | Stoker analytical solution | `dam_break_1d_FVM_PINN.py` (strategy=standard) |
| [`channel_with_bump`](../../../examples/FVM_PINN/channel_with_bump/) | 1D transcritical over a bump | FullSWOF analytical + SRH-2D | `channel_with_bump_FVM_PINN.py` (strategy=teacher) |
| [`block_in_channel`](../../../examples/FVM_PINN/block_in_channel/) | 2D wake around a block, 1326 cells | SRH-2D XMDFC h5 | `block_in_channel_FVM_PINN.py` (strategy=teacher) |
| [`savannah_river`](../../../examples/FVM_PINN/savannah_river/) | 2D real river reach, 1306 cells | SRH-2D XMDFC h5 (5 snapshots) | `savannah_river_FVM_PINN.py` (strategy=teacher) |

Every PINN script supports:

```bash
python <case>_FVM_PINN.py                       # train + evaluate
python <case>_FVM_PINN.py --device cuda          # override device
python <case>_FVM_PINN.py --post-only            # reload final checkpoint, skip training
python <case>_FVM_PINN.py --checkpoint path.pt   # load a specific checkpoint
```

The `*_fvm_only.py` drivers share the same YAML but skip all neural-network
training; they write VTKs at evenly spaced snapshots and compare the final
state to the case's reference solution. Use them to set the **accuracy floor**
(teacher mode can at best match the FVM-only result, since the teacher IS
the FVM solution).

## Extending

- **New case** → add `examples/FVM_PINN/<name>/` with `data/` (SRH-2D files),
  a `fvm_pinn_config.yaml` (copy/tweak from an existing case), and two entry
  scripts. If the mesh is synthetic (no SRH-2D files), use
  `FVM_PINNDataset.from_mesh(config, mesh, ic_data, ...)` — see the
  `dam_break_1d` case.
- **New boundary type** → extend `_internal/fvm/riemann_solver._build_ghost_states`
  and the soft-BC branch of `_internal/pinn/loss._bc_loss`. Wire the type
  through `FVM_PINNDataset._build_bc_ghost` / `_build_bc_data`.
- **New training strategy** → add a trainer class to `_internal/trainers/`
  following the `setup(ic, bc, ref) → train() → predict(xyt)` shape,
  register it in `_internal/trainers/__init__._TRAINER_REGISTRY`, and
  handle it in `FVM_PINNTrainer.__init__`'s strategy dispatch.

## Known limits

- **Python 3.10+** required (type-hint syntax in some internal files).
- **Teacher trajectory generation is sequential in time** — doesn't parallelize
  to GPU well for small meshes; expect kernel-launch-bound throughput on
  <2000 cells.
- **Classical PINN (strategy ≠ teacher) can land on spurious multi-steady-state
  branches** for transcritical flows (the bump case is a known example).
  Use `teacher` for those.
- **Inlet-q BC distributes discharge by Manning conveyance** proportionally to
  the current interior depth. On poor initial conditions (uneven IC at the
  inlet face), this can feed back and blow up — cold flat-WSE ICs are the
  safest starting point.
- **SRH-2D is the only mesh reader** currently. GMSH-native support would be
  straightforward to add inside `_internal/mesh/`.
