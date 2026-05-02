"""
FVM-PINN example: 1D dam-break validation.

Classical Riemann problem with h_L/h_R initial discontinuity over a flat
frictionless bed. The analytical Stoker rarefaction-shock solution provides
ground truth — no SRH-2D reference needed.

Pipeline
--------
1. Build a 1-cell-wide strip mesh programmatically and wrap it in the
   internal UnstructuredMesh dataclass.
2. Evaluate the Stoker IC at cell centres.
3. Hand the mesh + IC to ``FVM_PINNDataset.from_mesh()``.
4. Instantiate ``FVM_SWE_PINN`` + ``FVM_PINNTrainer`` via ``Config``.
5. Train, then compare the prediction at ``t_end`` with Stoker.

Usage
-----
    cd examples/FVM_PINN/dam_break_1d
    python dam_break_1d_FVM_PINN.py                # CPU
    python dam_break_1d_FVM_PINN.py --device cuda   # GPU (CLI override)
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

# Make HydroNet importable when running this script directly.
script_path = os.path.abspath(__file__)
examples_dir = os.path.dirname(os.path.dirname(os.path.dirname(script_path)))
project_root = os.path.dirname(examples_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from HydroNet import Config, FVM_SWE_PINN, FVM_PINNTrainer, FVM_PINNDataset
# The internal mesh module isn't part of the public HydroNet API, but a
# synthetic-mesh case like the dam-break legitimately needs it. We import
# directly — none of the other 3 cases (bump, block, Savannah) need this.
from HydroNet.models.FVM_PINN._internal.mesh.mesh_topology import build_mesh
from HydroNet.models.FVM_PINN._internal.mesh.srh2d_reader import SRH2DRawData

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

G = 9.81


# ---------------------------------------------------------------------------
# Analytical Stoker dam-break solution (wet bed, h_L > h_R > 0)
# ---------------------------------------------------------------------------

def dam_break_exact(x: np.ndarray, t: float, h_L: float, h_R: float,
                    x_dam: float):
    """Stoker's exact 1D rarefaction-shock solution for a flat frictionless bed."""
    c_L = np.sqrt(G * h_L)
    if t == 0:
        return np.where(x < x_dam, h_L, h_R), np.zeros_like(x)

    # Newton iteration for h2 (depth behind shock).
    h2 = h_L
    for _ in range(50):
        c2 = np.sqrt(G * h2)
        f = (2 * (c_L - c2)
             - (h2 - h_R) * np.sqrt(G * (h2 + h_R) / (2 * h2 * h_R + 1e-30)))
        dh = 1e-6
        c2_p = np.sqrt(G * (h2 + dh))
        f_p = (2 * (c_L - c2_p)
               - (h2 + dh - h_R) * np.sqrt(G * (h2 + dh + h_R) /
                                           (2 * (h2 + dh) * h_R + 1e-30)))
        df = (f_p - f) / dh
        if abs(df) < 1e-30:
            break
        h2 = max(h2 - f / df, 1e-6)

    c2 = np.sqrt(G * h2)
    u2 = 2 * (c_L - c2)
    W = (h2 * u2) / (h2 - h_R + 1e-30)   # shock speed

    h = np.zeros_like(x)
    u = np.zeros_like(x)
    for i, xi in enumerate(x):
        xr = xi - x_dam
        if xr / t < -c_L:
            h[i], u[i] = h_L, 0.0
        elif xr / t < u2 - c2:
            u_fan = 2.0 / 3.0 * (xr / t + c_L)
            c_fan = c_L - u_fan / 2.0
            h[i], u[i] = c_fan ** 2 / G, u_fan
        elif xr / t < W:
            h[i], u[i] = h2, u2
        else:
            h[i], u[i] = h_R, 0.0
    return h, u


# ---------------------------------------------------------------------------
# Programmatic 1D strip mesh
# ---------------------------------------------------------------------------

def build_1d_strip_mesh(L: float, n_cells: int, width: float = 1.0):
    """
    Build a 1-cell-wide strip of quads as a synthetic UnstructuredMesh.
    Bed elevation is zero everywhere; Manning's n is zero (frictionless).
    """
    x_nodes = np.linspace(0.0, L, n_cells + 1)
    y_bot = np.zeros(n_cells + 1)
    y_top = np.full(n_cells + 1, width)

    nodes_xy = np.vstack([
        np.column_stack([x_nodes, y_bot]),      # nodes 0..n
        np.column_stack([x_nodes, y_top]),      # nodes n+1..2n+1
    ])
    nodes_z = np.zeros(len(nodes_xy))

    n_per_row = n_cells + 1
    elem_nodes = [
        np.array([i, i + 1, i + 1 + n_per_row, i + n_per_row])
        for i in range(n_cells)
    ]
    elem_material = np.zeros(n_cells, dtype=np.int64)
    manning_n = {0: 0.0}

    raw = SRH2DRawData()
    raw.node_coords = np.column_stack([nodes_xy, nodes_z])
    raw.elem_nodes = elem_nodes
    raw.elem_material = elem_material
    raw.manning_n = manning_n
    # raw.node_strings stays empty — strip ends keep face_bc_id == -1 and
    # fall through to the FVM residual's default reflective ghost.
    return build_mesh(raw)


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def build_ic_from_case(config, mesh):
    """Riemann initial condition in perturbation form."""
    h_L = float(config.get_required_config('case.h_L'))
    h_R = float(config.get_required_config('case.h_R'))
    x_dam = float(config.get_required_config('case.x_dam'))
    wse_still = float(config.get_required_config('physics.wse_still'))

    xc = mesh.cell_center[:, 0]
    yc = mesh.cell_center[:, 1]
    n = mesh.n_cells
    h_init, u_init = dam_break_exact(xc, 0.0, h_L, h_R, x_dam)

    # Bed is zero → h_still = max(0, wse_still - 0) = wse_still (uniform).
    # xi = h - h_still. With wse_still=0 (default), xi = h.
    xi_init = h_init - wse_still

    xyt = torch.column_stack([
        torch.tensor(xc, dtype=torch.float64),
        torch.tensor(yc, dtype=torch.float64),
        torch.zeros(n, dtype=torch.float64),
    ])
    U_true = torch.stack([
        torch.tensor(xi_init,            dtype=torch.float64),
        torch.tensor(h_init * u_init,    dtype=torch.float64),
        torch.zeros(n,                   dtype=torch.float64),
    ], dim=-1)
    return {"xyt": xyt, "U_true": U_true}


def evaluate_and_plot(model, config, mesh, out_dir: Path):
    """Compare the trained model with Stoker's analytical solution at t_end."""
    h_L = float(config.get_required_config('case.h_L'))
    h_R = float(config.get_required_config('case.h_R'))
    x_dam = float(config.get_required_config('case.x_dam'))
    t_end = float(config.get_required_config('training.t_end'))

    xc = mesh.cell_center[:, 0]
    yc = mesh.cell_center[:, 1]
    n = mesh.n_cells

    # Analytical reference
    h_exact, u_exact = dam_break_exact(xc, t_end, h_L, h_R, x_dam)

    # PINN prediction — public forward returns [h, u, v]
    device = model.get_device()
    xyt = torch.column_stack([
        torch.tensor(xc, dtype=torch.float64, device=device),
        torch.tensor(yc, dtype=torch.float64, device=device),
        torch.full((n,), t_end, dtype=torch.float64, device=device),
    ])
    with torch.no_grad():
        Q_phys = model(xyt).cpu().numpy()
    h_pred = Q_phys[:, 0]
    u_pred = Q_phys[:, 1]

    # L2 / max errors
    l2_h = float(np.sqrt(np.mean((h_pred - h_exact) ** 2)))
    max_h = float(np.max(np.abs(h_pred - h_exact)))
    l2_u = float(np.sqrt(np.mean((u_pred - u_exact) ** 2)))
    max_u = float(np.max(np.abs(u_pred - u_exact)))
    logger.info(f"h: L2={l2_h:.4e}, max={max_h:.4e}")
    logger.info(f"u: L2={l2_u:.4e}, max={max_u:.4e}")

    # Profiles
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(xc, h_exact, "k-",  lw=2, label="Stoker (exact)")
    axes[0].plot(xc, h_pred,  "r--", lw=2, label="FVM-PINN")
    axes[0].set_xlabel("x [m]"); axes[0].set_ylabel("h [m]")
    axes[0].set_title(f"Water depth at t = {t_end} s")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(xc, u_exact, "k-",  lw=2, label="Stoker (exact)")
    axes[1].plot(xc, u_pred,  "r--", lw=2, label="FVM-PINN")
    axes[1].set_xlabel("x [m]"); axes[1].set_ylabel("u [m/s]")
    axes[1].set_title(f"Velocity at t = {t_end} s")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()

    path = out_dir / "dam_break_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Profile plot: {path}")

    # Also save raw predictions + reference for downstream analysis
    np.savez(
        out_dir / "predictions.npz",
        x=xc, h_pred=h_pred, u_pred=u_pred,
        h_exact=h_exact, u_exact=u_exact,
        L2_h=l2_h, max_h=max_h, L2_u=l2_u, max_u=max_u,
    )

    return {"L2_h": l2_h, "max_h": max_h, "L2_u": l2_u, "max_u": max_u}


def plot_loss_history(history: dict, out_dir: Path) -> None:
    """Three-panel loss plot: aggregate + per-component FVM + per-component IC."""
    def _nonempty(key):
        s = history.get(key) or []
        return s if (s and any(v > 0 for v in s)) else None

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # Panel 1 — aggregate
    ax = axes[0]
    for key in ("total", "fvm", "ic", "bc", "data"):
        s = _nonempty(key)
        if s:
            ax.semilogy(range(1, len(s) + 1), s, label=key)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Aggregate losses"); ax.legend(); ax.grid(True, alpha=0.3)

    # Panel 2 — per-SWE-component FVM residual
    ax = axes[1]
    for key in ("fvm_xi", "fvm_hu", "fvm_hv"):
        s = _nonempty(key)
        if s:
            ax.semilogy(range(1, len(s) + 1), s, label=key.split("_", 1)[1])
    ax.set_xlabel("Epoch"); ax.set_ylabel("FVM residual MSE")
    ax.set_title("FVM residual per SWE component")
    ax.legend(); ax.grid(True, alpha=0.3)

    # Panel 3 — per-SWE-component IC fit
    ax = axes[2]
    for key in ("ic_xi", "ic_hu", "ic_hv"):
        s = _nonempty(key)
        if s:
            ax.semilogy(range(1, len(s) + 1), s, label=key.split("_", 1)[1])
    ax.set_xlabel("Epoch"); ax.set_ylabel("IC MSE")
    ax.set_title("IC loss per SWE component")
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = out_dir / "loss_history.png"
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    logger.info(f"Loss plot: {path}")


def _default_checkpoint_path(config: Config) -> Path:
    """Best-guess checkpoint path from ``training.strategy`` naming."""
    strategy = str(config.get("training.strategy", "standard"))
    ckpt_dir = Path(str(
        config.get("training.logging.checkpoint_dir", "./checkpoints")
    ))
    if strategy == "teacher":
        return ckpt_dir / "teacher_final.pt"
    return ckpt_dir / "ckpt_final.pt"


def _load_checkpoint_into_model(model: FVM_SWE_PINN, ckpt_path: Path) -> dict:
    """Load ``network_state`` / ``x_mean`` / ``x_std`` from a trainer
    checkpoint directly onto ``model``'s internal SWENet. Returns the
    stored ``history`` dict (may be empty)."""
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No checkpoint at {ckpt_path}. Either train first (drop the "
            "--post-only flag) or pass --checkpoint <path>."
        )
    ck = torch.load(ckpt_path, map_location=model.get_device(), weights_only=False)
    net = model.get_internal_network()
    net.load_state_dict(ck["network_state"])
    net.set_normalisation(ck["x_mean"], ck["x_std"])
    logger.info(f"Loaded checkpoint: {ckpt_path}")
    return ck.get("history", {})


def main() -> None:
    parser = argparse.ArgumentParser(description="FVM-PINN 1D dam-break example")
    parser.add_argument(
        "--config", default="fvm_pinn_config.yaml",
        help="Path to the YAML config (default: fvm_pinn_config.yaml)",
    )
    parser.add_argument(
        "--device", default=None, choices=["cpu", "cuda"],
        help="Override device.type from the config",
    )
    parser.add_argument(
        "--post-only", action="store_true",
        help="Skip training; load an existing trainer checkpoint and only "
             "run evaluation + plotting.",
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="Explicit checkpoint path (default: based on training.strategy).",
    )
    args = parser.parse_args()

    np.random.seed(123456)
    torch.manual_seed(123456)

    wall_start = time.time()
    config = Config(args.config)
    if args.device:
        config.set("device.type", args.device)

    # ---- Build strip mesh + analytical IC ----
    L = float(config.get('case.L', 20.0))
    n_cells = int(config.get('case.n_cells', 100))
    width = float(config.get('case.width', 1.0))
    logger.info(f"Mesh: 1D strip, L={L} m, n_cells={n_cells}, width={width}")
    mesh = build_1d_strip_mesh(L, n_cells, width)
    ic_data = build_ic_from_case(config, mesh)

    # ---- Build model + dataset (needed by both paths) ----
    model = FVM_SWE_PINN(config)
    logger.info(f"Model on device={model.get_device()}, dtype={model.dtype}")

    dataset = FVM_PINNDataset.from_mesh(config, mesh, ic_data=ic_data)

    if args.post_only:
        # Wire the mesh context onto the model (normally FVM_PINNTrainer
        # does this in its __init__). Then load the checkpoint.
        model.set_mesh_context(
            h_still_cells=dataset.get_h_still(),
            cell_xy=dataset.get_cell_xy(),
        )
        ckpt_path = Path(args.checkpoint) if args.checkpoint else _default_checkpoint_path(config)
        history = _load_checkpoint_into_model(model, ckpt_path)
    else:
        trainer = FVM_PINNTrainer(model, dataset, config)
        logger.info(f"Trainer: strategy={trainer.strategy}")
        history, _ = trainer.train()

    # ---- Evaluate / plot ----
    out_dir = Path("plots")
    metrics = evaluate_and_plot(model, config, mesh, out_dir)
    if history:
        plot_loss_history(history, out_dir)

    # Training history dump (matches the PINN example idiom)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    hist_file = Path(f"history_fvm_pinn_{ts}.json")
    with open(hist_file, "w") as f:
        json.dump({"history": history, "metrics": metrics}, f, indent=2)
    logger.info(f"Training history: {hist_file}")

    logger.info(f"Total wall time: {time.time() - wall_start:.1f} s")
    logger.info("Done.")


if __name__ == "__main__":
    main()
