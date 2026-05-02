"""
FVM-PINN example: 2D block-in-channel (teacher mode).

A rectangular channel with a block obstruction. 2D unsteady flow with a
recirculation wake behind the block that settles into quasi-steady state
by t = 360 s. Reference: SRH-2D XMDFC at t = 360 s (bundled h5 file).

Pipeline
--------
1. Load YAML config.
2. Instantiate ``FVM_SWE_PINN``, ``FVM_PINNDataset`` (reads SRH-2D files
   under ``data/``), ``FVM_PINNTrainer`` with strategy=teacher.
3. Teacher trainer generates a cached FVM trajectory (0 → 360 s, 31
   snapshots) and distills the network onto it with a soft SRH-2D
   anchor at t = 360 s.
4. Compare PINN prediction at t_end against the SRH-2D reference; save
   depth / velocity contours + loss history + predictions.npz.

Usage
-----
    cd examples/FVM_PINN/block_in_channel
    python block_in_channel_FVM_PINN.py                          # CPU
    python block_in_channel_FVM_PINN.py --device cuda             # GPU (recommended for 1326 cells)
    python block_in_channel_FVM_PINN.py --post-only               # skip training, plot from checkpoint
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
import matplotlib.tri as tri
import torch

# Make HydroNet importable when running this script directly.
script_path = os.path.abspath(__file__)
examples_dir = os.path.dirname(os.path.dirname(os.path.dirname(script_path)))
project_root = os.path.dirname(examples_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from HydroNet import Config, FVM_SWE_PINN, FVM_PINNTrainer, FVM_PINNDataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Checkpoint helpers (identical pattern to the bump / dam-break scripts)
# ---------------------------------------------------------------------------

def _default_checkpoint_path(config: Config) -> Path:
    strategy = str(config.get("training.strategy", "standard"))
    ckpt_dir = Path(str(
        config.get("training.logging.checkpoint_dir", "./checkpoints")
    ))
    if strategy == "teacher":
        return ckpt_dir / "teacher_final.pt"
    return ckpt_dir / "ckpt_final.pt"


def _load_checkpoint_into_model(model: FVM_SWE_PINN, ckpt_path: Path) -> dict:
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


# ---------------------------------------------------------------------------
# SRH-2D reference + evaluation
# ---------------------------------------------------------------------------

def load_srh2d_reference_at(h5_path: Path, t_target: float):
    """Return (h, u, v) SRH-2D arrays at the snapshot nearest to t_target."""
    import h5py
    with h5py.File(h5_path, "r") as f:
        times = f["Water_Depth_m/Times"][:].astype(np.float64)
        ti = int(np.argmin(np.abs(times - t_target)))
        h = f["Water_Depth_m/Values"][ti, :].astype(np.float64)
        vel = f["Velocity_m_p_s/Values"][ti, :, :].astype(np.float64)
    logger.info(f"SRH-2D reference loaded at t={times[ti]:.0f}s (target={t_target})")
    return h, vel[:, 0], vel[:, 1]


def _triangulation_from_mesh(mesh):
    """Build a matplotlib Triangulation from an UnstructuredMesh (triangles
    or quads). Quads are split into two triangles for plotting only."""
    triangles = []
    for cn in mesh.cell_nodes:
        if len(cn) == 3:
            triangles.append(cn)
        elif len(cn) == 4:
            triangles.append([cn[0], cn[1], cn[2]])
            triangles.append([cn[0], cn[2], cn[3]])
    return tri.Triangulation(mesh.node_xy[:, 0], mesh.node_xy[:, 1], triangles)


def _cell_to_node(mesh, cell_vals):
    """Simple mean of cell values at each shared node (for contourf)."""
    nv = np.zeros(len(mesh.node_xy))
    nc = np.zeros(len(mesh.node_xy))
    for ci, cn in enumerate(mesh.cell_nodes):
        for ni in cn:
            nv[ni] += cell_vals[ci]
            nc[ni] += 1
    nc[nc == 0] = 1
    return nv / nc


def evaluate_and_plot(
    model: FVM_SWE_PINN,
    config: Config,
    dataset: FVM_PINNDataset,
    out_dir: Path,
):
    t_end = float(config.get_required_config("training.t_end"))
    mesh = dataset.get_mesh()
    md = dataset.get_mesh_data()
    xc = mesh.cell_center[:, 0]
    yc = mesh.cell_center[:, 1]
    n = mesh.n_cells

    # ---- Reference from SRH-2D ----
    h5_path = Path(str(config.get_required_config("data.srh2d_h5_file")))
    h_ref, u_ref, v_ref = load_srh2d_reference_at(h5_path, t_end)
    vel_ref = np.sqrt(u_ref ** 2 + v_ref ** 2)

    # ---- PINN prediction at t_end (public forward returns [h, u, v]) ----
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
    v_pred = Q_phys[:, 2]
    vel_pred = np.sqrt(u_pred ** 2 + v_pred ** 2)

    # ---- Errors over wet cells ----
    mask = h_ref > 0.01
    l2_h = float(np.sqrt(np.mean((h_pred[mask] - h_ref[mask]) ** 2)))
    max_h = float(np.max(np.abs(h_pred[mask] - h_ref[mask])))
    l2_vel = float(np.sqrt(np.mean((vel_pred[mask] - vel_ref[mask]) ** 2)))
    max_vel = float(np.max(np.abs(vel_pred[mask] - vel_ref[mask])))
    logger.info(f"vs SRH-2D (t={t_end}s):")
    logger.info(f"  h:   L2={l2_h:.4e}  max={max_h:.4e}  "
                f"range pred=[{h_pred.min():.3f},{h_pred.max():.3f}]  "
                f"ref=[{h_ref.min():.3f},{h_ref.max():.3f}]")
    logger.info(f"  |V|: L2={l2_vel:.4e}  max={max_vel:.4e}  "
                f"range pred=[{vel_pred.min():.3f},{vel_pred.max():.3f}]  "
                f"ref=[{vel_ref.min():.3f},{vel_ref.max():.3f}]")

    # ---- Contour plots: FVM-PINN vs SRH-2D ----
    out_dir.mkdir(parents=True, exist_ok=True)
    triang = _triangulation_from_mesh(mesh)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    ax = axes[0, 0]
    tcf = ax.tricontourf(triang, _cell_to_node(mesh, h_pred), levels=25, cmap="viridis")
    plt.colorbar(tcf, ax=ax); ax.set_aspect("equal")
    ax.set_title(f"FVM-PINN: Water depth at t={t_end}s [m]")

    ax = axes[0, 1]
    tcf = ax.tricontourf(triang, _cell_to_node(mesh, vel_pred), levels=25, cmap="hot_r")
    plt.colorbar(tcf, ax=ax); ax.set_aspect("equal")
    ax.set_title(f"FVM-PINN: Velocity magnitude [m/s]")

    ax = axes[1, 0]
    tcf = ax.tricontourf(triang, _cell_to_node(mesh, h_ref), levels=25, cmap="viridis")
    plt.colorbar(tcf, ax=ax); ax.set_aspect("equal")
    ax.set_title(f"SRH-2D: Water depth at t={t_end}s [m]")

    ax = axes[1, 1]
    tcf = ax.tricontourf(triang, _cell_to_node(mesh, vel_ref), levels=25, cmap="hot_r")
    plt.colorbar(tcf, ax=ax); ax.set_aspect("equal")
    ax.set_title(f"SRH-2D: Velocity magnitude [m/s]")

    plt.tight_layout()
    path = out_dir / "block_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Contour plot: {path}")

    np.savez(
        out_dir / "predictions.npz",
        xc=xc, yc=yc,
        h_pred=h_pred, u_pred=u_pred, v_pred=v_pred, vel_pred=vel_pred,
        h_ref=h_ref, u_ref=u_ref, v_ref=v_ref, vel_ref=vel_ref,
        L2_h=l2_h, max_h=max_h, L2_vel=l2_vel, max_vel=max_vel,
    )

    return {"L2_h": l2_h, "max_h": max_h,
            "L2_vel": l2_vel, "max_vel": max_vel}


def plot_loss_history(history: dict, out_dir: Path) -> None:
    """Three-panel log-scale loss history (aggregate / distill / phys)."""
    def _nonempty(key):
        s = history.get(key) or []
        return s if (s and any(v > 0 for v in s)) else None

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    ax = axes[0]
    for key in ("total", "distill", "bc", "phys", "anchor"):
        s = _nonempty(key)
        if s:
            ax.semilogy(range(1, len(s) + 1), s, label=key)
    ax.set_xlabel("Gradient step"); ax.set_ylabel("Loss")
    ax.set_title("Aggregate losses"); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    for key in ("distill_xi", "distill_hu", "distill_hv"):
        s = _nonempty(key)
        if s:
            ax.semilogy(range(1, len(s) + 1), s, label=key.split("_", 1)[1])
    ax.set_xlabel("Gradient step"); ax.set_ylabel("Distill MSE")
    ax.set_title("Distill loss per SWE component")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[2]
    for key in ("phys_xi", "phys_hu", "phys_hv"):
        s = _nonempty(key)
        if s:
            ax.semilogy(range(1, len(s) + 1), s, label=key.split("_", 1)[1])
    ax.set_xlabel("Gradient step"); ax.set_ylabel("||dQ/dt + R||² MSE")
    ax.set_title("Physics residual per SWE component")
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = out_dir / "loss_history.png"
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    logger.info(f"Loss plot: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="FVM-PINN block-in-channel example")
    parser.add_argument("--config", default="fvm_pinn_config.yaml",
                        help="Path to the YAML config")
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"],
                        help="Override device.type from the config")
    parser.add_argument("--post-only", action="store_true",
                        help="Skip training; load existing checkpoint and evaluate.")
    parser.add_argument("--checkpoint", default=None,
                        help="Explicit checkpoint path (default: strategy-based).")
    args = parser.parse_args()

    np.random.seed(42)
    torch.manual_seed(42)

    wall_start = time.time()
    config = Config(args.config)
    if args.device:
        config.set("device.type", args.device)

    # ---- Build model + dataset ----
    model = FVM_SWE_PINN(config)
    logger.info(f"Model on device={model.get_device()}, dtype={model.dtype}")

    dataset = FVM_PINNDataset(config)
    md = dataset.get_mesh_data()
    logger.info(f"Mesh: {md['n_cells']} cells, {md['n_faces']} faces")
    logger.info(f"h_still: [{dataset.get_h_still().min().item():.3f}, "
                f"{dataset.get_h_still().max().item():.3f}]")

    if args.post_only:
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
    metrics = evaluate_and_plot(model, config, dataset, out_dir)
    if history:
        plot_loss_history(history, out_dir)

    # Training history dump
    if not args.post_only or history:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        hist_file = Path(f"history_fvm_pinn_{ts}.json")
        with open(hist_file, "w") as f:
            json.dump({"history": history, "metrics": metrics}, f, indent=2)
        logger.info(f"Training history: {hist_file}")

    logger.info(f"Total wall time: {time.time() - wall_start:.1f} s")
    logger.info("Done.")


if __name__ == "__main__":
    main()
