"""
FVM-PINN example: Savannah River reach (~1 km, 1306 cells).

Real-world unstructured mesh with non-trivial bathymetry (z_b ∈ [22, 29] m),
five Manning zones, and an SRH-2D XMDFC reference at 5 snapshots
[720, 1440, 2160, 2880, 3600] s. Cold flat-WSE IC at t=0 (stable under
the well-balanced perturbation form + Manning-conveyance inlet BC).

Pipeline
--------
1. Load YAML config.
2. Build model / dataset / teacher trainer.
3. Teacher generates a cached FVM trajectory (0 → 3600 s) and distills
   the network against 31 snapshots + SRH-2D anchor.
4. Evaluate at t_end against SRH-2D, also report L2 at each anchor time.
5. Save contour comparison + per-SWE-component loss history.

Usage
-----
    cd examples/FVM_PINN/savannah_river
    python savannah_river_FVM_PINN.py                          # uses config's device (default: cuda)
    python savannah_river_FVM_PINN.py --device cpu              # override to CPU
    python savannah_river_FVM_PINN.py --post-only               # skip training, plot from checkpoint
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
# Checkpoint helpers (identical pattern to the other case scripts)
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
            f"No checkpoint at {ckpt_path}. Train first (drop --post-only) "
            "or pass --checkpoint <path>."
        )
    ck = torch.load(ckpt_path, map_location=model.get_device(), weights_only=False)
    net = model.get_internal_network()
    net.load_state_dict(ck["network_state"])
    net.set_normalisation(ck["x_mean"], ck["x_std"])
    logger.info(f"Loaded checkpoint: {ckpt_path}")
    return ck.get("history", {})


# ---------------------------------------------------------------------------
# SRH-2D reference + plotting
# ---------------------------------------------------------------------------

def load_all_srh2d_snapshots(h5_path: Path):
    """Return dict of times + [h, u, v] arrays across all SRH-2D snapshots."""
    import h5py
    with h5py.File(h5_path, "r") as f:
        times = f["Water_Depth_m/Times"][:].astype(np.float64)
        h_all = f["Water_Depth_m/Values"][:, :].astype(np.float64)
        vel_all = f["Velocity_m_p_s/Values"][:, :, :].astype(np.float64)
    return {"times": times, "h": h_all, "u": vel_all[..., 0], "v": vel_all[..., 1]}


def _triangulation_from_mesh(mesh):
    triangles = []
    for cn in mesh.cell_nodes:
        if len(cn) == 3:
            triangles.append(cn)
        elif len(cn) == 4:
            triangles.append([cn[0], cn[1], cn[2]])
            triangles.append([cn[0], cn[2], cn[3]])
    return tri.Triangulation(mesh.node_xy[:, 0], mesh.node_xy[:, 1], triangles)


def _cell_to_node(mesh, cell_vals):
    nv = np.zeros(len(mesh.node_xy))
    nc = np.zeros(len(mesh.node_xy))
    for ci, cn in enumerate(mesh.cell_nodes):
        for ni in cn:
            nv[ni] += cell_vals[ci]
            nc[ni] += 1
    nc[nc == 0] = 1
    return nv / nc


def _predict_at_t(
    model: FVM_SWE_PINN,
    mesh,
    t_val: float,
    trainer: "FVM_PINNTrainer | None" = None,
) -> np.ndarray:
    """Evaluate the (trained) model at all cell centres at time ``t_val``.
    Returns an [n_cells, 3] array of physical [h, u, v].

    For the time-window strategy ``model(xyt)`` only sees the last window's
    parameters and silently extrapolates earlier-time queries — passing
    ``trainer`` enables the window-aware ``trainer.predict(xyt)`` path,
    which routes each (x,y,t) to its owning sub-network.
    """
    xc = mesh.cell_center[:, 0]
    yc = mesh.cell_center[:, 1]
    n = mesh.n_cells
    device = model.get_device()
    xyt = torch.column_stack([
        torch.tensor(xc, dtype=torch.float64, device=device),
        torch.tensor(yc, dtype=torch.float64, device=device),
        torch.full((n,), t_val, dtype=torch.float64, device=device),
    ])
    with torch.no_grad():
        if trainer is not None and trainer.strategy == "window":
            return trainer.predict(xyt).cpu().numpy()
        return model(xyt).cpu().numpy()


def evaluate_and_plot(
    model: FVM_SWE_PINN,
    config: Config,
    dataset: FVM_PINNDataset,
    out_dir: Path,
    trainer: "FVM_PINNTrainer | None" = None,
):
    mesh = dataset.get_mesh()
    md = dataset.get_mesh_data()
    bed = md["bed_elev"].cpu().numpy()
    h_dry = float(config.get("physics.h_dry", 0.1))
    t_end = float(config.get_required_config("training.t_end"))

    # Load SRH-2D snapshots (all times)
    h5_path = Path(str(config.get_required_config("data.srh2d_h5_file")))
    snaps = load_all_srh2d_snapshots(h5_path)

    # ---- L2 at every SRH-2D anchor time ----
    logger.info("PINN vs SRH-2D at each anchor time:")
    per_time_metrics = []
    for ti, t_val in enumerate(snaps["times"]):
        h_ref = snaps["h"][ti]
        u_ref = snaps["u"][ti]
        v_ref = snaps["v"][ti]
        vel_ref = np.sqrt(u_ref ** 2 + v_ref ** 2)

        Q_phys = _predict_at_t(model, mesh, float(t_val), trainer=trainer)
        h_pred = Q_phys[:, 0]
        u_pred = Q_phys[:, 1]
        v_pred = Q_phys[:, 2]
        vel_pred = np.sqrt(u_pred ** 2 + v_pred ** 2)

        mask = h_ref > h_dry
        l2_h = float(np.sqrt(np.mean((h_pred[mask] - h_ref[mask]) ** 2)))
        l2_v = float(np.sqrt(np.mean((vel_pred[mask] - vel_ref[mask]) ** 2)))
        logger.info(f"  t={t_val:7.1f}s   L2(h)={l2_h:.4e}   L2(|V|)={l2_v:.4e}")
        per_time_metrics.append({"t": float(t_val), "L2_h": l2_h, "L2_vel": l2_v})

    # ---- Contour comparison at t_end ----
    ti_end = int(np.argmin(np.abs(snaps["times"] - t_end)))
    h_ref = snaps["h"][ti_end]
    u_ref = snaps["u"][ti_end]
    v_ref = snaps["v"][ti_end]
    vel_ref = np.sqrt(u_ref ** 2 + v_ref ** 2)

    Q_phys = _predict_at_t(model, mesh, t_end, trainer=trainer)
    h_pred = Q_phys[:, 0]
    u_pred = Q_phys[:, 1]
    v_pred = Q_phys[:, 2]
    vel_pred = np.sqrt(u_pred ** 2 + v_pred ** 2)

    mask = h_ref > h_dry
    l2_h = float(np.sqrt(np.mean((h_pred[mask] - h_ref[mask]) ** 2)))
    max_h = float(np.max(np.abs(h_pred[mask] - h_ref[mask])))
    l2_v = float(np.sqrt(np.mean((vel_pred[mask] - vel_ref[mask]) ** 2)))
    max_v = float(np.max(np.abs(vel_pred[mask] - vel_ref[mask])))

    out_dir.mkdir(parents=True, exist_ok=True)
    triang = _triangulation_from_mesh(mesh)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax = axes[0, 0]
    tcf = ax.tricontourf(triang, _cell_to_node(mesh, h_pred), levels=25, cmap="viridis")
    plt.colorbar(tcf, ax=ax); ax.set_aspect("equal")
    ax.set_title(f"FVM-PINN: Depth at t={t_end}s [m]")

    ax = axes[0, 1]
    tcf = ax.tricontourf(triang, _cell_to_node(mesh, vel_pred), levels=25, cmap="hot_r")
    plt.colorbar(tcf, ax=ax); ax.set_aspect("equal")
    ax.set_title(f"FVM-PINN: Velocity magnitude [m/s]")

    ax = axes[1, 0]
    tcf = ax.tricontourf(triang, _cell_to_node(mesh, h_ref), levels=25, cmap="viridis")
    plt.colorbar(tcf, ax=ax); ax.set_aspect("equal")
    ax.set_title(f"SRH-2D: Depth at t={t_end}s [m]")

    ax = axes[1, 1]
    tcf = ax.tricontourf(triang, _cell_to_node(mesh, vel_ref), levels=25, cmap="hot_r")
    plt.colorbar(tcf, ax=ax); ax.set_aspect("equal")
    ax.set_title(f"SRH-2D: Velocity magnitude [m/s]")

    plt.tight_layout()
    path = out_dir / "savannah_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    logger.info(f"Contour plot: {path}")

    np.savez(
        out_dir / "predictions.npz",
        xc=mesh.cell_center[:, 0], yc=mesh.cell_center[:, 1], bed=bed,
        h_pred=h_pred, u_pred=u_pred, v_pred=v_pred, vel_pred=vel_pred,
        h_ref=h_ref, u_ref=u_ref, v_ref=v_ref, vel_ref=vel_ref,
        L2_h=l2_h, max_h=max_h, L2_vel=l2_v, max_vel=max_v,
    )

    return {
        "L2_h_at_t_end": l2_h, "max_h_at_t_end": max_h,
        "L2_vel_at_t_end": l2_v, "max_vel_at_t_end": max_v,
        "per_time": per_time_metrics,
    }


def plot_loss_history(history: dict, out_dir: Path) -> None:
    """Three-panel loss plot (aggregate / per-component distill / per-component phys)."""
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
    parser = argparse.ArgumentParser(description="FVM-PINN Savannah River example")
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
    logger.info(f"Bed elev: [{md['bed_elev'].min().item():.2f}, "
                f"{md['bed_elev'].max().item():.2f}]m")
    logger.info(f"h_still:  [{dataset.get_h_still().min().item():.3f}, "
                f"{dataset.get_h_still().max().item():.3f}]m")

    trainer = None
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
    metrics = evaluate_and_plot(model, config, dataset, out_dir, trainer=trainer)
    if history:
        plot_loss_history(history, out_dir)

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
