"""
FVM-PINN example: 1D transcritical flow over a parabolic bump.

Classic Goutal & Maurel benchmark — the flow is subcritical upstream,
passes through critical at the bump crest, accelerates to supercritical on
the downstream face, then jumps back to subcritical through a hydraulic
jump. Reference: FullSWOF analytical steady-state profile.

Pipeline
--------
1. Load YAML config.
2. Instantiate ``FVM_SWE_PINN``, ``FVM_PINNDataset`` (reads the SRH-2D
   files under ``data/``), and ``FVM_PINNTrainer`` with strategy=teacher.
3. Teacher trainer: generates a cached FVM trajectory (0 → 72 s), then
   distils the network onto the 51 snapshots.
4. Compare the prediction at ``t_end`` against the FullSWOF analytical
   profile; save WSE / h / velocity plots + loss history + predictions.npz.

Usage
-----
    cd examples/FVM_PINN/channel_with_bump
    python channel_with_bump_FVM_PINN.py                # CPU
    python channel_with_bump_FVM_PINN.py --device cuda   # GPU (CLI override)
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

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Analytical reference (FullSWOF steady-state 1D profile)
# ---------------------------------------------------------------------------

def load_analytical(dat_path: Path):
    """Parse the 4-column (x, h, zb, wse) FullSWOF reference file."""
    data = np.loadtxt(dat_path, skiprows=1)
    return {
        "x":   data[:, 0],
        "h":   data[:, 1],
        "zb":  data[:, 2],
        "wse": data[:, 3],
    }


def interp_at_x(analytical, xq):
    """Linearly interpolate h/zb/wse at query x coordinates."""
    return {
        k: np.interp(xq, analytical["x"], analytical[k])
        for k in ("h", "zb", "wse")
    }


# ---------------------------------------------------------------------------
# Evaluation / plotting
# ---------------------------------------------------------------------------

def evaluate_and_plot(model, config, dataset, out_dir: Path):
    """Compare trained model with FullSWOF analytical at t = t_end."""
    t_end = float(config.get_required_config("training.t_end"))

    # Cell centres (from dataset) and bed elevation from mesh_data
    mesh = dataset.get_mesh()
    mesh_data = dataset.get_mesh_data()
    xc = mesh.cell_center[:, 0]
    yc = mesh.cell_center[:, 1]
    bed = mesh_data["bed_elev"].cpu().numpy()
    n = mesh.n_cells

    # Analytical reference
    dat = load_analytical(Path("data/analytical_solution_from_FullSWOF.dat"))
    ref = interp_at_x(dat, xc)
    h_exact = ref["h"]
    wse_exact = ref["wse"]

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
    wse_pred = h_pred + bed

    # Expected analytical velocity: q = inlet_q (conservation). u_exact = q/h_exact
    # ``Config.get`` can't traverse integer-keyed sub-dicts via dotted
    # strings, so read the whole BC block and look up the inlet-q entry.
    bc_block = config.get_required_config("boundary_conditions")
    inlet_q = next(
        float(v["value"]) for v in bc_block.values()
        if isinstance(v, dict) and v.get("type") == "inlet-q"
    )
    u_exact = np.where(h_exact > 1e-6, inlet_q / h_exact, 0.0)

    # Errors — sort by x first so profiles plot cleanly
    order = np.argsort(xc)
    xc_s = xc[order]
    h_pred_s, h_exact_s = h_pred[order], h_exact[order]
    u_pred_s, u_exact_s = u_pred[order], u_exact[order]
    wse_pred_s, wse_exact_s = wse_pred[order], wse_exact[order]
    bed_s = bed[order]

    l2_h = float(np.sqrt(np.mean((h_pred - h_exact) ** 2)))
    max_h = float(np.max(np.abs(h_pred - h_exact)))
    l2_u = float(np.sqrt(np.mean((u_pred - u_exact) ** 2)))
    max_u = float(np.max(np.abs(u_pred - u_exact)))
    l2_wse = float(np.sqrt(np.mean((wse_pred - wse_exact) ** 2)))
    logger.info(f"h:   L2={l2_h:.4e}, max={max_h:.4e}")
    logger.info(f"u:   L2={l2_u:.4e}, max={max_u:.4e}")
    logger.info(f"WSE: L2={l2_wse:.4e}")

    # ---- Profile plot ----
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    ax = axes[0]
    ax.fill_between(xc_s, 0, bed_s, color="#d2b48c", alpha=0.5, label="Bed")
    ax.plot(xc_s, wse_exact_s, "k-", lw=2, label="Analytical WSE")
    ax.plot(xc_s, wse_pred_s, "r--", lw=1.6, label="FVM-PINN WSE")
    ax.set_xlabel("x [m]"); ax.set_ylabel("Elevation [m]")
    ax.set_title(f"Water surface elevation at t = {t_end} s")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(xc_s, u_exact_s, "k-", lw=2, label="Analytical u")
    ax.plot(xc_s, u_pred_s, "r--", lw=1.6, label="FVM-PINN u")
    ax.set_xlabel("x [m]"); ax.set_ylabel("u [m/s]")
    ax.set_title(f"Velocity at t = {t_end} s")
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = out_dir / "bump_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Profile plot: {path}")

    np.savez(
        out_dir / "predictions.npz",
        x=xc, bed=bed,
        h_pred=h_pred, u_pred=u_pred, wse_pred=wse_pred,
        h_exact=h_exact, u_exact=u_exact, wse_exact=wse_exact,
        L2_h=l2_h, max_h=max_h, L2_u=l2_u, max_u=max_u, L2_wse=l2_wse,
    )

    return {"L2_h": l2_h, "max_h": max_h, "L2_u": l2_u, "max_u": max_u,
            "L2_wse": l2_wse}


def plot_loss_history(history: dict, out_dir: Path) -> None:
    """Three-panel loss plot:
       (1) aggregate curves (total / distill / bc / phys / anchor),
       (2) per-SWE-component distill loss (xi / hu / hv),
       (3) per-SWE-component physics-residual loss (xi / hu / hv).
    """
    def _nonempty(key):
        s = history.get(key) or []
        return s if (s and any(v > 0 for v in s)) else None

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # Panel 1 — aggregate
    ax = axes[0]
    for key in ("total", "distill", "bc", "phys", "anchor"):
        s = _nonempty(key)
        if s:
            ax.semilogy(range(1, len(s) + 1), s, label=key)
    ax.set_xlabel("Gradient step"); ax.set_ylabel("Loss")
    ax.set_title("Aggregate losses"); ax.legend(); ax.grid(True, alpha=0.3)

    # Panel 2 — per-component distill
    ax = axes[1]
    for key in ("distill_xi", "distill_hu", "distill_hv"):
        s = _nonempty(key)
        if s:
            ax.semilogy(range(1, len(s) + 1), s, label=key.split("_", 1)[1])
    ax.set_xlabel("Gradient step"); ax.set_ylabel("Distill MSE")
    ax.set_title("Distill loss per SWE component")
    ax.legend(); ax.grid(True, alpha=0.3)

    # Panel 3 — per-component physics-residual
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
    parser = argparse.ArgumentParser(description="FVM-PINN channel-with-bump example")
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
        help="Explicit checkpoint path (default: "
             "checkpoints/teacher_final.pt for strategy=teacher, "
             "checkpoints/ckpt_final.pt otherwise).",
    )
    args = parser.parse_args()

    np.random.seed(42)
    torch.manual_seed(42)

    wall_start = time.time()
    config = Config(args.config)
    if args.device:
        config.set("device.type", args.device)

    # ---- Build model + dataset (needed by both paths) ----
    model = FVM_SWE_PINN(config)
    logger.info(f"Model on device={model.get_device()}, dtype={model.dtype}")

    dataset = FVM_PINNDataset(config)
    md = dataset.get_mesh_data()
    logger.info(f"Mesh: {md['n_cells']} cells, {md['n_faces']} faces")
    logger.info(f"h_still: [{dataset.get_h_still().min().item():.3f}, "
                f"{dataset.get_h_still().max().item():.3f}]")

    if args.post_only:
        # Wire the mesh context onto the model (normally FVM_PINNTrainer
        # does this in its __init__). Then load the checkpoint and skip
        # straight to evaluation.
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

    # Training history dump (skipped in post-only if the checkpoint had no
    # history — ``evaluate_and_plot`` already wrote the metrics + npz).
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
