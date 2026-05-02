"""
Loss-landscape diagnostic for the FVM-PINN failure-mode story.

Motivates the "physics-only FVM-PINN collapses to a trivial low-momentum
state" claim in §6 of the manuscript by showing that the FVM residual
loss has a shallow / flat basin near alpha = 0 (trivial momentum) that
the optimizer can easily fall into, whereas adding the data loss
produces a sharp minimum at alpha = 1 (the trained solution).

What it does
------------
1. Load a trained FVM-PINN checkpoint (default: BIC-B -- the lightest
   data-guided run that reaches the true solution).
2. Wrap the network in a scaling module that multiplies the momentum
   outputs (hu, hv) by a scalar alpha, leaving xi (depth perturbation)
   unchanged.
3. Sweep alpha in [alpha_min, alpha_max] and evaluate:
       L_fvm(alpha)   — physics loss only
       L_data(alpha)  — data loss only
       L_total(alpha) — weighted sum using the run's own lambdas
4. Save a 2x1 figure (fig_loss_landscape.pdf) and a raw-data JSON.

The script is idempotent and cheap (~1 minute on CPU for the 1326-cell
block-in-channel case; no training is performed).

Usage
-----
    cd <HydroNet root>
    python scripts/diagnose_loss_landscape.py \\
        --config  examples/FVM_PINN/block_in_channel/fvm_pinn_config_BIC_B.yaml \\
        --checkpoint examples/FVM_PINN/block_in_channel/checkpoints/BIC_B/ckpt_final.pt \\
        --out plan/figures/

Defaults target BIC-B when invoked with no arguments.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# Make HydroNet importable when running from the repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from HydroNet import Config, FVM_SWE_PINN, FVM_PINNDataset  # noqa: E402
from HydroNet.models.FVM_PINN._internal.pinn.loss import (  # noqa: E402
    FVMPINNLoss, LossConfig,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


class MomentumScaledNetwork(nn.Module):
    """Wraps a trained SWENet and post-scales its momentum outputs by alpha.

    The network returns ``Q = [xi, hu, hv]``. This wrapper rescales the
    last two components to ``Q_scaled = [xi, alpha * hu, alpha * hv]``,
    leaving xi unchanged so the water-depth field stays physical.
    """

    def __init__(self, base: nn.Module, alpha: float):
        super().__init__()
        self.base = base
        self.register_buffer(
            "_scale",
            torch.tensor([1.0, alpha, alpha], dtype=torch.float64),
        )
        self.alpha = float(alpha)

    def forward(self, xyt: torch.Tensor) -> torch.Tensor:
        Q = self.base(xyt)
        scale = self._scale.to(device=Q.device, dtype=Q.dtype)
        return Q * scale


# ---------------------------------------------------------------------------
# Core diagnostic
# ---------------------------------------------------------------------------

def build_components(
    config: Config,
) -> Tuple[FVM_SWE_PINN, FVM_PINNDataset, FVMPINNLoss, torch.Tensor]:
    """Instantiate model, dataset, loss, and a fixed time-sample vector."""
    dataset = FVM_PINNDataset(config)
    model = FVM_SWE_PINN(config)
    model.set_mesh_context(
        h_still_cells=dataset.get_h_still(),
        cell_xy=dataset.get_cell_xy(),
    )

    loss_cfg = LossConfig(
        lambda_fvm=float(config.get("training.loss_weights.lambda_fvm", 1.0)),
        lambda_ic=float(config.get("training.loss_weights.lambda_ic", 10.0)),
        lambda_bc=float(config.get("training.loss_weights.lambda_bc", 30.0)),
        lambda_data=float(config.get("training.loss_weights.lambda_data", 10.0)),
        lambda_xi=float(config.get("training.component_weights.lambda_xi", 1.0)),
        lambda_hu=float(config.get("training.component_weights.lambda_hu", 1.0)),
        lambda_hv=float(config.get("training.component_weights.lambda_hv", 1.0)),
        h_dry=float(config.get("physics.h_dry", 1e-2)),
        use_grad_checkpoint=False,
    )
    loss = FVMPINNLoss(
        cfg=loss_cfg,
        mesh_data=dataset.get_mesh_data(),
        h_still=dataset.get_h_still(),
        bc_config=None,
    )

    # Fixed, reproducible time samples -- a few points in [t_start, t_end]
    # so the diagnostic isn't dominated by a single snapshot.
    t_start = float(config.get("training.t_start", 0.0))
    t_end = float(config.get("training.t_end", 1.0))
    n_t = int(config.get("training.n_time_samples", 8))
    # Deterministic stratified sample -- shared across all alphas so each
    # alpha evaluation is compared at the same times.
    t_samples = torch.linspace(
        t_start, t_end, max(n_t, 3), dtype=torch.float64,
        device=model.get_device(),
    )
    return model, dataset, loss, t_samples


def load_checkpoint(model: FVM_SWE_PINN, ckpt_path: Path) -> bool:
    """Load network weights + x_mean / x_std into model's internal network.

    Returns True if weights were loaded, False if the checkpoint was missing
    (the diagnostic still runs on random-init weights but is less meaningful).
    """
    if not ckpt_path.exists():
        logger.warning(
            "Checkpoint %s not found. Running on random-init weights -- "
            "curves are shape-only, not quantitative.", ckpt_path,
        )
        return False
    ck = torch.load(
        ckpt_path, map_location=model.get_device(), weights_only=False
    )
    net = model.get_internal_network()
    net.load_state_dict(ck["network_state"])
    net.set_normalisation(ck["x_mean"], ck["x_std"])
    logger.info("Loaded checkpoint %s", ckpt_path)
    return True


@torch.no_grad()
def _wrap_without_grad(
    fn, *args, **kwargs,
):
    """FVM residual needs autograd through time; data/IC don't. Helper is only
    here so the caller can write grad_enabled(True) at the outer loop level."""
    return fn(*args, **kwargs)


def scan_alpha(
    model: FVM_SWE_PINN,
    dataset: FVM_PINNDataset,
    loss_fn: FVMPINNLoss,
    t_samples: torch.Tensor,
    alphas: np.ndarray,
) -> Dict[str, list]:
    """For each alpha, evaluate FVMPINNLoss and collect the components."""
    ic_data = dataset.get_ic_data()
    ref_data = dataset.get_ref_data()
    bc_data = dataset.get_bc_data()

    base_net = model.get_internal_network()
    base_net.eval()

    out: Dict[str, list] = {
        "alpha":    list(map(float, alphas)),
        "fvm":      [], "fvm_xi": [], "fvm_hu": [], "fvm_hv": [],
        "data":     [], "data_xi": [], "data_hu": [], "data_hv": [],
        "ic":       [],
        "bc":       [],
        "total":    [],
    }
    for alpha in alphas:
        wrapped = MomentumScaledNetwork(base_net, float(alpha))
        # FVM loss uses autograd through t -> must enable grad.
        with torch.enable_grad():
            losses = loss_fn.forward(
                network=wrapped,
                t=t_samples,
                ic_data=ic_data,
                bc_data=bc_data,
                ref_data=ref_data,
            )
        for k in ("fvm", "fvm_xi", "fvm_hu", "fvm_hv",
                  "data", "data_xi", "data_hu", "data_hv",
                  "ic", "bc", "total"):
            v = losses.get(k)
            out[k].append(float(v.detach().cpu()) if v is not None else float("nan"))
    return out


# ---------------------------------------------------------------------------
# Plot + JSON dump
# ---------------------------------------------------------------------------

def save_plot(curves: Dict[str, list], out_path: Path) -> None:
    alpha = np.array(curves["alpha"])
    fvm = np.array(curves["fvm"])
    data = np.array(curves["data"])
    total = np.array(curves["total"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    ax.semilogy(alpha, np.maximum(fvm, 1e-20), "-", color="tab:blue", lw=2,
                label=r"$\mathcal{L}_{\mathrm{fvm}}(\alpha)$")
    ax.semilogy(alpha, np.maximum(data, 1e-20), "-", color="tab:orange", lw=2,
                label=r"$\mathcal{L}_{\mathrm{data}}(\alpha)$")
    ax.axvline(0.0, color="gray", ls=":", lw=1)
    ax.axvline(1.0, color="gray", ls=":", lw=1)
    ax.set_xlabel(r"momentum scale $\alpha$ (1 = trained, 0 = zero-momentum)")
    ax.set_ylabel(r"loss (log scale)")
    ax.set_title("FVM and data losses along the momentum-scaling line")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best")

    ax = axes[1]
    ax.plot(alpha, total, "-", color="tab:green", lw=2,
            label=r"$\mathcal{L}_{\mathrm{total}}(\alpha)$")
    ax.axvline(0.0, color="gray", ls=":", lw=1)
    ax.axvline(1.0, color="gray", ls=":", lw=1)
    ax.set_xlabel(r"momentum scale $\alpha$")
    ax.set_ylabel(r"weighted total loss")
    ax.set_title("Total loss along the momentum-scaling line")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s.{pdf,png}", out_path)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Loss-landscape diagnostic for the FVM-PINN manuscript.",
    )
    p.add_argument(
        "--config",
        default=str(
            REPO_ROOT / "examples" / "FVM_PINN" / "block_in_channel"
            / "fvm_pinn_config_BIC_B.yaml"
        ),
        help="YAML config of the trained run (default: BIC-B).",
    )
    p.add_argument(
        "--checkpoint",
        default=str(
            REPO_ROOT / "examples" / "FVM_PINN" / "block_in_channel"
            / "checkpoints" / "BIC_B" / "ckpt_final.pt"
        ),
        help="Path to the trained .pt checkpoint (default: BIC-B final).",
    )
    p.add_argument(
        "--out",
        default=str(REPO_ROOT / "plan" / "figures"),
        help="Output directory for fig_loss_landscape.{pdf,png} and .json.",
    )
    p.add_argument("--alpha-min", type=float, default=0.0)
    p.add_argument("--alpha-max", type=float, default=1.5)
    p.add_argument("--n-alpha",   type=int,   default=31)
    p.add_argument(
        "--device", default=None, choices=[None, "cpu", "cuda"],
        help="Override device.type in the YAML (use 'cpu' for reproducibility).",
    )
    args = p.parse_args()

    cfg_path = Path(args.config).resolve()
    if not cfg_path.exists():
        logger.error("Config not found: %s", cfg_path)
        return 2

    # Load config and optionally override the device.
    # Config is opened relative to cfg_path's directory so that relative
    # data/ paths inside the YAML resolve.
    import os
    cwd0 = os.getcwd()
    os.chdir(cfg_path.parent)
    try:
        config = Config(str(cfg_path.name))
        if args.device is not None:
            config.set("device.type", args.device)
        model, dataset, loss_fn, t_samples = build_components(config)
        load_checkpoint(model, Path(args.checkpoint).resolve())

        alphas = np.linspace(args.alpha_min, args.alpha_max, args.n_alpha)
        curves = scan_alpha(model, dataset, loss_fn, t_samples, alphas)
    finally:
        os.chdir(cwd0)

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = out_dir / "fig_loss_landscape"
    save_plot(curves, fig_path)
    with (out_dir / "loss_landscape_data.json").open("w") as f:
        json.dump({
            "source_config":     str(cfg_path),
            "source_checkpoint": str(args.checkpoint),
            "alpha_range":       [args.alpha_min, args.alpha_max],
            "n_alpha":           args.n_alpha,
            "curves":            curves,
        }, f, indent=2)
    logger.info("Wrote %s", out_dir / "loss_landscape_data.json")

    # One-line summary the user can grep from a log.
    i_zero = int(np.argmin(np.abs(np.array(curves["alpha"]) - 0.0)))
    i_one  = int(np.argmin(np.abs(np.array(curves["alpha"]) - 1.0)))
    fvm0, fvm1 = curves["fvm"][i_zero], curves["fvm"][i_one]
    dat0, dat1 = curves["data"][i_zero], curves["data"][i_one]
    logger.info(
        "At alpha=0: L_fvm=%.3e  L_data=%.3e   |   at alpha=1: L_fvm=%.3e  L_data=%.3e",
        fvm0, dat0, fvm1, dat1,
    )
    logger.info(
        "Ratios -- L_fvm(0)/L_fvm(1) = %.2f   L_data(0)/L_data(1) = %.2f",
        (fvm0 / fvm1 if fvm1 > 0 else float("nan")),
        (dat0 / dat1 if dat1 > 0 else float("nan")),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
