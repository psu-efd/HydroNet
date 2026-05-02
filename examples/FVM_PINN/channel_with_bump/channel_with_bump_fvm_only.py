"""
FVM-only validation baseline for the bump case (no PINN training).

Reuses the case's existing ``fvm_pinn_config.yaml`` — the physics, BCs,
mesh paths, time window, and CFL are all the same as the teacher PINN run;
this script just skips the training step and integrates the FVM solver
from the flat-WSE IC to ``training.t_end`` with the shared Heun RK2
helper. Output: per-snapshot VTKs + a final profile comparison against
the FullSWOF analytical steady-state solution.

Use this as a pure-FVM sanity check — if the profile here already matches
the analytical, the Roe solver + bed-slope + Manning source are all
working and any residual error in the PINN run is coming from the
network, not the physics.

Usage
-----
    cd examples/FVM_PINN/channel_with_bump
    python channel_with_bump_fvm_only.py                # CPU
    python channel_with_bump_fvm_only.py --device cuda   # GPU
    python channel_with_bump_fvm_only.py --n-vtk 20 --cfl 0.2
"""

import argparse
import logging
import os
import sys
import time
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

from HydroNet import Config, FVM_PINNDataset
# FVM-only scripts legitimately need the internal helpers (no public-API
# wrapper exposes raw run_fvm_rk2 / vtk_writer today).
from HydroNet.models.FVM_PINN._internal.fvm import run_fvm_rk2
from HydroNet.models.FVM_PINN._internal.utils import write_vtk_solution

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
# Evaluation / plotting — mirrors the PINN script so the two outputs are
# directly comparable side-by-side.
# ---------------------------------------------------------------------------

def evaluate_and_plot(
    Q_final: torch.Tensor,
    h_still: torch.Tensor,
    mesh,
    mesh_data,
    config,
    t_end: float,
    out_dir: Path,
):
    xc = mesh.cell_center[:, 0]
    bed = mesh_data["bed_elev"].cpu().numpy()

    # Physical state
    Q_np = Q_final.cpu().numpy()
    h_still_np = h_still.cpu().numpy()
    h_pred = Q_np[:, 0] + h_still_np
    h_safe = np.maximum(h_pred, 1e-6)
    u_pred = Q_np[:, 1] / h_safe
    wse_pred = h_pred + bed

    # Analytical reference
    dat = load_analytical(Path("data/analytical_solution_from_FullSWOF.dat"))
    ref = interp_at_x(dat, xc)
    h_exact = ref["h"]
    wse_exact = ref["wse"]
    # Look up the inlet-q entry by type — ``Config.get`` can't traverse
    # integer-keyed sub-dicts via dotted strings, so access the whole
    # block and iterate.
    bc_block = config.get_required_config("boundary_conditions")
    inlet_q = next(
        float(v["value"]) for v in bc_block.values()
        if isinstance(v, dict) and v.get("type") == "inlet-q"
    )
    u_exact = np.where(h_exact > 1e-6, inlet_q / h_exact, 0.0)

    # Errors
    l2_h = float(np.sqrt(np.mean((h_pred - h_exact) ** 2)))
    max_h = float(np.max(np.abs(h_pred - h_exact)))
    l2_u = float(np.sqrt(np.mean((u_pred - u_exact) ** 2)))
    max_u = float(np.max(np.abs(u_pred - u_exact)))
    l2_wse = float(np.sqrt(np.mean((wse_pred - wse_exact) ** 2)))
    logger.info(f"h:   L2={l2_h:.4e}, max={max_h:.4e}")
    logger.info(f"u:   L2={l2_u:.4e}, max={max_u:.4e}")
    logger.info(f"WSE: L2={l2_wse:.4e}")

    # Sort by x for clean profiles
    order = np.argsort(xc)
    xc_s = xc[order]
    bed_s = bed[order]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    ax = axes[0]
    ax.fill_between(xc_s, 0, bed_s, color="#d2b48c", alpha=0.5, label="Bed")
    ax.plot(xc_s, wse_exact[order], "k-", lw=2, label="Analytical WSE")
    ax.plot(xc_s, wse_pred[order], "b--", lw=1.6, label="FVM-only WSE")
    ax.set_xlabel("x [m]"); ax.set_ylabel("Elevation [m]")
    ax.set_title(f"Water surface elevation at t = {t_end} s (FVM only)")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(xc_s, u_exact[order], "k-", lw=2, label="Analytical u")
    ax.plot(xc_s, u_pred[order], "b--", lw=1.6, label="FVM-only u")
    ax.set_xlabel("x [m]"); ax.set_ylabel("u [m/s]")
    ax.set_title(f"Velocity at t = {t_end} s (FVM only)")
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "bump_fvm_only_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Profile plot: {path}")

    np.savez(
        out_dir / "fvm_only_predictions.npz",
        x=xc, bed=bed,
        h_pred=h_pred, u_pred=u_pred, wse_pred=wse_pred,
        h_exact=h_exact, u_exact=u_exact, wse_exact=wse_exact,
        L2_h=l2_h, max_h=max_h, L2_u=l2_u, max_u=max_u, L2_wse=l2_wse,
    )

    return {"L2_h": l2_h, "max_h": max_h,
            "L2_u": l2_u, "max_u": max_u, "L2_wse": l2_wse}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="FVM-only baseline for the channel-with-bump case"
    )
    parser.add_argument(
        "--config", default="fvm_pinn_config.yaml",
        help="Shared YAML config (same as the PINN run)",
    )
    parser.add_argument(
        "--device", default=None, choices=["cpu", "cuda"],
        help="Override device.type",
    )
    parser.add_argument(
        "--n-vtk", type=int, default=10,
        help="Number of VTK snapshots (evenly spaced over [t_start, t_end])",
    )
    parser.add_argument(
        "--cfl", type=float, default=None,
        help="Override training.teacher.cfl (the FVM stepper CFL)",
    )
    args = parser.parse_args()

    np.random.seed(42); torch.manual_seed(42)

    wall_start = time.time()
    config = Config(args.config)
    if args.device:
        config.set("device.type", args.device)

    # ---- Build dataset (reads SRH-2D, builds mesh + h_still + IC) ----
    dataset = FVM_PINNDataset(config)
    md = dataset.get_mesh_data()
    mesh = dataset.get_mesh()
    h_still = dataset.get_h_still()
    n_cells = int(md["n_cells"])
    logger.info(f"Mesh: {n_cells} cells, {md['n_faces']} faces")
    logger.info(f"h_still: [{h_still.min().item():.3f}, {h_still.max().item():.3f}]")

    # ---- Time window + CFL from config (CLI can override CFL) ----
    t_start = float(config.get("training.t_start", 0.0))
    t_end = float(config.get_required_config("training.t_end"))
    cfl = args.cfl if args.cfl is not None else float(
        config.get("training.teacher.cfl", 0.3)
    )
    h_dry = float(config.get("physics.h_dry", 0.01))

    # ---- Initial state (perturbation form from the dataset) ----
    Q0 = dataset.get_ic_data()["U_true"]
    h0 = Q0[:, 0] + h_still
    logger.info(f"IC h range: [{h0.min().item():.3f}, {h0.max().item():.3f}]")

    # ---- Snapshot times for VTK output ----
    snapshot_times = torch.linspace(
        t_start, t_end, args.n_vtk + 1,
        dtype=torch.float64, device=h_still.device,
    )
    logger.info(
        f"FVM: [{t_start:.2f}, {t_end:.2f}]s, CFL={cfl}, h_dry={h_dry}, "
        f"n_snapshots={args.n_vtk}"
    )

    # ---- Run FVM ----
    fvm_t0 = time.time()
    t_snaps, Q_snaps = run_fvm_rk2(
        Q0, md, h_still, snapshot_times, cfl=cfl, h_dry=h_dry,
    )
    logger.info(f"FVM integration: {time.time() - fvm_t0:.1f}s, "
                f"{Q_snaps.shape[0]} snapshots saved")

    # ---- Write VTKs ----
    vtk_dir = Path("vtk_fvm_only")
    vtk_dir.mkdir(parents=True, exist_ok=True)
    h_still_np = h_still.cpu().numpy()
    for k in range(Q_snaps.shape[0]):
        Q_np = Q_snaps[k].cpu().numpy()
        U_phys = Q_np.copy()
        U_phys[:, 0] = Q_np[:, 0] + h_still_np       # xi -> h for VTK writer
        write_vtk_solution(
            vtk_dir / f"fvm_t{k:04d}.vtk", mesh, U_phys, float(t_snaps[k]),
        )
    logger.info(f"VTK snapshots: {vtk_dir}")

    # ---- Final state analysis at t_end ----
    out_dir = Path("plots")
    metrics = evaluate_and_plot(
        Q_snaps[-1], h_still, mesh, md, config, t_end, out_dir,
    )

    logger.info(f"Metrics: {metrics}")
    logger.info(f"Total wall time: {time.time() - wall_start:.1f} s")
    logger.info("Done.")


if __name__ == "__main__":
    main()
