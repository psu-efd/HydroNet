"""
FVM-only validation baseline for the Savannah River case (no PINN training).

Reuses the case's ``fvm_pinn_config.yaml`` (same physics, BCs, mesh paths,
time window, and CFL as the teacher PINN run). Integrates Heun-RK2 FVM
from the flat-WSE IC to ``training.t_end`` using the shared
``run_fvm_rk2`` helper. Writes VTKs and compares to the SRH-2D reference
at t = t_end.

Use this as a pure-FVM sanity check — the PINN's teacher trajectory IS
this solution, so the PINN's accuracy floor is set by the FVM-only L2.

Usage
-----
    cd examples/FVM_PINN/savannah_river
    python savannah_river_fvm_only.py                # uses config's device
    python savannah_river_fvm_only.py --device cpu    # override to CPU
    python savannah_river_fvm_only.py --n-vtk 20 --cfl 0.2
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
import matplotlib.tri as tri
import torch

# Make HydroNet importable when running this script directly.
script_path = os.path.abspath(__file__)
examples_dir = os.path.dirname(os.path.dirname(os.path.dirname(script_path)))
project_root = os.path.dirname(examples_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from HydroNet import Config, FVM_PINNDataset
from HydroNet.models.FVM_PINN._internal.fvm import run_fvm_rk2
from HydroNet.models.FVM_PINN._internal.utils import write_vtk_solution

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reference + plotting helpers (mirror the PINN script for comparability)
# ---------------------------------------------------------------------------

def load_srh2d_reference_at(h5_path: Path, t_target: float):
    import h5py
    with h5py.File(h5_path, "r") as f:
        times = f["Water_Depth_m/Times"][:].astype(np.float64)
        ti = int(np.argmin(np.abs(times - t_target)))
        h = f["Water_Depth_m/Values"][ti, :].astype(np.float64)
        vel = f["Velocity_m_p_s/Values"][ti, :, :].astype(np.float64)
    logger.info(f"SRH-2D reference loaded at t={times[ti]:.0f}s (target={t_target})")
    return h, vel[:, 0], vel[:, 1]


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


def evaluate_and_plot(
    Q_final: torch.Tensor,
    h_still: torch.Tensor,
    mesh,
    mesh_data,
    config: Config,
    t_end: float,
    out_dir: Path,
):
    xc = mesh.cell_center[:, 0]
    yc = mesh.cell_center[:, 1]
    bed = mesh_data["bed_elev"].cpu().numpy()

    Q_np = Q_final.cpu().numpy()
    h_still_np = h_still.cpu().numpy()
    h_pred = Q_np[:, 0] + h_still_np
    h_safe = np.maximum(h_pred, 1e-6)
    u_pred = Q_np[:, 1] / h_safe
    v_pred = Q_np[:, 2] / h_safe
    vel_pred = np.sqrt(u_pred ** 2 + v_pred ** 2)

    h5_path = Path(str(config.get_required_config("data.srh2d_h5_file")))
    h_ref, u_ref, v_ref = load_srh2d_reference_at(h5_path, t_end)
    vel_ref = np.sqrt(u_ref ** 2 + v_ref ** 2)

    h_dry = float(config.get("physics.h_dry", 0.1))
    mask = h_ref > h_dry
    l2_h = float(np.sqrt(np.mean((h_pred[mask] - h_ref[mask]) ** 2)))
    max_h = float(np.max(np.abs(h_pred[mask] - h_ref[mask])))
    l2_v = float(np.sqrt(np.mean((vel_pred[mask] - vel_ref[mask]) ** 2)))
    max_v = float(np.max(np.abs(vel_pred[mask] - vel_ref[mask])))
    logger.info(f"vs SRH-2D (t={t_end}s):")
    logger.info(f"  h:   L2={l2_h:.4e}  max={max_h:.4e}  "
                f"range pred=[{h_pred.min():.3f},{h_pred.max():.3f}]  "
                f"ref=[{h_ref.min():.3f},{h_ref.max():.3f}]")
    logger.info(f"  |V|: L2={l2_v:.4e}  max={max_v:.4e}  "
                f"range pred=[{vel_pred.min():.3f},{vel_pred.max():.3f}]  "
                f"ref=[{vel_ref.min():.3f},{vel_ref.max():.3f}]")

    out_dir.mkdir(parents=True, exist_ok=True)
    triang = _triangulation_from_mesh(mesh)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax = axes[0, 0]
    tcf = ax.tricontourf(triang, _cell_to_node(mesh, h_pred), levels=25, cmap="viridis")
    plt.colorbar(tcf, ax=ax); ax.set_aspect("equal")
    ax.set_title(f"FVM-only: Depth at t={t_end}s [m]")
    ax = axes[0, 1]
    tcf = ax.tricontourf(triang, _cell_to_node(mesh, vel_pred), levels=25, cmap="hot_r")
    plt.colorbar(tcf, ax=ax); ax.set_aspect("equal")
    ax.set_title(f"FVM-only: Velocity magnitude [m/s]")
    ax = axes[1, 0]
    tcf = ax.tricontourf(triang, _cell_to_node(mesh, h_ref), levels=25, cmap="viridis")
    plt.colorbar(tcf, ax=ax); ax.set_aspect("equal")
    ax.set_title(f"SRH-2D: Depth at t={t_end}s [m]")
    ax = axes[1, 1]
    tcf = ax.tricontourf(triang, _cell_to_node(mesh, vel_ref), levels=25, cmap="hot_r")
    plt.colorbar(tcf, ax=ax); ax.set_aspect("equal")
    ax.set_title(f"SRH-2D: Velocity magnitude [m/s]")
    plt.tight_layout()
    path = out_dir / "savannah_fvm_only_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    logger.info(f"Contour plot: {path}")

    np.savez(
        out_dir / "fvm_only_predictions.npz",
        xc=xc, yc=yc, bed=bed,
        h_pred=h_pred, u_pred=u_pred, v_pred=v_pred, vel_pred=vel_pred,
        h_ref=h_ref, u_ref=u_ref, v_ref=v_ref, vel_ref=vel_ref,
        L2_h=l2_h, max_h=max_h, L2_vel=l2_v, max_vel=max_v,
    )

    return {"L2_h": l2_h, "max_h": max_h,
            "L2_vel": l2_v, "max_vel": max_v}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="FVM-only baseline for Savannah River")
    parser.add_argument("--config", default="fvm_pinn_config.yaml",
                        help="Shared YAML config (same as the PINN run)")
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"])
    parser.add_argument("--n-vtk", type=int, default=10,
                        help="Number of VTK snapshots (evenly over [t_start, t_end])")
    parser.add_argument("--cfl", type=float, default=None,
                        help="Override training.teacher.cfl")
    args = parser.parse_args()

    np.random.seed(42); torch.manual_seed(42)

    wall_start = time.time()
    config = Config(args.config)
    if args.device:
        config.set("device.type", args.device)

    dataset = FVM_PINNDataset(config)
    md = dataset.get_mesh_data()
    mesh = dataset.get_mesh()
    h_still = dataset.get_h_still()
    logger.info(f"Mesh: {md['n_cells']} cells, {md['n_faces']} faces")
    logger.info(f"Bed elev: [{md['bed_elev'].min().item():.2f}, "
                f"{md['bed_elev'].max().item():.2f}]m")
    logger.info(f"h_still:  [{h_still.min().item():.3f}, {h_still.max().item():.3f}]m")

    t_start = float(config.get("training.t_start", 0.0))
    t_end = float(config.get_required_config("training.t_end"))
    cfl = args.cfl if args.cfl is not None else float(
        config.get("training.teacher.cfl", 0.1)
    )
    h_dry = float(config.get("physics.h_dry", 0.1))

    Q0 = dataset.get_ic_data()["U_true"]
    h0 = Q0[:, 0] + h_still
    logger.info(f"IC h range: [{h0.min().item():.3f}, {h0.max().item():.3f}]")

    snapshot_times = torch.linspace(
        t_start, t_end, args.n_vtk + 1,
        dtype=torch.float64, device=h_still.device,
    )
    logger.info(f"FVM: [{t_start:.2f}, {t_end:.2f}]s, CFL={cfl}, h_dry={h_dry}, "
                f"n_snapshots={args.n_vtk}")

    fvm_t0 = time.time()
    t_snaps, Q_snaps = run_fvm_rk2(
        Q0, md, h_still, snapshot_times, cfl=cfl, h_dry=h_dry,
    )
    logger.info(f"FVM integration: {time.time() - fvm_t0:.1f}s, "
                f"{Q_snaps.shape[0]} snapshots saved")

    # VTK snapshots
    vtk_dir = Path("vtk_fvm_only")
    vtk_dir.mkdir(parents=True, exist_ok=True)
    h_still_np = h_still.cpu().numpy()
    for k in range(Q_snaps.shape[0]):
        Q_np = Q_snaps[k].cpu().numpy()
        U_phys = Q_np.copy()
        U_phys[:, 0] = Q_np[:, 0] + h_still_np
        write_vtk_solution(
            vtk_dir / f"fvm_t{k:04d}.vtk", mesh, U_phys, float(t_snaps[k]),
        )
    logger.info(f"VTK snapshots: {vtk_dir}")

    out_dir = Path("plots")
    metrics = evaluate_and_plot(
        Q_snaps[-1], h_still, mesh, md, config, t_end, out_dir,
    )

    logger.info(f"Metrics: {metrics}")
    logger.info(f"Total wall time: {time.time() - wall_start:.1f} s")
    logger.info("Done.")


if __name__ == "__main__":
    main()
