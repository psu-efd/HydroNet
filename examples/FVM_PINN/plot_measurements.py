"""
Plot the FVM-PINN measurement data a config will train on -- no training.

Builds ``FVM_PINNDataset`` from the YAML (so random / ``points_file`` /
``flags_file`` selection, ``variables`` and ``noise_sigma`` are applied
exactly as in training) and draws, for each measurement time, the measured
vectors over the SRH-2D velocity magnitude.

Arrows show exactly what the data loss sees: ``u, v`` for velocity
measurements (``variables: [u, v]``), ``hu, hv`` otherwise; a component
masked out at a point (``var_mask``) is drawn as zero. In ``both`` mode only
the sparse block is drawn, not the dense anchor snapshots.

Usage (run from the case directory, since YAML paths are relative)
-----
    cd examples/FVM_PINN/savannah_river
    python ../plot_measurements.py --config fvm_pinn_config_SR_F.yaml
    python ../plot_measurements.py --config my_piv.yaml --out plots/piv.png
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as tri

# Make HydroNet importable when running this script directly.
script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_path)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from HydroNet import Config, FVM_PINNDataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


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


def _measurement_rows(ref, mode):
    """Rows to draw: those supervising u/v (or hu/hv); sparse block only in 'both'."""
    vm = ref["var_mask"].cpu().numpy()
    rows = vm[:, 1:].any(axis=1)
    if mode == "both":
        if "vel_target" in ref:
            rows &= ref["vel_target"].cpu().numpy() > 0.5
        else:
            # Dense rows supervise all three components; sparse rows do not
            # unless variables lists all three (then they can't be told apart).
            rows &= ~vm.all(axis=1)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", default="fvm_pinn_config.yaml")
    parser.add_argument("--out", default="plots/measurements.png")
    args = parser.parse_args()

    config = Config(args.config)
    config.set("device.type", "cpu")
    mode = str(config.get("data.measurements.mode", "dense")).lower()

    dataset = FVM_PINNDataset(config)
    ref = dataset.get_ref_data()
    if ref is None:
        raise SystemExit("No measurement data: set data.srh2d_h5_file / data.measurements.")

    is_vel = "vel_target" in ref
    rows = _measurement_rows(ref, mode)
    xyt = ref["xyt"].cpu().numpy()[rows]
    U = ref["U_ref"].cpu().numpy()[rows]
    vm = ref["var_mask"].cpu().numpy()[rows]
    vx, vy = U[:, 1] * vm[:, 1], U[:, 2] * vm[:, 2]
    if len(xyt) == 0:
        raise SystemExit("No u/v (or hu/hv) measurement rows to plot.")

    mesh = dataset.get_mesh()
    triang = _triangulation_from_mesh(mesh)
    import h5py
    with h5py.File(str(config.get("data.srh2d_h5_file")), "r") as f:
        h5_times = f["Water_Depth_m/Times"][:].astype(np.float64)
        h_all = f["Water_Depth_m/Values"][:, :].astype(np.float64)
        vel_all = f["Velocity_m_p_s/Values"][:, :, :].astype(np.float64)

    times = np.unique(xyt[:, 2])
    t_idx = [int(np.argmin(np.abs(h5_times - t))) for t in times]
    # Background speed on wet cells only (dry cells can carry junk velocity).
    speed = np.hypot(vel_all[..., 0], vel_all[..., 1])
    speed = np.where(h_all > dataset.h_dry, speed, 0.0)
    ncols = min(len(times), 3)
    nrows = int(np.ceil(len(times) / ncols))
    Lx = np.ptp(mesh.node_xy[:, 0])
    Ly = np.ptp(mesh.node_xy[:, 1])
    fig, axes = plt.subplots(
        nrows, ncols, squeeze=False,
        figsize=(6.0 * ncols, 6.0 * nrows * max(Ly / Lx, 0.35) + 1.0),
    )

    # Common arrow scale: the largest measured vector spans ~6% of the domain.
    mag = np.hypot(vx, vy)
    scale = max(mag.max(), 1e-12) / (0.06 * max(Lx, Ly))
    label = "u, v [m/s]" if is_vel else "hu, hv [m²/s]"
    bg_max = max(speed[t_idx].max(), 1e-12)

    for k, t in enumerate(times):
        ax = axes[k // ncols, k % ncols]
        tcf = ax.tricontourf(triang, _cell_to_node(mesh, speed[t_idx[k]]),
                             levels=np.linspace(0.0, bg_max, 26), cmap="Blues")
        ax.triplot(triang, color="k", lw=0.1, alpha=0.3)
        sel = np.isclose(xyt[:, 2], t)
        ax.quiver(xyt[sel, 0], xyt[sel, 1], vx[sel], vy[sel], color="crimson",
                  angles="xy", scale_units="xy", scale=scale, width=0.003)
        ax.plot(xyt[sel, 0], xyt[sel, 1], ".", color="crimson", ms=2)
        ax.set_title(f"t = {t:g} s  —  {sel.sum()} points")
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(tcf, ax=ax, shrink=0.8, label="SRH-2D |V| [m/s]")
    for k in range(len(times), nrows * ncols):
        axes[k // ncols, k % ncols].axis("off")

    fig.suptitle(f"Measurements ({mode}, {label}); arrows share one scale")
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    logger.info(
        f"{len(xyt)} measurement rows at {len(times)} time(s); "
        f"|{label.split(' [')[0]}| range [{mag.min():.3g}, {mag.max():.3g}]; "
        f"saved {out}"
    )


if __name__ == "__main__":
    main()
