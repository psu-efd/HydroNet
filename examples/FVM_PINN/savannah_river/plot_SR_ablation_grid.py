"""
7-row ablation comparison for the Savannah River SR-A ... SR-G runs.

A single combined figure is generated:

    plots/SR_ablation_combined.png      7 rows x 4 cols
        col 0 : water-depth field of each run at t = 3600 s
        col 1 : signed difference (run - SRH-2D) for depth
        col 2 : velocity magnitude of each run at t = 3600 s
        col 3 : signed difference (run - SRH-2D) for velocity magnitude

Each row corresponds to one SR ablation run loaded directly from
``checkpoints/SR_X/...``. The SRH-2D ground truth comes from
``data/Savana_XMDFC.h5`` at the snapshot closest to ``training.t_end``
(t = 3600 s for this case).

Usage
-----
    cd examples/FVM_PINN/savannah_river
    python plot_SR_ablation_grid.py                # CPU, 200 dpi
    python plot_SR_ablation_grid.py --device cuda  # GPU forward passes
    python plot_SR_ablation_grid.py --no-tex       # disable LaTeX rendering
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import PolyCollection
from matplotlib.ticker import FormatStrFormatter

# Make HydroNet importable when running this script directly.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]  # savannah_river/ -> FVM_PINN/ -> examples/ -> root
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from HydroNet import Config, FVM_PINNDataset, FVM_SWE_PINN

plt.rc('text', usetex=True)  #allow the use of Latex for math expressions and equations
plt.rc('font', family='serif') #specify the default font family to be "serif"

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


# Each entry: (run id, YAML used to train it, checkpoint relative path, row label).
# For windowed runs (SR-B, SR-C) we plot the final window's network, which
# covers t = t_end. The checkpoint is therefore inside ``window_NNN/``.
SR_RUNS: List[Tuple[str, str, str, str]] = [
    ("SR_A", "fvm_pinn_config_SR_A.yaml", "ckpt_final.pt",
     "SR-A\nphysics only"),
    ("SR_B", "fvm_pinn_config_SR_B.yaml", "ckpt_final.pt",
     "SR-B\nstandard"),
    ("SR_C", "fvm_pinn_config_SR_C.yaml", "window_004/ckpt_final.pt",
     "SR-C\nwindow(5)"),
    ("SR_D", "fvm_pinn_config_SR_D.yaml", "window_009/ckpt_final.pt",
     "SR-D\nwindow(10)"),
    ("SR_E", "fvm_pinn_config_SR_E.yaml", "teacher_final.pt",
     "SR-E\nFVM teacher"),
    ("SR_F", "fvm_pinn_config_SR_F.yaml", "ckpt_final.pt",
     "SR-F\nsparse $N_d=200$"),
    ("SR_G", "fvm_pinn_config_SR_G.yaml", "ckpt_final.pt",
     "SR-G\nanchor, $\\lambda_\\mathrm{fvm-pinn}=0$"),
]

H5_REF = SCRIPT_DIR / "data" / "Savana_XMDFC.h5"


# ---------------------------------------------------------------------------
# Style + helpers
# ---------------------------------------------------------------------------

def setup_publication_style(use_tex: bool) -> None:
    plt.rcdefaults()
    if use_tex:
        try:
            plt.rc("text", usetex=True)
        except Exception:
            logger.warning("LaTeX not available; falling back to mathtext.")
            plt.rc("text", usetex=False)
    else:
        plt.rc("text", usetex=False)
    plt.rc("font", family="serif", size=21)
    plt.rc("axes", labelsize=21, titlesize=21)
    plt.rc("xtick", labelsize=21)
    plt.rc("ytick", labelsize=21)


def domain_extent(mesh) -> Tuple[float, float, float, float]:
    nodes = mesh.node_xy
    return (float(nodes[:, 0].min()), float(nodes[:, 0].max()),
            float(nodes[:, 1].min()), float(nodes[:, 1].max()))


def cell_polygons(mesh) -> list:
    """Return a list of [n_verts, 2] polygon vertex arrays, one per cell."""
    return [mesh.node_xy[ids] for ids in mesh.cell_nodes]


# ---------------------------------------------------------------------------
# Inference + reference loaders
# ---------------------------------------------------------------------------

def load_srh2d_reference(
    h5_path: Path, target_time: float,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Return ``(actual_time, depth[n_cells], vel[n_cells, 2])`` from the
    SRH-2D snapshot closest to ``target_time``."""
    with h5py.File(h5_path, "r") as f:
        times = f["Water_Depth_m/Times"][:].astype(np.float64)
        depth = f["Water_Depth_m/Values"][:].astype(np.float64)
        vel = f["Velocity_m_p_s/Values"][:].astype(np.float64)
    ti = int(np.argmin(np.abs(times - target_time)))
    return float(times[ti]), depth[ti], vel[ti]


def predict_at_time(
    config: Config, dataset: FVM_PINNDataset,
    ckpt_path: Path, t_query: float,
) -> np.ndarray:
    """Build a fresh ``FVM_SWE_PINN``, load the checkpoint, run forward
    at ``t_query`` on every cell centre. Returns ``[n_cells, 3]`` of
    physical ``[h, u, v]``."""
    model = FVM_SWE_PINN(config)
    model.set_mesh_context(
        h_still_cells=dataset.get_h_still(),
        cell_xy=dataset.get_cell_xy(),
    )
    ck = torch.load(ckpt_path, map_location=model.get_device(), weights_only=False)
    net = model.get_internal_network()
    net.load_state_dict(ck["network_state"])
    net.set_normalisation(ck["x_mean"], ck["x_std"])

    mesh = dataset.get_mesh()
    xc = mesh.cell_center[:, 0]
    yc = mesh.cell_center[:, 1]
    n = mesh.n_cells
    device = model.get_device()
    xyt = torch.column_stack([
        torch.tensor(xc, dtype=torch.float64, device=device),
        torch.tensor(yc, dtype=torch.float64, device=device),
        torch.full((n,), t_query, dtype=torch.float64, device=device),
    ])
    with torch.no_grad():
        Q = model(xyt).cpu().numpy()
    return Q


# ---------------------------------------------------------------------------
# Plotter
# ---------------------------------------------------------------------------

def plot_combined_grid(
    mesh,
    runs_h: List[Tuple[str, np.ndarray]],         # (row_label, h[n_cells])
    h_srh: np.ndarray,                            # [n_cells] SRH-2D depth
    runs_vmag: List[Tuple[str, np.ndarray]],      # (row_label, |u|[n_cells])
    vmag_srh: np.ndarray,                         # [n_cells] SRH-2D velocity mag
    column_titles: Tuple[str, str, str, str],
    column_labels: Tuple[str, str, str, str],
    out_path: Path,
    dpi: int,
    diff_pctile: float = 99.0,
) -> None:
    """6-row x 4-col compact comparison grid (mirrors the BIC ablation grid).

    Columns:
      0  water depth ``h``
      1  signed difference ``h - h_SRH-2D``
      2  velocity magnitude ``|u|``
      3  signed difference ``|u| - |u|_SRH-2D``
    """
    polys = cell_polygons(mesh)
    extent = domain_extent(mesh)
    n_runs = len(runs_h)
    if n_runs == 0:
        logger.warning("No runs to plot; skipping figure.")
        return
    if len(runs_vmag) != n_runs:
        raise ValueError(
            f"Mismatched run counts: depth={n_runs}, velocity={len(runs_vmag)}"
        )

    # ---- Shared per-column color scales ----
    vmin_h = float(np.nanmin(h_srh))
    vmax_h = float(np.nanmax(h_srh))
    if vmax_h - vmin_h < 1e-12:
        vmax_h = vmin_h + 1e-12
    vmin_v = float(np.nanmin(vmag_srh))
    vmax_v = float(np.nanmax(vmag_srh))
    if vmax_v - vmin_v < 1e-12:
        vmax_v = vmin_v + 1e-12

    diffs_h = [np.asarray(rv) - h_srh for _, rv in runs_h]
    diffs_v = [np.asarray(rv) - vmag_srh for _, rv in runs_vmag]
    abs_max_h = float(np.nanpercentile(np.abs(np.concatenate(diffs_h)),
                                       diff_pctile))
    abs_max_v = float(np.nanpercentile(np.abs(np.concatenate(diffs_v)),
                                       diff_pctile))
    if abs_max_h < 1e-12:
        abs_max_h = 1e-12
    if abs_max_v < 1e-12:
        abs_max_v = 1e-12

    col_clims = [
        (vmin_h, vmax_h),
        (-abs_max_h, abs_max_h),
        (vmin_v, vmax_v),
        (-abs_max_v, abs_max_v),
    ]

    # ---- Figure geometry ----
    # Aspect ratio of Savannah reach is ~ wide-and-short, but we keep the
    # same per-row height as BIC for visual consistency (set aspect="equal"
    # on each panel will leave whitespace).
    # Savannah reach is wide-and-short (~2000 m x ~700 m), so each
    # equal-aspect panel is much wider than tall. We give the figure a
    # larger fig_w to leave room for 4 columns side-by-side without
    # title collisions, and use wider wspace than BIC.
    fig_w = 11.0
    panel_h = 1.10
    fig_h = panel_h * n_runs + 1.9   # extra slack for top titles + cbar
    fig = plt.figure(figsize=(fig_w, fig_h))

    gs = fig.add_gridspec(
        nrows=n_runs, ncols=4,
        hspace=0.05, wspace=0.12,
        left=0.01, right=0.99, top=0.82, bottom=0.09,
    )

    first_pcs: List[Optional[PolyCollection]] = [None, None, None, None]

    for r in range(n_runs):
        row_label, h_pred = runs_h[r]
        _, vmag_pred = runs_vmag[r]
        diff_h = np.asarray(h_pred) - h_srh
        diff_v = np.asarray(vmag_pred) - vmag_srh
        col_arrays = [h_pred, diff_h, vmag_pred, diff_v]

        l2_h = float(np.sqrt(np.mean(diff_h ** 2)))
        l2_v = float(np.sqrt(np.mean(diff_v ** 2)))

        for c in range(4):
            ax = fig.add_subplot(gs[r, c])
            pc = PolyCollection(
                polys, array=np.asarray(col_arrays[c]), cmap="RdBu_r",
                edgecolors="face", linewidth=0, antialiased=False,
            )
            pc.set_clim(*col_clims[c])
            ax.add_collection(pc)
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
            ax.set_aspect("equal", adjustable="box")

            # Hide per-panel y-tick labels (they would otherwise collide
            # with the row label on col 0). The y-extent is the same on
            # every panel so this loses no information.
            ax.set_yticklabels([])
            if c == 0:
                ax.text(
                    -0.04, 0.5, row_label,
                    transform=ax.transAxes, ha="right", va="center",
                    fontsize=18, linespacing=1.15,
                )

            if r < n_runs - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel(r"$x$ (m)")

            if first_pcs[c] is None:
                first_pcs[c] = pc

            if c == 1:
                ax.text(
                    0.99, 0.92, rf"$L_2 = {l2_h:.2e}$",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=14,
                    bbox=dict(facecolor="white", alpha=0.75,
                              edgecolor="none", pad=2.0),
                )
            elif c == 3:
                ax.text(
                    0.99, 0.92, rf"$L_2 = {l2_v:.2e}$",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=14,
                    bbox=dict(facecolor="white", alpha=0.75,
                              edgecolor="none", pad=2.0),
                )

    # ---- Top colorbars + column titles ----
    first_row = sorted(
        (a for a in fig.axes if a.get_subplotspec().rowspan.start == 0),
        key=lambda a: a.get_position().x0,
    )
    bboxes = [a.get_position() for a in first_row]

    cbar_h = 0.014
    cbar_y = bboxes[0].y1 + 0.060

    for c in range(4):
        cax = fig.add_axes([bboxes[c].x0, cbar_y, bboxes[c].width, cbar_h])
        cbar = fig.colorbar(first_pcs[c], cax=cax, orientation="horizontal")
        cbar.ax.xaxis.set_ticks_position("bottom")
        cbar.ax.xaxis.set_label_position("top")
        cbar.set_ticks(np.linspace(*col_clims[c], 3))
        cbar.ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        cbar.ax.tick_params(labelsize=18)
        cbar.set_label(column_labels[c], fontsize=20, labelpad=8)

    title_y = cbar_y + cbar_h + 0.055
    for c in range(4):
        cx = (bboxes[c].x0 + bboxes[c].x1) / 2
        fig.text(cx, title_y, column_titles[c],
                 ha="center", va="bottom", fontsize=18)

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    logger.info(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="7-row ablation grid: SR-A...SR-G vs SRH-2D reference.",
    )
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"],
                        help="Override device.type from the configs.")
    parser.add_argument("--dpi", type=int, default=200, help="Figure DPI.")
    parser.add_argument("--no-tex", action="store_true",
                        help="Disable LaTeX rendering.")
    parser.add_argument("--out-dir", default=str(SCRIPT_DIR / "plots"),
                        help="Output directory for the combined PNG.")
    args = parser.parse_args()

    use_tex = True
    setup_publication_style(use_tex=use_tex)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build a base Config + dataset once. All SR variants share the mesh,
    # IC, BC, and architecture; only the trained network weights differ.
    # We use SR-B (the standard + anchor run) as the base config because
    # SR-A may not be trained yet when this script runs.
    base_yaml = SCRIPT_DIR / "fvm_pinn_config_SR_B.yaml"
    base_config = Config(str(base_yaml))
    if args.device:
        base_config.set("device.type", args.device)

    dataset = FVM_PINNDataset(base_config)
    mesh = dataset.get_mesh()
    logger.info(f"Mesh: {mesh.n_cells} cells, {mesh.node_xy.shape[0]} nodes")

    # ---- SRH-2D reference at t = t_end (closest snapshot) ----
    t_target = float(base_config.get("training.t_end", 3600.0))
    t_actual, h_srh, vel_srh = load_srh2d_reference(H5_REF, t_target)
    vmag_srh = np.sqrt(vel_srh[:, 0] ** 2 + vel_srh[:, 1] ** 2)
    logger.info(f"SRH-2D reference: t = {t_actual:.2f} s "
                f"(requested {t_target:.2f} s)")
    logger.info(f"  h:    [{h_srh.min():.4f}, {h_srh.max():.4f}] m")
    logger.info(f"  |u|:  [{vmag_srh.min():.4f}, {vmag_srh.max():.4f}] m/s")

    # ---- Run inference for each SR ----
    runs_h: List[Tuple[str, np.ndarray]] = []
    runs_vmag: List[Tuple[str, np.ndarray]] = []
    for run_id, _yaml, ckpt_relpath, label in SR_RUNS:
        ckpt_path = SCRIPT_DIR / "checkpoints" / run_id / ckpt_relpath
        if not ckpt_path.exists():
            logger.warning(f"[{run_id}] checkpoint missing -> skip: {ckpt_path}")
            continue
        try:
            Q = predict_at_time(base_config, dataset, ckpt_path, t_actual)
        except Exception as exc:
            logger.warning(f"[{run_id}] inference failed: {exc}")
            continue
        h_pred = Q[:, 0]
        u_pred = Q[:, 1]
        v_pred = Q[:, 2]
        vmag_pred = np.sqrt(u_pred ** 2 + v_pred ** 2)

        l2_h = float(np.sqrt(np.mean((h_pred - h_srh) ** 2)))
        l2_v = float(np.sqrt(np.mean((vmag_pred - vmag_srh) ** 2)))
        logger.info(f"[{run_id}] L2(h) = {l2_h:.4e}, L2(|u|) = {l2_v:.4e}")

        runs_h.append((label, h_pred))
        runs_vmag.append((label, vmag_pred))

    if not runs_h:
        logger.error("No SR checkpoints loaded — nothing to plot.")
        return

    # ---- Plot (single 6x4 combined figure) ----
    plot_combined_grid(
        mesh,
        runs_h=runs_h, h_srh=h_srh,
        runs_vmag=runs_vmag, vmag_srh=vmag_srh,
        column_titles=(
            "Water depth",
            "Depth difference",
            "Velocity magnitude",
            "Velocity difference",
        ),
        column_labels=(
            r"$h$ (m)",
            (r"$h - h_{\mathrm{SRH\mbox{-}2D}}$ (m)" if use_tex
             else r"$h - h_{\mathrm{SRH\!-\!2D}}$ (m)"),
            (r"$|\mathbf{u}|$ (m\,s$^{-1}$)" if use_tex
             else r"$|\mathbf{u}|$ (m s$^{-1}$)"),
            (r"$|\mathbf{u}| - |\mathbf{u}|_{\mathrm{SRH\mbox{-}2D}}$ (m\,s$^{-1}$)"
             if use_tex else
             r"$|\mathbf{u}| - |\mathbf{u}|_{\mathrm{SRH\!-\!2D}}$ (m s$^{-1}$)"),
        ),
        out_path=out_dir / "SR_ablation_combined.png",
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
