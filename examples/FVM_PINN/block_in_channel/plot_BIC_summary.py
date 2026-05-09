"""
Summary plots for the block-in-channel BIC-A ... BIC-H ablation.

Generates four separate PNGs:

  plots/BIC_L2_h.png        bar chart of L2(h - h_SRH-2D)
  plots/BIC_L2_vel.png      bar chart of L2(|u| - |u|_SRH-2D)
  plots/BIC_profile_h.png   h(x, y=2.5 m) — channel centerline profile
  plots/BIC_profile_vel.png |u|(x, y=2.5 m) — channel centerline profile

Each profile plot overlays the 8 BIC runs against the SRH-2D reference.
The centerline y = 2.5 m cuts through the block obstruction
(x ∈ [4.5, 6.5] m), where ``griddata`` returns ``NaN`` and plotting
naturally leaves a gap.

Style follows ``plot_BIC_ablation_grid.py`` and ``data/plot_case_overview.py``:
serif fonts, LaTeX rendering by default with mathtext fallback.

Usage
-----
    cd examples/FVM_PINN/block_in_channel
    python plot_BIC_summary.py                    # CPU, default 200 dpi
    python plot_BIC_summary.py --device cuda
    python plot_BIC_summary.py --no-tex
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

# Make HydroNet importable when running this script directly.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]  # block_in_channel/ -> FVM_PINN/ -> examples/ -> root
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from HydroNet import Config, FVM_PINNDataset

# Reuse the run table + inference helpers + style setup so this script
# stays in lock-step with the grid plotter.
from plot_BIC_ablation_grid import (
    BIC_RUNS,
    H5_REF,
    load_srh2d_reference,
    predict_at_time,
    setup_publication_style,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Inference: build the per-run prediction table once
# ---------------------------------------------------------------------------

def collect_run_results(
    base_config: Config,
    dataset: FVM_PINNDataset,
    t_query: float,
) -> List[Tuple[str, str, np.ndarray, np.ndarray]]:
    """For each BIC run with a valid checkpoint, return
    ``(run_id, label, h_pred[n_cells], vmag_pred[n_cells])``.

    Skips runs whose checkpoint is missing or whose inference raises.
    """
    results: List[Tuple[str, str, np.ndarray, np.ndarray]] = []
    for run_id, _yaml, ckpt_filename, label in BIC_RUNS:
        ckpt_path = SCRIPT_DIR / "checkpoints" / run_id / ckpt_filename
        if not ckpt_path.exists():
            logger.warning(f"[{run_id}] checkpoint missing -> skip: {ckpt_path}")
            continue
        try:
            Q = predict_at_time(base_config, dataset, ckpt_path, t_query)
        except Exception as exc:
            logger.warning(f"[{run_id}] inference failed: {exc}")
            continue
        h_pred = Q[:, 0]
        u_pred = Q[:, 1]
        v_pred = Q[:, 2]
        vmag_pred = np.sqrt(u_pred ** 2 + v_pred ** 2)
        results.append((run_id, label, h_pred, vmag_pred))
        logger.info(
            f"[{run_id}] inference complete "
            f"(h_range=[{h_pred.min():.3f},{h_pred.max():.3f}], "
            f"|u|_range=[{vmag_pred.min():.3f},{vmag_pred.max():.3f}])"
        )
    return results


# ---------------------------------------------------------------------------
# Bar chart
# ---------------------------------------------------------------------------

def plot_bar(
    ids: List[str],
    l2_values: List[float],
    *,
    ylabel: str,
    title: str,
    out_path: Path,
    dpi: int,
    xtick_labels: Optional[List[str]] = None,
    highlight_id: Optional[str] = "BIC-A",
) -> None:
    """Bar chart of L2 errors with one bar per BIC run.

    ``xtick_labels`` gives the per-bar x-axis description (typically the
    run id plus the run condition on a second line). When provided the
    labels are rotated 45 degrees and right-aligned so they read cleanly
    even when they're long.
    """
    fig, ax = plt.subplots(figsize=(10, 7.0))

    colors = [
        "#d62728" if rid == highlight_id else "#1f77b4"
        for rid in ids
    ]
    bars = ax.bar(
        ids, l2_values, color=colors,
        edgecolor="black", linewidth=0.6,
    )

    # Numeric value above each bar
    for bar, val in zip(bars, l2_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.2e}",
            ha="center", va="bottom",
            fontsize=18,
        )

    ax.set_yscale("log")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", which="both", alpha=0.3)
    ax.set_axisbelow(True)
    # Slight upper headroom so the value labels don't crowd the spine
    y_top = max(l2_values)
    ax.set_ylim(top=y_top * 3.0)

    # X-tick labels: BIC-X on the first line, run condition on the second.
    # Rotated 45 degrees with ha='right' so the right edge anchors near
    # the tick.
    if xtick_labels is not None:
        ax.set_xticks(range(len(ids)))
        ax.set_xticklabels(
            xtick_labels, rotation=45, ha="right", rotation_mode="anchor",
            fontsize=16, linespacing=1.15,
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Centerline profile
# ---------------------------------------------------------------------------

def interp_along_line(
    cell_xy: np.ndarray,
    values: np.ndarray,
    *,
    y_line: float = 2.5,
    x_min: float = 0.0,
    x_max: float = 15.0,
    n: int = 400,
    block_x: Tuple[float, float] = (4.5, 6.5),
    block_y: Tuple[float, float] = (2.0, 3.0),
) -> Tuple[np.ndarray, np.ndarray]:
    """Linear interpolation of cell-centred ``values`` onto the
    ``y = y_line`` line over ``[x_min, x_max]``.

    ``griddata`` interpolates across the block hole because the cell-centre
    convex hull contains the block. To restore physical correctness we
    explicitly NaN-out samples that fall inside the block rectangle so
    matplotlib draws a clean gap there.
    """
    xs = np.linspace(x_min, x_max, n)
    line = np.column_stack([xs, np.full_like(xs, y_line)])
    vals = griddata(cell_xy, values, line, method="linear")

    if (block_x is not None and block_y is not None
            and block_y[0] <= y_line <= block_y[1]):
        in_block = (xs >= block_x[0]) & (xs <= block_x[1])
        vals = np.where(in_block, np.nan, vals)
    return xs, vals


def plot_profile(
    runs: List[Tuple[str, np.ndarray]],   # (run_id, values[n_cells])
    srh_values: np.ndarray,
    cell_xy: np.ndarray,
    *,
    ylabel: str,
    title: str,
    out_path: Path,
    dpi: int,
    y_line: float = 2.5,
    x_range: Tuple[float, float] = (0.0, 15.0),
) -> None:
    """Overlay the SRH-2D reference and all BIC runs along ``y = y_line``."""
    fig, ax = plt.subplots(figsize=(11.5, 6.0))

    # SRH-2D reference: thick black, drawn last (highest z) for visibility
    xs, srh_line = interp_along_line(cell_xy, srh_values, y_line=y_line,
                                     x_min=x_range[0], x_max=x_range[1])

    # Distinct colour per BIC run
    cmap = plt.get_cmap("tab10")
    for i, (run_id, vals) in enumerate(runs):
        _, run_line = interp_along_line(cell_xy, vals, y_line=y_line,
                                        x_min=x_range[0], x_max=x_range[1])
        ax.plot(
            xs, run_line,
            color=cmap(i % 10), lw=1.6, alpha=0.9,
            label=run_id.replace("_", "-"),
        )
    ax.plot(xs, srh_line, "k-", lw=2.6, label="SRH-2D", zorder=20)

    # Mark the block extent (x in [4.5, 6.5]) with a faint shaded region
    ax.axvspan(4.5, 6.5, color="0.85", alpha=0.6, zorder=1, label="Block")

    ax.set_xlabel(r"$x$ (m)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(x_range)
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(loc="best", ncol=3, fontsize=14, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=("Bar charts and centerline profiles for the BIC-A...BIC-H "
                     "ablation against the SRH-2D reference."),
    )
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"],
                        help="Override device.type from the configs.")
    parser.add_argument("--dpi", type=int, default=200, help="Figure DPI.")
    parser.add_argument("--no-tex", action="store_true",
                        help="Disable LaTeX rendering.")
    parser.add_argument("--out-dir", default=str(SCRIPT_DIR / "plots"),
                        help="Output directory for the four PNGs.")
    parser.add_argument("--y-line", type=float, default=2.5,
                        help="y-coordinate of the profile slice. Default: 2.5 m.")
    args = parser.parse_args()

    setup_publication_style(use_tex=not args.no_tex)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Build base config + dataset (once) ----
    base_yaml = SCRIPT_DIR / "fvm_pinn_config_BIC_A.yaml"
    base_config = Config(str(base_yaml))
    if args.device:
        base_config.set("device.type", args.device)

    dataset = FVM_PINNDataset(base_config)
    mesh = dataset.get_mesh()
    cell_xy = np.asarray(mesh.cell_center)
    logger.info(f"Mesh: {mesh.n_cells} cells, {mesh.node_xy.shape[0]} nodes")

    # ---- SRH-2D reference at t = t_end (closest snapshot) ----
    t_target = float(base_config.get("training.t_end", 360.0))
    t_actual, h_srh, vel_srh = load_srh2d_reference(H5_REF, t_target)
    vmag_srh = np.sqrt(vel_srh[:, 0] ** 2 + vel_srh[:, 1] ** 2)
    logger.info(f"SRH-2D reference: t = {t_actual:.2f} s "
                f"(requested {t_target:.2f} s)")

    # ---- Inference for every BIC run ----
    results = collect_run_results(base_config, dataset, t_actual)
    if not results:
        logger.error("No BIC checkpoints loaded — nothing to plot.")
        return

    ids = [r[0].replace("_", "-") for r in results]
    # Per-bar x-tick labels: full BIC-X + run condition (two lines).
    # The labels in BIC_RUNS already follow that convention.
    xtick_labels = [r[1] for r in results]
    l2_h = [float(np.sqrt(np.mean((r[2] - h_srh) ** 2))) for r in results]
    l2_v = [float(np.sqrt(np.mean((r[3] - vmag_srh) ** 2))) for r in results]

    for rid, lh, lv in zip(ids, l2_h, l2_v):
        logger.info(f"[{rid}] L2(h) = {lh:.4e}, L2(|u|) = {lv:.4e}")

    # ---- Bar charts ----
    plot_bar(
        ids, l2_h,
        ylabel=r"$L_2(h)$ (m)",
        title=r"Water depth $L_2$ error vs SRH-2D",
        out_path=out_dir / "BIC_L2_h.png",
        dpi=args.dpi,
        xtick_labels=xtick_labels,
    )
    plot_bar(
        ids, l2_v,
        ylabel=r"$L_2(|\mathbf{u}|)$ (m\,s$^{-1}$)",
        title=r"Velocity magnitude $L_2$ error vs SRH-2D",
        out_path=out_dir / "BIC_L2_vel.png",
        dpi=args.dpi,
        xtick_labels=xtick_labels,
    )

    # ---- Centerline profiles ----
    domain_x = (
        float(np.min(cell_xy[:, 0])),
        float(np.max(cell_xy[:, 0])),
    )

    h_runs: List[Tuple[str, np.ndarray]] = [(r[0], r[2]) for r in results]
    v_runs: List[Tuple[str, np.ndarray]] = [(r[0], r[3]) for r in results]

    plot_profile(
        h_runs, h_srh, cell_xy,
        ylabel=r"$h$ (m)",
        title=rf"Water depth along $y = {args.y_line:g}$ m centerline",
        out_path=out_dir / "BIC_profile_h.png",
        dpi=args.dpi,
        y_line=args.y_line,
        x_range=domain_x,
    )
    plot_profile(
        v_runs, vmag_srh, cell_xy,
        ylabel=r"$|\mathbf{u}|$ (m\,s$^{-1}$)",
        title=rf"Velocity magnitude along $y = {args.y_line:g}$ m centerline",
        out_path=out_dir / "BIC_profile_vel.png",
        dpi=args.dpi,
        y_line=args.y_line,
        x_range=domain_x,
    )


if __name__ == "__main__":
    main()
