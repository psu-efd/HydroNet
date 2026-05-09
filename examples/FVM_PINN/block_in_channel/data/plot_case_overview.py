"""
Plot the block-in-channel case: mesh, water depth contour, and velocity-magnitude contour.

Reads:
  - block_in_channel.srhhydro / .srhgeom / .srhmat  (SRH-2D case files)
  - block_in_channel_XMDFC.h5                        (SRH-2D output snapshots)

For this case the bed is flat, so water depth ``h`` equals water surface
elevation; the depth contour is therefore equivalent to a WSE contour.
Velocity magnitude is :math:`\\|\\mathbf{u}\\| = \\sqrt{u^2 + v^2}`.

Output (each as a separate PNG in this directory by default):
  - block_in_channel_mesh.png
  - block_in_channel_depth.png      (h, equivalent to WSE for flat bed)
  - block_in_channel_velocity.png   (velocity magnitude)

Usage
-----
    cd examples/FVM_PINN/block_in_channel/data
    python plot_case_overview.py                 # last snapshot, 300 dpi
    python plot_case_overview.py --time-index 0  # initial snapshot
    python plot_case_overview.py --no-tex        # disable LaTeX rendering
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection

# Make HydroNet importable when running this script directly.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]  # data/ -> case/ -> FVM_PINN/ -> examples/ -> root
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from HydroNet.models.FVM_PINN._internal.mesh.mesh_topology import build_mesh
from HydroNet.models.FVM_PINN._internal.mesh.srh2d_reader import (
    SRH2DMeshReader,
    SRH2DRawData,
)

plt.rc('text', usetex=True)  #allow the use of Latex for math expressions and equations
plt.rc('font', family='serif') #specify the default font family to be "serif"

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# Default file locations (relative to this script)
HYDRO_PATH = SCRIPT_DIR / "block_in_channel.srhhydro"
H5_PATH = SCRIPT_DIR / "block_in_channel_XMDFC.h5"


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_case() -> Tuple[object, SRH2DRawData]:
    """Parse the SRH-2D files and build the unstructured mesh."""
    reader = SRH2DMeshReader(str(HYDRO_PATH))
    raw = reader.read()
    mesh = build_mesh(raw)
    logger.info(f"Mesh: {mesh.n_cells} cells, "
                f"{len(raw.node_coords)} nodes, {mesh.n_faces} faces")
    return mesh, raw


def load_h5_fields(
    h5_path: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(times, depth, vel)`` from the SRH-2D XMDFC h5 file.

    Shapes
    ------
    times : ``[n_t]``
    depth : ``[n_t, n_cells]``
    vel   : ``[n_t, n_cells, 2]``  (u, v components)
    """
    with h5py.File(h5_path, "r") as f:
        times = f["Water_Depth_m/Times"][:].astype(np.float64)
        depth = f["Water_Depth_m/Values"][:].astype(np.float64)
        vel = f["Velocity_m_p_s/Values"][:].astype(np.float64)
    return times, depth, vel


def cell_polygons(raw: SRH2DRawData) -> list:
    """List of [n_verts, 2] node-coordinate polygons, one per cell."""
    nodes_xy = raw.node_coords[:, :2]
    return [nodes_xy[ids] for ids in raw.elem_nodes]


def cell_centers(raw: SRH2DRawData) -> np.ndarray:
    """Return ``[n_cells, 2]`` cell-centroid coordinates (vertex average)."""
    nodes_xy = raw.node_coords[:, :2]
    centers = np.empty((len(raw.elem_nodes), 2), dtype=np.float64)
    for i, ids in enumerate(raw.elem_nodes):
        centers[i] = nodes_xy[ids].mean(axis=0)
    return centers


def sparse_sample_indices(
    h5_path: Path,
    *,
    n_points: int = 50,
    seed: int = 42,
    h_dry: float = 0.01,
    target_time: float = 360.0,
) -> np.ndarray:
    """Reproduce the BIC-C sparse-sampling rule from
    ``HydroNet/models/FVM_PINN/data.py:_ref_data_sparse``: pick
    ``n_points`` wet cells uniformly at random (seeded) at the snapshot
    closest to ``target_time``.

    With ``seed=42`` the 50-point set is a strict subset of the 200-point
    set (BIC-B) — both calls share an RNG stream and ``rng.choice`` with
    ``replace=False`` returns the prefix of an internal permutation.
    """
    with h5py.File(h5_path, "r") as f:
        times = f["Water_Depth_m/Times"][:].astype(np.float64)
        depth = f["Water_Depth_m/Values"][:].astype(np.float64)
    ti = int(np.argmin(np.abs(times - target_time)))
    h_at_t = depth[ti]
    wet = np.where(h_at_t > h_dry)[0]
    take = min(n_points, len(wet))
    rng = np.random.default_rng(seed)
    return rng.choice(wet, size=take, replace=False)


def read_vtk_cell_scalars(vtk_path: Path) -> Tuple[float, Dict[str, np.ndarray]]:
    """Parse an ASCII VTK file's ``CELL_DATA`` scalar arrays.

    Lightweight reader sufficient for the ``fvm_t*.vtk`` outputs written by
    ``run_fvm_rk2`` (header line, ``UNSTRUCTURED_GRID``, then a ``CELL_DATA``
    block of named scalars). Vector arrays in the same file are skipped —
    velocity components are already exposed as ``Velocity_X_m_s`` /
    ``Velocity_Y_m_s`` scalars.

    Returns
    -------
    sim_time : float
        Simulation time parsed from the VTK title line (e.g.
        ``"FVM-PINN SWE solution t=360.0000"``). Falls back to ``0.0`` if
        the title format is not recognised.
    fields : dict
        Mapping ``name -> [n_cells]`` array of cell-centred values.
    """
    with open(vtk_path, "r") as f:
        lines = f.readlines()

    sim_time = 0.0
    if len(lines) >= 2 and "t=" in lines[1]:
        try:
            sim_time = float(lines[1].strip().split("t=")[-1].strip())
        except ValueError:
            pass

    n_cells: Optional[int] = None
    cell_data_idx: Optional[int] = None
    for i, line in enumerate(lines):
        if line.startswith("CELL_DATA"):
            n_cells = int(line.split()[1])
            cell_data_idx = i + 1
            break
    if n_cells is None or cell_data_idx is None:
        raise ValueError(f"No CELL_DATA section found in {vtk_path}")

    fields: Dict[str, np.ndarray] = {}
    i = cell_data_idx
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("SCALARS"):
            name = line.split()[1]
            # Next line is ``LOOKUP_TABLE default``; values follow.
            i += 2
            values = np.fromiter(
                (float(lines[i + k].strip()) for k in range(n_cells)),
                dtype=np.float64, count=n_cells,
            )
            fields[name] = values
            i += n_cells
        elif line.startswith("VECTORS"):
            # Skip vector block (n_cells lines of "vx vy vz")
            i += 1 + n_cells
        else:
            i += 1

    return sim_time, fields


# ---------------------------------------------------------------------------
# Plot styling helpers
# ---------------------------------------------------------------------------

def setup_publication_style(use_tex: bool) -> None:
    """Apply consistent serif + size settings for publication-quality output."""
    plt.rcdefaults()
    if use_tex:
        try:
            plt.rc("text", usetex=True)
            plt.rc("font", family="serif")
        except Exception:
            logger.warning("LaTeX not available — falling back to mathtext.")
            plt.rc("text", usetex=False)
            plt.rc("font", family="serif")
    else:
        plt.rc("text", usetex=False)
        plt.rc("font", family="serif")
    plt.rc("font", size=16)
    plt.rc("axes", labelsize=22, titlesize=22)
    plt.rc("xtick", labelsize=18)
    plt.rc("ytick", labelsize=18)
    plt.rc("legend", fontsize=18)


def domain_extent(raw: SRH2DRawData) -> Tuple[float, float, float, float]:
    nodes = raw.node_coords[:, :2]
    return float(nodes[:, 0].min()), float(nodes[:, 0].max()), \
           float(nodes[:, 1].min()), float(nodes[:, 1].max())


def figsize_for_aspect(extent, target_height: float = 4.0) -> Tuple[float, float]:
    """Pick a (width, height) preserving the domain aspect ratio."""
    x_min, x_max, y_min, y_max = extent
    dx = max(x_max - x_min, 1e-9)
    dy = max(y_max - y_min, 1e-9)
    aspect = dx / dy
    width = target_height * aspect + 1.8  # slack for colorbar/labels
    return (width, target_height + 1.0)


# ---------------------------------------------------------------------------
# Plotters
# ---------------------------------------------------------------------------

def plot_mesh(
    raw: SRH2DRawData,
    out_path: Path,
    dpi: int,
    *,
    sample_xy: Optional[np.ndarray] = None,
    sample_label: Optional[str] = None,
) -> None:
    """Draw the mesh wireframe; optionally overlay sparse measurement
    locations as red dots (used to visualise the BIC-B/C random sample)."""
    polys = cell_polygons(raw)
    extent = domain_extent(raw)

    fig, ax = plt.subplots(figsize=figsize_for_aspect(extent))
    pc = PolyCollection(
        polys, facecolor="white", edgecolor="black", linewidth=0.35,
    )
    ax.add_collection(pc)

    if sample_xy is not None and len(sample_xy) > 0:
        ax.scatter(
            sample_xy[:, 0], sample_xy[:, 1],
            s=28, c="#d62728", marker="o",
            edgecolor="black", linewidth=0.4,
            zorder=5,
            label=sample_label or f"Sparse measurements ({len(sample_xy)})",
        )
        ax.legend(loc="upper right", fontsize=18, framealpha=0.85)

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$x$ (m)")
    ax.set_ylabel(r"$y$ (m)")
    ax.set_title(
        rf"Block-in-channel mesh",
        fontsize=16,
    )

    # Subtle box frame
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {out_path}")


def plot_field(
    raw: SRH2DRawData,
    values: np.ndarray,
    cbar_label: str,
    title: str,
    out_path: Path,
    *,
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    dpi: int = 300,
    show_mesh_overlay: bool = False,
) -> None:
    polys = cell_polygons(raw)
    extent = domain_extent(raw)

    if vmin is None:
        vmin = float(np.nanmin(values))
    if vmax is None:
        vmax = float(np.nanmax(values))
    # Avoid degenerate colormaps when the field is uniform
    if vmax - vmin < 1e-12:
        vmax = vmin + 1e-12

    fig, ax = plt.subplots(figsize=figsize_for_aspect(extent))
    pc = PolyCollection(
        polys,
        array=np.asarray(values),
        cmap=cmap,
        edgecolor=("0.4" if show_mesh_overlay else "none"),
        linewidth=(0.15 if show_mesh_overlay else 0.0),
    )
    pc.set_clim(vmin, vmax)
    ax.add_collection(pc)

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$x$ (m)")
    ax.set_ylabel(r"$y$ (m)")
    ax.set_title(title)

    #set tick fontsize
    #ax.tick_params(axis='both', which='major', labelsize=14)

    cbar = fig.colorbar(pc, ax=ax, shrink=0.9, pad=0.02)
    cbar.set_label(cbar_label)
    cbar.outline.set_linewidth(0.6)

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=("Generate publication-quality plots of the block-in-channel case: "
                     "mesh, water-depth contour, velocity-magnitude contour."),
    )
    parser.add_argument(
        "--time-index", type=int, default=-1,
        help="Snapshot index in the XMDFC h5 (negative indexing allowed). Default: -1 (last).",
    )
    parser.add_argument(
        "--dpi", type=int, default=300,
        help="Figure DPI. Default: 300.",
    )
    parser.add_argument(
        "--out-dir", default=str(SCRIPT_DIR),
        help="Output directory for the three PNGs. Default: this script's directory.",
    )
    parser.add_argument(
        "--depth-cmap", default=plt.cm.RdBu_r,
        help="Colormap for the water-depth field (default: cividis).",
    )
    parser.add_argument(
        "--vel-cmap", default=plt.cm.RdBu_r,
        help="Colormap for velocity magnitude (default: magma).",
    )
    parser.add_argument(
        "--no-tex", action="store_true",
        help="Disable LaTeX rendering (use matplotlib mathtext instead).",
    )
    parser.add_argument(
        "--mesh-overlay", action="store_true",
        help="Overlay light cell edges on the contour plots.",
    )
    parser.add_argument(
        "--vtk-path",
        default=str(SCRIPT_DIR.parent / "vtk_fvm_only" / "fvm_t0010.vtk"),
        help=("FVM-only (teacher) VTK snapshot to plot alongside the SRH-2D reference. "
              "Default: ../vtk_fvm_only/fvm_t0010.vtk. Pass empty string to skip."),
    )
    parser.add_argument(
        "--vtk-out-dir",
        default=str(SCRIPT_DIR.parent / "vtk_fvm_only"),
        help="Output directory for the FVM-only PNGs. Default: ../vtk_fvm_only.",
    )
    parser.add_argument(
        "--n-sparse", type=int, default=50,
        help=("Number of sparse measurement points to overlay on the mesh "
              "plot (random wet cells, seed=42 — matches BIC-C). "
              "Pass 0 to disable. Default: 50."),
    )
    parser.add_argument(
        "--sparse-seed", type=int, default=42,
        help="RNG seed for the sparse sampling. Default: 42 (matches BIC-C/B).",
    )
    parser.add_argument(
        "--sparse-time", type=float, default=360.0,
        help="Snapshot time at which to evaluate `h > h_dry` for the wet-cell "
             "filter when sampling. Default: 360.0 s.",
    )
    args = parser.parse_args()

    setup_publication_style(use_tex=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load case + reference fields ----
    mesh, raw = load_case()
    times, depth, vel = load_h5_fields(H5_PATH)
    logger.info(f"H5: {len(times)} snapshots, t = [{times[0]:.3f}, {times[-1]:.3f}] s")

    ti = args.time_index if args.time_index >= 0 else len(times) + args.time_index
    if not (0 <= ti < len(times)):
        raise IndexError(
            f"--time-index {args.time_index} out of range for {len(times)} snapshots"
        )
    t_val = float(times[ti])
    h_t = depth[ti]                                 # [n_cells]
    vel_t = vel[ti]                                 # [n_cells, 2]
    vel_mag = np.sqrt(vel_t[:, 0] ** 2 + vel_t[:, 1] ** 2)

    logger.info(f"Snapshot t = {t_val:.3f} s")
    logger.info(f"  h:      [{h_t.min():.4f}, {h_t.max():.4f}] m")
    logger.info(f"  |u|:    [{vel_mag.min():.4f}, {vel_mag.max():.4f}] m/s")

    # ---- Sparse sample overlay (matches BIC-C config: n=50, seed=42) ----
    sample_xy: Optional[np.ndarray] = None
    sample_label: Optional[str] = None
    if args.n_sparse > 0:
        h_dry = 0.01  # matches BIC configs (`physics.h_dry`)
        idx = sparse_sample_indices(
            H5_PATH,
            n_points=args.n_sparse,
            seed=args.sparse_seed,
            h_dry=h_dry,
            target_time=args.sparse_time,
        )
        sample_xy = cell_centers(raw)[idx]
        sample_label = (rf"{len(idx)} sparse points ")
        logger.info(
            f"Sparse sample: {len(idx)} cells "
            f"at t = {args.sparse_time:.1f} s, seed={args.sparse_seed}"
        )

    # ---- Plot ----
    plot_mesh(
        raw, out_dir / "block_in_channel_mesh.png", dpi=args.dpi,
        sample_xy=sample_xy, sample_label=sample_label,
    )

    plot_field(
        raw,
        h_t,
        cbar_label=r"$h$ (m)",
        title=rf"SRH-2D: Water depth $h$ at steady state (flat bed: $h \equiv \mathrm{{WSE}}$)",
        out_path=out_dir / "block_in_channel_depth.png",
        cmap=args.depth_cmap,
        dpi=args.dpi,
        show_mesh_overlay=args.mesh_overlay,
    )

    plot_field(
        raw,
        vel_mag,
        cbar_label=r"$|\mathbf{u}|$ (m\,s$^{-1}$)",
        title=rf"SRH-2D: Velocity magnitude $|\mathbf{{u}}|$ at steady state",
        out_path=out_dir / "block_in_channel_velocity.png",
        cmap=args.vel_cmap,
        dpi=args.dpi,
        show_mesh_overlay=args.mesh_overlay,
    )

    # ---- FVM-only (teacher) VTK snapshot, if requested ----
    if args.vtk_path:
        vtk_path = Path(args.vtk_path)
        if not vtk_path.exists():
            logger.warning(f"VTK file not found, skipping FVM-only plots: {vtk_path}")
        else:
            sim_time, fvm_fields = read_vtk_cell_scalars(vtk_path)
            required = {"Water_Depth_m", "Velocity_X_m_s", "Velocity_Y_m_s"}
            missing = required - set(fvm_fields)
            if missing:
                logger.warning(
                    f"VTK {vtk_path.name} missing fields {missing} — skipping FVM-only plots"
                )
            else:
                if len(fvm_fields["Water_Depth_m"]) != len(raw.elem_nodes):
                    logger.warning(
                        f"VTK cell count {len(fvm_fields['Water_Depth_m'])} does not match "
                        f"mesh cell count {len(raw.elem_nodes)} — plotting may be misaligned."
                    )
                h_fvm = fvm_fields["Water_Depth_m"]
                vx_fvm = fvm_fields["Velocity_X_m_s"]
                vy_fvm = fvm_fields["Velocity_Y_m_s"]
                vmag_fvm = np.sqrt(vx_fvm ** 2 + vy_fvm ** 2)

                logger.info(f"FVM-only VTK: t = {sim_time:.3f} s ({vtk_path.name})")
                logger.info(f"  h:      [{h_fvm.min():.4f}, {h_fvm.max():.4f}] m")
                logger.info(f"  |u|:    [{vmag_fvm.min():.4f}, {vmag_fvm.max():.4f}] m/s")

                vtk_out_dir = Path(args.vtk_out_dir)
                vtk_out_dir.mkdir(parents=True, exist_ok=True)

                plot_field(
                    raw,
                    h_fvm,
                    cbar_label=r"$h$ (m)",
                    title=(rf"FVM teacher: water depth $h$ at steady state "
                           rf"(flat bed: $h \equiv \mathrm{{WSE}}$)"),
                    out_path=vtk_out_dir / "fvm_only_depth.png",
                    cmap=args.depth_cmap,
                    dpi=args.dpi,
                    show_mesh_overlay=args.mesh_overlay,
                )

                plot_field(
                    raw,
                    vmag_fvm,
                    cbar_label=r"$|\mathbf{u}|$ (m\,s$^{-1}$)",
                    title=(rf"FVM teacher: velocity magnitude $|\mathbf{{u}}|$ "
                           rf"at steady state"),
                    out_path=vtk_out_dir / "fvm_only_velocity.png",
                    cmap=args.vel_cmap,
                    dpi=args.dpi,
                    show_mesh_overlay=args.mesh_overlay,
                )


if __name__ == "__main__":
    main()
