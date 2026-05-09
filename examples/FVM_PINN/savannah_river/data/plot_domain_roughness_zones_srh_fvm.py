"""
Plot the Savannah River case overview as six separate PNGs:

  savannah_bathymetry.png        bed elevation z_b
  savannah_manning_zones.png     Manning's n roughness zones (discrete)
  savannah_srh2d_h.png           SRH-2D water depth at the chosen snapshot
  savannah_srh2d_velocity.png    SRH-2D velocity magnitude at the same snapshot
  savannah_fvm_h.png             FVM-only water depth (vtk_fvm_only/fvm_t0010.vtk)
  savannah_fvm_velocity.png      FVM-only velocity magnitude

Sources
-------
  - savana_SI.srhhydro / .srhgeom / .srhmat   (mesh + Manning zones)
  - Savana_XMDFC.h5                            (SRH-2D reference snapshots)
  - ../vtk_fvm_only/fvm_t0010.vtk              (FVM teacher result)

Plotting style mirrors
``examples/FVM_PINN/block_in_channel/data/plot_case_overview.py``:
serif fonts (LaTeX), per-cell ``PolyCollection`` with
``edgecolors="face"`` + ``antialiased=False`` to suppress sub-pixel mesh
artefacts, equal-aspect axes, colorbar on the right.

Usage
-----
    cd examples/FVM_PINN/savannah_river/data
    python plot_domain_roughness_zones_srh_fvm.py                   # last h5 snapshot, 300 dpi
    python plot_domain_roughness_zones_srh_fvm.py --time-index 0    # initial snapshot
    python plot_domain_roughness_zones_srh_fvm.py --no-tex          # mathtext fallback
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
from matplotlib.colors import ListedColormap

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
HYDRO_PATH = SCRIPT_DIR / "savana_SI.srhhydro"
H5_PATH = SCRIPT_DIR / "Savana_XMDFC.h5"
FVM_VTK_PATH = SCRIPT_DIR.parent / "vtk_fvm_only" / "fvm_t0010.vtk"


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_case() -> Tuple[object, SRH2DRawData]:
    """Parse the SRH-2D files and build the unstructured mesh."""
    reader = SRH2DMeshReader(str(HYDRO_PATH))
    raw = reader.read()
    mesh = build_mesh(raw)
    logger.info(
        f"Mesh: {mesh.n_cells} cells, "
        f"{len(raw.node_coords)} nodes, {mesh.n_faces} faces"
    )
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
    """List of [n_verts, 2] node-coordinate polygons, one per cell.

    PolyCollection accepts triangles, quads, or arbitrary polygons —
    no triangulation step needed.
    """
    nodes_xy = raw.node_coords[:, :2]
    return [nodes_xy[ids] for ids in raw.elem_nodes]


def cell_centered_from_nodal(
    raw: SRH2DRawData, nodal_values: np.ndarray,
) -> np.ndarray:
    """Vertex-average a nodal field onto cell centres for PolyCollection."""
    cell_vals = np.empty(len(raw.elem_nodes), dtype=np.float64)
    for i, ids in enumerate(raw.elem_nodes):
        cell_vals[i] = nodal_values[ids].mean()
    return cell_vals


def read_vtk_cell_scalars(vtk_path: Path) -> Tuple[float, Dict[str, np.ndarray]]:
    """Parse an ASCII VTK file's ``CELL_DATA`` scalar arrays.

    Lightweight reader sufficient for the ``fvm_t*.vtk`` outputs written
    by ``run_fvm_rk2`` (header line, ``UNSTRUCTURED_GRID``, then a
    ``CELL_DATA`` block of named scalars). Vector arrays are skipped —
    velocity components are exposed as ``Velocity_X_m_s`` /
    ``Velocity_Y_m_s`` scalars.

    Returns ``(sim_time, {name: [n_cells] array})``. Time is parsed from
    the title line ``"FVM-PINN SWE solution t=<value>"``; falls back to 0.
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
            i += 2  # skip LOOKUP_TABLE line
            values = np.fromiter(
                (float(lines[i + k].strip()) for k in range(n_cells)),
                dtype=np.float64, count=n_cells,
            )
            fields[name] = values
            i += n_cells
        elif line.startswith("VECTORS"):
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
    return (
        float(nodes[:, 0].min()), float(nodes[:, 0].max()),
        float(nodes[:, 1].min()), float(nodes[:, 1].max()),
    )


def figsize_for_aspect(extent, target_height: float = 5.0) -> Tuple[float, float]:
    """Pick a (width, height) preserving the domain aspect ratio."""
    x_min, x_max, y_min, y_max = extent
    dx = max(x_max - x_min, 1e-9)
    dy = max(y_max - y_min, 1e-9)
    aspect = dx / dy
    width = target_height * aspect + 1.8  # slack for colorbar/labels
    return (width, target_height + 1.0)


# ---------------------------------------------------------------------------
# Continuous field plot (bathymetry, h, |u|)
# ---------------------------------------------------------------------------

def plot_field(
    raw: SRH2DRawData,
    values: np.ndarray,
    cbar_label: str,
    title: str,
    out_path: Path,
    *,
    cmap=plt.cm.viridis,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    dpi: int = 300,
) -> None:
    polys = cell_polygons(raw)
    extent = domain_extent(raw)

    if vmin is None:
        vmin = float(np.nanmin(values))
    if vmax is None:
        vmax = float(np.nanmax(values))
    if vmax - vmin < 1e-12:
        vmax = vmin + 1e-12

    fig, ax = plt.subplots(figsize=figsize_for_aspect(extent))
    pc = PolyCollection(
        polys,
        array=np.asarray(values),
        cmap=cmap,
        edgecolors="face",
        linewidth=0,
        antialiased=False,
    )
    pc.set_clim(vmin, vmax)
    ax.add_collection(pc)

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$x$ (m)")
    ax.set_ylabel(r"$y$ (m)")
    ax.set_title(title)

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
# Manning zones (discrete colorbar)
# ---------------------------------------------------------------------------

def plot_manning_zones(
    raw: SRH2DRawData, out_path: Path, dpi: int = 300,
) -> None:
    """Plot Manning's-n roughness zones with a discrete colorbar.

    Each zone gets one colour from ``tab10`` (or ``viridis`` if more than
    10 zones), and the colorbar labels the actual ``n`` value of each zone.
    """
    polys = cell_polygons(raw)
    extent = domain_extent(raw)

    sorted_zone_ids = sorted(raw.manning_n.keys())
    n_zones = len(sorted_zone_ids)
    zone_n_values = np.array([raw.manning_n[zid] for zid in sorted_zone_ids])
    zone_id_to_idx = {zid: i for i, zid in enumerate(sorted_zone_ids)}
    cell_idx = np.array(
        [zone_id_to_idx[mat] for mat in raw.elem_material],
        dtype=np.float64,
    )

    if n_zones <= 10:
        cmap_zones = ListedColormap(plt.cm.tab10.colors[:n_zones])
    else:
        cmap_zones = ListedColormap(plt.cm.viridis(np.linspace(0, 1, n_zones)))

    fig, ax = plt.subplots(figsize=figsize_for_aspect(extent))
    pc = PolyCollection(
        polys,
        array=cell_idx,
        cmap=cmap_zones,
        edgecolors="face",
        linewidth=0,
        antialiased=False,
    )
    pc.set_clim(-0.5, n_zones - 0.5)
    ax.add_collection(pc)

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$x$ (m)")
    ax.set_ylabel(r"$y$ (m)")
    ax.set_title(rf"Manning's $n$ roughness zones ({n_zones} zones)")

    cbar = fig.colorbar(
        pc, ax=ax, shrink=0.9, pad=0.02, ticks=np.arange(n_zones),
    )
    cbar.ax.set_yticklabels([f"$n = {v:.3f}$" for v in zone_n_values])
    cbar.set_label(r"Manning's $n$ (s\,m$^{-1/3}$)")
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
        description=("Generate publication-quality Savannah River plots: "
                     "bathymetry, Manning zones, SRH-2D h/|u|, FVM-only h/|u|."),
    )
    parser.add_argument(
        "--time-index", type=int, default=-1,
        help="Snapshot index in Savana_XMDFC.h5 (negative allowed). Default: -1 (last).",
    )
    parser.add_argument(
        "--dpi", type=int, default=300,
        help="Figure DPI. Default: 300.",
    )
    parser.add_argument(
        "--out-dir", default=str(SCRIPT_DIR),
        help="Output directory for the six PNGs. Default: this script's directory.",
    )
    parser.add_argument(
        "--no-tex", action="store_true",
        help="Disable LaTeX rendering (use matplotlib mathtext instead).",
    )
    parser.add_argument(
        "--bathy-cmap", default=plt.cm.terrain,
        help="Colormap for the bathymetry field (default: terrain).",
    )
    parser.add_argument(
        "--depth-cmap", default=plt.cm.RdBu_r,
        help="Colormap for the water-depth field (default: RdBu_r).",
    )
    parser.add_argument(
        "--vel-cmap", default=plt.cm.RdBu_r,
        help="Colormap for velocity magnitude (default: RdBu_r).",
    )
    parser.add_argument(
        "--vtk-path", default=str(FVM_VTK_PATH),
        help=("FVM-only VTK snapshot. Default: ../vtk_fvm_only/fvm_t0010.vtk. "
              "Pass empty string to skip the FVM plots."),
    )
    args = parser.parse_args()

    setup_publication_style(use_tex=not args.no_tex)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load case ----
    mesh, raw = load_case()

    # ---- 1. Bathymetry ----
    bed_at_nodes = raw.node_coords[:, 2]
    bed_at_cells = cell_centered_from_nodal(raw, bed_at_nodes)
    logger.info(
        f"Bathymetry: z_b in [{bed_at_cells.min():.3f}, "
        f"{bed_at_cells.max():.3f}] m"
    )
    plot_field(
        raw, bed_at_cells,
        cbar_label=r"$z_b$ (m)",
        title="Bathymetry",
        out_path=out_dir / "savannah_bathymetry.png",
        cmap=args.bathy_cmap,
        dpi=args.dpi,
    )

    # ---- 2. Manning's n zones ----
    plot_manning_zones(raw, out_dir / "savannah_manning_zones.png", dpi=args.dpi)

    # ---- 3-4. SRH-2D h and |u| at the chosen snapshot ----
    if H5_PATH.exists():
        times, depth, vel = load_h5_fields(H5_PATH)
        logger.info(f"H5: {len(times)} snapshots, "
                    f"t = [{times[0]:.3f}, {times[-1]:.3f}] s")
        ti = args.time_index if args.time_index >= 0 \
            else len(times) + args.time_index
        if not (0 <= ti < len(times)):
            raise IndexError(
                f"--time-index {args.time_index} out of range for "
                f"{len(times)} snapshots"
            )
        t_val = float(times[ti])
        h_t = depth[ti]
        vel_t = vel[ti]
        vmag_srh = np.sqrt(vel_t[:, 0] ** 2 + vel_t[:, 1] ** 2)
        logger.info(f"SRH-2D snapshot t = {t_val:.3f} s")
        logger.info(f"  h:    [{h_t.min():.4f}, {h_t.max():.4f}] m")
        logger.info(f"  |u|:  [{vmag_srh.min():.4f}, {vmag_srh.max():.4f}] m/s")

        plot_field(
            raw, h_t,
            cbar_label=r"$h$ (m)",
            title=rf"SRH-2D: water depth $h$ at $t = {t_val:.2f}$ s",
            out_path=out_dir / "savannah_srh2d_h.png",
            cmap=args.depth_cmap,
            dpi=args.dpi,
        )
        plot_field(
            raw, vmag_srh,
            cbar_label=r"$|\mathbf{u}|$ (m\,s$^{-1}$)",
            title=(rf"SRH-2D: velocity magnitude $|\mathbf{{u}}|$ "
                   rf"at $t = {t_val:.2f}$ s"),
            out_path=out_dir / "savannah_srh2d_velocity.png",
            cmap=args.vel_cmap,
            dpi=args.dpi,
        )
    else:
        logger.warning(f"SRH-2D h5 file not found, skipping: {H5_PATH}")

    # ---- 5-6. FVM-only h and |u| from the chosen VTK ----
    if args.vtk_path:
        vtk_path = Path(args.vtk_path)
        if not vtk_path.exists():
            logger.warning(f"FVM VTK not found, skipping: {vtk_path}")
        else:
            sim_time, fvm_fields = read_vtk_cell_scalars(vtk_path)
            required = {"Water_Depth_m", "Velocity_X_m_s", "Velocity_Y_m_s"}
            missing = required - set(fvm_fields)
            if missing:
                logger.warning(
                    f"VTK {vtk_path.name} missing fields {missing} "
                    "— skipping FVM plots"
                )
            else:
                if len(fvm_fields["Water_Depth_m"]) != len(raw.elem_nodes):
                    logger.warning(
                        f"VTK cell count {len(fvm_fields['Water_Depth_m'])} "
                        f"!= mesh cell count {len(raw.elem_nodes)} — "
                        "plot may be misaligned."
                    )
                h_fvm = fvm_fields["Water_Depth_m"]
                vx_fvm = fvm_fields["Velocity_X_m_s"]
                vy_fvm = fvm_fields["Velocity_Y_m_s"]
                vmag_fvm = np.sqrt(vx_fvm ** 2 + vy_fvm ** 2)

                logger.info(f"FVM-only VTK: t = {sim_time:.3f} s "
                            f"({vtk_path.name})")
                logger.info(f"  h:    [{h_fvm.min():.4f}, {h_fvm.max():.4f}] m")
                logger.info(f"  |u|:  [{vmag_fvm.min():.4f}, "
                            f"{vmag_fvm.max():.4f}] m/s")

                plot_field(
                    raw, h_fvm,
                    cbar_label=r"$h$ (m)",
                    title=(rf"FVM teacher: water depth $h$ "
                           rf"at $t = {sim_time:.2f}$ s"),
                    out_path=out_dir / "savannah_fvm_h.png",
                    cmap=args.depth_cmap,
                    dpi=args.dpi,
                )
                plot_field(
                    raw, vmag_fvm,
                    cbar_label=r"$|\mathbf{u}|$ (m\,s$^{-1}$)",
                    title=(rf"FVM teacher: velocity magnitude $|\mathbf{{u}}|$ "
                           rf"at $t = {sim_time:.2f}$ s"),
                    out_path=out_dir / "savannah_fvm_velocity.png",
                    cmap=args.vel_cmap,
                    dpi=args.dpi,
                )


if __name__ == "__main__":
    main()
