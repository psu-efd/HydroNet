"""Case 4 overview figure (six panels) for the manuscript, composed in one script.

(a) Manning's n zones, (b) bathymetry, (c, d) SRH-2D depth and velocity magnitude at the
last snapshot, (e, f) the same from the FVM teacher. Replaces the earlier composition of
separate PNGs from plot_domain_roughness_zones_srh_fvm.py, so that the text is large
enough at the printed size. Depth and velocity panels share colour limits.

Needs an SRH-2D XMDFC file (default Savana_XMDFC.h5 in this folder) and
../vtk_fvm_only/fvm_t0010.vtk (written by ../savannah_river_fvm_only.py).

Usage (from this folder):
    python fig_case_overview.py <output file> [SRH-2D h5 file]
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.colors import ListedColormap

import plot_domain_roughness_zones_srh_fvm as pdz

FS = 20                                   # base font size at a figure width of 14 in
HERE = Path(__file__).resolve().parent
H5 = Path(sys.argv[2]) if len(sys.argv) > 2 else pdz.H5_PATH

plt.rc("text", usetex=True)
plt.rc("font", family="serif", size=FS)
plt.rc("axes", labelsize=FS, titlesize=FS)
plt.rc("xtick", labelsize=FS - 2)
plt.rc("ytick", labelsize=FS - 2)

_, raw = pdz.load_case()
polys = pdz.cell_polygons(raw)
x0, x1, y0, y1 = pdz.domain_extent(raw)
z_b = pdz.cell_centered_from_nodal(raw, raw.node_coords[:, 2])
times, depth, vel = pdz.load_h5_fields(H5)
h_srh, u_srh = depth[-1], np.hypot(vel[-1][:, 0], vel[-1][:, 1])
_, fvm = pdz.read_vtk_cell_scalars(HERE.parent / "vtk_fvm_only" / "fvm_t0010.vtk")
h_fvm, u_fvm = fvm["Water_Depth_m"], np.hypot(fvm["Velocity_X_m_s"], fvm["Velocity_Y_m_s"])
print(f"SRH-2D snapshot t = {times[-1]} (file units), cells = {len(h_srh)}")

fig, axes = plt.subplots(3, 2, figsize=(14, 11.4), constrained_layout=True)


def panel(ax, values, title, cmap, clim=None, label=None, **cbar_kw):
    pc = PolyCollection(polys, array=np.asarray(values, dtype=float), cmap=cmap,
                        edgecolors="face", linewidth=0, antialiased=False)
    if clim:
        pc.set_clim(*clim)
    ax.add_collection(pc)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$x$ (m)")
    ax.set_ylabel(r"$y$ (m)")
    ax.set_xticks(np.arange(0, x1, 200))
    ax.set_yticks(np.arange(0, y1, 100))
    ax.set_title(title, loc="left")
    cbar = fig.colorbar(pc, ax=ax, shrink=0.95, pad=0.02, **cbar_kw)
    if label:
        cbar.set_label(label)
    return cbar


# (a) Manning zones: only the zones that occur in the mesh
mat = np.asarray(raw.elem_material)
zones = sorted(set(mat.tolist()))
zone_index = {z: i for i, z in enumerate(zones)}
cbar = panel(axes[0, 0], [zone_index[m] for m in mat], r"(a) Manning's $n$ zones",
             ListedColormap(plt.cm.tab10.colors[1:len(zones) + 1]),
             clim=(-0.5, len(zones) - 0.5), ticks=np.arange(len(zones)))
cbar.ax.set_yticklabels([f"{raw.manning_n[z]:g}" for z in zones])
cbar.set_label(r"$n$ (s\,m$^{-1/3}$)")

# (b) bathymetry
panel(axes[0, 1], z_b, "(b) Bathymetry", plt.cm.terrain, label=r"$z_b$ (m)")

# (c)-(f) fields
lims = [(min(h_srh.min(), h_fvm.min()), max(h_srh.max(), h_fvm.max())),
        (min(u_srh.min(), u_fvm.min()), max(u_srh.max(), u_fvm.max()))]
labels = (r"$h$ (m)", r"$|\mathbf{u}|$ (m/s)")
for ax, values, title, q in (
    (axes[1, 0], h_srh, "(c) SRH-2D: water depth", 0),
    (axes[1, 1], u_srh, "(d) SRH-2D: velocity magnitude", 1),
    (axes[2, 0], h_fvm, "(e) FVM teacher: water depth", 0),
    (axes[2, 1], u_fvm, "(f) FVM teacher: velocity magnitude", 1),
):
    panel(ax, values, title, plt.cm.RdBu_r, clim=lims[q], label=labels[q])

fig.savefig(sys.argv[1], dpi=300)
print("h limits", lims[0], "u limits", lims[1], "zones", [(z, raw.manning_n[z]) for z in zones])
