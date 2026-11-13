"""Case 3 overview figure (six panels) for the manuscript, composed in one script.

(a) case setup, (b) mesh with the 50 sparse data points, (c, d) SRH-2D depth and velocity
magnitude at t = 360 s, (e, f) the same from the FVM teacher. Replaces the earlier
composition of separate PNGs from plot_case_overview.py, so that the text is large
enough at the printed size. Depth and velocity panels share colour limits.

Needs block_in_channel_XMDFC.h5 in this folder and ../vtk_fvm_only/fvm_t0010.vtk
(written by ../block_in_channel_fvm_only.py).

Usage (from this folder):
    python fig_case_overview.py <output file, e.g. block_in_channel_mesh_srh2d_fvm.png>
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.patches import Rectangle

import plot_case_overview as pco

FS = 20                                   # base font size at a figure width of 14 in
BLOCK = (4.5, 2.0, 2.0, 1.0)              # x0, y0, length, width of the block (m)
HERE = Path(__file__).resolve().parent

plt.rc("text", usetex=True)
plt.rc("font", family="serif", size=FS)
plt.rc("axes", labelsize=FS, titlesize=FS)
plt.rc("xtick", labelsize=FS - 2)
plt.rc("ytick", labelsize=FS - 2)

_, raw = pco.load_case()
polys = pco.cell_polygons(raw)
x0, x1, y0, y1 = pco.domain_extent(raw)
times, depth, vel = pco.load_h5_fields(pco.H5_PATH)
h_srh, u_srh = depth[-1], np.hypot(vel[-1][:, 0], vel[-1][:, 1])
_, fvm = pco.read_vtk_cell_scalars(HERE.parent / "vtk_fvm_only" / "fvm_t0010.vtk")
h_fvm, u_fvm = fvm["Water_Depth_m"], np.hypot(fvm["Velocity_X_m_s"], fvm["Velocity_Y_m_s"])
idx = pco.sparse_sample_indices(pco.H5_PATH, n_points=50, seed=42, h_dry=0.01, target_time=360.0)
pts = pco.cell_centers(raw)[idx]

fig, axes = plt.subplots(3, 2, figsize=(14, 8.6), constrained_layout=True)


def frame(ax, title):
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$x$ (m)")
    ax.set_ylabel(r"$y$ (m)")
    ax.set_xticks(np.arange(0, 15.1, 3))
    ax.set_yticks(np.arange(0, 5.1, 1))
    ax.set_title(title, loc="left")


# (a) case setup: the axes frame is the channel boundary
ax = axes[0, 0]
ax.add_patch(Rectangle(BLOCK[:2], BLOCK[2], BLOCK[3], fill=True, fc="0.85", ec="k", lw=1.5))
dim = dict(arrowstyle="<->", lw=1.2, shrinkA=0, shrinkB=0)
out = dict(annotation_clip=False)
ax.annotate("", (x0, -0.55), (x1, -0.55), arrowprops=dim, **out)
ax.text(7.5, -0.75, "15 m", ha="center", va="top")
ax.annotate("", (-0.45, y0), (-0.45, y1), arrowprops=dim, **out)
ax.text(-0.65, 2.5, "5 m", ha="right", va="center", rotation=90)
ax.annotate("", (BLOCK[0], 1.6), (BLOCK[0] + BLOCK[2], 1.6), arrowprops=dim)
ax.text(BLOCK[0] + BLOCK[2] / 2, 1.45, "2 m", ha="center", va="top")
ax.annotate("", (6.95, BLOCK[1]), (6.95, BLOCK[1] + BLOCK[3]), arrowprops=dim)
ax.text(7.2, 2.5, "1 m", ha="left", va="center")
ax.annotate("", (2.3, 2.5), (0.3, 2.5), arrowprops=dict(arrowstyle="->", lw=1.5))
ax.text(0.3, 2.75, "Inlet", ha="left", va="bottom")
ax.text(14.7, 2.5, "Outlet", ha="right", va="center")
ax.text(10.5, 4.8, "Wall", ha="center", va="top")
ax.text(10.5, 0.2, "Wall", ha="center", va="bottom")
ax.text(BLOCK[0] + BLOCK[2] / 2, 3.15, "Block", ha="center", va="bottom")
ax.set_xlim(x0, x1)
ax.set_ylim(y0, y1)
ax.set_aspect("equal", adjustable="box")
ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_linewidth(1.5)
ax.set_title("(a) Case setup", loc="left")

# (b) mesh and sparse points
ax = axes[0, 1]
ax.add_collection(PolyCollection(polys, facecolor="white", edgecolor="0.25", linewidth=0.35))
ax.scatter(pts[:, 0], pts[:, 1], s=30, c="#d62728", edgecolor="k", linewidth=0.4, zorder=5)
frame(ax, "(b) Mesh and 50 sparse data points")

# (c)-(f) fields
panels = (
    (axes[1, 0], h_srh, "(c) SRH-2D: water depth", 0),
    (axes[1, 1], u_srh, "(d) SRH-2D: velocity magnitude", 1),
    (axes[2, 0], h_fvm, "(e) FVM teacher: water depth", 0),
    (axes[2, 1], u_fvm, "(f) FVM teacher: velocity magnitude", 1),
)
lims = [(min(h_srh.min(), h_fvm.min()), max(h_srh.max(), h_fvm.max())),
        (min(u_srh.min(), u_fvm.min()), max(u_srh.max(), u_fvm.max()))]
labels = (r"$h$ (m)", r"$|\mathbf{u}|$ (m/s)")
for ax, values, title, q in panels:
    pc = PolyCollection(polys, array=np.asarray(values), cmap=plt.cm.RdBu_r, edgecolor="none")
    pc.set_clim(*lims[q])
    ax.add_collection(pc)
    frame(ax, title)
    cbar = fig.colorbar(pc, ax=ax, shrink=0.95, pad=0.02)
    cbar.set_label(labels[q])
    if q == 0:
        cbar.set_ticks(np.arange(0.398, 0.4041, 0.002))

fig.savefig(sys.argv[1], dpi=300)
print("h limits", lims[0], "u limits", lims[1])
