"""Case 5 snapshot figure with larger text, recomposed from the original vector figure.

fig_bicu_unsteady.py draws this figure from the trained networks and the unsteady SRH-2D
result. This script takes the PDF written by that script, renders the nine contour panels
and the two colour bars from it unchanged, and lays them out again with larger labels,
so the figure can be restyled without the checkpoints. The colour bars go in the empty
cell at the top right. Needs pymupdf.

Usage (from this folder):
    python fig_bicu_snapshot_replot.py <original fig_bicu_snapshot_t400.pdf> <output without extension>
Writes <output>.pdf and <output>.png.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pymupdf

FS = 20                                   # base font size at a figure width of 12.5 in
ZOOM = 600 / 72                           # render the source at 600 dpi

# Panel frames in the source PDF (points), row by row: left, right
X_LEFT, X_RIGHT, W, H = 45.62, 497.12, 312.01, 104.0
Y_ROWS = (24.65, 169.40, 314.16, 458.91, 603.67)
# Colour bars: outline box, and the positions of two labelled ticks
CB_U = dict(box=(13.57, 783.62, 389.70, 794.36), ticks=((13.57, 0.0), (361.83, 0.5)))
CB_D = dict(box=(465.10, 775.20, 841.20, 785.95), ticks=((497.74, -0.4), (808.51, 0.4)))
# Titles; the L2 values are those printed in the source figure
LEFT = ("(a) SRH-2D", "(b) BICU-HW (dense data only)", "(d) BICU-EW (physics + dense)",
        "(f) BICU-C (physics + 10 gauges)", "(h) BICU-A (physics only)")
RIGHT = (None, "(c) Difference, $L_2 = 0.046$~m/s", "(e) Difference, $L_2 = 0.314$~m/s",
         "(g) Difference, $L_2 = 0.102$~m/s", "(i) Difference, $L_2 = 0.312$~m/s")

page = pymupdf.open(sys.argv[1])[0]


def crop(x0, y0, x1, y1):
    pix = page.get_pixmap(matrix=pymupdf.Matrix(ZOOM, ZOOM), clip=pymupdf.Rect(x0, y0, x1, y1), alpha=False)
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)


plt.rc("text", usetex=True)
plt.rc("font", family="serif", size=FS)
plt.rc("axes", labelsize=FS, titlesize=FS)
plt.rc("xtick", labelsize=FS - 3)
plt.rc("ytick", labelsize=FS - 3)

fig = plt.figure(figsize=(12.5, 12.4), constrained_layout=True)
gs = fig.add_gridspec(5, 2)
for row, y in enumerate(Y_ROWS):
    for col, (x, titles) in enumerate(((X_LEFT, LEFT), (X_RIGHT, RIGHT))):
        if titles[row] is None:
            continue
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(crop(x, y, x + W, y + H), extent=(0, 15, 0, 5), aspect="equal", interpolation="lanczos")
        ax.set_title(titles[row], loc="left")
        ax.set_xticks(np.arange(0, 15.1, 3))
        ax.set_yticks((0, 2, 4))
        ax.tick_params(labelbottom=(row == 4), labelleft=(col == 0))
        if row == 4:
            ax.set_xlabel(r"$x$ (m)")
        if col == 0:
            ax.set_ylabel(r"$y$ (m)")

# colour bars in the empty top-right cell
sub = gs[0, 1].subgridspec(5, 1, height_ratios=(0.5, 1, 2.6, 1, 2.2))
for k, (cb, label, step) in enumerate(((CB_U, r"$|\mathbf{u}|$ (m/s)", 0.1),
                                       (CB_D, r"$|\mathbf{u}| - |\mathbf{u}|_{\mathrm{SRH}}$ (m/s)", 0.2))):
    (xa, va), (xb, vb) = cb["ticks"]
    value = lambda x: va + (x - xa) / (xb - xa) * (vb - va)
    x0, y0, x1, y1 = cb["box"]
    pad = 0.6                                       # keep the outline of the colour bar
    ax = fig.add_subplot(sub[1 + 2 * k])
    ax.imshow(crop(x0 - pad, y0 - pad, x1 + pad, y1 + pad), extent=(value(x0 - pad), value(x1 + pad), 0, 1),
              aspect="auto", interpolation="lanczos")
    ax.set_yticks([])
    ax.set_xticks(np.arange(va, vb + 1e-9, step))
    ax.set_xlabel(label, labelpad=2)
    for spine in ax.spines.values():
        spine.set_visible(False)

out = Path(sys.argv[2])
fig.savefig(out.with_suffix(".pdf"), dpi=300)
fig.savefig(out.with_suffix(".png"), dpi=200)
