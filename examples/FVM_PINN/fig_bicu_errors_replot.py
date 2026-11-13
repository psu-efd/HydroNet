"""Case 5 error figure with larger text, drawn from stored values.

fig_bicu_unsteady.py makes this figure from the trained networks and the unsteady SRH-2D
result. This script redraws it from fig_bicu_errors_data.json, which holds the plotted
values read from the vector file written by that script, so the figure can be restyled
without the checkpoints.

Usage (from this folder):
    python fig_bicu_errors_replot.py <output path without extension>
Writes <output>.pdf and <output>.png.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FS = 21                                   # base font size at a figure width of 13 in
DENSE = np.arange(360, 601, 30)
RUNS = (  # key, label, colour, linestyle
    ("BICU_A", "BICU-A (physics only)", "#7f7f7f", "--"),
    ("BICU_C", "BICU-C (physics + 10 gauges)", "#9467bd", "-."),
    ("BICU_EW", "BICU-EW (physics + dense)", "#d62728", "-"),
    ("BICU_HW", "BICU-HW (dense data only)", "#2ca02c", "-"),
)
SWEEP_COLOURS = ("#d62728", "#ff7f0e", "#bcbd22", "#2ca02c")

d = json.loads((Path(__file__).resolve().parent / "fig_bicu_errors_data.json").read_text())
a, b = d["panel_a"], d["panel_b"]

plt.rc("text", usetex=True)
plt.rc("font", family="serif", size=FS)
plt.rc("axes", labelsize=FS, titlesize=FS)
plt.rc("xtick", labelsize=FS - 2)
plt.rc("ytick", labelsize=FS - 2)

fig = plt.figure(figsize=(13, 7.0), constrained_layout=True)
gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1])
ax0, ax1 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

# (a) error through time
for t in DENSE:
    ax0.axvline(t, color="0.85", lw=1.0, zorder=0)
ax0.semilogy(a["interp"]["t"], a["interp"]["L2_u"], "ks", mfc="white", ms=8, ls="none",
             label="Snapshot interpolation")
ax0.semilogy(a["fvm"]["t"], a["fvm"]["L2_u"], "k:", lw=2.4, label="FVM solver (no data)")
for key, label, colour, ls in RUNS:
    ax0.semilogy(a[key]["t"], a[key]["L2_u"], color=colour, ls=ls, lw=2.0, marker="o", ms=4.5, label=label)
ax0.set_xlim(360, 600)
ax0.set_ylim(1e-3, 0.8)
ax0.set_xticks(np.arange(360, 601, 60))
ax0.set_xlabel(r"$t$ (s)")
ax0.set_ylabel(r"$L_2(|\mathbf{u}|)$ (m/s)")
ax0.grid(True, which="both", alpha=0.25)
ax0.set_title("(a)", loc="left")
handles, labels = ax0.get_legend_handles_labels()
fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=FS - 4, frameon=False,
           columnspacing=1.2, handletextpad=0.5)

# (b) held-out error against the residual for the lambda sweep
ax1.plot(b["residual"], b["heldout_L2_u"], "-", color="0.5", lw=1.4, zorder=1)
offsets = {1.0: (12, -6), 0.1: (12, 2), 0.01: (-12, 12), 0.0: (4, 16)}
for lam, x, y, colour in zip(b["lambda"], b["residual"], b["heldout_L2_u"], SWEEP_COLOURS):
    ax1.plot(x, y, "o", ms=13, color=colour, mec="k", zorder=3)
    dx, dy = offsets[lam]
    ax1.annotate(rf"$\lambda = {lam:g}$", (x, y), xytext=(dx, dy), textcoords="offset points",
                 fontsize=FS - 2, ha="left" if dx > 0 else "right")
ax1.axhline(b["interp_heldout_L2_u"], color="k", ls="--", lw=1.4)
ax1.text(8.2e-4, b["interp_heldout_L2_u"] + 0.004, "snapshot interpolation", fontsize=FS - 4, va="bottom")
ax1.set_xscale("log")
ax1.set_xlim(8.0e-4, 3.3e-3)
ax1.set_ylim(0, 0.195)
ax1.minorticks_off()
ax1.set_xticks([1e-3, 2e-3])
ax1.set_xticklabels([r"$1{\times}10^{-3}$", r"$2{\times}10^{-3}$"])
ax1.set_yticks(np.arange(0, 0.151, 0.05))
ax1.set_xlabel(r"Mean FVM-PINN residual")
ax1.set_ylabel(r"Held-out $L_2(|\mathbf{u}|)$ (m/s)")
ax1.grid(True, which="both", alpha=0.25)
ax1.set_title("(b)", loc="left")

out = Path(sys.argv[1])
fig.savefig(out.with_suffix(".pdf"))
fig.savefig(out.with_suffix(".png"), dpi=200)
