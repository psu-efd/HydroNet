"""Case 1 profiles at t = 1 s from the A100 run behind the dam-break table (DB_1).

Reads the saved predictions and checks them against the training log before plotting.
Writes figures/fig_dambreak_profiles.{pdf,png} in the manuscript.
"""
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CASE = Path("/storage/work/axt5780/Projects/Hydro/HydroNet_a100_timing/examples/FVM_PINN/dam_break_1d")
OUT = Path("/storage/work/axt5780/Projects/Hydro/fvm_pinn_swe_manuscript/figures/fig_dambreak_profiles")

d = np.load(CASE / "plots" / "DB_1" / "predictions.npz")
o = np.argsort(d["x"])
x, h, u, he, ue = d["x"][o], d["h_pred"][o], d["u_pred"][o], d["h_exact"][o], d["u_exact"][o]
l2h, l2u, maxh = np.sqrt(np.mean((h - he) ** 2)), np.sqrt(np.mean((u - ue) ** 2)), np.abs(h - he).max()
log = (CASE / "runs" / "DB_1" / "stdout.log").read_text(errors="ignore")
logh, logmax = map(float, re.findall(r"h: L2=(\S+), max=(\S+)", log)[-1])
logu = float(re.findall(r"u: L2=(\S+), max=", log)[-1])
if max(abs(l2h / logh - 1), abs(l2u / logu - 1), abs(maxh / logmax - 1)) > 1e-3:
    raise SystemExit(f"predictions {l2h}, {l2u}, {maxh} differ from log {logh}, {logu}, {logmax}")
print(f"L2(h)={l2h:.4e} L2(u)={l2u:.4e} Linf(h)={maxh:.4e}")

plt.rc("text", usetex=True)
plt.rc("font", family="serif")
fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
for ax, pred, exact, yl, lab in ((axes[0], h, he, "$h$ (m)", "(a)"), (axes[1], u, ue, "$u$ (m/s)", "(b)")):
    ax.plot(x, pred, color="#0072B2", lw=2, label="FVM-PINN")
    ax.plot(x, exact, color="#D55E00", ls="--", lw=2, label="Exact solution")
    ax.set_xlabel("$x$ (m)", fontsize=16)
    ax.set_ylabel(yl, fontsize=16)
    ax.tick_params(labelsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=14)
    ax.set_title(lab, fontsize=16, loc="left")
fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
fig.savefig(OUT.with_suffix(".png"), dpi=200, bbox_inches="tight")
