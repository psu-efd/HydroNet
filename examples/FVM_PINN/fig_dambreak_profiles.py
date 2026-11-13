"""Case 1 profiles at t = 1 s: FVM-PINN (DB_1) and strong-form PINN (DB_1_SF) against the exact solution.

Reads the predictions saved by dam_break_1d_FVM_PINN.py and checks them against the training logs
before plotting.

Usage (from this folder, after both runs):
    python fig_dambreak_profiles.py <output path without extension>
Writes <output>.pdf and <output>.png.
"""
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CASE = Path(__file__).resolve().parent / "dam_break_1d"
OUT = Path(sys.argv[1])
RUNS = (("DB_1", "FVM-PINN", "#0072B2", "-"), ("DB_1_SF", "Strong-form PINN", "#009E73", "-."))


def load(run_id):
    """Profiles of one run sorted by x, checked against the errors in its training log."""
    d = np.load(CASE / "plots" / run_id / "predictions.npz")
    o = np.argsort(d["x"])
    x, h, u, he, ue = d["x"][o], d["h_pred"][o], d["u_pred"][o], d["h_exact"][o], d["u_exact"][o]
    l2h, l2u, maxh = np.sqrt(np.mean((h - he) ** 2)), np.sqrt(np.mean((u - ue) ** 2)), np.abs(h - he).max()
    log = (CASE / "runs" / run_id / "stdout.log").read_text(errors="ignore")
    logh, logmax = map(float, re.findall(r"h: L2=(\S+), max=(\S+)", log)[-1])
    logu = float(re.findall(r"u: L2=(\S+), max=", log)[-1])
    if max(abs(l2h / logh - 1), abs(l2u / logu - 1), abs(maxh / logmax - 1)) > 1e-3:
        raise SystemExit(f"{run_id}: predictions {l2h}, {l2u}, {maxh} differ from log {logh}, {logu}, {logmax}")
    print(f"{run_id}: L2(h)={l2h:.4e} L2(u)={l2u:.4e} Linf(h)={maxh:.4e}")
    return x, h, u, he, ue


data = {run_id: load(run_id) for run_id, *_ in RUNS}
x, _, _, he, ue = data["DB_1"]

plt.rc("text", usetex=True)
plt.rc("font", family="serif")
fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
for k, (ax, exact, yl, lab) in enumerate(((axes[0], he, "$h$ (m)", "(a)"), (axes[1], ue, "$u$ (m/s)", "(b)"))):
    for run_id, label, color, ls in RUNS:
        ax.plot(data[run_id][0], data[run_id][1 + k], color=color, ls=ls, lw=2, label=label)
    ax.plot(x, exact, color="#D55E00", ls="--", lw=2, label="Exact solution")
    ax.set_xlabel("$x$ (m)", fontsize=16)
    ax.set_ylabel(yl, fontsize=16)
    ax.tick_params(labelsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=13, loc="upper right" if k == 0 else "upper left")
    ax.set_title(lab, fontsize=16, loc="left")
fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
fig.savefig(OUT.with_suffix(".png"), dpi=200, bbox_inches="tight")
