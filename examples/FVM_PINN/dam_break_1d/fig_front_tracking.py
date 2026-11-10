"""Shock position against time, Case 1 (R3-B1c). Uses front_tracking.npz and .json from front_tracking.py.

Line styles follow the manuscript: exact in solid black, FVM-PINN in dashed red.
Writes the figure to the path given as the first argument.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rc("text", usetex=True)
plt.rc("font", family="serif", size=11)
here = Path(__file__).resolve().parent
d = np.load(here / "front_tracking.npz")
ex = json.loads((here / "front_tracking.json").read_text())["exact"]
t = d["times"]
x_dam = 10.0
tt = np.linspace(0, t[-1], 50)

fig, ax = plt.subplots(figsize=(5.0, 3.4), constrained_layout=True)
ax.plot(tt, x_dam + ex["shock"] * tt, "k-", lw=2, label="Exact (Rankine--Hugoniot)")
ax.plot(t, d["FVM_shock_mid"], "r--o", lw=1.6, ms=4.5, label="FVM-PINN")
ax.plot(t, d["SF_shock_mid"], "b:s", lw=1.6, ms=4.5, label="Strong-form PINN")
ax.set_xlabel("$t$ (s)")
ax.set_ylabel("Shock position $x$ (m)")
ax.set_xlim(0, 1.02)
ax.grid(alpha=0.3)
ax.legend(fontsize=9.5, loc="upper left")
fig.savefig(sys.argv[1])
