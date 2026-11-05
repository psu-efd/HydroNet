"""Dam-break time-sampling sweep (reviewer R1-M5): error against n_t for redrawn
and fixed time levels, and the velocity profile at t_end.

Reads dam_break_1d/plots/DB_{NT,FX}<n>_S<s>/ and writes
fig_db_time_sampling.{pdf,png} to the manuscript figures folder.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
PLOTS = HERE / "dam_break_1d" / "plots"
OUT = Path("/storage/work/axt5780/Projects/Hydro/fvm_pinn_swe_manuscript/figures/fig_db_time_sampling")

REDRAWN, FIXED = "#1f77b4", "#d62728"
MODES = {"NT": ("redrawn levels", REDRAWN, "o", "-", (1, 2, 5, 10, 20)),
         "FX": ("fixed levels", FIXED, "s", "--", (2, 5, 20))}
SEEDS = (1, 2, 3)


def load(mode, n):
    out = []
    for s in SEEDS:
        d = np.load(PLOTS / f"DB_{mode}{n}_S{s}" / "spacetime_errors.npz")
        out.append((float(d["spacetime_L2_u"]), float(d["spacetime_L2_h"])))
    return np.array(out)


def main():
    plt.rc("font", family="serif", size=13)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)

    for col, (ax, ylabel) in enumerate(zip(axes[:2], (r"Space-time $L_2(u)$ (m/s)",
                                                       r"Space-time $L_2(h)$ (m)"))):
        for mode, (label, color, marker, ls, ns) in MODES.items():
            vals = np.array([load(mode, n)[:, col] for n in ns])
            ax.plot(ns, vals.mean(1), ls=ls, color=color, lw=2, zorder=2, marker=marker, ms=0)
            ax.plot([], [], ls=ls, color=color, lw=2, marker=marker, ms=8, mec="white", label=label)
            for n, v in zip(ns, vals):
                ax.plot([n] * len(v), v, marker, ms=8, mfc=color, mec="white",
                        mew=1.0, color=color, zorder=3)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks([1, 2, 5, 10, 20])
        ax.set_xticklabels(["1", "2", "5", "10", "20"])
        ax.minorticks_off()
        ax.set_xlabel(r"Time levels per step, $n_t$")
        ax.set_ylabel(ylabel)
        ax.grid(True, color="0.9", lw=0.8)
        ax.set_axisbelow(True)
    yticks = ([0.2, 0.5, 1.0], [0.05, 0.1, 0.2, 0.5, 1.0])
    for ax, t in zip(axes[:2], yticks):
        ax.set_yticks(t)
        ax.set_yticklabels([f"{x:g}" for x in t])
    axes[0].legend(frameon=False, loc="center left", bbox_to_anchor=(0.0, 0.42))

    ax = axes[2]
    d = np.load(PLOTS / "DB_NT5_S1" / "predictions.npz")
    ax.plot(d["x"], d["u_exact"], color="k", lw=2, label="exact (Stoker)", zorder=1)
    ax.plot(d["x"], d["u_pred"], color=REDRAWN, lw=2, ls="-", label=r"redrawn, $n_t = 5$", zorder=3)
    d = np.load(PLOTS / "DB_FX5_S1" / "predictions.npz")
    ax.plot(d["x"], d["u_pred"], color=FIXED, lw=2, ls="--", label=r"fixed, $n_t = 5$", zorder=2)
    ax.set_xlabel(r"$x$ (m)")
    ax.set_ylabel(r"$u$ at $t = 1$ s (m/s)")
    ax.grid(True, color="0.9", lw=0.8)
    ax.set_axisbelow(True)
    ax.set_ylim(top=3.4)
    ax.legend(frameon=False, loc="upper left")

    for ax, lab in zip(axes, "abc"):
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(-0.02, 1.03, f"({lab})", transform=ax.transAxes, fontweight="bold", ha="right")

    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
