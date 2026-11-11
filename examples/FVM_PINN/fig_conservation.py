"""Figure and table of the mass-conservation errors, from conservation_diagnostics.json.

Writes figures/fig_conservation.{pdf,png} in the manuscript and prints the LaTeX table rows.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = Path("/storage/work/axt5780/Projects/Hydro/fvm_pinn_swe_manuscript/figures/fig_conservation")
D = json.loads((HERE / "conservation_diagnostics.json").read_text())

# Colour follows the kind of training, across the three cases
PHYS, SPARSE, DENSE, WIND, TEACH, TUNE, DATA = "#333333", "#D55E00", "#0072B2", "#009E73", "#E69F00", "#CC79A7", "#56B4E9"
PANELS = [
    ("Case 3, block in channel", "BIC", [("BIC_A", "BIC-A, physics only", PHYS, "o"), ("BIC_B", "BIC-B, sparse at $t=T$", SPARSE, "s"),
                                         ("BIC_H", "BIC-H, dense, no physics", DATA, "P"), ("BIC_D", "BIC-D, teacher", TEACH, "v")]),
    ("Case 4, Savannah River", "SR", [("SR_A", "SR-A, physics only", PHYS, "o"), ("SR_B", "SR-B, dense", DENSE, "s"),
                                      ("SR_G", "SR-G, dense, no physics", DATA, "P"), ("SR_D", "SR-D, 10 windows", WIND, "D"),
                                      ("SR_E", "SR-E, teacher", TEACH, "v")]),
    ("Case 5, inflow pulse", "BICU", [("BICU_A", "BICU-A, physics only", PHYS, "o"), ("BICU_EW", "BICU-EW, windows", WIND, "D"),
                                      ("BICU_HW", "BICU-HW, windows, no physics", DATA, "P"), ("BICU_T", "BICU-T, teacher", TEACH, "v"),
                                      ("BICU_PT", "BICU-PT, from T, physics only", TUNE, "X")]),
]


def figure():
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")
    fig, axes = plt.subplots(2, 3, figsize=(12, 9.2), sharex="col", sharey="row", constrained_layout=True)
    for j, (title, case, runs) in enumerate(PANELS):
        for rid, label, color, marker in runs:
            rows = D[rid]["rows"]
            t = np.array([r["t"] for r in rows])
            kw = dict(color=color, lw=1.5, marker=marker, ms=6, markevery=14, label=label)
            axes[0, j].plot(t, np.clip([r["eps_V"] for r in rows], 1e-4, None), **kw)
            axes[1, j].plot(t, [r["eps_rms"] for r in rows], **kw)
        ref = D[f"{case}_SRH2D"]["rows"]
        kw = dict(ls="none", marker="+", ms=8, mew=1.2, color="#777777", label="SRH-2D output")
        rt = [r["t"] for r in ref]
        axes[0, j].plot(rt, np.clip([r["eps_V"] for r in ref], 1e-4, None), **kw)
        axes[1, j].plot(rt, [r["eps_rms"] for r in ref], **kw)
        axes[0, j].set_title(f"({'abc'[j]}) {title}", fontsize=15)
        axes[1, j].set_title(f"({'def'[j]})", fontsize=15)
        axes[1, j].set_xlabel("$t$ (s)", fontsize=15)
        axes[1, j].legend(fontsize=10.5, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=1, frameon=False)
    axes[0, 0].set_ylabel(r"$\varepsilon_V$", fontsize=16)
    axes[1, 0].set_ylabel(r"$\varepsilon_{\mathrm{rms}}$", fontsize=16)
    for ax in axes.ravel():
        ax.set_yscale("log")
        ax.grid(True, which="major", alpha=0.3)
        ax.tick_params(labelsize=12)
    axes[0, 0].set_ylim(1e-4, 3)
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), dpi=200, bbox_inches="tight")


def sig(x, n=2):
    return f"{x:.{n}g}"


def table():
    groups = [["BIC_A", "BIC_B", "BIC_C", "BIC_F", "BIC_G", "BIC_E", "BIC_H", "BIC_D", "BIC_SRH2D", "BIC_FVM"],
              ["SR_A", "SR_A_SF", "SR_B", "SR_B_SF", "SR_G", "SR_C", "SR_D", "SR_E", "SR_F", "SR_F_SF", "SR_SRH2D", "SR_FVM"],
              ["BICU_A", "BICU_C", "BICU_EW", "BICU_EWL1", "BICU_EWL2", "BICU_HW", "BICU_T", "BICU_CT", "BICU_GT", "BICU_PT", "BICU_SRH2D", "BICU_FVM"]]
    for g in groups:
        for rid in g:
            s, rows = D[rid]["summary"], D[rid]["rows"]
            name = rid.replace("_SRH2D", " SRH-2D").replace("_FVM", " FVM solver").replace("_SF", "-SF").replace("_", "-")
            e = "--" if rid.endswith(("SRH2D", "FVM")) else f"{100 * s['E_V_final']:+.1f}"
            print(f"{name} & {sig(s['eps_inf_median'])} & {sig(s['eps_rms_median'])} & {100 * s['eps_V_median']:.2g} & "
                  f"{e} & {s['q_out_num_ratio_final']:.2f} \\\\")
        print("\\midrule")


if __name__ == "__main__":
    figure()
    table()
