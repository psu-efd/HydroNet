"""
Plot peak GPU memory vs. wall-clock time for SR-A through SR-G.

Reads each run's ``checkpoints/SR_X/memory_history.json`` (written by
``FVM_PINNTrainer`` during training) and produces a single comparison
figure showing how peak memory evolves over training time for each
strategy.

Usage
-----
    cd examples/FVM_PINN/savannah_river
    python plot_SR_memory_history.py
    python plot_SR_memory_history.py --out plots/memory_comparison.png
    python plot_SR_memory_history.py --no-tex
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent

plt.rc('text', usetex=True)  #allow the use of Latex for math expressions and equations
plt.rc('font', family='serif') #specify the default font family to be "serif"

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


# Match the colors used by fig_savannah_l2_by_time so the two figures
# are visually consistent.
SR_RUNS: List[Tuple[str, str, str, str]] = [
    ("SR_A", "physics only",                 "#333333", "o"),
    ("SR_B", "standard",                     "#D55E00", "s"),
    ("SR_C", "window(5)",                    "#0072B2", "^"),
    ("SR_D", "window(10)",                   "#009E73", "D"),
    ("SR_E", "FVM teacher",                  "#E69F00", "v"),
    ("SR_F", r"sparse $N_d=200$",            "#CC79A7", "P"),
    ("SR_G", r"anchor, $\lambda_{\mathrm{fvm\mbox{-}pinn}}=0$",  "#56B4E9", "X"),
]


def load_history(run_id: str, ckpt_dir: Path = SCRIPT_DIR / "checkpoints") -> Optional[Dict]:
    path = ckpt_dir / run_id / "memory_history.json"
    if not path.exists():
        logger.warning(f"[{run_id}] memory_history.json missing -> skip")
        return None
    with path.open() as f:
        data = json.load(f)
    if not data.get("records"):
        logger.warning(f"[{run_id}] empty records (CPU run?) -> skip")
        return None
    return data


def setup_publication_style(use_tex: bool) -> None:
    plt.rcdefaults()
    if use_tex:
        try:
            plt.rc("text", usetex=True)
        except Exception:
            plt.rc("text", usetex=False)
    else:
        plt.rc("text", usetex=False)
    plt.rc("font", family="serif", size=13)
    plt.rc("axes", labelsize=14, titlesize=14)
    plt.rc("xtick", labelsize=12)
    plt.rc("ytick", labelsize=12)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot peak GPU memory vs wall-clock time for SR-A...SR-G.",
    )
    parser.add_argument("--out", default=str(SCRIPT_DIR / "plots" / "SR_memory_history.png"))
    parser.add_argument("--no-tex", action="store_true")
    parser.add_argument("--checkpoints", type=Path, default=SCRIPT_DIR / "checkpoints",
                        help="Folder holding <run>/memory_history.json (the manuscript "
                             "figure uses the A100 timing tree).")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    setup_publication_style(use_tex=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax_p, ax_a) = plt.subplots(2, 1, figsize=(8, 9.5), constrained_layout=True)

    n_loaded = 0
    for run_id, label, color, marker in SR_RUNS:
        data = load_history(run_id, args.checkpoints)
        if data is None:
            continue
        records = data["records"]
        wall = np.array([r["wall_s"] for r in records])
        # The JSON stores peak-since-last-reset (the trainer resets peak at
        # the start of each phase / window). For a clean "max since start"
        # curve, accumulate the running maximum.
        peak_phase = np.array([r["peak_mib"] for r in records])
        peak = np.maximum.accumulate(peak_phase)
        alloc = np.array([r["alloc_mib"] for r in records])

        ax_p.plot(wall, peak, marker=marker, markersize=4.5, linewidth=1.4,
                  color=color, label=f"{run_id.replace('_', '-')} ({label})", alpha=0.9)
        ax_a.plot(wall, alloc, marker=marker, markersize=4.5, linewidth=1.4,
                  color=color, alpha=0.9)
        n_loaded += 1

        # Log a quick summary
        logger.info(
            f"[{run_id}] n={len(records)}  "
            f"peak max={peak.max():.0f} MiB  "
            f"alloc final={alloc[-1]:.0f} MiB  "
            f"duration={wall.max():.0f}s"
        )

    if n_loaded == 0:
        logger.error("No memory_history.json files found — nothing to plot.")
        return

    for ax, ylabel, title in [
        (ax_p, "Peak GPU memory (MiB)", "Peak (running max since start)"),
        (ax_a, "Live GPU allocation (MiB)", "Currently allocated"),
    ]:
        ax.set_xlabel("Wall-clock time (s)", fontsize=18)
        ax.set_ylabel(ylabel, fontsize=18)
        ax.set_title(title, fontsize=18)
        #set tick label size
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.grid(True, alpha=0.3)

        #add (a) and (b) labels to the subplots
        if ax == ax_p:
            ax.text(-0.15, 1.05, "(a)", transform=ax.transAxes, fontsize=18)
        else:
            ax.text(-0.15, 1.05, "(b)", transform=ax.transAxes, fontsize=18)

    # One shared legend below the panels, so it covers no data
    handles, labels = ax_p.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=12, frameon=False)

    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    pdf_path = out_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {out_path}")
    logger.info(f"Saved: {pdf_path}")


if __name__ == "__main__":
    main()
