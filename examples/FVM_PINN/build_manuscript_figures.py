"""
Build the publication-quality figures for papers/fvm_pinn_swe/main.tex.

For each Case 3 (block-in-channel) and Case 4 (Savannah River) run, the
trained checkpoint is re-loaded, the network is evaluated at every cell
centre at the relevant snapshot time, and per-figure matplotlib code
assembles the panels into PDFs that the LaTeX manuscript references.

Outputs:
    examples/FVM_PINN/block_in_channel/plots/
        fig_ablation_contours.pdf / .png
        fig_ablation_bar.pdf       / .png
        fig_ablation_profiles.pdf  / .png
    examples/FVM_PINN/savannah_river/plots/
        fig_savannah_contours.pdf   / .png
        fig_savannah_l2_by_time.pdf / .png

Copy the desired figures into ``papers/fvm_pinn_swe/figures/`` manually.

Usage
-----
    python examples/FVM_PINN/build_manuscript_figures.py            # everything
    python examples/FVM_PINN/build_manuscript_figures.py --skip-sr  # only Case 3
    python examples/FVM_PINN/build_manuscript_figures.py --skip-bic # only Case 4

Each subcommand caches per-run prediction arrays in
``plan/figures/_data/<case>_predictions.npz`` so subsequent runs are
fast (set ``--regenerate`` to force a fresh sweep).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from HydroNet import Config, FVM_SWE_PINN, FVM_PINNDataset  # noqa: E402

plt.rc('text', usetex=True)  #allow the use of Latex for math expressions and equations
plt.rc('font', family='serif') #specify the default font family to be "serif"

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

CACHE_DIR = REPO_ROOT / "plan" / "figures" / "_data"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Run registry
# ---------------------------------------------------------------------------

BIC_DIR = REPO_ROOT / "examples" / "FVM_PINN" / "block_in_channel"
SR_DIR = REPO_ROOT / "examples" / "FVM_PINN" / "savannah_river"

BIC_OUT_DIR = BIC_DIR / "plots"
SR_OUT_DIR = SR_DIR / "plots"
BIC_OUT_DIR.mkdir(parents=True, exist_ok=True)
SR_OUT_DIR.mkdir(parents=True, exist_ok=True)

BIC_RUNS = [
    ("BIC_A", "fvm_pinn_config_BIC_A.yaml", "ckpt_final.pt"),
    ("BIC_B", "fvm_pinn_config_BIC_B.yaml", "ckpt_final.pt"),
    ("BIC_C", "fvm_pinn_config_BIC_C.yaml", "ckpt_final.pt"),
    ("BIC_D", "fvm_pinn_config.yaml",       "teacher_final.pt"),
    ("BIC_E", "fvm_pinn_config_BIC_E.yaml", "ckpt_final.pt"),
    ("BIC_F", "fvm_pinn_config_BIC_F.yaml", "ckpt_final.pt"),
    ("BIC_G", "fvm_pinn_config_BIC_G.yaml", "ckpt_final.pt"),
    ("BIC_H", "fvm_pinn_config_BIC_H.yaml", "ckpt_final.pt"),
]
BIC_T_END = 360.0

SR_RUNS = [
    ("SR_A", "fvm_pinn_config_SR_A.yaml", "ckpt_final.pt"),
    ("SR_B", "fvm_pinn_config_SR_B.yaml", "ckpt_final.pt"),
    ("SR_C", "fvm_pinn_config_SR_C.yaml", "window_004/ckpt_final.pt"),
    ("SR_D", "fvm_pinn_config_SR_D.yaml", "window_009/ckpt_final.pt"),
    ("SR_E", "fvm_pinn_config_SR_E.yaml", "teacher_final.pt"),
    ("SR_F", "fvm_pinn_config_SR_F.yaml", "ckpt_final.pt"),
    ("SR_G", "fvm_pinn_config_SR_G.yaml", "ckpt_final.pt"),
]
SR_T_END = 3600.0

# Per-time L2 source per Savannah run. Each run persists a printable
# "PINN vs SRH-2D at each anchor time" table to ``runs/<id>/stdout.log``.
SR_PER_TIME_SOURCES: Dict[str, Tuple[str, str]] = {
    "SR_A": ("log",  "runs/SR_A/stdout.log"),
    "SR_B": ("log",  "runs/SR_B/stdout.log"),
    "SR_C": ("log",  "runs/SR_C/stdout.log"),
    "SR_D": ("log",  "runs/SR_D/stdout.log"),
    "SR_E": ("log",  "runs/SR_E/stdout.log"),
}


# ---------------------------------------------------------------------------
# Per-run prediction (load checkpoint + evaluate at all cell centres)
# ---------------------------------------------------------------------------

def _load_run(
    case_dir: Path,
    config_name: str,
    ckpt_relpath: str,
) -> Tuple[FVM_SWE_PINN, FVM_PINNDataset]:
    cwd0 = os.getcwd()
    os.chdir(case_dir)
    try:
        cfg = Config(config_name)
        cfg.set("device.type", "cpu")
        ds = FVM_PINNDataset(cfg)
        model = FVM_SWE_PINN(cfg)
        model.set_mesh_context(
            h_still_cells=ds.get_h_still(),
            cell_xy=ds.get_cell_xy(),
        )
        ckpt_dir = Path(str(
            cfg.get("training.logging.checkpoint_dir", "./checkpoints")
        ))
        ckpt_path = (ckpt_dir / ckpt_relpath).resolve()
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        net = model.get_internal_network()
        net.load_state_dict(ck["network_state"])
        net.set_normalisation(ck["x_mean"], ck["x_std"])
    finally:
        os.chdir(cwd0)
    return model, ds


def _predict_at_t(
    model: FVM_SWE_PINN,
    ds: FVM_PINNDataset,
    t_val: float,
) -> np.ndarray:
    """Evaluate model at every cell centre, returning [h, u, v] in physical units."""
    cell_xy = ds.get_cell_xy().detach().cpu().numpy()
    n = cell_xy.shape[0]
    xyt = torch.column_stack([
        torch.tensor(cell_xy[:, 0], dtype=torch.float64),
        torch.tensor(cell_xy[:, 1], dtype=torch.float64),
        torch.full((n,), t_val, dtype=torch.float64),
    ])
    model.eval()
    with torch.no_grad():
        Q_phys = model(xyt).cpu().numpy()
    return Q_phys


def _srh2d_at_t(h5_path: Path, t_val: float) -> Tuple[np.ndarray, ...]:
    with h5py.File(h5_path, "r") as f:
        times = f["Water_Depth_m/Times"][:].astype(np.float64)
        ti = int(np.argmin(np.abs(times - t_val)))
        h = f["Water_Depth_m/Values"][ti, :].astype(np.float64)
        vel = f["Velocity_m_p_s/Values"][ti, :, :].astype(np.float64)
    return h, vel[:, 0], vel[:, 1]


def collect_predictions(
    case_dir: Path,
    runs: List[Tuple[str, str, str]],
    t_end: float,
    cache_path: Path,
    regenerate: bool = False,
) -> Dict[str, Dict[str, np.ndarray]]:
    if cache_path.exists() and not regenerate:
        logger.info("Loading cached predictions from %s", cache_path)
        npz = np.load(cache_path, allow_pickle=True)
        out = {k: dict(npz[k].item()) for k in npz.files}
        return out

    out: Dict[str, Dict[str, np.ndarray]] = {}
    # Reference mesh (any run shares the same mesh)
    _, ds_ref = _load_run(case_dir, runs[0][1], runs[0][2])
    mesh = ds_ref.get_mesh()
    h5_path = case_dir / "data" / Path(
        Config(str(case_dir / runs[0][1])).get_required_config("data.srh2d_h5_file")
    ).name
    h_ref, u_ref, v_ref = _srh2d_at_t(h5_path, t_end)
    out["_ref"] = {
        "x": mesh.cell_center[:, 0].astype(np.float64),
        "y": mesh.cell_center[:, 1].astype(np.float64),
        "h": h_ref, "u": u_ref, "v": v_ref,
        "node_xy": mesh.node_xy.astype(np.float64),
        "cell_nodes": np.array(mesh.cell_nodes, dtype=object),
    }

    for run_id, config_name, ckpt_rel in runs:
        try:
            model, ds = _load_run(case_dir, config_name, ckpt_rel)
            Q = _predict_at_t(model, ds, t_end)
            out[run_id] = {"h": Q[:, 0], "u": Q[:, 1], "v": Q[:, 2]}
            logger.info(
                "%s :: h ∈ [%.3f, %.3f]  |V|_max = %.3f",
                run_id, Q[:, 0].min(), Q[:, 0].max(),
                np.sqrt(Q[:, 1] ** 2 + Q[:, 2] ** 2).max(),
            )
        except Exception as e:
            logger.warning("%s prediction failed (%s) - skipping", run_id, e)
            out[run_id] = None

    np.savez(cache_path, **{k: np.array(v, dtype=object) for k, v in out.items()})
    logger.info("Cached predictions -> %s", cache_path)
    return out


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _triangulation_from_meta(node_xy: np.ndarray, cell_nodes) -> mtri.Triangulation:
    triangles = []
    for cn in cell_nodes:
        cn = list(cn)
        if len(cn) == 3:
            triangles.append(cn)
        elif len(cn) == 4:
            triangles.append([cn[0], cn[1], cn[2]])
            triangles.append([cn[0], cn[2], cn[3]])
    return mtri.Triangulation(node_xy[:, 0], node_xy[:, 1], triangles)


def _cell_to_node(node_xy: np.ndarray, cell_nodes, cell_vals: np.ndarray) -> np.ndarray:
    nv = np.zeros(len(node_xy))
    nc = np.zeros(len(node_xy))
    for ci, cn in enumerate(cell_nodes):
        for ni in cn:
            nv[ni] += cell_vals[ci]
            nc[ni] += 1
    nc[nc == 0] = 1
    return nv / nc


def _wet_l2(pred: np.ndarray, ref: np.ndarray, h_ref: np.ndarray, h_dry: float) -> float:
    mask = h_ref > h_dry
    return float(np.sqrt(np.mean((pred[mask] - ref[mask]) ** 2)))


# ---------------------------------------------------------------------------
# Case 3 figures
# ---------------------------------------------------------------------------

def _bic_metrics(preds: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Dict[str, float]]:
    ref = preds["_ref"]
    out: Dict[str, Dict[str, float]] = {}
    for rid, run in preds.items():
        if rid.startswith("_") or run is None:
            continue
        speed_pred = np.sqrt(run["u"] ** 2 + run["v"] ** 2)
        speed_ref = np.sqrt(ref["u"] ** 2 + ref["v"] ** 2)
        out[rid] = {
            "L2_h": _wet_l2(run["h"], ref["h"], ref["h"], 0.01),
            "L2_V": _wet_l2(speed_pred, speed_ref, ref["h"], 0.01),
        }
    return out


def fig_ablation_contours(preds: Dict[str, Dict[str, np.ndarray]]) -> None:
    """4-panel velocity-magnitude contour comparison: SRH-2D ref, BIC_A, BIC_B, BIC_E."""
    ref = preds["_ref"]
    triang = _triangulation_from_meta(ref["node_xy"], ref["cell_nodes"])

    panels = [
        ("SRH-2D reference", np.sqrt(ref["u"] ** 2 + ref["v"] ** 2)),
    ]
    for rid, label in [("BIC_A", "BIC-A: physics-only"),
                       ("BIC_B", "BIC-B: + 200 sparse vel"),
                       ("BIC_E", "BIC-E: + sparse + dense")]:
        run = preds[rid]
        if run is None:
            continue
        panels.append((label, np.sqrt(run["u"] ** 2 + run["v"] ** 2)))

    vmax = max(p[1].max() for p in panels)
    vmin = 0.0

    fig, axes = plt.subplots(2, 2, figsize=(11, 5.5), constrained_layout=True)
    for ax, (title, field) in zip(axes.ravel(), panels):
        nv = _cell_to_node(ref["node_xy"], ref["cell_nodes"], field)
        tcf = ax.tricontourf(triang, nv, levels=25, cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    cbar = fig.colorbar(tcf, ax=axes.ravel().tolist(), shrink=0.85, label=r"$|\mathbf{V}|$ [m/s]")

    out = BIC_OUT_DIR / "fig_ablation_contours"
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s.{pdf,png}", out)


def fig_ablation_bar(preds: Dict[str, Dict[str, np.ndarray]]) -> None:
    """Bar chart of L2(|V|) across all BIC runs."""
    metrics = _bic_metrics(preds)
    order = ["BIC_A", "BIC_B", "BIC_C", "BIC_D", "BIC_E", "BIC_F", "BIC_G", "BIC_H"]
    labels = ["A: phys-only", "B: 200 vel", "C: 50 vel", "D: teacher",
              "E: sparse+dense", "F: 200 vel +5% noise", "G: data-only sparse",
              "H: data-only dense"]
    vals = [metrics[r]["L2_V"] if r in metrics else np.nan for r in order]

    colors = ["#d62728"] + ["#1f77b4"] * 5 + ["#2ca02c"] * 2  # phys-only red, FVM+data blue, data-only green

    fig, ax = plt.subplots(figsize=(9, 4.2))
    bars = ax.bar(np.arange(len(order)), vals, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_yscale("log")
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel(r"$L_2(|\mathbf{V}|)$  (log scale, m/s)")
    ax.set_title("Block-in-channel: data-guidance ablation, velocity error vs run")
    ax.grid(True, axis="y", alpha=0.3, which="both")
    for b, v in zip(bars, vals):
        if not np.isnan(v):
            ax.text(b.get_x() + b.get_width() / 2, v * 1.10, f"{v:.2e}",
                    ha="center", va="bottom", fontsize=8)

    out = BIC_OUT_DIR / "fig_ablation_bar"
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s.{pdf,png}", out)


def fig_ablation_profiles(preds: Dict[str, Dict[str, np.ndarray]]) -> None:
    """Centerline (y ≈ 2.5 m) profiles of h(x) and |V|(x) for SRH ref + BIC_A + BIC_B + BIC_E."""
    ref = preds["_ref"]
    y_target = 2.5
    mask = np.abs(ref["y"] - y_target) < 0.30
    x_line = ref["x"][mask]
    order = np.argsort(x_line)

    speed_ref = np.sqrt(ref["u"] ** 2 + ref["v"] ** 2)
    series = [
        ("SRH-2D ref", "k", "-", ref["h"][mask][order], speed_ref[mask][order]),
    ]
    for rid, color, ls, lab in [
        ("BIC_A", "#d62728", "--", "BIC-A: phys-only"),
        ("BIC_B", "#1f77b4", "-",  "BIC-B: 200 sparse"),
        ("BIC_E", "#2ca02c", "-",  "BIC-E: sparse+dense"),
    ]:
        run = preds.get(rid)
        if run is None:
            continue
        h_p = run["h"][mask][order]
        v_p = np.sqrt(run["u"] ** 2 + run["v"] ** 2)[mask][order]
        series.append((lab, color, ls, h_p, v_p))

    x = x_line[order]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), constrained_layout=True)
    for label, color, ls, h_arr, v_arr in series:
        axes[0].plot(x, h_arr, color=color, ls=ls, label=label, lw=1.4)
        axes[1].plot(x, v_arr, color=color, ls=ls, label=label, lw=1.4)
    axes[0].set_xlabel("x [m]"); axes[0].set_ylabel("h [m]")
    axes[0].set_title(r"Depth profile at $y \approx 2.5$ m")
    axes[1].set_xlabel("x [m]"); axes[1].set_ylabel(r"$|\mathbf{V}|$ [m/s]")
    axes[1].set_title(r"Velocity-magnitude profile at $y \approx 2.5$ m")
    for ax in axes:
        ax.grid(True, alpha=0.3); ax.legend(fontsize=8, loc="best")

    out = BIC_OUT_DIR / "fig_ablation_profiles"
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s.{pdf,png}", out)


# ---------------------------------------------------------------------------
# Case 4 figures
# ---------------------------------------------------------------------------

def fig_savannah_contours(preds: Dict[str, Dict[str, np.ndarray]]) -> None:
    """2x2 contours: best run (SR_D, window 10) h and |V|, vs SRH-2D h and |V|."""
    ref = preds["_ref"]
    best = preds.get("SR_D")
    if best is None:
        logger.warning("SR_D predictions unavailable; skipping savannah contours")
        return

    triang = _triangulation_from_meta(ref["node_xy"], ref["cell_nodes"])
    speed_ref = np.sqrt(ref["u"] ** 2 + ref["v"] ** 2)
    speed_pred = np.sqrt(best["u"] ** 2 + best["v"] ** 2)

    panels = [
        ("FVM-PINN (SR-D)  h [m]",  best["h"], "viridis"),
        (r"FVM-PINN (SR-D)  $|\mathbf{V}|$ [m/s]", speed_pred, "hot_r"),
        ("SRH-2D  h [m]",  ref["h"], "viridis"),
        (r"SRH-2D  $|\mathbf{V}|$ [m/s]", speed_ref, "hot_r"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), constrained_layout=True)
    for ax, (title, field, cmap) in zip(axes.ravel(), panels):
        nv = _cell_to_node(ref["node_xy"], ref["cell_nodes"], field)
        tcf = ax.tricontourf(triang, nv, levels=25, cmap=cmap)
        fig.colorbar(tcf, ax=ax, shrink=0.85)
        ax.set_aspect("equal"); ax.set_title(title, fontsize=10)
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")

    out = SR_OUT_DIR / "fig_savannah_contours"
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s.{pdf,png}", out)


def _parse_per_time_l2(stdout_path: Path) -> Dict[float, Tuple[float, float]]:
    """Pull the 'PINN vs SRH-2D at each anchor time' table from a run's stdout.log."""
    if not stdout_path.exists():
        return {}
    pattern = re.compile(
        r"t=\s*([\d.]+)s\s+L2\(h\)=([\d.eE+-]+)\s+L2\(\|V\|\)=([\d.eE+-]+)"
    )
    out: Dict[float, Tuple[float, float]] = {}
    for line in stdout_path.read_text(errors="ignore").splitlines():
        m = pattern.search(line)
        if m:
            t, lh, lv = float(m.group(1)), float(m.group(2)), float(m.group(3))
            out[t] = (lh, lv)
    return out


def _per_time_l2_from_json(json_path: Path) -> Dict[float, Tuple[float, float]]:
    """Read the per-time L2 table from a HydroNet ``history_fvm_pinn_*.json``.

    The file's ``metrics.per_time`` field is a list of
    ``{"t": ..., "L2_h": ..., "L2_vel": ...}`` dicts (written by the
    standard / teacher example scripts). Used as a fallback for runs
    whose stdout was not preserved (e.g. SR-E).
    """
    if not json_path.exists():
        return {}
    with json_path.open() as f:
        data = json.load(f)
    out: Dict[float, Tuple[float, float]] = {}
    for entry in (data.get("metrics") or {}).get("per_time", []) or []:
        try:
            t = float(entry["t"])
            lh = float(entry["L2_h"])
            lv = float(entry["L2_vel"])
        except (KeyError, TypeError, ValueError):
            continue
        out[t] = (lh, lv)
    return out


def _load_per_time_l2(run_id: str) -> Dict[float, Tuple[float, float]]:
    """Dispatch to the log parser or JSON parser per ``SR_PER_TIME_SOURCES``."""
    src = SR_PER_TIME_SOURCES.get(run_id)
    if src is None:
        return {}
    kind, relpath = src
    full = SR_DIR / relpath
    if kind == "log":
        return _parse_per_time_l2(full)
    if kind == "json":
        return _per_time_l2_from_json(full)
    raise ValueError(f"Unknown per-time source kind {kind!r} for {run_id}")


def fig_savannah_l2_by_time() -> None:
    """L2(h) and L2(|V|) vs time, for SR-A...SR-E (the strategy ablation set)."""
    runs = [("SR_A", "physics only"),
            ("SR_B", "single net"),
            ("SR_C", "window(5)"),
            ("SR_D", "window(10)"),
            ("SR_E", "FVM teacher")]
    colors = {
        "SR_A": "#7f7f7f",
        "SR_B": "#d62728",
        "SR_C": "#1f77b4",
        "SR_D": "#2ca02c",
        "SR_E": "#ff7f0e",
    }

    fig, axes = plt.subplots(2, 1, figsize=(5.5, 7.4), constrained_layout=True)
    for rid, label in runs:
        per_t = _load_per_time_l2(rid)
        if not per_t:
            logger.warning("No per-time L2 data found for %s", rid)
            continue
        ts = sorted(per_t.keys())
        l2h = [per_t[t][0] for t in ts]
        l2v = [per_t[t][1] for t in ts]
        axes[0].plot(ts, l2h, "o-", color=colors[rid], label=f"{rid} ({label})", lw=1.6, ms=5)
        # Mask out the t=720 spike for the V plot — it's a div-by-h artefact
        ts_v = [t for t in ts if per_t[t][1] < 1.0]
        l2v_clean = [per_t[t][1] for t in ts_v]
        axes[1].plot(ts_v, l2v_clean, "o-", color=colors[rid], label=f"{rid} ({label})", lw=1.6, ms=5)

    axes[0].set_xlabel("t (s)", fontsize=18); 
    axes[0].set_ylabel(r"$L_2(h)$ [m]", fontsize=18)
    axes[0].set_title(r"Depth error vs time", fontsize=16)
    axes[0].set_yscale("log"); axes[0].grid(True, which="both", alpha=0.3)
    #set tick label font size
    axes[0].tick_params(axis='both', which='major', labelsize=14)
    axes[0].legend(fontsize=10)
    axes[1].set_xlabel("t (s)", fontsize=18); 
    axes[1].set_ylabel(r"$L_2(|\mathbf{u}|)$ (m/s)", fontsize=18)
    axes[1].set_title(r"Velocity error vs time (t=720 s spike excluded)", fontsize=16)
    axes[1].tick_params(axis='both', which='major', labelsize=14)
    axes[1].set_yscale("log"); axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(fontsize=10)

    #add (a) and (b) labels to the subplots at the upper left corner
    axes[0].text(-0.15, 1.05, "(a)", transform=axes[0].transAxes, fontsize=18)
    axes[1].text(-0.15, 1.05, "(b)", transform=axes[1].transAxes, fontsize=18)

    out = SR_OUT_DIR / "fig_savannah_l2_by_time"
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s.{pdf,png}", out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description="Build manuscript figures.")
    p.add_argument("--skip-bic", action="store_true")
    p.add_argument("--skip-sr",  action="store_true")
    p.add_argument("--regenerate", action="store_true",
                   help="Force re-prediction (ignore cache).")
    args = p.parse_args()

    if not args.skip_bic:
        logger.info("=== Building Case 3 (block-in-channel) figures ===")
        bic_preds = collect_predictions(
            BIC_DIR, BIC_RUNS, BIC_T_END,
            CACHE_DIR / "bic_predictions.npz",
            regenerate=args.regenerate,
        )
        fig_ablation_contours(bic_preds)
        fig_ablation_bar(bic_preds)
        fig_ablation_profiles(bic_preds)

    if not args.skip_sr:
        logger.info("=== Building Case 4 (Savannah River) figures ===")
        sr_preds = collect_predictions(
            SR_DIR, SR_RUNS, SR_T_END,
            CACHE_DIR / "sr_predictions.npz",
            regenerate=args.regenerate,
        )
        #fig_savannah_contours(sr_preds)
        fig_savannah_l2_by_time()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
