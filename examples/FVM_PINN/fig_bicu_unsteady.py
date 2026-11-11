"""
Unsteady block-in-channel (BICU, inflow pulse): manuscript figures.

    fig_bicu_setup       (a) inflow hydrograph with the dense-snapshot and
                         held-out times, (b) mesh with block, gauges, probes
    fig_bicu_probes      h(t) and |u|(t) at three probe points: SRH-2D outputs
                         (10 s) vs networks evaluated continuously (1 s)
    fig_bicu_errors      (a) L2(|u|) at every SRH-2D output time, with
                         snapshot interpolation and the FVM solver as
                         references; (b) held-out error vs FVM residual
                         for lambda_fvm = 1, 0.1, 0.01, 0
    fig_bicu_snapshot    |u| at a held-out time (default 400 s, 10 s after
                         the inflow peak): SRH-2D and the networks, with
                         each network's difference from SRH-2D

Windowed runs are routed to their time windows as in
TimeWindowTrainer.predict. Predictions (every 1 s, all cells) and the FVM
solver trajectory are cached in plan/figures/_data/.

Usage (needs `module load texlive/2024` for the LaTeX labels)
-----
    python examples/FVM_PINN/fig_bicu_unsteady.py [--regenerate]
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch

import build_manuscript_figures as bmf
from HydroNet import Config, FVM_SWE_PINN, FVM_PINNDataset
from HydroNet.models.FVM_PINN._internal.fvm import run_fvm_rk2
from HydroNet.models.FVM_PINN._internal.fvm.riemann_solver import compute_fvm_residual
from HydroNet.models.FVM_PINN._internal.trainers.window_trainer import _split_windows

logger = logging.getLogger(__name__)

CASE = bmf.BIC_DIR
OUT = bmf.BIC_OUT_DIR
SRH_H5 = CASE / "data" / "hydrograph" / "block_in_channel_hydrograph_XMDFC.h5"
T0, T1 = 360.0, 600.0
T_FINE = np.arange(T0, T1 + 0.5, 1.0)                 # network evaluation grid
DENSE = [360.0, 390.0, 420.0, 450.0, 480.0, 510.0, 540.0, 570.0, 600.0]
H_WET = 0.01

RUNS = {   # id: (config, label, colour, linestyle)
    "BICU_A":    ("fvm_pinn_config_BICU_A.yaml",    "BICU-A (physics only)",                  "#7f7f7f", "--"),
    "BICU_C":    ("fvm_pinn_config_BICU_C.yaml",    "BICU-C (physics + 10 velocity gauges)",  "#9467bd", "-."),
    "BICU_EW":   ("fvm_pinn_config_BICU_EW.yaml",   r"BICU-EW (physics + dense, $\lambda_{\mathrm{fvm\mbox{-}pinn}}=1$)", "#d62728", "-"),
    "BICU_EWL1": ("fvm_pinn_config_BICU_EWL1.yaml", r"$\lambda_{\mathrm{fvm\mbox{-}pinn}}=0.1$", "#ff7f0e", "-"),
    "BICU_EWL2": ("fvm_pinn_config_BICU_EWL2.yaml", r"$\lambda_{\mathrm{fvm\mbox{-}pinn}}=0.01$", "#bcbd22", "-"),
    "BICU_HW":   ("fvm_pinn_config_BICU_HW.yaml",   "BICU-HW (dense data only)",              "#2ca02c", "-"),
    # physics prior: distilled from the FVM solver (T), then fine-tuned from T
    "BICU_T":    ("fvm_pinn_config_BICU_T.yaml",    "BICU-T (distilled from FVM solver)",     "#1f77b4", "-"),
    "BICU_CT":   ("fvm_pinn_config_BICU_CT.yaml",   "BICU-CT (from T: physics + 10 velocity gauges)", "#9467bd", "-"),
    "BICU_PT":   ("fvm_pinn_config_BICU_PT.yaml",   "BICU-PT (from T: physics only)",         "#7f7f7f", "--"),
    "BICU_GT":   ("fvm_pinn_config_BICU_GT.yaml",   "BICU-GT (from T: 10 velocity gauges only)", "#ff7f0e", ":"),
}
SHOWN = ["BICU_A", "BICU_C", "BICU_EW", "BICU_HW"]          # time-series / error panels
SHOWN_PRIOR = ["BICU_T", "BICU_CT", "BICU_PT", "BICU_GT"]   # fig_bicu_probes_teacher
LAMBDA_SWEEP = [("BICU_EW", 1.0), ("BICU_EWL1", 0.1), ("BICU_EWL2", 0.01), ("BICU_HW", 0.0)]
PROBES = {"P1 (upstream)": (3.0, 2.5), "P2 (beside block)": (5.5, 4.0), "P3 (wake)": (7.5, 2.5)}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _in_case(fn):
    cwd = os.getcwd()
    os.chdir(CASE)
    try:
        return fn()
    finally:
        os.chdir(cwd)


def load_dataset(config_name: str):
    def _load():
        cfg = Config(config_name)
        cfg.set("device.type", "cpu")
        return cfg, FVM_PINNDataset(cfg), FVM_SWE_PINN(cfg)
    return _in_case(_load)


def predict_run(rid: str) -> np.ndarray:
    """Q = [xi, hu, hv] at all cells for every t in T_FINE -> [nt, n_cells, 3]."""
    cfg, ds, model = load_dataset(RUNS[rid][0])
    ck_dir = CASE / str(cfg.get("training.logging.checkpoint_dir"))
    if str(cfg.get("training.strategy")) == "window":
        wins = _split_windows(T0, T1, int(cfg.get("training.window.n_windows")),
                              float(cfg.get("training.window.overlap_frac", 0.1)))
        paths = [ck_dir / f"window_{k:03d}" / "ckpt_final.pt" for k in range(len(wins))]
    elif str(cfg.get("training.strategy")) == "teacher":
        wins, paths = [(T0, T1)], [ck_dir / "teacher_final.pt"]
    else:
        wins, paths = [(T0, T1)], [ck_dir / "ckpt_final.pt"]
    states = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]

    def window_of(t):   # same routing as TimeWindowTrainer.predict
        for k in range(len(wins) - 1):
            if t <= wins[k][1]:
                return k
        return len(wins) - 1

    net = model.get_internal_network()
    xy = ds.get_cell_xy().double()
    out = np.zeros((len(T_FINE), xy.shape[0], 3))
    loaded = None
    for i, t in enumerate(T_FINE):
        k = window_of(t)
        if k != loaded:
            net.load_state_dict(states[k]["network_state"])
            net.set_normalisation(states[k]["x_mean"], states[k]["x_std"])
            loaded = k
        with torch.no_grad():
            out[i] = net(torch.cat([xy, torch.full((len(xy), 1), float(t), dtype=torch.float64)], 1)).numpy()
    logger.info("%s predicted (%d windows)", rid, len(wins))
    return out


def fvm_trajectory(ds) -> np.ndarray:
    """In-house FVM solver from the same IC, at the SRH-2D output times."""
    Q0 = ds.get_ic_data()["U_true"]
    _, Qs = run_fvm_rk2(Q0, ds.get_mesh_data(), ds.get_h_still(), np.arange(T0, T1 + 0.5, 10.0).tolist(), cfl=0.3)
    return Qs.numpy()


def load_all(regenerate: bool):
    cache = bmf.CACHE_DIR / "bicu_unsteady.npz"
    _, ds, _ = load_dataset(RUNS["BICU_A"][0])
    Q, Q_fvm = {}, None
    if cache.exists() and not regenerate:
        z = np.load(cache)
        Q = {r: z[r] for r in RUNS if r in z.files}
        Q_fvm = z["_fvm"] if "_fvm" in z.files else None
        logger.info("Loaded cached predictions from %s", cache)
    missing = [r for r in RUNS if r not in Q]
    if missing or Q_fvm is None:
        Q.update({r: predict_run(r) for r in missing})
        if Q_fvm is None:
            Q_fvm = fvm_trajectory(ds)
        np.savez(cache, _fvm=Q_fvm, **Q)
        logger.info("Cached predictions -> %s (added %s)", cache, missing or "FVM only")
    return ds, Q, Q_fvm


def srh2d():
    with h5py.File(SRH_H5, "r") as f:
        T = f["Water_Depth_m/Times"][:]
        m = (T >= T0 - 1e-6) & (T <= T1 + 1e-6)
        return T[m], f["Water_Depth_m/Values"][m], f["Velocity_m_p_s/Values"][m]


def to_phys(Q, hs):
    h = Q[..., 0] + hs
    hsafe = np.maximum(h, 1e-6)
    return h, np.hypot(Q[..., 1], Q[..., 2]) / hsafe


def l2(a, b, wet):
    return float(np.sqrt(np.mean((a - b)[wet] ** 2)))


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_setup(ds, T_srh):
    cfg = _in_case(lambda: Config(RUNS["BICU_C"][0]))
    # BC ids are int keys in the YAML, so look the inlet up by type
    hyd = np.array(next(bc["hydrograph"] for bc in cfg.get("boundary_conditions").values()
                        if bc.get("type") == "inlet-q"), dtype=float)
    md = ds.get_mesh_data()
    _, ds_c, _ = load_dataset(RUNS["BICU_C"][0])
    gauges = np.unique(ds_c.get_ref_data()["xyt"][:, :2].numpy(), axis=0)

    fig = plt.figure(figsize=(11, 7.6), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.15])
    ax0, ax1 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    # (a) hydrograph
    tt = np.linspace(0, T1, 601)
    ax0.plot(tt, np.interp(tt, hyd[:, 0], hyd[:, 1]), "k-", lw=2.0)
    ax0.axvspan(0, T0, color="0.9", zorder=0)
    ax0.text(T0 / 2, 0.47, "spin-up (SRH-2D only)", ha="center", fontsize=14, color="0.35")
    ax0.text(500, 0.785, "FVM-PINN training window", ha="center", fontsize=14)
    held = [t for t in T_srh if t not in DENSE]
    ax0.plot(DENSE, np.interp(DENSE, hyd[:, 0], hyd[:, 1]), "ks", ms=7, label="dense SRH-2D snapshots (every 30 s)")
    ax0.plot(held, np.interp(held, hyd[:, 0], hyd[:, 1]), "o", mfc="white", mec="k", ms=6,
             label="held-out SRH-2D outputs (10 s)")
    ax0.set_xlim(0, T1); ax0.set_ylim(0.3, 0.82)
    ax0.set_xlabel(r"$t$ (s)", fontsize=18); ax0.set_ylabel(r"$Q_{\mathrm{in}}$ (m$^3$\,s$^{-1}$)", fontsize=18)
    ax0.set_title("Inflow hydrograph", fontsize=16)
    ax0.legend(fontsize=12, loc="upper left", framealpha=0.95)    # empty spin-up area
    ax0.tick_params(labelsize=14); ax0.grid(True, alpha=0.3)

    # (b) mesh, block, gauges, probes
    ref = {"node_xy": ds.get_mesh().node_xy, "cell_nodes": np.array(ds.get_mesh().cell_nodes, dtype=object)}
    tri = bmf._triangulation_from_meta(ref["node_xy"], ref["cell_nodes"])
    ax1.triplot(tri, color="0.8", lw=0.4)
    fc, bid, fr = md["face_center"].numpy(), md["face_bc_id"].numpy(), md["face_right"].numpy()
    blk = fc[(fr < 0) & (bid == 5)]
    ax1.add_patch(plt.Rectangle((blk[:, 0].min(), blk[:, 1].min()), np.ptp(blk[:, 0]), np.ptp(blk[:, 1]),
                                color="0.35", zorder=3))
    ax1.plot(gauges[:, 0], gauges[:, 1], "^", color="#9467bd", ms=10, mec="k", zorder=4,
             label="fixed gauges (BICU-B: depth; BICU-C: velocity)")
    for name, (px, py) in PROBES.items():
        ax1.plot(px, py, "o", color="#1f77b4", ms=10, mec="k", zorder=5)
        ax1.annotate(name.split(" ")[0], (px, py), xytext=(8, 6), textcoords="offset points", fontsize=14,
                     bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85), zorder=6)
    ax1.plot([], [], "o", color="#1f77b4", mec="k", ms=10, label="probe points P1--P3")
    ax1.annotate("", xy=(1.2, 2.5), xytext=(-0.2, 2.5), arrowprops=dict(arrowstyle="->", lw=2))
    ax1.text(0.05, 2.75, r"$Q_{\mathrm{in}}(t)$", fontsize=14)
    ax1.text(14.0, 2.75, r"$z_s = 0.4$~m", fontsize=14, ha="right")
    ax1.set_aspect("equal"); ax1.set_xlim(-0.3, 15.3); ax1.set_ylim(-0.1, 5.1)
    ax1.set_xlabel(r"$x$ (m)", fontsize=18); ax1.set_ylabel(r"$y$ (m)", fontsize=18)
    ax1.set_title("Mesh (1,326 cells), block, gauges and probes", fontsize=16)
    ax1.legend(fontsize=12, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False)
    ax1.tick_params(labelsize=14)
    for ax, lab in zip((ax0, ax1), ("(a)", "(b)")):
        ax.text(-0.08, 1.04, lab, transform=ax.transAxes, fontsize=18)
    _save(fig, "fig_bicu_setup")


def fig_probes(ds, Q, T_srh, H_srh, V_srh, runs=SHOWN, name="fig_bicu_probes", Q_fvm=None):
    """Probe time series for `runs`; dense-snapshot times are marked unless
    `Q_fvm` is given, in which case the FVM solver (10 s) is drawn instead."""
    xy = ds.get_cell_xy().numpy(); hs = ds.get_h_still().numpy()
    cells = {n: int(np.argmin(np.hypot(xy[:, 0] - p[0], xy[:, 1] - p[1]))) for n, p in PROBES.items()}
    T_fvm = np.arange(T0, T1 + 0.5, 10.0)
    fig, axes = plt.subplots(len(PROBES), 2, figsize=(12, 10), sharex=True, constrained_layout=True)
    for row, (probe, c) in enumerate(cells.items()):
        for col, ylab in enumerate((r"$h$ (m)", r"$|\mathbf{u}|$ (m/s)")):
            ax = axes[row, col]
            if Q_fvm is None:
                for t in DENSE:
                    ax.axvline(t, color="0.85", lw=1.0, zorder=0)
            for r in runs:
                h, sp = to_phys(Q[r][:, c], hs[c])
                _, lab, color, ls = RUNS[r]
                ax.plot(T_FINE, (h, sp)[col], color=color, ls=ls, lw=1.7, label=lab)
            if Q_fvm is not None:
                h, sp = to_phys(Q_fvm[:, c], hs[c])
                ax.plot(T_fvm, (h, sp)[col], "s", ms=6, mfc="none", mec="#8c564b", mew=1.4,
                        label="FVM solver (10 s)", zorder=4)
            srh = H_srh[:, c] if col == 0 else np.hypot(V_srh[:, c, 0], V_srh[:, c, 1])
            ax.plot(T_srh, srh, "ko", ms=5, mfc="k", label="SRH-2D (10 s outputs)", zorder=5)
            ax.set_xlim(T0, T1); ax.set_ylabel(ylab, fontsize=16); ax.tick_params(labelsize=13)
            ax.grid(True, alpha=0.3)
            ax.set_title(f"{probe}, $({xy[c, 0]:.1f}, {xy[c, 1]:.1f})$~m", fontsize=14)
            if row == len(PROBES) - 1:
                ax.set_xlabel(r"$t$ (s)", fontsize=18)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=3, fontsize=12, frameon=False)
    for i, ax in enumerate(axes.ravel()):
        ax.text(-0.13, 1.04, f"({'abcdef'[i]})", transform=ax.transAxes, fontsize=16)
    _save(fig, name)


def fig_errors(ds, Q, Q_fvm, T_srh, H_srh, V_srh):
    hs = ds.get_h_still().numpy(); md = ds.get_mesh_data()
    sp_srh = np.hypot(V_srh[..., 0], V_srh[..., 1])
    idx = {float(t): int(np.argmin(abs(T_FINE - t))) for t in T_srh}
    held = [t for t in T_srh if t not in DENSE]

    def err_series(sp_of_t):
        return np.array([l2(sp_of_t(t, k), sp_srh[k], H_srh[k] > H_WET) for k, t in enumerate(T_srh)])

    def interp_sp(t, k):
        if t in DENSE:
            return sp_srh[k]
        i = np.searchsorted(DENSE, t); a, b = DENSE[i - 1], DENSE[i]; w = (t - a) / (b - a)
        ka, kb = int(np.argmin(abs(T_srh - a))), int(np.argmin(abs(T_srh - b)))
        return (1 - w) * sp_srh[ka] + w * sp_srh[kb]

    e_interp = err_series(interp_sp)
    e_fvm = err_series(lambda t, k: to_phys(Q_fvm[k], hs)[1])
    e_run = {r: err_series(lambda t, k, r=r: to_phys(Q[r][idx[float(t)]], hs)[1]) for r in RUNS}

    def residual(r):   # mean FVM-PINN residual over the interior 10 s output times
        vals = []
        for t in T_srh[1:-1]:
            i = idx[float(t)]
            Q0 = torch.tensor(Q[r][i])
            dqdt = torch.tensor((Q[r][i + 1] - Q[r][i - 1]) / (T_FINE[i + 1] - T_FINE[i - 1]))
            R = compute_fvm_residual(Q0, md, ds.get_h_still().double(), H_WET, t=float(t))
            wet = (Q0[:, 0] + ds.get_h_still().double()) > H_WET
            vals.append(((dqdt + R)[wet] ** 2).mean(0).sum().item())
        return float(np.mean(vals))

    fig = plt.figure(figsize=(13, 5.8), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.6, 1])
    ax0, ax1 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    for t in DENSE:
        ax0.axvline(t, color="0.85", lw=1.0, zorder=0)
    # Interpolation is exact at the snapshots and the FVM solver starts from
    # the SRH-2D state, so both are shown only where they are non-zero
    held_t = np.array([t not in DENSE for t in T_srh])
    ax0.semilogy(T_srh[held_t], e_interp[held_t], "ks", mfc="white", ms=6, ls="none",
                 label="snapshot interpolation (held-out times)")
    ax0.semilogy(T_srh[1:], e_fvm[1:], "k:", lw=2.0, label="FVM solver (no data)")
    for r in SHOWN:
        _, lab, color, ls = RUNS[r]
        ax0.semilogy(T_srh, np.maximum(e_run[r], 1e-4), color=color, ls=ls, lw=1.7, marker="o", ms=3.5, label=lab)
    ax0.set_xlim(T0, T1); ax0.set_ylim(1e-3, 0.8); ax0.set_xlabel(r"$t$ (s)", fontsize=18)
    ax0.set_ylabel(r"$L_2(|\mathbf{u}|)$ (m/s)", fontsize=18)
    ax0.set_title("Velocity error vs SRH-2D (grey lines: dense snapshots)", fontsize=15)
    h0, l0 = ax0.get_legend_handles_labels()
    fig.legend(h0, l0, loc="outside lower center", ncol=3, fontsize=11, frameon=False); ax0.tick_params(labelsize=13)
    ax0.grid(True, which="both", alpha=0.25)

    # (b) lambda trade-off: held-out error vs FVM residual
    held_mask = np.array([t not in DENSE for t in T_srh])
    xs, ys = [], []
    for r, lam in LAMBDA_SWEEP:
        xs.append(residual(r)); ys.append(float(np.mean(e_run[r][held_mask])))
        logger.info("%s lambda=%g: held-out L2(|u|) %.4f, mean residual %.2e", r, lam, ys[-1], xs[-1])
    ax1.plot(xs, ys, "-", color="0.5", lw=1.2, zorder=1)
    offsets = {1.0: (10, -4), 0.1: (10, 2), 0.01: (-12, 10), 0.0: (0, 14)}
    for (r, lam), x, y in zip(LAMBDA_SWEEP, xs, ys):
        ax1.plot(x, y, "o", ms=10, color=RUNS[r][2], mec="k", zorder=3)
        dx, dy = offsets[lam]
        ax1.annotate(rf"$\lambda = {lam:g}$", (x, y), xytext=(dx, dy), textcoords="offset points",
                     fontsize=13, ha="left" if dx > 0 else ("right" if dx < 0 else "center"))
    ax1.margins(x=0.1)
    ax1.axhline(float(np.mean(e_interp[held_mask])), color="k", ls="--", lw=1.2, label="snapshot interpolation")
    ax1.set_xscale("log")
    ax1.set_xlabel(r"mean FVM-PINN residual $\mathcal{L}_{\mathrm{fvm\mbox{-}pinn}}$", fontsize=16)
    ax1.set_ylabel(r"held-out $L_2(|\mathbf{u}|)$ (m/s)", fontsize=16)
    ax1.set_title(r"Residual vs accuracy ($\lambda_{\mathrm{fvm\mbox{-}pinn}}$ sweep)", fontsize=15)
    ax1.legend(fontsize=11, loc="center right"); ax1.tick_params(labelsize=13); ax1.grid(True, which="both", alpha=0.25)
    for ax, lab in zip((ax0, ax1), ("(a)", "(b)")):
        ax.text(-0.1, 1.04, lab, transform=ax.transAxes, fontsize=18)
    logger.info("held-out mean L2(|u|): interp %.4f, FVM solver %.4f; " % (
        np.mean(e_interp[held_mask]), np.mean(e_fvm[held_mask]))
        + ", ".join(f"{r} {np.mean(e_run[r][held_mask]):.4f}" for r in RUNS))
    _save(fig, "fig_bicu_errors")


SNAPSHOT_RUNS = ["BICU_HW", "BICU_EW", "BICU_C", "BICU_A"]


def fig_snapshot(ds, Q, T_srh, H_srh, V_srh, t_snap: float = 400.0):
    hs = ds.get_h_still().numpy()
    mesh = ds.get_mesh()
    node_xy, cell_nodes = mesh.node_xy, np.array(mesh.cell_nodes, dtype=object)
    tri = bmf._triangulation_from_meta(node_xy, cell_nodes)
    nodal = lambda f: bmf._cell_to_node(node_xy, cell_nodes, f)
    k = int(np.argmin(abs(T_srh - t_snap)))
    i = int(np.argmin(abs(T_FINE - t_snap)))
    sp_ref = np.hypot(V_srh[k, :, 0], V_srh[k, :, 1])
    wet = H_srh[k] > H_WET
    sp = {r: to_phys(Q[r][i], hs)[1] for r in SNAPSHOT_RUNS}
    d = {r: sp[r] - sp_ref for r in SNAPSHOT_RUNS}

    v_max = float(np.percentile(np.concatenate([sp_ref] + [sp[r] for r in SNAPSHOT_RUNS]), 99.5))
    lv = np.linspace(0.0, v_max, 25)
    d_max = float(np.percentile(np.abs(np.concatenate([d[r] for r in SNAPSHOT_RUNS])), 99.0))
    ld = np.linspace(-d_max, d_max, 25)

    n = len(SNAPSHOT_RUNS) + 1
    fig, axes = plt.subplots(n, 2, figsize=(12.5, 2.05 * n + 1.2), constrained_layout=True)
    held = "held-out" if t_snap not in DENSE else "training snapshot"
    axes[0, 0].tricontourf(tri, nodal(sp_ref), levels=lv, cmap="RdBu_r", extend="max")
    axes[0, 0].set_title(rf"SRH-2D, $t = {t_snap:.0f}$~s ({held})", fontsize=14)
    axes[0, 1].axis("off")
    axes[0, 1].text(0.5, 0.5, rf"$t = {t_snap:.0f}$~s: {t_snap - 390:.0f}~s after the inflow peak," + "\n"
                    + rf"{min(abs(t_snap - np.array(DENSE))):.0f}~s from the nearest training snapshot",
                    ha="center", va="center", fontsize=14, transform=axes[0, 1].transAxes)
    for row, r in enumerate(SNAPSHOT_RUNS, start=1):
        lab = RUNS[r][1]
        cf = axes[row, 0].tricontourf(tri, nodal(sp[r]), levels=lv, cmap="RdBu_r", extend="max")
        axes[row, 0].set_title(lab, fontsize=14)
        cd = axes[row, 1].tricontourf(tri, nodal(d[r]), levels=ld, cmap="PuOr_r", extend="both")
        axes[row, 1].set_title(rf"{lab.split(' (')[0]} $-$ SRH-2D, $L_2 = {l2(sp[r], sp_ref, wet):.3f}$~m/s",
                               fontsize=14)
    blk = None
    md = ds.get_mesh_data()
    fc, bid, fr = md["face_center"].numpy(), md["face_bc_id"].numpy(), md["face_right"].numpy()
    blk = fc[(fr < 0) & (bid == 5)]
    for ax in axes.ravel():
        if not ax.axison:
            continue
        ax.add_patch(plt.Rectangle((blk[:, 0].min(), blk[:, 1].min()), np.ptp(blk[:, 0]), np.ptp(blk[:, 1]),
                                   color="0.35", zorder=3))
        ax.set_aspect("equal"); ax.set_xlim(0, 15); ax.set_ylim(0, 5)
        ax.tick_params(labelsize=11)
    for ax in axes[-1, :]:
        ax.set_xlabel(r"$x$ (m)", fontsize=15)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$y$ (m)", fontsize=15)
    cb = fig.colorbar(cf, ax=axes[:, 0], orientation="horizontal", location="bottom", shrink=0.85, aspect=35,
                      ticks=np.arange(0.0, v_max + 1e-9, 0.1))
    cb.set_label(r"$|\mathbf{u}|$ (m/s)", fontsize=15)
    cb2 = fig.colorbar(cd, ax=axes[1:, 1], orientation="horizontal", location="bottom", shrink=0.85, aspect=35,
                       ticks=np.arange(-np.floor(d_max * 10) / 10, d_max + 1e-9, 0.1))
    cb2.set_label(r"$|\mathbf{u}| - |\mathbf{u}|_{\mathrm{SRH}}$ (m/s)", fontsize=15)
    letters = iter("abcdefghij")
    for ax in axes.ravel():
        if ax.axison:
            ax.text(-0.09, 1.06, f"({next(letters)})", transform=ax.transAxes, fontsize=15)
    _save(fig, f"fig_bicu_snapshot_t{t_snap:.0f}")


def _save(fig, name):
    out = OUT / name
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s.{pdf,png}", out)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--regenerate", action="store_true", help="Re-predict instead of using the cache")
    p.add_argument("--t-snap", type=float, default=400.0, help="Time of the snapshot figure [s]")
    args = p.parse_args()
    ds, Q, Q_fvm = load_all(args.regenerate)
    T_srh, H_srh, V_srh = srh2d()
    fig_setup(ds, T_srh)
    fig_probes(ds, Q, T_srh, H_srh, V_srh)
    fig_probes(ds, Q, T_srh, H_srh, V_srh, runs=SHOWN_PRIOR, name="fig_bicu_probes_teacher", Q_fvm=Q_fvm)
    fig_errors(ds, Q, Q_fvm, T_srh, H_srh, V_srh)
    fig_snapshot(ds, Q, T_srh, H_srh, V_srh, args.t_snap)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
