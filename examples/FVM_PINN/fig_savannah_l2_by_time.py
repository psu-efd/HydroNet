"""Case 4 L2 error at each SRH-2D reference time, from the A100 runs behind Table 4.

Recomputes predictions from the final checkpoints on CPU. Two velocity metrics:
  raw     : u = hu / max(h, 1e-6), as in the training logs (checked against them);
  wet/dry : velocity set to zero where the predicted depth is below h_dry, the rule
            the FVM teacher uses. Plotted for every run and time; nothing is dropped.
Writes figures/fig_savannah_l2_by_time.{pdf,png} and a JSON table next to this script.
"""
import json
import logging
import os
import re
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

logging.disable(logging.CRITICAL)
sys.path.insert(0, "/storage/work/axt5780/Projects/Hydro/HydroNet")
from HydroNet import Config, FVM_SWE_PINN, FVM_PINNDataset
from HydroNet.models.FVM_PINN._internal.trainers.window_trainer import _split_windows

HERE = Path(__file__).resolve().parent
A100 = Path("/storage/work/axt5780/Projects/Hydro/HydroNet_a100_timing/examples/FVM_PINN/savannah_river")
OUT = Path("/storage/work/axt5780/Projects/Hydro/fvm_pinn_swe_manuscript/figures/fig_savannah_l2_by_time")

RUNS = [("SR_A", "SR-A (physics only)", "#333333", "o"),
        ("SR_B", "SR-B (single network)", "#D55E00", "s"),
        ("SR_C", "SR-C (5 windows)", "#0072B2", "^"),
        ("SR_D", "SR-D (10 windows)", "#009E73", "D"),
        ("SR_E", "SR-E (FVM teacher)", "#E69F00", "v")]
LOG_RE = re.compile(r"t=\s*([\d.]+)s\s+L2\(h\)=(\S+)\s+L2\(\|V\|\)=(\S+)")


def logged(rid):
    out = {}
    for line in (A100 / "runs" / rid / "stdout.log").read_text(errors="ignore").splitlines():
        m = LOG_RE.search(line)
        if m:
            out[float(m.group(1))] = (float(m.group(2)), float(m.group(3)))
    return out


def load_net(model, path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = model.get_internal_network()
    net.load_state_dict(ck["network_state"])
    net.set_normalisation(ck["x_mean"], ck["x_std"])


def checkpoint_for(rid, cfg, t):
    """Same routing as WindowTrainer.predict for windowed runs."""
    d = A100 / "checkpoints" / rid
    if (d / "teacher_final.pt").exists():
        return d / "teacher_final.pt"
    if not (d / "window_000").exists():
        return d / "ckpt_final.pt"
    win = _split_windows(float(cfg.get("training.t_start", 0.0)),
                         float(cfg.get_required_config("training.t_end")),
                         int(cfg.get_required_config("training.window.n_windows")),
                         float(cfg.get("training.window.overlap_frac", 0.10)))
    for wk, (_, hi) in enumerate(win):
        if t <= hi or wk == len(win) - 1:
            if wk > 0 and t <= win[wk - 1][1]:
                continue
            return d / f"window_{wk:03d}" / "ckpt_final.pt"
    raise ValueError(t)


def evaluate():
    os.chdir(A100)
    with h5py.File("data/Savana_XMDFC.h5", "r") as f:
        times = f["Water_Depth_m/Times"][()]
        H = f["Water_Depth_m/Values"][()]
        V = f["Vel_Mag_m_p_s/Values"][()]
    table = {}
    for rid, *_ in RUNS:
        cfg = Config(f"fvm_pinn_config_{rid}.yaml")
        cfg.set("device.type", "cpu")
        model = FVM_SWE_PINN(cfg)
        ds = FVM_PINNDataset(cfg)
        model.set_mesh_context(h_still_cells=ds.get_h_still(), cell_xy=ds.get_cell_xy())
        mesh = ds.get_mesh()
        xc = torch.tensor(mesh.cell_center[:, 0], dtype=torch.float64)
        yc = torch.tensor(mesh.cell_center[:, 1], dtype=torch.float64)
        h_dry = float(cfg.get("physics.h_dry", 0.1))
        log = logged(rid)
        rows = []
        for ti, t in enumerate(times):
            load_net(model, checkpoint_for(rid, cfg, float(t)))
            with torch.no_grad():
                P = model(torch.column_stack([xc, yc, torch.full_like(xc, float(t))])).numpy()
            h, sp = P[:, 0], np.hypot(P[:, 1], P[:, 2])
            mask = H[ti] > h_dry
            l2h = float(np.sqrt(np.mean((h[mask] - H[ti][mask]) ** 2)))
            raw = float(np.sqrt(np.mean((sp[mask] - V[ti][mask]) ** 2)))
            sp_wd = np.where(h < h_dry, 0.0, sp)
            wd = float(np.sqrt(np.mean((sp_wd[mask] - V[ti][mask]) ** 2)))
            lh, lv = log[float(t)]
            if abs(l2h / lh - 1) > 1e-3 or abs(raw / lv - 1) > 1e-3:
                raise SystemExit(f"{rid} t={t}: recomputed {l2h:.5e},{raw:.5e} != log {lh:.5e},{lv:.5e}")
            rows.append(dict(t=float(t), L2_h=l2h, L2_u_raw=raw, L2_u_wetdry=wd,
                             n_wet=int(mask.sum()), n_dry_pred=int((h[mask] < h_dry).sum()),
                             n_nonpos=int((h[mask] <= 0).sum()), max_speed_raw=float(sp[mask].max())))
        table[rid] = rows
    return table


def plot(table):
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")
    fig, axes = plt.subplots(2, 1, figsize=(5.5, 7.4), constrained_layout=True)
    for rid, label, color, marker in RUNS:
        ts = [r["t"] for r in table[rid]]
        kw = dict(color=color, marker=marker, ms=6, lw=1.6, label=label)
        axes[0].plot(ts, [r["L2_h"] for r in table[rid]], **kw)
        axes[1].plot(ts, [r["L2_u_wetdry"] for r in table[rid]], **kw)
    for ax, yl, title, lab in zip(axes, (r"$L_2(h)$ (m)", r"$L_2(|\mathbf{u}|)$ (m/s)"),
                                  ("Depth error vs time", "Velocity error vs time"), "ab"):
        ax.set_yscale("log")
        ax.set_xlabel("$t$ (s)", fontsize=18)
        ax.set_ylabel(yl, fontsize=18)
        ax.set_title(title, fontsize=16)
        ax.grid(True, which="both", alpha=0.3)
        ax.tick_params(axis="both", which="major", labelsize=14)
        ax.text(-0.15, 1.05, f"({lab})", transform=ax.transAxes, fontsize=18)
    axes[0].legend(fontsize=10, loc="upper right")
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    table = evaluate()
    (HERE / "fig_savannah_l2_by_time.json").write_text(json.dumps(table, indent=1))
    plot(table)
    for rid, rows in table.items():
        print(rid)
        for r in rows:
            print(f"  t={r['t']:6.0f} h={r['L2_h']:.4e} u_raw={r['L2_u_raw']:.4e} u_wd={r['L2_u_wetdry']:.4e} "
                  f"wet={r['n_wet']} below_hdry={r['n_dry_pred']} nonpos={r['n_nonpos']} maxV={r['max_speed_raw']:.3g}")
