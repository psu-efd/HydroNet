"""Mass-conservation errors of the trained networks, Cases 3-5 (reviewer items R2-2, R3-B1a).

Every model is queried at the cell centres and passed through the same finite-volume
continuity balance, so networks trained with any loss (FVM, strong form, data only) and
the SRH-2D output are measured the same way:

    r_i(t) = d(xi_i)/dt + (1/A_i) sum_f F_f l_f                (F = Roe mass flux)
    eps_inf(t) = max_wet |r_i| / (Q_in / A_domain)
    eps_V(t)   = |sum_i A_i r_i| / Q_in = |dV/dt - (Q_in,num - Q_out,num)| / Q_in

Interior fluxes telescope in sum_i A_i r_i; the script checks this to round-off. Also stored:
the numerical inlet and outlet discharge, and the accumulated volume error
    E_V(t) = [V(t) - V(t0) - int (Q_in - Q_out,num) dt] / int Q_in dt,
which uses the prescribed Q_in and sees jumps between time windows, unlike the rate eps_V.

d(xi)/dt comes from autograd for networks, and from central differences of the outputs
for SRH-2D. Each checkpoint is first checked against the L2 errors in the manuscript tables.

Usage: conservation_diagnostics.py [--levels 100] [--only BIC_B SR_B ...]
Writes conservation_diagnostics.json next to this script.
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

logging.disable(logging.CRITICAL)
sys.path.insert(0, "/storage/work/axt5780/Projects/Hydro/HydroNet")
from HydroNet import Config, FVM_SWE_PINN, FVM_PINNDataset
from HydroNet.models.FVM_PINN._internal.fvm.riemann_solver import (
    _build_ghost_states, compute_fvm_residual, interp_hydrograph, roe_flux_2d)
from HydroNet.models.FVM_PINN._internal.pinn.loss import _batch_time_grad
from HydroNet.models.FVM_PINN._internal.trainers.window_trainer import _split_windows

HERE = Path(__file__).resolve().parent
A100 = Path("/storage/work/axt5780/Projects/Hydro/HydroNet_a100_timing/examples/FVM_PINN")
BIC_A100, SR_A100 = A100 / "block_in_channel", A100 / "savannah_river"
BIC_LIVE, SR_LIVE = HERE / "block_in_channel", HERE / "savannah_river"
SR_REF = "data_savannah_flat/Savana_flat_XMDFC.h5"
BICU_REF = "data/hydrograph/block_in_channel_hydrograph_XMDFC.h5"
BIC_REF = "data/block_in_channel_XMDFC.h5"

# id -> (case, tree, config stem, checkpoint id, table check (t, L2(h), L2(|u|)); None = no value in a table)
RUNS = {}
_bic = {"A": (2.51e-3, 1.66e-1), "B": (4.40e-3, 7.81e-3), "C": (4.03e-3, 2.22e-2), "E": (3.63e-4, 9.84e-3),
        "F": (4.60e-3, 1.21e-2), "G": (1.58e-3, 9.84e-3), "H": (3.29e-4, 7.80e-3)}
for k, v in _bic.items():
    RUNS[f"BIC_{k}"] = ("BIC", BIC_A100, f"BIC_{k}", f"BIC_{k}", (360.0, *v))
RUNS["BIC_D"] = ("BIC", BIC_A100, "", "", (360.0, 5.91e-4, 7.47e-3))
_sr = {"B": (7.05e-2, 2.44e-2), "C": (5.28e-2, 1.44e-2), "D": (3.67e-2, 1.07e-2), "E": (7.43e-2, 3.75e-2),
       "F": (1.86e-1, 7.38e-2), "G": (6.84e-2, 2.43e-2), "B_SF": (7.06e-2, 2.52e-2), "F_SF": (2.12e-1, 7.61e-2)}
RUNS["SR_A"] = ("SR", SR_A100, "SR_A", "SR_A", (3600.0, 1.45e-1, 5.99e-1))
RUNS["SR_A_SF"] = ("SR", SR_LIVE, "SR_A_SF", "SR_A_SF", (3600.0, 1.45e-1, 5.97e-1))
for k, v in _sr.items():
    RUNS[f"SR_{k}"] = ("SR", SR_LIVE, f"SR_{k}_FL", f"SR_{k}_FL", (3600.0, *v))
_bicu = {"A": (3.12e-1,), "C": (1.02e-1,), "EW": (3.14e-1,), "EWL1": (2.35e-1,), "EWL2": (5.60e-2,),
         "HW": (4.63e-2,), "T": (2.34e-2,), "CT": (3.81e-2,), "GT": (2.55e-2,), "PT": (2.32e-1,)}
for k, v in _bicu.items():
    RUNS[f"BICU_{k}"] = ("BICU", BIC_LIVE, f"BICU_{k}", f"BICU_{k}", (400.0, None, v[0]))

# Same-code control and area-weighted side test (R3-B1a)
RUNS["BIC_B_CTL"] = ("BIC", BIC_LIVE, "BIC_B_CTL", "BIC_B_CTL", (360.0, 3.80e-3, 7.77e-3))
RUNS["BIC_B_AW"] = ("BIC", BIC_LIVE, "BIC_B_AW", "BIC_B_AW", (360.0, 4.14e-3, 7.77e-3))
RUNS["SR_B_AW"] = ("SR", SR_LIVE, "SR_B_AW_FL", "SR_B_AW_FL", (3600.0, 6.99e-2, 2.48e-2))

CASE_REF = {"BIC": (BIC_LIVE, BIC_REF, ""), "SR": (SR_LIVE, SR_REF, "SR_B_FL"), "BICU": (BIC_LIVE, BICU_REF, "BICU_A")}
REF_WINDOW = {"BIC": (0.0, 360.0), "SR": (0.0, 3600.0), "BICU": (360.0, 600.0)}


class Case:
    """Mesh, bed and boundary data for one configuration."""

    def __init__(self, tree, stem):
        os.chdir(tree)
        name = f"fvm_pinn_config_{stem}.yaml" if stem else "fvm_pinn_config.yaml"
        self.cfg = Config(name)
        self.cfg.set("device.type", "cpu")
        self.model = FVM_SWE_PINN(self.cfg)
        self.ds = FVM_PINNDataset(self.cfg)
        self.model.set_mesh_context(h_still_cells=self.ds.get_h_still(), cell_xy=self.ds.get_cell_xy())
        self.net = self.model.get_internal_network()
        self.md = self.ds.get_mesh_data()
        self.hs = self.ds.get_h_still().to(torch.float64)
        self.xy = self.md["cell_center"].to(torch.float64)
        self.area = self.md["cell_area"].to(torch.float64)
        self.h_dry = float(self.cfg.get("physics.h_dry", 0.01))
        self.t0 = float(self.cfg.get("training.t_start", 0.0))
        self.t1 = float(self.cfg.get_required_config("training.t_end"))
        self.Adom = float(self.area.sum())
        self.inlet = [k for k, b in self.md["bc_ghost"].items() if b["type"] == "inlet-q"]
        self.exit = [k for k, b in self.md["bc_ghost"].items() if b["type"] == "exit-h"]
        if len(self.inlet) != 1:
            raise ValueError(f"expected one inlet, got {self.inlet}")
        bnd = self.md["face_right"] < 0
        ids, fl = self.md["face_bc_id"][bnd], self.md["face_left"][bnd]
        self.inlet_cells = fl[ids == self.inlet[0]].unique()
        self.open_cells = fl[torch.isin(ids, torch.tensor(self.inlet + self.exit))].unique()

    def q_in(self, t):
        b = self.md["bc_ghost"][self.inlet[0]]
        if "hydrograph" in b:
            return float(interp_hydrograph(b["hydrograph"], torch.tensor(float(t), dtype=torch.float64)))
        return float(b["value"])

    def boundary_discharge(self, Q, t):
        """Outward numerical mass flux summed by boundary id, as the FVM residual computes it."""
        md = self.md
        bnd = md["face_right"] < 0
        fl = md["face_left"][bnd]
        nx, ny = md["face_normal"][bnd, 0], md["face_normal"][bnd, 1]
        Q_L, hs_L, zb_L = Q[fl], self.hs[fl], md["bed_elev"][fl]
        Q_R, hs_R, zb_R = _build_ghost_states(Q_L, hs_L, zb_L, nx, ny, md, bnd, self.h_dry,
                                              torch.tensor(float(t), dtype=torch.float64))
        flux = roe_flux_2d(Q_L, Q_R, hs_L, hs_R, zb_L, zb_R, nx, ny, self.h_dry)[:, 0] * md["face_length"][bnd]
        ids = md["face_bc_id"][bnd]
        return {int(i): float(flux[ids == i].sum()) for i in ids.unique()}

    def balance(self, Q, dxidt, t):
        """All conservation measures for one state."""
        r = dxidt + compute_fvm_residual(Q, self.md, self.hs, self.h_dry,
                                         t=torch.tensor(float(t), dtype=torch.float64))[:, 0]
        h = Q[:, 0] + self.hs
        wet = h > self.h_dry
        qin = self.q_in(t)
        q_bnd = self.boundary_discharge(Q, t)
        q_in_num = -q_bnd[self.inlet[0]]
        q_out_num = sum(q_bnd[k] for k in self.exit)
        q_wall = sum(v for k, v in q_bnd.items() if k != self.inlet[0] and k not in self.exit)
        dvdt = float((self.area * dxidt).sum())
        total = float((self.area * r).sum())
        telescoping = abs(total - (dvdt + sum(q_bnd.values())))
        ra = r[wet].abs()
        scale = qin / self.Adom
        interior = wet.clone()
        interior[self.open_cells] = False
        mass = (self.area * r).abs()
        return dict(t=float(t), eps_inf=float(ra.max()) / scale, eps_rms=float(ra.pow(2).mean().sqrt()) / scale,
                    eps_mean=float((ra * self.area[wet]).sum() / self.area[wet].sum()) / scale,
                    i_max=int(torch.nonzero(wet).view(-1)[ra.argmax()]),
                    eps_inf_interior=float(r[interior].abs().max()) / scale,
                    inlet_share=float(mass[self.inlet_cells].sum() / mass.sum()),
                    inlet_imbalance=float((self.area * r)[self.inlet_cells].sum()) / qin,
                    eps_V=abs(total) / qin, q_in=qin, q_in_num=q_in_num, q_out_num=q_out_num, q_wall=q_wall,
                    dvdt=dvdt, V=float((self.area * h.clamp(min=0.0)).sum()), V_signed=float((self.area * h).sum()),
                    n_dry=int((~wet).sum()), telescoping=telescoping / qin)


def ckpt_for(tree, ckpt_id, case, t):
    """Same routing as the existing figure scripts and WindowTrainer.predict."""
    d = tree / "checkpoints" / ckpt_id if ckpt_id else tree / "checkpoints"
    if (d / "teacher_final.pt").exists():
        return d / "teacher_final.pt"
    if not (d / "window_000").exists():
        return d / "ckpt_final.pt"
    win = _split_windows(case.t0, case.t1, int(case.cfg.get_required_config("training.window.n_windows")),
                         float(case.cfg.get("training.window.overlap_frac", 0.10)))
    for k, (_, hi) in enumerate(win):
        if t <= hi or k == len(win) - 1:
            if k > 0 and t <= win[k - 1][1]:
                continue
            return d / f"window_{k:03d}" / "ckpt_final.pt"
    raise ValueError(t)


def load(case, path):
    if getattr(case, "loaded", None) != path:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        case.net.load_state_dict(ck["network_state"])
        case.net.set_normalisation(ck["x_mean"], ck["x_std"])
        case.loaded = path


def net_state(case, t):
    xyt = torch.column_stack([case.xy, torch.full((len(case.xy), 1), float(t), dtype=torch.float64)])
    xyt.requires_grad_(True)
    Q = case.net(xyt)
    return Q.detach(), _batch_time_grad(Q, xyt)[:, 0].detach()


def ref_fields(ref_path):
    with h5py.File(ref_path, "r") as f:
        return (f["Water_Depth_m/Times"][()].astype(float), f["Water_Depth_m/Values"][()].astype(float),
                f["Velocity_m_p_s/Values"][()].astype(float), f["Vel_Mag_m_p_s/Values"][()].astype(float))


def table_check(case, rid, spec, tree, ckpt_id, ref):
    t, want_h, want_u = spec
    times, H, _, V = ref
    k = int(np.argmin(np.abs(times - t)))
    if abs(times[k] - t) > 1e-6:
        raise SystemExit(f"{rid}: reference has no output at t={t}")
    load(case, ckpt_for(tree, ckpt_id, case, t))
    with torch.no_grad():
        P = case.model(torch.column_stack([case.xy, torch.full((len(case.xy), 1), t, dtype=torch.float64)])).numpy()
    mask = H[k] > case.h_dry
    h_err = float(np.sqrt(np.mean((P[mask, 0] - H[k][mask]) ** 2)))
    u_err = float(np.sqrt(np.mean((np.hypot(P[mask, 1], P[mask, 2]) - V[k][mask]) ** 2)))
    ok = abs(u_err / want_u - 1) < 0.01 and (want_h is None or abs(h_err / want_h - 1) < 0.01)
    return dict(t=t, L2_h=h_err, L2_u=u_err, table_h=want_h, table_u=want_u, ok=bool(ok))


def summarise(rows):
    a = {k: np.array([r[k] for r in rows]) for k in ("t", "eps_inf", "eps_rms", "eps_mean", "eps_V", "q_in", "q_in_num",
                                                       "q_out_num", "V_signed", "eps_inf_interior", "inlet_share")}
    t = a["t"]
    trap = lambda y: np.concatenate([[0.0], np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(t))])
    # V_signed: Q is compared as the network gives it, without clipping negative depths
    E = (a["V_signed"] - a["V_signed"][0] - trap(a["q_in"] - a["q_out_num"])) / trap(a["q_in"]).clip(min=1e-30)
    E[0] = 0.0
    for r, e in zip(rows, E):
        r["E_V"] = float(e)
    out = {}
    for k in ("eps_inf", "eps_rms", "eps_mean", "eps_V", "eps_inf_interior", "inlet_share"):
        out[f"{k}_median"], out[f"{k}_max"], out[f"{k}_final"] = float(np.median(a[k])), float(a[k].max()), float(a[k][-1])
    out["E_V_final"], out["E_V_maxabs"] = float(E[-1]), float(np.abs(E).max())
    out["q_in_num_ratio_final"] = float(a["q_in_num"][-1] / a["q_in"][-1])
    out["q_out_num_ratio_final"] = float(a["q_out_num"][-1] / a["q_in"][-1])
    return out


def run_network(rid, levels):
    case_name, tree, stem, ckpt_id, spec = RUNS[rid]
    case = Case(tree, stem)
    os.chdir(CASE_REF[case_name][0])
    ref = ref_fields(CASE_REF[case_name][1])
    check = table_check(case, rid, spec, tree, ckpt_id, ref)
    rows = []
    for t in np.linspace(case.t0, case.t1, levels):
        load(case, ckpt_for(tree, ckpt_id, case, float(t)))
        Q, dxi = net_state(case, float(t))
        rows.append(case.balance(Q, dxi, float(t)))
    summary = summarise(rows)
    summary["window_jump_max"] = window_jump(case, tree, ckpt_id)
    return dict(case=case_name, check=check, summary=summary, rows=rows)


def window_jump(case, tree, ckpt_id):
    """Largest volume jump between successive windows at the routing switch, over int Q_in dt."""
    d = tree / "checkpoints" / ckpt_id
    if not (d / "window_000").exists():
        return 0.0
    win = _split_windows(case.t0, case.t1, int(case.cfg.get_required_config("training.window.n_windows")),
                         float(case.cfg.get("training.window.overlap_frac", 0.10)))
    ts = np.linspace(case.t0, case.t1, 2001)
    inflow = float(np.trapezoid([case.q_in(t) for t in ts], ts))
    jumps = []
    for k in range(len(win) - 1):
        tb = float(win[k][1])
        vols = []
        for j in (k, k + 1):
            load(case, d / f"window_{j:03d}" / "ckpt_final.pt")
            Q, _ = net_state(case, tb)
            vols.append(float((case.area * (Q[:, 0] + case.hs)).sum()))
        jumps.append(abs(vols[1] - vols[0]) / inflow)
    return max(jumps)


def run_control(case_name, tree, stem, traj):
    """Positive control: the FVM solver's own trajectory, which should nearly close the balance."""
    case = Case(tree, stem)
    d = torch.load(traj, map_location="cpu", weights_only=False)
    ts, Qs = d["t_snaps"].to(torch.float64), d["Q_snaps"].to(torch.float64)
    rows = []
    for k in range(1, len(ts) - 1):
        dxi = (Qs[k + 1, :, 0] - Qs[k - 1, :, 0]) / (ts[k + 1] - ts[k - 1])
        rows.append(case.balance(Qs[k], dxi, float(ts[k])))
    return dict(case=case_name, check=None, summary=summarise(rows), rows=rows)


def run_reference(case_name, depth_from_wse=True):
    """SRH-2D output through the same balance, at interior outputs with central time differences.

    depth_from_wse: depth = SRH-2D water surface - mesh bed, which keeps the SRH-2D free surface.
    Otherwise the SRH-2D depth is used, which carries the bed difference of Case 4 into xi.
    """
    tree, ref_path, stem = CASE_REF[case_name]
    case = Case(tree, stem)
    times, H, Vel, _ = ref_fields(ref_path)
    with h5py.File(ref_path, "r") as f:
        W = f["Water_Elev_m/Values"][()].astype(float)
    bed = case.md["bed_elev"].numpy()
    rows = []
    lo, hi = REF_WINDOW[case_name]
    for k in range(1, len(times) - 1):
        if not lo <= times[k] <= hi:
            continue
        h_srh = torch.tensor(H[k], dtype=torch.float64)
        h = torch.tensor(W[k] - bed, dtype=torch.float64) if depth_from_wse else h_srh
        q = torch.tensor(Vel[k, :, :2], dtype=torch.float64) * h_srh.unsqueeze(-1)
        Q = torch.column_stack([h - case.hs, q])
        dxi = torch.tensor((H[k + 1] - H[k - 1]) / (times[k + 1] - times[k - 1]), dtype=torch.float64)
        rows.append(case.balance(Q, dxi, times[k]))
    return dict(case=case_name, check=None, summary=summarise(rows), rows=rows) if rows else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", type=int, default=100)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--out", default=str(HERE / "conservation_diagnostics.json"))
    args = ap.parse_args()
    torch.set_num_threads(4)
    torch.set_default_dtype(torch.float64)
    table = {}
    for rid in (args.only or RUNS):
        table[rid] = run_network(rid, args.levels)
        s, c = table[rid]["summary"], table[rid]["check"]
        tele = max(r["telescoping"] for r in table[rid]["rows"])
        print(f"{rid:9s} check={'ok ' if c['ok'] else 'BAD'} h={c['L2_h']:.3e}/{c['table_h']} u={c['L2_u']:.3e}/{c['table_u']:.3e} "
              f"inf med={s['eps_inf_median']:.3g} max={s['eps_inf_max']:.3g} rms={s['eps_rms_median']:.3g} "
              f"V med={s['eps_V_median']:.3g} max={s['eps_V_max']:.3g} E={s['E_V_final']:+.3g} "
              f"Qin={s['q_in_num_ratio_final']:.3f} Qout={s['q_out_num_ratio_final']:.3f} tele={tele:.1e}", flush=True)
    if args.only is None:
        for c in CASE_REF:
            res = run_reference(c)
            if res:
                table[f"{c}_SRH2D"] = res
                s = res["summary"]
                print(f"{c}_SRH2D n={len(res['rows'])} inf med={s['eps_inf_median']:.3g} max={s['eps_inf_max']:.3g} "
                      f"rms={s['eps_rms_median']:.3g} V med={s['eps_V_median']:.3g} Qin={s['q_in_num_ratio_final']:.3f} "
                      f"Qout={s['q_out_num_ratio_final']:.3f}", flush=True)
        table["SR_SRH2D_depth"] = run_reference("SR", depth_from_wse=False)
        for name, c, tree, stem, traj in (
                ("BIC_FVM", "BIC", BIC_A100, "", BIC_A100 / "checkpoints/fvm_trajectory.pt"),
                ("SR_FVM", "SR", SR_LIVE, "SR_E_FL", SR_LIVE / "checkpoints/SR_E_FL/fvm_trajectory.pt"),
                ("BICU_FVM", "BICU", BIC_LIVE, "BICU_T", BIC_LIVE / "checkpoints/BICU_T/fvm_trajectory.pt")):
            if traj.exists():
                res = run_control(c, tree, stem, traj)
                table[name] = res
                s = res["summary"]
                print(f"{name} n={len(res['rows'])} inf med={s['eps_inf_median']:.3g} final={s['eps_inf_final']:.3g} "
                      f"rms med={s['eps_rms_median']:.3g} final={s['eps_rms_final']:.3g} "
                      f"V med={s['eps_V_median']:.3g} final={s['eps_V_final']:.3g}", flush=True)
            else:
                print(name, "missing", traj, flush=True)
    Path(args.out).write_text(json.dumps(table, indent=1))


if __name__ == "__main__":
    main()
