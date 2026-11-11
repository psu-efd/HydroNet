"""Off-mesh and off-snapshot evaluation of the trained networks (reviewer item R2-3).

time  : on the training mesh, at SRH-2D output times not used for training. Baseline: linear-in-time
        interpolation of the anchor snapshots.
space : at the cell centres of the second mesh (make_second_mesh.py), against SRH-2D run on that mesh.
        Depth uses the second mesh's own bed, h = xi + max(wse_still - z, 0). This avoids the
        nearest-training-cell lookup of h_still in the model's public forward. Baselines: the training-mesh
        SRH-2D field interpolated linearly in space onto the second mesh, and the same network on the
        training mesh.
L2 is the root-mean-square difference over wet reference cells, as in the manuscript tables.
Writes offmesh_eval.json next to this script. The space part runs only once the SRH-2D file exists.
"""
import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

import conservation_diagnostics as cd

HERE = cd.HERE
ANCHORS = {"BIC": [72.0, 144.0, 216.0, 288.0, 360.0], "SR": [720.0, 1440.0, 2160.0, 2880.0, 3600.0]}
TIME_REF = {"BIC": (cd.BIC_LIVE, cd.BICU_REF), "SR": (cd.SR_LIVE, cd.SR_REF)}
ANCHOR_REF = {"BIC": (cd.BIC_LIVE, cd.BIC_REF), "SR": (cd.SR_LIVE, cd.SR_REF)}
MESH2 = {"BIC": (cd.BIC_LIVE, "data_mesh2/block_in_channel_mesh2.srhhydro", "data_mesh2/block_in_channel_mesh2_XMDFC.h5"),
         "SR": (cd.SR_LIVE, "data_mesh2/savana_mesh2.srhhydro", "data_mesh2/Savana_mesh2_XMDFC.h5")}
MESH1_STEM = {"BIC": "", "SR": "SR_B_FL"}


def fields(path):
    with h5py.File(path, "r") as f:
        return {k: f[n].astype(float)[()] for k, n in (("H", "Water_Depth_m/Values"), ("W", "Water_Elev_m/Values"),
                                                      ("V", "Vel_Mag_m_p_s/Values"), ("Vel", "Velocity_m_p_s/Values"))} | {
            "t": f["Water_Depth_m/Times"][()].astype(float)}


def predict(case, tree, ckpt_id, xy, hs, t):
    cd.load(case, cd.ckpt_for(tree, ckpt_id, case, float(t)))
    xyt = torch.column_stack([xy, torch.full((len(xy), 1), float(t), dtype=torch.float64)])
    with torch.no_grad():
        Q = case.net(xyt)
    h = Q[:, 0] + hs
    speed = torch.hypot(Q[:, 1], Q[:, 2]) / h.clamp(min=1e-6)
    return h.numpy(), speed.numpy(), Q[:, 0].numpy()


def rms(a):
    return float(np.sqrt(np.mean(a ** 2)))


def score(h, sp, xi, ref, k, h_dry, wse):
    m = ref["H"][k] > h_dry
    out = dict(t=float(ref["t"][k]), L2_h=rms(h[m] - ref["H"][k][m]), L2_u=rms(sp[m] - ref["V"][k][m]),
               mean_u=float(sp[m].mean()), mean_u_ref=float(ref["V"][k][m].mean()))
    if wse is not None:
        out["L2_ws"] = rms(xi[m] + wse - ref["W"][k][m])
    return out


def interp_anchor(ref_anchor, t, field, ic):
    """Linear in time between anchors; the initial state before the first anchor."""
    ts = ref_anchor["t"]
    if t >= ts[-1]:
        return ref_anchor[field][-1]
    j = int(np.searchsorted(ts, t))
    if j == 0:
        lo, tlo = ic, 0.0
    else:
        lo, tlo = ref_anchor[field][j - 1], ts[j - 1]
    w = (t - tlo) / (ts[j] - tlo)
    return (1 - w) * lo + w * ref_anchor[field][j]


def time_part(cname, runs):
    tree_ref, ref_path = TIME_REF[cname]
    out = {}
    os.chdir(tree_ref)
    ref = fields(ref_path)
    anc_tree, anc_path = ANCHOR_REF[cname]
    anc = fields(anc_path)
    keep = np.array([np.isclose(t, ANCHORS[cname]).any() for t in anc["t"]])
    anc = {k: (v[keep] if k != "t" else v[keep]) for k, v in anc.items()}
    T = ANCHORS[cname][-1]
    idx = [k for k, t in enumerate(ref["t"]) if t <= T and not np.isclose(t, ANCHORS[cname]).any()]
    common = [k for k, t in enumerate(ref["t"]) if np.isclose(t, T)][0]
    out["reference_check"] = dict(t=T, rms_h=rms(ref["H"][common] - anc["H"][-1]), rms_V=rms(ref["V"][common] - anc["V"][-1]),
                                  rms_h_72=None)
    for rid in runs:
        _, tree, stem, ck, _ = cd.RUNS[rid]
        case = cd.Case(tree, stem)
        hs = case.hs
        wse = float(case.cfg.get("physics.wse_still", 0.0)) if cname == "SR" else None
        held = [score(*predict(case, tree, ck, case.xy, hs, ref["t"][k]), ref, k, case.h_dry, wse) for k in idx]
        out[rid] = dict(held_mean_L2_u=float(np.mean([r["L2_u"] for r in held])), held_mean_L2_h=float(np.mean([r["L2_h"] for r in held])),
                        held_mean_u=float(np.mean([r["mean_u"] for r in held])), held=held)
    # interpolation baseline (speed and depth interpolated directly)
    case = cd.Case(cd.RUNS[runs[0]][1], cd.RUNS[runs[0]][2])
    ic_h, ic_v = case.hs.numpy(), np.zeros(len(case.hs))
    base = []
    for k in idx:
        t = ref["t"][k]
        h, v = interp_anchor(anc, t, "H", ic_h), interp_anchor(anc, t, "V", ic_v)
        m = ref["H"][k] > case.h_dry
        base.append(dict(t=float(t), L2_h=rms(h[m] - ref["H"][k][m]), L2_u=rms(v[m] - ref["V"][k][m])))
    out["interpolation"] = dict(held_mean_L2_u=float(np.mean([r["L2_u"] for r in base])), held_mean_L2_h=float(np.mean([r["L2_h"] for r in base])), held=base)
    out["mean_u_ref"] = float(np.mean([ref["V"][k][ref["H"][k] > case.h_dry].mean() for k in idx]))
    out["n_times"] = len(idx)
    return out


def space_part(cname, runs):
    tree2, hydro, h5 = MESH2[cname]
    if not (tree2 / h5).exists():
        return None
    from HydroNet import Config, FVM_PINNDataset
    os.chdir(tree2)
    ref2 = fields(h5)
    stem = cd.CASE_REF[cname][2] or None
    name = f"fvm_pinn_config_{stem}.yaml" if stem else "fvm_pinn_config.yaml"
    c = Config(name)
    c.set("device.type", "cpu")
    c.set("data.srhhydro", hydro)
    c.set("data.srh2d_h5_file", h5)
    ds2 = FVM_PINNDataset(c)
    xy2, hs2 = ds2.get_cell_xy().to(torch.float64), ds2.get_h_still().to(torch.float64)
    h_dry = float(c.get("physics.h_dry", 0.01))
    wse = float(c.get("physics.wse_still", 0.0)) if cname == "SR" else None
    anc_tree, anc_path = ANCHOR_REF[cname]
    os.chdir(anc_tree)
    anc = fields(anc_path)
    out = {}
    tidx = [k for k, t in enumerate(ref2["t"]) if np.isclose(t, ANCHORS[cname]).any()]
    # Solver and interpolation floor: SRH-2D on the training mesh, interpolated onto the second mesh
    case1 = cd.Case(cd.RUNS[runs[0]][1], cd.RUNS[runs[0]][2])
    x1, x2 = case1.xy.numpy(), xy2.numpy()

    def to2(v):
        a = LinearNDInterpolator(x1, v)(x2)
        miss = np.isnan(a)
        a[miss] = NearestNDInterpolator(x1, v)(x2[miss])
        return a
    floor = []
    for k in tidx:
        j = int(np.argmin(np.abs(anc["t"] - ref2["t"][k])))
        m = ref2["H"][k] > h_dry
        floor.append(dict(t=float(ref2["t"][k]), L2_h=rms(to2(anc["H"][j])[m] - ref2["H"][k][m]), L2_u=rms(to2(anc["V"][j])[m] - ref2["V"][k][m])))
    out["SRH2D_mesh1_interpolated"] = dict(final=floor[-1], rows=floor)
    for rid in runs:
        _, tree, st, ck, _ = cd.RUNS[rid]
        case = cd.Case(tree, st)
        rows = [score(*predict(case, tree, ck, xy2, hs2, ref2["t"][k]), ref2, k, h_dry, wse) for k in tidx]
        # same network on the training mesh against the training-mesh SRH-2D
        last = int(np.argmin(np.abs(anc["t"] - ANCHORS[cname][-1])))
        on = score(*predict(case, tree, ck, case.xy, case.hs, anc["t"][last]), anc, last, case.h_dry, wse)
        # the network on the second mesh against the training-mesh field interpolated there
        h, sp, _ = predict(case, tree, ck, xy2, hs2, anc["t"][last])
        m = to2(anc["H"][last]) > h_dry
        vs1 = dict(L2_h=rms(h[m] - to2(anc["H"][last])[m]), L2_u=rms(sp[m] - to2(anc["V"][last])[m]))
        out[rid] = dict(final=rows[-1], on_mesh_final=on, vs_interpolated_mesh1=vs1, rows=rows)
    return out


if __name__ == "__main__":
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float64)
    bic = [f"BIC_{k}" for k in "ABCEFGHD"]
    sr = [f"SR_{k}" for k in ("A", "B", "C", "D", "E", "F", "G", "B_SF", "F_SF")]
    res = {"time": {"BIC": time_part("BIC", bic), "SR": time_part("SR", sr)},
           "space": {"BIC": space_part("BIC", bic), "SR": space_part("SR", sr)}}
    (HERE / "offmesh_eval.json").write_text(json.dumps(res, indent=1))
    for c, d in res["time"].items():
        print(c, "reference check", d["reference_check"], "n held-out", d["n_times"], "mean |u| ref %.3f" % d["mean_u_ref"])
        for k, v in d.items():
            if isinstance(v, dict) and "held_mean_L2_u" in v:
                print(f"  {k:14s} held-out mean L2(u) {v['held_mean_L2_u']:.4f} L2(h) {v['held_mean_L2_h']:.4f} mean|u| {v.get('held_mean_u', float('nan')):.3f}")
    for c, d in res["space"].items():
        print(c, "space:", "waiting for the SRH-2D file" if d is None else "done")
