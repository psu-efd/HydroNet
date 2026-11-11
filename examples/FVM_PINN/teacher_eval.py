"""FVM teacher trajectory against SRH-2D, and as a space-time interpolant (reviewer item R3-C6).

The cached 31-snapshot teacher trajectories are compared with SRH-2D at the five anchor times, at the
held-out SRH-2D output times (linear in time between snapshots) and on the second mesh (xi and speed linear in space,
depth from the second-mesh bed as for the networks). L2 is the RMS over wet reference cells, as in the tables.
Writes teacher_eval.json next to this script.
"""
import json

import numpy as np
import torch
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

import conservation_diagnostics as cd
import offmesh_eval as oe
from HydroNet import Config, FVM_PINNDataset

TRAJ = {"BIC": (cd.BIC_A100, "", cd.BIC_A100 / "checkpoints/fvm_trajectory.pt"),
        "SR": (cd.SR_LIVE, "SR_E_FL", cd.SR_LIVE / "checkpoints/SR_E_FL/fvm_trajectory.pt")}


def teacher(tr, hs, t):
    ts, Q = tr["t_snaps"].numpy(), tr["Q_snaps"].numpy()
    j = min(max(int(np.searchsorted(ts, t)), 1), len(ts) - 1)
    w = np.clip((t - ts[j - 1]) / (ts[j] - ts[j - 1]), 0.0, 1.0)
    q = (1 - w) * Q[j - 1] + w * Q[j]
    h = q[:, 0] + hs
    return h, np.hypot(q[:, 1], q[:, 2]) / np.maximum(h, 1e-6), q[:, 0]


def run(cname):
    tree, stem, path = TRAJ[cname]
    tr = torch.load(path, map_location="cpu", weights_only=False)
    case = cd.Case(tree, stem)
    hs = case.hs.numpy()
    wse = float(case.cfg.get("physics.wse_still", 0.0)) if cname == "SR" else None
    a_tree, a_path = oe.ANCHOR_REF[cname]
    import os
    os.chdir(a_tree)
    anc = oe.fields(a_path)
    keep = np.array([np.isclose(t, oe.ANCHORS[cname]).any() for t in anc["t"]])
    anc = {k: v[keep] for k, v in anc.items()}
    out = {"t_snaps": [float(tr["t_snaps"][0]), float(tr["t_snaps"][-1])], "meta": {k: str(v) for k, v in tr["meta"].items()}}
    out["anchors"] = [oe.score(*teacher(tr, hs, anc["t"][k]), anc, k, case.h_dry, wse) for k in range(len(anc["t"]))]
    r_tree, r_path = oe.TIME_REF[cname]
    os.chdir(r_tree)
    ref = oe.fields(r_path)
    T = oe.ANCHORS[cname][-1]
    idx = [k for k, t in enumerate(ref["t"]) if t <= T and not np.isclose(t, oe.ANCHORS[cname]).any()]
    held = [oe.score(*teacher(tr, hs, ref["t"][k]), ref, k, case.h_dry, wse) for k in idx]
    out["held"] = dict(n=len(held), mean_L2_u=float(np.mean([r["L2_u"] for r in held])), mean_L2_h=float(np.mean([r["L2_h"] for r in held])), rows=held)
    t2, hydro, h5 = oe.MESH2[cname]
    os.chdir(t2)
    ref2 = oe.fields(h5)
    st = cd.CASE_REF[cname][2] or None
    c = Config(f"fvm_pinn_config_{st}.yaml" if st else "fvm_pinn_config.yaml")
    c.set("device.type", "cpu"); c.set("data.srhhydro", hydro); c.set("data.srh2d_h5_file", h5)
    ds2 = FVM_PINNDataset(c)
    x1, x2 = case.xy.numpy(), ds2.get_cell_xy().numpy()
    h_dry = float(c.get("physics.h_dry", 0.01))
    hs2 = ds2.get_h_still().numpy()

    def to2(v):
        a = LinearNDInterpolator(x1, v)(x2)
        miss = np.isnan(a)
        a[miss] = NearestNDInterpolator(x1, v)(x2[miss])
        return a
    rows = []
    for k, t in enumerate(ref2["t"]):
        if np.isclose(t, oe.ANCHORS[cname]).any():
            _, sp, xi = teacher(tr, hs, t)
            xi2 = to2(xi)
            m = ref2["H"][k] > h_dry
            r = dict(t=float(t), L2_h=oe.rms(xi2[m] + hs2[m] - ref2["H"][k][m]), L2_u=oe.rms(to2(sp)[m] - ref2["V"][k][m]))
            if wse is not None:
                r["L2_ws"] = oe.rms(xi2[m] + wse - ref2["W"][k][m])
            rows.append(r)
    out["mesh2"] = dict(final=rows[-1], rows=rows)
    return out


if __name__ == "__main__":
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float64)
    res = {c: run(c) for c in TRAJ}
    (cd.HERE / "teacher_eval.json").write_text(json.dumps(res, indent=1))
    prev = json.loads((cd.HERE / "offmesh_eval.json").read_text())
    for c, net in (("BIC", "BIC_D"), ("SR", "SR_E")):
        r = res[c]
        print(c, r["t_snaps"], r["meta"])
        for a in r["anchors"]:
            print("  anchor t=%g L2h=%.4e L2u=%.4e%s" % (a["t"], a["L2_h"], a["L2_u"], " L2ws=%.4e" % a["L2_ws"] if "L2_ws" in a else ""), "mean|u| %.3f ref %.3f" % (a["mean_u"], a["mean_u_ref"]))
        t, s = prev["time"][c], prev["space"][c]
        print("  held n=%d teacher L2u=%.4e L2h=%.4e | %s L2u=%.4e L2h=%.4e | snapshot-interp L2u=%.4e" % (r["held"]["n"], r["held"]["mean_L2_u"], r["held"]["mean_L2_h"], net, t[net]["held_mean_L2_u"], t[net]["held_mean_L2_h"], t["interpolation"]["held_mean_L2_u"]))
        print("  mesh2 final teacher", r["mesh2"]["final"], "|", net, {k: v for k, v in s[net]["final"].items() if k.startswith("L2")}, "| SRH mesh1", s["SRH2D_mesh1_interpolated"]["final"])
