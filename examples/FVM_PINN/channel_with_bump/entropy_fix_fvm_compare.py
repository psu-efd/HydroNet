"""
FVM solver on Case 2 with and without the Harten-Hyman entropy fix (R3-B1c).

Integrates the same flat-WSE start to t_end twice and compares each result
with the FullSWOF analytical steady state, overall and around the sonic point
at the crest. Writes entropy_fix_fvm_compare.json and .npz next to this script.

Usage (from this folder):
    python entropy_fix_fvm_compare.py [--config fvm_pinn_config.yaml]
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

project_root = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from HydroNet import Config, FVM_PINNDataset
from HydroNet.models.FVM_PINN._internal.fvm import run_fvm_rk2

G = 9.81
CREST = (9.5, 10.5)     # window around the sonic point at x = 10 m


def summarise(Q, h_still, xc, dat, q_in):
    h = Q[:, 0] + h_still
    u = Q[:, 1] / np.maximum(h, 1e-6)
    h_ex = np.interp(xc, dat[:, 0], dat[:, 1])
    u_ex = q_in / h_ex
    order = np.argsort(xc)
    x, h, u, h_ex, u_ex = xc[order], h[order], u[order], h_ex[order], u_ex[order]
    lam = u - np.sqrt(G * h)
    in_crest = (x >= CREST[0]) & (x <= CREST[1])
    # Excess cell-to-cell step in h relative to the exact profile, at the crest
    excess = np.abs(np.diff(h) - np.diff(h_ex))
    excess_mask = in_crest[:-1] & in_crest[1:]
    # Sonic face: u - c changes sign from negative to positive
    sonic = np.where((lam[:-1] < 0) & (lam[1:] >= 0))[0]
    return {
        "L2_h": float(np.sqrt(np.mean((h - h_ex) ** 2))),
        "L2_u": float(np.sqrt(np.mean((u - u_ex) ** 2))),
        "max_h": float(np.abs(h - h_ex).max()),
        "crest_L2_h": float(np.sqrt(np.mean((h[in_crest] - h_ex[in_crest]) ** 2))),
        "crest_max_h": float(np.abs(h[in_crest] - h_ex[in_crest]).max()),
        "crest_max_step_excess": float(excess[excess_mask].max()),
        "sonic_faces": [
            {"x": float(0.5 * (x[i] + x[i + 1])), "lam_left": float(lam[i]),
             "lam_right": float(lam[i + 1]), "h_left": float(h[i]), "h_right": float(h[i + 1]),
             "h_exact_left": float(h_ex[i]), "h_exact_right": float(h_ex[i + 1])}
            for i in sonic
        ],
        "q_min": float(Q[:, 1].min()),
        "q_max": float(Q[:, 1].max()),
    }, dict(x=x, h=h, u=u, h_exact=h_ex, u_exact=u_ex)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="fvm_pinn_config.yaml")
    args = ap.parse_args()

    config = Config(args.config)
    config.set("device.type", "cpu")
    ds = FVM_PINNDataset(config)
    md, h_still = ds.get_mesh_data(), ds.get_h_still()
    Q0 = ds.get_ic_data()["U_true"]
    t0, t1 = float(config.get("training.t_start", 0.0)), float(config.get_required_config("training.t_end"))
    cfl = float(config.get("training.teacher.cfl", 0.3))
    h_dry = float(config.get("physics.h_dry", 0.01))
    snaps = torch.linspace(t0, t1, 51, dtype=torch.float64, device=h_still.device)
    q_in = next(float(v["value"]) for v in config.get_required_config("boundary_conditions").values()
                if isinstance(v, dict) and v.get("type") == "inlet-q")
    dat = np.loadtxt("data/analytical_solution_from_FullSWOF.dat", skiprows=1)
    xc = md["cell_center"][:, 0].cpu().numpy()
    hs = h_still.cpu().numpy()

    out, arrays = {}, {}
    for name, fix in (("no_fix", False), ("harten_hyman", True)):
        _, Q = run_fvm_rk2(Q0, md, h_still, snaps, cfl=cfl, h_dry=h_dry, entropy_fix=fix)
        out[name], fields = summarise(Q[-1].cpu().numpy(), hs, xc, dat, q_in)
        arrays.update({f"{k}_{name}": v for k, v in fields.items()})

    here = Path(__file__).resolve().parent
    (here / "entropy_fix_fvm_compare.json").write_text(json.dumps(out, indent=2))
    np.savez(here / "entropy_fix_fvm_compare.npz", **arrays)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
