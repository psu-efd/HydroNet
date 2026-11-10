"""
Front tracking on the Case 1 dam break against the exact Riemann solution
(R3-B1c): shock, rarefaction head and rarefaction tail positions through time.

The network is queried on a fine x grid at 20 uniform times. Positions are
compared with the exact trajectories, and speeds come from a least-squares
line through the positions.

Usage (from this folder):
    python front_tracking.py --config fvm_pinn_config.yaml \
        --runs FVM=path/ckpt_final.pt SF=path/ckpt_final.pt
Writes front_tracking.json and front_tracking.npz next to this script.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import brentq

project_root = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from HydroNet import Config, FVM_SWE_PINN

G = 9.81


def exact_speeds(h_L, h_R):
    """Star state and the shock, head and tail speeds of the Stoker solution."""
    c_L = np.sqrt(G * h_L)

    def f(h):
        return 2 * (c_L - np.sqrt(G * h)) - (h - h_R) * np.sqrt(0.5 * G * (h + h_R) / (h * h_R))

    h_s = brentq(f, h_R * (1 + 1e-9), h_L)
    u_s = 2 * (c_L - np.sqrt(G * h_s))
    return {"h_star": h_s, "u_star": u_s, "shock": h_s * u_s / (h_s - h_R),
            "head": -c_L, "tail": u_s - np.sqrt(G * h_s)}


def first_below(x, y, level):
    """First x (scanning left to right) at which y falls below level, or NaN."""
    idx = np.where((y[:-1] >= level) & (y[1:] < level))[0]
    if len(idx) == 0:
        return np.nan
    i = idx[0]
    return x[i] + (level - y[i]) / (y[i + 1] - y[i]) * (x[i + 1] - x[i])


def last_below(x, y, level):
    """Last downward crossing of level, or NaN."""
    idx = np.where((y[:-1] >= level) & (y[1:] < level))[0]
    if len(idx) == 0:
        return np.nan
    i = idx[-1]
    return x[i] + (level - y[i]) / (y[i + 1] - y[i]) * (x[i + 1] - x[i])


def track(h, x, x_dam, h_L, h_R, h_s):
    dn = x > x_dam
    grad = np.abs(np.gradient(h, x))
    d_h = h_s - h_R
    x_up, x_lo = (last_below(x[dn], h[dn], h_R + f * d_h) for f in (0.9, 0.1))
    return {
        "shock_grad": float(x[dn][np.argmax(grad[dn])]),
        "shock_mid": float(last_below(x[dn], h[dn], h_R + 0.5 * d_h)),
        "shock_width": float(x_lo - x_up),
        "head": float(first_below(x, h, 0.99 * h_L)),
        "tail": float(first_below(x, h, 1.01 * h_s)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="fvm_pinn_config.yaml")
    ap.add_argument("--runs", nargs="+", required=True, help="label=checkpoint ...")
    ap.add_argument("--n-times", type=int, default=20)
    args = ap.parse_args()

    config = Config(args.config)
    config.set("device.type", "cpu")
    case = config.get_required_config("case")
    L, x_dam, h_L, h_R = (float(case[k]) for k in ("L", "x_dam", "h_L", "h_R"))
    dx = L / int(case["n_cells"])
    y = 0.5 * float(case["width"])
    T = float(config.get_required_config("training.t_end"))
    ex = exact_speeds(h_L, h_R)

    x = np.linspace(0.0, L, 8001)
    times = np.linspace(T / args.n_times, T, args.n_times)
    out = {"exact": {k: float(v) for k, v in ex.items()}, "dx": dx, "runs": {}}
    arrays = {"times": times}
    for spec in args.runs:
        label, path = spec.split("=", 1)
        model = FVM_SWE_PINN(config)
        net = model.get_internal_network()
        ck = torch.load(path, map_location="cpu", weights_only=False)
        net.load_state_dict(ck["network_state"])
        net.set_normalisation(ck["x_mean"], ck["x_std"])
        rows = []
        for t in times:
            xyt = torch.column_stack([torch.tensor(x), torch.full((x.size,), y, dtype=torch.float64),
                                      torch.full((x.size,), float(t), dtype=torch.float64)])
            with torch.no_grad():
                h = net(xyt)[:, 0].numpy()     # wse_still = 0, so xi = h
            rows.append(track(h, x, x_dam, h_L, h_R, ex["h_star"]))
        res = {}
        for key, speed in (("shock_grad", "shock"), ("shock_mid", "shock"), ("head", "head"), ("tail", "tail")):
            pos = np.array([r[key] for r in rows])
            exact = x_dam + ex[speed] * times
            ok = np.isfinite(pos)
            res[key] = {
                "n_valid": int(ok.sum()),
                "speed": float(np.polyfit(times[ok], pos[ok], 1)[0]) if ok.sum() > 1 else None,
                "speed_exact": float(ex[speed]),
                "mean_abs_error_m": float(np.abs(pos[ok] - exact[ok]).mean()) if ok.any() else None,
                "max_abs_error_m": float(np.abs(pos[ok] - exact[ok]).max()) if ok.any() else None,
                "final_error_m": float(pos[-1] - exact[-1]),
            }
            arrays[f"{label}_{key}"] = pos
        width = np.array([r["shock_width"] for r in rows])
        res["shock_width_cells_mean"] = float(np.nanmean(width) / dx)
        res["shock_width_cells_final"] = float(width[-1] / dx)
        out["runs"][label] = res

    here = Path(__file__).resolve().parent
    (here / "front_tracking.json").write_text(json.dumps(out, indent=2))
    np.savez(here / "front_tracking.npz", **arrays)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
