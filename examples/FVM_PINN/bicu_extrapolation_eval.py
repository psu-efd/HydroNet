"""
R2-4 temporal extrapolation test on Case 5 (unsteady block-in-channel).

Networks trained on [360, S] s (S = 450, 540) are evaluated against SRH-2D
at every 10 s output, inside the training window and after it, (S, 600] s.

    BICU-TX{S}   distilled from the FVM solver on [360, S] only
    BICU-CTX{S}  from TX{S}: FVM-PINN residual on [360, 600], gauges up to S
    BICU-GTX{S}  from TX{S}: gauges up to S, no physics loss (lambda_fvm = 0)

References: BICU-T/CT/GT trained on the full window, the FVM solver run
alone, and persistence (the SRH-2D state at t = S held fixed), which is the
trivial forecast a model has to beat.

Writes bicu_extrapolation_eval.json next to this script.

Usage
-----
    python examples/FVM_PINN/bicu_extrapolation_eval.py
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

import fig_bicu_unsteady as fb

logger = logging.getLogger(__name__)
SPLITS = [450.0, 540.0]
NEW = {
    f"BICU_{k}X{int(s)}": (f"fvm_pinn_config_BICU_{k}X{int(s)}.yaml", f"BICU-{k}X{int(s)}", "k", "-")
    for s in SPLITS for k in ("T", "CT", "GT")
}
FULL = ["BICU_T", "BICU_CT", "BICU_GT"]


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    fb.RUNS.update(NEW)
    _, ds, _ = fb.load_dataset(fb.RUNS["BICU_A"][0])
    hs = ds.get_h_still().numpy()
    T_srh, H_srh, V_srh = fb.srh2d()
    U_srh = np.hypot(V_srh[..., 0], V_srh[..., 1])
    i_fine = [int(np.argmin(np.abs(fb.T_FINE - t))) for t in T_srh]

    def metrics(h, u):
        """Per SRH-2D output time: L2(|u|), L2(h), wet-cell mean |u|."""
        rows = []
        for j in range(len(T_srh)):
            wet = H_srh[j] > fb.H_WET
            rows.append((fb.l2(u[j], U_srh[j], wet), fb.l2(h[j], H_srh[j], wet), float(u[j][wet].mean())))
        return np.array(rows)

    per = {}
    for rid in list(NEW) + FULL:
        h, u = fb.to_phys(fb.predict_run(rid)[i_fine], hs)
        per[rid] = metrics(h, u)
    h, u = fb.to_phys(fb.fvm_trajectory(ds), hs)   # at the 10 s outputs, 360-600 s
    per["FVM solver"] = metrics(h, u)

    out = {"times": T_srh.tolist(), "srh_mean_u": [float(U_srh[j][H_srh[j] > fb.H_WET].mean())
                                                   for j in range(len(T_srh))], "splits": {}}
    for s in SPLITS:
        j_s = int(np.argmin(np.abs(T_srh - s)))
        per[f"Persistence {int(s)}"] = metrics(np.repeat(H_srh[j_s:j_s + 1], len(T_srh), 0),
                                               np.repeat(U_srh[j_s:j_s + 1], len(T_srh), 0))
        inside = (T_srh > 360.0 + 1e-6) & (T_srh <= s + 1e-6)
        after = T_srh > s + 1e-6
        rows = {}
        names = [f"BICU_{k}X{int(s)}" for k in ("T", "CT", "GT")] + FULL + ["FVM solver", f"Persistence {int(s)}"]
        for n in names:
            m = per[n]
            rows[n] = {
                "u_inside": float(m[inside, 0].mean()), "u_after": float(m[after, 0].mean()),
                "u_final": float(m[-1, 0]), "h_after": float(m[after, 1].mean()),
                "mean_u_after": float(m[after, 2].mean()),
                "u_by_time": {f"{t:.0f}": float(v) for t, v in zip(T_srh, m[:, 0])},
            }
        out["splits"][str(int(s))] = rows
        logger.info("\nSplit %d s: L2(|u|) inside / after / at 600 s, L2(h) after, mean|u| after", s)
        for n, r in rows.items():
            logger.info("  %-16s %.4f  %.4f  %.4f   %.2e   %.3f", n, r["u_inside"], r["u_after"],
                        r["u_final"], r["h_after"], r["mean_u_after"])
    Path(__file__).with_suffix(".json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
