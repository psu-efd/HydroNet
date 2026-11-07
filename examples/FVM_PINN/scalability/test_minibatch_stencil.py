"""Check that the mini-batch FVM residual equals the full-mesh residual at the sampled cells.

Usage (from examples/FVM_PINN/savannah_river or any case folder):
    python ../scalability/test_minibatch_stencil.py --config fvm_pinn_config_SR_A.yaml
Exits non-zero on a mismatch.
"""
import argparse
import sys

import torch

from HydroNet import Config, FVM_PINNDataset
from HydroNet.models.FVM_PINN._internal.fvm.riemann_solver import compute_fvm_residual
from HydroNet.models.FVM_PINN._internal.pinn.loss import (
    FVMPINNLoss, LossConfig, _build_stencil, _local_mesh_data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--fracs", type=float, nargs="+", default=[0.02, 0.1, 0.25, 0.5])
    ap.add_argument("--trials", type=int, default=5)
    a = ap.parse_args()
    torch.manual_seed(0)
    cfg = Config(a.config)
    cfg.set("device.type", "cpu")
    ds = FVM_PINNDataset(cfg)
    md = {k: (v.double() if torch.is_tensor(v) and v.is_floating_point() else v)
          for k, v in ds.get_mesh_data().items()}
    hs = ds.get_h_still().double()
    n = md["n_cells"]
    loss = FVMPINNLoss(LossConfig(), md, hs)
    inlet = loss.inlet_cells
    print(f"{n} cells, {md['n_faces']} faces, {0 if inlet is None else len(inlet)} inlet cells")

    # A smooth, wet, moving state: xi small, momentum O(1) with spatial variation
    xy = md["cell_center"]
    s = (xy - xy.mean(0)) / xy.std(0)
    Q = torch.stack([0.05 * torch.sin(s[:, 0]), 0.5 + 0.2 * torch.cos(s[:, 1]),
                     0.1 * torch.sin(s[:, 0] + s[:, 1])], -1)
    R_full = compute_fvm_residual(Q, md, hs, 0.1, t=torch.tensor(100.0, dtype=torch.float64))
    scale = R_full.abs().max().item()

    worst, worst_old = 0.0, 0.0
    for frac in a.fracs:
        for _ in range(a.trials):
            mask = torch.zeros(n, dtype=torch.bool)
            mask[torch.randperm(n)[: max(1, int(frac * n))]] = True
            for always, tag in ((inlet, "new"), (None, "no-inlet")):
                idx, ev = _build_stencil(mask, md["face_left"], md["face_right"], always_include=always)
                assert torch.equal(idx[ev], mask.nonzero().view(-1))
                R = compute_fvm_residual(Q[idx], _local_mesh_data(md, idx, n), hs[idx], 0.1,
                                         t=torch.tensor(100.0, dtype=torch.float64))
                err = (R[ev] - R_full[mask]).abs().max().item() / scale
                if tag == "new":
                    worst = max(worst, err)
                else:
                    worst_old = max(worst_old, err)
    print(f"max relative mismatch at sampled cells: with inlet cells kept {worst:.2e}; "
          f"without (previous behaviour) {worst_old:.2e}")
    ok = worst < 1e-12
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
