"""
Mass-balance diagnostics for trained FVM-PINN models.

For each requested time, reports the discharge entering through the upstream
boundary, leaving through the other boundaries, the rate of change of stored
volume, the resulting global imbalance, and the discharge profile along the
channel (across nested cuts from ``MassBalance``). Everything is evaluated
with the same Roe fluxes as the training residual, so a physically plausible
solution shows a small imbalance and — once the flow settles — a flat
discharge profile. Optional reference states (e.g. SRH-2D snapshots) are
passed through the same fluxes for comparison.
"""

from typing import Dict, Optional, Sequence

import numpy as np
import torch

from ._internal.fvm.conservation import MassBalance
from ._internal.fvm.riemann_solver import compute_fvm_residual
from .data import FVM_PINNDataset
from .model import FVM_SWE_PINN


def mass_balance_report(
    model: FVM_SWE_PINN,
    dataset: FVM_PINNDataset,
    times: Sequence[float],
    ref_states: Optional[Dict[str, np.ndarray]] = None,
    n_regions: int = 10,
    upstream_bc: Optional[Sequence[int]] = None,
) -> dict:
    """
    Mass balance of the network solution at ``times``.

    Parameters
    ----------
    model, dataset : trained model (normalisation set) and its dataset
    times          : evaluation times [s]
    ref_states     : optional {"h", "u", "v"}: arrays [len(times), n_cells]
                     of a reference solution at the same times
    n_regions      : number of nested regions (cuts = n_regions - 1)
    upstream_bc    : BC ids of the upstream end (default: inlet-q ids)

    Returns
    -------
    JSON-serialisable dict with ``s_cut`` [m], ``Q_prescribed`` [m³/s or
    None] and ``per_time`` entries: Q_in, Q_out, dVdt, imbalance,
    imbalance_pct, Q_cut (and ref_* when ``ref_states`` is given).
    """
    md = dataset.get_mesh_data()
    h_still = dataset.get_h_still()
    cell_xy = dataset.get_cell_xy()
    area = md["cell_area"]
    h_dry = float(getattr(dataset, "h_dry", 1e-2))
    net = model.get_internal_network()
    net.eval()

    mb = MassBalance(md, upstream_bc, n_regions)
    q_prescribed = [
        float(md["bc_ghost"][b]["value"]) for b in mb.upstream_bc
        if md.get("bc_ghost", {}).get(b, {}).get("type") == "inlet-q"
    ]
    q_prescribed = sum(q_prescribed) if q_prescribed else None

    def fluxes(Q):
        _, F = compute_fvm_residual(Q, md, h_still, h_dry, return_face_flux=True)
        q_in, q_out = mb.boundary_discharge(F[..., 0])
        return q_in, q_out, mb.cut_discharge(F[..., 0])

    per_time = []
    for k, t in enumerate(times):
        xyt = torch.cat([cell_xy, torch.full_like(cell_xy[:, :1], float(t))], dim=-1)
        xyt.requires_grad_(True)
        Q = net(xyt)
        dxi_dt = torch.autograd.grad(Q[:, 0].sum(), xyt)[0][:, 2]
        Q = Q.detach()
        with torch.no_grad():
            q_in, q_out, q_cut = fluxes(Q)
            dvdt = (area * dxi_dt).sum()
        imbalance = dvdt + q_out - q_in            # = Σ A (∂ξ/∂t + R_ξ)
        scale = q_prescribed if q_prescribed else max(abs(float(q_in)), 1e-12)
        entry = {
            "t": float(t),
            "Q_in": float(q_in), "Q_out": float(q_out), "dVdt": float(dvdt),
            "imbalance": float(imbalance),
            "imbalance_pct": 100.0 * float(imbalance) / scale,
            "Q_cut": q_cut.cpu().tolist(),
        }
        if ref_states is not None:
            h = torch.as_tensor(ref_states["h"][k], dtype=h_still.dtype, device=h_still.device)
            u = torch.as_tensor(ref_states["u"][k], dtype=h_still.dtype, device=h_still.device)
            v = torch.as_tensor(ref_states["v"][k], dtype=h_still.dtype, device=h_still.device)
            with torch.no_grad():
                r_in, r_out, r_cut = fluxes(torch.stack([h - h_still, h * u, h * v], dim=-1))
            entry.update(ref_Q_in=float(r_in), ref_Q_out=float(r_out),
                         ref_Q_cut=r_cut.cpu().tolist())
        per_time.append(entry)

    return {
        "s_cut": mb.s_edges[:-1].cpu().tolist(),
        "Q_prescribed": q_prescribed,
        "upstream_bc": mb.upstream_bc,
        "per_time": per_time,
    }


def plot_mass_balance(report: dict, path) -> None:
    """Discharge along the channel over time + global imbalance per time."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = np.asarray(report["s_cut"])
    entries = report["per_time"]
    colors = plt.cm.viridis(np.linspace(0.0, 0.9, len(entries)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5),
                                   gridspec_kw={"width_ratios": [2, 1]})
    for e, c in zip(entries, colors):
        ax1.plot(s, e["Q_cut"], "-o", color=c, ms=3, label=f"PINN t={e['t']:.0f}s")
        if "ref_Q_cut" in e:
            ax1.plot(s, e["ref_Q_cut"], "--", color=c, lw=1)
    if report.get("Q_prescribed"):
        ax1.axhline(report["Q_prescribed"], color="k", lw=1, ls=":", label="prescribed Q")
    ax1.set_xlabel("Distance along channel from inlet (m)")
    ax1.set_ylabel("Discharge across section (m³/s)")
    ax1.set_title("Discharge along the channel (dashed: reference)")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8)

    t = [e["t"] for e in entries]
    ax2.bar(range(len(t)), [e["imbalance_pct"] for e in entries], color=colors)
    ax2.set_xticks(range(len(t)), [f"{v:.0f}" for v in t])
    ax2.set_xlabel("t (s)")
    ax2.set_ylabel("Global imbalance (% of inflow)")
    ax2.set_title("dV/dt + Q_out − Q_in")
    ax2.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
