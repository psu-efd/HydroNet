"""
Shared Heun RK2 time-stepper for the FVM solver.

Single source of truth for:
    * ``compute_dt_cfl`` — CFL-limited time step from the current state.
    * ``run_fvm_rk2``    — Heun RK2 integration with per-snapshot output.

Both are used by the internal ``TeacherTrainer`` to generate its teacher
trajectory and by standalone ``<case>_fvm_only.py`` drivers for pure-FVM
validation, so the clamp / dry-cell / Manning-blow-up handling lives in
exactly one place.

Conventions
-----------
``Q`` stores conserved variables in **perturbation form** ``[xi, hu, hv]``
where ``xi = h - h_still``. The physical depth ``h = xi + h_still`` is
clamped to ``>= 0`` (not ``xi >= 0``) so the free surface can drop below
the still-water reference without being artificially filled up. Momentum
is zeroed in cells whose physical depth drops below ``h_dry`` to keep
``hu / h`` from blowing up the Manning friction source on the next
residual evaluation.
"""

from typing import Optional, Sequence, Tuple, Union

import torch

from .riemann_solver import compute_fvm_residual

G: float = 9.81


# ---------------------------------------------------------------------------
# CFL time-step
# ---------------------------------------------------------------------------

def compute_dt_cfl(
    Q: torch.Tensor,
    h_still: torch.Tensor,
    mesh_data: dict,
    cfl: float = 0.3,
    h_small: float = 1e-3,
) -> float:
    """
    CFL-limited time step ``dt = cfl * min(dx / (|u| + |v| + c))`` over
    wet cells, where ``dx ≈ sqrt(cell_area)`` is a characteristic length.

    Returns at least ``1e-10`` to avoid zero/negative dt if every cell is
    dry. Callers should further cap ``dt`` by any outer target time.
    """
    xi = Q[:, 0]
    hu = Q[:, 1]
    hv = Q[:, 2]
    h = (xi + h_still).clamp(min=0.0)
    cell_area = mesh_data["cell_area"]
    dx = torch.sqrt(cell_area)

    h_safe = h.clamp(min=h_small)
    u = hu / h_safe
    v = hv / h_safe
    c = torch.sqrt(torch.tensor(G, dtype=h.dtype, device=h.device) * h_safe)
    speed = torch.abs(u) + torch.abs(v) + c

    wet = h > h_small
    if wet.any():
        dt = cfl * (dx[wet] / speed[wet]).min().item()
    else:
        dt = 1e-6
    return max(dt, 1e-10)


# ---------------------------------------------------------------------------
# Heun RK2 integrator with per-snapshot output
# ---------------------------------------------------------------------------

def run_fvm_rk2(
    Q_init: torch.Tensor,
    mesh_data: dict,
    h_still: torch.Tensor,
    snapshot_times: Union[torch.Tensor, Sequence[float]],
    cfl: float = 0.3,
    h_dry: float = 1e-2,
    progress_cb: Optional[callable] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Heun RK2 integration of the well-balanced FVM SWE residual from
    ``snapshot_times[0]`` to ``snapshot_times[-1]``, saving state at each
    requested snapshot time.

    Parameters
    ----------
    Q_init          : ``[n_cells, 3]`` initial conserved variables
                      ``[xi, hu, hv]`` at ``snapshot_times[0]``.
    mesh_data       : dict of mesh tensors (face/cell topology, bed, Manning,
                      S0, ``bc_ghost``, ...). See ``compute_cell_geometry``.
    h_still         : ``[n_cells]`` still-water depth for the perturbation form.
    snapshot_times  : 1-D tensor/list of snapshot times. Must be monotonically
                      non-decreasing. ``Q_init`` is taken to be the state at
                      ``snapshot_times[0]``.
    cfl             : CFL number.
    h_dry           : depth threshold below which momentum is zeroed after
                      each RK stage.
    progress_cb     : optional callback ``fn(k, t_k, Q_k)`` invoked after
                      each snapshot is captured (``k`` is the snapshot index
                      ≥ 1; the initial state at k=0 is not reported).

    Returns
    -------
    t_snaps : ``[N+1]`` tensor of snapshot times (same dtype/device as Q_init).
    Q_snaps : ``[N+1, n_cells, 3]`` tensor; ``Q_snaps[k]`` is the state at
              ``t_snaps[k]``. ``Q_snaps[0] == Q_init`` (a detached clone).
    """
    if isinstance(snapshot_times, torch.Tensor):
        t_snaps = snapshot_times.to(dtype=Q_init.dtype, device=Q_init.device)
    else:
        t_snaps = torch.tensor(
            list(snapshot_times), dtype=Q_init.dtype, device=Q_init.device
        )
    if t_snaps.dim() != 1:
        raise ValueError("snapshot_times must be 1-D")
    if len(t_snaps) < 2:
        raise ValueError("snapshot_times must contain at least two entries "
                         "(start and end)")

    n_cells = Q_init.shape[0]
    Q_snaps = torch.zeros(len(t_snaps), n_cells, 3,
                          dtype=Q_init.dtype, device=Q_init.device)
    Q_snaps[0] = Q_init.detach().clone()

    Q_cur = Q_init.detach().clone()
    t_cur = float(t_snaps[0].item())

    with torch.no_grad():
        for k in range(1, len(t_snaps)):
            t_target = float(t_snaps[k].item())
            while t_cur < t_target - 1e-12:
                dt = compute_dt_cfl(Q_cur, h_still, mesh_data, cfl=cfl)
                if t_cur + dt > t_target:
                    dt = t_target - t_cur

                # ---- Heun RK2 stage 1 ----
                R1 = compute_fvm_residual(Q_cur, mesh_data, h_still)
                Q_star = Q_cur - dt * R1
                h_star = (Q_star[:, 0] + h_still).clamp(min=0.0)
                Q_star[:, 0] = h_star - h_still
                dry1 = h_star < h_dry
                Q_star[dry1, 1] = 0.0
                Q_star[dry1, 2] = 0.0

                # ---- Heun RK2 stage 2 ----
                R2 = compute_fvm_residual(Q_star, mesh_data, h_still)
                Q_cur = Q_cur - 0.5 * dt * (R1 + R2)
                h_new = (Q_cur[:, 0] + h_still).clamp(min=0.0)
                Q_cur[:, 0] = h_new - h_still
                dry2 = h_new < h_dry
                Q_cur[dry2, 1] = 0.0
                Q_cur[dry2, 2] = 0.0

                t_cur += float(dt)

            Q_snaps[k] = Q_cur.clone()
            if progress_cb is not None:
                progress_cb(k, t_cur, Q_snaps[k])

    return t_snaps, Q_snaps
