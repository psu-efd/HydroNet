"""
FVM-PINN loss function for 2D Shallow Water Equations.

Uses the well-balanced perturbation form from Hydrograd:
    Q = [xi, hu, hv]  where xi = h - h_still

Total loss:
    L = lambda_fvm * L_fvm + lambda_ic * L_ic + lambda_bc * L_bc + lambda_data * L_data

where:
    L_fvm  = FVM cell-residual loss (physics, weak form, well-balanced)
    L_ic   = initial condition loss
    L_bc   = boundary condition loss (inflow/outflow/wall)
    L_data = optional data fitting loss
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from ..fvm.conservation import MassBalance
from ..fvm.riemann_solver import compute_fvm_residual

logger = logging.getLogger(__name__)

G = 9.81


@dataclass
class LossConfig:
    """Loss weights and settings.

    Component weights
    -----------------
    ``lambda_xi`` / ``lambda_hu`` / ``lambda_hv`` scale the contribution of
    each SWE conserved variable to the **elementwise** losses (FVM residual,
    IC MSE, data MSE). They do NOT affect BC losses, which are already
    per-type scalars. Defaults are all 1.0 (uniform); raise one of them to
    emphasize fitting that variable when its magnitude is typically smaller
    than the others (e.g. ``lambda_xi > 1`` when the well-balanced
    perturbation keeps xi << hu/hv in your case).
    """
    lambda_fvm: float = 1.0
    lambda_ic: float = 10.0
    lambda_bc: float = 10.0
    lambda_data: float = 1.0
    # Per-conserved-variable weights applied inside elementwise losses
    lambda_xi: float = 1.0
    lambda_hu: float = 1.0
    lambda_hv: float = 1.0
    h_dry: float = 1e-4            # wet/dry threshold
    use_grad_checkpoint: bool = False
    # Mass conservation on nested control volumes along the channel
    # (see fvm/conservation.py). Full-batch only.
    use_mass: bool = False
    lambda_mass: float = 1.0
    mass_n_regions: int = 10
    mass_upstream_bc: Tuple[int, ...] = ()   # default: inlet-q BC ids
    mass_flux_scale: Optional[float] = None  # m³/s; default: prescribed inlet Q


class FVMPINNLoss(nn.Module):
    """
    Well-balanced FVM-PINN composite loss.

    The network outputs Q = [xi, hu, hv] in perturbation form.
    The FVM residual uses the well-balanced Roe solver with:
        dQ/dt + R(Q) = 0
    where R includes flux divergence and source terms (bed slope + friction).

    Parameters
    ----------
    cfg       : LossConfig
    mesh_data : dict of PyTorch tensors from compute_cell_geometry()
    h_still   : [n_cells] tensor — still water reference depth
    bc_config : optional dict of BC metadata
    """

    def __init__(
        self,
        cfg: LossConfig,
        mesh_data: Dict[str, torch.Tensor],
        h_still: Optional[torch.Tensor] = None,
        bc_config: Optional[Dict] = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.mesh_data = mesh_data
        self.bc_config = bc_config or {}

        # h_still: still water reference. Default to zero (then xi = h).
        n_cells = mesh_data["n_cells"]
        device = mesh_data["cell_center"].device
        dtype = mesh_data["cell_center"].dtype
        if h_still is not None:
            self.h_still = h_still.to(device=device, dtype=dtype)
        else:
            self.h_still = torch.zeros(n_cells, device=device, dtype=dtype)

        self.mass_balance: Optional[MassBalance] = None
        if cfg.use_mass:
            self.mass_balance = MassBalance(
                mesh_data, cfg.mass_upstream_bc or None, cfg.mass_n_regions
            )
            if cfg.mass_flux_scale is not None:
                self.mass_flux_scale = float(cfg.mass_flux_scale)
            else:
                q_in = [
                    float(mesh_data["bc_ghost"][b]["value"])
                    for b in self.mass_balance.upstream_bc
                    if mesh_data.get("bc_ghost", {}).get(b, {}).get("type") == "inlet-q"
                ]
                if not q_in:
                    raise ValueError(
                        "training.mass_conservation.flux_scale is required when "
                        "the upstream BC has no prescribed inlet discharge."
                    )
                self.mass_flux_scale = sum(q_in)
            logger.info(
                f"Mass conservation: {self.mass_balance.n_regions} nested regions "
                f"from BC {self.mass_balance.upstream_bc}, "
                f"flux scale {self.mass_flux_scale:.3g} m³/s"
            )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def forward(
        self,
        network: nn.Module,
        t: torch.Tensor,
        ic_data: Optional[Dict] = None,
        bc_data: Optional[Dict] = None,
        ref_data: Optional[Dict] = None,
        cell_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        device = self.mesh_data["cell_center"].device
        zero = torch.tensor(0.0, device=device, dtype=t.dtype)

        # FVM loss with per-equation breakdown
        fvm_result = self._fvm_loss(network, t, cell_mask)

        losses = {
            "fvm":       fvm_result["fvm"],
            "fvm_xi":    fvm_result["fvm_xi"],
            "fvm_hu":    fvm_result["fvm_hu"],
            "fvm_hv":    fvm_result["fvm_hv"],
        }
        if "mass" in fvm_result:
            losses["mass"] = fvm_result["mass"]
            losses["mass_global"] = fvm_result["mass_global"]

        # IC loss with per-equation breakdown
        if ic_data is not None:
            ic_result = self._ic_loss(network, ic_data)
            losses["ic"]    = ic_result["ic"]
            losses["ic_xi"] = ic_result["ic_xi"]
            losses["ic_hu"] = ic_result["ic_hu"]
            losses["ic_hv"] = ic_result["ic_hv"]
        else:
            losses["ic"] = zero
            losses["ic_xi"] = zero
            losses["ic_hu"] = zero
            losses["ic_hv"] = zero

        # Data loss with per-equation breakdown
        if ref_data is not None:
            data_result = self._data_loss(network, ref_data)
            losses["data"]    = data_result["data"]
            losses["data_xi"] = data_result["data_xi"]
            losses["data_hu"] = data_result["data_hu"]
            losses["data_hv"] = data_result["data_hv"]
        else:
            losses["data"] = zero
            losses["data_xi"] = zero
            losses["data_hu"] = zero
            losses["data_hv"] = zero

        # BC loss with per-boundary breakdown
        if bc_data is not None:
            bc_result = self._bc_loss(network, t, bc_data)
            losses["bc"] = bc_result["bc"]
            for k, v in bc_result.items():
                if k.startswith("bc_"):
                    losses[k] = v
        else:
            losses["bc"] = zero

        cfg = self.cfg
        losses["total"] = (
            cfg.lambda_fvm  * losses["fvm"]
            + cfg.lambda_ic   * losses["ic"]
            + cfg.lambda_bc   * losses["bc"]
            + cfg.lambda_data * losses["data"]
        )
        if "mass" in losses:
            losses["total"] = losses["total"] + cfg.lambda_mass * losses["mass"]
        return losses

    # ------------------------------------------------------------------
    # FVM physics loss (well-balanced)
    # ------------------------------------------------------------------

    def _fvm_loss(
        self,
        network: nn.Module,
        t: torch.Tensor,
        cell_mask: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Well-balanced FVM residual loss: dQ/dt + R(Q) = 0

        Network outputs Q = [xi, hu, hv].
        R = flux_divergence - source (bed slope + friction).

        All sampled time levels are evaluated in one batch: a single network
        forward over [n_t * n_stencil] points, a single dQ/dt backward, and a
        single residual evaluation on [n_t, n_stencil, 3]. The loss is the
        same as evaluating each time level separately: per time level, the
        mean squared residual over its *wet* cells (0 if none are wet), then
        averaged over the n_t time levels.

        Returns dict with 'fvm' (total) and per-equation 'fvm_xi', 'fvm_hu', 'fvm_hv'.
        """
        cell_xy = self.mesh_data["cell_center"]
        n_cells = cell_xy.shape[0]

        if cell_mask is not None:
            if self.mass_balance is not None:
                raise ValueError(
                    "Mass-conservation loss needs every cell of each region; it is "
                    "not supported with a cell mask (minibatch strategy)."
                )
            stencil_idx, eval_mask = _build_stencil(
                cell_mask, self.mesh_data["face_left"], self.mesh_data["face_right"]
            )
            eval_idx = eval_mask.nonzero(as_tuple=False).view(-1)
            stencil_xy = cell_xy[stencil_idx]
            stencil_hs = self.h_still[stencil_idx]
            md = _local_mesh_data(self.mesh_data, stencil_idx, n_cells)
        else:
            eval_idx = None
            stencil_xy = cell_xy
            stencil_hs = self.h_still
            md = self.mesh_data

        n_t = len(t)
        n_s = stencil_xy.shape[0]

        # Time-major batch: rows [k * n_s, (k + 1) * n_s) hold time level t[k]
        xy_all = stencil_xy.unsqueeze(0).expand(n_t, n_s, 2)
        t_all = t.view(n_t, 1, 1).expand(n_t, n_s, 1)
        xyt = torch.cat([xy_all, t_all], dim=-1).reshape(n_t * n_s, 3)
        xyt = xyt.requires_grad_(True)

        # Network forward: outputs [xi, hu, hv]
        Q_flat = self._network_forward(network, xyt)

        # dQ/dt via autograd (each output row depends only on its own input row)
        dQ_dt = _batch_time_grad(Q_flat, xyt).view(n_t, n_s, 3)
        Q_stencil = Q_flat.view(n_t, n_s, 3)

        # FVM residual (well-balanced, includes bed slope + friction)
        R = compute_fvm_residual(Q_stencil, md, stencil_hs, self.cfg.h_dry)

        residual = dQ_dt + R                                  # [n_t, n_s, 3]
        h_check = Q_stencil[..., 0].detach() + stencil_hs     # [n_t, n_s]

        # Restrict to sampled cells
        if eval_idx is not None:
            residual = residual.index_select(1, eval_idx)
            h_check = h_check.index_select(1, eval_idx)

        # Per-time-level mean over wet cells, without a host sync. torch.where
        # (not multiplication by the mask) keeps any non-finite residual at a
        # dry cell out of the forward value, exactly like boolean indexing did.
        wet = h_check > self.cfg.h_dry                        # [n_t, n_e]
        n_wet = wet.sum(dim=-1).clamp(min=1).to(residual.dtype)
        r_sq = torch.where(wet.unsqueeze(-1), residual, torch.zeros_like(residual)) ** 2
        per_t = r_sq.sum(dim=1) / n_wet.unsqueeze(-1)         # [n_t, 3]; 0 if no wet cell
        l_xi, l_hu, l_hv = per_t.unbind(dim=-1)               # each [n_t]

        # Weighted sum across conserved variables. Component weights
        # default to 1.0 (uniform), but can be tuned per case to
        # balance scale mismatches between xi and hu/hv.
        total_res = (
            self.cfg.lambda_xi * l_xi
            + self.cfg.lambda_hu * l_hu
            + self.cfg.lambda_hv * l_hv
        )

        out = {
            "fvm": total_res.sum() / n_t,
            "fvm_xi": l_xi.sum() / n_t,
            "fvm_hu": l_hu.sum() / n_t,
            "fvm_hv": l_hv.sum() / n_t,
        }

        # Region mass balances: area-weighted signed sums of the continuity
        # residual over all cells (wet or not) — a systematic loss or gain of
        # water cannot average out here the way it does cell-by-cell.
        if self.mass_balance is not None:
            E = self.mass_balance.imbalance(residual[..., 0])     # [n_t, N]
            E_rel = E / self.mass_flux_scale
            out["mass"] = (E_rel ** 2).mean()
            out["mass_global"] = E_rel[:, -1].abs().mean().detach()
        return out

    def _network_forward(self, network: nn.Module, xyt: torch.Tensor) -> torch.Tensor:
        if self.cfg.use_grad_checkpoint:
            return torch_checkpoint(network, xyt, use_reentrant=False)
        return network(xyt)

    # ------------------------------------------------------------------
    # IC / BC / data losses
    # ------------------------------------------------------------------

    def _ic_loss(self, network: nn.Module, ic_data: Dict) -> Dict[str, torch.Tensor]:
        Q_pred = self._network_forward(network, ic_data["xyt"])
        diff_sq = (Q_pred - ic_data["U_true"]) ** 2
        l_xi = diff_sq[:, 0].mean()
        l_hu = diff_sq[:, 1].mean()
        l_hv = diff_sq[:, 2].mean()
        total = (
            self.cfg.lambda_xi * l_xi
            + self.cfg.lambda_hu * l_hu
            + self.cfg.lambda_hv * l_hv
        )
        return {"ic": total, "ic_xi": l_xi, "ic_hu": l_hu, "ic_hv": l_hv}

    def _bc_loss(
        self, network: nn.Module, t: torch.Tensor, bc_data: Dict
    ) -> Dict[str, torch.Tensor]:
        """
        BC loss evaluated at all sampled times.

        Network outputs [xi, hu, hv].
        For exit-h BC: xi should equal h_target - h_still at the boundary.
        For inlet-q BC: hu_n = -q (inflow opposes outward normal).

        Returns dict with 'bc' (total) and 'bc_{bc_id}_{type}' per boundary.
        """
        total = torch.tensor(0.0, device=t.device, dtype=t.dtype)
        result = {}
        n_bc = 0
        n_t = len(t)

        for bc_id, bc_info in bc_data.items():
            bc_type = bc_info["type"]
            xyt_base = bc_info["xyt"]
            xy = xyt_base[:, :2]
            n_pts = xy.shape[0]

            xy_rep = xy.unsqueeze(0).expand(n_t, -1, -1).reshape(-1, 2)
            t_rep = t.unsqueeze(1).expand(-1, n_pts).reshape(-1, 1)
            xyt_all = torch.cat([xy_rep, t_rep], dim=-1)

            Q_pred = self._network_forward(network, xyt_all)
            bc_loss_val = torch.tensor(0.0, device=t.device, dtype=t.dtype)

            if bc_type == "exit-h":
                val = bc_info["value"]
                if isinstance(val, torch.Tensor):
                    val = val.repeat(n_t)
                h_still_bc = bc_info.get("h_still", 0.0)
                if isinstance(h_still_bc, torch.Tensor):
                    h_still_bc = h_still_bc.repeat(n_t)
                h_pred = Q_pred[:, 0] + h_still_bc
                bc_loss_val = ((h_pred - val) ** 2).mean()

            elif bc_type == "inlet-q":
                nx = bc_info["nx"].repeat(n_t)
                ny = bc_info["ny"].repeat(n_t)
                val = bc_info["value"]
                if isinstance(val, torch.Tensor):
                    val = val.repeat(n_t)
                hu_n = Q_pred[:, 1] * nx + Q_pred[:, 2] * ny
                bc_loss_val = ((hu_n + val) ** 2).mean()

            elif bc_type in ("wall", "symmetry"):
                nx = bc_info["nx"].repeat(n_t)
                ny = bc_info["ny"].repeat(n_t)
                un = Q_pred[:, 1] * nx + Q_pred[:, 2] * ny
                bc_loss_val = (un ** 2).mean()

            total = total + bc_loss_val
            result[f"bc_{bc_id}_{bc_type}"] = bc_loss_val
            n_bc += 1

        result["bc"] = total / max(n_bc, 1)
        return result

    def _data_loss(self, network: nn.Module, ref_data: Dict) -> Dict[str, torch.Tensor]:
        Q_pred = self._network_forward(network, ref_data["xyt"])
        diff_sq = (Q_pred - ref_data["U_ref"]) ** 2
        device, dtype = diff_sq.device, diff_sq.dtype

        # Optional variable mask: zero out components not in the mask so the
        # per-component breakdown stays faithful (a masked-out variable has
        # zero data loss, independent of its current residual).
        # Accepts either a global mask of shape [3] or a per-point mask of
        # shape [n_pts, 3] (useful when concatenating sparse-velocity and
        # dense all-variable reference data in a single ref_data dict).
        if "var_mask" in ref_data:
            mask = ref_data["var_mask"]
            keep = mask.to(dtype=dtype, device=device)
            if keep.dim() == 1:
                keep = keep.unsqueeze(0)  # [3] -> [1, 3] broadcast
            diff_sq = diff_sq * keep

        def _component(j):
            # If this column is fully zeroed by var_mask, mean() is still 0
            return diff_sq[:, j].mean()

        l_xi = _component(0)
        l_hu = _component(1)
        l_hv = _component(2)
        total = (
            self.cfg.lambda_xi * l_xi
            + self.cfg.lambda_hu * l_hu
            + self.cfg.lambda_hv * l_hv
        )
        return {"data": total, "data_xi": l_xi, "data_hu": l_hu, "data_hv": l_hv}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _batch_time_grad(Q: torch.Tensor, xyt: torch.Tensor) -> torch.Tensor:
    """
    Compute dQ/dt for all 3 conserved components.

    Uses a single vmapped backward pass (``is_grads_batched``) seeded with
    the 3×3 identity rather than one ``autograd.grad`` call per component.
    The FLOP count is the same, but the three cotangents propagate through
    the network graph as batched matmuls instead of three separate small
    ones — fewer kernel launches and better BLAS utilisation. This runs
    once per time level per optimizer step, so the saving compounds.
    """
    n_out = Q.shape[-1]
    # basis[j] is the cotangent selecting output component j, broadcast
    # over all rows: shape [n_out, n_points, n_out].
    basis = torch.eye(n_out, dtype=Q.dtype, device=Q.device)
    grad_outputs = basis.unsqueeze(1).expand(n_out, Q.shape[0], n_out)
    g = torch.autograd.grad(
        Q, xyt,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        is_grads_batched=True,
    )[0]                      # [n_out, n_points, 3] — last dim is (x, y, t)
    return g[..., 2].transpose(0, 1)     # [n_points, n_out]


def _build_stencil(
    cell_mask: torch.Tensor,
    face_left: torch.Tensor,
    face_right: torch.Tensor,
) -> tuple:
    """Build stencil for mini-batch FVM evaluation."""
    sample_set = set(cell_mask.nonzero(as_tuple=False).view(-1).tolist())
    stencil_set = set(sample_set)
    fl = face_left.tolist()
    fr = face_right.tolist()
    for fi, (l, r) in enumerate(zip(fl, fr)):
        if l in sample_set and r >= 0:
            stencil_set.add(r)
        if r in sample_set:
            stencil_set.add(l)

    stencil_list = sorted(stencil_set)
    stencil_idx = torch.tensor(stencil_list, dtype=torch.long, device=cell_mask.device)

    stencil_to_local = {g: loc for loc, g in enumerate(stencil_list)}
    eval_positions = [stencil_to_local[g] for g in sorted(sample_set)]
    eval_mask = torch.zeros(len(stencil_list), dtype=torch.bool, device=cell_mask.device)
    eval_mask[eval_positions] = True

    return stencil_idx, eval_mask


def _local_mesh_data(
    mesh_data: Dict[str, torch.Tensor],
    stencil_idx: torch.Tensor,
    n_cells_global: int,
) -> Dict[str, torch.Tensor]:
    """Build a local mesh_data view for the stencil subset."""
    device = stencil_idx.device

    global_to_local = torch.full((n_cells_global,), -1, dtype=torch.long, device=device)
    global_to_local[stencil_idx] = torch.arange(len(stencil_idx), device=device)

    fl_g = mesh_data["face_left"]
    fr_g = mesh_data["face_right"]

    left_in = global_to_local[fl_g] >= 0
    face_sel = left_in

    fl_local = global_to_local[fl_g[face_sel]]
    fr_raw = fr_g[face_sel]
    fr_local = torch.where(fr_raw >= 0, global_to_local[fr_raw], torch.full_like(fr_raw, -1))

    local_md = {
        "face_left":    fl_local,
        "face_right":   fr_local,
        "face_normal":  mesh_data["face_normal"][face_sel],
        "face_length":  mesh_data["face_length"][face_sel],
        "cell_area":    mesh_data["cell_area"][stencil_idx],
        "bed_elev":     mesh_data["bed_elev"][stencil_idx],
        "cell_center":  mesh_data["cell_center"][stencil_idx],
        "cell_manning": mesh_data["cell_manning"][stencil_idx],
        "S0_cells":     mesh_data["S0_cells"][stencil_idx],
        "n_cells":      len(stencil_idx),
        "n_faces":      int(face_sel.sum().item()),
    }

    # Propagate bc_ghost and face_bc_id if present
    if "bc_ghost" in mesh_data:
        local_md["bc_ghost"] = mesh_data["bc_ghost"]
    if "face_bc_id" in mesh_data:
        local_md["face_bc_id"] = mesh_data["face_bc_id"][face_sel]

    return local_md
