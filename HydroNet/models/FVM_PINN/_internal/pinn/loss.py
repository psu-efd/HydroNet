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
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint

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

        Returns dict with 'fvm' (total) and per-equation 'fvm_xi', 'fvm_hu', 'fvm_hv'.
        """
        cell_xy = self.mesh_data["cell_center"]
        n_cells = cell_xy.shape[0]
        device = cell_xy.device

        if cell_mask is not None:
            stencil_idx, eval_mask = _build_stencil(
                cell_mask, self.mesh_data["face_left"], self.mesh_data["face_right"]
            )
            stencil_xy = cell_xy[stencil_idx]
            stencil_hs = self.h_still[stencil_idx]
        else:
            stencil_idx = None
            stencil_xy = cell_xy
            stencil_hs = self.h_still

        zero = torch.tensor(0.0, device=device, dtype=cell_xy.dtype)
        total_res = zero.clone()
        total_xi = zero.clone()
        total_hu = zero.clone()
        total_hv = zero.clone()
        n_t = len(t)

        for ti in range(n_t):
            t_k = t[ti : ti + 1].expand(stencil_xy.shape[0])
            xyt = torch.cat([stencil_xy, t_k.unsqueeze(-1)], dim=-1)
            xyt = xyt.requires_grad_(True)

            # Network forward: outputs [xi, hu, hv]
            Q_stencil = self._network_forward(network, xyt)

            # dQ/dt via autograd
            dQ_dt = _batch_time_grad(Q_stencil, xyt)

            # FVM residual (well-balanced, includes bed slope + friction)
            if stencil_idx is not None:
                local_md = _local_mesh_data(self.mesh_data, stencil_idx, n_cells)
                R = compute_fvm_residual(Q_stencil, local_md, stencil_hs, self.cfg.h_dry)
            else:
                R = compute_fvm_residual(Q_stencil, self.mesh_data, stencil_hs, self.cfg.h_dry)

            residual = dQ_dt + R

            # Restrict to sampled cells
            if cell_mask is not None:
                residual = residual[eval_mask]
                xi_check = Q_stencil[eval_mask, 0].detach()
                h_check = (xi_check + stencil_hs[eval_mask]).detach()
            else:
                h_check = (Q_stencil[:, 0].detach() + stencil_hs).detach()

            wet = h_check > self.cfg.h_dry
            if wet.any():
                r_wet = residual[wet]
                l_xi = (r_wet[:, 0] ** 2).mean()
                l_hu = (r_wet[:, 1] ** 2).mean()
                l_hv = (r_wet[:, 2] ** 2).mean()
                # Weighted sum across conserved variables. Component weights
                # default to 1.0 (uniform), but can be tuned per case to
                # balance scale mismatches between xi and hu/hv.
                total_res = total_res + (
                    self.cfg.lambda_xi * l_xi
                    + self.cfg.lambda_hu * l_hu
                    + self.cfg.lambda_hv * l_hv
                )
                total_xi = total_xi + l_xi
                total_hu = total_hu + l_hu
                total_hv = total_hv + l_hv

        return {
            "fvm": total_res / n_t,
            "fvm_xi": total_xi / n_t,
            "fvm_hu": total_hu / n_t,
            "fvm_hv": total_hv / n_t,
        }

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
    """Compute dQ/dt for all 3 components."""
    grads: List[torch.Tensor] = []
    for col in range(Q.shape[-1]):
        s = Q[:, col].sum()
        g = torch.autograd.grad(s, xyt, create_graph=True, retain_graph=True)[0]
        grads.append(g[:, 2:3])
    return torch.cat(grads, dim=-1)


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
