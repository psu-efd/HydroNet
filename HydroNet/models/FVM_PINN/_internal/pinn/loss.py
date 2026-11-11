"""
FVM-PINN loss function for 2D Shallow Water Equations.

Uses the well-balanced perturbation form from Hydrograd:
    Q = [xi, hu, hv]  where xi = h - h_still

Total loss:
    L = lambda_fvm * L_fvm + lambda_ic * L_ic + lambda_bc * L_bc + lambda_data * L_data

where:
    L_fvm  = FVM cell-residual loss (physics, weak form, well-balanced), or
             with LossConfig.residual = "strong_form" the pointwise SWE
             residual by automatic differentiation (baseline PINN)
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

from ..fvm.riemann_solver import compute_fvm_residual, interp_hydrograph
from ..fvm.smooth_ops import smooth_pow, smooth_sqrt

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
    # Physics residual: "fvm" (well-balanced Roe finite-volume residual) or
    # "strong_form" (pointwise PDE residual by autograd at the same points,
    # the standard-PINN baseline). lambda_fvm weights either; the history
    # keys stay "fvm*".
    residual: str = "fvm"
    # Steady-form residual (FVM only): drop dQ/dt, so R(Q) = 0 is enforced at
    # every sampled time independently. Pair with lambda_ic = 0 (R1-M4 test).
    steady: bool = False
    # Weight each cell's squared FVM residual by A_i / mean(A). Off by default:
    # the loss is an unweighted mean over wet cells. Normalising by the mean
    # keeps the loss scale, and leaves uniform meshes unchanged (side test).
    area_weighted: bool = False
    # Harten-Hyman entropy fix on the acoustic waves of the Roe flux in the
    # FVM residual. Off by default (ablation only).
    entropy_fix: bool = False


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

        # Cells adjacent to inlet-q faces. The inlet ghost state shares the
        # discharge over all inlet faces by conveyance, so a mini-batch stencil
        # must contain every inlet cell or the discharge is concentrated on the
        # sampled ones. Empty when there is no inlet-q boundary.
        inlet_ids = [k for k, v in mesh_data.get("bc_ghost", {}).items()
                     if v.get("type") == "inlet-q"]
        if inlet_ids and "face_bc_id" in mesh_data:
            fb = mesh_data["face_bc_id"]
            on_inlet = (mesh_data["face_right"] < 0) & torch.isin(
                fb, torch.tensor(inlet_ids, device=fb.device, dtype=fb.dtype))
            self.inlet_cells = torch.unique(mesh_data["face_left"][on_inlet])
        else:
            self.inlet_cells = None

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
        point_weights: Optional[Dict] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        point_weights : optional per-point loss weights (self-adaptive
            weighting): ``{"fvm": [n_cells], "ic": [n_ic], "bc": {bc_id:
            [n_pts]}}``, any key may be absent. Weighted terms are returned
            as ``fvm_w`` / ``ic_w`` / ``bc_w`` and replace the plain terms in
            ``total``; ``fvm`` / ``ic`` / ``bc`` stay the unweighted MSEs.
        """
        device = self.mesh_data["cell_center"].device
        zero = torch.tensor(0.0, device=device, dtype=t.dtype)
        pw = point_weights or {}

        # FVM loss with per-equation breakdown
        if self.cfg.residual == "fvm":
            fvm_result = self._fvm_loss(network, t, cell_mask, pw.get("fvm"))
        elif self.cfg.residual == "strong_form":
            fvm_result = self._strong_form_loss(network, t, cell_mask, pw.get("fvm"))
        else:
            raise ValueError(f"LossConfig.residual must be 'fvm' or 'strong_form', "
                             f"got {self.cfg.residual!r}")

        losses = {
            "fvm":       fvm_result["fvm"],
            "fvm_xi":    fvm_result["fvm_xi"],
            "fvm_hu":    fvm_result["fvm_hu"],
            "fvm_hv":    fvm_result["fvm_hv"],
        }
        if "fvm_w" in fvm_result:
            losses["fvm_w"] = fvm_result["fvm_w"]

        # IC loss with per-equation breakdown
        if ic_data is not None:
            ic_result = self._ic_loss(network, ic_data, pw.get("ic"))
            losses["ic"]    = ic_result["ic"]
            losses["ic_xi"] = ic_result["ic_xi"]
            losses["ic_hu"] = ic_result["ic_hu"]
            losses["ic_hv"] = ic_result["ic_hv"]
            if "ic_w" in ic_result:
                losses["ic_w"] = ic_result["ic_w"]
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
            bc_result = self._bc_loss(network, t, bc_data, pw.get("bc"))
            losses["bc"] = bc_result["bc"]
            for k, v in bc_result.items():
                if k.startswith("bc_"):
                    losses[k] = v
        else:
            losses["bc"] = zero

        losses["total"] = self.weighted_total(losses)
        return losses

    def weighted_total(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Total loss from the current ``cfg.lambda_*`` (re-read each call,
        so adaptive schemes can change the weights between steps). Uses the
        point-weighted terms when present."""
        cfg = self.cfg
        return (
            cfg.lambda_fvm  * losses.get("fvm_w", losses["fvm"])
            + cfg.lambda_ic   * losses.get("ic_w", losses["ic"])
            + cfg.lambda_bc   * losses.get("bc_w", losses["bc"])
            + cfg.lambda_data * losses["data"]
        )

    # ------------------------------------------------------------------
    # FVM physics loss (well-balanced)
    # ------------------------------------------------------------------

    def _fvm_loss(
        self,
        network: nn.Module,
        t: torch.Tensor,
        cell_mask: Optional[torch.Tensor],
        cell_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Well-balanced FVM residual loss: dQ/dt + R(Q) = 0

        Network outputs Q = [xi, hu, hv].
        R = flux_divergence - source (bed slope + friction).

        Returns dict with 'fvm' (total) and per-equation 'fvm_xi', 'fvm_hu', 'fvm_hv'.
        With ``cell_weights`` [n_cells] it also returns the weighted 'fvm_w'.
        """
        cell_xy = self.mesh_data["cell_center"]
        n_cells = cell_xy.shape[0]
        device = cell_xy.device

        if cell_mask is not None:
            stencil_idx, eval_mask = _build_stencil(
                cell_mask, self.mesh_data["face_left"], self.mesh_data["face_right"],
                always_include=self.inlet_cells,
            )
            stencil_xy = cell_xy[stencil_idx]
            stencil_hs = self.h_still[stencil_idx]
        else:
            stencil_idx = None
            stencil_xy = cell_xy
            stencil_hs = self.h_still

        # Per-cell weights restricted to the cells that are evaluated
        if cell_weights is not None and cell_mask is not None:
            cell_weights = cell_weights[stencil_idx][eval_mask]

        zero = torch.tensor(0.0, device=device, dtype=cell_xy.dtype)
        total_res = zero.clone()
        total_w = zero.clone()
        total_xi = zero.clone()
        total_hu = zero.clone()
        total_hv = zero.clone()
        n_t = len(t)

        area_w = None
        if self.cfg.area_weighted:
            area = self.mesh_data["cell_area"]
            area_w = area / area.mean()
            if cell_mask is not None:
                area_w = area_w[stencil_idx][eval_mask]

        for ti in range(n_t):
            t_k = t[ti : ti + 1].expand(stencil_xy.shape[0])
            xyt = torch.cat([stencil_xy, t_k.unsqueeze(-1)], dim=-1)
            xyt = xyt.requires_grad_(True)

            # Network forward: outputs [xi, hu, hv]
            Q_stencil = self._network_forward(network, xyt)

            # FVM residual (well-balanced, includes bed slope + friction)
            if stencil_idx is not None:
                local_md = _local_mesh_data(self.mesh_data, stencil_idx, n_cells)
                R = compute_fvm_residual(Q_stencil, local_md, stencil_hs, self.cfg.h_dry, t=t[ti],
                                         entropy_fix=self.cfg.entropy_fix)
            else:
                R = compute_fvm_residual(Q_stencil, self.mesh_data, stencil_hs, self.cfg.h_dry, t=t[ti],
                                         entropy_fix=self.cfg.entropy_fix)

            if self.cfg.steady:
                residual = R                                       # steady form: R(Q) = 0
            else:
                residual = _batch_time_grad(Q_stencil, xyt) + R    # dQ/dt + R(Q) = 0

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
                r2_wet = r_wet ** 2
                if area_w is not None:
                    r2_wet = r2_wet * area_w[wet].unsqueeze(-1)
                l_xi = r2_wet[:, 0].mean()
                l_hu = r2_wet[:, 1].mean()
                l_hv = r2_wet[:, 2].mean()
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
                if cell_weights is not None:
                    r2w = (r_wet ** 2 * cell_weights[wet].unsqueeze(-1)).mean(0)
                    total_w = total_w + self._combine_components(r2w)

        result = {
            "fvm": total_res / n_t,
            "fvm_xi": total_xi / n_t,
            "fvm_hu": total_hu / n_t,
            "fvm_hv": total_hv / n_t,
        }
        if cell_weights is not None:
            result["fvm_w"] = total_w / n_t
        return result

    def _combine_components(self, v: torch.Tensor) -> torch.Tensor:
        """Component-weighted sum of a [3] tensor (xi, hu, hv)."""
        return (
            self.cfg.lambda_xi * v[0]
            + self.cfg.lambda_hu * v[1]
            + self.cfg.lambda_hv * v[2]
        )

    def _strong_form_loss(
        self,
        network: nn.Module,
        t: torch.Tensor,
        cell_mask: Optional[torch.Tensor],
        cell_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Pointwise (strong-form) SWE residual at the cell centres, by autograd.

        Same PDE as the FVM residual, in perturbation variables
        Q = [xi, hu, hv], h = xi + h_still:
            d(xi)/dt + d(hu)/dx + d(hv)/dy = 0
            d(hu)/dt + d(hu^2/h)/dx + d(hu hv/h)/dy + g h d(xi)/dx + tau_x = 0
            d(hv)/dt + d(hu hv/h)/dx + d(hv^2/h)/dy + g h d(xi)/dy + tau_y = 0
        with Manning friction tau = g n^2 |q| q / h^(7/3) as in the FVM source.
        The pressure term g h grad(xi) = g h grad(h + zb) needs no bed
        derivative; the advection terms need grad h = grad xi + grad h_still,
        so h_still is extended linearly about each cell centre with
        grad h_still = -grad zb = S0_cells.

        Returns the same keys as ``_fvm_loss``.
        """
        md = self.mesh_data
        idx = (torch.arange(md["cell_center"].shape[0], device=t.device)
               if cell_mask is None else cell_mask.nonzero(as_tuple=False).view(-1))
        xy = md["cell_center"][idx]
        hs = self.h_still[idx]
        S0 = md["S0_cells"][idx]
        n_man = md["cell_manning"][idx]
        if cell_weights is not None:
            cell_weights = cell_weights[idx]

        zero = torch.tensor(0.0, device=xy.device, dtype=xy.dtype)
        total_res, total_w = zero.clone(), zero.clone()
        total_xi, total_hu, total_hv = zero.clone(), zero.clone(), zero.clone()
        n_t = len(t)
        h_dry = self.cfg.h_dry

        # A one-cell-wide (quasi-1D) mesh has no extent in one coordinate. The
        # input normalisation then clamps that std to 1e-6, so autograd would
        # return a spurious derivative about 1e6 times too large. The flow does
        # not vary along such a coordinate, so its derivatives are dropped.
        centres = md["cell_center"]
        has_extent = (centres.max(0)[0] - centres.min(0)[0]) > 1e-9
        axis_mask = torch.cat([has_extent.to(xy.dtype), torch.ones(1, device=xy.device, dtype=xy.dtype)])

        for ti in range(n_t):
            t_k = t[ti:ti + 1].expand(xy.shape[0])
            xyt = torch.cat([xy, t_k.unsqueeze(-1)], dim=-1).requires_grad_(True)
            Q = self._network_forward(network, xyt)
            xi, hu, hv = Q[:, 0], Q[:, 1], Q[:, 2]
            # Zero at the centres; carries grad h_still into autograd
            h = xi + hs + ((xyt[:, :2] - xy) * S0).sum(-1)
            hc = h.clamp(min=h_dry)

            def d(f):   # [n, 3] = (d/dx, d/dy, d/dt)
                return torch.autograd.grad(f.sum(), xyt, create_graph=True)[0] * axis_mask

            d_xi, d_hu, d_hv = d(xi), d(hu), d(hv)
            d_uu, d_uv, d_vv = d(hu * hu / hc), d(hu * hv / hc), d(hv * hv / hc)
            fric = G * n_man ** 2 * smooth_sqrt(hu * hu + hv * hv) / smooth_pow(h + h_dry, 7.0 / 3.0)

            residual = torch.stack([
                d_xi[:, 2] + d_hu[:, 0] + d_hv[:, 1],
                d_hu[:, 2] + d_uu[:, 0] + d_uv[:, 1] + G * h * d_xi[:, 0] + fric * hu,
                d_hv[:, 2] + d_uv[:, 0] + d_vv[:, 1] + G * h * d_xi[:, 1] + fric * hv,
            ], dim=-1)

            wet = h.detach() > h_dry
            if wet.any():
                r2 = residual[wet] ** 2
                l_xi, l_hu, l_hv = r2[:, 0].mean(), r2[:, 1].mean(), r2[:, 2].mean()
                total_res = total_res + (
                    self.cfg.lambda_xi * l_xi
                    + self.cfg.lambda_hu * l_hu
                    + self.cfg.lambda_hv * l_hv
                )
                total_xi = total_xi + l_xi
                total_hu = total_hu + l_hu
                total_hv = total_hv + l_hv
                if cell_weights is not None:
                    total_w = total_w + self._combine_components(
                        (r2 * cell_weights[wet].unsqueeze(-1)).mean(0))

        result = {
            "fvm": total_res / n_t,
            "fvm_xi": total_xi / n_t,
            "fvm_hu": total_hu / n_t,
            "fvm_hv": total_hv / n_t,
        }
        if cell_weights is not None:
            result["fvm_w"] = total_w / n_t
        return result

    def _network_forward(self, network: nn.Module, xyt: torch.Tensor) -> torch.Tensor:
        if self.cfg.use_grad_checkpoint:
            return torch_checkpoint(network, xyt, use_reentrant=False)
        return network(xyt)

    # ------------------------------------------------------------------
    # IC / BC / data losses
    # ------------------------------------------------------------------

    def _ic_loss(
        self, network: nn.Module, ic_data: Dict,
        point_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
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
        result = {"ic": total, "ic_xi": l_xi, "ic_hu": l_hu, "ic_hv": l_hv}
        if point_weights is not None:
            result["ic_w"] = self._combine_components(
                (diff_sq * point_weights.unsqueeze(-1)).mean(0)
            )
        return result

    def _bc_loss(
        self, network: nn.Module, t: torch.Tensor, bc_data: Dict,
        point_weights: Optional[Dict] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        BC loss evaluated at all sampled times.

        Network outputs [xi, hu, hv].
        For exit-h BC: xi should equal h_target - h_still at the boundary.
        For inlet-q BC: hu_n = -q (inflow opposes outward normal).

        Returns dict with 'bc' (total) and 'bc_{bc_id}_{type}' per boundary.
        With ``point_weights`` ({bc_id: [n_pts]}, shared over the sampled
        times) it also returns the weighted 'bc_w'.
        """
        total = torch.tensor(0.0, device=t.device, dtype=t.dtype)
        total_w = total.clone()
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
            sq = None   # pointwise squared error [n_t * n_pts]

            if bc_type == "exit-h":
                val = bc_info["value"]
                if isinstance(val, torch.Tensor):
                    val = val.repeat(n_t)
                h_still_bc = bc_info.get("h_still", 0.0)
                if isinstance(h_still_bc, torch.Tensor):
                    h_still_bc = h_still_bc.repeat(n_t)
                h_pred = Q_pred[:, 0] + h_still_bc
                sq = (h_pred - val) ** 2

            elif bc_type == "inlet-q":
                nx = bc_info["nx"].repeat(n_t)
                ny = bc_info["ny"].repeat(n_t)
                val = bc_info["value"]
                if isinstance(val, torch.Tensor):
                    val = val.repeat(n_t)
                if "hydrograph" in bc_info:
                    # value = 1 / inlet width; scale by Q(t), rows time-major
                    val = val * interp_hydrograph(bc_info["hydrograph"], t).repeat_interleave(n_pts)
                hu_n = Q_pred[:, 1] * nx + Q_pred[:, 2] * ny
                sq = (hu_n + val) ** 2

            elif bc_type in ("wall", "symmetry"):
                nx = bc_info["nx"].repeat(n_t)
                ny = bc_info["ny"].repeat(n_t)
                un = Q_pred[:, 1] * nx + Q_pred[:, 2] * ny
                sq = un ** 2

            if sq is not None:
                bc_loss_val = sq.mean()
            if point_weights is not None:
                # Rows are time-major (t * n_pts + p), as for nx.repeat(n_t)
                w = point_weights.get(bc_id)
                total_w = total_w + (
                    (sq * w.repeat(n_t)).mean() if (w is not None and sq is not None)
                    else bc_loss_val
                )

            total = total + bc_loss_val
            result[f"bc_{bc_id}_{bc_type}"] = bc_loss_val
            n_bc += 1

        result["bc"] = total / max(n_bc, 1)
        if point_weights is not None:
            result["bc_w"] = total_w / max(n_bc, 1)
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
    always_include: Optional[torch.Tensor] = None,
) -> tuple:
    """Build the stencil for a mini-batch FVM evaluation.

    The stencil holds the sampled cells, their face neighbours (so every face
    flux of a sampled cell is exact) and ``always_include`` (the inlet-q
    cells, whose states set the inlet discharge distribution).

    Returns ``stencil_idx`` (sorted global cell indices) and ``eval_mask``
    (True at the sampled cells, the only ones whose residual is used).
    """
    in_stencil = cell_mask.clone()
    interior = face_right >= 0
    fl, fr = face_left[interior], face_right[interior]
    in_stencil[fr[cell_mask[fl]]] = True
    in_stencil[fl[cell_mask[fr]]] = True
    if always_include is not None:
        in_stencil[always_include] = True
    stencil_idx = in_stencil.nonzero(as_tuple=False).view(-1)
    eval_mask = cell_mask[stencil_idx]
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
