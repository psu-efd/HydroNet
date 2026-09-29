"""
Mass-conservation bookkeeping on nested control volumes along the channel.

The FV fluxes are conservative, so for any set of cells Ω the area-weighted
sum of the continuity residual telescopes to the region's mass balance:

    E_Ω(t) = Σ_{i∈Ω} A_i · (∂ξ_i/∂t + R_ξ,i)
           = dV_Ω/dt + (net outflow across ∂Ω)

Interior-face fluxes cancel pairwise; inflow enters through the inlet ghost
flux inside R_ξ. E_Ω therefore needs no prescribed discharge: it holds
unchanged when the inlet Q is unknown or learnable.

``MassBalance`` builds, once per mesh:

* ``s``        : along-channel distance of every cell from the upstream
                 boundary (shortest path through the cell-adjacency graph,
                 so it follows bends);
* nested regions Ω_k = {s ≤ (k/N)·s_max}, k = 1..N (Ω_N = whole domain);
* ``W``        : [N, n_cells] area-weighted membership, so region imbalances
                 are one matmul ``r_xi @ W.T`` (no host syncs);
* cut faces    : interior faces with exactly one side in Ω_k, with signs, so
                 the discharge crossing each cut can be read off face fluxes.
"""

from typing import Iterable, Optional

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra


class MassBalance:
    """
    Parameters
    ----------
    mesh_data    : dict from ``compute_cell_geometry`` (+ ``bc_ghost``).
    upstream_bc  : BC ids whose faces mark the upstream end of the reach.
                   Default: every ``inlet-q`` id in ``mesh_data["bc_ghost"]``.
    n_regions    : number of nested regions N (the last is the whole domain).
    """

    def __init__(
        self,
        mesh_data: dict,
        upstream_bc: Optional[Iterable[int]] = None,
        n_regions: int = 10,
    ) -> None:
        device = mesh_data["cell_area"].device
        dtype = mesh_data["cell_area"].dtype

        face_left = mesh_data["face_left"].cpu().numpy()
        face_right = mesh_data["face_right"].cpu().numpy()
        face_bc_id = mesh_data["face_bc_id"].cpu().numpy()
        centers = mesh_data["cell_center"].cpu().numpy()
        n_cells = int(mesh_data["n_cells"])

        if upstream_bc is None or len(list(upstream_bc)) == 0:
            upstream_bc = [
                bc_id for bc_id, info in mesh_data.get("bc_ghost", {}).items()
                if info.get("type") == "inlet-q"
            ]
        self.upstream_bc = sorted(int(b) for b in upstream_bc)
        if not self.upstream_bc:
            raise ValueError(
                "MassBalance needs at least one upstream BC id "
                "(set training.mass_conservation.upstream_bc)."
            )

        # --- Along-channel distance from the upstream boundary ---
        interior = face_right >= 0
        fl, fr = face_left[interior], face_right[interior]
        w = np.linalg.norm(centers[fl] - centers[fr], axis=1)
        graph = coo_matrix((w, (fl, fr)), shape=(n_cells, n_cells)).tocsr()
        upstream_faces = np.isin(face_bc_id, self.upstream_bc) & ~interior
        sources = np.unique(face_left[upstream_faces])
        if sources.size == 0:
            raise ValueError(
                f"No boundary faces carry upstream BC ids {self.upstream_bc}."
            )
        s = dijkstra(graph, directed=False, indices=sources, min_only=True)
        finite = np.isfinite(s)
        s_max = float(s[finite].max()) if finite.any() else 0.0

        # --- Nested regions Ω_k = {s <= (k/N) s_max}; last = whole domain ---
        members = []
        edges = []
        for k in range(1, int(n_regions) + 1):
            edge = s_max * k / n_regions
            m = finite & (s <= edge)
            if k == n_regions:
                m = np.ones(n_cells, dtype=bool)   # unreachable cells too
            if m.any() and not (members and np.array_equal(m, members[-1])):
                members.append(m)
                edges.append(edge)
        M = np.stack(members)                        # [N, n_cells] bool
        self.n_regions = M.shape[0]

        # --- Cut faces of each region (interior faces crossing ∂Ω_k) ---
        # sign +1 when the region is on the face's left: the face flux
        # (left → right) then leaves the region, i.e. flows downstream.
        cut_face, cut_region, cut_sign = [], [], []
        idx_int = np.nonzero(interior)[0]
        for k in range(self.n_regions - 1):          # Ω_N has no interior cut
            in_l, in_r = M[k][fl], M[k][fr]
            cross = in_l != in_r
            cut_face.append(idx_int[cross])
            cut_region.append(np.full(int(cross.sum()), k))
            cut_sign.append(np.where(in_l[cross], 1.0, -1.0))

        area = mesh_data["cell_area"]
        self.s = torch.tensor(np.where(finite, s, s_max), dtype=dtype, device=device)
        self.s_edges = torch.tensor(edges, dtype=dtype, device=device)
        self.W = torch.tensor(M, dtype=dtype, device=device) * area.unsqueeze(0)
        self.cut_face = torch.tensor(np.concatenate(cut_face), dtype=torch.long, device=device)
        self.cut_region = torch.tensor(np.concatenate(cut_region), dtype=torch.long, device=device)
        self.cut_sign = torch.tensor(np.concatenate(cut_sign), dtype=dtype, device=device)

        # Boundary faces split into upstream (inflow) and the rest (outflow)
        bnd = ~interior
        self.inlet_faces = torch.tensor(np.nonzero(bnd & np.isin(face_bc_id, self.upstream_bc))[0],
                                        dtype=torch.long, device=device)
        self.outlet_faces = torch.tensor(np.nonzero(bnd & ~np.isin(face_bc_id, self.upstream_bc))[0],
                                         dtype=torch.long, device=device)

    # ------------------------------------------------------------------

    def imbalance(self, r_xi: torch.Tensor) -> torch.Tensor:
        """
        Region mass imbalance E_k = Σ_{i∈Ω_k} A_i r_i  [m³/s].

        r_xi : [..., n_cells] continuity residual ∂ξ/∂t + R_ξ (all cells)
        returns [..., N]
        """
        return r_xi @ self.W.T

    def cut_discharge(self, flux_xi: torch.Tensor) -> torch.Tensor:
        """
        Discharge crossing each interior cut, positive downstream [m³/s].

        flux_xi : [..., n_faces] mass flux × face length (left → right)
        returns [..., N-1]
        """
        signed = flux_xi.index_select(-1, self.cut_face) * self.cut_sign
        out = torch.zeros(*flux_xi.shape[:-1], self.n_regions - 1,
                          dtype=flux_xi.dtype, device=flux_xi.device)
        return out.index_add(-1, self.cut_region, signed)

    def boundary_discharge(self, flux_xi: torch.Tensor):
        """(Q_in, Q_out) through the upstream and all other boundary faces [m³/s]."""
        q_in = -flux_xi.index_select(-1, self.inlet_faces).sum(-1)
        q_out = flux_xi.index_select(-1, self.outlet_faces).sum(-1)
        return q_in, q_out
