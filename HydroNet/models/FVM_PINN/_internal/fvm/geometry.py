"""
Utility: convert UnstructuredMesh geometry to PyTorch tensors.
Returns a mesh_data dict ready for compute_fvm_residual().

Includes precomputation of bed slope S0 via the Green-Gauss
divergence theorem, following the approach in Hydrograd.jl.
"""

import torch
from typing import Dict

from ..mesh.mesh_topology import UnstructuredMesh


def compute_cell_geometry(mesh: UnstructuredMesh, device: str = "cpu") -> Dict[str, torch.Tensor]:
    """
    Pack mesh topology arrays into a dict of PyTorch tensors.

    Also precomputes bed slope S0_cells via the Green-Gauss gradient.

    Parameters
    ----------
    mesh   : UnstructuredMesh
    device : torch device string

    Returns
    -------
    dict with keys:
        face_left, face_right, face_normal, face_length,
        face_bc_id, cell_area, bed_elev, cell_center, cell_manning,
        S0_cells, n_cells, n_faces
    """
    def t(arr, dtype=torch.float64):
        return torch.tensor(arr, dtype=dtype, device=device)

    face_left = torch.tensor(mesh.face_left, dtype=torch.long, device=device)
    face_right = torch.tensor(mesh.face_right, dtype=torch.long, device=device)
    face_normal = t(mesh.face_normal)
    face_length = t(mesh.face_length)
    cell_area = t(mesh.cell_area)
    bed_elev = t(mesh.bed_elev)

    # Precompute bed slope S0 = -grad(zb) via Green-Gauss divergence theorem
    S0_cells = _compute_bed_slope_S0(
        bed_elev, face_left, face_right, face_normal, face_length,
        cell_area, mesh.n_cells,
    )

    return {
        "face_left":    face_left,
        "face_right":   face_right,
        "face_normal":  face_normal,
        "face_length":  face_length,
        "face_center":  t(mesh.face_center),
        "face_bc_id":   torch.tensor(mesh.face_bc_id, dtype=torch.long, device=device),
        "cell_area":    cell_area,
        "bed_elev":     bed_elev,
        "cell_center":  t(mesh.cell_center),
        "cell_manning": t(mesh.cell_manning),
        "S0_cells":     S0_cells,
        "n_cells":      mesh.n_cells,
        "n_faces":      mesh.n_faces,
    }


def _compute_bed_slope_S0(
    bed_elev: torch.Tensor,
    face_left: torch.Tensor,
    face_right: torch.Tensor,
    face_normal: torch.Tensor,
    face_length: torch.Tensor,
    cell_area: torch.Tensor,
    n_cells: int,
) -> torch.Tensor:
    """
    Compute bed slope S0 = -grad(zb) at cell centers using the
    Green-Gauss divergence theorem.

    Following Hydrograd.jl (fvm_schemes_2D.jl, process_bed_2D.jl):

    Step 1: Interpolate zb from cells to faces (simple average).
    Step 2: Green-Gauss gradient:
        grad(zb)_i = (1/A_i) * sum_faces [ zb_face * n_outward * L_face ]
    Step 3: S0 = -grad(zb)

    Parameters
    ----------
    bed_elev   : [n_cells]  bed elevation at cell centers
    face_left  : [n_faces]  left cell index per face
    face_right : [n_faces]  right cell index (-1 for boundary)
    face_normal: [n_faces, 2]  unit normal (from left to right)
    face_length: [n_faces]  face edge lengths
    cell_area  : [n_cells]  cell areas
    n_cells    : int

    Returns
    -------
    S0_cells : [n_cells, 2]  bed slope vector (S0_x, S0_y) per cell
    """
    interior_mask = face_right >= 0

    # Step 1: Interpolate zb to faces
    zb_L = bed_elev[face_left]
    zb_R = torch.zeros_like(zb_L)
    if interior_mask.any():
        zb_R[interior_mask] = bed_elev[face_right[interior_mask]]
    # Boundary faces: zero gradient → zb_face = zb_left
    bnd_mask = ~interior_mask
    if bnd_mask.any():
        zb_R[bnd_mask] = zb_L[bnd_mask]

    zb_face = 0.5 * (zb_L + zb_R)   # [n_faces]

    # Step 2: Green-Gauss gradient
    # For left cell: outward normal = +face_normal
    # For right cell: outward normal = -face_normal
    nx = face_normal[:, 0]
    ny = face_normal[:, 1]

    # Contribution: zb_face * n * L
    contrib_x = zb_face * nx * face_length   # [n_faces]
    contrib_y = zb_face * ny * face_length

    # Scatter to cells
    grad_zb = torch.zeros(n_cells, 2, dtype=bed_elev.dtype, device=bed_elev.device)

    # Left cell gets +normal contribution
    grad_zb[:, 0].scatter_add_(0, face_left, contrib_x)
    grad_zb[:, 1].scatter_add_(0, face_left, contrib_y)

    # Right cell gets -normal contribution (outward normal is flipped)
    if interior_mask.any():
        idx_r = face_right[interior_mask]
        grad_zb[:, 0].scatter_add_(0, idx_r, -contrib_x[interior_mask])
        grad_zb[:, 1].scatter_add_(0, idx_r, -contrib_y[interior_mask])

    # Divide by cell area
    grad_zb = grad_zb / cell_area.unsqueeze(-1)

    # Step 3: S0 = -grad(zb)
    S0_cells = -grad_zb

    return S0_cells
