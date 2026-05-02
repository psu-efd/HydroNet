"""
Well-balanced Roe Riemann solver and FVM residual for 2D SWE.

Follows the Hydrograd.jl formulation (Liu & Song, WRR):

Conserved variables (perturbation form):
    Q = [xi, hu, hv]
    xi = h - h_still    (free surface perturbation above still water)
    h  = xi + h_still   (total water depth)
    hu = h * u           (x-momentum / unit discharge)
    hv = h * v           (y-momentum / unit discharge)

Well-balanced SWE (Eqs. 1-3 in Hydrograd paper):
    d(xi)/dt  + d(uh)/dx + d(vh)/dy = 0
    d(uh)/dt  + d(u²h + 0.5*g*(xi² + 2*xi*h_s))/dx + d(uvh)/dy
              = -g*xi*S0x - tau_bx/rho
    d(vh)/dt  + d(uvh)/dx + d(v²h + 0.5*g*(xi² + 2*xi*h_s))/dy
              = -g*xi*S0y - tau_by/rho

The pressure term 0.5*g*(xi² + 2*xi*h_still) replaces 0.5*g*h² to achieve
exact balance at rest: when xi=0, both flux and source are zero.

Source term split (Rogers et al. 2001, 2003; Liu et al. 2008):
    g*h*d(xi)/dx = d[0.5*g*(xi² + 2*xi*h_s)]/dx - g*h*d(zb)/dx
"""

import torch

from .smooth_ops import smooth_abs, smooth_sqrt, smooth_pow

G: float = 9.81   # gravitational acceleration (m/s²)


# ---------------------------------------------------------------------------
# Roe Riemann solver for 2D SWE (well-balanced, perturbation form)
# ---------------------------------------------------------------------------

def roe_flux_2d(
    Q_L: torch.Tensor,
    Q_R: torch.Tensor,
    h_still_L: torch.Tensor,
    h_still_R: torch.Tensor,
    zb_L: torch.Tensor,
    zb_R: torch.Tensor,
    nx: torch.Tensor,
    ny: torch.Tensor,
    h_small: float = 1e-3,
) -> torch.Tensor:
    """
    Compute Roe numerical flux for 2D SWE in perturbation form.

    Following Hydrograd.jl/src/fvm/discretization/Riemman_solvers/swe_2D_solvers.jl

    Parameters
    ----------
    Q_L : [n_faces, 3]  conserved variables on left  [xi, hu, hv]
    Q_R : [n_faces, 3]  conserved variables on right [xi, hu, hv]
    h_still_L : [n_faces]  still water depth on left
    h_still_R : [n_faces]  still water depth on right
    zb_L : [n_faces]  bed elevation on left
    zb_R : [n_faces]  bed elevation on right
    nx, ny : [n_faces]  outward unit face normal (left → right)
    h_small : float  dry threshold

    Returns
    -------
    flux : [n_faces, 3]  numerical flux [xi_flux, hu_flux, hv_flux]
    """
    xi_L = Q_L[:, 0]
    hu_L = Q_L[:, 1]
    hv_L = Q_L[:, 2]
    xi_R = Q_R[:, 0]
    hu_R = Q_R[:, 1]
    hv_R = Q_R[:, 2]

    h_L = (xi_L + h_still_L).clamp(min=0.0)
    h_R = (xi_R + h_still_R).clamp(min=0.0)

    # --- Dry bed handling (following Hydrograd lines 16-76) ---
    both_dry = (h_L <= h_small) & (h_R <= h_small)
    wse_L = h_L + zb_L
    wse_R = h_R + zb_R

    # Left WSE below right bed AND right dry → wall
    left_below_right = (wse_L < (zb_R + h_small)) & (h_R <= h_small)
    # Right WSE below left bed AND left dry → wall
    right_below_left = (wse_R < (zb_L + h_small)) & (h_L <= h_small)
    # One side dry (but WSE covers opposite bed)
    left_dry_only = (h_L <= h_small) & ~left_below_right & ~both_dry
    right_dry_only = (h_R <= h_small) & ~right_below_left & ~both_dry

    # Apply wall reflection for appropriate cases
    # Left below right: reflect left state as wall
    h_R = torch.where(left_below_right, h_L, h_R)
    hu_R = torch.where(left_below_right, -hu_L, hu_R)
    hv_R = torch.where(left_below_right, -hv_L, hv_R)
    xi_R = torch.where(left_below_right, xi_L, xi_R)
    h_still_R = torch.where(left_below_right, h_still_L, h_still_R)

    # Right below left: reflect right state as wall
    h_L = torch.where(right_below_left, h_R, h_L)
    hu_L = torch.where(right_below_left, -hu_R, hu_L)
    hv_L = torch.where(right_below_left, -hv_R, hv_L)
    xi_L = torch.where(right_below_left, xi_R, xi_L)
    h_still_L = torch.where(right_below_left, h_still_R, h_still_L)

    # Clamp h for velocity computation
    h_L_safe = h_L.clamp(min=h_small)
    h_R_safe = h_R.clamp(min=h_small)

    u_L = hu_L / h_L_safe
    v_L = hv_L / h_L_safe
    u_R = hu_R / h_R_safe
    v_R = hv_R / h_R_safe

    # Normal velocities
    un_L = u_L * nx + v_L * ny
    un_R = u_R * nx + v_R * ny

    # --- Roe averages ---
    sqrt_hL = smooth_sqrt(h_L)
    sqrt_hR = smooth_sqrt(h_R)
    denom = sqrt_hL + sqrt_hR + 1e-12

    h_Roe = 0.5 * (h_L + h_R)
    u_Roe = (sqrt_hL * u_L + sqrt_hR * u_R) / denom
    v_Roe = (sqrt_hL * v_L + sqrt_hR * v_R) / denom
    un_Roe = u_Roe * nx + v_Roe * ny
    c_Roe = smooth_sqrt(torch.tensor(G, dtype=h_L.dtype, device=h_L.device) * h_Roe)

    # --- Physical fluxes (well-balanced perturbation form) ---
    # F_xi  = hu*nx + hv*ny
    # F_hu  = (hu*u + 0.5*g*(xi² + 2*xi*h_still)) * nx + hu*v * ny
    # F_hv  = hv*u * nx + (hv*v + 0.5*g*(xi² + 2*xi*h_still)) * ny
    g = G

    pressure_L = 0.5 * g * (xi_L * xi_L + 2.0 * xi_L * h_still_L)
    pressure_R = 0.5 * g * (xi_R * xi_R + 2.0 * xi_R * h_still_R)

    F_xi_L = hu_L * nx + hv_L * ny
    F_hu_L = (hu_L * u_L + pressure_L) * nx + hu_L * v_L * ny
    F_hv_L = hv_L * u_L * nx + (hv_L * v_L + pressure_L) * ny

    F_xi_R = hu_R * nx + hv_R * ny
    F_hu_R = (hu_R * u_R + pressure_R) * nx + hu_R * v_R * ny
    F_hv_R = hv_R * u_R * nx + (hv_R * v_R + pressure_R) * ny

    # --- Roe dissipation: R |Lambda| L * dQ ---
    # dQ = [xi_R - xi_L, hu_R - hu_L, hv_R - hv_L]
    dxi = xi_R - xi_L
    dhu = hu_R - hu_L
    dhv = hv_R - hv_L

    # Eigenvalues (smoothed absolute values)
    lam1 = smooth_abs(un_Roe)           # contact wave
    lam2 = smooth_abs(un_Roe - c_Roe)   # left acoustic
    lam3 = smooth_abs(un_Roe + c_Roe)   # right acoustic

    # R |Lambda| L * dQ  (following Hydrograd's matrix form)
    # L * dQ (project onto characteristic variables)
    over_2c = 0.5 / (c_Roe + 1e-12)
    alpha1 = -(u_Roe * ny - v_Roe * nx) * dxi + ny * dhu - nx * dhv
    alpha2 = (un_Roe * over_2c + 0.5) * dxi - nx * over_2c * dhu - ny * over_2c * dhv
    alpha3 = (-un_Roe * over_2c + 0.5) * dxi + nx * over_2c * dhu + ny * over_2c * dhv

    # |Lambda| * (L * dQ)
    alpha1 = lam1 * alpha1
    alpha2 = lam2 * alpha2
    alpha3 = lam3 * alpha3

    # R * |Lambda| * L * dQ (back to conserved variables)
    absA_dQ_xi = alpha2 + alpha3
    absA_dQ_hu = ny * alpha1 + (u_Roe - c_Roe * nx) * alpha2 + (u_Roe + c_Roe * nx) * alpha3
    absA_dQ_hv = -nx * alpha1 + (v_Roe - c_Roe * ny) * alpha2 + (v_Roe + c_Roe * ny) * alpha3

    # --- Roe flux = (F_L + F_R)/2 - absA_dQ/2 ---
    flux_xi = 0.5 * (F_xi_L + F_xi_R - absA_dQ_xi)
    flux_hu = 0.5 * (F_hu_L + F_hu_R - absA_dQ_hu)
    flux_hv = 0.5 * (F_hv_L + F_hv_R - absA_dQ_hv)

    # --- Handle fully dry faces ---
    flux_xi = torch.where(both_dry, torch.zeros_like(flux_xi), flux_xi)
    flux_hu = torch.where(both_dry, torch.zeros_like(flux_hu), flux_hu)
    flux_hv = torch.where(both_dry, torch.zeros_like(flux_hv), flux_hv)

    # One-sided flux for single dry side
    F_xi_R_only = hu_R * nx + hv_R * ny
    F_hu_R_only = (hu_R * u_R + 0.5 * g * smooth_pow(h_R, 2)) * nx + hu_R * v_R * ny
    F_hv_R_only = hv_R * u_R * nx + (hv_R * v_R + 0.5 * g * smooth_pow(h_R, 2)) * ny

    F_xi_L_only = hu_L * nx + hv_L * ny
    F_hu_L_only = (hu_L * u_L + 0.5 * g * smooth_pow(h_L, 2)) * nx + hu_L * v_L * ny
    F_hv_L_only = hv_L * u_L * nx + (hv_L * v_L + 0.5 * g * smooth_pow(h_L, 2)) * ny

    flux_xi = torch.where(left_dry_only, F_xi_R_only, flux_xi)
    flux_hu = torch.where(left_dry_only, F_hu_R_only, flux_hu)
    flux_hv = torch.where(left_dry_only, F_hv_R_only, flux_hv)

    flux_xi = torch.where(right_dry_only, F_xi_L_only, flux_xi)
    flux_hu = torch.where(right_dry_only, F_hu_L_only, flux_hu)
    flux_hv = torch.where(right_dry_only, F_hv_L_only, flux_hv)

    return torch.stack([flux_xi, flux_hu, flux_hv], dim=-1)


# ---------------------------------------------------------------------------
# Source terms
# ---------------------------------------------------------------------------

def compute_source_terms(
    Q_cells: torch.Tensor,
    mesh_data: dict,
    h_still: torch.Tensor,
    h_small: float = 1e-2,
) -> torch.Tensor:
    """
    Compute source terms: bed slope + Manning friction.

    Following Hydrograd.jl (semi_discretize_swe_2D.jl, lines 474-475):
        source_x = g * xi * S0_x - friction_x
        source_y = g * xi * S0_y - friction_y

    Parameters
    ----------
    Q_cells : [n_cells, 3]  conserved vars [xi, hu, hv]
    mesh_data : dict with S0_cells, cell_manning
    h_still : [n_cells]  still water depth
    h_small : float  dry threshold

    Returns
    -------
    source : [n_cells, 3]  source terms [0, source_x, source_y]
    """
    xi = Q_cells[:, 0]
    hu = Q_cells[:, 1]
    hv = Q_cells[:, 2]
    h = (xi + h_still).clamp(min=0.0)

    S0 = mesh_data["S0_cells"]          # [n_cells, 2]
    manning_n = mesh_data["cell_manning"]  # [n_cells]

    g = G

    # Wet cell mask (only apply source where wet)
    wet = (h > h_small).float()

    # --- Bed slope source (well-balanced): g * xi * S0 ---
    bed_x = g * xi * S0[:, 0]
    bed_y = g * xi * S0[:, 1]

    # --- Manning friction (discharge form, following Hydrograd line 544-546) ---
    # friction = g * n^2 / h^(7/3) * sqrt(hu^2 + hv^2) * (hu, hv)
    mag_q = smooth_sqrt(hu * hu + hv * hv)
    h_pow = smooth_pow(h + h_small, 7.0 / 3.0)
    friction_coeff = g * manning_n * manning_n / h_pow * mag_q

    friction_x = friction_coeff * hu
    friction_y = friction_coeff * hv

    # Combined source (zero for dry cells)
    source = torch.zeros_like(Q_cells)
    source[:, 1] = wet * (bed_x - friction_x)
    source[:, 2] = wet * (bed_y - friction_y)

    return source


# ---------------------------------------------------------------------------
# Full FVM residual
# ---------------------------------------------------------------------------

def compute_fvm_residual(
    Q_cells: torch.Tensor,
    mesh_data: dict,
    h_still: torch.Tensor,
    h_small: float = 1e-2,
) -> torch.Tensor:
    """
    Compute FVM residual for the well-balanced SWE.

    dQ/dt = -flux_divergence + source_terms
    Residual R = flux_divergence - source_terms
    PINN loss enforces: dQ/dt + R = 0

    Parameters
    ----------
    Q_cells : [n_cells, 3]  conserved vars [xi, hu, hv]
    mesh_data : dict with face/cell topology and bed data
    h_still : [n_cells]  still water depth reference
    h_small : float  dry threshold

    Returns
    -------
    residual : [n_cells, 3]  R = flux_divergence - source
    """
    face_left = mesh_data["face_left"]
    face_right = mesh_data["face_right"]
    face_normal = mesh_data["face_normal"]
    nx = face_normal[:, 0]
    ny = face_normal[:, 1]
    face_length = mesh_data["face_length"]
    cell_area = mesh_data["cell_area"]
    bed_elev = mesh_data["bed_elev"]
    n_cells = Q_cells.shape[0]

    interior_mask = face_right >= 0

    # --- Gather left/right states ---
    Q_L = Q_cells[face_left]
    hs_L = h_still[face_left]
    zb_L = bed_elev[face_left]

    Q_R = torch.zeros_like(Q_L)
    hs_R = torch.zeros_like(hs_L)
    zb_R = torch.zeros_like(zb_L)

    if interior_mask.any():
        idx_r = face_right[interior_mask]
        Q_R[interior_mask] = Q_cells[idx_r]
        hs_R[interior_mask] = h_still[idx_r]
        zb_R[interior_mask] = bed_elev[idx_r]

    # --- Boundary ghost cells ---
    bnd_mask = ~interior_mask
    if bnd_mask.any():
        Q_R[bnd_mask], hs_R[bnd_mask], zb_R[bnd_mask] = _build_ghost_states(
            Q_L[bnd_mask], hs_L[bnd_mask], zb_L[bnd_mask],
            nx[bnd_mask], ny[bnd_mask],
            mesh_data, bnd_mask, h_small,
        )

    # --- Roe flux ---
    flux = roe_flux_2d(Q_L, Q_R, hs_L, hs_R, zb_L, zb_R, nx, ny, h_small)

    # --- Scatter fluxes to cells ---
    flux_scaled = flux * face_length.unsqueeze(-1)

    flux_div = torch.zeros(n_cells, 3, dtype=Q_cells.dtype, device=Q_cells.device)
    flux_div.scatter_add_(0, face_left.unsqueeze(-1).expand_as(flux_scaled), flux_scaled)

    if interior_mask.any():
        idx_r = face_right[interior_mask]
        flux_div.scatter_add_(
            0, idx_r.unsqueeze(-1).expand(-1, 3),
            -flux_scaled[interior_mask],
        )

    flux_div = flux_div / cell_area.unsqueeze(-1)

    # --- Source terms ---
    source = compute_source_terms(Q_cells, mesh_data, h_still, h_small)

    # Residual: R = flux_divergence - source
    return flux_div - source


# ---------------------------------------------------------------------------
# Ghost cell construction
# ---------------------------------------------------------------------------

def _build_ghost_states(
    Q_L: torch.Tensor,
    hs_L: torch.Tensor,
    zb_L: torch.Tensor,
    nx: torch.Tensor,
    ny: torch.Tensor,
    mesh_data: dict,
    bnd_mask: torch.Tensor,
    h_small: float,
):
    """
    Build ghost states for boundary faces.

    Default: reflective (wall/symmetry).
    Overrides for inlet-q and exit-h via mesh_data["bc_ghost"].
    """
    # Default: reflective ghost (wall/symmetry)
    Q_ghost = _reflective_ghost(Q_L, nx, ny)
    hs_ghost = hs_L.clone()
    zb_ghost = zb_L.clone()

    # Override for specific BC types
    bc_ghost = mesh_data.get("bc_ghost", {})
    if bc_ghost and "face_bc_id" in mesh_data:
        face_bc_id = mesh_data["face_bc_id"]
        # Get the bc_ids for the boundary faces only
        bc_ids_at_bnd = face_bc_id[bnd_mask]

        # Per-face length and Manning n at boundary faces (for conveyance BC)
        face_length_bnd = mesh_data["face_length"][bnd_mask]
        manning_bnd = mesh_data["cell_manning"][mesh_data["face_left"][bnd_mask]]

        for bc_id, bc_info in bc_ghost.items():
            bc_type = bc_info["type"]
            bc_val = bc_info.get("value", 0.0)

            # Which boundary faces have this BC ID
            local_mask = (bc_ids_at_bnd == bc_id)
            if not local_mask.any():
                continue

            if bc_type == "inlet-q":
                Q_ghost[local_mask] = _inlet_q_ghost(
                    Q_L[local_mask], hs_L[local_mask],
                    nx[local_mask], ny[local_mask],
                    face_length_bnd[local_mask],
                    manning_bnd[local_mask],
                    bc_val, h_small,
                )
            elif bc_type == "exit-h":
                Q_ghost[local_mask], hs_ghost[local_mask] = _exit_h_ghost(
                    Q_L[local_mask], hs_L[local_mask],
                    zb_L[local_mask], bc_val,
                )

    return Q_ghost, hs_ghost, zb_ghost


def _reflective_ghost(
    Q_L: torch.Tensor, nx: torch.Tensor, ny: torch.Tensor
) -> torch.Tensor:
    """Reflective (wall/symmetry) ghost: mirror normal momentum."""
    xi = Q_L[:, 0]
    hu = Q_L[:, 1]
    hv = Q_L[:, 2]

    # Normal momentum component
    hun = hu * nx + hv * ny

    return torch.stack([
        xi,
        hu - 2.0 * hun * nx,
        hv - 2.0 * hun * ny,
    ], dim=-1)


def _inlet_q_ghost(
    Q_L: torch.Tensor,
    hs_L: torch.Tensor,
    nx: torch.Tensor,
    ny: torch.Tensor,
    face_length: torch.Tensor,
    manning_n: torch.Tensor,
    Q_total: float,
    h_small: float,
) -> torch.Tensor:
    """
    Inlet discharge ghost state, Manning-conveyance distribution.

    Matches SRH-2D's "IQParams ... CONVEYANCE" option and Hydrograd.jl
    (bc_2D.jl, lines 663-694):

        total_A  = sum_f [ L_f^(5/3) * h_int_f / n_f * wet_f ]
        v_n,f    = (Q_total / total_A) * L_f^(2/3) / n_f
        (hu,hv)_ghost,f = -h_int_f * v_n,f * (nx,ny)_f  * wet_f

    Ghost depth equals interior depth. No tangential momentum is imposed.

    Parameters
    ----------
    Q_L         : [n_face, 3]  interior conserved vars on inlet faces
    hs_L        : [n_face]     still-water depth on interior side
    nx, ny      : [n_face]     outward unit normal at each inlet face
    face_length : [n_face]     edge length of each inlet face
    manning_n   : [n_face]     Manning's n of the interior cell adjacent to each face
    Q_total     : float        total volumetric discharge into the domain (m^3/s)
    h_small     : float        wet/dry depth threshold
    """
    xi_L = Q_L[:, 0]
    h_L = (xi_L + hs_L).clamp(min=0.0)

    wet = (h_L > h_small).to(h_L.dtype)

    # Conveyance weight per face: L^(5/3) * h / n, zero on dry faces
    conveyance = face_length.pow(5.0 / 3.0) * h_L / manning_n * wet
    total_A = conveyance.sum().clamp(min=1e-12)

    # Manning velocity (normal, positive into domain) per face
    v_n = (Q_total / total_A) * face_length.pow(2.0 / 3.0) / manning_n

    # (hu, hv) pointing into the domain: magnitude h * v_n along -n
    hu_ghost = -h_L * v_n * nx * wet
    hv_ghost = -h_L * v_n * ny * wet

    # Ghost depth same as interior (xi = h - h_still; hs is shared)
    return torch.stack([xi_L, hu_ghost, hv_ghost], dim=-1)


def _exit_h_ghost(
    Q_L: torch.Tensor,
    hs_L: torch.Tensor,
    zb_L: torch.Tensor,
    wse_exit: float,
):
    """
    Exit ghost state from a prescribed water-surface elevation.

    Matches Hydrograd.jl (bc_2D.jl, line 763-770): per-face ghost depth
    is h_ghost = max(wse_exit - zb_L, 0); momentum (hu, hv) is zero-
    gradient (extrapolated from the interior). This preserves the
    discharge crossing the exit, which is the conserved quantity, rather
    than preserving velocity and letting the discharge drift when
    h_ghost != h_L.

    Returns (Q_ghost, hs_ghost).
    """
    hu_L = Q_L[:, 1]
    hv_L = Q_L[:, 2]

    # Ghost depth from prescribed WSE and local bed
    h_ghost = (wse_exit - zb_L).clamp(min=0.0)
    xi_ghost = h_ghost - hs_L

    # Zero-gradient on momentum (not velocity)
    Q_ghost = torch.stack([xi_ghost, hu_L, hv_L], dim=-1)
    hs_ghost = hs_L   # same reference

    return Q_ghost, hs_ghost
