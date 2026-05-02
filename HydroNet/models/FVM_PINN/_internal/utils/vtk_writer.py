"""
Simple VTK legacy ASCII writer for unstructured 2D mesh solutions.
Writes cell-centred data (h, u, v, WSE) for Paraview/VisIt visualisation.
"""

import numpy as np
from pathlib import Path


def write_vtk_solution(
    path: "str | Path",
    mesh,
    U: np.ndarray,
    t: float,
    S0_cells: "np.ndarray | None" = None,
) -> None:
    """
    Write FVM-PINN solution to VTK legacy format.

    Parameters
    ----------
    path      : output .vtk file path
    mesh      : UnstructuredMesh (provides bed_elev and cell_manning)
    U         : [n_cells, 3]  conserved variables [h, hu, hv]
    t         : simulation time
    S0_cells  : [n_cells, 2]  optional bed slope vector (S0_x, S0_y)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    nodes_xy = mesh.node_xy         # [n_nodes, 2]
    nodes_z = mesh.node_z           # [n_nodes]
    elem_nodes = mesh.cell_nodes    # list of node arrays
    n_nodes = len(nodes_xy)
    n_cells = len(elem_nodes)

    h = U[:, 0]
    u = U[:, 1] / (h + 1e-8)
    v = U[:, 2] / (h + 1e-8)
    wse = h + mesh.bed_elev

    with open(path, "w") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write(f"FVM-PINN SWE solution t={t:.4f}\n")
        f.write("ASCII\n")
        f.write("DATASET UNSTRUCTURED_GRID\n\n")

        # Points
        f.write(f"POINTS {n_nodes} double\n")
        for i in range(n_nodes):
            f.write(f"{nodes_xy[i, 0]:.6f} {nodes_xy[i, 1]:.6f} {nodes_z[i]:.6f}\n")

        # Cells
        total_ids = sum(len(ns) + 1 for ns in elem_nodes)
        f.write(f"\nCELLS {n_cells} {total_ids}\n")
        vtk_type = []
        for ns in elem_nodes:
            n_verts = len(ns)
            ids = " ".join(str(int(n)) for n in ns)
            f.write(f"{n_verts} {ids}\n")
            vtk_type.append(9 if n_verts == 4 else 5)   # VTK_QUAD=9, VTK_TRIANGLE=5

        f.write(f"\nCELL_TYPES {n_cells}\n")
        for vt in vtk_type:
            f.write(f"{vt}\n")

        # Cell data
        f.write(f"\nCELL_DATA {n_cells}\n")

        scalar_fields = [
            ("Water_Depth_m", h),
            ("Velocity_X_m_s", u),
            ("Velocity_Y_m_s", v),
            ("WSE_m", wse),
            ("Bed_Elevation_m", np.asarray(mesh.bed_elev)),
            ("Manning_n", np.asarray(mesh.cell_manning)),
        ]

        for name, values in scalar_fields:
            f.write(f"\nSCALARS {name} double 1\n")
            f.write("LOOKUP_TABLE default\n")
            for val in values:
                f.write(f"{val:.6f}\n")

        # Velocity vector
        f.write("\nVECTORS Velocity_m_s double\n")
        for i in range(n_cells):
            f.write(f"{u[i]:.6f} {v[i]:.6f} 0.0\n")

        # Bed slope vector S0 = -grad(zb)
        if S0_cells is not None:
            S0 = np.asarray(S0_cells)
            f.write("\nVECTORS Bed_Slope_S0 double\n")
            for i in range(n_cells):
                f.write(f"{S0[i, 0]:.6e} {S0[i, 1]:.6e} 0.0\n")
