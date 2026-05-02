"""
Unstructured mesh topology builder.

Converts raw SRH-2D node/element data into a face-based data structure
suitable for FVM flux computation:
  - cell centres
  - face connectivity (interior + boundary)
  - outward face normal vectors (unit)
  - face lengths
  - cell areas
  - cell-to-face and face-to-cell mappings
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .srh2d_reader import SRH2DRawData

logger = logging.getLogger(__name__)


@dataclass
class UnstructuredMesh:
    """
    Face-based unstructured mesh topology.

    Conventions
    -----------
    - All indices are 0-based.
    - Interior faces: face_left[f] >= 0 and face_right[f] >= 0
    - Boundary faces: face_right[f] == -1 (ghost / exterior)
    - face_normal[f] points from left cell to right cell (outward for right cell).
    """

    # ---- Node data ----
    node_xy: np.ndarray = None      # [n_nodes, 2]  x, y coordinates
    node_z: np.ndarray = None       # [n_nodes]     bed elevation z

    # ---- Cell data ----
    n_cells: int = 0
    cell_center: np.ndarray = None  # [n_cells, 2]
    cell_area: np.ndarray = None    # [n_cells]
    cell_nodes: List[np.ndarray] = field(default_factory=list)  # list of node-index arrays
    cell_manning: np.ndarray = None # [n_cells]  Manning's n per cell

    # ---- Face data ----
    n_faces: int = 0
    face_center: np.ndarray = None  # [n_faces, 2]
    face_normal: np.ndarray = None  # [n_faces, 2]  unit outward normal (for left cell)
    face_length: np.ndarray = None  # [n_faces]
    face_left: np.ndarray = None    # [n_faces]  left-cell index
    face_right: np.ndarray = None   # [n_faces]  right-cell index (-1 = boundary)
    face_bc_id: np.ndarray = None   # [n_faces]  BC ID for boundary faces, -1 otherwise

    # ---- Cell-face adjacency ----
    cell_faces: List[np.ndarray] = field(default_factory=list)  # list of face-index arrays
    cell_face_signs: List[np.ndarray] = field(default_factory=list)  # +1 / -1 per face

    # ---- Boundary data ----
    # bc_id -> array of face indices
    bc_faces: Dict[int, np.ndarray] = field(default_factory=dict)

    # ---- Bed elevation at cell centres ----
    bed_elev: np.ndarray = None     # [n_cells]  z_b (average of node z values)


def build_mesh(raw: SRH2DRawData) -> UnstructuredMesh:
    """
    Build UnstructuredMesh from SRH2DRawData.

    Steps
    -----
    1. Store node coordinates.
    2. Compute cell centres and areas.
    3. Extract all edges (faces in 2D).
    4. Deduplicate → interior faces (shared by 2 cells) and boundary faces.
    5. Compute face normals and lengths.
    6. Assign boundary conditions from node-string data.
    """
    mesh = UnstructuredMesh()

    coords = raw.node_coords            # [n_nodes, 3]
    mesh.node_xy = coords[:, :2].copy()
    mesh.node_z = coords[:, 2].copy()

    elem_nodes = raw.elem_nodes
    n_cells = len(elem_nodes)
    mesh.n_cells = n_cells
    mesh.cell_nodes = elem_nodes

    # Manning's n per cell
    mesh.cell_manning = _assign_manning(raw)

    # Cell centres and areas
    mesh.cell_center = np.zeros((n_cells, 2), dtype=np.float64)
    mesh.cell_area = np.zeros(n_cells, dtype=np.float64)
    mesh.bed_elev = np.zeros(n_cells, dtype=np.float64)

    for ci, ns in enumerate(elem_nodes):
        pts = mesh.node_xy[ns]                  # [nv, 2]
        mesh.cell_center[ci] = pts.mean(axis=0)
        mesh.cell_area[ci] = _polygon_area(pts)
        mesh.bed_elev[ci] = mesh.node_z[ns].mean()

    # Build face structure
    _build_faces(mesh, elem_nodes, raw)

    logger.info(
        f"Mesh topology: {n_cells} cells, {mesh.n_faces} faces "
        f"({(mesh.face_right >= 0).sum()} interior, "
        f"{(mesh.face_right < 0).sum()} boundary)"
    )
    return mesh


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _assign_manning(raw: SRH2DRawData) -> np.ndarray:
    """
    Assign Manning's n per cell from material zone lookup.

    Manning's n mapping (from .srhhydro ManningsN lines):
      - ID 0: default value (cells not in any named material zone)
      - ID 1..N: per-zone values

    Cells with material ID not found in manning_n fall back to
    the default (ID 0), then to 0.03 as last resort.
    """
    n_cells = len(raw.elem_nodes)
    default_n = raw.manning_n.get(0, 0.03)
    result = np.full(n_cells, default_n, dtype=np.float64)
    for ci in range(n_cells):
        mat = int(raw.elem_material[ci])
        result[ci] = raw.manning_n.get(mat, default_n)
    return result


def _polygon_area(pts: np.ndarray) -> float:
    """Shoelace formula for a polygon given vertices [n, 2]."""
    n = len(pts)
    x, y = pts[:, 0], pts[:, 1]
    area = 0.5 * abs(
        sum(x[i] * y[(i + 1) % n] - x[(i + 1) % n] * y[i] for i in range(n))
    )
    return area


def _build_faces(
    mesh: UnstructuredMesh,
    elem_nodes: List[np.ndarray],
    raw: SRH2DRawData,
) -> None:
    """
    Build face connectivity, normals, and lengths.

    Uses a dict keyed by sorted (nodeA, nodeB) tuples to detect shared edges.
    """
    # Map (minN, maxN) -> [left_cell, right_cell]
    edge_map: Dict[Tuple[int, int], List[int]] = {}

    for ci, ns in enumerate(elem_nodes):
        nv = len(ns)
        for k in range(nv):
            a = int(ns[k])
            b = int(ns[(k + 1) % nv])
            key = (min(a, b), max(a, b))
            if key not in edge_map:
                # Store as [left_cell, right_cell, raw_a, raw_b]
                # raw_a/b preserve orientation for normal direction
                edge_map[key] = [ci, -1, a, b]
            else:
                edge_map[key][1] = ci   # second cell = right cell

    n_faces = len(edge_map)
    mesh.n_faces = n_faces

    face_center = np.zeros((n_faces, 2), dtype=np.float64)
    face_normal = np.zeros((n_faces, 2), dtype=np.float64)
    face_length = np.zeros(n_faces, dtype=np.float64)
    face_left = np.zeros(n_faces, dtype=np.int64)
    face_right = np.full(n_faces, -1, dtype=np.int64)
    face_bc_id = np.full(n_faces, -1, dtype=np.int64)

    # cell -> list of face indices and signs
    cell_faces_raw: List[List[int]] = [[] for _ in range(mesh.n_cells)]
    cell_face_signs_raw: List[List[int]] = [[] for _ in range(mesh.n_cells)]

    xy = mesh.node_xy
    for fi, (key, val) in enumerate(edge_map.items()):
        left_cell, right_cell, a, b = val[0], val[1], val[2], val[3]
        pa = xy[a]
        pb = xy[b]
        face_center[fi] = 0.5 * (pa + pb)
        dx = pb[0] - pa[0]
        dy = pb[1] - pa[1]
        length = np.sqrt(dx**2 + dy**2)
        face_length[fi] = length

        # Outward normal for left cell: rotate edge vector 90° to the right
        # If left-cell centre is to the left of the edge vector, this is outward
        nx = dy / length
        ny = -dx / length

        # Verify normal points away from left cell
        cc = mesh.cell_center[left_cell]
        mid = face_center[fi]
        if (nx * (cc[0] - mid[0]) + ny * (cc[1] - mid[1])) > 0:
            # Normal points toward left cell — flip
            nx, ny = -nx, -ny

        face_normal[fi] = [nx, ny]
        face_left[fi] = left_cell
        face_right[fi] = right_cell

        cell_faces_raw[left_cell].append(fi)
        cell_face_signs_raw[left_cell].append(+1)   # outward for left
        if right_cell >= 0:
            cell_faces_raw[right_cell].append(fi)
            cell_face_signs_raw[right_cell].append(-1)  # inward for right

    mesh.face_center = face_center
    mesh.face_normal = face_normal
    mesh.face_length = face_length
    mesh.face_left = face_left
    mesh.face_right = face_right
    mesh.face_bc_id = face_bc_id

    mesh.cell_faces = [np.array(f, dtype=np.int64) for f in cell_faces_raw]
    mesh.cell_face_signs = [np.array(s, dtype=np.float64) for s in cell_face_signs_raw]

    # Assign BC IDs to boundary faces
    _assign_bc_to_faces(mesh, raw)


def _assign_bc_to_faces(mesh: UnstructuredMesh, raw: SRH2DRawData) -> None:
    """
    Map node-string boundary IDs to boundary face indices.

    A boundary face (a,b) belongs to BC i if both its nodes are in the
    node string for BC i.
    """
    # Build set per bc_id
    bc_node_sets: Dict[int, set] = {
        bc_id: set(nids) for bc_id, nids in raw.node_strings.items()
    }

    # Map (min,max) -> face index for boundary faces
    bf_map: Dict[Tuple[int, int], int] = {}
    for fi in range(mesh.n_faces):
        if mesh.face_right[fi] < 0:
            # Find the two nodes of this face
            left_cell = mesh.face_left[fi]
            fc = mesh.face_center[fi]
            ns = mesh.cell_nodes[left_cell]
            a, b = _find_face_nodes(mesh.node_xy, ns, fc)
            if a is not None:
                bf_map[(min(a, b), max(a, b))] = fi

    # Assign BC IDs
    bc_faces_raw: Dict[int, List[int]] = {}
    for bc_id, node_set in bc_node_sets.items():
        for (a, b), fi in bf_map.items():
            if a in node_set and b in node_set:
                mesh.face_bc_id[fi] = bc_id
                bc_faces_raw.setdefault(bc_id, []).append(fi)

    mesh.bc_faces = {k: np.array(v, dtype=np.int64) for k, v in bc_faces_raw.items()}


def _find_face_nodes(
    xy: np.ndarray,
    cell_nodes: np.ndarray,
    face_center: np.ndarray,
    tol: float = 1e-6,
) -> Tuple[Optional[int], Optional[int]]:
    """
    Find the two consecutive nodes in cell_nodes whose midpoint matches face_center.
    Returns 0-indexed node IDs.
    """
    nv = len(cell_nodes)
    for k in range(nv):
        a = cell_nodes[k]
        b = cell_nodes[(k + 1) % nv]
        mid = 0.5 * (xy[a] + xy[b])
        if np.linalg.norm(mid - face_center) < tol * (np.linalg.norm(xy[a] - xy[b]) + 1e-12):
            return int(a), int(b)
    return None, None
