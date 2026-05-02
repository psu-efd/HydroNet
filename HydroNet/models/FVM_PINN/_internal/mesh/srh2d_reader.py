"""
SRH-2D case file reader.

Parses the SMS 2D mesh format (.srhgeom) used by SRH-2D.
Also reads Manning's n from (.srhmat) and boundary conditions from (.srhhydro).

SRH-2D geometry file (.srhgeom) is in SMS 2DM format:
    MESH2D
    ND  nodeID  x  y  z        # node definition
    E3T elemID n1 n2 n3 matID  # triangular element
    E4Q elemID n1 n2 n3 n4 matID  # quad element
    NS  n1 n2 ... -nLast  bcID  # node string (boundary)
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BoundaryCondition:
    """Stores a single boundary condition definition."""
    bc_id: int
    bc_type: str        # e.g., "inlet-q", "exit-h", "wall"
    bc_name: str
    value: Optional[float] = None
    node_ids: List[int] = field(default_factory=list)


@dataclass
class SRH2DRawData:
    """Raw parsed data from SRH-2D case files."""
    # Node data: shape [n_nodes, 3] — (x, y, z_bed)
    node_coords: np.ndarray = None

    # Element connectivity (0-indexed): list of arrays, each [n_verts_in_elem]
    elem_nodes: List[np.ndarray] = field(default_factory=list)

    # Material ID per element (0-indexed material zones)
    elem_material: np.ndarray = None

    # Manning's n per material zone: {material_id: n_value}
    manning_n: Dict[int, float] = field(default_factory=dict)

    # Boundary conditions
    boundary_conditions: List[BoundaryCondition] = field(default_factory=list)

    # Node strings: {bc_id: [node_ids]} (0-indexed)
    node_strings: Dict[int, List[int]] = field(default_factory=dict)

    # Simulation metadata from .srhhydro
    case_name: str = ""
    time_type: str = "STEADY"          # STEADY or UNSTEADY
    end_time: float = 0.0              # seconds
    time_step: float = 0.0             # seconds
    init_cond: str = ""                # DRY, RESTART, etc.


class SRH2DMeshReader:
    """
    Reads SRH-2D case files and returns structured raw data.

    Supported files:
      - <case>.srhgeom  : SMS 2DM mesh format (nodes + elements)
      - <case>.srhmat   : Manning's n per material zone
      - <case>.srhhydro : Simulation parameters and BCs

    Usage:
        reader = SRH2DMeshReader("path/to/case.srhhydro")
        raw = reader.read()
    """

    def __init__(self, hydro_file: str) -> None:
        self.hydro_path = Path(hydro_file)
        if not self.hydro_path.exists():
            raise FileNotFoundError(f"SRH-2D hydro file not found: {hydro_file}")
        self.case_dir = self.hydro_path.parent
        logger.info(f"SRH2DMeshReader initialised: {self.hydro_path}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def read(self) -> SRH2DRawData:
        """Parse all case files and return SRH2DRawData."""
        raw = SRH2DRawData()
        raw.case_name = self.hydro_path.stem

        self._read_hydro(raw)
        geom_file = self._locate_file(raw, ".srhgeom")
        mat_file = self._locate_file(raw, ".srhmat")

        self._read_geom(geom_file, raw)
        if mat_file is not None:
            self._read_mat(mat_file, raw)
        else:
            logger.warning("No .srhmat file found; defaulting Manning's n = 0.03")
            raw.manning_n = {0: 0.03}

        n_nodes = len(raw.node_coords)
        n_elems = len(raw.elem_nodes)
        logger.info(f"Mesh loaded: {n_nodes} nodes, {n_elems} elements")
        return raw

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _locate_file(self, raw: SRH2DRawData, ext: str) -> Optional[Path]:
        """Find companion file with given extension next to .srhhydro."""
        candidate = self.case_dir / (raw.case_name + ext)
        if candidate.exists():
            return candidate
        # Search for any file with this extension in the case directory
        matches = list(self.case_dir.glob(f"*{ext}"))
        if matches:
            logger.info(f"Found {ext}: {matches[0]}")
            return matches[0]
        logger.warning(f"No {ext} file found in {self.case_dir}")
        return None

    # ------------------------------------------------------------------
    # .srhhydro parser
    # ------------------------------------------------------------------

    def _read_hydro(self, raw: SRH2DRawData) -> None:
        """Parse .srhhydro file for simulation parameters and BCs."""
        lines = self.hydro_path.read_text().splitlines()

        # Temp storage for v30-format BC params
        bc_types: Dict[int, str] = {}   # bc_id -> type string
        bc_values: Dict[int, float] = {}

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            if not line or line.startswith("//"):
                i += 1
                continue

            tokens = line.split()
            key = tokens[0].lower() if tokens else ""

            if key == "steady-or-unsteady":
                raw.time_type = tokens[1].upper() if len(tokens) > 1 else "STEADY"

            elif key == "simulation-time-step":
                raw.time_step = float(tokens[1]) if len(tokens) > 1 else 0.0

            elif key == "simulation-end-time":
                raw.end_time = float(tokens[1]) if len(tokens) > 1 else 0.0

            elif key == "simtime":
                # v30 format: SimTime t_start_hr dt_sec t_end_hr
                # t_start and t_end are in hours; dt is in seconds
                if len(tokens) >= 4:
                    raw.time_step = float(tokens[2])           # seconds
                    raw.end_time = float(tokens[3]) * 3600.0   # hours → seconds

            elif key == "initcondoption":
                # InitCondOption DRY | RESTART | ...
                raw.init_cond = tokens[1].upper() if len(tokens) > 1 else ""

            elif key == "case":
                # Case "name"
                raw.case_name = " ".join(tokens[1:]).strip('"')

            elif key == "manningsn":
                # v30 format: ManningsN matID value
                if len(tokens) >= 3:
                    mat_id = int(tokens[1])
                    n_val = float(tokens[2])
                    raw.manning_n[mat_id] = n_val

            elif key == "bc":
                if len(tokens) >= 3:
                    bc_id = int(tokens[1])
                    bc_type_raw = tokens[2].upper()
                    # Map SRH-2D BC types to internal names
                    if bc_type_raw == "INLET-Q":
                        bc_types[bc_id] = "inlet-q"
                    elif bc_type_raw in ("EXIT-H", "EXIT-WS"):
                        bc_types[bc_id] = "exit-h"
                    elif bc_type_raw == "WALL":
                        bc_types[bc_id] = "wall"
                    elif bc_type_raw == "SYMM":
                        bc_types[bc_id] = "symmetry"
                    else:
                        bc_types[bc_id] = bc_type_raw.lower()

            elif key == "iqparams":
                # IQParams bcID value unit method
                if len(tokens) >= 3:
                    bc_id = int(tokens[1])
                    bc_values[bc_id] = float(tokens[2])

            elif key in ("ewsparamsc", "ewsparams"):
                # EWSParamsC / EWSParams bcID value unit method
                if len(tokens) >= 3:
                    bc_id = int(tokens[1])
                    bc_values[bc_id] = float(tokens[2])

            i += 1

        # Build BC objects from v30-format entries
        for bc_id, bc_type in bc_types.items():
            bc = BoundaryCondition(
                bc_id=bc_id,
                bc_type=bc_type,
                bc_name=f"BC_{bc_id}",
                value=bc_values.get(bc_id),
            )
            raw.boundary_conditions.append(bc)

    def _parse_bc_line(self, tokens: List[str]) -> Optional[BoundaryCondition]:
        """Parse a BC line from .srhhydro."""
        try:
            bc_id = int(tokens[1])
            bc_name = tokens[2]
            bc_type = tokens[3].lower()
            value = float(tokens[4]) if len(tokens) > 4 else None
            return BoundaryCondition(
                bc_id=bc_id, bc_type=bc_type, bc_name=bc_name, value=value
            )
        except (IndexError, ValueError) as e:
            logger.debug(f"Skipping BC line: {tokens} ({e})")
            return None

    # ------------------------------------------------------------------
    # .srhgeom parser (SMS 2DM format)
    # ------------------------------------------------------------------

    def _read_geom(self, geom_file: Path, raw: SRH2DRawData) -> None:
        """Parse SMS 2DM .srhgeom file."""
        if geom_file is None:
            raise FileNotFoundError("No .srhgeom file found")

        nodes: Dict[int, np.ndarray] = {}   # nodeID -> [x, y, z]
        elems: Dict[int, Tuple[np.ndarray, int]] = {}  # elemID -> (nodes_0idx, matID)
        node_strings: Dict[int, List[int]] = {}        # bcID -> [nodeIDs_0idx]
        current_ns_bc_id: Optional[int] = None         # track NodeString continuation

        with open(geom_file) as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                tokens = line.split()
                record = tokens[0].upper()

                # Check if this is a continuation of a NodeString
                # (line starts with a number, not a keyword)
                if current_ns_bc_id is not None and record.lstrip("-").isdigit():
                    # Continuation line: just node IDs
                    for tok in tokens:
                        try:
                            nid = int(tok)
                            node_strings[current_ns_bc_id].append(abs(nid) - 1)
                        except ValueError:
                            pass
                    continue

                # Any non-continuation line ends the NodeString context
                current_ns_bc_id = None

                if record == "ND":
                    # ND nodeID x y z
                    nid = int(tokens[1])
                    x, y, z = float(tokens[2]), float(tokens[3]), float(tokens[4])
                    nodes[nid] = np.array([x, y, z], dtype=np.float64)

                elif record == "NODE":
                    # Node nodeID x y z  (alternate SRH-2D format)
                    nid = int(tokens[1])
                    x, y, z = float(tokens[2]), float(tokens[3]), float(tokens[4])
                    nodes[nid] = np.array([x, y, z], dtype=np.float64)

                elif record == "E3T":
                    # E3T elemID n1 n2 n3 matID
                    eid = int(tokens[1])
                    ns = np.array([int(tokens[2]) - 1, int(tokens[3]) - 1,
                                   int(tokens[4]) - 1], dtype=np.int64)
                    mat = int(tokens[5]) - 1  # 0-indexed
                    elems[eid] = (ns, mat)

                elif record == "E4Q":
                    # E4Q elemID n1 n2 n3 n4 matID
                    eid = int(tokens[1])
                    ns = np.array([int(tokens[2]) - 1, int(tokens[3]) - 1,
                                   int(tokens[4]) - 1, int(tokens[5]) - 1],
                                  dtype=np.int64)
                    mat = int(tokens[6]) - 1
                    elems[eid] = (ns, mat)

                elif record == "ELEM":
                    # Elem elemID n1 n2 n3 [n4]  (alternate format, no matID)
                    eid = int(tokens[1])
                    node_ids = [int(t) for t in tokens[2:]]
                    ns = np.array([n - 1 for n in node_ids], dtype=np.int64)
                    elems[eid] = (ns, 0)

                elif record == "NS":
                    # NS n1 n2 ... -nLast  bcID
                    # Negative sign on last node ID marks end; bcID is last token
                    self._parse_node_string(tokens[1:], node_strings)

                elif record == "NODESTRING":
                    # NodeString bcID n1 n2 ... (may continue on next lines)
                    current_ns_bc_id = self._parse_nodestring_alt(
                        tokens[1:], node_strings
                    )

        # Build sorted arrays
        n_nodes = max(nodes.keys())
        coords = np.zeros((n_nodes, 3), dtype=np.float64)
        for nid, xyz in nodes.items():
            coords[nid - 1] = xyz
        raw.node_coords = coords

        n_elems = max(elems.keys())
        elem_list = [None] * n_elems
        mat_list = np.zeros(n_elems, dtype=np.int64)
        for eid, (ns, mat) in elems.items():
            elem_list[eid - 1] = ns
            mat_list[eid - 1] = mat
        raw.elem_nodes = elem_list
        raw.elem_material = mat_list

        raw.node_strings = node_strings

        # Assign node IDs to boundary conditions
        for bc in raw.boundary_conditions:
            if bc.bc_id in node_strings:
                bc.node_ids = node_strings[bc.bc_id]

    def _parse_node_string(
        self, tokens: List[str], node_strings: Dict[int, List[int]]
    ) -> None:
        """Parse NS record tokens into node_strings dict."""
        # Last token is BC ID; nodes may be negative (end marker)
        try:
            bc_id = int(tokens[-1])
        except ValueError:
            return

        node_ids = []
        for tok in tokens[:-1]:
            try:
                nid = int(tok)
                node_ids.append(abs(nid) - 1)   # 0-indexed
            except ValueError:
                continue

        if bc_id not in node_strings:
            node_strings[bc_id] = []
        node_strings[bc_id].extend(node_ids)

    def _parse_nodestring_alt(
        self, tokens: List[str], node_strings: Dict[int, List[int]]
    ) -> Optional[int]:
        """
        Parse NodeString record: NodeString bcID n1 n2 n3 ...

        Returns the bc_id so the caller can track continuation lines.
        """
        if len(tokens) < 2:
            return None
        bc_id = int(tokens[0])
        node_ids = [int(tok) - 1 for tok in tokens[1:]]  # 0-indexed
        if bc_id not in node_strings:
            node_strings[bc_id] = []
        node_strings[bc_id].extend(node_ids)
        return bc_id

    # ------------------------------------------------------------------
    # .srhmat parser
    # ------------------------------------------------------------------

    def _read_mat(self, mat_file: Path, raw: SRH2DRawData) -> None:
        """
        Parse .srhmat file for material zone assignments.

        Supports two formats:
        - v30: ``Material <matID> <cell1> <cell2> ...`` (cells on multiple lines)
        - Legacy: ``VAR`` block with ``<matID> <manning_n>`` lines

        Manning's n values are read from the .srhhydro ManningsN lines.
        The .srhmat only assigns cells to material zones.

        The ManningsN mapping in .srhhydro uses:
        - ID 0 = default Manning's n (applies to cells not in any zone)
        - ID 1..N = material zone Manning's n
        """
        with open(mat_file) as f:
            content = f.read()
        lines = content.splitlines()

        in_var_block = False
        current_mat_id = None
        current_cells: List[int] = []

        for line in lines:
            line = line.strip()
            if not line or line.startswith("//"):
                continue

            tokens = line.split()
            key = tokens[0].upper()

            # Legacy VAR block format
            if key == "VAR":
                in_var_block = True
                continue

            if in_var_block and len(tokens) >= 2:
                try:
                    mat_id = int(tokens[0]) - 1   # 0-indexed
                    n_val = float(tokens[1])
                    raw.manning_n[mat_id] = n_val
                except ValueError:
                    in_var_block = False

            # v30 Material block: "Material <matID> <cell1> <cell2> ..."
            # Cell IDs may continue on following lines (just numbers)
            if key == "MATERIAL":
                # Flush previous material
                if current_mat_id is not None:
                    self._assign_material_cells(
                        raw, current_mat_id, current_cells
                    )
                current_mat_id = int(tokens[1])
                current_cells = []
                # Remaining tokens on this line are cell IDs
                for tok in tokens[2:]:
                    try:
                        current_cells.append(int(tok))
                    except ValueError:
                        pass
            elif current_mat_id is not None and key not in (
                "SRHMAT", "NMATERIALS", "MATNAME"
            ):
                # Continuation lines: just cell IDs
                for tok in tokens:
                    try:
                        current_cells.append(int(tok))
                    except ValueError:
                        # Hit a non-integer line → end of this material block
                        self._assign_material_cells(
                            raw, current_mat_id, current_cells
                        )
                        current_mat_id = None
                        current_cells = []
                        break

        # Flush last material
        if current_mat_id is not None:
            self._assign_material_cells(raw, current_mat_id, current_cells)

        if not raw.manning_n:
            raw.manning_n = {0: 0.03}
            logger.warning("No Manning's n parsed; defaulting to 0.03")
        logger.info(f"Manning's n: {raw.manning_n}")

    def _assign_material_cells(
        self, raw: SRH2DRawData, mat_id: int, cell_ids: List[int]
    ) -> None:
        """Assign material ID to cells listed in a Material block."""
        if raw.elem_material is None or len(cell_ids) == 0:
            return
        for cid in cell_ids:
            idx = cid - 1  # 0-indexed
            if 0 <= idx < len(raw.elem_material):
                raw.elem_material[idx] = mat_id
