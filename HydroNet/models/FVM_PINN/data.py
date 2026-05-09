"""
Dataset wrapper for FVM-PINN.

``FVM_PINNDataset`` reads an SRH-2D case (``.srhhydro``/``.srhgeom``/
``.srhmat``, plus optional ``_XMDFC.h5`` reference) and exposes the mesh
data, well-balanced still-water reference, and IC/BC/ref dicts in the
exact shape the internal ``FVMPINNLoss`` and ``BaseTrainer.setup`` expect.

Unlike ``PINNDataset`` this dataset does not consume preprocessed
``.npy`` files — the FVM residual needs the full face-based mesh
topology, which is built on-the-fly from the raw SRH-2D files. The
dataset is therefore "one case per instance" and the ``__len__`` /
``__getitem__`` methods exist only for DataLoader compatibility;
training is full-batch, driven by ``get_*_data()`` accessors.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from ...utils.config import Config
from ._internal.mesh.srh2d_reader import SRH2DMeshReader
from ._internal.mesh.mesh_topology import build_mesh
from ._internal.fvm.geometry import compute_cell_geometry


class FVM_PINNDataset(Dataset):
    """
    Self-contained FVM-PINN dataset for a single SRH-2D case.

    Public accessors
    ----------------
    get_mesh_data()  -> dict of torch tensors (face topology, cell geometry, bed, Manning, S0, ...)
    get_h_still()    -> [n_cells] tensor, max(0, wse_still - bed)
    get_cell_xy()    -> [n_cells, 2] tensor
    get_ic_data()    -> {xyt: [n, 3], U_true: [n, 3]}  with U_true in perturbation form [xi, hu, hv]
    get_bc_data()    -> {bc_id: {type, xyt, value, nx, ny, ...}}
    get_ref_data()   -> Optional SRH-2D snapshots (all times) as {xyt, U_ref}, U_ref in xi-form
    """

    def __init__(self, config):
        if not isinstance(config, Config):
            raise ValueError("config must be a Config object")
        self.config = config

        # ---- Device / dtype ----
        device_type = config.get('device.type', 'cpu')
        device_index = int(config.get('device.index', 0))
        if device_type == 'cuda' and torch.cuda.is_available():
            self.device = torch.device(f'cuda:{device_index}')
        else:
            self.device = torch.device('cpu')
        self.dtype = torch.float64

        # ---- Case paths ----
        self.srhhydro = str(config.get_required_config('data.srhhydro'))
        self.srh2d_h5_file = config.get('data.srh2d_h5_file', None)
        if self.srh2d_h5_file is not None:
            self.srh2d_h5_file = str(self.srh2d_h5_file)

        # ---- Physics knobs (perturbation form + dry threshold) ----
        self.wse_still = float(config.get_required_config('physics.wse_still'))
        self.h_dry = float(config.get('physics.h_dry', 1e-2))

        # ---- Time window for unsteady training ----
        self.t_start = float(config.get('training.t_start', 0.0))
        self.t_end = float(config.get('training.t_end', 1.0))

        # ---- Build mesh + mesh_data ----
        reader = SRH2DMeshReader(self.srhhydro)
        raw = reader.read()
        self._mesh = build_mesh(raw)
        self._mesh_data = compute_cell_geometry(
            self._mesh, device=str(self.device)
        )

        # ---- Well-balanced h_still = max(0, wse_still - bed) ----
        bed = self._mesh_data["bed_elev"]
        self._h_still = (self.wse_still - bed).clamp(min=0.0).to(
            dtype=self.dtype, device=self.device
        )

        # ---- Attach h_still to mesh_data for BaseTrainer ----
        self._mesh_data["h_still"] = self._h_still

        # ---- Attach bc_ghost (used by the FVM residual) ----
        self._mesh_data["bc_ghost"] = self._build_bc_ghost()

        # ---- Build IC / BC / ref data ----
        self._ic_data = self._build_ic_data()
        self._bc_data = self._build_bc_data()
        self._ref_data = self._build_ref_data()

        # ---- Convenience cell_xy (CPU tensor — FVM_SWE_PINN moves it) ----
        self._cell_xy = self._mesh_data["cell_center"].detach().clone()

    # ------------------------------------------------------------------
    # Alternate constructor: build from a pre-made mesh (no SRH-2D I/O)
    # ------------------------------------------------------------------

    @classmethod
    def from_mesh(
        cls,
        config,
        mesh,
        ic_data: Dict[str, torch.Tensor],
        bc_data: Optional[Dict] = None,
        ref_data: Optional[Dict[str, torch.Tensor]] = None,
    ) -> "FVM_PINNDataset":
        """
        Construct a dataset from a programmatically-built ``UnstructuredMesh``
        and pre-computed training data, bypassing the SRH-2D file reader.

        Use this for synthetic cases (e.g., the 1D dam-break) where the mesh
        is built in Python and the IC comes from an analytical expression.

        Parameters
        ----------
        config   : Config
        mesh     : UnstructuredMesh (from the internal mesh module)
        ic_data  : {"xyt": [n, 3], "U_true": [n, 3]} with U_true in the
                   **perturbation form** ``[xi, hu, hv]`` where
                   ``xi = h - h_still``.
        bc_data  : optional soft-BC dict (see ``_build_bc_data``); pass None
                   if the case relies only on the FVM residual's default
                   reflective walls.
        ref_data : optional reference/anchor data in perturbation form.

        Notes
        -----
        - ``physics.wse_still`` / ``physics.h_dry`` / ``training.t_start`` /
          ``training.t_end`` are still read from ``config``.
        - ``initial_condition.*`` and ``data.*`` entries in the YAML are
          ignored on this path (the caller owns the IC and the mesh).
        - Any ``boundary_conditions`` block in the YAML is still consumed to
          populate ``mesh_data["bc_ghost"]`` for the FVM residual (i.e.
          inlet-q / exit-h types). Walls default to reflective and need no
          config entry.
        """
        if not isinstance(config, Config):
            raise ValueError("config must be a Config object")

        self = cls.__new__(cls)
        self.config = config

        # ---- Device / dtype ----
        device_type = config.get('device.type', 'cpu')
        device_index = int(config.get('device.index', 0))
        if device_type == 'cuda' and torch.cuda.is_available():
            self.device = torch.device(f'cuda:{device_index}')
        else:
            self.device = torch.device('cpu')
        self.dtype = torch.float64

        # Unused on this path but keep the attribute for API symmetry.
        self.srhhydro = None
        self.srh2d_h5_file = None

        # ---- Physics + time window ----
        self.wse_still = float(config.get_required_config('physics.wse_still'))
        self.h_dry = float(config.get('physics.h_dry', 1e-2))
        self.t_start = float(config.get('training.t_start', 0.0))
        self.t_end = float(config.get('training.t_end', 1.0))

        # ---- Build mesh_data + h_still from the given UnstructuredMesh ----
        self._mesh = mesh
        self._mesh_data = compute_cell_geometry(mesh, device=str(self.device))
        bed = self._mesh_data["bed_elev"]
        self._h_still = (self.wse_still - bed).clamp(min=0.0).to(
            dtype=self.dtype, device=self.device
        )
        self._mesh_data["h_still"] = self._h_still
        self._mesh_data["bc_ghost"] = self._build_bc_ghost()

        # ---- User-provided training data (moved to device) ----
        self._ic_data = {
            k: v.to(device=self.device, dtype=self.dtype)
            if torch.is_tensor(v) else v
            for k, v in ic_data.items()
        }
        self._bc_data = bc_data
        self._ref_data = ref_data

        self._cell_xy = self._mesh_data["cell_center"].detach().clone()
        return self

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    def get_mesh_data(self) -> Dict[str, Any]:
        return self._mesh_data

    def get_h_still(self) -> torch.Tensor:
        return self._h_still

    def get_cell_xy(self) -> torch.Tensor:
        return self._cell_xy

    def get_mesh(self):
        """Underlying UnstructuredMesh (for VTK writer)."""
        return self._mesh

    def get_ic_data(self) -> Dict[str, torch.Tensor]:
        return self._ic_data

    def get_bc_data(self) -> Optional[Dict]:
        return self._bc_data

    def get_ref_data(self) -> Optional[Dict[str, torch.Tensor]]:
        return self._ref_data

    # ------------------------------------------------------------------
    # PyTorch Dataset protocol (single-case, full-batch; only here for
    # DataLoader compatibility — see module docstring)
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx != 0:
            raise IndexError("FVM_PINNDataset has a single implicit item")
        return {
            "mesh_data": self._mesh_data,
            "h_still": self._h_still,
            "ic_data": self._ic_data,
            "bc_data": self._bc_data,
            "ref_data": self._ref_data,
        }

    # ------------------------------------------------------------------
    # BC ghost setup
    # ------------------------------------------------------------------

    def _build_bc_ghost(self) -> Dict[int, Dict[str, Any]]:
        """
        Translate ``boundary_conditions`` YAML block to the ``bc_ghost``
        dict that ``compute_fvm_residual`` consumes. The inlet ``value``
        is total discharge Q [m³/s]; the ghost builder distributes it
        across inlet faces by Manning conveyance.
        """
        bc_ghost: Dict[int, Dict[str, Any]] = {}
        bc_block = self.config.get('boundary_conditions', None)
        if bc_block is None:
            return bc_ghost

        # YAML load returns a dict with int keys if written with bare
        # integers, string keys otherwise. Normalise.
        for raw_id, bc_info in bc_block.items():
            bc_id = int(raw_id)
            bc_type = bc_info.get("type")
            if bc_type == "inlet-q":
                bc_ghost[bc_id] = {
                    "type": "inlet-q",
                    "value": float(bc_info["value"]),
                }
            elif bc_type == "exit-h":
                # Caller specifies WSE (not depth); ghost builder will
                # subtract per-face bed to get h_ghost.
                bc_ghost[bc_id] = {
                    "type": "exit-h",
                    "value": float(bc_info["value"]),
                }
            elif bc_type in ("wall", "symmetry"):
                # Default reflective ghost is handled by the FVM residual
                # for unmarked boundaries; no entry needed here.
                pass
            else:
                raise ValueError(
                    f"Unknown boundary condition type: {bc_type!r} for bc_id {bc_id}"
                )
        return bc_ghost

    # ------------------------------------------------------------------
    # IC setup
    # ------------------------------------------------------------------

    def _build_ic_data(self) -> Dict[str, torch.Tensor]:
        """
        Build the IC dict consumed by ``BaseTrainer.setup``.

        ``U_true`` is in perturbation form ``[xi, hu, hv]`` — the network
        predicts ``xi`` directly, so the IC loss compares like to like.
        """
        ic_mode = str(self.config.get('initial_condition.mode', 'flat-wse'))
        cell_xy_np = self._mesh.cell_center  # numpy
        n_cells = cell_xy_np.shape[0]

        if ic_mode == 'flat-wse':
            wse_init = float(
                self.config.get('initial_condition.wse_init', self.wse_still)
            )
            bed_np = self._mesh.bed_elev
            h_ic = np.clip(wse_init - bed_np, a_min=0.0, a_max=None)
            hu_ic = np.zeros(n_cells, dtype=np.float64)
            hv_ic = np.zeros(n_cells, dtype=np.float64)
        elif ic_mode == 'uniform-h':
            h_value = float(self.config.get_required_config('initial_condition.h'))
            u_value = float(self.config.get('initial_condition.u', 0.0))
            v_value = float(self.config.get('initial_condition.v', 0.0))
            h_ic = np.full(n_cells, h_value, dtype=np.float64)
            hu_ic = h_ic * u_value
            hv_ic = h_ic * v_value
        elif ic_mode == 'srh2d-snapshot':
            # Load IC from an SRH-2D h5 snapshot at t=t_start
            h_ic, hu_ic, hv_ic = self._load_srh2d_ic_at(self.t_start)
        else:
            raise ValueError(
                f"Unknown initial_condition.mode: {ic_mode!r} "
                "(expected 'flat-wse', 'uniform-h', or 'srh2d-snapshot')"
            )

        # Convert physical h → xi = h - h_still (network output convention)
        h_still_np = self._h_still.cpu().numpy()
        xi_ic = h_ic - h_still_np

        xyt = np.column_stack([
            cell_xy_np[:, 0],
            cell_xy_np[:, 1],
            np.full(n_cells, self.t_start, dtype=np.float64),
        ])
        U_true = np.column_stack([xi_ic, hu_ic, hv_ic])

        return {
            "xyt":    torch.tensor(xyt,    dtype=self.dtype, device=self.device),
            "U_true": torch.tensor(U_true, dtype=self.dtype, device=self.device),
        }

    # ------------------------------------------------------------------
    # BC setup (for the PINN soft-BC loss; the hard BC is in the FVM residual)
    # ------------------------------------------------------------------

    def _build_bc_data(self) -> Optional[Dict[int, Dict[str, Any]]]:
        """
        Build per-face soft-BC data for ``FVMPINNLoss._bc_loss``.

        Inlet-q soft loss: ``(hu·n + q_per_width)²``.
        Exit-h soft loss: ``(xi + h_still - h_target)²``.
        Wall/symmetry soft loss: ``(hu·n)²``.
        """
        bc_block = self.config.get('boundary_conditions', None)
        if bc_block is None:
            return None

        bc_data: Dict[int, Dict[str, Any]] = {}
        face_bc_id = self._mesh_data["face_bc_id"]
        face_center = self._mesh_data.get("face_center", None)
        face_normal = self._mesh_data["face_normal"]
        face_length = self._mesh_data["face_length"]
        face_left = self._mesh_data["face_left"]
        bed_elev = self._mesh_data["bed_elev"]
        t_mid = 0.5 * (self.t_start + self.t_end)

        for raw_id, bc_info in bc_block.items():
            bc_id = int(raw_id)
            bc_type = bc_info.get("type")
            mask = face_bc_id == bc_id
            if not bool(mask.any()):
                continue
            fc = face_center[mask]
            n_bc = fc.shape[0]
            xyt_bc = torch.cat([
                fc,
                torch.full((n_bc, 1), t_mid, dtype=self.dtype, device=fc.device),
            ], dim=-1)

            if bc_type == "inlet-q":
                total_Q = float(bc_info["value"])
                inlet_len = face_length[mask].sum()
                q_per_width = total_Q / inlet_len.clamp(min=1e-12)
                bc_data[bc_id] = {
                    "type": "inlet-q",
                    "xyt":   xyt_bc,
                    "value": q_per_width * torch.ones(n_bc, dtype=self.dtype, device=fc.device),
                    "nx":    face_normal[mask, 0],
                    "ny":    face_normal[mask, 1],
                }
            elif bc_type == "exit-h":
                wse_exit = float(bc_info["value"])
                exit_cells = face_left[mask]
                exit_bed = bed_elev[exit_cells]
                exit_h = (wse_exit - exit_bed).clamp(min=0.0)
                exit_hstill = self._h_still[exit_cells]
                bc_data[bc_id] = {
                    "type":    "exit-h",
                    "xyt":     xyt_bc,
                    "value":   exit_h.to(dtype=self.dtype),
                    "h_still": exit_hstill.to(dtype=self.dtype),
                }
            elif bc_type in ("wall", "symmetry"):
                bc_data[bc_id] = {
                    "type": bc_type,
                    "xyt":  xyt_bc,
                    "nx":   face_normal[mask, 0],
                    "ny":   face_normal[mask, 1],
                }
            else:
                raise ValueError(
                    f"Unknown boundary condition type: {bc_type!r} for bc_id {bc_id}"
                )
        return bc_data if bc_data else None

    # ------------------------------------------------------------------
    # Reference data (optional, from SRH-2D h5)
    # ------------------------------------------------------------------

    def _build_ref_data(self) -> Optional[Dict[str, torch.Tensor]]:
        """Load SRH-2D reference snapshots as anchor/measurement data.

        Four modes, controlled by the ``data.measurements`` YAML block:

        ``measurements`` absent, or ``mode: "dense"`` (default)
            Load every wet cell at every in-window snapshot (backward-
            compatible behaviour). Used by teacher-mode anchors and by the
            BIC-D / BIC-H / SR-B/C/D/G runs.

        ``mode: "sparse"``
            Subsample ``n_points`` random cells at each snapshot listed in
            ``times`` (or at all in-window snapshots if ``times`` is
            empty/absent). Used by the sparse-measurement ablation runs
            (BIC-B/C/F/G, SR-F). Optional ``variables`` restricts which
            conserved components are supervised via ``var_mask``; optional
            ``noise_sigma`` adds Gaussian noise with standard deviation
            equal to ``noise_sigma * max|U_ref|`` per component.

        ``mode: "both"``
            Concatenate a sparse block (at ``sparse_times``, typically just
            ``t_end``) with the full dense snapshot block. Produces a single
            ``ref_data`` dict with a **per-point** ``var_mask`` of shape
            ``[N, 3]`` so sparse rows can be velocity-only while dense rows
            supervise all three components. Used by BIC-E.

        Returns None when neither an h5 file nor the measurements block
        provides usable reference data.
        """
        if self.srh2d_h5_file is None:
            return None
        h5_path = Path(self.srh2d_h5_file)
        if not h5_path.exists():
            return None

        import h5py
        with h5py.File(h5_path, "r") as f:
            times_all = f["Water_Depth_m/Times"][:].astype(np.float64)
            h_all = f["Water_Depth_m/Values"][:, :].astype(np.float64)
            vel_all = f["Velocity_m_p_s/Values"][:, :, :].astype(np.float64)

        xc = self._mesh.cell_center[:, 0]
        yc = self._mesh.cell_center[:, 1]
        h_still_np = self._h_still.cpu().numpy()

        meas_cfg = self.config.get("data.measurements", None)
        mode = "dense"
        if meas_cfg is not None:
            mode = str(meas_cfg.get("mode", "dense")).lower()
            if mode not in ("dense", "sparse", "both"):
                raise ValueError(
                    f"data.measurements.mode must be 'dense', 'sparse', or 'both', "
                    f"got {mode!r}"
                )

        if mode == "dense":
            return self._ref_data_dense(
                times_all, h_all, vel_all, xc, yc, h_still_np
            )

        # Sparse-specific options
        default_seed = int(meas_cfg.get("seed", 42))
        default_noise = float(meas_cfg.get("noise_sigma", 0.0))
        default_vars = list(meas_cfg.get("variables", ["xi", "hu", "hv"]))

        if mode == "sparse":
            return self._ref_data_sparse(
                times_all, h_all, vel_all, xc, yc, h_still_np,
                n_points=int(meas_cfg.get("n_points", 200)),
                req_times=list(meas_cfg.get("times", [])),
                variables=default_vars,
                noise_sigma=default_noise,
                seed=default_seed,
            )

        # mode == "both"
        sparse_block = self._ref_data_sparse(
            times_all, h_all, vel_all, xc, yc, h_still_np,
            n_points=int(meas_cfg.get("n_points", 200)),
            req_times=list(
                meas_cfg.get("sparse_times", meas_cfg.get("times", [self.t_end]))
            ),
            variables=default_vars,
            noise_sigma=default_noise,
            seed=default_seed,
        )
        dense_block = self._ref_data_dense(
            times_all, h_all, vel_all, xc, yc, h_still_np,
            restrict_times=list(meas_cfg.get("dense_times", [])),
        )
        return self._concat_ref_blocks(sparse_block, dense_block)

    # ------------------------------------------------------------------
    # ref_data helpers (dense + sparse + concat)
    # ------------------------------------------------------------------

    def _ref_data_dense(
        self,
        times_all: np.ndarray,
        h_all: np.ndarray,
        vel_all: np.ndarray,
        xc: np.ndarray,
        yc: np.ndarray,
        h_still_np: np.ndarray,
        restrict_times: Optional[list] = None,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Every wet cell at every in-window (and optionally restricted) time."""
        all_xyt: list = []
        all_U: list = []
        for ti, t_val in enumerate(times_all):
            if t_val < self.t_start or t_val > self.t_end:
                continue
            if restrict_times:
                # nearest-match tolerance of 1% of the training window
                tol = 0.01 * max(self.t_end - self.t_start, 1.0)
                if not any(abs(t_val - tr) <= tol for tr in restrict_times):
                    continue
            h_ref = h_all[ti]
            vel_ref = vel_all[ti]
            wet = h_ref > self.h_dry
            idx = np.where(wet)[0]
            if len(idx) == 0:
                continue
            xyt_t = np.column_stack([
                xc[idx], yc[idx], np.full(len(idx), t_val, dtype=np.float64)
            ])
            xi_t = h_ref[idx] - h_still_np[idx]
            hu_t = h_ref[idx] * vel_ref[idx, 0]
            hv_t = h_ref[idx] * vel_ref[idx, 1]
            all_xyt.append(xyt_t)
            all_U.append(np.column_stack([xi_t, hu_t, hv_t]))

        if not all_xyt:
            return None

        xyt = np.concatenate(all_xyt)
        U = np.concatenate(all_U)
        return {
            "xyt":   torch.tensor(xyt, dtype=self.dtype, device=self.device),
            "U_ref": torch.tensor(U,   dtype=self.dtype, device=self.device),
            # Per-point all-ones mask so "both" mode can stack cleanly.
            "var_mask": torch.ones((xyt.shape[0], 3),
                                   dtype=self.dtype, device=self.device),
        }

    def _ref_data_sparse(
        self,
        times_all: np.ndarray,
        h_all: np.ndarray,
        vel_all: np.ndarray,
        xc: np.ndarray,
        yc: np.ndarray,
        h_still_np: np.ndarray,
        *,
        n_points: int,
        req_times: list,
        variables: list,
        noise_sigma: float,
        seed: int,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Subsample ``n_points`` wet cells at each selected snapshot time."""
        # Resolve requested times: empty list => all in-window snapshots.
        tol = 0.01 * max(self.t_end - self.t_start, 1.0)
        if not req_times:
            sel_ti = [
                ti for ti, tv in enumerate(times_all)
                if self.t_start <= tv <= self.t_end
            ]
        else:
            sel_ti = []
            for tr in req_times:
                ti = int(np.argmin(np.abs(times_all - float(tr))))
                if abs(times_all[ti] - float(tr)) > tol:
                    raise ValueError(
                        f"Requested measurement time {tr}s has no SRH-2D snapshot "
                        f"within {tol}s (nearest: {times_all[ti]}s)."
                    )
                sel_ti.append(ti)

        # Variable mask: [xi, hu, hv] in that order
        var_ix = {"xi": 0, "hu": 1, "hv": 2}
        mask_vec = np.zeros(3, dtype=np.float64)
        for v in variables:
            if v not in var_ix:
                raise ValueError(
                    f"data.measurements.variables entry {v!r} must be one of "
                    f"'xi', 'hu', 'hv'"
                )
            mask_vec[var_ix[v]] = 1.0
        if mask_vec.sum() == 0:
            raise ValueError("data.measurements.variables cannot be empty.")

        rng = np.random.default_rng(seed)
        all_xyt: list = []
        all_U: list = []
        for ti in sel_ti:
            t_val = float(times_all[ti])
            h_ref = h_all[ti]
            vel_ref = vel_all[ti]
            wet = np.where(h_ref > self.h_dry)[0]
            if len(wet) == 0:
                continue
            take = min(n_points, len(wet))
            pick = rng.choice(wet, size=take, replace=False)

            xi_t = h_ref[pick] - h_still_np[pick]
            hu_t = h_ref[pick] * vel_ref[pick, 0]
            hv_t = h_ref[pick] * vel_ref[pick, 1]
            U_t = np.column_stack([xi_t, hu_t, hv_t])

            if noise_sigma > 0.0:
                # Per-component noise scaled by the max |U| in the snapshot.
                scales = np.array([
                    np.abs(h_ref - h_still_np).max(),
                    np.abs(h_ref * vel_ref[:, 0]).max(),
                    np.abs(h_ref * vel_ref[:, 1]).max(),
                ], dtype=np.float64)
                scales = np.where(scales > 0, scales, 1.0)
                U_t = U_t + noise_sigma * scales * rng.standard_normal(U_t.shape)

            xyt_t = np.column_stack([
                xc[pick], yc[pick], np.full(take, t_val, dtype=np.float64)
            ])
            all_xyt.append(xyt_t)
            all_U.append(U_t)

        if not all_xyt:
            return None

        xyt = np.concatenate(all_xyt)
        U = np.concatenate(all_U)
        # Broadcast per-point var_mask of shape [N, 3] so it can be stacked
        # with dense blocks that use different variable subsets.
        vm = np.broadcast_to(mask_vec[None, :], (xyt.shape[0], 3)).copy()
        return {
            "xyt":      torch.tensor(xyt, dtype=self.dtype, device=self.device),
            "U_ref":    torch.tensor(U,   dtype=self.dtype, device=self.device),
            "var_mask": torch.tensor(vm,  dtype=self.dtype, device=self.device),
        }

    @staticmethod
    def _concat_ref_blocks(
        a: Optional[Dict[str, torch.Tensor]],
        b: Optional[Dict[str, torch.Tensor]],
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Concatenate two ref_data blocks along the point dimension."""
        if a is None:
            return b
        if b is None:
            return a
        return {
            "xyt":      torch.cat([a["xyt"],      b["xyt"]],      dim=0),
            "U_ref":    torch.cat([a["U_ref"],    b["U_ref"]],    dim=0),
            "var_mask": torch.cat([a["var_mask"], b["var_mask"]], dim=0),
        }

    def _load_srh2d_ic_at(self, t_target: float):
        """Load h/hu/hv from an SRH-2D h5 snapshot nearest to ``t_target``."""
        if self.srh2d_h5_file is None:
            raise ValueError(
                "initial_condition.mode='srh2d-snapshot' requires data.srh2d_h5_file"
            )
        h5_path = Path(self.srh2d_h5_file)
        if not h5_path.exists():
            raise FileNotFoundError(f"SRH-2D H5 file not found: {h5_path}")

        import h5py
        with h5py.File(h5_path, "r") as f:
            times = f["Water_Depth_m/Times"][:].astype(np.float64)
            ti = int(np.argmin(np.abs(times - t_target)))
            h_ic = f["Water_Depth_m/Values"][ti, :].astype(np.float64).copy()
            vel_ic = f["Velocity_m_p_s/Values"][ti, :, :].astype(np.float64)

        dry = h_ic < self.h_dry
        h_ic[dry] = 0.0
        hu_ic = h_ic * vel_ic[:, 0]
        hv_ic = h_ic * vel_ic[:, 1]
        hu_ic[dry] = 0.0
        hv_ic[dry] = 0.0
        return h_ic, hu_ic, hv_ic
