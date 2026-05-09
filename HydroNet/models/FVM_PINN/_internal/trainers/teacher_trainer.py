"""
Strategy 4: FVM Teacher-Trajectory Trainer.

Generates a reference trajectory with the FVM solver once (cached to disk),
then trains the network to match every snapshot via a distillation loss.
This sidesteps the multiple-steady-state branches of a pure physics-residual
loss — the IC + FVM trajectory uniquely picks out one solution, and the
network regresses to it snapshot-by-snapshot.

Loss per epoch
--------------
    L = Σ_k [ L_distill(Q_pred(t_k) | Q_teacher[k])
            + λ_bc   · L_bc(Q_pred, t_k)
            + λ_phys · L_phys(Q_pred, t_k) ] / (M + 1)
        + λ_anchor · L_anchor(Q_pred | ref_data)

where L_phys = ||∂Q/∂t + R(Q)||² is the residual form of the SWE (matches
the standard FVM-PINN loss), used as a smooth regulariser after a warm-up
phase. L_anchor is a direct MSE against optional reference data (e.g.,
SRH-2D snapshots or sparse field measurements).

Design
------
- Stands alongside ``TimeWindowTrainer``: owns its own ``network`` and
  training loop rather than subclassing ``BaseTrainer``. Exposes the same
  public API (``setup``/``train``/``predict``) so it slots into
  ``TrainerFactory("teacher")`` the same way.
- The teacher trajectory is cached to ``trajectory_cache`` on first run;
  subsequent runs with matching mesh / snapshots / cfl / IC reuse the cache.
  Set ``regen_trajectory=True`` to force regeneration.
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from ..fvm.riemann_solver import compute_fvm_residual
from ..fvm.time_stepping import run_fvm_rk2
from ..pinn.network import SWENet, SirenSWENet, NetworkConfig
from ..pinn.loss import LossConfig
from .base_trainer import _to_device
from .memory_tracker import MemoryTracker

logger = logging.getLogger(__name__)


@dataclass
class TeacherTrainerConfig:
    """
    Hyperparameters for the FVM-teacher trainer.

    Trajectory
    ----------
    n_snapshots : number of FVM snapshots generated between ``t_start`` and
                  ``t_end`` (M+1 total, inclusive of t_start).
    cfl         : CFL for the trajectory generator.
    h_dry       : depth threshold below which momentum is zeroed during the
                  trajectory RK2 stage (prevents hu/h blow-up on dry fronts).
    trajectory_cache_path : optional cache file path.
    regen_trajectory      : if True, ignore any cached file and re-run FVM.

    Training
    --------
    adam_epochs    : Adam phase epochs (each epoch loops over M+1 snapshots).
    adam_lr        : Adam learning rate.
    lbfgs_steps    : L-BFGS outer steps for the polish phase (0 disables).
    lbfgs_max_iter : max_iter per lbfgs.step() call.
    phys_warmup    : epochs before λ_phys ramps in (0 disables warm-up).

    Loss weights
    ------------
    lambda_bc, lambda_phys, lambda_anchor : scalars. Distillation weight is
                                            implicit (= 1).

    I/O / hardware
    --------------
    log_every       : print every N epochs (Adam) or closures (L-BFGS).
    checkpoint_every: save every N epochs.
    output_dir, device, dtype : as in BaseTrainerConfig.

    Arch / IC sampling
    ------------------
    t_start, t_end : trajectory interval; Q_teacher[0] = Q_init at t_start.
    """

    # Trajectory
    n_snapshots: int = 30
    cfl: float = 0.1
    h_dry: float = 1e-2
    trajectory_cache_path: str = ""
    regen_trajectory: bool = False

    # Training
    adam_epochs: int = 3000
    adam_lr: float = 1e-3
    adam_decay_every: int = 1000
    adam_lr_decay: float = 0.95
    lbfgs_steps: int = 200
    lbfgs_max_iter: int = 20
    lbfgs_lr: float = 1.0
    phys_warmup: int = 200

    # Loss weights (distill weight is implicit 1.0)
    lambda_bc: float = 1.0
    lambda_phys: float = 0.05
    lambda_anchor: float = 1.0

    # Per-SWE-component weights applied inside the distill and physics-
    # residual losses to balance the scale mismatch between xi and hu/hv.
    # Defaults are 1.0 (uniform); raise lambda_xi if xi << hu,hv in your case.
    lambda_xi: float = 1.0
    lambda_hu: float = 1.0
    lambda_hv: float = 1.0

    # I/O
    log_every: int = 50
    checkpoint_every: int = 1000
    output_dir: str = "outputs"

    # Hardware
    device: str = "cpu"
    dtype: str = "float64"

    # Time window
    t_start: float = 0.0
    t_end: float = 1.0


class TeacherTrainer:
    """
    Standalone FVM-teacher trainer (``strategy='teacher'``).

    Constructor signature matches the other internal trainers so
    ``TrainerFactory("teacher")(cfg, net_cfg, loss_cfg, mesh_data)`` works.
    """

    def __init__(
        self,
        teacher_cfg: TeacherTrainerConfig,
        net_cfg: NetworkConfig,
        loss_cfg: LossConfig,
        mesh_data: Dict[str, torch.Tensor],
        bc_config: Optional[Dict] = None,
    ) -> None:
        self.cfg = teacher_cfg
        self.net_cfg = net_cfg
        self.loss_cfg = loss_cfg  # kept for parity/h_dry/grad_checkpoint
        self.bc_config = bc_config

        self.device = torch.device(teacher_cfg.device)
        self.dtype = torch.float64 if teacher_cfg.dtype == "float64" else torch.float32

        # Move mesh tensors to device/dtype (matches BaseTrainer)
        self.mesh_data = _to_device(mesh_data, self.device, self.dtype)

        # h_still for the FVM residual and trajectory generator
        h_still = mesh_data.get("h_still", None)
        if h_still is None:
            n_cells = int(self.mesh_data["n_cells"])
            self.h_still = torch.zeros(n_cells, dtype=self.dtype, device=self.device)
        else:
            self.h_still = h_still.to(dtype=self.dtype, device=self.device)

        # Build the network. If FVM_PINNTrainer is wrapping us, it will
        # immediately reassign ``self.network`` to the model's internal net;
        # otherwise we own this one.
        self.network = SWENet(net_cfg).to(self.device).to(self.dtype)

        # Data (set by setup())
        self.ic_data: Optional[Dict] = None
        self.bc_data: Optional[Dict] = None
        self.ref_data: Optional[Dict] = None

        # Teacher trajectory (built during setup())
        self._t_snaps: Optional[torch.Tensor] = None
        self._Q_snaps: Optional[torch.Tensor] = None

        # History — per-step losses. Core scalars + per-SWE-component
        # breakdowns for the elementwise losses (distill, phys) so scale
        # imbalances between xi and hu/hv are visible in the log / plot.
        core_keys = (
            "total",
            "distill", "distill_xi", "distill_hu", "distill_hv",
            "bc",
            "phys", "phys_xi", "phys_hu", "phys_hv",
            "anchor",
        )
        self.history: Dict[str, List[float]] = {k: [] for k in core_keys}

        self.output_dir = Path(teacher_cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Optional GPU memory tracker (attached externally by FVM_PINNTrainer).
        self.memory_tracker: Optional[MemoryTracker] = None

    # ------------------------------------------------------------------
    # Public API (parity with BaseTrainer / TimeWindowTrainer)
    # ------------------------------------------------------------------

    def setup(
        self,
        ic_data: Dict,
        bc_data: Optional[Dict] = None,
        ref_data: Optional[Dict] = None,
    ) -> None:
        """
        Move training data to device, set input normalisation, generate
        (or load) the teacher trajectory.
        """
        self.ic_data = _to_device(ic_data, self.device, self.dtype)
        if bc_data is not None:
            self.bc_data = {
                k: _to_device(v, self.device, self.dtype) if isinstance(v, dict) else v
                for k, v in bc_data.items()
            }
        if ref_data is not None:
            self.ref_data = _to_device(ref_data, self.device, self.dtype)

        # Build trajectory
        self._build_trajectory()

        # Normalisation over (cell × snapshot) grid
        self._set_normalisation()

        logger.info(
            f"TeacherTrainer ready: {self.mesh_data['n_cells']} cells, "
            f"{self._Q_snaps.shape[0]} snapshots, "
            f"t=[{self.cfg.t_start:.2f}, {self.cfg.t_end:.2f}]"
        )

    def train(self) -> float:
        """Run Adam phase + optional L-BFGS polish. Returns final loss."""
        logger.info("=== TeacherTrainer training start ===")
        t0 = time.perf_counter()

        self._adam_phase()
        self._lbfgs_phase()

        elapsed = time.perf_counter() - t0
        final_loss = self.history["total"][-1] if self.history["total"] else float("nan")
        logger.info(f"Training complete: {elapsed:.1f}s, final loss={final_loss:.4e}")
        self._save_checkpoint("final")
        return final_loss

    def predict(self, xyt: torch.Tensor) -> torch.Tensor:
        """No-grad forward at arbitrary points. Returns [xi, hu, hv]."""
        self.network.eval()
        with torch.no_grad():
            xyt = xyt.to(self.device, dtype=self.dtype)
            return self.network(xyt).detach()

    def load_checkpoint(self, path: str) -> None:
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.network.load_state_dict(ck["network_state"])
        self.network.set_normalisation(ck["x_mean"], ck["x_std"])
        if "history" in ck:
            self.history = ck["history"]
        logger.info(f"Checkpoint loaded: {path}")

    # ------------------------------------------------------------------
    # Adam / L-BFGS phases
    # ------------------------------------------------------------------

    def _adam_phase(self) -> None:
        if self.cfg.adam_epochs <= 0:
            return
        has_anchor = self.ref_data is not None and self.cfg.lambda_anchor > 0
        logger.info(
            f"Adam: {self.cfg.adam_epochs} × {self._Q_snaps.shape[0]} snapshots, "
            f"lr={self.cfg.adam_lr}, lambda_bc={self.cfg.lambda_bc}, "
            f"lambda_phys={self.cfg.lambda_phys}, "
            f"lambda_anchor={self.cfg.lambda_anchor if has_anchor else 0.0}, "
            f"phys_warmup={self.cfg.phys_warmup}"
        )

        if self.memory_tracker is not None:
            self.memory_tracker.reset_peak("teacher-adam-start")
            self.memory_tracker.sample("adam", 0)

        optimizer = optim.Adam(self.network.parameters(), lr=self.cfg.adam_lr)
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=self.cfg.adam_decay_every,
            gamma=self.cfg.adam_lr_decay,
        )
        norm = float(self._Q_snaps.shape[0])

        # Per-component accumulator keys for the snapshot-averaged logs
        comp_keys = ("distill", "distill_xi", "distill_hu", "distill_hv",
                     "phys", "phys_xi", "phys_hu", "phys_hv")

        for epoch in range(1, self.cfg.adam_epochs + 1):
            optimizer.zero_grad()
            lam_phys = self._phys_weight(epoch)
            sums = {k: 0.0 for k in comp_keys}
            sum_b = 0.0
            for k in range(self._Q_snaps.shape[0]):
                t_k = float(self._t_snaps[k])
                parts = self._snapshot_losses(t_k, self._Q_snaps[k], lam_phys)
                l_k = (parts["distill"]
                       + self.cfg.lambda_bc * parts["bc"]
                       + lam_phys * parts["phys"]) / norm
                l_k.backward()
                for key in comp_keys:
                    sums[key] += parts[key].item()
                sum_b += parts["bc"].item()

            sum_a = 0.0
            if has_anchor:
                l_a = self._measurement_loss(self.ref_data)
                (self.cfg.lambda_anchor * l_a).backward()
                sum_a = l_a.item()

            optimizer.step()
            scheduler.step()

            avgs = {k: v / norm for k, v in sums.items()}
            avg_b = sum_b / norm
            total = (avgs["distill"] + self.cfg.lambda_bc * avg_b
                     + lam_phys * avgs["phys"]
                     + (self.cfg.lambda_anchor * sum_a if has_anchor else 0.0))
            self.history["total"].append(total)
            self.history["bc"].append(avg_b)
            self.history["anchor"].append(sum_a)
            for key in comp_keys:
                self.history[key].append(avgs[key])

            if epoch % self.cfg.log_every == 0 or epoch == 1:
                msg = (
                    f"Teacher [{epoch}/{self.cfg.adam_epochs}]  "
                    f"total={total:.4e}  distill={avgs['distill']:.4e}  "
                    f"bc={avg_b:.4e}  phys={avgs['phys']:.4e}  lp={lam_phys:.3f}"
                )
                if has_anchor:
                    msg += f"  anchor={sum_a:.4e}"
                msg += f"  lr={scheduler.get_last_lr()[0]:.2e}"
                logger.info(msg)
                # Per-SWE-component detail (distill + phys) at every log.
                logger.info(
                    "  detail: "
                    f"d_xi={avgs['distill_xi']:.3e}  d_hu={avgs['distill_hu']:.3e}  "
                    f"d_hv={avgs['distill_hv']:.3e}  p_xi={avgs['phys_xi']:.3e}  "
                    f"p_hu={avgs['phys_hu']:.3e}  p_hv={avgs['phys_hv']:.3e}"
                )
                if self.memory_tracker is not None:
                    self.memory_tracker.sample("adam", epoch)
            if epoch % self.cfg.checkpoint_every == 0:
                self._save_checkpoint(f"adam_{epoch}")

    def _lbfgs_phase(self) -> None:
        if self.cfg.lbfgs_steps <= 0:
            return
        has_anchor = self.ref_data is not None and self.cfg.lambda_anchor > 0
        logger.info(f"L-BFGS polish: {self.cfg.lbfgs_steps} outer steps")
        if self.memory_tracker is not None:
            self.memory_tracker.reset_peak("teacher-lbfgs-start")
            self.memory_tracker.sample("lbfgs", 0)
        optimizer = optim.LBFGS(
            self.network.parameters(),
            lr=self.cfg.lbfgs_lr,
            max_iter=self.cfg.lbfgs_max_iter,
            history_size=50,
            line_search_fn="strong_wolfe",
        )
        norm = float(self._Q_snaps.shape[0])
        counter = [0]

        comp_keys = ("distill", "distill_xi", "distill_hu", "distill_hv",
                     "phys", "phys_xi", "phys_hu", "phys_hv")

        def closure():
            optimizer.zero_grad()
            sums = {k: 0.0 for k in comp_keys}
            sum_b = 0.0
            for k in range(self._Q_snaps.shape[0]):
                t_k = float(self._t_snaps[k])
                parts = self._snapshot_losses(
                    t_k, self._Q_snaps[k], self.cfg.lambda_phys
                )
                l_k = (parts["distill"]
                       + self.cfg.lambda_bc * parts["bc"]
                       + self.cfg.lambda_phys * parts["phys"]) / norm
                l_k.backward()
                for key in comp_keys:
                    sums[key] += parts[key].item()
                sum_b += parts["bc"].item()

            sa = 0.0
            if has_anchor:
                l_a = self._measurement_loss(self.ref_data)
                (self.cfg.lambda_anchor * l_a).backward()
                sa = l_a.item()

            avgs = {k: v / norm for k, v in sums.items()}
            avg_b = sum_b / norm
            total = (avgs["distill"] + self.cfg.lambda_bc * avg_b
                     + self.cfg.lambda_phys * avgs["phys"]
                     + (self.cfg.lambda_anchor * sa if has_anchor else 0.0))
            self.history["total"].append(total)
            self.history["bc"].append(avg_b)
            self.history["anchor"].append(sa)
            for key in comp_keys:
                self.history[key].append(avgs[key])

            counter[0] += 1
            if counter[0] % self.cfg.log_every == 0:
                msg = (f"L-BFGS [{counter[0]}]  total={total:.4e}  "
                       f"distill={avgs['distill']:.4e}  bc={avg_b:.4e}  "
                       f"phys={avgs['phys']:.4e}")
                if has_anchor:
                    msg += f"  anchor={sa:.4e}"
                logger.info(msg)
                logger.info(
                    "  detail: "
                    f"d_xi={avgs['distill_xi']:.3e}  d_hu={avgs['distill_hu']:.3e}  "
                    f"d_hv={avgs['distill_hv']:.3e}  p_xi={avgs['phys_xi']:.3e}  "
                    f"p_hu={avgs['phys_hu']:.3e}  p_hv={avgs['phys_hv']:.3e}"
                )
                if self.memory_tracker is not None:
                    self.memory_tracker.sample("lbfgs", counter[0])
            return torch.tensor(total, device=self.device, dtype=self.dtype)

        for _ in range(self.cfg.lbfgs_steps):
            optimizer.step(closure)

    # ------------------------------------------------------------------
    # Losses (shared between Adam and L-BFGS phases)
    # ------------------------------------------------------------------

    def _snapshot_losses(
        self, t_k: float, Q_target: torch.Tensor, lam_phys: float,
    ) -> Dict[str, torch.Tensor]:
        """distill + BC + (optional) physics-residual at a single time.

        Returns a dict with per-component breakdowns so the Adam / L-BFGS
        loops can record and log the SWE-variable split. The total
        ``distill`` and ``phys`` values are already weighted sums across
        ``lambda_xi`` / ``lambda_hu`` / ``lambda_hv``.
        """
        cell_xy = self.mesh_data["cell_center"]
        n_cells = cell_xy.shape[0]
        w_xi, w_hu, w_hv = self.cfg.lambda_xi, self.cfg.lambda_hu, self.cfg.lambda_hv

        # --- Distill (no grad on t) ---
        xyt_d = torch.cat(
            [cell_xy, torch.full((n_cells, 1), t_k, dtype=self.dtype, device=self.device)],
            dim=-1,
        )
        Q_pred_d = self.network(xyt_d)
        diff_d = (Q_pred_d - Q_target) ** 2
        d_xi = diff_d[:, 0].mean()
        d_hu = diff_d[:, 1].mean()
        d_hv = diff_d[:, 2].mean()
        l_distill = w_xi * d_xi + w_hu * d_hu + w_hv * d_hv

        # --- BC loss at t_k ---
        l_bc = self._unsteady_bc_loss(t_k)

        # --- Physics residual (only if λ_phys > 0) ---
        zero = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        if lam_phys > 0:
            t_col = torch.full(
                (n_cells, 1), t_k, dtype=self.dtype,
                device=self.device, requires_grad=True,
            )
            xyt_p = torch.cat([cell_xy, t_col], dim=-1)
            Q_pred_p = self.network(xyt_p)
            dQ_dt = torch.zeros_like(Q_pred_p)
            for j in range(3):
                grads = torch.autograd.grad(
                    Q_pred_p[:, j].sum(), t_col, create_graph=True,
                )[0]
                dQ_dt[:, j] = grads.squeeze(-1)
            R = compute_fvm_residual(Q_pred_p, self.mesh_data, self.h_still,
                                     self.cfg.h_dry)
            diff_p = (dQ_dt + R) ** 2
            p_xi = diff_p[:, 0].mean()
            p_hu = diff_p[:, 1].mean()
            p_hv = diff_p[:, 2].mean()
            l_phys = w_xi * p_xi + w_hu * p_hu + w_hv * p_hv
        else:
            p_xi = p_hu = p_hv = zero
            l_phys = zero

        return {
            "distill": l_distill, "distill_xi": d_xi,
            "distill_hu": d_hu,   "distill_hv": d_hv,
            "bc":        l_bc,
            "phys":      l_phys,  "phys_xi":    p_xi,
            "phys_hu":   p_hu,    "phys_hv":    p_hv,
        }

    def _unsteady_bc_loss(self, t_val: float) -> torch.Tensor:
        """Per-BC soft loss at a single time — same formulas as the standard
        FVM-PINN BC loss. Matches fvm_pinn_swe's ``unsteady_bc_loss`` exactly."""
        if self.bc_data is None:
            return torch.tensor(0.0, device=self.device, dtype=self.dtype)
        total: Optional[torch.Tensor] = None
        n_bc = 0
        for bc_info in self.bc_data.values():
            bc_type = bc_info["type"]
            xy = bc_info["xyt"][:, :2]
            n = xy.shape[0]
            t_col = torch.full((n, 1), t_val, dtype=xy.dtype, device=xy.device)
            xyt = torch.cat([xy, t_col], dim=-1)
            Q_pred = self.network(xyt)

            if bc_type == "inlet-q":
                nx, ny = bc_info["nx"], bc_info["ny"]
                val = bc_info["value"]
                hu_n = Q_pred[:, 1] * nx + Q_pred[:, 2] * ny
                l = ((hu_n + val) ** 2).mean()
            elif bc_type == "exit-h":
                val = bc_info["value"]
                h_still_bc = bc_info.get("h_still", 0.0)
                h_pred = Q_pred[:, 0] + h_still_bc
                l = ((h_pred - val) ** 2).mean()
            elif bc_type in ("wall", "symmetry"):
                nx, ny = bc_info["nx"], bc_info["ny"]
                un = Q_pred[:, 1] * nx + Q_pred[:, 2] * ny
                l = (un ** 2).mean()
            else:
                continue

            total = l if total is None else total + l
            n_bc += 1
        if total is None:
            return torch.tensor(0.0, device=self.device, dtype=self.dtype)
        return total / max(n_bc, 1)

    def _measurement_loss(self, ref_data: Dict) -> torch.Tensor:
        """MSE over anchor (reference) points. Respects var_mask if provided.

        ``var_mask`` is accepted in three shapes for compatibility with
        FVM_PINNDataset's sparse / dense / both modes:

        * ``[3]`` bool      — legacy global per-component selector
        * ``[3]`` float/int — broadcast multiplicative mask
        * ``[N, 3]`` any    — per-point multiplicative mask (sparse + dense
                              concatenated with different supervised vars)

        For multiplicative masks (float/int, any shape) the mean is taken
        over the masked-element count rather than over all elements, so
        zeroed-out columns/rows don't dilute the loss.
        """
        xyt = ref_data["xyt"]
        U_ref = ref_data["U_ref"]
        Q_pred = self.network(xyt)
        diff = (Q_pred - U_ref) ** 2

        var_mask = ref_data.get("var_mask")
        if var_mask is None:
            return diff.mean()

        if var_mask.dtype == torch.bool and var_mask.dim() == 1:
            return diff[:, var_mask].mean()

        keep = var_mask.to(dtype=diff.dtype, device=diff.device)
        if keep.dim() == 1:
            keep = keep.unsqueeze(0)  # [3] -> [1, 3] broadcast across points
        masked = diff * keep
        denom = keep.sum().clamp(min=1.0)
        return masked.sum() / denom

    # ------------------------------------------------------------------
    # Teacher trajectory (FVM RK2, cached)
    # ------------------------------------------------------------------

    def _build_trajectory(self) -> None:
        """Load from cache if compatible, else run FVM and save."""
        Q_init = self.ic_data["U_true"].to(device=self.device, dtype=self.dtype)
        M = self.cfg.n_snapshots
        duration = self.cfg.t_end - self.cfg.t_start
        n_cells = int(self.mesh_data["n_cells"])

        cache_path = (
            Path(self.cfg.trajectory_cache_path)
            if self.cfg.trajectory_cache_path
            else self.output_dir / "fvm_trajectory.pt"
        )
        meta = self._trajectory_meta(n_cells)

        if not self.cfg.regen_trajectory and cache_path.exists():
            try:
                data = torch.load(cache_path, map_location="cpu", weights_only=False)
                got = data.get("meta", {})
                mismatch = [k for k in meta if got.get(k) != meta[k]]
                if not mismatch:
                    self._t_snaps = data["t_snaps"].to(device=self.device, dtype=self.dtype)
                    self._Q_snaps = data["Q_snaps"].to(device=self.device, dtype=self.dtype)
                    logger.info(
                        f"Loaded cached FVM trajectory from {cache_path} "
                        f"({self._Q_snaps.shape[0]} snapshots)"
                    )
                    return
                logger.info(
                    f"Trajectory cache invalid (differs in {mismatch}); regenerating"
                )
            except Exception as e:  # noqa: BLE001 — any cache read error is recoverable
                logger.info(f"Cache load failed ({e}); regenerating trajectory")

        # Run FVM to generate snapshots
        logger.info(
            f"Generating FVM trajectory: {M+1} snapshots over "
            f"[{self.cfg.t_start}, {self.cfg.t_end}]s (duration {duration}s, cfl={self.cfg.cfl})"
        )
        t0 = time.time()
        t_snaps_rel, Q_snaps = self._run_trajectory(Q_init, duration, M)
        self._t_snaps = t_snaps_rel + self.cfg.t_start
        self._Q_snaps = Q_snaps
        logger.info(f"Trajectory generated in {time.time() - t0:.1f}s")

        # Save cache
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "t_snaps": self._t_snaps.detach().cpu(),
            "Q_snaps": self._Q_snaps.detach().cpu(),
            "meta": meta,
        }, cache_path)
        logger.info(f"Cached trajectory to {cache_path}")

    def _run_trajectory(
        self, Q_init: torch.Tensor, t_end: float, n_snapshots: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Thin wrapper over the shared ``run_fvm_rk2`` helper.

        Produces ``n_snapshots+1`` snapshots evenly spaced over ``[0, t_end]``
        (including t=0). All step logic — CFL dt, Heun RK2 stages, physical-h
        clamp, dry-cell momentum zeroing — lives in
        ``_internal/fvm/time_stepping.py`` so the teacher trainer and the
        standalone ``<case>_fvm_only.py`` drivers share a single
        implementation.
        """
        t_snaps = torch.linspace(
            0.0, t_end, n_snapshots + 1, dtype=self.dtype, device=self.device,
        )
        return run_fvm_rk2(
            Q_init, self.mesh_data, self.h_still,
            snapshot_times=t_snaps,
            cfl=self.cfg.cfl,
            h_dry=self.cfg.h_dry,
        )

    def _trajectory_meta(self, n_cells: int) -> Dict[str, Any]:
        """Cache-invalidation key. Must include anything that would change Q_snaps."""
        return {
            "n_cells":    int(n_cells),
            "t_start":    float(self.cfg.t_start),
            "t_end":      float(self.cfg.t_end),
            "snapshots":  int(self.cfg.n_snapshots),
            "cfl":        float(self.cfg.cfl),
            "h_dry":      float(self.cfg.h_dry),
        }

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def _phys_weight(self, epoch: int) -> float:
        """Linear warm-up: 0 for the first ``phys_warmup`` epochs, then
        ramp linearly to ``lambda_phys`` over another ``phys_warmup``."""
        warmup = self.cfg.phys_warmup
        if warmup <= 0:
            return self.cfg.lambda_phys
        if epoch <= warmup:
            return 0.0
        if epoch <= 2 * warmup:
            return self.cfg.lambda_phys * (epoch - warmup) / warmup
        return self.cfg.lambda_phys

    def _set_normalisation(self) -> None:
        """Normalise over the (cell × snapshot) grid spanned by the trajectory."""
        cell_xy = self.mesh_data["cell_center"]
        n_cells = cell_xy.shape[0]
        n_snaps = self._Q_snaps.shape[0]

        xy_rep = cell_xy.unsqueeze(0).expand(n_snaps, -1, -1).reshape(-1, 2)
        t_rep = self._t_snaps.view(-1, 1, 1).expand(n_snaps, n_cells, 1).reshape(-1, 1)
        xyt_all = torch.cat([xy_rep, t_rep], dim=-1)
        x_mean = xyt_all.mean(0)
        x_std = xyt_all.std(0).clamp(min=1e-6)
        self.network.set_normalisation(x_mean, x_std)

    def _save_checkpoint(self, tag: str) -> None:
        path = self.output_dir / f"teacher_{tag}.pt"
        torch.save({
            "network_state": self.network.state_dict(),
            "x_mean": self.network.x_mean.detach().cpu(),
            "x_std":  self.network.x_std.detach().cpu(),
            "history": self.history,
            "tag": tag,
        }, path)
