"""
Training wrapper for FVM-PINN.

``FVM_PINNTrainer`` is the HydroNet-shape (``(model, dataset, config)``)
facade over the internal trainer family (``StandardTrainer``,
``MiniBatchTrainer``, ``TimeWindowTrainer``). It:

    1. Translates the YAML ``training.*`` block into the matching internal
       dataclass config (``BaseTrainerConfig`` / ``MiniBatchTrainerConfig``
       / ``TimeWindowConfig``).
    2. Builds the chosen internal trainer from ``TrainerFactory(strategy)``.
    3. Replaces the internal trainer's network with the wrapped model's
       network so training and inference share the same parameters.
    4. Wires ``set_mesh_context`` on the model so ``model.forward`` returns
       physical ``[h, u, v]`` after training.
    5. Delegates ``setup``/``train``/``predict`` to the internal trainer.

After ``train()`` returns, callers can either:
    * call ``model(xyt)`` (physical ``[h, u, v]``) — works for
      ``standard`` and ``minibatch`` strategies, and for ``window`` reflects
      the final window's parameters;
    * call ``trainer.predict(xyt)`` — returns physical ``[h, u, v]`` and, for
      the ``window`` strategy, automatically routes each query to the
      correct window's network.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from torch.utils.data import Dataset

from ...utils.config import Config
from .data import FVM_PINNDataset
from .model import FVM_SWE_PINN
from ._internal.trainers import (
    BaseTrainer, BaseTrainerConfig,
    StandardTrainer, StandardTrainerConfig,
    MiniBatchTrainer, MiniBatchTrainerConfig,
    TimeWindowTrainer, TimeWindowConfig,
    TeacherTrainer, TeacherTrainerConfig,
)
from ._internal.trainers.memory_tracker import MemoryTracker
from ._internal.pinn.loss import LossConfig, FVMPINNLoss

logger = logging.getLogger(__name__)


class FVM_PINNTrainer:
    """
    Trainer for :class:`FVM_SWE_PINN`.
    """

    def __init__(
        self,
        model: FVM_SWE_PINN,
        dataset: Dataset,
        config: Config,
    ) -> None:
        if not isinstance(model, FVM_SWE_PINN):
            raise ValueError("model must be an FVM_SWE_PINN")
        if not isinstance(dataset, FVM_PINNDataset):
            raise ValueError("dataset must be an FVM_PINNDataset")
        if not isinstance(config, Config):
            raise ValueError("config must be a Config object")

        self.model = model
        self.dataset = dataset
        self.config = config

        self.strategy = str(config.get('training.strategy', 'standard'))
        self.device = model.get_device()
        self.dtype = model.dtype

        # ---- Wire mesh context into the model so public forward works ----
        model.set_mesh_context(
            h_still_cells=dataset.get_h_still(),
            cell_xy=dataset.get_cell_xy(),
        )

        # ---- Build internal trainer ----
        net_cfg = model.get_network_config()
        loss_cfg = self._build_loss_cfg()
        mesh_data = dataset.get_mesh_data()

        self._trainer: Any
        if self.strategy == 'standard':
            trainer_cfg = self._build_standard_cfg()
            self._trainer = StandardTrainer(
                trainer_cfg, net_cfg, loss_cfg, mesh_data,
            )
            self._patch_network(self._trainer)
        elif self.strategy == 'minibatch':
            trainer_cfg = self._build_minibatch_cfg()
            self._trainer = MiniBatchTrainer(
                trainer_cfg, net_cfg, loss_cfg, mesh_data,
            )
            self._patch_network(self._trainer)
        elif self.strategy == 'window':
            window_cfg = self._build_window_cfg()
            base_cls = StandardTrainer  # per-window trainer; could be made configurable
            self._trainer = TimeWindowTrainer(
                window_cfg, net_cfg, loss_cfg, mesh_data,
                base_trainer_cls=base_cls,
            )
            # TimeWindowTrainer builds its per-window networks lazily inside
            # its own ``train()`` — we can't swap them here. The model's
            # ``_net`` therefore reflects only the final window after train;
            # use ``trainer.predict(xyt)`` for across-window inference.
        elif self.strategy == 'teacher':
            teacher_cfg = self._build_teacher_cfg()
            self._trainer = TeacherTrainer(
                teacher_cfg, net_cfg, loss_cfg, mesh_data,
            )
            # TeacherTrainer's constructor builds its own network; swap it
            # for the model's so parameters are shared.
            self._trainer.network = self.model.get_internal_network().to(
                device=self._trainer.device, dtype=self._trainer.dtype
            )
        else:
            raise ValueError(
                f"Unknown training.strategy: {self.strategy!r} "
                "(expected 'standard', 'minibatch', 'window', or 'teacher')"
            )

        # ---- Feed training data to the internal trainer ----
        self._trainer.setup(
            ic_data=dataset.get_ic_data(),
            bc_data=dataset.get_bc_data(),
            ref_data=dataset.get_ref_data(),
        )

        # ---- GPU memory tracking ----
        # Always attach a tracker. On CPU it is a no-op; on CUDA it logs a
        # ``MEM`` line every ``log_every`` epochs and saves
        # ``memory_history.json`` next to the checkpoints when train() finishes.
        ckpt_dir = Path(
            config.get('training.logging.checkpoint_dir',
                       config.get('training.output_dir', 'outputs'))
        )
        run_label = ckpt_dir.name or "fvm_pinn"
        self._memory_tracker = MemoryTracker(self.device, label=run_label)
        self._trainer.memory_tracker = self._memory_tracker
        self._memory_history_path = ckpt_dir / "memory_history.json"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self) -> Tuple[Dict[str, list], Dict[str, Any]]:
        """
        Run training (Adam + L-BFGS).

        Returns
        -------
        history : dict[str, list[float]]
            Per-step losses (``total``, ``fvm``, ``ic``, ``bc``, ``data``, ...).
        predictions_and_true_values : dict[str, Any]
            End-of-training snapshot at the dataset's IC coords — used by the
            example scripts to render VTKs / plots without reloading a
            checkpoint.
        """
        final_loss = self._trainer.train()

        # Persist GPU memory history (no-op on CPU — empty records list).
        try:
            self._memory_tracker.save_json(self._memory_history_path)
        except Exception as exc:  # noqa: BLE001 — memory log is best-effort
            logger.warning(f"Could not save memory_history.json: {exc}")

        # Build a lightweight predictions/targets bundle in the same spirit
        # as PINNTrainer (which returns a JSON-serialisable dict the example
        # scripts dump to disk). For FVM-PINN we only have the IC dataset as
        # a natural set of (xyt, U_true) pairs; anchor/ref data is also
        # available if the user provided it.
        ic_data = self.dataset.get_ic_data()
        ref_data = self.dataset.get_ref_data()

        with torch.no_grad():
            Q_ic = self._trainer.predict(ic_data["xyt"])
        out: Dict[str, Any] = {
            "ic_xyt":    ic_data["xyt"].detach().cpu(),
            "ic_U_true": ic_data["U_true"].detach().cpu(),
            "ic_U_pred": Q_ic.detach().cpu(),
        }
        if ref_data is not None:
            with torch.no_grad():
                Q_ref = self._trainer.predict(ref_data["xyt"])
            out["ref_xyt"]   = ref_data["xyt"].detach().cpu()
            out["ref_U_ref"] = ref_data["U_ref"].detach().cpu()
            out["ref_U_pred"] = Q_ref.detach().cpu()

        history = self._trainer.history
        out["final_loss"] = float(final_loss) if final_loss is not None else float("nan")
        return history, out

    def predict(self, xyt: torch.Tensor) -> torch.Tensor:
        """
        Inference convenience: returns physical ``[h, u, v]`` on
        arbitrary ``(x, y, t)`` points. Routes through the internal
        trainer (which handles per-window routing for strategy='window')
        and converts xi → h via the model's h_still context.
        """
        Q = self._trainer.predict(xyt)
        h_still = self.model._h_still_at(xyt)
        h = Q[..., 0] + h_still.to(dtype=Q.dtype, device=Q.device)
        h_safe = h.clamp(min=1e-6)
        u = Q[..., 1] / h_safe
        v = Q[..., 2] / h_safe
        return torch.stack([h, u, v], dim=-1)

    # Accessors kept for parity with PINNTrainer / debugging
    @property
    def internal_trainer(self):
        return self._trainer

    @property
    def history(self) -> Dict[str, list]:
        return self._trainer.history

    def load_checkpoint(self, path: str) -> None:
        self._trainer.load_checkpoint(path)

    # ------------------------------------------------------------------
    # Config translation (YAML -> internal dataclass configs)
    # ------------------------------------------------------------------

    def _build_loss_cfg(self) -> LossConfig:
        lw = 'training.loss_weights'
        cw = 'training.component_weights'
        cfg = self.config
        return LossConfig(
            lambda_fvm=float(cfg.get(f'{lw}.lambda_fvm', 1.0)),
            lambda_ic=float(cfg.get(f'{lw}.lambda_ic', 10.0)),
            lambda_bc=float(cfg.get(f'{lw}.lambda_bc', 30.0)),
            lambda_data=float(cfg.get(f'{lw}.lambda_data', 10.0)),
            lambda_xi=float(cfg.get(f'{cw}.lambda_xi', 1.0)),
            lambda_hu=float(cfg.get(f'{cw}.lambda_hu', 1.0)),
            lambda_hv=float(cfg.get(f'{cw}.lambda_hv', 1.0)),
            h_dry=float(cfg.get('physics.h_dry', 1e-2)),
            use_grad_checkpoint=bool(
                cfg.get('training.use_grad_checkpoint', False)
            ),
        )

    def _fill_base_cfg(self, cfg_obj: BaseTrainerConfig) -> BaseTrainerConfig:
        """Populate the shared BaseTrainerConfig fields from the YAML."""
        cfg = self.config
        cfg_obj.adam_epochs = int(cfg.get('training.adam_epochs', 5000))
        cfg_obj.adam_lr = float(cfg.get('training.adam_lr', 1e-3))
        cfg_obj.adam_lr_decay = float(cfg.get('training.adam_lr_decay', 0.9))
        cfg_obj.adam_decay_every = int(cfg.get('training.adam_decay_every', 1000))
        cfg_obj.lbfgs_epochs = int(cfg.get('training.lbfgs_epochs', 500))
        cfg_obj.lbfgs_max_iter = int(cfg.get('training.lbfgs_max_iter', 20))
        cfg_obj.lbfgs_lr = float(cfg.get('training.lbfgs_lr', 1.0))
        cfg_obj.n_time_samples = int(cfg.get('training.n_time_samples', 8))
        cfg_obj.t_start = float(cfg.get('training.t_start', 0.0))
        cfg_obj.t_end = float(cfg.get('training.t_end', 1.0))
        cfg_obj.use_grad_checkpoint = bool(
            cfg.get('training.use_grad_checkpoint', False)
        )
        cfg_obj.log_every = int(
            cfg.get('training.logging.print_freq',
                    cfg.get('training.log_every', 200))
        )
        cfg_obj.checkpoint_every = int(
            cfg.get('training.logging.save_freq',
                    cfg.get('training.checkpoint_every', 1000))
        )
        cfg_obj.output_dir = str(
            cfg.get('training.logging.checkpoint_dir',
                    cfg.get('training.output_dir', 'outputs'))
        )
        cfg_obj.device = str(self.device)
        cfg_obj.dtype = "float64"
        return cfg_obj

    def _build_standard_cfg(self) -> StandardTrainerConfig:
        return self._fill_base_cfg(StandardTrainerConfig())

    def _build_minibatch_cfg(self) -> MiniBatchTrainerConfig:
        cfg = MiniBatchTrainerConfig()
        self._fill_base_cfg(cfg)
        cfg.cell_sample_frac = float(
            self.config.get('training.minibatch.cell_sample_frac', 0.2)
        )
        cfg.sampling = str(
            self.config.get('training.minibatch.sampling', 'random')
        )
        return cfg

    def _build_window_cfg(self) -> TimeWindowConfig:
        per_window_cfg = StandardTrainerConfig()
        self._fill_base_cfg(per_window_cfg)
        return TimeWindowConfig(
            n_windows=int(self.config.get('training.window.n_windows', 5)),
            overlap_frac=float(self.config.get('training.window.overlap_frac', 0.10)),
            warm_start=bool(self.config.get('training.window.warm_start', True)),
            lambda_continuity=float(
                self.config.get('training.window.lambda_continuity', 5.0)
            ),
            per_window_cfg=per_window_cfg,
        )

    def _build_teacher_cfg(self) -> TeacherTrainerConfig:
        """Translate ``training.teacher.*`` and shared ``training.*`` keys
        into a TeacherTrainerConfig.

        Note: the teacher strategy uses its own loss weights (distill is
        implicit 1.0; bc/phys/anchor are specified under ``training.teacher.*``
        so they do not collide with ``training.loss_weights`` used by the
        other strategies).
        """
        cfg = self.config
        return TeacherTrainerConfig(
            # Trajectory
            n_snapshots=int(cfg.get('training.teacher.n_snapshots', 30)),
            cfl=float(cfg.get('training.teacher.cfl', 0.1)),
            h_dry=float(cfg.get('physics.h_dry', 1e-2)),
            trajectory_cache_path=str(
                cfg.get('training.teacher.trajectory_cache', '')
            ),
            regen_trajectory=bool(
                cfg.get('training.teacher.regen_trajectory', False)
            ),
            # Optimisation
            adam_epochs=int(cfg.get('training.teacher.adam_epochs',
                                    cfg.get('training.adam_epochs', 3000))),
            adam_lr=float(cfg.get('training.teacher.lr',
                                  cfg.get('training.adam_lr', 1e-3))),
            adam_decay_every=int(cfg.get('training.teacher.adam_decay_every',
                                         cfg.get('training.adam_decay_every', 1000))),
            adam_lr_decay=float(cfg.get('training.teacher.adam_lr_decay',
                                        cfg.get('training.adam_lr_decay', 0.95))),
            lbfgs_steps=int(cfg.get('training.teacher.lbfgs_steps',
                                    cfg.get('training.lbfgs_epochs', 200))),
            lbfgs_max_iter=int(cfg.get('training.lbfgs_max_iter', 20)),
            lbfgs_lr=float(cfg.get('training.lbfgs_lr', 1.0)),
            phys_warmup=int(cfg.get('training.teacher.phys_warmup', 200)),
            # Loss weights
            lambda_bc=float(cfg.get('training.teacher.lambda_bc', 1.0)),
            lambda_phys=float(cfg.get('training.teacher.lambda_phys', 0.05)),
            lambda_anchor=float(cfg.get('training.teacher.lambda_anchor', 1.0)),
            # Per-conserved-variable weights (shared with FVMPINNLoss via
            # training.component_weights.*). Defaults are uniform 1.0.
            lambda_xi=float(cfg.get('training.component_weights.lambda_xi', 1.0)),
            lambda_hu=float(cfg.get('training.component_weights.lambda_hu', 1.0)),
            lambda_hv=float(cfg.get('training.component_weights.lambda_hv', 1.0)),
            # I/O
            log_every=int(cfg.get('training.logging.print_freq',
                                  cfg.get('training.log_every', 50))),
            checkpoint_every=int(cfg.get('training.logging.save_freq',
                                         cfg.get('training.checkpoint_every', 1000))),
            output_dir=str(cfg.get('training.logging.checkpoint_dir',
                                   cfg.get('training.output_dir', 'outputs'))),
            # Hardware
            device=str(self.device),
            dtype="float64",
            # Time window
            t_start=float(cfg.get('training.t_start', 0.0)),
            t_end=float(cfg.get('training.t_end', 1.0)),
        )

    # ------------------------------------------------------------------
    # Network-swap: share parameters between model and internal trainer
    # ------------------------------------------------------------------

    def _patch_network(self, trainer: BaseTrainer) -> None:
        """
        Replace the internal trainer's freshly-built SWENet with the one
        owned by ``self.model`` so ``model(xyt)`` and ``trainer.predict(xyt)``
        share the same parameters.

        Also rebuild the internal ``FVMPINNLoss`` so ``loss_fn.h_still``
        matches ``dataset.h_still`` (the internal trainer's constructor
        pulled ``h_still`` from ``mesh_data`` and re-cast it; rebuilding
        here keeps the network swap + h_still path consistent).
        """
        trainer.network = self.model.get_internal_network().to(
            device=trainer.device, dtype=trainer.dtype
        )
        trainer.loss_fn = FVMPINNLoss(
            trainer.loss_fn.cfg,
            trainer.mesh_data,
            self.dataset.get_h_still(),
            trainer.loss_fn.bc_config,
        )
