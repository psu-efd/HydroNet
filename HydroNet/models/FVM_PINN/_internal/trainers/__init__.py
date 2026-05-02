"""
Trainer module — factory-registered FVM-PINN trainers.

Available trainers
------------------
"standard"  : StandardTrainer  — full-batch baseline
"minibatch" : MiniBatchTrainer — Strategy 1: cell mini-batching
"window"    : TimeWindowTrainer — Strategy 2: time-window decomposition
"teacher"   : TeacherTrainer    — Strategy 4: FVM-trajectory distillation

Strategy 3 (gradient checkpointing) is a flag on all trainers:
    cfg.use_grad_checkpoint = True

Usage
-----
    from HydroNet.models.FVM_PINN._internal.trainers import TrainerFactory

    cfg = StandardTrainerConfig(adam_epochs=3000, use_grad_checkpoint=True)
    trainer = TrainerFactory("standard")(cfg, net_cfg, loss_cfg, mesh_data)
    trainer.setup(ic_data)
    trainer.train()
"""

from .base_trainer import BaseTrainer, BaseTrainerConfig
from .standard_trainer import StandardTrainer, StandardTrainerConfig
from .minibatch_trainer import MiniBatchTrainer, MiniBatchTrainerConfig
from .window_trainer import TimeWindowTrainer, TimeWindowConfig
from .teacher_trainer import TeacherTrainer, TeacherTrainerConfig

# Registry
_TRAINER_REGISTRY = {
    "standard":  StandardTrainer,
    "minibatch": MiniBatchTrainer,
    "window":    TimeWindowTrainer,
    "teacher":   TeacherTrainer,
}


def TrainerFactory(name: str):
    """Return trainer class by name. Raises KeyError for unknown names."""
    if name not in _TRAINER_REGISTRY:
        raise KeyError(
            f"Unknown trainer '{name}'. "
            f"Available: {list(_TRAINER_REGISTRY.keys())}"
        )
    return _TRAINER_REGISTRY[name]


__all__ = [
    "BaseTrainer", "BaseTrainerConfig",
    "StandardTrainer", "StandardTrainerConfig",
    "MiniBatchTrainer", "MiniBatchTrainerConfig",
    "TimeWindowTrainer", "TimeWindowConfig",
    "TeacherTrainer", "TeacherTrainerConfig",
    "TrainerFactory",
]
