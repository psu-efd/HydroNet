"""
Adaptive loss weighting for FVM-PINN.

Tests whether the physics-only failure (BIC-A) is an artefact of the static,
hand-tuned loss weights. Selected by ``training.loss_weighting.method``:

``grad_norm``
    Learning-rate annealing of Wang, Teng & Perdikaris (2021), "Understanding
    and mitigating gradient flow pathologies in physics-informed neural
    networks", SIAM J. Sci. Comput. 43(5), A3055-A3081, Algorithm 1. Every
    ``update_every`` Adam steps the scalar weight of each constraint term
    i in {ic, bc, data} is moved towards

        lam_hat_i = max |grad_theta L_fvm| / mean |grad_theta L_i|

    by lam_i <- (1 - alpha) * lam_i + alpha * lam_hat_i. ``lambda_fvm`` stays
    fixed; terms whose configured weight is 0 stay switched off.

``self_adaptive``
    SA-PINN of McClenny & Braga-Neto (2023), "Self-adaptive physics-informed
    neural networks", J. Comput. Phys. 474, 111722. One trainable weight w per
    cell for the FVM residual and the IC, and one per boundary point for each
    BC (shared over the sampled times). The loss terms become
    mean(m(w) * r**2) with the polynomial mask m(w) = w**q, and the weights
    are updated by gradient *ascent* with their own optimiser, so that
    points with persistently large error gain weight.

Both schemes act in the Adam phase only. The weights are frozen at their
final values for L-BFGS, whose line search needs a fixed objective.
"""

import logging
from typing import Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


def build_loss_weighting(cfg: Optional[Dict], trainer) -> Optional["LossWeighting"]:
    """Return the weighting scheme named in ``cfg['method']`` (None = static)."""
    if not cfg:
        return None
    method = str(cfg.get("method", "static"))
    if method == "static":
        return None
    if method == "grad_norm":
        return GradNormAnnealing(cfg, trainer)
    if method == "self_adaptive":
        return SelfAdaptiveWeights(cfg, trainer)
    raise ValueError(
        f"Unknown training.loss_weighting.method: {method!r} "
        "(expected 'static', 'grad_norm', or 'self_adaptive')"
    )


class LossWeighting:
    """Hooks called by BaseTrainer around each Adam step."""

    def point_weights(self) -> Optional[Dict]:
        """Per-point weights passed to FVMPINNLoss (None = none)."""
        return None

    def before_backward(self, losses: Dict[str, torch.Tensor], step: int) -> None:
        """May update weights and recompute ``losses['total']`` in place."""

    def after_step(self) -> None:
        """Called after the network optimiser step."""

    def freeze(self) -> None:
        """Fix the weights (end of the Adam phase)."""

    def scalars(self) -> Dict[str, torch.Tensor]:
        """Weight summaries recorded in the history every step."""
        return {}

    def describe(self) -> str:
        """Short text appended to the training log line."""
        return ""

    def state_dict(self) -> Dict:
        return {}

    def load_state_dict(self, state: Dict) -> None:
        pass


# ---------------------------------------------------------------------------
# Learning-rate annealing (Wang, Teng & Perdikaris 2021)
# ---------------------------------------------------------------------------

class GradNormAnnealing(LossWeighting):

    def __init__(self, cfg: Dict, trainer) -> None:
        self.trainer = trainer
        self.loss_cfg = trainer.loss_fn.cfg     # weights are edited in place
        self.update_every = int(cfg.get("update_every", 10))
        self.alpha = float(cfg.get("alpha", 0.9))
        self.terms: List[str] = [
            k for k in ("ic", "bc", "data")
            if getattr(self.loss_cfg, f"lambda_{k}") > 0
        ]
        logger.info(
            f"Loss weighting: grad_norm (Wang et al. 2021), terms={self.terms}, "
            f"update_every={self.update_every}, alpha={self.alpha}, initial "
            + ", ".join(f"lambda_{k}={getattr(self.loss_cfg, f'lambda_{k}'):.3g}"
                        for k in self.terms)
        )

    def before_backward(self, losses: Dict[str, torch.Tensor], step: int) -> None:
        if step % self.update_every != 0:
            return
        params = [p for p in self.trainer.network.parameters() if p.requires_grad]

        def _grads(loss: torch.Tensor) -> Optional[torch.Tensor]:
            if not loss.requires_grad:
                return None
            gs = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
            gs = [g.reshape(-1) for g in gs if g is not None]
            return torch.cat(gs).abs() if gs else None

        g_res = _grads(losses["fvm"])
        if g_res is None:
            return
        g_max = g_res.max()
        for k in self.terms:
            g_k = _grads(losses[k])
            if g_k is None or g_k.mean() <= 0:
                continue
            lam_hat = (g_max / g_k.mean()).item()
            lam = getattr(self.loss_cfg, f"lambda_{k}")
            setattr(self.loss_cfg, f"lambda_{k}",
                    (1.0 - self.alpha) * lam + self.alpha * lam_hat)
        losses["total"] = self.trainer.loss_fn.weighted_total(losses)

    def scalars(self) -> Dict[str, torch.Tensor]:
        return {f"lam_{k}": torch.tensor(getattr(self.loss_cfg, f"lambda_{k}"))
                for k in self.terms}

    def describe(self) -> str:
        return "  ".join(f"lam_{k}={getattr(self.loss_cfg, f'lambda_{k}'):.2e}"
                         for k in self.terms)

    def state_dict(self) -> Dict:
        return {f"lambda_{k}": getattr(self.loss_cfg, f"lambda_{k}") for k in self.terms}

    def load_state_dict(self, state: Dict) -> None:
        for k, v in state.items():
            setattr(self.loss_cfg, k, float(v))


# ---------------------------------------------------------------------------
# Self-adaptive weights (McClenny & Braga-Neto 2023)
# ---------------------------------------------------------------------------

class SelfAdaptiveWeights(LossWeighting):

    def __init__(self, cfg: Dict, trainer) -> None:
        self.q = float(cfg.get("q", 2.0))
        init = float(cfg.get("init", 1.0))
        groups = list(cfg.get("groups", ["fvm", "ic", "bc"]))
        dev, dt = trainer.device, trainer.dtype

        def _w(n: int) -> torch.Tensor:
            return torch.full((n,), init, device=dev, dtype=dt, requires_grad=True)

        self.w: Dict[str, torch.Tensor] = {}
        if "fvm" in groups:
            self.w["fvm"] = _w(int(trainer.mesh_data["n_cells"]))
        if "ic" in groups and trainer.ic_data is not None:
            self.w["ic"] = _w(trainer.ic_data["xyt"].shape[0])
        self.bc_ids: List = []
        if "bc" in groups and trainer.bc_data is not None:
            for bc_id, info in trainer.bc_data.items():
                if isinstance(info, dict) and "xyt" in info:
                    self.w[f"bc_{bc_id}"] = _w(info["xyt"].shape[0])
                    self.bc_ids.append(bc_id)
        # Log label per weight vector, e.g. "bc_1_inlet-q"
        self._tag = {k: k for k in self.w}
        self._tag.update({f"bc_{b}": f"bc_{b}_{trainer.bc_data[b]['type']}" for b in self.bc_ids})

        lr = float(cfg.get("lr", 5e-3))
        opt = str(cfg.get("optimizer", "adam"))
        params = list(self.w.values())
        if opt == "adam":
            self.optimizer = torch.optim.Adam(params, lr=lr, maximize=True)
        elif opt == "sgd":
            self.optimizer = torch.optim.SGD(params, lr=lr, maximize=True)
        else:
            raise ValueError(f"loss_weighting.optimizer must be 'adam' or 'sgd', got {opt!r}")
        logger.info(
            f"Loss weighting: self_adaptive (McClenny & Braga-Neto 2023), "
            f"mask m(w)=w^{self.q:g}, init={init}, {opt} lr={lr}, "
            + ", ".join(f"{k}:{v.numel()}" for k, v in self.w.items())
        )

    def _mask(self, w: torch.Tensor) -> torch.Tensor:
        return w ** self.q

    def point_weights(self) -> Optional[Dict]:
        pw: Dict = {}
        for k in ("fvm", "ic"):
            if k in self.w:
                pw[k] = self._mask(self.w[k])
        if self.bc_ids:
            pw["bc"] = {bc_id: self._mask(self.w[f"bc_{bc_id}"]) for bc_id in self.bc_ids}
        return pw

    def after_step(self) -> None:
        if self.optimizer is None:
            return
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

    def freeze(self) -> None:
        for w in self.w.values():
            w.requires_grad_(False)
            w.grad = None
        self.optimizer = None
        logger.info(f"Self-adaptive weights frozen: {self.describe()}")

    def scalars(self) -> Dict[str, torch.Tensor]:
        out = {}
        with torch.no_grad():
            for k, w in self.w.items():
                m = self._mask(w)
                out[f"saw_{k}_mean"] = m.mean()
                out[f"saw_{k}_max"] = m.max()
        return out

    def describe(self) -> str:
        with torch.no_grad():
            parts = []
            for k, w in self.w.items():
                m = self._mask(w)
                parts.append(f"saw[{self._tag[k]}]={m.mean().item():.2e}/{m.max().item():.2e}")
            return "  ".join(parts)

    def state_dict(self) -> Dict:
        return {
            "w": {k: w.detach().clone() for k, w in self.w.items()},
            "optimizer": self.optimizer.state_dict() if self.optimizer else None,
        }

    def load_state_dict(self, state: Dict) -> None:
        with torch.no_grad():
            for k, v in state["w"].items():
                self.w[k].copy_(v.to(self.w[k]))
        if state.get("optimizer") is not None and self.optimizer is not None:
            self.optimizer.load_state_dict(state["optimizer"])
