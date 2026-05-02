"""
FVM-PINN model for 2D shallow water equations.

``FVM_SWE_PINN`` pairs a SWENet (MLP with Fourier positional encoding) with a
well-balanced Roe FVM residual computed on an unstructured SRH-2D mesh. The
internal network predicts the perturbation-form conserved variables
``[xi, hu, hv]`` where ``xi = h - h_still``; the public ``forward`` wraps this
to ``[h, u, v]`` so the model slots into the same post-training evaluation
path (VTK export, L2 comparison) as ``SWE_PINN``.

The perturbation form is what makes the well-balanced property exact —
changing it would break the Roe flux / bed-slope cancellation on which the
stability of the solver depends. Do not "simplify" this.

Normalization choices (differs from PI-DeepONet)
------------------------------------------------
- **Input normalization**: z-score (``(xyt − x_mean) / x_std``). The
  internal mesh is unstructured and cells cluster where the mesh is
  refined; z-score reflects that sample distribution, whereas min-max
  (which PI-DeepONet uses for ``x, y, t``) weights the bounding box
  corners even where no cells live.
- **Output normalization**: none. The network predicts physical
  ``[xi, hu, hv]`` in SI units so the Roe solver and FVM residual can
  consume the output directly without per-call denormalization. Scale
  mismatches between xi (often << 1 under the well-balanced form) and
  hu/hv are handled via explicit per-component loss weights
  (``LossConfig.lambda_xi`` / ``lambda_hu`` / ``lambda_hv``), not by
  rescaling the output.
"""

import torch
import torch.nn as nn
from typing import Optional

from ...utils.config import Config
from ._internal.pinn.network import SWENet, SirenSWENet, NetworkConfig


class FVM_SWE_PINN(nn.Module):
    """
    Physics-Informed Neural Network for 2D SWE backed by a differentiable
    well-balanced Roe FVM residual on an unstructured mesh.

    The physics loss and all training-time bookkeeping happen inside
    ``FVM_PINNTrainer`` — ``FVM_SWE_PINN`` itself is just the network
    wrapper with the ``[xi, hu, hv] → [h, u, v]`` conversion at its public
    boundary.
    """

    def __init__(self, config):
        super().__init__()

        if not isinstance(config, Config):
            raise ValueError("config must be a Config object")

        expected_name = "FVM_SWE_PINN"
        actual_name = config.get_required_config('model.name')
        if actual_name != expected_name:
            raise ValueError(
                f"model.name must be '{expected_name}', got '{actual_name}'"
            )

        self.config = config

        # ---- Device ----
        device_type = config.get('device.type', 'cpu')
        device_index = int(config.get('device.index', 0))
        if device_type == 'cuda' and torch.cuda.is_available():
            self.device = torch.device(f'cuda:{device_index}')
        else:
            self.device = torch.device('cpu')

        # ---- Dtype ----
        # FP64 throughout the internal network. The FVM residual requires
        # FP64 for the Roe pressure/source cancellation; a mixed-precision
        # mode (network FP32, residual FP64) is a potential future
        # optimisation but would need explicit casts at the model boundary.
        self.dtype = torch.float64

        # ---- Build internal SWENet ----
        arch = config.get('model.arch', 'mlp')
        net_cfg = NetworkConfig(
            hidden_dim=int(config.get('model.hidden_dim', 128)),
            n_layers=int(config.get('model.n_layers', 6)),
            n_fourier=int(config.get('model.n_fourier', 64)),
            fourier_scale=float(config.get('model.fourier_scale', 1.0)),
            activation=str(config.get('model.activation', 'tanh')),
            use_residual=bool(config.get('model.use_residual', True)),
            output_dim=3,
        )
        self._net_cfg = net_cfg

        if arch == 'siren':
            w0_first = float(config.get('model.siren_w0', 30.0))
            self._net: nn.Module = SirenSWENet(net_cfg, w0_first=w0_first)
        elif arch == 'mlp':
            self._net = SWENet(net_cfg)
        else:
            raise ValueError(
                f"Unknown model.arch: {arch!r} (expected 'mlp' or 'siren')"
            )
        self._net = self._net.to(device=self.device, dtype=self.dtype)

        # ---- Mesh context (wired in by FVM_PINNTrainer after dataset build) ----
        # ``_h_still_cells``: per-cell h_still = max(0, wse_still - bed).
        # ``_cell_xy``      : per-cell (x, y) for nearest-cell lookup used by
        # the public forward when called at arbitrary points (e.g. VTK nodes).
        self._h_still_cells: Optional[torch.Tensor] = None
        self._cell_xy: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_device(self) -> torch.device:
        """Mirror ``SWE_PINN.get_device`` for HydroNet helper compatibility."""
        return self.device

    def get_internal_network(self) -> nn.Module:
        """Return the raw SWENet so the trainer can share its parameters."""
        return self._net

    def get_network_config(self) -> NetworkConfig:
        """Return the internal NetworkConfig so the trainer can match the arch."""
        return self._net_cfg

    # ------------------------------------------------------------------
    # Mesh context injection (called by FVM_PINNTrainer after dataset build)
    # ------------------------------------------------------------------

    def set_mesh_context(
        self,
        h_still_cells: torch.Tensor,
        cell_xy: torch.Tensor,
    ) -> None:
        """
        Attach per-cell ``h_still`` and cell coordinates so ``forward()`` can
        convert the internal ``[xi, hu, hv]`` output to physical ``[h, u, v]``
        at arbitrary query points via nearest-cell lookup.
        """
        self._h_still_cells = h_still_cells.to(
            device=self.device, dtype=self.dtype
        )
        self._cell_xy = cell_xy.to(device=self.device, dtype=self.dtype)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _internal_forward(self, xyt: torch.Tensor) -> torch.Tensor:
        """Raw network output ``[xi, hu, hv]``. Used by the trainer."""
        return self._net(xyt)

    def forward(self, xyt: torch.Tensor) -> torch.Tensor:
        """
        Public forward: returns physical ``[h, u, v]``.

        ``h = xi + h_still`` using a nearest-cell lookup of ``h_still`` at the
        query ``(x, y)``. ``u = hu / max(h, 1e-6)``, ``v = hv / max(h, 1e-6)``
        to keep velocities finite on dry cells.

        If ``set_mesh_context`` has not been called (e.g. the model is used
        standalone before a dataset is attached), ``h_still`` defaults to 0
        and the output is equivalent to the raw network prediction reshaped
        into ``[h, u, v]`` form — useful for architecture smoke tests, not
        for meaningful physical output.
        """
        Q = self._internal_forward(xyt)
        h_still = self._h_still_at(xyt)
        h = Q[..., 0] + h_still
        h_safe = h.clamp(min=1e-6)
        u = Q[..., 1] / h_safe
        v = Q[..., 2] / h_safe
        return torch.stack([h, u, v], dim=-1)

    # ------------------------------------------------------------------
    # Private: nearest-cell h_still lookup
    # ------------------------------------------------------------------

    def _h_still_at(self, xyt: torch.Tensor) -> torch.Tensor:
        """Nearest-cell ``h_still`` at the ``(x, y)`` slice of each input row."""
        if self._h_still_cells is None or self._cell_xy is None:
            shape = xyt.shape[:-1]
            return torch.zeros(shape, dtype=xyt.dtype, device=xyt.device)

        xy = xyt[..., :2].reshape(-1, 2).to(
            dtype=self._cell_xy.dtype, device=self._cell_xy.device
        )
        # torch.cdist is O(N·M); fine for post-training VTK export sizes
        # (tens of thousands of query points × thousands of cells).
        d = torch.cdist(xy, self._cell_xy)
        idx = d.argmin(dim=-1)
        return self._h_still_cells[idx].reshape(xyt.shape[:-1])
