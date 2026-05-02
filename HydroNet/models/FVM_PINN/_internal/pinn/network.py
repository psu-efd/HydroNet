"""
Neural network architecture for FVM-PINN SWE solver.

Input:  (x, y, t)  — spatial position and time
Output: (h, hu, hv) — conserved SWE variables

Architecture: MLP with Fourier feature positional encoding.

Fourier features improve the network's ability to represent high-frequency
spatial variation across large domains (typical for hydraulic applications).
"""

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class NetworkConfig:
    """PINN network hyperparameters."""
    hidden_dim: int = 128
    n_layers: int = 6
    n_fourier: int = 64           # number of Fourier feature pairs
    fourier_scale: float = 1.0    # scale for random Fourier frequencies
    activation: str = "tanh"      # tanh or gelu
    use_residual: bool = True      # residual connections every 2 layers
    output_dim: int = 3           # [h, hu, hv]


class FourierEmbedding(nn.Module):
    """
    Random Fourier feature embedding for (x, y, t).

    Maps input [x, y, t] to [cos(B x), sin(B x)] where B is fixed random.
    Output dim = 2 * n_fourier.
    """

    def __init__(self, in_dim: int, n_fourier: int, scale: float) -> None:
        super().__init__()
        B = torch.randn(in_dim, n_fourier) * scale
        self.register_buffer("B", B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., in_dim]
        proj = x @ self.B           # [..., n_fourier]
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)


class SWENet(nn.Module):
    """
    PINN network for 2D Shallow Water Equations.

    Input normalisation is handled inside the network to keep
    the training loop clean. Call set_normalisation() after mesh
    coordinates are known.
    """

    def __init__(self, cfg: NetworkConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # Fourier embedding: 3 input dims → 2*n_fourier
        self.embed = FourierEmbedding(3, cfg.n_fourier, cfg.fourier_scale)
        embed_dim = 2 * cfg.n_fourier

        act_map = {"tanh": nn.Tanh, "gelu": nn.GELU, "silu": nn.SiLU}
        Act = act_map.get(cfg.activation, nn.Tanh)

        # Input projection: embed_dim → hidden_dim
        self.input_layer = nn.Sequential(
            nn.Linear(embed_dim, cfg.hidden_dim), Act(),
        )

        # Hidden layers (grouped into residual blocks of 2 if use_residual)
        self.use_residual = cfg.use_residual
        n_hidden = cfg.n_layers - 1  # first layer is input_layer
        self.blocks = nn.ModuleList()
        for _ in range(n_hidden):
            self.blocks.append(nn.Sequential(
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim), Act(),
            ))

        self.output_layer = nn.Linear(cfg.hidden_dim, cfg.output_dim)

        # Normalisation buffers (set by trainer)
        self.register_buffer("x_mean", torch.zeros(3))
        self.register_buffer("x_std", torch.ones(3))

        self._init_weights()
        res_str = " (residual)" if cfg.use_residual else ""
        logger.info(
            f"SWENet: {cfg.n_fourier} Fourier features, "
            f"{cfg.n_layers} hidden layers × {cfg.hidden_dim}{res_str}"
        )

    def set_normalisation(
        self,
        x_mean: torch.Tensor,
        x_std: torch.Tensor,
    ) -> None:
        """Set input normalisation statistics (call once before training)."""
        self.x_mean.copy_(x_mean)
        self.x_std.copy_(x_std)

    def forward(self, xyt: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        xyt : [..., 3]   columns are (x, y, t)

        Returns
        -------
        Q   : [..., 3]   conserved vars in perturbation form [xi, hu, hv]
                         xi = h - h_still (can be negative).
                         Positivity of h = xi + h_still is enforced in the
                         FVM residual, not here.
        """
        x_norm = (xyt - self.x_mean) / (self.x_std + 1e-8)
        z = self.embed(x_norm)
        z = self.input_layer(z)
        for i, block in enumerate(self.blocks):
            if self.use_residual and i % 2 == 1:
                z = block(z) + z  # skip connection every 2 layers
            else:
                z = block(z)
        return self.output_layer(z)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)


class SineLayer(nn.Module):
    """Sine-activated linear layer, SIREN style (Sitzmann et al., 2020).

    First layer uses uniform init in [-1/fan_in, 1/fan_in] and a large w0
    (default 30) so the pre-activation spans many periods across the
    (normalized) domain. Hidden layers use [-sqrt(6/fan_in)/w0, +sqrt(6/fan_in)/w0]
    which preserves output variance through successive sine nonlinearities.
    """

    def __init__(self, in_dim: int, out_dim: int, w0: float = 1.0,
                 is_first: bool = False) -> None:
        super().__init__()
        self.w0 = w0
        self.linear = nn.Linear(in_dim, out_dim)
        with torch.no_grad():
            if is_first:
                bound = 1.0 / in_dim
            else:
                bound = (6.0 / in_dim) ** 0.5 / w0
            self.linear.weight.uniform_(-bound, bound)
            self.linear.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w0 * self.linear(x))


class SirenSWENet(nn.Module):
    """SIREN variant of SWENet.

    Sine activations replace tanh; no Fourier feature embedding (the first
    sine layer with w0>>1 is itself a learned frequency encoding). Drop-in
    for SWENet: same forward signature, same set_normalisation contract.
    """

    def __init__(self, cfg: NetworkConfig, w0_first: float = 30.0,
                 w0_hidden: float = 1.0) -> None:
        super().__init__()
        self.cfg = cfg
        self.w0_first = w0_first
        self.w0_hidden = w0_hidden
        self.use_residual = cfg.use_residual

        self.input_layer = SineLayer(3, cfg.hidden_dim,
                                     w0=w0_first, is_first=True)
        n_hidden = cfg.n_layers - 1
        self.blocks = nn.ModuleList([
            SineLayer(cfg.hidden_dim, cfg.hidden_dim,
                      w0=w0_hidden, is_first=False)
            for _ in range(n_hidden)
        ])

        self.output_layer = nn.Linear(cfg.hidden_dim, cfg.output_dim)
        with torch.no_grad():
            bound = (6.0 / cfg.hidden_dim) ** 0.5 / w0_hidden
            self.output_layer.weight.uniform_(-bound, bound)
            self.output_layer.bias.zero_()

        self.register_buffer("x_mean", torch.zeros(3))
        self.register_buffer("x_std", torch.ones(3))

        res_str = " (residual)" if cfg.use_residual else ""
        logger.info(
            f"SirenSWENet: {cfg.n_layers} layers × {cfg.hidden_dim}, "
            f"w0_first={w0_first}, w0_hidden={w0_hidden}{res_str}"
        )

    def set_normalisation(self, x_mean: torch.Tensor,
                          x_std: torch.Tensor) -> None:
        self.x_mean.copy_(x_mean)
        self.x_std.copy_(x_std)

    def forward(self, xyt: torch.Tensor) -> torch.Tensor:
        z = (xyt - self.x_mean) / (self.x_std + 1e-8)
        z = self.input_layer(z)
        for i, block in enumerate(self.blocks):
            if self.use_residual and i % 2 == 1:
                z = block(z) + z
            else:
                z = block(z)
        return self.output_layer(z)
