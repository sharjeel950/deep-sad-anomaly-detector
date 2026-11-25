"""Network architectures used by Deep SAD.

Design notes (following Ruff et al., 2018 and 2020)
---------------------------------------------------
* Every linear layer is **bias-free** and batch-norm has **no affine
  parameters**.  With biases the network could trivially satisfy the
  one-class objective by mapping every input to the constant ``c``
  ("hypersphere collapse").  Removing them makes that solution unreachable.
* The encoder ``phi(.; W)`` is pretrained as the encoder half of an
  autoencoder; the Deep SAD objective then fine-tunes it.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


def _mlp(in_dim: int, hidden: Sequence[int], out_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = in_dim
    for h in hidden:
        layers += [
            nn.Linear(d, h, bias=False),
            nn.BatchNorm1d(h, eps=1e-4, affine=False),
            nn.LeakyReLU(0.1),
        ]
        d = h
    layers.append(nn.Linear(d, out_dim, bias=False))
    return nn.Sequential(*layers)


class Encoder(nn.Module):
    """phi(x; W): R^d -> R^rep_dim."""

    def __init__(self, in_dim: int, hidden: Sequence[int] = (128, 64), rep_dim: int = 32):
        super().__init__()
        self.in_dim = in_dim
        self.hidden = tuple(hidden)
        self.rep_dim = rep_dim
        self.net = _mlp(in_dim, self.hidden, rep_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.net(x)


class Decoder(nn.Module):
    """Mirror of the encoder, used only for autoencoder pretraining."""

    def __init__(self, out_dim: int, hidden: Sequence[int], rep_dim: int):
        super().__init__()
        self.net = _mlp(rep_dim, tuple(reversed(tuple(hidden))), out_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class AutoEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: Sequence[int] = (128, 64), rep_dim: int = 32):
        super().__init__()
        self.encoder = Encoder(in_dim, hidden, rep_dim)
        self.decoder = Decoder(in_dim, hidden, rep_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
