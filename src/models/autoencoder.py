from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class AutoencoderConfig:
    input_size: int = 6
    latent_size: int = 3
    hidden_size: int = 16


class FeedforwardAutoencoder(nn.Module):
    """
    Simple fully-connected autoencoder for per-timestep feature vectors.

    Input:
        x: (batch, input_size)

    Output:
        x_hat: (batch, input_size)
    """

    def __init__(self, cfg: AutoencoderConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.encoder = nn.Sequential(
            nn.Linear(cfg.input_size, cfg.hidden_size),
            nn.ReLU(),
            nn.Linear(cfg.hidden_size, cfg.latent_size),
        )

        self.decoder = nn.Sequential(
            nn.Linear(cfg.latent_size, cfg.hidden_size),
            nn.ReLU(),
            nn.Linear(cfg.hidden_size, cfg.input_size),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat
