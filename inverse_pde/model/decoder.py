from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import MCDropout


class ProbabilisticDecoder(nn.Module):
    def __init__(self, d_model: int = 128, dropout: float = 0.1, n_targets: int = 1):
        super().__init__()
        if n_targets not in (1, 2):
            raise ValueError("n_targets must be 1 (k only) or 2 (k and r)")
        self.n_targets = n_targets
        self.fc1 = nn.Linear(d_model, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 2 * n_targets)
        self.dropout = MCDropout(dropout)

    def forward(self, latent_grid: torch.Tensor, mc_dropout: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.fc1(latent_grid)
        x = F.gelu(x)
        x = self.dropout(x, force_mc=mc_dropout)
        x = self.fc2(x)
        x = F.gelu(x)
        x = self.fc3(x)

        x = x.view(*x.shape[:-1], self.n_targets, 2)
        mu = x[..., 0]
        log_sigma = x[..., 1]
        sigma = F.softplus(log_sigma) + 0.1
        if self.n_targets == 1:
            mu = mu.squeeze(-1)
            sigma = sigma.squeeze(-1)
            log_sigma = log_sigma.squeeze(-1)
        return mu, sigma, log_sigma
