from __future__ import annotations

import torch
import torch.nn as nn

from .decoder import ProbabilisticDecoder
from .encoder import CrossAttentionEncoder


class AmortizedInversePDEModel(nn.Module):
    def __init__(
        self,
        grid_size: int = 32,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
        mc_samples: int = 50,
        include_time: bool = False,
        n_targets: int = 1,
    ):
        super().__init__()
        self.encoder = CrossAttentionEncoder(
            grid_size=grid_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            include_time=include_time,
        )
        self.decoder = ProbabilisticDecoder(
            d_model=d_model,
            dropout=dropout,
            n_targets=n_targets,
        )
        self.mc_samples = mc_samples
        self.n_targets = n_targets

    def forward_with_logsigma(
        self,
        obs_coords: torch.Tensor,
        obs_times: torch.Tensor | None,
        obs_values: torch.Tensor,
        obs_key_padding_mask: torch.Tensor | None = None,
        mc_dropout: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent = self.encoder(
            obs_coords=obs_coords,
            obs_times=obs_times,
            obs_values=obs_values,
            obs_key_padding_mask=obs_key_padding_mask,
            mc_dropout=mc_dropout,
        )
        mu_k, sigma_k, log_sigma_k = self.decoder(latent, mc_dropout=mc_dropout)
        return mu_k, sigma_k, log_sigma_k

    def forward(
        self,
        obs_coords: torch.Tensor,
        obs_times: torch.Tensor | None,
        obs_values: torch.Tensor,
        obs_key_padding_mask: torch.Tensor | None = None,
        mc_dropout: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mu_k, sigma_k, _ = self.forward_with_logsigma(
            obs_coords=obs_coords,
            obs_times=obs_times,
            obs_values=obs_values,
            obs_key_padding_mask=obs_key_padding_mask,
            mc_dropout=mc_dropout,
        )
        return mu_k, sigma_k

    @torch.no_grad()
    def predict_with_uncertainty(
        self,
        obs_coords: torch.Tensor,
        obs_times: torch.Tensor | None,
        obs_values: torch.Tensor,
        obs_key_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu_samples = []
        aleatoric_samples = []

        for _ in range(self.mc_samples):
            mu_k, sigma_k = self.forward(
                obs_coords=obs_coords,
                obs_times=obs_times,
                obs_values=obs_values,
                obs_key_padding_mask=obs_key_padding_mask,
                mc_dropout=True,
            )
            mu_samples.append(mu_k)
            aleatoric_samples.append(sigma_k)

        mu_stack = torch.stack(mu_samples, dim=0)
        sigma_stack = torch.stack(aleatoric_samples, dim=0)

        pred_mean = mu_stack.mean(dim=0)
        epistemic_std = mu_stack.std(dim=0)
        aleatoric_sigma = sigma_stack.mean(dim=0)
        return pred_mean, epistemic_std, aleatoric_sigma
