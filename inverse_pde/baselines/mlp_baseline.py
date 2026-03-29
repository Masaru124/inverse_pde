from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn


def build_padded_feature(obs_coords: torch.Tensor, obs_values: torch.Tensor, max_m: int = 100) -> torch.Tensor:
    bsz = obs_coords.shape[0]
    feat = torch.zeros(bsz, max_m, 3, device=obs_coords.device)
    n = min(max_m, obs_coords.shape[1])
    feat[:, :n, :2] = obs_coords[:, :n]
    feat[:, :n, 2] = obs_values[:, :n, 0]
    return feat.reshape(bsz, -1)


class MLPBaseline(nn.Module):
    def __init__(self, max_m: int = 100, grid_size: int = 32):
        super().__init__()
        self.max_m = max_m
        self.grid_size = grid_size
        in_dim = max_m * 3
        out_dim = grid_size * grid_size

        self.net = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, out_dim),
        )

    def forward(self, obs_coords: torch.Tensor, obs_values: torch.Tensor) -> torch.Tensor:
        x = build_padded_feature(obs_coords, obs_values, max_m=self.max_m)
        y = self.net(x)
        return y.view(obs_coords.shape[0], self.grid_size, self.grid_size)


def train_mlp_baseline(
    model: MLPBaseline,
    train_loader: Iterable,
    device: torch.device,
    epochs: int = 30,
    lr: float = 1e-3,
) -> MLPBaseline:
    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for _ in range(epochs):
        for batch in train_loader:
            obs_coords = batch["obs_coords"].to(device)
            obs_values = batch["obs_values"].to(device)
            target = batch["k_grid"].to(device)

            pred = model(obs_coords, obs_values)
            loss = loss_fn(pred, target)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    return model
