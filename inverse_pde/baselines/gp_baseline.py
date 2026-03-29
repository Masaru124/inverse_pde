from __future__ import annotations

from dataclasses import dataclass

import gpytorch
import numpy as np
import torch
from sklearn.decomposition import PCA

from baselines.mlp_baseline import build_padded_feature


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


@dataclass
class GPBaseline:
    max_m: int = 100
    grid_size: int = 32
    n_components: int = 16
    training_iters: int = 40

    def __post_init__(self):
        self.pca = PCA(n_components=self.n_components)
        self.models: list[ExactGPModel] = []
        self.likelihoods: list[gpytorch.likelihoods.GaussianLikelihood] = []

    def _extract_numpy(self, loader):
        feats = []
        targets = []
        for batch in loader:
            x = build_padded_feature(batch["obs_coords"], batch["obs_values"], max_m=self.max_m)
            y = batch["k_grid"].reshape(batch["k_grid"].shape[0], -1)
            feats.append(x.cpu())
            targets.append(y.cpu())

        x_all = torch.cat(feats, dim=0).numpy()
        y_all = torch.cat(targets, dim=0).numpy()
        return x_all, y_all

    def fit(self, train_loader, max_samples: int = 2000) -> None:
        x_np, y_np = self._extract_numpy(train_loader)
        x_np = x_np[:max_samples]
        y_np = y_np[:max_samples]

        y_latent = self.pca.fit_transform(y_np)

        train_x = torch.tensor(x_np, dtype=torch.float32)
        train_y_latent = torch.tensor(y_latent, dtype=torch.float32)

        self.models = []
        self.likelihoods = []

        for c in range(self.n_components):
            y_c = train_y_latent[:, c]
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            model = ExactGPModel(train_x, y_c, likelihood)

            model.train()
            likelihood.train()

            optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

            for _ in range(self.training_iters):
                optimizer.zero_grad()
                output = model(train_x)
                loss = -mll(output, y_c)
                loss.backward()
                optimizer.step()

            self.models.append(model)
            self.likelihoods.append(likelihood)

    @torch.no_grad()
    def predict(self, obs_coords: torch.Tensor, obs_values: torch.Tensor) -> torch.Tensor:
        x = build_padded_feature(obs_coords, obs_values, max_m=self.max_m).cpu()

        latents = []
        for model, likelihood in zip(self.models, self.likelihoods):
            model.eval()
            likelihood.eval()
            preds = likelihood(model(x)).mean
            latents.append(preds.unsqueeze(-1))

        latent_tensor = torch.cat(latents, dim=-1).numpy()
        recon = self.pca.inverse_transform(latent_tensor)
        recon_t = torch.tensor(recon, dtype=torch.float32)
        return recon_t.view(obs_coords.shape[0], self.grid_size, self.grid_size)
