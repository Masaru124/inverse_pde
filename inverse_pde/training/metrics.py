from __future__ import annotations

import math

import torch


def _ensure_target_shape(target: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    if target.shape == pred.shape:
        return target
    if target.dim() == pred.dim() - 1:
        return target.unsqueeze(-1)
    raise ValueError(f"Target shape {target.shape} is incompatible with prediction shape {pred.shape}")


def rmse(mu: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target = _ensure_target_shape(target, mu)
    return torch.sqrt(torch.mean((mu - target) ** 2))


def coverage(mu: torch.Tensor, sigma: torch.Tensor, target: torch.Tensor, level: float = 0.90) -> torch.Tensor:
    target = _ensure_target_shape(target, mu)
    alpha = 1.0 - level
    z = torch.tensor(
        torch.distributions.Normal(0.0, 1.0).icdf(torch.tensor(1.0 - alpha / 2.0)).item(),
        device=mu.device,
    )
    lower = mu - z * sigma
    upper = mu + z * sigma
    covered = (target >= lower) & (target <= upper)
    return covered.float().mean()


def ece_regression(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    target: torch.Tensor,
    n_bins: int = 15,
    level: float = 0.90,
) -> torch.Tensor:
    target = _ensure_target_shape(target, mu)
    # Confidence proxy: inverse uncertainty, normalized per batch.
    conf = 1.0 / (sigma + 1e-6)
    conf = (conf - conf.min()) / (conf.max() - conf.min() + 1e-6)

    alpha = 1.0 - level
    z = torch.distributions.Normal(0.0, 1.0).icdf(torch.tensor(1.0 - alpha / 2.0, device=mu.device))
    in_interval = ((target >= mu - z * sigma) & (target <= mu + z * sigma)).float()

    conf_flat = conf.reshape(-1)
    acc_flat = in_interval.reshape(-1)

    bin_edges = torch.linspace(0.0, 1.0, n_bins + 1, device=mu.device)
    ece = torch.tensor(0.0, device=mu.device)

    n = conf_flat.numel()
    for i in range(n_bins):
        lo = bin_edges[i]
        hi = bin_edges[i + 1]
        mask = (conf_flat >= lo) & (conf_flat < hi if i < n_bins - 1 else conf_flat <= hi)
        if mask.any():
            bin_conf = conf_flat[mask].mean()
            bin_acc = acc_flat[mask].mean()
            ece += (mask.float().mean()) * torch.abs(bin_acc - bin_conf)

    return ece


def batch_metrics(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    target: torch.Tensor,
    ece_bins: int = 15,
    coverage_level: float = 0.90,
) -> dict:
    target = _ensure_target_shape(target, mu)
    return {
        "rmse": rmse(mu=mu, target=target).item(),
        "ece": ece_regression(mu=mu, sigma=sigma, target=target, n_bins=ece_bins, level=coverage_level).item(),
        "coverage": coverage(mu=mu, sigma=sigma, target=target, level=coverage_level).item(),
    }
