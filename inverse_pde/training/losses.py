from __future__ import annotations

import torch


def _ensure_target_shape(target: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    if target.shape == pred.shape:
        return target
    if target.dim() == pred.dim() - 1:
        return target.unsqueeze(-1)
    raise ValueError(f"Target shape {target.shape} is incompatible with prediction shape {pred.shape}")


def gaussian_nll_loss(k_true: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    k_true = _ensure_target_shape(k_true, mu)
    sigma = torch.clamp(sigma, min=1e-6)
    nll = ((k_true - mu) ** 2) / (2.0 * sigma**2) + torch.log(sigma)
    return nll.mean()


def total_loss(
    k_true: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    log_sigma: torch.Tensor,
    lambda_reg: float = 0.01,
    sigma_floor: float = 0.1,
    sigma_reg_weight: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    k_true = _ensure_target_shape(k_true, mu)
    nll = gaussian_nll_loss(k_true=k_true, mu=mu, sigma=sigma)
    log_sigma_l2 = (log_sigma**2).mean()
    sigma_reg = torch.relu(sigma_floor - sigma).mean()
    loss = nll + lambda_reg * log_sigma_l2 + sigma_reg_weight * sigma_reg
    return loss, nll, sigma_reg
