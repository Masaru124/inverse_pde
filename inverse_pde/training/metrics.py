from __future__ import annotations

import math

import numpy as np
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
    # Quantile-based regression calibration error (Kuleshov-style):
    # for nominal quantiles q, compare q to empirical CDF proportion.
    sigma = torch.clamp(sigma, min=1e-6)
    z = (target - mu) / sigma
    normal = torch.distributions.Normal(
        torch.tensor(0.0, device=mu.device),
        torch.tensor(1.0, device=mu.device),
    )
    cdf_vals = normal.cdf(z).reshape(-1)

    n_quantiles = max(2, int(n_bins))
    quantiles = torch.linspace(
        1.0 / (n_quantiles + 1),
        n_quantiles / (n_quantiles + 1),
        n_quantiles,
        device=mu.device,
    )

    empirical = (cdf_vals.unsqueeze(0) <= quantiles.unsqueeze(1)).float().mean(dim=1)
    return torch.abs(empirical - quantiles).mean()


def ssim_2d(mu: torch.Tensor | None, target: torch.Tensor, sigma: float = 1.5) -> float:
    """
    Structural Similarity Index (SSIM) for 2D fields.
    
    Computes SSIM between prediction and target on 2D grids (e.g., k_pred[32,32]).
    Better than RMSE for evaluating spatial structure (edges, patterns, correlations).
    
    Args:
        mu: prediction, shape (H, W) or (batch, H, W)
        target: ground truth, same shape as mu
        sigma: Gaussian kernel std dev (larger = coarser structure)
        
    Returns:
        SSIM score in [-1, 1], typically 0.0-1.0 (higher = more similar)
    """
    # For now, return a placeholder or simplified MSE-based score
    # Full SSIM with Gaussian kernels requires scipy/skimage
    try:
        from scipy.ndimage import gaussian_filter
    except ImportError:
        # Fallback: return normalized negative RMSE as proxy
        mse = torch.mean((mu - target) ** 2)
        return float(1.0 - torch.sqrt(mse).item())
    
    # Convert to numpy for scipy operations
    if isinstance(mu, torch.Tensor):
        mu_np = mu.detach().cpu().numpy().astype(np.float64)
    else:
        mu_np = np.array(mu, dtype=np.float64)
    
    if isinstance(target, torch.Tensor):
        target_np = target.detach().cpu().numpy().astype(np.float64)
    else:
        target_np = np.array(target, dtype=np.float64)
    
    # Handle batch dimension
    if mu_np.ndim == 3:
        # (batch, H, W) -> take first element
        mu_np = mu_np[0]
        target_np = target_np[0]
    
    if mu_np.ndim != 2 or target_np.ndim != 2:
        raise ValueError(f"Expected 2D or 3D tensors, got {mu_np.ndim}D and {target_np.ndim}D")
    
    # Normalize
    mu_mean = mu_np.mean()
    target_mean = target_np.mean()
    mu_std = mu_np.std()
    target_std = target_np.std()
    
    if mu_std < 1e-10 or target_std < 1e-10:
        return 0.0
    
    mu_norm = (mu_np - mu_mean) / (mu_std + 1e-10)
    target_norm = (target_np - target_mean) / (target_std + 1e-10)
    
    # Cross-correlation with Gaussian weighting
    c1, c2 = 0.01, 0.03  # stability constants
    try:
        mu_smooth = gaussian_filter(mu_norm, sigma=sigma)
        target_smooth = gaussian_filter(target_norm, sigma=sigma)
        var_mu = gaussian_filter(mu_norm ** 2, sigma=sigma) - mu_smooth ** 2
        var_target = gaussian_filter(target_norm ** 2, sigma=sigma) - target_smooth ** 2
        cov = gaussian_filter(mu_norm * target_norm, sigma=sigma) - mu_smooth * target_smooth
        
        num = (2 * mu_smooth * target_smooth + c1) * (2 * cov + c2)
        denom = (mu_smooth ** 2 + target_smooth ** 2 + c1) * (var_mu + var_target + c2)
        
        ssim_map = num / denom
        return float(np.mean(ssim_map))
    except Exception:
        # Final fallback: correlation coefficient
        return float(np.corrcoef(mu_np.ravel(), target_np.ravel())[0, 1])


def batch_metrics(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    target: torch.Tensor,
    ece_bins: int = 15,
    coverage_level: float = 0.90,
    compute_ssim: bool = False,
) -> dict:
    target = _ensure_target_shape(target, mu)
    result = {
        "rmse": rmse(mu=mu, target=target).item(),
        "ece": ece_regression(mu=mu, sigma=sigma, target=target, n_bins=ece_bins, level=coverage_level).item(),
        "coverage": coverage(mu=mu, sigma=sigma, target=target, level=coverage_level).item(),
    }
    
    if compute_ssim:
        try:
            result["ssim"] = ssim_2d(mu, target)
        except Exception:
            result["ssim"] = None
    
    return result
