from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Dict, List, Literal, Tuple

import torch
import torch.nn.functional as F

PDEFamily = Literal["diffusion", "reaction_diffusion"]
KType = Literal["gp", "inclusion", "checkerboard", "mixed"]
NoiseType = Literal["gaussian", "correlated", "outlier", "mixed"]


def _build_grid(grid_size: int, device: torch.device) -> torch.Tensor:
    x = torch.linspace(0.0, 1.0, grid_size, device=device)
    y = torch.linspace(0.0, 1.0, grid_size, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    coords = torch.stack([xx, yy], dim=-1).reshape(-1, 2)
    return coords


def _build_time_grid(n_times: int, device: torch.device) -> torch.Tensor:
    return torch.linspace(0.0, 1.0, n_times, device=device)


def _pairwise_distance(coords: torch.Tensor) -> torch.Tensor:
    return torch.cdist(coords, coords, p=2)


def _sample_rff_field(
    coords: torch.Tensor,
    lengthscale: float,
    variance: float,
    n_features: int = 96,
) -> torch.Tensor:
    dim = coords.shape[-1]
    omega = torch.randn(n_features, dim, device=coords.device, dtype=coords.dtype) / max(lengthscale, 1e-6)
    phase = 2.0 * math.pi * torch.rand(n_features, device=coords.device, dtype=coords.dtype)
    coeff = torch.randn(n_features, device=coords.device, dtype=coords.dtype)
    arg = coords @ omega.transpose(0, 1) + phase.unsqueeze(0)
    scale = math.sqrt(2.0 * variance / max(1, n_features))
    return scale * torch.cos(arg) @ coeff


def _matern_kernel(distance: torch.Tensor, nu: float, lengthscale: torch.Tensor, variance: torch.Tensor) -> torch.Tensor:
    eps = 1e-12
    r = distance / (lengthscale + eps)
    if nu == 0.5:
        base = torch.exp(-r)
    elif nu == 1.5:
        sqrt3 = math.sqrt(3.0)
        base = (1.0 + sqrt3 * r) * torch.exp(-sqrt3 * r)
    elif nu == 2.5:
        sqrt5 = math.sqrt(5.0)
        base = (1.0 + sqrt5 * r + (5.0 / 3.0) * r**2) * torch.exp(-sqrt5 * r)
    else:
        raise ValueError(f"Unsupported Matérn nu={nu}. Expected one of 0.5, 1.5, 2.5")
    return variance * base


def _sample_gp_field(
    coords: torch.Tensor,
    nu: float,
    lengthscale: torch.Tensor,
    variance: torch.Tensor,
    jitter: float = 1e-6,
) -> torch.Tensor:
    n = coords.shape[0]
    distance = _pairwise_distance(coords)
    cov = _matern_kernel(distance, nu=nu, lengthscale=lengthscale, variance=variance)
    cov = cov + jitter * torch.eye(n, device=coords.device, dtype=coords.dtype)
    chol = torch.linalg.cholesky(cov)
    eps = torch.randn(n, device=coords.device, dtype=coords.dtype)
    return chol @ eps


def _compute_forcing_autograd(coords: torch.Tensor, u_flat: torch.Tensor, k_flat: torch.Tensor) -> torch.Tensor:
    grad_u = torch.autograd.grad(u_flat.sum(), coords, create_graph=True, retain_graph=True)[0]
    flux = k_flat.unsqueeze(-1) * grad_u
    div_x = torch.autograd.grad(flux[:, 0].sum(), coords, create_graph=True, retain_graph=True)[0][:, 0]
    div_y = torch.autograd.grad(flux[:, 1].sum(), coords, create_graph=True, retain_graph=True)[0][:, 1]
    return -(div_x + div_y)


def _sample_inclusion_k_grid(grid_size: int, device: torch.device) -> torch.Tensor:
    coords = _build_grid(grid_size=grid_size, device=device)
    x = coords[:, 0]
    y = coords[:, 1]
    bg_val = float(torch.empty(1, device=device).uniform_(0.3, 0.8).item())
    k = torch.full((coords.shape[0],), bg_val, device=device)

    n_inc = int(torch.randint(1, 4, (1,), device=device).item())
    for _ in range(n_inc):
        cx, cy = torch.rand(2, device=device)
        r = float(torch.empty(1, device=device).uniform_(0.05, 0.20).item())
        val = float(torch.empty(1, device=device).uniform_(1.0, 2.0).item())
        mask = ((x - cx) ** 2 + (y - cy) ** 2) <= (r**2)
        k[mask] = val
    return (F.softplus(k) + 1e-6).reshape(grid_size, grid_size)


def _sample_checkerboard_k_grid(grid_size: int, device: torch.device) -> torch.Tensor:
    x = torch.arange(grid_size, device=device)
    y = torch.arange(grid_size, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    tile = int(torch.randint(2, 6, (1,), device=device).item())
    pattern = ((xx // tile + yy // tile) % 2).float()
    return 0.25 + 1.2 * pattern


def _sample_k_from_type(
    k_type: KType,
    grid_size: int,
    coords: torch.Tensor,
    nu_choices: Tuple[float, ...],
    device: torch.device,
) -> Tuple[torch.Tensor, str]:
    if k_type == "mixed":
        # 70% GP + 30% inclusion; piecewise is intentionally removed.
        p = float(torch.rand(1, device=device).item())
        k_type = "gp" if p < 0.7 else "inclusion"

    if k_type == "gp":
        nu_k = float(nu_choices[torch.randint(0, len(nu_choices), (1,)).item()])
        ls_k = torch.empty(1, device=device, dtype=torch.float64).uniform_(0.8, 1.6)
        var_k = torch.empty(1, device=device, dtype=torch.float64).uniform_(0.3, 0.8)
        k_raw = _sample_gp_field(coords=coords, nu=nu_k, lengthscale=ls_k, variance=var_k)
        k_flat = F.softplus(k_raw) + 1e-6
        return k_flat.reshape(grid_size, grid_size), "gp"

    if k_type == "inclusion":
        return _sample_inclusion_k_grid(grid_size=grid_size, device=device), "inclusion"
    if k_type == "checkerboard":
        return _sample_checkerboard_k_grid(grid_size=grid_size, device=device), "checkerboard"

    raise ValueError(f"Unsupported k_type={k_type}")


def _apply_observation_noise(
    obs_values_clean: torch.Tensor,
    obs_coords: torch.Tensor,
    noise_min: float,
    noise_max: float,
    noise_type: NoiseType,
) -> Tuple[torch.Tensor, float, str]:
    device = obs_values_clean.device
    if noise_type == "mixed":
        options = ["gaussian", "correlated", "outlier"]
        noise_type = options[int(torch.randint(0, len(options), (1,), device=device).item())]

    sigma = float(torch.empty(1, device=device).uniform_(noise_min, noise_max).item())

    if noise_type == "gaussian":
        noise = sigma * torch.randn_like(obs_values_clean)
        return (obs_values_clean + noise).to(obs_values_clean.dtype), sigma, "gaussian"

    if noise_type == "correlated":
        d = torch.cdist(obs_coords, obs_coords)
        cov = torch.exp(-d / 0.2) + 1e-5 * torch.eye(obs_coords.shape[0], device=device)
        chol = torch.linalg.cholesky(cov)
        eps = chol @ torch.randn(obs_coords.shape[0], device=device, dtype=obs_coords.dtype)
        eps = eps / (eps.std() + 1e-6)
        return (obs_values_clean + sigma * eps).to(obs_values_clean.dtype), sigma, "correlated"

    if noise_type == "outlier":
        noise = sigma * torch.randn_like(obs_values_clean)
        out = obs_values_clean + noise
        n_outliers = max(1, int(0.05 * obs_values_clean.numel()))
        idx = torch.randperm(obs_values_clean.numel(), device=device)[:n_outliers]
        out[idx] = torch.empty(n_outliers, device=device, dtype=out.dtype).uniform_(-2.0, 2.0)
        return out.to(obs_values_clean.dtype), sigma, "outlier"

    raise ValueError(f"Unsupported noise_type={noise_type}")


def generate_instance(
    grid_size: int = 32,
    m_min: int = 20,
    m_max: int = 100,
    noise_min: float = 1e-3,
    noise_max: float = 5e-2,
    nu_choices: Tuple[float, ...] = (0.5, 1.5, 2.5),
    pde_family: PDEFamily = "diffusion",
    k_type: KType = "gp",
    noise_type: NoiseType = "gaussian",
    n_time_snapshots: int = 3,
    device: torch.device | None = None,
) -> Dict[str, torch.Tensor | str | float]:
    if pde_family == "reaction_diffusion":
        return generate_reaction_diffusion_instance(
            grid_size=grid_size,
            m_min=m_min,
            m_max=m_max,
            noise_min=noise_min,
            noise_max=noise_max,
            n_time_snapshots=n_time_snapshots,
            noise_type=noise_type,
            device=device,
        )

    device = device or torch.device("cpu")
    coords = _build_grid(grid_size=grid_size, device=device).to(torch.float64).requires_grad_(True)

    nu_u = float(nu_choices[torch.randint(0, len(nu_choices), (1,)).item()])
    # Smoother priors improve consistency between autograd and finite-difference checks.
    ls_u = torch.empty(1, device=device, dtype=torch.float64).uniform_(0.8, 1.6)
    var_u = torch.empty(1, device=device, dtype=torch.float64).uniform_(0.3, 0.8)

    u_flat = _sample_gp_field(coords=coords, nu=nu_u, lengthscale=ls_u, variance=var_u)
    k_grid64, resolved_k_type = _sample_k_from_type(
        k_type=k_type,
        grid_size=grid_size,
        coords=coords,
        nu_choices=nu_choices,
        device=device,
    )
    k_flat = k_grid64.reshape(-1)

    if resolved_k_type == "gp":
        f_flat = _compute_forcing_autograd(coords=coords, u_flat=u_flat, k_flat=k_flat)
    else:
        u_grid_tmp = u_flat.detach().reshape(grid_size, grid_size).to(torch.float32)
        f_grid_tmp = _finite_diff_forcing(u_grid=u_grid_tmp, k_grid=k_grid64.to(torch.float32))
        f_flat = f_grid_tmp.to(torch.float64).reshape(-1)

    m_obs = int(torch.randint(low=m_min, high=m_max + 1, size=(1,)).item())
    obs_indices = torch.randperm(coords.shape[0], device=device)[:m_obs]
    obs_coords = coords.detach()[obs_indices]
    obs_values_clean = u_flat.detach()[obs_indices]

    obs_values, sigma, resolved_noise_type = _apply_observation_noise(
        obs_values_clean=obs_values_clean,
        obs_coords=obs_coords,
        noise_min=noise_min,
        noise_max=noise_max,
        noise_type=noise_type,
    )

    k_grid = k_flat.detach().reshape(grid_size, grid_size).to(torch.float32)
    f_grid = f_flat.detach().reshape(grid_size, grid_size).to(torch.float32)
    u_grid = u_flat.detach().reshape(grid_size, grid_size).to(torch.float32)

    return {
        "obs_coords": obs_coords.to(torch.float32).cpu(),
        "obs_times": torch.zeros(obs_coords.shape[0], dtype=torch.float32).cpu(),
        "obs_values": obs_values.to(torch.float32).cpu(),
        "k_grid": k_grid.cpu(),
        "r_grid": torch.zeros_like(k_grid).cpu(),
        "f_grid": f_grid.cpu(),
        "u_grid": u_grid.cpu(),
        "pde_family": "diffusion",
        "k_type": resolved_k_type,
        "noise_type": resolved_noise_type,
        "noise_sigma": float(sigma),
    }


def generate_reaction_diffusion_instance(
    grid_size: int = 32,
    m_min: int = 20,
    m_max: int = 100,
    noise_min: float = 1e-3,
    noise_max: float = 5e-2,
    n_time_snapshots: int = 3,
    noise_type: NoiseType = "gaussian",
    device: torch.device | None = None,
) -> Dict[str, torch.Tensor | str | float]:
    device = device or torch.device("cpu")
    n_time_snapshots = max(2, int(n_time_snapshots))

    spatial_coords = _build_grid(grid_size=grid_size, device=device).to(torch.float64)
    times = _build_time_grid(n_time_snapshots, device=device).to(torch.float64)

    spatial_rep = spatial_coords.unsqueeze(1).expand(-1, n_time_snapshots, -1)
    time_rep = times.unsqueeze(0).expand(spatial_coords.shape[0], -1).unsqueeze(-1)
    coords_3d = torch.cat([spatial_rep, time_rep], dim=-1).reshape(-1, 3).requires_grad_(True)

    u_field = _sample_rff_field(
        coords=coords_3d,
        lengthscale=float(torch.empty(1, device=device).uniform_(0.6, 1.2).item()),
        variance=float(torch.empty(1, device=device).uniform_(0.2, 0.6).item()),
        n_features=128,
    )

    k_raw = _sample_rff_field(
        coords=coords_3d[:, :2],
        lengthscale=float(torch.empty(1, device=device).uniform_(0.7, 1.4).item()),
        variance=float(torch.empty(1, device=device).uniform_(0.2, 0.8).item()),
        n_features=96,
    )
    k_field = F.softplus(k_raw) + 0.05

    grad_u = torch.autograd.grad(u_field.sum(), coords_3d, create_graph=True, retain_graph=True)[0]
    ux = grad_u[:, 0]
    uy = grad_u[:, 1]
    ut = grad_u[:, 2]

    flux_x = k_field * ux
    flux_y = k_field * uy
    div_x = torch.autograd.grad(flux_x.sum(), coords_3d, create_graph=True, retain_graph=True)[0][:, 0]
    div_y = torch.autograd.grad(flux_y.sum(), coords_3d, create_graph=True, retain_graph=True)[0][:, 1]
    diffusion_term = div_x + div_y

    denom = torch.where(torch.abs(u_field) < 0.1, 0.1 * torch.sign(u_field + 1e-6), u_field)
    r_field = torch.clamp((ut - diffusion_term) / denom, min=-2.0, max=2.0)

    u_3d = u_field.reshape(grid_size * grid_size, n_time_snapshots)
    k_3d = k_field.reshape(grid_size * grid_size, n_time_snapshots)
    r_3d = r_field.reshape(grid_size * grid_size, n_time_snapshots)
    ut_3d = ut.reshape(grid_size * grid_size, n_time_snapshots)

    # Coefficients are time-invariant; use first slice.
    k_grid = k_3d[:, 0].detach().reshape(grid_size, grid_size).to(torch.float32)
    r_grid = r_3d[:, 0].detach().reshape(grid_size, grid_size).to(torch.float32)

    obs_coords_list: List[torch.Tensor] = []
    obs_times_list: List[torch.Tensor] = []
    obs_values_clean_list: List[torch.Tensor] = []

    for t_idx in range(n_time_snapshots):
        m_obs = int(torch.randint(low=m_min, high=m_max + 1, size=(1,), device=device).item())
        obs_indices = torch.randperm(spatial_coords.shape[0], device=device)[:m_obs]
        obs_coords_t = spatial_coords.detach()[obs_indices].to(torch.float32)
        obs_values_t = u_3d[obs_indices, t_idx].detach().to(torch.float32)
        obs_times_t = torch.full((m_obs,), float(times[t_idx].item()), device=device, dtype=torch.float32)

        obs_coords_list.append(obs_coords_t)
        obs_times_list.append(obs_times_t)
        obs_values_clean_list.append(obs_values_t)

    obs_coords = torch.cat(obs_coords_list, dim=0)
    obs_times = torch.cat(obs_times_list, dim=0)
    obs_values_clean = torch.cat(obs_values_clean_list, dim=0)

    obs_values, sigma, resolved_noise_type = _apply_observation_noise(
        obs_values_clean=obs_values_clean,
        obs_coords=obs_coords.to(torch.float64),
        noise_min=noise_min,
        noise_max=noise_max,
        noise_type=noise_type,
    )

    t_mid = n_time_snapshots // 2
    u_grid = u_3d[:, t_mid].detach().reshape(grid_size, grid_size).to(torch.float32)
    f_grid = ut_3d[:, t_mid].detach().reshape(grid_size, grid_size).to(torch.float32)

    return {
        "obs_coords": obs_coords.cpu(),
        "obs_times": obs_times.cpu(),
        "obs_values": obs_values.cpu(),
        "k_grid": k_grid.cpu(),
        "r_grid": r_grid.cpu(),
        "f_grid": f_grid.cpu(),
        "u_grid": u_grid.cpu(),
        "pde_family": "reaction_diffusion",
        "k_type": "gp",
        "noise_type": resolved_noise_type,
        "noise_sigma": float(sigma),
    }


def _finite_diff_forcing(u_grid: torch.Tensor, k_grid: torch.Tensor) -> torch.Tensor:
    n = u_grid.shape[0]
    h = 1.0 / (n - 1)

    du_dy, du_dx = torch.gradient(u_grid, spacing=(h, h), edge_order=2)
    flux_x = k_grid * du_dx
    flux_y = k_grid * du_dy

    dfluxx_dy, dfluxx_dx = torch.gradient(flux_x, spacing=(h, h), edge_order=2)
    dfluxy_dy, dfluxy_dx = torch.gradient(flux_y, spacing=(h, h), edge_order=2)

    div_flux = dfluxx_dx + dfluxy_dy
    return -div_flux


def validate_generator(num_samples: int = 100, tol: float = 1e-3, grid_size: int = 32) -> None:
    max_err = 0.0
    for _ in range(num_samples):
        sample = generate_instance(grid_size=grid_size)
        k_grid = sample["k_grid"]
        f_grid = sample["f_grid"]
        u_grid = sample["u_grid"]
        f_fd = _finite_diff_forcing(u_grid=u_grid, k_grid=k_grid)
        interior_err = torch.abs(f_grid[1:-1, 1:-1] - f_fd[1:-1, 1:-1])
        err = torch.max(interior_err).item()
        max_err = max(max_err, err)

    if max_err >= tol:
        raise AssertionError(
            f"Generator validation failed: max error {max_err:.6e} exceeds tolerance {tol:.6e}"
        )


def generate_dataset_to_disk(
    out_dir: str | Path,
    n_instances: int = 50_000,
    shard_size: int = 1_000,
    grid_size: int = 32,
    m_min: int = 20,
    m_max: int = 100,
    noise_min: float = 1e-3,
    noise_max: float = 5e-2,
    nu_choices: Tuple[float, ...] = (0.5, 1.5, 2.5),
    pde_family: PDEFamily = "diffusion",
    k_type: KType = "gp",
    noise_type: NoiseType = "gaussian",
    n_time_snapshots: int = 3,
    device: torch.device | None = None,
) -> Dict[str, float]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    shard: List[dict] = []
    shard_id = 0
    start = time.time()

    for i in range(n_instances):
        sample = generate_instance(
            grid_size=grid_size,
            m_min=m_min,
            m_max=m_max,
            noise_min=noise_min,
            noise_max=noise_max,
            nu_choices=nu_choices,
            pde_family=pde_family,
            k_type=k_type,
            noise_type=noise_type,
            n_time_snapshots=n_time_snapshots,
            device=device,
        )
        shard.append(sample)

        if len(shard) >= shard_size or i == n_instances - 1:
            torch.save(shard, out_path / f"dataset_shard_{shard_id:05d}.pt")
            shard.clear()
            shard_id += 1

    elapsed = time.time() - start
    return {
        "n_instances": float(n_instances),
        "n_shards": float(shard_id),
        "elapsed_sec": elapsed,
    }
