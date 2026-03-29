from __future__ import annotations

import time

import torch
import torch.nn as nn
import torch.nn.functional as F


class KFieldMLP(nn.Module):
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.net(coords)).squeeze(-1) + 1e-6


class UFieldMLP(nn.Module):
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.net(coords).squeeze(-1)


def _build_grid(grid_size: int, device: torch.device) -> torch.Tensor:
    x = torch.linspace(0.0, 1.0, grid_size, device=device)
    y = torch.linspace(0.0, 1.0, grid_size, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack([xx, yy], dim=-1).reshape(-1, 2)


def _finite_diff_forcing(u_grid: torch.Tensor, k_grid: torch.Tensor) -> torch.Tensor:
    n = u_grid.shape[0]
    h = 1.0 / (n - 1)
    du_dy, du_dx = torch.gradient(u_grid, spacing=(h, h), edge_order=2)
    flux_x = k_grid * du_dx
    flux_y = k_grid * du_dy
    _, dfluxx_dx = torch.gradient(flux_x, spacing=(h, h), edge_order=2)
    dfluxy_dy, _ = torch.gradient(flux_y, spacing=(h, h), edge_order=2)
    return -(dfluxx_dx + dfluxy_dy)


def run_pinn_inversion(
    obs_coords: torch.Tensor,
    obs_values: torch.Tensor,
    f_grid: torch.Tensor,
    u_grid: torch.Tensor,
    grid_size: int = 32,
    steps: int = 1000,
    min_steps: int = 1,
    convergence_tol: float = 1e-6,
    convergence_patience: int = 200,
    lr: float = 1e-3,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, float, dict]:
    start = time.perf_counter()

    device = device or torch.device("cpu")
    obs_coords = obs_coords.to(device)
    obs_values = torch.nan_to_num(obs_values.to(device).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    f_grid = torch.nan_to_num(f_grid.to(device), nan=0.0, posinf=0.0, neginf=0.0)
    u_grid = torch.nan_to_num(u_grid.to(device), nan=0.0, posinf=0.0, neginf=0.0)

    coords = _build_grid(grid_size, device=device)
    k_model = KFieldMLP().to(device)
    optimizer = torch.optim.Adam(k_model.parameters(), lr=lr)

    # Precompute nearest grid index for each observation for a stable observation loss.
    gx = torch.round(obs_coords[:, 0] * (grid_size - 1)).long().clamp(0, grid_size - 1)
    gy = torch.round(obs_coords[:, 1] * (grid_size - 1)).long().clamp(0, grid_size - 1)
    obs_idx = gy * grid_size + gx

    best_loss = float("inf")
    best_pred = None
    non_finite_count = 0
    no_improve_steps = 0
    attempted_steps = 0
    update_steps = 0
    converged = False
    final_loss = float("inf")

    for step in range(1, steps + 1):
        attempted_steps = step
        pred_k_flat = torch.clamp(k_model(coords), min=1e-4, max=10.0)
        pred_k = pred_k_flat.reshape(grid_size, grid_size)
        pred_f = _finite_diff_forcing(u_grid=u_grid, k_grid=pred_k)

        residual_loss = ((pred_f - f_grid) ** 2).mean()
        smoothness = ((pred_k[1:, :] - pred_k[:-1, :]) ** 2).mean() + ((pred_k[:, 1:] - pred_k[:, :-1]) ** 2).mean()
        # Keep observation term as a soft stabilizer based on nearest u-grid consistency.
        obs_u_pred = u_grid.reshape(-1)[obs_idx]
        obs_loss = ((obs_u_pred - obs_values) ** 2).mean()

        loss = residual_loss + 1e-4 * smoothness + 0.1 * obs_loss

        if not torch.isfinite(loss):
            non_finite_count += 1
            # Damp the optimizer LR if loss explodes so optimization can recover.
            for group in optimizer.param_groups:
                group["lr"] = max(1e-6, float(group["lr"]) * 0.5)
            if non_finite_count > 10 and step >= min_steps:
                break
            continue

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = 0.0
        for p in k_model.parameters():
            if p.grad is not None:
                grad_norm += float(torch.sum(p.grad.detach() ** 2).item())
        grad_norm = grad_norm**0.5
        torch.nn.utils.clip_grad_norm_(k_model.parameters(), max_norm=1.0)
        optimizer.step()
        update_steps += 1

        loss_item = float(loss.item())
        final_loss = loss_item
        if loss_item < best_loss and torch.isfinite(pred_k).all():
            improvement = best_loss - loss_item if best_loss < float("inf") else float("inf")
            best_loss = loss_item
            best_pred = pred_k.detach().clone()
            if improvement <= convergence_tol:
                no_improve_steps += 1
            else:
                no_improve_steps = 0
        else:
            no_improve_steps += 1

        # Primary convergence criteria requested for comparability.
        if step >= min_steps and (loss_item < 1e-3 or grad_norm < 1e-4):
            converged = True
            break

        if step >= min_steps and no_improve_steps >= convergence_patience:
            converged = True
            break

    elapsed = time.perf_counter() - start
    if best_pred is None:
        pred_k = torch.ones(grid_size, grid_size, device=device)
    else:
        pred_k = best_pred

    pred_k = torch.nan_to_num(pred_k, nan=1.0, posinf=1.0, neginf=1.0).detach().cpu()
    if best_loss == float("inf"):
        best_loss = final_loss if torch.isfinite(torch.tensor(final_loss)) else float("nan")

    meta = {
        "attempted_steps": int(attempted_steps),
        "update_steps": int(update_steps),
        "configured_steps": int(steps),
        "converged": bool(converged),
        "best_loss": float(best_loss),
        "final_loss": float(final_loss),
        "non_finite_count": int(non_finite_count),
        "optimizer": "adam",
        "lr": float(lr),
    }
    return pred_k, elapsed, meta
