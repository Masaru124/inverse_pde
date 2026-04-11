from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.generator import generate_instance
from model.model import AmortizedInversePDEModel
from training.losses import total_loss


def _checkpoint_score(path: Path) -> tuple[int, float, str]:
    import re

    match = re.search(r"val_nll_(-?\d+(?:\.\d+)?)", path.name)
    if match:
        return (0, float(match.group(1)), path.name)
    return (1, float("inf"), path.name)


def discover_checkpoints() -> list[Path]:
    patterns = (
        "outputs*/checkpoints/*.pt",
        "outputs_recovery*/checkpoints/*.pt",
        "outputs_improved_v1/checkpoints/*.pt",
    )
    paths: list[Path] = []
    for pat in patterns:
        paths.extend(ROOT.glob(pat))
    return sorted({p.resolve() for p in paths}, key=_checkpoint_score)


def load_model(checkpoint_path: Path | None) -> tuple[AmortizedInversePDEModel, dict[str, Any], torch.device]:
    checkpoints = discover_checkpoints()
    if checkpoint_path is None:
        preferred = ROOT / "outputs_recovery_gpu_fast_stable" / "checkpoints" / "epoch_006_val_nll_-0.618860.pt"
        checkpoint_path = preferred if preferred.exists() else (checkpoints[0] if checkpoints else None)

    if checkpoint_path is None or not checkpoint_path.exists():
        raise FileNotFoundError("No checkpoint found. Ensure outputs*/checkpoints/*.pt is present.")

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg = ckpt.get("config", {})
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})
    target_fields = list(train_cfg.get("target_fields", ["k_grid"]))

    model = AmortizedInversePDEModel(
        grid_size=int(model_cfg.get("grid_size", data_cfg.get("grid_size", 32))),
        d_model=int(model_cfg.get("d_model", 96)),
        n_heads=int(model_cfg.get("n_heads", 4)),
        n_layers=int(model_cfg.get("n_layers", 3)),
        dropout=float(model_cfg.get("dropout", 0.15)),
        mc_samples=int(model_cfg.get("mc_samples", 25)),
        include_time=bool(model_cfg.get("include_time", False)),
        n_targets=int(model_cfg.get("n_targets", len(target_fields))),
    )
    model.load_state_dict(ckpt["model_state_dict"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, {"checkpoint": str(checkpoint_path), "config": cfg}, device


def make_boundary_coords(n_boundary: int, device: torch.device) -> torch.Tensor:
    theta = torch.linspace(0.0, 2.0 * np.pi, n_boundary + 1, device=device)[:-1]
    x = 0.5 + 0.5 * torch.cos(theta)
    y = 0.5 + 0.5 * torch.sin(theta)
    return torch.stack([x, y], dim=-1)


def sample_u_at_coords(u_grid: torch.Tensor, coords_01: torch.Tensor) -> torch.Tensor:
    # u_grid: (H, W), coords_01: (M, 2) in [0,1]
    grid = u_grid.unsqueeze(0).unsqueeze(0)
    sample_coords = coords_01 * 2.0 - 1.0
    sample_coords = sample_coords.view(1, -1, 1, 2)
    vals = F.grid_sample(grid, sample_coords, mode="bilinear", padding_mode="border", align_corners=True)
    return vals.view(-1)


@dataclass
class EvalSummary:
    rmse: float
    mae: float
    coverage_90: float


def evaluate_boundary(model: AmortizedInversePDEModel, device: torch.device, n_cases: int, n_boundary: int, seed: int = 0) -> EvalSummary:
    rmses: list[float] = []
    maes: list[float] = []
    coverages: list[float] = []

    model.eval()
    for i in range(n_cases):
        torch.manual_seed(seed + i)
        np.random.seed(seed + i)

        sample = generate_instance(
            grid_size=32,
            m_min=64,
            m_max=64,
            noise_min=0.01,
            noise_max=0.01,
            pde_family="diffusion",
            k_type="mixed",
            noise_type="gaussian",
            fast_gp=True,
        )

        k_true = sample["k_grid"].to(device)
        u_grid = sample["u_grid"].to(device)

        coords = make_boundary_coords(n_boundary=n_boundary, device=device)
        obs = sample_u_at_coords(u_grid=u_grid, coords_01=coords)
        obs = obs + 0.01 * torch.randn_like(obs)

        obs_coords = coords.unsqueeze(0)
        obs_values = obs.unsqueeze(0).unsqueeze(-1)
        obs_mask = torch.zeros((1, n_boundary), dtype=torch.bool, device=device)

        pred_mean, pred_epi, pred_ale = model.predict_with_uncertainty(
            obs_coords=obs_coords,
            obs_times=None,
            obs_values=obs_values,
            obs_key_padding_mask=obs_mask,
        )

        pred = pred_mean.squeeze(0)
        epi = pred_epi.squeeze(0)
        ale = pred_ale.squeeze(0)
        total_unc = epi + ale

        err = pred - k_true
        rmse = torch.sqrt(torch.mean(err**2)).item()
        mae = torch.mean(torch.abs(err)).item()
        cov = torch.mean((torch.abs(err) <= 1.64 * torch.clamp(total_unc, min=1e-6)).float()).item()

        rmses.append(rmse)
        maes.append(mae)
        coverages.append(cov)

    return EvalSummary(
        rmse=float(np.mean(rmses)),
        mae=float(np.mean(maes)),
        coverage_90=float(np.mean(coverages)),
    )


def fine_tune_boundary(
    model: AmortizedInversePDEModel,
    device: torch.device,
    steps: int,
    batch_size: int,
    n_boundary: int,
    lr: float,
) -> list[float]:
    # Option B: adapt only decoder to new observation geometry quickly.
    for p in model.parameters():
        p.requires_grad = False
    for p in model.decoder.parameters():
        p.requires_grad = True

    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    losses: list[float] = []
    model.train()

    for step in range(steps):
        batch_coords = []
        batch_values = []
        batch_targets = []

        for b in range(batch_size):
            seed = 1000 + step * batch_size + b
            torch.manual_seed(seed)
            np.random.seed(seed)

            sample = generate_instance(
                grid_size=32,
                m_min=64,
                m_max=64,
                noise_min=0.01,
                noise_max=0.01,
                pde_family="diffusion",
                k_type="mixed",
                noise_type="gaussian",
                fast_gp=True,
            )
            k_true = sample["k_grid"].to(device)
            u_grid = sample["u_grid"].to(device)

            coords = make_boundary_coords(n_boundary=n_boundary, device=device)
            obs = sample_u_at_coords(u_grid=u_grid, coords_01=coords)
            obs = obs + 0.01 * torch.randn_like(obs)

            batch_coords.append(coords)
            batch_values.append(obs.unsqueeze(-1))
            batch_targets.append(k_true)

        obs_coords = torch.stack(batch_coords, dim=0)
        obs_values = torch.stack(batch_values, dim=0)
        k_true = torch.stack(batch_targets, dim=0)
        obs_mask = torch.zeros((batch_size, n_boundary), dtype=torch.bool, device=device)

        mu, sigma, log_sigma = model.forward_with_logsigma(
            obs_coords=obs_coords,
            obs_times=None,
            obs_values=obs_values,
            obs_key_padding_mask=obs_mask,
            mc_dropout=True,
        )

        loss, _, _ = total_loss(k_true=k_true, mu=mu, sigma=sigma, log_sigma=log_sigma)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        losses.append(float(loss.item()))

    model.eval()
    return losses


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Option A and Option B boundary-observation validation.")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--eval-cases", type=int, default=20)
    parser.add_argument("--n-boundary", type=int, default=32)
    parser.add_argument("--finetune-steps", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--output", type=str, default="results_option_ab_boundary.json")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint).resolve() if args.checkpoint else None
    base_model, meta, device = load_model(checkpoint)

    # Option A: zero-shot under boundary-only observations.
    option_a = evaluate_boundary(
        model=base_model,
        device=device,
        n_cases=args.eval_cases,
        n_boundary=args.n_boundary,
        seed=42,
    )

    # Option B: quick boundary adaptation (decoder-only), then evaluate.
    adapted_model = copy.deepcopy(base_model)
    ft_losses = fine_tune_boundary(
        model=adapted_model,
        device=device,
        steps=args.finetune_steps,
        batch_size=args.batch_size,
        n_boundary=args.n_boundary,
        lr=args.lr,
    )
    option_b = evaluate_boundary(
        model=adapted_model,
        device=device,
        n_cases=args.eval_cases,
        n_boundary=args.n_boundary,
        seed=42,
    )

    payload = {
        "checkpoint": meta["checkpoint"],
        "device": str(device),
        "setup": {
            "eval_cases": args.eval_cases,
            "n_boundary": args.n_boundary,
            "finetune_steps": args.finetune_steps,
            "batch_size": args.batch_size,
            "lr": args.lr,
        },
        "option_a_zero_shot": asdict(option_a),
        "option_b_decoder_finetune": asdict(option_b),
        "delta_b_minus_a": {
            "rmse": option_b.rmse - option_a.rmse,
            "mae": option_b.mae - option_a.mae,
            "coverage_90": option_b.coverage_90 - option_a.coverage_90,
        },
        "finetune_loss": {
            "first": ft_losses[0] if ft_losses else None,
            "last": ft_losses[-1] if ft_losses else None,
            "min": min(ft_losses) if ft_losses else None,
        },
    }

    out_path = ROOT / args.output
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("=== Option A vs Option B (Boundary Observation Shift) ===")
    print(json.dumps(payload, indent=2))
    print(f"Saved report to: {out_path}")


if __name__ == "__main__":
    main()
