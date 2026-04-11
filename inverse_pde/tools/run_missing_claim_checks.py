from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.generator import _finite_diff_forcing, generate_instance
from model.model import AmortizedInversePDEModel
from training.metrics import batch_metrics
from utils import get_device, load_config, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run missing claim verification checks.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--sweep-samples", type=int, default=64)
    parser.add_argument("--reaction-samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ece-bins", type=int, default=15)
    parser.add_argument("--coverage-level", type=float, default=0.9)
    return parser.parse_args()


def _build_model(config: dict, device: torch.device) -> AmortizedInversePDEModel:
    target_fields = list(config["training"].get("target_fields", ["k_grid"]))
    model = AmortizedInversePDEModel(
        grid_size=int(config["data"]["grid_size"]),
        d_model=int(config["model"]["d_model"]),
        n_heads=int(config["model"]["n_heads"]),
        n_layers=int(config["model"]["n_layers"]),
        dropout=float(config["model"]["dropout"]),
        mc_samples=int(config["model"]["mc_samples"]),
        include_time=bool(config["model"].get("include_time", False)),
        n_targets=len(target_fields),
    ).to(device)
    return model


def _sample_layered(grid_size: int, device: torch.device) -> torch.Tensor:
    k = torch.ones(grid_size, grid_size, device=device, dtype=torch.float32) * 0.35
    n_layers = int(torch.randint(3, 6, (1,), device=device).item())
    boundaries = torch.linspace(0, grid_size, n_layers + 1, device=device).long()
    for i in range(n_layers):
        val = torch.empty(1, device=device).uniform_(0.25, 1.6).item()
        k[int(boundaries[i].item()) : int(boundaries[i + 1].item()), :] = val
    return k


def _sample_channelized(grid_size: int, device: torch.device) -> torch.Tensor:
    k = torch.ones(grid_size, grid_size, device=device, dtype=torch.float32) * 0.25
    center = int(torch.randint(grid_size // 4, 3 * grid_size // 4, (1,), device=device).item())
    width = int(torch.randint(2, max(3, grid_size // 6), (1,), device=device).item())
    y = torch.arange(grid_size, device=device)
    x = torch.arange(grid_size, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    wav = center + (0.15 * grid_size * torch.sin(2 * torch.pi * xx.float() / float(grid_size))).long()
    mask = (yy >= (wav - width)) & (yy <= (wav + width))
    k[mask] = torch.empty(1, device=device).uniform_(1.0, 2.0).item()
    return k


def _sample_realistic_gp(grid_size: int, device: torch.device) -> torch.Tensor:
    sample = generate_instance(
        grid_size=grid_size,
        m_min=20,
        m_max=100,
        noise_min=1e-3,
        noise_max=5e-2,
        nu_choices=(1.5, 2.5),
        pde_family="diffusion",
        k_type="gp",
        noise_type="gaussian",
        fast_gp=False,
        device=device,
    )
    return sample["k_grid"].to(device)


def _build_diffusion_sample_from_k(k_grid: torch.Tensor, device: torch.device) -> dict[str, torch.Tensor]:
    grid_size = k_grid.shape[0]
    # Reuse generator for consistent smooth u and then overwrite k/f to target chosen structure.
    base = generate_instance(
        grid_size=grid_size,
        m_min=20,
        m_max=100,
        noise_min=1e-3,
        noise_max=5e-2,
        pde_family="diffusion",
        k_type="gp",
        noise_type="gaussian",
        fast_gp=False,
        device=device,
    )
    u_grid = base["u_grid"].to(device)
    f_grid = _finite_diff_forcing(u_grid=u_grid, k_grid=k_grid)

    m_obs = int(torch.randint(20, 101, (1,), device=device).item())
    coords = torch.stack(torch.meshgrid(
        torch.linspace(0.0, 1.0, grid_size, device=device),
        torch.linspace(0.0, 1.0, grid_size, device=device),
        indexing="ij",
    ), dim=-1).reshape(-1, 2)
    idx = torch.randperm(coords.shape[0], device=device)[:m_obs]
    obs_coords = coords[idx].to(torch.float32)
    u_flat = u_grid.reshape(-1)
    obs_values = u_flat[idx] + 0.01 * torch.randn(m_obs, device=device)

    return {
        "obs_coords": obs_coords,
        "obs_times": torch.zeros(m_obs, device=device, dtype=torch.float32),
        "obs_values": obs_values.to(torch.float32),
        "k_grid": k_grid.to(torch.float32),
        "f_grid": f_grid.to(torch.float32),
        "u_grid": u_grid.to(torch.float32),
    }


def _eval_case(
    model: AmortizedInversePDEModel,
    samples: int,
    case_name: str,
    sample_builder,
    device: torch.device,
    use_mc_dropout: bool,
    ece_bins: int,
    coverage_level: float,
) -> dict[str, float]:
    rmse_vals: list[float] = []
    ece_vals: list[float] = []
    cov_vals: list[float] = []

    for _ in range(samples):
        s = sample_builder()
        obs_coords = s["obs_coords"].unsqueeze(0).to(device)
        obs_times = s["obs_times"].unsqueeze(0).unsqueeze(-1).to(device)
        obs_values = s["obs_values"].unsqueeze(0).unsqueeze(-1).to(device)
        target = s["k_grid"].unsqueeze(0).to(device)
        mask = torch.zeros(1, obs_coords.shape[1], dtype=torch.bool, device=device)

        if use_mc_dropout:
            mu, epistemic_std, aleatoric_sigma = model.predict_with_uncertainty(
                obs_coords=obs_coords,
                obs_times=obs_times,
                obs_values=obs_values,
                obs_key_padding_mask=mask,
            )
            sigma = torch.sqrt(torch.clamp(aleatoric_sigma**2 + epistemic_std**2, min=1e-12))
        else:
            mu, sigma = model(obs_coords, obs_times, obs_values, mask, mc_dropout=False)

        m = batch_metrics(mu=mu, sigma=sigma, target=target, ece_bins=ece_bins, coverage_level=coverage_level)
        rmse_vals.append(float(m["rmse"]))
        ece_vals.append(float(m["ece"]))
        cov_vals.append(float(m["coverage"]))

    n = max(1, len(rmse_vals))
    return {
        "case": case_name,
        "rmse": float(sum(rmse_vals) / n),
        "ece": float(sum(ece_vals) / n),
        "coverage": float(sum(cov_vals) / n),
        "n_samples": int(samples),
    }


def _run_reaction_probe(
    model: AmortizedInversePDEModel,
    samples: int,
    device: torch.device,
    use_mc_dropout: bool,
    ece_bins: int,
    coverage_level: float,
) -> dict[str, Any]:
    rmse_vals: list[float] = []
    ece_vals: list[float] = []
    cov_vals: list[float] = []

    for _ in range(samples):
        s = generate_instance(
            grid_size=32,
            m_min=20,
            m_max=100,
            noise_min=1e-3,
            noise_max=5e-2,
            pde_family="reaction_diffusion",
            n_time_snapshots=3,
            noise_type="gaussian",
            device=device,
        )

        obs_coords = s["obs_coords"].unsqueeze(0).to(device)
        obs_times = s["obs_times"].unsqueeze(0).unsqueeze(-1).to(device)
        obs_values = s["obs_values"].unsqueeze(0).unsqueeze(-1).to(device)
        target_k = s["k_grid"].unsqueeze(0).to(device)
        mask = torch.zeros(1, obs_coords.shape[1], dtype=torch.bool, device=device)

        if use_mc_dropout:
            mu, epistemic_std, aleatoric_sigma = model.predict_with_uncertainty(
                obs_coords=obs_coords,
                obs_times=obs_times,
                obs_values=obs_values,
                obs_key_padding_mask=mask,
            )
            sigma = torch.sqrt(torch.clamp(aleatoric_sigma**2 + epistemic_std**2, min=1e-12))
        else:
            mu, sigma = model(obs_coords, obs_times, obs_values, mask, mc_dropout=False)

        m = batch_metrics(mu=mu, sigma=sigma, target=target_k, ece_bins=ece_bins, coverage_level=coverage_level)
        rmse_vals.append(float(m["rmse"]))
        ece_vals.append(float(m["ece"]))
        cov_vals.append(float(m["coverage"]))

    n = max(1, len(rmse_vals))
    return {
        "status": "tested_with_diffusion_checkpoint",
        "note": "This is a probe using diffusion checkpoint on reaction-diffusion generated data (k only), not the paper's dedicated reaction model.",
        "k_rmse": float(sum(rmse_vals) / n),
        "k_ece": float(sum(ece_vals) / n),
        "k_coverage": float(sum(cov_vals) / n),
        "n_samples": int(samples),
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(args.seed, deterministic=bool(config.get("deterministic", False)))
    device = get_device()

    model = _build_model(config=config, device=device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    use_mc_dropout = bool(config.get("evaluation", {}).get("inference", {}).get("use_mc_dropout", True))

    zero_shot = {
        "layered": _eval_case(
            model,
            args.samples,
            "layered",
            lambda: _build_diffusion_sample_from_k(_sample_layered(32, device=device), device=device),
            device,
            use_mc_dropout,
            args.ece_bins,
            args.coverage_level,
        ),
        "channelized": _eval_case(
            model,
            args.samples,
            "channelized",
            lambda: _build_diffusion_sample_from_k(_sample_channelized(32, device=device), device=device),
            device,
            use_mc_dropout,
            args.ece_bins,
            args.coverage_level,
        ),
        "inclusion": _eval_case(
            model,
            args.samples,
            "inclusion",
            lambda: generate_instance(
                grid_size=32,
                m_min=20,
                m_max=100,
                noise_min=1e-3,
                noise_max=5e-2,
                pde_family="diffusion",
                k_type="inclusion",
                noise_type="gaussian",
                device=device,
            ),
            device,
            use_mc_dropout,
            args.ece_bins,
            args.coverage_level,
        ),
        "realistic_gp": _eval_case(
            model,
            args.samples,
            "realistic_gp",
            lambda: _build_diffusion_sample_from_k(_sample_realistic_gp(32, device=device), device=device),
            device,
            use_mc_dropout,
            args.ece_bins,
            args.coverage_level,
        ),
        "darcy": {
            "status": "blocked",
            "reason": "No local Darcy dataset artifact was found in workspace.",
        },
    }

    sweep = []
    sweep_seeds = [42, 43, 44, 45, 46]
    for s in sweep_seeds:
        set_seed(s, deterministic=False)
        met = _eval_case(
            model,
            args.sweep_samples,
            "realistic_gp",
            lambda: _build_diffusion_sample_from_k(_sample_realistic_gp(32, device=device), device=device),
            device,
            use_mc_dropout,
            args.ece_bins,
            args.coverage_level,
        )
        met["seed"] = s
        sweep.append(met)

    def _mean_std(key: str) -> dict[str, float]:
        vals = torch.tensor([float(x[key]) for x in sweep], dtype=torch.float32)
        return {
            "mean": float(vals.mean().item()),
            "std": float(vals.std(unbiased=False).item()),
        }

    sweep_summary = {
        "rmse": _mean_std("rmse"),
        "ece": _mean_std("ece"),
        "coverage": _mean_std("coverage"),
        "n_seeds": len(sweep),
        "samples_per_seed": int(args.sweep_samples),
        "seed_results": sweep,
    }

    reaction_probe = _run_reaction_probe(
        model=model,
        samples=args.reaction_samples,
        device=device,
        use_mc_dropout=use_mc_dropout,
        ece_bins=args.ece_bins,
        coverage_level=args.coverage_level,
    )

    out = {
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "zero_shot_transfer_suite": zero_shot,
        "realistic_gp_five_seed_sweep": sweep_summary,
        "reaction_diffusion_probe": reaction_probe,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
