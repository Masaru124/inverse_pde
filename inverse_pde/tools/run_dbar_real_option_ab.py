from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.model import AmortizedInversePDEModel
from training.losses import total_loss


def _checkpoint_score(path: Path) -> tuple[int, float, str]:
    match = re.search(r"val_nll_(-?\d+(?:\.\d+)?)", path.name)
    if match:
        return (0, float(match.group(1)), path.name)
    return (1, float("inf"), path.name)


def discover_checkpoints() -> list[Path]:
    paths: list[Path] = []
    for pattern in ("outputs*/checkpoints/*.pt", "outputs_recovery*/checkpoints/*.pt", "outputs_improved_v1/checkpoints/*.pt"):
        paths.extend(ROOT.glob(pattern))
    return sorted({p.resolve() for p in paths}, key=_checkpoint_score)


def load_model(checkpoint_path: Path | None) -> tuple[AmortizedInversePDEModel, dict[str, Any], torch.device]:
    checkpoints = discover_checkpoints()
    if checkpoint_path is None:
        preferred = ROOT / "outputs_recovery_gpu_fast_stable" / "checkpoints" / "epoch_006_val_nll_-0.618860.pt"
        checkpoint_path = preferred if preferred.exists() else (checkpoints[0] if checkpoints else None)

    if checkpoint_path is None or not checkpoint_path.exists():
        raise FileNotFoundError("No checkpoint found in outputs*/checkpoints/*.pt")

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
    return model, {"checkpoint": str(checkpoint_path)}, device


def boundary_coords(n: int, device: torch.device) -> torch.Tensor:
    theta = torch.linspace(0, 2 * np.pi, n + 1, device=device)[:-1]
    x = 0.5 + 0.5 * torch.cos(theta)
    y = 0.5 + 0.5 * torch.sin(theta)
    return torch.stack([x, y], dim=-1)


def load_dbar_case(data_dir: Path, grid_size: int = 32) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    nd = sio.loadmat(data_dir / "ND.mat", squeeze_me=True, struct_as_record=False)["NtoD"]
    recon = sio.loadmat(data_dir / "recon.mat", squeeze_me=True, struct_as_record=False)["recon"]

    nd = np.asarray(nd)
    if nd.shape != (32, 32):
        raise ValueError(f"Expected NtoD shape (32,32), got {nd.shape}")

    # Real-valued proxy for scalar observation channel expected by the current model.
    obs = np.real(nd).astype(np.float32)
    obs = (obs - obs.mean(axis=1, keepdims=True)) / (obs.std(axis=1, keepdims=True) + 1e-6)

    recon = np.real(np.asarray(recon)).astype(np.float32).reshape(64, 64)
    recon_t = torch.from_numpy(recon).unsqueeze(0).unsqueeze(0)
    target = F.interpolate(recon_t, size=(grid_size, grid_size), mode="bilinear", align_corners=True).squeeze(0).squeeze(0)

    obs_t = torch.from_numpy(obs)  # (32 stim patterns, 32 electrodes)
    coords_t = boundary_coords(32, device=torch.device("cpu"))
    return obs_t, coords_t, target


@dataclass
class Metrics:
    rmse: float
    mae: float
    coverage_90: float


def evaluate(model: AmortizedInversePDEModel, obs_rows: torch.Tensor, coords: torch.Tensor, target: torch.Tensor, device: torch.device) -> Metrics:
    model.eval()
    rmses: list[float] = []
    maes: list[float] = []
    covs: list[float] = []

    coords_b = coords.to(device).unsqueeze(0)
    mask = torch.zeros((1, coords.shape[0]), dtype=torch.bool, device=device)
    target_d = target.to(device)

    for i in range(obs_rows.shape[0]):
        values = obs_rows[i].to(device).unsqueeze(0).unsqueeze(-1)
        mean, epi, ale = model.predict_with_uncertainty(
            obs_coords=coords_b,
            obs_times=None,
            obs_values=values,
            obs_key_padding_mask=mask,
        )
        pred = mean.squeeze(0)
        unc = torch.clamp((epi + ale).squeeze(0), min=1e-6)
        err = pred - target_d
        rmses.append(torch.sqrt(torch.mean(err**2)).item())
        maes.append(torch.mean(torch.abs(err)).item())
        covs.append(torch.mean((torch.abs(err) <= 1.64 * unc).float()).item())

    return Metrics(float(np.mean(rmses)), float(np.mean(maes)), float(np.mean(covs)))


def finetune_decoder(model: AmortizedInversePDEModel, obs_rows: torch.Tensor, coords: torch.Tensor, target: torch.Tensor, device: torch.device, steps: int, lr: float) -> list[float]:
    for p in model.parameters():
        p.requires_grad = False
    for p in model.decoder.parameters():
        p.requires_grad = True

    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    losses: list[float] = []

    bsz, m = obs_rows.shape
    coords_b = coords.to(device).unsqueeze(0).repeat(bsz, 1, 1)
    values_b = obs_rows.to(device).unsqueeze(-1)
    mask = torch.zeros((bsz, m), dtype=torch.bool, device=device)
    target_b = target.to(device).unsqueeze(0).repeat(bsz, 1, 1)

    model.train()
    for _ in range(steps):
        mu, sigma, log_sigma = model.forward_with_logsigma(
            obs_coords=coords_b,
            obs_times=None,
            obs_values=values_b,
            obs_key_padding_mask=mask,
            mc_dropout=True,
        )
        loss, _, _ = total_loss(k_true=target_b, mu=mu, sigma=sigma, log_sigma=log_sigma)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

    model.eval()
    return losses


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Option A/B on DBAR real-data files")
    parser.add_argument("--data-dir", type=str, default="dbar/data")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--finetune-steps", type=int, default=120)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output", type=str, default="results_option_ab_dbar_real.json")
    args = parser.parse_args()

    model, meta, device = load_model(Path(args.checkpoint).resolve() if args.checkpoint else None)
    obs_rows, coords, target = load_dbar_case(ROOT / args.data_dir, grid_size=32)

    option_a = evaluate(model, obs_rows, coords, target, device)

    adapted = copy.deepcopy(model)
    losses = finetune_decoder(adapted, obs_rows, coords, target, device, steps=args.finetune_steps, lr=args.lr)
    option_b = evaluate(adapted, obs_rows, coords, target, device)

    result = {
        "checkpoint": meta["checkpoint"],
        "data_dir": str((ROOT / args.data_dir).resolve()),
        "note": "Real DBAR files used. NtoD complex matrix reduced to real part for scalar-input model compatibility.",
        "setup": {
            "n_cases": int(obs_rows.shape[0]),
            "n_boundary": int(obs_rows.shape[1]),
            "finetune_steps": args.finetune_steps,
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
            "first": losses[0] if losses else None,
            "last": losses[-1] if losses else None,
            "min": min(losses) if losses else None,
        },
    }

    out_path = ROOT / args.output
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("=== Option A/B on DBAR real data ===")
    print(json.dumps(result, indent=2))
    print(f"Saved report to: {out_path}")


if __name__ == "__main__":
    main()
