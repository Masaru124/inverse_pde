from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from training.metrics import ece_regression
from utils import ensure_dir


@torch.no_grad()
def _single_calibration_points(mu: torch.Tensor, sigma: torch.Tensor, target: torch.Tensor, n_bins: int = 15):
    conf = 1.0 / (sigma + 1e-6)
    conf = (conf - conf.min()) / (conf.max() - conf.min() + 1e-6)

    z = torch.tensor(1.6448536269514722, device=mu.device)
    in_interval = ((target >= mu - z * sigma) & (target <= mu + z * sigma)).float()

    conf_flat = conf.reshape(-1)
    acc_flat = in_interval.reshape(-1)

    edges = torch.linspace(0, 1, n_bins + 1, device=mu.device)
    xs, ys = [], []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (conf_flat >= lo) & (conf_flat < hi if i < n_bins - 1 else conf_flat <= hi)
        if mask.any():
            xs.append(conf_flat[mask].mean().item())
            ys.append(acc_flat[mask].mean().item())
    return xs, ys


@torch.no_grad()
def save_instance_figures(
    model,
    test_loader,
    device: torch.device,
    out_dir: str | Path = "results/figures",
    n_instances: int = 6,
):
    out_path = ensure_dir(out_dir)
    model.eval()

    examples = []
    for batch in test_loader:
        bsz = batch["obs_coords"].shape[0]
        for i in range(bsz):
            obs_coords = batch["obs_coords"][i : i + 1].to(device)
            obs_values = batch["obs_values"][i : i + 1].to(device)
            mask = batch["obs_key_padding_mask"][i : i + 1].to(device)

            pred_mean, epistemic, aleatoric = model.predict_with_uncertainty(
                obs_coords=obs_coords,
                obs_times=batch.get("obs_times", None)[i : i + 1].to(device) if "obs_times" in batch else None,
                obs_values=obs_values,
                obs_key_padding_mask=mask,
            )
            target = batch["k_grid"][i : i + 1].to(device)
            pred_for_plot = pred_mean[..., 0] if pred_mean.dim() == 4 else pred_mean
            epi_for_plot = epistemic[..., 0] if epistemic.dim() == 4 else epistemic
            ale_for_plot = aleatoric[..., 0] if aleatoric.dim() == 4 else aleatoric
            rmse = torch.sqrt(((pred_for_plot - target) ** 2).mean()).item()

            examples.append(
                {
                    "rmse": rmse,
                    "obs_coords": batch["obs_coords"][i],
                    "u_grid": batch["u_grid"][i],
                    "target": target[0].cpu(),
                    "pred": pred_for_plot[0].cpu(),
                    "epi": epi_for_plot[0].cpu(),
                    "ale": ale_for_plot[0].cpu(),
                }
            )

        if len(examples) >= max(30, n_instances):
            break

    examples = sorted(examples, key=lambda x: x["rmse"])
    selected = examples[: max(0, n_instances - 1)] + [examples[-1]]

    for idx, item in enumerate(selected[:n_instances]):
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        ax = axes.ravel()

        im0 = ax[0].imshow(item["target"], cmap="viridis")
        ax[0].set_title("True k")
        plt.colorbar(im0, ax=ax[0], fraction=0.046)

        im1 = ax[1].imshow(item["pred"], cmap="viridis")
        ax[1].set_title("Predicted mean k")
        plt.colorbar(im1, ax=ax[1], fraction=0.046)

        total_unc = item["epi"] + item["ale"]
        im2 = ax[2].imshow(total_unc, cmap="magma")
        ax[2].set_title("Uncertainty (epi+ale)")
        plt.colorbar(im2, ax=ax[2], fraction=0.046)

        ax[3].imshow(item["u_grid"], cmap="coolwarm")
        obs = item["obs_coords"].numpy()
        ax[3].scatter(obs[:, 0] * 31.0, obs[:, 1] * 31.0, s=8, c="white", edgecolors="black", linewidths=0.3)
        ax[3].set_title("Obs locations over u")

        abs_err = torch.abs(item["pred"] - item["target"])
        im4 = ax[4].imshow(abs_err, cmap="inferno")
        ax[4].set_title("Absolute error")
        plt.colorbar(im4, ax=ax[4], fraction=0.046)

        xs, ys = _single_calibration_points(
            mu=item["pred"],
            sigma=item["ale"],
            target=item["target"],
            n_bins=15,
        )
        ax[5].plot([0, 1], [0, 1], "k--", linewidth=1)
        ax[5].plot(xs, ys, marker="o")
        ece_val = ece_regression(
            mu=item["pred"],
            sigma=item["ale"],
            target=item["target"],
            n_bins=15,
        ).item()
        ax[5].set_title(f"Calibration (ECE={ece_val:.3f})")
        ax[5].set_xlim(0, 1)
        ax[5].set_ylim(0, 1)

        fig.tight_layout()
        fig.savefig(Path(out_path) / f"instance_{idx}.png", dpi=150)
        plt.close(fig)


@torch.no_grad()
def save_attention_figures(
    model,
    test_loader,
    device: torch.device,
    out_dir: str | Path = "results/figures/attention",
    n_instances: int = 4,
    layer_idx: int = -1,
    head_idx: int = 0,
) -> None:
    out_path = ensure_dir(out_dir)
    if not hasattr(model, "encoder") or not hasattr(model.encoder, "enable_attention_capture"):
        return

    model.eval()
    model.encoder.enable_attention_capture()

    saved = 0
    for batch in test_loader:
        bsz = batch["obs_coords"].shape[0]
        for i in range(bsz):
            if saved >= n_instances:
                break

            obs_coords = batch["obs_coords"][i : i + 1].to(device)
            obs_times = batch.get("obs_times", None)
            obs_times = obs_times[i : i + 1].to(device) if obs_times is not None else None
            obs_values = batch["obs_values"][i : i + 1].to(device)
            mask = batch["obs_key_padding_mask"][i : i + 1].to(device)

            _ = model(obs_coords=obs_coords, obs_times=obs_times, obs_values=obs_values, obs_key_padding_mask=mask)
            attn = model.encoder.get_last_attention_weights()
            if not attn:
                continue
            keys = sorted(attn.keys())
            chosen = keys[layer_idx] if layer_idx < 0 else layer_idx
            w = attn.get(chosen)
            if w is None:
                continue

            # (B, H, Q, K) -> mean over heads and keys gives query-level attention intensity.
            q_map = w[0, head_idx].mean(dim=-1).reshape(model.encoder.grid_size, model.encoder.grid_size).cpu().numpy()
            fig, ax = plt.subplots(1, 1, figsize=(4, 4))
            im = ax.imshow(q_map, cmap="magma")
            obs_xy = obs_coords[0].cpu().numpy()
            ax.scatter(obs_xy[:, 0] * (model.encoder.grid_size - 1), obs_xy[:, 1] * (model.encoder.grid_size - 1), s=8, c="cyan")
            ax.set_title(f"Attention map #{saved}")
            plt.colorbar(im, ax=ax, fraction=0.046)
            fig.tight_layout()
            fig.savefig(Path(out_path) / f"attention_{saved}.png", dpi=160)
            plt.close(fig)
            saved += 1

        if saved >= n_instances:
            break

    model.encoder.disable_attention_capture()
