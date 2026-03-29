from __future__ import annotations

import csv
import heapq
import time
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Subset

from training.losses import total_loss
from training.metrics import batch_metrics
from utils import ensure_dir

try:
    import wandb  # type: ignore
except Exception:
    wandb = None


class CSVLogger:
    def __init__(self, csv_path: str | Path):
        self.csv_path = Path(csv_path)
        ensure_dir(self.csv_path.parent)
        self._header_written = False

    def log(self, row: Dict[str, float]) -> None:
        write_header = not self.csv_path.exists() or not self._header_written
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
                self._header_written = True
            writer.writerow(row)


class TopKCheckpoints:
    def __init__(self, out_dir: str | Path, k: int = 3):
        self.out_dir = ensure_dir(out_dir)
        self.k = k
        self.heap: list[tuple[float, Path]] = []

    def add(self, score: float, payload: dict, epoch: int) -> None:
        ckpt_path = self.out_dir / f"epoch_{epoch:03d}_valnll_{score:.6f}.pt"
        torch.save(payload, ckpt_path)

        entry = (-score, ckpt_path)
        if len(self.heap) < self.k:
            heapq.heappush(self.heap, entry)
            return

        worst_neg_score, worst_path = self.heap[0]
        worst_score = -worst_neg_score
        if score < worst_score:
            heapq.heapreplace(self.heap, entry)
            if worst_path.exists():
                worst_path.unlink()
        else:
            if ckpt_path.exists():
                ckpt_path.unlink()


def _move_batch(batch: dict, device: torch.device) -> dict:
    out = {}
    for key, value in batch.items():
        out[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    return out


def _stack_targets(batch: dict, target_fields: list[str]) -> torch.Tensor:
    if not target_fields:
        target_fields = ["k_grid"]
    tensors = [batch[name] for name in target_fields]
    if len(tensors) == 1:
        return tensors[0]
    return torch.stack(tensors, dim=-1)


def _evaluate(
    model: nn.Module,
    loader,
    device: torch.device,
    lambda_reg: float,
    sigma_floor: float,
    sigma_reg_weight: float,
    ece_bins: int,
    coverage_level: float,
    target_fields: list[str],
) -> Dict[str, float]:
    model.eval()
    val_loss = 0.0
    val_nll = 0.0
    val_rmse = 0.0
    val_ece = 0.0
    val_cov = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in loader:
            batch = _move_batch(batch, device)
            mu, sigma, log_sigma = model.forward_with_logsigma(
                batch["obs_coords"],
                batch.get("obs_times"),
                batch["obs_values"],
                batch["obs_key_padding_mask"],
                mc_dropout=False,
            )
            target = _stack_targets(batch=batch, target_fields=target_fields)
            loss, nll, _ = total_loss(
                k_true=target,
                mu=mu,
                sigma=sigma,
                log_sigma=log_sigma,
                lambda_reg=lambda_reg,
                sigma_floor=sigma_floor,
                sigma_reg_weight=sigma_reg_weight,
            )
            metrics = batch_metrics(
                mu=mu,
                sigma=sigma,
                target=target,
                ece_bins=ece_bins,
                coverage_level=coverage_level,
            )

            val_loss += float(loss.item())
            val_nll += float(nll.item())
            val_rmse += metrics["rmse"]
            val_ece += metrics["ece"]
            val_cov += metrics["coverage"]
            n_batches += 1

    if n_batches == 0:
        return {"loss": 0.0, "nll": 0.0, "rmse": 0.0, "ece": 0.0, "coverage": 0.0}

    return {
        "loss": val_loss / n_batches,
        "nll": val_nll / n_batches,
        "rmse": val_rmse / n_batches,
        "ece": val_ece / n_batches,
        "coverage": val_cov / n_batches,
    }


def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    config: dict,
    device: torch.device,
    output_dir: str | Path = "outputs",
) -> dict:
    output_dir = ensure_dir(output_dir)
    ckpt_mgr = TopKCheckpoints(output_dir / "checkpoints", k=3)

    lr = float(config["training"]["lr"])
    weight_decay = float(config["training"]["weight_decay"])
    epochs = int(config["training"]["epochs"])
    patience = int(config["training"]["patience"])
    lambda_reg = float(config["training"]["lambda_reg"])
    grad_clip = float(config["training"]["grad_clip"])
    sigma_floor = float(config["training"].get("sigma_floor", 0.1))
    sigma_reg_weight = float(config["training"].get("sigma_reg_weight", 0.1))
    min_epochs_before_early_stop = int(config["training"].get("min_epochs_before_early_stop", 0))
    warmup_epochs = int(config["training"].get("warmup_epochs", 0))

    ece_bins = int(config["evaluation"]["ece_bins"])
    coverage_level = float(config["evaluation"]["coverage_level"])
    target_fields = list(config["training"].get("target_fields", ["k_grid"]))

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if warmup_epochs > 0:
        warmup_epochs = min(warmup_epochs, max(1, epochs - 1))
        warmup = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine = CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup_epochs))
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_epochs],
        )
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    use_wandb = wandb is not None
    csv_logger = CSVLogger(output_dir / "training_log.csv")

    if use_wandb:
        wandb.init(project="amortized-inverse-pde", config=config)

    model.to(device)
    best_val_nll = float("inf")
    wait = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_train_nll = 0.0
        n_batches = 0

        start = time.perf_counter()
        for batch in train_loader:
            batch = _move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)

            mu, sigma, log_sigma = model.forward_with_logsigma(
                batch["obs_coords"],
                batch.get("obs_times"),
                batch["obs_values"],
                batch["obs_key_padding_mask"],
                mc_dropout=False,
            )
            target = _stack_targets(batch=batch, target_fields=target_fields)
            loss, nll, _ = total_loss(
                k_true=target,
                mu=mu,
                sigma=sigma,
                log_sigma=log_sigma,
                lambda_reg=lambda_reg,
                sigma_floor=sigma_floor,
                sigma_reg_weight=sigma_reg_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

            epoch_train_nll += float(nll.item())
            n_batches += 1

        scheduler.step()

        train_nll = epoch_train_nll / max(1, n_batches)
        val_stats = _evaluate(
            model=model,
            loader=val_loader,
            device=device,
            lambda_reg=lambda_reg,
            sigma_floor=sigma_floor,
            sigma_reg_weight=sigma_reg_weight,
            ece_bins=ece_bins,
            coverage_level=coverage_level,
            target_fields=target_fields,
        )

        row = {
            "epoch": float(epoch),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_nll": train_nll,
            "val_nll": val_stats["nll"],
            "val_rmse": val_stats["rmse"],
            "val_ece": val_stats["ece"],
            "val_coverage": val_stats["coverage"],
            "epoch_time_sec": time.perf_counter() - start,
        }
        history.append(row)
        csv_logger.log(row)

        if use_wandb:
            wandb.log(row)

        ckpt_payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_nll": val_stats["nll"],
            "config": config,
        }
        ckpt_mgr.add(score=val_stats["nll"], payload=ckpt_payload, epoch=epoch)

        if val_stats["nll"] < best_val_nll:
            best_val_nll = val_stats["nll"]
            wait = 0
        else:
            wait += 1
            if wait >= patience and epoch >= min_epochs_before_early_stop:
                break

    if use_wandb:
        wandb.finish()

    return {"best_val_nll": best_val_nll, "history": history}


def run_overfit_sanity(
    model: nn.Module,
    train_loader,
    config: dict,
    device: torch.device,
    output_dir: str | Path = "outputs/sanity",
) -> dict:
    output_dir = ensure_dir(output_dir)
    csv_logger = CSVLogger(output_dir / "sanity_log.csv")

    sanity_samples = int(config["training"].get("sanity_samples", 10))
    sanity_epochs = int(config["training"].get("sanity_epochs", 200))
    lambda_reg = float(config["training"]["lambda_reg"])
    sigma_floor = float(config["training"].get("sigma_floor", 0.1))
    sigma_reg_weight = float(config["training"].get("sigma_reg_weight", 0.1))
    grad_clip = float(config["training"]["grad_clip"])
    lr = float(config["training"].get("sanity_lr", config["training"]["lr"]))
    weight_decay = float(config["training"]["weight_decay"])
    target_fields = list(config["training"].get("target_fields", ["k_grid"]))

    subset_count = min(sanity_samples, len(train_loader.dataset))
    subset = Subset(train_loader.dataset, range(subset_count))
    sanity_loader = DataLoader(
        subset,
        batch_size=min(int(config["training"]["batch_size"]), subset_count),
        shuffle=True,
        num_workers=0,
        collate_fn=train_loader.collate_fn,
    )

    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_rmse = float("inf")
    history = []

    for epoch in range(1, sanity_epochs + 1):
        model.train()
        nll_sum = 0.0
        n_train_batches = 0

        for batch in sanity_loader:
            batch = _move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)

            mu, sigma, log_sigma = model.forward_with_logsigma(
                batch["obs_coords"],
                batch.get("obs_times"),
                batch["obs_values"],
                batch["obs_key_padding_mask"],
                mc_dropout=False,
            )
            target = _stack_targets(batch=batch, target_fields=target_fields)
            loss, nll, _ = total_loss(
                k_true=target,
                mu=mu,
                sigma=sigma,
                log_sigma=log_sigma,
                lambda_reg=lambda_reg,
                sigma_floor=sigma_floor,
                sigma_reg_weight=sigma_reg_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

            nll_sum += float(nll.item())
            n_train_batches += 1

        sanity_nll = nll_sum / max(1, n_train_batches)

        model.eval()
        rmse_sum = 0.0
        n_eval_batches = 0
        with torch.no_grad():
            for batch in sanity_loader:
                batch = _move_batch(batch, device)
                mu, _ = model(
                    batch["obs_coords"],
                    batch.get("obs_times"),
                    batch["obs_values"],
                    batch["obs_key_padding_mask"],
                    mc_dropout=False,
                )
                target = _stack_targets(batch=batch, target_fields=target_fields)
                rmse = torch.sqrt(torch.mean((mu - target) ** 2))
                rmse_sum += float(rmse.item())
                n_eval_batches += 1

        sanity_rmse = rmse_sum / max(1, n_eval_batches)
        best_rmse = min(best_rmse, sanity_rmse)

        row = {
            "epoch": float(epoch),
            "sanity_nll": sanity_nll,
            "sanity_rmse": sanity_rmse,
        }
        history.append(row)
        csv_logger.log(row)

    final_rmse = history[-1]["sanity_rmse"] if history else float("inf")
    return {
        "final_rmse": final_rmse,
        "best_rmse": best_rmse,
        "history": history,
    }
