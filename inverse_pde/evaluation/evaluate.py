from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from baselines.gp_baseline import GPBaseline
from baselines.mlp_baseline import MLPBaseline, train_mlp_baseline
from baselines.pinn_baseline import run_pinn_inversion
from data.generator import generate_instance
from training.losses import gaussian_nll_loss
from training.metrics import batch_metrics
from utils import ensure_dir


def _move_batch(batch: dict, device: torch.device) -> dict:
    return {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


def _stack_targets(batch: dict, target_fields: list[str]) -> torch.Tensor:
    if not target_fields:
        target_fields = ["k_grid"]
    tensors = [batch[name] for name in target_fields]
    if len(tensors) == 1:
        return tensors[0]
    return torch.stack(tensors, dim=-1)


def _point_estimate_to_probabilistic_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    ece_bins: int,
    coverage_level: float,
) -> dict:
    sigma = (pred - target).abs().mean(dim=(-2, -1), keepdim=True).expand_as(pred) + 1e-3
    return batch_metrics(mu=pred, sigma=sigma, target=target, ece_bins=ece_bins, coverage_level=coverage_level)


def _aggregate_grouped_metrics(
    grouped: dict[str, dict[str, float]],
    counts: dict[str, int],
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for key, vals in grouped.items():
        c = max(1, counts.get(key, 1))
        out[key] = {metric: value / c for metric, value in vals.items()}
    return out


@torch.no_grad()
def evaluate_main_model(
    model,
    test_loader,
    device: torch.device,
    ece_bins: int,
    coverage_level: float,
    target_fields: list[str],
    temperature: float = 1.0,
    use_mc_dropout: bool = False,
) -> dict:
    model.eval()
    stats = {"rmse": 0.0, "ece": 0.0, "coverage": 0.0}
    grouped_by_k_type: dict[str, dict[str, float]] = {}
    grouped_count: dict[str, int] = {}
    n = 0

    for batch in test_loader:
        batch = _move_batch(batch, device)
        if use_mc_dropout:
            mu, epistemic_std, aleatoric_sigma = model.predict_with_uncertainty(
                obs_coords=batch["obs_coords"],
                obs_times=batch.get("obs_times"),
                obs_values=batch["obs_values"],
                obs_key_padding_mask=batch["obs_key_padding_mask"],
            )
            sigma = torch.sqrt(torch.clamp(aleatoric_sigma**2 + epistemic_std**2, min=1e-12))
        else:
            mu, sigma = model(
                batch["obs_coords"],
                batch.get("obs_times"),
                batch["obs_values"],
                batch["obs_key_padding_mask"],
                mc_dropout=False,
            )
        sigma = sigma * temperature
        target = _stack_targets(batch=batch, target_fields=target_fields)
        met = batch_metrics(mu=mu, sigma=sigma, target=target, ece_bins=ece_bins, coverage_level=coverage_level)
        for key in stats:
            stats[key] += met[key]
        n += 1

        k_types = list(batch.get("k_type", ["unknown"] * target.shape[0]))
        unique_k_types = set(k_types)
        for k_type in unique_k_types:
            idxs = [i for i, t in enumerate(k_types) if t == k_type]
            idx_tensor = torch.tensor(idxs, device=target.device, dtype=torch.long)
            sub_met = batch_metrics(
                mu=mu.index_select(0, idx_tensor),
                sigma=sigma.index_select(0, idx_tensor),
                target=target.index_select(0, idx_tensor),
                ece_bins=ece_bins,
                coverage_level=coverage_level,
            )
            if k_type not in grouped_by_k_type:
                grouped_by_k_type[k_type] = {"rmse": 0.0, "ece": 0.0, "coverage": 0.0}
                grouped_count[k_type] = 0
            for m in grouped_by_k_type[k_type]:
                grouped_by_k_type[k_type][m] += sub_met[m] * len(idxs)
            grouped_count[k_type] += len(idxs)

    out = {key: value / max(1, n) for key, value in stats.items()}
    out["by_k_type"] = _aggregate_grouped_metrics(grouped_by_k_type, grouped_count)
    return out


@torch.no_grad()
def fit_temperature_scaling(
    model,
    val_loader,
    device: torch.device,
    target_fields: list[str],
    t_min: float = 0.5,
    t_max: float = 3.0,
    n_steps: int = 51,
    objective: str = "nll",
    ece_bins: int = 15,
    coverage_level: float = 0.90,
    coverage_min: float | None = None,
    enforce_non_sharpening: bool = False,
    use_mc_dropout: bool = False,
) -> dict:
    model.eval()
    coverage_min = coverage_level - 0.05 if coverage_min is None else coverage_min

    candidates: list[dict] = []

    temps = torch.linspace(t_min, t_max, n_steps, device=device)
    if enforce_non_sharpening:
        temps = temps[temps >= 1.0]
        if temps.numel() == 0:
            temps = torch.tensor([1.0], device=device)

    for t in temps:
        obj_sum = 0.0
        cov_sum = 0.0
        n_batches = 0
        for batch in val_loader:
            batch = _move_batch(batch, device)
            if use_mc_dropout:
                mu, epistemic_std, aleatoric_sigma = model.predict_with_uncertainty(
                    obs_coords=batch["obs_coords"],
                    obs_times=batch.get("obs_times"),
                    obs_values=batch["obs_values"],
                    obs_key_padding_mask=batch["obs_key_padding_mask"],
                )
                sigma = torch.sqrt(torch.clamp(aleatoric_sigma**2 + epistemic_std**2, min=1e-12))
            else:
                mu, sigma = model(
                    batch["obs_coords"],
                    batch.get("obs_times"),
                    batch["obs_values"],
                    batch["obs_key_padding_mask"],
                    mc_dropout=False,
                )
            target = _stack_targets(batch=batch, target_fields=target_fields)
            sigma_t = sigma * t
            met = batch_metrics(
                mu=mu,
                sigma=sigma_t,
                target=target,
                ece_bins=ece_bins,
                coverage_level=coverage_level,
            )
            if objective.lower() == "ece":
                obj_sum += float(met["ece"])
            else:
                nll = gaussian_nll_loss(target, mu, sigma_t)
                obj_sum += float(nll.item())
            cov_sum += float(met["coverage"])
            n_batches += 1

        mean_obj = obj_sum / max(1, n_batches)
        mean_cov = cov_sum / max(1, n_batches)
        candidates.append(
            {
                "temperature": float(t.item()),
                "objective": mean_obj,
                "coverage": mean_cov,
                "feasible": bool(mean_cov >= coverage_min),
            }
        )

    feasible = [c for c in candidates if c["feasible"]]

    if feasible:
        feasible.sort(key=lambda c: (c["objective"], abs(c["coverage"] - coverage_level)))
        selected = feasible[0]
        return {
            "temperature": selected["temperature"],
            "objective": selected["objective"],
            "coverage": selected["coverage"],
            "feasible_candidate_count": len(feasible),
            "fallback_reason": None,
        }

    fallback_obj = float("inf")
    fallback_cov = 0.0
    for c in candidates:
        if abs(c["temperature"] - 1.0) < 1e-6:
            fallback_obj = c["objective"]
            fallback_cov = c["coverage"]
            break

    return {
        "temperature": 1.0,
        "objective": fallback_obj,
        "coverage": fallback_cov,
        "feasible_candidate_count": 0,
        "fallback_reason": "no_feasible_temperature_for_coverage_constraint",
    }


@torch.no_grad()
def evaluate_mlp_baseline(
    model: MLPBaseline,
    test_loader,
    device: torch.device,
    ece_bins: int,
    coverage_level: float,
) -> dict:
    model.eval()
    model.to(device)
    rmse_vals = []
    ece_vals = []
    cov_vals = []

    for batch in test_loader:
        obs_coords = batch["obs_coords"].to(device)
        obs_values = batch["obs_values"].to(device)
        target = batch["k_grid"].to(device)
        pred = model(obs_coords, obs_values)
        metrics = _point_estimate_to_probabilistic_metrics(
            pred=pred,
            target=target,
            ece_bins=ece_bins,
            coverage_level=coverage_level,
        )
        rmse_vals.append(metrics["rmse"])
        ece_vals.append(metrics["ece"])
        cov_vals.append(metrics["coverage"])

    n = max(1, len(rmse_vals))
    return {
        "rmse": float(sum(rmse_vals) / n),
        "ece": float(sum(ece_vals) / n),
        "coverage": float(sum(cov_vals) / n),
    }


@torch.no_grad()
def evaluate_gp_baseline(
    gp: GPBaseline,
    test_loader,
    device: torch.device,
    ece_bins: int,
    coverage_level: float,
) -> dict:
    rmse_vals = []
    ece_vals = []
    cov_vals = []
    for batch in test_loader:
        pred = gp.predict(batch["obs_coords"], batch["obs_values"]).to(device)
        target = batch["k_grid"].to(device)
        metrics = _point_estimate_to_probabilistic_metrics(
            pred=pred,
            target=target,
            ece_bins=ece_bins,
            coverage_level=coverage_level,
        )
        rmse_vals.append(metrics["rmse"])
        ece_vals.append(metrics["ece"])
        cov_vals.append(metrics["coverage"])

    n = max(1, len(rmse_vals))
    return {
        "rmse": float(sum(rmse_vals) / n),
        "ece": float(sum(ece_vals) / n),
        "coverage": float(sum(cov_vals) / n),
    }


def evaluate_pinn_baseline(
    test_loader,
    device: torch.device,
    ece_bins: int,
    coverage_level: float,
    max_instances: int,
    pinn_steps: int,
    pinn_min_steps: int,
    pinn_convergence_tol: float,
    pinn_convergence_patience: int,
    lr: float,
) -> dict:
    rmse_vals = []
    ece_vals = []
    cov_vals = []
    times = []
    step_counts = []
    update_counts = []
    converged_flags = []
    best_losses = []
    count = 0

    for batch in test_loader:
        bsz = batch["obs_coords"].shape[0]
        for i in range(bsz):
            if count >= max_instances:
                break

            pred_k, elapsed, meta = run_pinn_inversion(
                obs_coords=batch["obs_coords"][i],
                obs_values=batch["obs_values"][i, :, 0],
                f_grid=batch["f_grid"][i],
                u_grid=batch["u_grid"][i],
                device=device,
                steps=pinn_steps,
                min_steps=pinn_min_steps,
                convergence_tol=pinn_convergence_tol,
                convergence_patience=pinn_convergence_patience,
                lr=lr,
            )
            pred = pred_k.unsqueeze(0).to(device)
            target = batch["k_grid"][i : i + 1].to(device)

            metrics = _point_estimate_to_probabilistic_metrics(
                pred=pred,
                target=target,
                ece_bins=ece_bins,
                coverage_level=coverage_level,
            )
            rmse_vals.append(metrics["rmse"])
            ece_vals.append(metrics["ece"])
            cov_vals.append(metrics["coverage"])
            times.append(elapsed)
            step_counts.append(int(meta.get("attempted_steps", 0)))
            update_counts.append(int(meta.get("update_steps", 0)))
            converged_flags.append(bool(meta.get("converged", False)))
            best_losses.append(float(meta.get("best_loss", 0.0)))
            count += 1

        if count >= max_instances:
            break

    n = max(1, len(rmse_vals))
    return {
        "rmse": float(sum(rmse_vals) / n),
        "ece": float(sum(ece_vals) / n),
        "coverage": float(sum(cov_vals) / n),
        "avg_time_sec": float(sum(times) / max(1, len(times))),
        "avg_steps": float(sum(step_counts) / max(1, len(step_counts))),
        "avg_update_steps": float(sum(update_counts) / max(1, len(update_counts))),
        "median_steps": float(statistics.median(step_counts)) if step_counts else 0.0,
        "converged_fraction": float(sum(1 for x in converged_flags if x) / max(1, len(converged_flags))),
        "avg_best_loss": float(sum(best_losses) / max(1, len(best_losses))),
        "optimizer": "adam",
        "lr": lr,
    }


@torch.no_grad()
def runtime_main_vs_pinn(
    model,
    test_loader,
    device: torch.device,
    n_instances: int = 100,
    use_mc_dropout: bool = False,
) -> dict:
    model.eval()

    main_times = []
    count = 0

    for batch in test_loader:
        bsz = batch["obs_coords"].shape[0]
        for i in range(bsz):
            if count >= n_instances:
                break

            obs_coords = batch["obs_coords"][i : i + 1].to(device)
            obs_times = batch.get("obs_times", None)
            obs_times = obs_times[i : i + 1].to(device) if obs_times is not None else None
            obs_values = batch["obs_values"][i : i + 1].to(device)
            mask = batch["obs_key_padding_mask"][i : i + 1].to(device)

            t0 = time.perf_counter()
            if use_mc_dropout:
                _ = model.predict_with_uncertainty(
                    obs_coords=obs_coords,
                    obs_times=obs_times,
                    obs_values=obs_values,
                    obs_key_padding_mask=mask,
                )
            else:
                _ = model(obs_coords, obs_times, obs_values, mask, mc_dropout=False)
            main_times.append(time.perf_counter() - t0)

            count += 1

        if count >= n_instances:
            break

    return {
        "main_avg_sec": float(sum(main_times) / max(1, len(main_times))),
    }


def evaluate_ood(
    model,
    device: torch.device,
    ece_bins: int,
    coverage_level: float,
    grid_size: int,
    target_fields: list[str],
    pde_family: str,
    n_samples: int = 256,
    temperature: float = 1.0,
    use_mc_dropout: bool = False,
) -> dict:
    cases = {
        "high_noise": {"noise_min": 0.1, "noise_max": 0.1, "m_min": 20, "m_max": 100, "noise_type": "gaussian"},
        "few_observations": {"noise_min": 1e-3, "noise_max": 5e-2, "m_min": 10, "m_max": 10, "noise_type": "gaussian"},
        "nu_0_5_only": {"noise_min": 1e-3, "noise_max": 5e-2, "m_min": 20, "m_max": 100, "nu_choices": (0.5,), "noise_type": "gaussian"},
        "non_smooth_checkerboard": {"noise_min": 1e-3, "noise_max": 5e-2, "m_min": 20, "m_max": 100, "k_type": "checkerboard", "noise_type": "gaussian"},
        "correlated_noise": {"noise_min": 1e-3, "noise_max": 5e-2, "m_min": 20, "m_max": 100, "noise_type": "correlated"},
        "outlier_noise": {"noise_min": 1e-3, "noise_max": 5e-2, "m_min": 20, "m_max": 100, "noise_type": "outlier"},
    }

    out = {}
    for case_name, kwargs in cases.items():
        rmse_vals, ece_vals, cov_vals = [], [], []
        for _ in range(n_samples):
            sample = generate_instance(grid_size=grid_size, pde_family=pde_family, **kwargs)
            obs_coords = sample["obs_coords"].unsqueeze(0).to(device).float()
            obs_times = sample["obs_times"].unsqueeze(0).unsqueeze(-1).to(device).float()
            obs_values = sample["obs_values"].unsqueeze(0).unsqueeze(-1).to(device).float()
            mask = torch.zeros(1, obs_coords.shape[1], dtype=torch.bool, device=device)

            targets = []
            for name in target_fields:
                targets.append(sample[name].unsqueeze(0).to(device).float())
            target = targets[0] if len(targets) == 1 else torch.stack(targets, dim=-1)

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
            sigma = sigma * temperature
            met = batch_metrics(mu=mu, sigma=sigma, target=target, ece_bins=ece_bins, coverage_level=coverage_level)
            rmse_vals.append(met["rmse"])
            ece_vals.append(met["ece"])
            cov_vals.append(met["coverage"])

        out[case_name] = {
            "rmse": float(sum(rmse_vals) / len(rmse_vals)),
            "ece": float(sum(ece_vals) / len(ece_vals)),
            "coverage": float(sum(cov_vals) / len(cov_vals)),
        }

    return out


def evaluate_resolution_transfer(
    model,
    device: torch.device,
    ece_bins: int,
    coverage_level: float,
    target_fields: list[str],
    pde_family: str,
    n_samples: int = 64,
) -> dict:
    rmse_native = []
    rmse_up64 = []

    for _ in range(n_samples):
        sample32 = generate_instance(grid_size=32, pde_family=pde_family)
        sample64 = generate_instance(grid_size=64, pde_family=pde_family)

        obs_coords = sample64["obs_coords"].unsqueeze(0).to(device).float()
        obs_times = sample64["obs_times"].unsqueeze(0).unsqueeze(-1).to(device).float()
        obs_values = sample64["obs_values"].unsqueeze(0).unsqueeze(-1).to(device).float()
        mask = torch.zeros(1, obs_coords.shape[1], dtype=torch.bool, device=device)

        mu32, _ = model(obs_coords, obs_times, obs_values, mask, mc_dropout=False)
        if mu32.dim() == 4:
            mu32_main = mu32[..., 0]
        else:
            mu32_main = mu32

        target32 = sample32[target_fields[0]].unsqueeze(0).to(device).float()
        target64 = sample64[target_fields[0]].unsqueeze(0).to(device).float()

        mu32_img = mu32_main.unsqueeze(1)
        mu64 = F.interpolate(mu32_img, size=(64, 64), mode="bicubic", align_corners=False).squeeze(1)

        rmse_native.append(torch.sqrt(torch.mean((mu32_main - target32) ** 2)).item())
        rmse_up64.append(torch.sqrt(torch.mean((mu64 - target64) ** 2)).item())

    return {
        "rmse_32_native": float(sum(rmse_native) / max(1, len(rmse_native))),
        "rmse_32_to_64_upsampled": float(sum(rmse_up64) / max(1, len(rmse_up64))),
        "n_samples": int(n_samples),
    }


def run_full_evaluation(
    model,
    train_loader,
    test_loader,
    config: dict,
    device: torch.device,
    output_dir: str | Path = "results",
    val_loader=None,
    calibration_only: bool = False,
) -> dict:
    out_dir = ensure_dir(output_dir)
    eval_start = time.perf_counter()
    print(f"[EVAL] Starting evaluation. Output dir: {out_dir}", flush=True)

    ece_bins = int(config["evaluation"]["ece_bins"])
    coverage_level = float(config["evaluation"]["coverage_level"])
    target_fields = list(config["training"].get("target_fields", ["k_grid"]))
    pde_family = str(config["data"].get("pde_family", "diffusion"))

    temp_cfg = config["evaluation"].get("temperature_scaling", {})
    temp_enabled = bool(temp_cfg.get("enabled", False))
    infer_cfg = config["evaluation"].get("inference", {})
    use_mc_dropout = bool(infer_cfg.get("use_mc_dropout", False))
    temperature = 1.0
    calibration_val_objective = None
    calibration_val_coverage = None
    calibration_feasible_candidates = None
    calibration_fallback_reason = None
    calibration_objective_name = str(temp_cfg.get("objective", "nll")).lower()

    if temp_enabled and val_loader is not None:
        print("[EVAL] Fitting temperature scaling...", flush=True)
        calibration_result = fit_temperature_scaling(
            model=model,
            val_loader=val_loader,
            device=device,
            target_fields=target_fields,
            t_min=float(temp_cfg.get("t_min", 0.5)),
            t_max=float(temp_cfg.get("t_max", 3.0)),
            n_steps=int(temp_cfg.get("n_steps", 51)),
            objective=calibration_objective_name,
            ece_bins=ece_bins,
            coverage_level=coverage_level,
            coverage_min=float(temp_cfg.get("coverage_min", coverage_level - 0.05)),
            enforce_non_sharpening=bool(temp_cfg.get("enforce_non_sharpening", False)),
            use_mc_dropout=use_mc_dropout,
        )
        temperature = float(calibration_result["temperature"])
        calibration_val_objective = float(calibration_result["objective"])
        calibration_val_coverage = float(calibration_result["coverage"])
        calibration_feasible_candidates = int(calibration_result["feasible_candidate_count"])
        calibration_fallback_reason = calibration_result["fallback_reason"]

    pinn_cfg = config["evaluation"].get("pinn", {})
    eval_budget = config["evaluation"].get("budget", {})

    print("[EVAL] Evaluating main model on test set...", flush=True)
    main_metrics = evaluate_main_model(
        model=model,
        test_loader=test_loader,
        device=device,
        ece_bins=ece_bins,
        coverage_level=coverage_level,
        target_fields=target_fields,
        temperature=temperature,
        use_mc_dropout=use_mc_dropout,
    )

    if calibration_only:
        print("[EVAL] Calibration-only mode enabled; skipping baselines, runtime, OOD, and resolution transfer.", flush=True)
        mlp_metrics = {}
        gp_metrics = {}
        pinn_metrics = {}
        timing = {}
        ood = {}
        resolution_transfer = {}
    else:
        max_m = int(config["data"].get("m_max", 100)) * max(1, int(config["data"].get("n_time_snapshots", 1)))
        print(f"[EVAL] Training MLP baseline (epochs={int(eval_budget.get('mlp_epochs', 20))})...", flush=True)
        mlp_baseline = MLPBaseline(max_m=max_m, grid_size=int(config["data"]["grid_size"]))
        mlp_baseline = train_mlp_baseline(
            mlp_baseline,
            train_loader,
            device=device,
            epochs=int(eval_budget.get("mlp_epochs", 20)),
        )
        mlp_metrics = evaluate_mlp_baseline(
            mlp_baseline,
            test_loader,
            device=device,
            ece_bins=ece_bins,
            coverage_level=coverage_level,
        )

        print(f"[EVAL] Fitting GP baseline (max_samples={int(eval_budget.get('gp_max_samples', 1500))})...", flush=True)
        gp_baseline = GPBaseline(max_m=max_m, grid_size=int(config["data"]["grid_size"]), n_components=16)
        gp_baseline.fit(train_loader, max_samples=int(eval_budget.get("gp_max_samples", 1500)))
        gp_metrics = evaluate_gp_baseline(
            gp_baseline,
            test_loader,
            device=device,
            ece_bins=ece_bins,
            coverage_level=coverage_level,
        )

        print(
            "[EVAL] Running PINN baseline "
            f"(max_instances={int(pinn_cfg.get('max_instances', 100))}, "
            f"steps={int(pinn_cfg.get('steps', 1000))})...",
            flush=True,
        )
        pinn_metrics = evaluate_pinn_baseline(
            test_loader=test_loader,
            device=device,
            ece_bins=ece_bins,
            coverage_level=coverage_level,
            max_instances=int(pinn_cfg.get("max_instances", 100)),
            pinn_steps=int(pinn_cfg.get("steps", 1000)),
            pinn_min_steps=int(pinn_cfg.get("min_steps", 1)),
            pinn_convergence_tol=float(pinn_cfg.get("convergence_tol", 1e-6)),
            pinn_convergence_patience=int(pinn_cfg.get("convergence_patience", 200)),
            lr=float(pinn_cfg.get("lr", 1e-3)),
        )

        print(f"[EVAL] Measuring runtime (instances={int(eval_budget.get('runtime_instances', 100))})...", flush=True)
        timing = runtime_main_vs_pinn(
            model=model,
            test_loader=test_loader,
            device=device,
            n_instances=int(eval_budget.get("runtime_instances", 100)),
            use_mc_dropout=bool(infer_cfg.get("timing_use_mc_dropout", False)),
        )
        print(f"[EVAL] Running OOD suite (samples/case={int(eval_budget.get('ood_samples', 64))})...", flush=True)
        ood = evaluate_ood(
            model=model,
            device=device,
            ece_bins=ece_bins,
            coverage_level=coverage_level,
            grid_size=int(config["data"]["grid_size"]),
            target_fields=target_fields,
            pde_family=pde_family,
            n_samples=int(eval_budget.get("ood_samples", 64)),
            temperature=temperature,
            use_mc_dropout=use_mc_dropout,
        )
        print(
            f"[EVAL] Running resolution transfer (samples={int(eval_budget.get('resolution_samples', 32))})...",
            flush=True,
        )
        resolution_transfer = evaluate_resolution_transfer(
            model=model,
            device=device,
            ece_bins=ece_bins,
            coverage_level=coverage_level,
            target_fields=target_fields,
            pde_family=pde_family,
            n_samples=int(eval_budget.get("resolution_samples", 32)),
        )

    best_pinn = pinn_metrics
    result = {
        "main_model": main_metrics,
        "baselines": {
            "mlp": mlp_metrics,
            "gp": gp_metrics,
            "pinn": best_pinn,
        },
        "timing": timing,
        "pinn_timing": {
            "pinn_avg_sec": best_pinn.get("avg_time_sec", 0.0),
            "pinn_avg_steps": best_pinn.get("avg_steps", 0.0),
            "pinn_avg_update_steps": best_pinn.get("avg_update_steps", 0.0),
            "pinn_median_steps": best_pinn.get("median_steps", 0.0),
            "pinn_converged_fraction": best_pinn.get("converged_fraction", 0.0),
        },
        "calibration": {
            "temperature_scaling_enabled": temp_enabled,
            "objective": calibration_objective_name,
            "temperature": float(temperature),
            "use_mc_dropout": use_mc_dropout,
            "val_objective_at_temperature": calibration_val_objective,
            "val_coverage_at_temperature": calibration_val_coverage,
            "feasible_candidate_count": calibration_feasible_candidates,
            "fallback_reason": calibration_fallback_reason,
        },
        "ood": ood,
        "resolution_transfer": resolution_transfer,
    }

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    elapsed = time.perf_counter() - eval_start
    print(f"[EVAL] Completed in {elapsed:.1f}s. Metrics written to {out_dir / 'metrics.json'}", flush=True)

    return result
