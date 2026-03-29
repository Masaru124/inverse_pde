from __future__ import annotations

import argparse
from pathlib import Path

import torch

from data.dataset import build_dataloaders
from data.generator import generate_dataset_to_disk, validate_generator
from evaluation.evaluate import run_full_evaluation
from evaluation.visualize import save_attention_figures, save_instance_figures
from model.model import AmortizedInversePDEModel
from training.train import run_overfit_sanity, train_model
from utils import get_device, load_config, set_seed


def _ensure_dataset(config: dict, data_dir: Path) -> None:
    shard_files = sorted(data_dir.glob("dataset_shard_*.pt"))
    if shard_files:
        return

    print("No dataset shards found. Generating synthetic dataset...")
    stats = generate_dataset_to_disk(
        out_dir=data_dir,
        n_instances=int(config["data"].get("n_total_generate", 50_000)),
        shard_size=1000,
        grid_size=int(config["data"]["grid_size"]),
        m_min=int(config["data"]["m_min"]),
        m_max=int(config["data"]["m_max"]),
        noise_min=float(config["data"]["noise_min"]),
        noise_max=float(config["data"]["noise_max"]),
        nu_choices=tuple(config["data"]["matern_nu_choices"]),
        pde_family=str(config["data"].get("pde_family", "diffusion")),
        k_type=str(config["data"].get("k_type", "gp")),
        noise_type=str(config["data"].get("noise_type", "gaussian")),
        n_time_snapshots=int(config["data"].get("n_time_snapshots", 3)),
    )
    print(f"Generated dataset shards: {stats}")


def _build_model(config: dict) -> AmortizedInversePDEModel:
    target_fields = list(config["training"].get("target_fields", ["k_grid"]))
    return AmortizedInversePDEModel(
        grid_size=int(config["data"]["grid_size"]),
        d_model=int(config["model"]["d_model"]),
        n_heads=int(config["model"]["n_heads"]),
        n_layers=int(config["model"]["n_layers"]),
        dropout=float(config["model"]["dropout"]),
        mc_samples=int(config["model"]["mc_samples"]),
        include_time=bool(config["model"].get("include_time", False)),
        n_targets=len(target_fields),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Amortized Inverse PDE")
    parser.add_argument("--mode", choices=["train", "evaluate", "baselines"], required=True)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default="data/generated")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--results-dir", type=str, default="results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    set_seed(int(config.get("seed", 42)), deterministic=bool(config.get("deterministic", False)))
    device = get_device()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    _ensure_dataset(config, data_dir)

    train_loader, val_loader, test_loader = build_dataloaders(
        data_dir=data_dir,
        n_train=int(config["data"]["n_train"]),
        n_val=int(config["data"]["n_val"]),
        n_test=int(config["data"]["n_test"]),
        batch_size=int(config["training"]["batch_size"]),
        seed=int(config.get("seed", 42)),
        num_workers=int(config.get("num_workers", 0)),
    )

    if args.mode == "train":
        if str(config["data"].get("pde_family", "diffusion")) == "diffusion":
            print("Validating generator before training...")
            validate_generator(num_samples=100, tol=1e-3, grid_size=int(config["data"]["grid_size"]))
            print("Generator validation passed.")

        if bool(config["training"].get("sanity_enabled", False)):
            print("Running overfit sanity check...")
            sanity_model = _build_model(config).to(device)
            sanity_report = run_overfit_sanity(
                model=sanity_model,
                train_loader=train_loader,
                config=config,
                device=device,
                output_dir=Path(args.output_dir) / "sanity",
            )
            target_rmse = float(config["training"].get("sanity_target_rmse", 0.1))
            print(
                f"Sanity done: best_rmse={sanity_report['best_rmse']:.6f}, "
                f"final_rmse={sanity_report['final_rmse']:.6f}, target={target_rmse:.6f}"
            )
            if sanity_report["best_rmse"] > target_rmse:
                raise RuntimeError(
                    f"Sanity check failed: best_rmse={sanity_report['best_rmse']:.6f} > target={target_rmse:.6f}"
                )

        model = _build_model(config).to(device)

        summary = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device,
            output_dir=args.output_dir,
        )
        print(f"Training complete. Best val NLL: {summary['best_val_nll']:.6f}")

    elif args.mode == "evaluate":
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required for evaluate mode")

        model = _build_model(config).to(device)

        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

        result = run_full_evaluation(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            config=config,
            device=device,
            output_dir=args.results_dir,
            val_loader=val_loader,
        )
        save_instance_figures(model=model, test_loader=test_loader, device=device, out_dir=Path(args.results_dir) / "figures")
        if bool(config["evaluation"].get("attention_visualization", {}).get("enabled", True)):
            attn_cfg = config["evaluation"].get("attention_visualization", {})
            save_attention_figures(
                model=model,
                test_loader=test_loader,
                device=device,
                out_dir=Path(args.results_dir) / "figures" / "attention",
                n_instances=int(attn_cfg.get("n_instances", 4)),
                layer_idx=int(attn_cfg.get("layer_idx", -1)),
                head_idx=int(attn_cfg.get("head_idx", 0)),
            )
        print("Evaluation complete.")
        print(result)

    elif args.mode == "baselines":
        model = _build_model(config).to(device)
        # Baselines are run inside the full evaluation pipeline.
        result = run_full_evaluation(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            config=config,
            device=device,
            output_dir=args.results_dir,
            val_loader=val_loader,
        )
        print("Baselines complete.")
        print(result.get("baselines", {}))


if __name__ == "__main__":
    main()
