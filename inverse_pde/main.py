from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from data.dataset import build_dataloaders
from data.generator import generate_dataset_to_disk, validate_generator
from evaluation.evaluate import run_full_evaluation
from evaluation.visualize import save_attention_figures, save_instance_figures
from model.model import AmortizedInversePDEModel
from training.train import run_overfit_sanity, train_model
from utils import get_device, load_config, set_seed


def _count_samples_in_shards(shard_files: list[Path]) -> int:
    total = 0
    for shard in shard_files:
        loaded = torch.load(shard, map_location="cpu", weights_only=True)
        if not isinstance(loaded, list):
            raise ValueError(f"Expected shard to contain a list of samples, got {type(loaded)} in {shard}")
        total += len(loaded)
    return total


def _build_expected_dataset_spec(config: dict, expected: int, required: int) -> dict:
    data_cfg = config["data"]
    return {
        "expected_samples": int(expected),
        "required_split_samples": int(required),
        "pde_family": str(data_cfg.get("pde_family", "diffusion")),
        "k_type": str(data_cfg.get("k_type", "gp")),
        "noise_type": str(data_cfg.get("noise_type", "gaussian")),
        "grid_size": int(data_cfg["grid_size"]),
        "m_min": int(data_cfg["m_min"]),
        "m_max": int(data_cfg["m_max"]),
        "noise_min": float(data_cfg["noise_min"]),
        "noise_max": float(data_cfg["noise_max"]),
        "matern_nu_choices": list(data_cfg["matern_nu_choices"]),
        "n_time_snapshots": int(data_cfg.get("n_time_snapshots", 3)),
    }


def _load_dataset_manifest(manifest_path: Path) -> dict | None:
    if not manifest_path.exists():
        return None
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _write_dataset_manifest(manifest_path: Path, spec: dict) -> None:
    manifest_path.write_text(json.dumps(spec, indent=2, sort_keys=True), encoding="utf-8")


def _dataset_manifest_matches(actual: dict | None, expected: dict) -> bool:
    if actual is None:
        return False
    for key, value in expected.items():
        if actual.get(key) != value:
            return False
    return True


def _ensure_dataset(config: dict, data_dir: Path) -> None:
    required = int(config["data"]["n_train"]) + int(config["data"]["n_val"]) + int(config["data"]["n_test"])
    target_generate = int(config["data"].get("n_total_generate", required))
    expected = max(required, target_generate)
    manifest_path = data_dir / "dataset_manifest.json"
    expected_manifest = _build_expected_dataset_spec(config=config, expected=expected, required=required)

    shard_files = sorted(data_dir.glob("dataset_shard_*.pt"))
    if shard_files:
        existing = _count_samples_in_shards(shard_files)
        manifest = _load_dataset_manifest(manifest_path)
        manifest_matches = _dataset_manifest_matches(actual=manifest, expected=expected_manifest)
        if existing == expected and manifest_matches:
            print(f"Found existing dataset shards with {existing} samples (required: {required}, target: {expected}).")
            return
        reason_parts = []
        if existing != expected:
            reason_parts.append(f"sample count {existing} != expected {expected}")
        if not manifest_matches:
            reason_parts.append("dataset manifest missing or does not match current config")
        print(f"Existing dataset is incompatible ({'; '.join(reason_parts)}). Regenerating dataset...")
        for shard in shard_files:
            shard.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)
    else:
        print("No dataset shards found. Generating synthetic dataset...")

    stats = generate_dataset_to_disk(
        out_dir=data_dir,
        n_instances=expected,
        shard_size=int(config["data"].get("generation_shard_size", 5000)),
        generate_batch_size=int(config["data"].get("generation_batch_size", 32)),
        progress_interval=int(config["data"].get("generation_progress_interval", 1000)),
        forcing_mode=str(config["data"].get("generation_forcing_mode", "finite_diff")),
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
    _write_dataset_manifest(manifest_path=manifest_path, spec=expected_manifest)
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
    parser.add_argument(
        "--calibration-only",
        action="store_true",
        help="Run a fast evaluation pass that performs temperature calibration and main-model metrics only.",
    )
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
        pin_memory=bool(config.get("pin_memory", device.type == "cuda")),
        persistent_workers=bool(config.get("persistent_workers", int(config.get("num_workers", 0)) > 0)),
        prefetch_factor=int(config.get("prefetch_factor", 2)),
    )

    if args.mode == "train":
        if str(config["data"].get("pde_family", "diffusion")) == "diffusion" and bool(
            config["training"].get("generator_validation_enabled", True)
        ):
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
        if "best_metric_name" in summary and "best_metric_value" in summary:
            print(f"Training complete. Best {summary['best_metric_name']}: {summary['best_metric_value']:.6f}")
        else:
            print(f"Training complete. Best val NLL: {summary['best_val_nll']:.6f}")

    elif args.mode == "evaluate":
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required for evaluate mode")

        model = _build_model(config).to(device)

        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])

        result = run_full_evaluation(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            config=config,
            device=device,
            output_dir=args.results_dir,
            val_loader=val_loader,
            calibration_only=bool(args.calibration_only),
        )
        if not bool(args.calibration_only):
            save_instance_figures(
                model=model,
                test_loader=test_loader,
                device=device,
                out_dir=Path(args.results_dir) / "figures",
            )
        if (not bool(args.calibration_only)) and bool(config["evaluation"].get("attention_visualization", {}).get("enabled", True)):
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
            calibration_only=bool(args.calibration_only),
        )
        print("Baselines complete.")
        print(result.get("baselines", {}))


if __name__ == "__main__":
    main()
