from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.gp_baseline import GPBaseline
from data.dataset import build_dataloaders
from utils import get_device, load_config, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure GP baseline inference timing.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--n-instances", type=int, default=100)
    parser.add_argument("--output", type=str, default="results_gp_timing.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)), deterministic=bool(config.get("deterministic", False)))
    device = get_device()

    train_loader, _, test_loader = build_dataloaders(
        data_dir=Path(args.data_dir),
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

    max_m = int(config["data"].get("m_max", 100)) * max(1, int(config["data"].get("n_time_snapshots", 1)))
    gp = GPBaseline(max_m=max_m, grid_size=int(config["data"]["grid_size"]), n_components=16)
    gp.fit(train_loader, max_samples=int(args.max_samples))

    times = []
    seen = 0
    for batch in test_loader:
        bsz = batch["obs_coords"].shape[0]
        for i in range(bsz):
            if seen >= args.n_instances:
                break
            obs_coords = batch["obs_coords"][i : i + 1]
            obs_values = batch["obs_values"][i : i + 1]
            t0 = time.perf_counter()
            _ = gp.predict(obs_coords, obs_values)
            times.append(time.perf_counter() - t0)
            seen += 1
        if seen >= args.n_instances:
            break

    avg_sec = float(sum(times) / max(1, len(times)))
    result = {
        "gp_avg_sec": avg_sec,
        "gp_avg_ms": avg_sec * 1000.0,
        "n_instances": int(len(times)),
        "fit_max_samples": int(args.max_samples),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
