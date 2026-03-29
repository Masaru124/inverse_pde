from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader, Dataset


class InversePDEDataset(Dataset):
    def __init__(self, shards: List[str | Path]):
        self.samples: List[dict] = []
        for shard in shards:
            loaded = torch.load(shard, map_location="cpu")
            if not isinstance(loaded, list):
                raise ValueError(f"Expected shard to contain a list of samples, got {type(loaded)}")
            self.samples.extend(loaded)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        obs_coords = sample["obs_coords"].float()
        obs_values = sample["obs_values"].float()
        obs_times = sample.get("obs_times", torch.zeros(obs_coords.shape[0])).float()
        k_grid = sample["k_grid"].float()
        r_grid = sample.get("r_grid", torch.zeros_like(k_grid)).float()
        return {
            "obs_coords": obs_coords,
            "obs_times": obs_times,
            "obs_values": obs_values,
            "k_grid": k_grid,
            "r_grid": r_grid,
            "f_grid": sample["f_grid"].float(),
            "u_grid": sample["u_grid"].float(),
            "pde_family": str(sample.get("pde_family", "diffusion")),
            "k_type": str(sample.get("k_type", "gp")),
            "noise_type": str(sample.get("noise_type", "gaussian")),
            "noise_sigma": float(sample.get("noise_sigma", 0.0)),
        }


def collate_variable_observations(batch: List[dict]) -> Dict[str, torch.Tensor]:
    batch_size = len(batch)
    lengths = torch.tensor([item["obs_coords"].shape[0] for item in batch], dtype=torch.long)
    max_len = int(lengths.max().item())

    obs_coords = torch.zeros(batch_size, max_len, 2, dtype=torch.float32)
    obs_times = torch.zeros(batch_size, max_len, 1, dtype=torch.float32)
    obs_values = torch.zeros(batch_size, max_len, 1, dtype=torch.float32)
    key_padding_mask = torch.ones(batch_size, max_len, dtype=torch.bool)

    for i, item in enumerate(batch):
        n = item["obs_coords"].shape[0]
        obs_coords[i, :n] = item["obs_coords"]
        obs_times[i, :n, 0] = item["obs_times"]
        obs_values[i, :n, 0] = item["obs_values"]
        key_padding_mask[i, :n] = False

    k_grid = torch.stack([item["k_grid"] for item in batch], dim=0)
    r_grid = torch.stack([item["r_grid"] for item in batch], dim=0)
    f_grid = torch.stack([item["f_grid"] for item in batch], dim=0)
    u_grid = torch.stack([item["u_grid"] for item in batch], dim=0)
    pde_family = [item["pde_family"] for item in batch]
    k_type = [item["k_type"] for item in batch]
    noise_type = [item["noise_type"] for item in batch]
    noise_sigma = torch.tensor([item["noise_sigma"] for item in batch], dtype=torch.float32)

    return {
        "obs_coords": obs_coords,
        "obs_times": obs_times,
        "obs_values": obs_values,
        "obs_key_padding_mask": key_padding_mask,
        "obs_lengths": lengths,
        "k_grid": k_grid,
        "r_grid": r_grid,
        "f_grid": f_grid,
        "u_grid": u_grid,
        "pde_family": pde_family,
        "k_type": k_type,
        "noise_type": noise_type,
        "noise_sigma": noise_sigma,
    }


def _sorted_shards(data_dir: str | Path) -> List[Path]:
    data_path = Path(data_dir)
    shards = sorted(data_path.glob("dataset_shard_*.pt"))
    if not shards:
        raise FileNotFoundError(f"No shard files found in {data_path}")
    return shards


def build_splits_from_shards(data_dir: str | Path, n_train: int, n_val: int, n_test: int, seed: int = 42):
    shards = _sorted_shards(data_dir)
    dataset = InversePDEDataset(shards)

    total = n_train + n_val + n_test
    if len(dataset) < total:
        raise ValueError(f"Dataset has {len(dataset)} samples but split requires {total}")

    train_set, val_set, test_set = torch.utils.data.random_split(
        dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(seed),
    )
    return train_set, val_set, test_set


def build_dataloaders(
    data_dir: str | Path,
    n_train: int,
    n_val: int,
    n_test: int,
    batch_size: int,
    seed: int = 42,
    num_workers: int = 0,
):
    train_set, val_set, test_set = build_splits_from_shards(
        data_dir=data_dir,
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        seed=seed,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_variable_observations,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_variable_observations,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_variable_observations,
    )

    return train_loader, val_loader, test_loader
