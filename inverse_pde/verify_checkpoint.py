import glob
from pathlib import Path

import torch

from model.model import AmortizedInversePDEModel


def _find_checkpoint() -> str:
    candidates = [
        "outputs_full_run_d96_restore_check/checkpoints/*.pt",
        "outputs_recovery_gpu_fast_stable/checkpoints/*.pt",
        "outputs_recovery_gpu_fast/checkpoints/*.pt",
        "outputs_recovery_gpu/checkpoints/*.pt",
    ]
    for pattern in candidates:
        paths = sorted(glob.glob(pattern))
        if paths:
            return paths[0]
    raise FileNotFoundError("No checkpoint files found in known output folders")


def main() -> None:
    ckpt_path = _find_checkpoint()
    print(f"Using checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    model = AmortizedInversePDEModel(
        grid_size=32,
        d_model=96,
        n_heads=4,
        n_layers=3,
        dropout=0.15,
        mc_samples=50,
        include_time=False,
        n_targets=1,
    )

    result = model.load_state_dict(ckpt["model_state_dict"], strict=True)
    print("Load result:", result)

    model.eval()
    obs_coords = torch.rand(1, 50, 2)
    obs_values = torch.randn(1, 50, 1)
    mask = torch.zeros(1, 50, dtype=torch.bool)

    with torch.no_grad():
        mu, sigma = model(
            obs_coords=obs_coords,
            obs_times=None,
            obs_values=obs_values,
            obs_key_padding_mask=mask,
        )

    print("mu shape:", tuple(mu.shape))
    print("sigma shape:", tuple(sigma.shape))
    print("mu range:", float(mu.min().item()), "to", float(mu.max().item()))
    print("sigma range:", float(sigma.min().item()), "to", float(sigma.max().item()))
    print("SUCCESS: Model loads and runs correctly")


if __name__ == "__main__":
    main()
