#!/usr/bin/env python3
"""
Investigate checkpoint selection impact on 5-seed RMSE.

Hypothesis: Paper may have used a single-epoch checkpoint (e.g., epoch 6, 14, or 24)
instead of the epoch-averaged checkpoint (epoch_avg_006_014_024.pt).

Results from previous sweeps show that neither m_min nor noise_max alone 
explains the 0.2958 RMSE claim. The checkpoint may be the key difference.
"""

import json
import sys
import time
from pathlib import Path
from typing import Optional

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.generator import generate_instance
from training.metrics import batch_metrics
from utils import set_seed

# Config
CHECKPOINT_DIR = Path(__file__).parent.parent / "outputs_recovery_gpu_fast_stable" / "checkpoints"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GRID_SIZE = 32

def load_specific_checkpoint(checkpoint_file: str):
    """Load a specific checkpoint file."""
    checkpoint_path = CHECKPOINT_DIR / checkpoint_file
    
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        return None
    
    try:
        from model.model import AmortizedInversePDEModel
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model_state = checkpoint.get("model_state", checkpoint.get("model"))
        
        model = AmortizedInversePDEModel()
        if isinstance(model_state, dict):
            model.load_state_dict(model_state)
        model.to(DEVICE)
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading {checkpoint_path}: {e}")
        return None


def test_checkpoint(checkpoint_file: str, n_samples: int = 48) -> Optional[dict]:
    """Test RMSE with a specific checkpoint."""
    
    print(f"\n{'='*70}")
    print(f"Testing checkpoint: {checkpoint_file}")
    print(f"{'='*70}")
    
    # Load checkpoint
    model = load_specific_checkpoint(checkpoint_file)
    if model is None:
        print(f"✗ Failed to load checkpoint")
        return None
    
    rmse_vals = []
    ece_vals = []
    start = time.time()
    
    for i in range(n_samples):
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            remaining = (n_samples - i - 1) / rate
            print(f"  Processed {i+1}/{n_samples} ({elapsed:.1f}s, ~{remaining:.0f}s remaining)")
        
        # Generate instance
        sample = generate_instance(
            grid_size=GRID_SIZE,
            m_min=20,
            m_max=100,
            noise_min=1e-3,
            noise_max=5e-2,
            nu_choices=(1.5, 2.5),
            pde_family="diffusion",
            k_type="gp",
            noise_type="gaussian",
            fast_gp=False,
            device=DEVICE,
        )
        
        # Prepare inputs
        obs_coords = sample["obs_coords"].unsqueeze(0).to(DEVICE)
        obs_values = sample["obs_values"].unsqueeze(0).unsqueeze(-1).to(DEVICE)
        target = sample["k_grid"].unsqueeze(0).to(DEVICE)
        mask = torch.zeros(1, obs_coords.shape[1], dtype=torch.bool, device=DEVICE)
        
        try:
            with torch.no_grad():
                mu, sigma = model(obs_coords, None, obs_values, mask, mc_dropout=False)
            
            # Compute metrics
            m = batch_metrics(mu=mu, sigma=sigma, target=target, ece_bins=10, coverage_level=0.9)
            rmse_vals.append(float(m["rmse"]))
            ece_vals.append(float(m["ece"]))
        except Exception as e:
            print(f"  Error on sample {i+1}: {e}")
            rmse_vals.append(1.0)
            ece_vals.append(1.0)
    
    # Compute statistics
    rmse_tensor = torch.tensor(rmse_vals, dtype=torch.float32)
    ece_tensor = torch.tensor(ece_vals, dtype=torch.float32)
    
    elapsed_total = time.time() - start
    
    result = {
        "checkpoint": checkpoint_file,
        "n_samples": n_samples,
        "rmse_mean": float(rmse_tensor.mean().item()),
        "rmse_std": float(rmse_tensor.std(unbiased=False).item()),
        "rmse_min": float(rmse_tensor.min().item()),
        "rmse_max": float(rmse_tensor.max().item()),
        "ece_mean": float(ece_tensor.mean().item()),
        "elapsed_seconds": elapsed_total,
    }
    
    # Print summary
    print(f"\nResults:")
    print(f"  RMSE:        {result['rmse_mean']:.6f} ± {result['rmse_std']:.6f}")
    print(f"  RMSE range:  [{result['rmse_min']:.6f}, {result['rmse_max']:.6f}]")
    print(f"  ECE:         {result['ece_mean']:.6f}")
    print(f"  Time:        {elapsed_total:.1f}s")
    
    # Compare to paper
    paper_rmse = 0.2958
    diff = abs(result['rmse_mean'] - paper_rmse)
    print(f"\n  Paper target: RMSE = {paper_rmse:.6f}")
    print(f"  Difference:   {diff:.6f} {'✓ MATCH' if diff < 0.01 else '✗ No match'}")
    
    return result


def list_available_checkpoints():
    """List all available checkpoints in the directory."""
    if not CHECKPOINT_DIR.exists():
        print(f"Checkpoint directory not found: {CHECKPOINT_DIR}")
        return []
    
    checkpoints = sorted(CHECKPOINT_DIR.glob("*.pt"))
    print(f"Available checkpoints in {CHECKPOINT_DIR}:")
    for cp in checkpoints:
        print(f"  - {cp.name}")
    return [cp.name for cp in checkpoints]


def main():
    """Run checkpoint investigation."""
    
    print("="*70)
    print("Five-Seed RMSE Investigation: Checkpoint Selection")
    print("="*70)
    print(f"Paper claim:    RMSE = 0.2958 ± 0.0008 (5 seeds × 48 samples)")
    print(f"Recovery shows: RMSE = 0.3400 ± 0.0162")
    print(f"Device: {DEVICE}")
    print(f"Hypothesis: Paper used single-epoch checkpoint, not epoch-averaged")
    
    # List and test available checkpoints
    checkpoints = list_available_checkpoints()
    
    print(f"\nTesting {len(checkpoints)} checkpoint(s)...")
    results = []
    
    for checkpoint in checkpoints:
        result = test_checkpoint(checkpoint, n_samples=48)
        if result is not None:
            results.append(result)
    
    # Summary table
    print("\n" + "="*70)
    print("Summary: Checkpoint Comparison Results")
    print("="*70)
    print(f"{'Checkpoint':<40} {'RMSE Mean':<12} {'Diff':<12} {'Match?':<8}")
    print("-" * 70)
    
    for r in results:
        checkpoint = r["checkpoint"]
        rmse_mean = r["rmse_mean"]
        diff = abs(rmse_mean - 0.2958)
        match = "✓ YES" if diff < 0.01 else "✗ NO"
        
        print(f"{checkpoint:<40} {rmse_mean:<12.6f} {diff:<12.6f} {match:<8}")
    
    # Save results
    output_file = Path(__file__).parent.parent / "checkpoint_sweep_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")
    
    if results:
        best_match = min(results, key=lambda r: abs(r["rmse_mean"] - 0.2958))
        print(f"\nBest match: {best_match['checkpoint']} with RMSE={best_match['rmse_mean']:.6f}")
        
        if abs(best_match["rmse_mean"] - 0.2958) < 0.01:
            print("✓ DISCREPANCY RESOLVED! Paper likely used this checkpoint")
        else:
            print("✗ Checkpoint selection alone does not explain the discrepancy")


if __name__ == "__main__":
    main()
