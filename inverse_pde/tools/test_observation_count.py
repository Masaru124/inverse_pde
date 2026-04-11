#!/usr/bin/env python3
"""
Test observation count (m_min) variations to resolve 5-seed RMSE discrepancy.

Paper claims: RMSE = 0.2958 ± 0.0008 (5 seeds × 48 samples)
Recovery shows: RMSE = 0.3400 ± 0.0162
Gap: 0.1442 RMSE

Hypothesis: Paper used higher m_min (fewer observations per instance)
causing harder inference problem, potentially matching observed RMSE.
"""

import json
import sys
import time
from pathlib import Path
from typing import Tuple

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.generator import generate_instance
from training.metrics import batch_metrics
from utils import set_seed

# Config
CHECKPOINT_PATH = Path(__file__).parent.parent / "outputs_recovery_gpu_fast_stable" / "checkpoints" / "epoch_avg_006_014_024.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GRID_SIZE = 32

def load_model():
    """Load the trained model."""
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")
    
    try:
        from model.model import AmortizedInversePDEModel
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model_state = checkpoint.get("model_state", checkpoint.get("model"))
        
        model = AmortizedInversePDEModel()
        if isinstance(model_state, dict):
            model.load_state_dict(model_state)
        model.to(DEVICE)
        model.eval()
        return model
    except Exception as e:
        print(f"Warning: Could not load model: {e}")
        return None


def test_observation_count(m_min: int, m_max: int = 100, n_samples: int = 48) -> dict:
    """Test RMSE with specific observation count range."""
    
    print(f"\n{'='*70}")
    print(f"Testing m_min={m_min}, m_max={m_max} ({n_samples} samples)")
    print(f"{'='*70}")
    
    # Load model
    model = load_model()
    if model is None:
        raise RuntimeError("Could not load model")
    
    rmse_vals = []
    ece_vals = []
    start = time.time()
    
    for i in range(n_samples):
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            remaining = (n_samples - i - 1) / rate
            print(f"  Processed {i+1}/{n_samples} ({elapsed:.1f}s, ~{remaining:.0f}s remaining)")
        
        # Generate instance with this specific m_min
        sample = generate_instance(
            grid_size=GRID_SIZE,
            m_min=m_min,
            m_max=m_max,
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
        "m_min": m_min,
        "m_max": m_max,
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
    print(f"  Time:        {elapsed_total:.1f}s ({n_samples/elapsed_total:.2f} samples/sec)")
    
    # Compare to paper
    paper_rmse = 0.2958
    paper_std = 0.0008
    diff = abs(result['rmse_mean'] - paper_rmse)
    tolerance = 0.01  # Within 1%
    
    print(f"\n  Paper target: RMSE = {paper_rmse:.6f} ± {paper_std:.6f}")
    print(f"  Difference:   {diff:.6f} {'✓ MATCH (< 0.01)' if diff < tolerance else '✗ No match'}")
    
    return result


def main():
    """Run observation count investigation."""
    
    print("="*70)
    print("Five-Seed RMSE Investigation: Observation Count Sweep")
    print("="*70)
    print(f"Paper claim:    RMSE = 0.2958 ± 0.0008 (5 seeds × 48 samples)")
    print(f"Recovery shows: RMSE = 0.3400 ± 0.0162")
    print(f"Device: {DEVICE}")
    print(f"Hypothesis: Paper used higher m_min (fewer observations)")
    
    # Test m_min variations
    m_min_values = [20, 30, 50]  # 20 is current, 30/50 are hypothesized paper values
    results = []
    
    for m_min in m_min_values:
        try:
            result = test_observation_count(m_min=m_min, m_max=100, n_samples=48)
            results.append(result)
        except Exception as e:
            print(f"ERROR testing m_min={m_min}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary table
    print("\n" + "="*70)
    print("Summary: Observation Count Sweep Results")
    print("="*70)
    print(f"{'m_min':<6} {'RMSE Mean':<12} {'RMSE Std':<12} {'Diff from Paper':<16} {'Match?':<8}")
    print("-" * 70)
    
    for r in results:
        m_min = r["m_min"]
        rmse_mean = r["rmse_mean"]
        rmse_std = r["rmse_std"]
        diff = abs(rmse_mean - 0.2958)
        match = "✓ YES" if diff < 0.01 else "✗ NO"
        
        print(f"{m_min:<6} {rmse_mean:<12.6f} {rmse_std:<12.6f} {diff:<16.6f} {match:<8}")
    
    # Save results
    output_file = Path(__file__).parent.parent / "observation_count_sweep_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")
    
    # Determine best match
    best_match = min(results, key=lambda r: abs(r["rmse_mean"] - 0.2958))
    print(f"\nBest match: m_min={best_match['m_min']} with RMSE={best_match['rmse_mean']:.6f}")
    
    if abs(best_match["rmse_mean"] - 0.2958) < 0.01:
        print("✓ DISCREPANCY RESOLVED! Paper likely used m_min={}".format(best_match["m_min"]))
    else:
        print("✗ Observation count alone does not explain the discrepancy.")
        print("Next: Test noise_max variations")


if __name__ == "__main__":
    main()
