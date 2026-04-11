#!/usr/bin/env python3
"""
Simple diagnostic: Test which nu_choices produce RMSE closest to 0.2958
by analyzing the k_grid statistics generated under different nu values.
"""

import json
import sys
from pathlib import Path
from typing import Tuple

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.generator import generate_instance

GRID_SIZE = 32
DEVICE = torch.device("cpu")


def test_nu_variation(nu_choices: Tuple[float, ...], n_samples: int = 48) -> dict:
    """Test RMS statistics of generated k_grids for given nu_choices."""
    
    print(f"\nTesting nu_choices={nu_choices} ({n_samples} samples)...")
    
    k_rms_values = []
    for i in range(n_samples):
        if (i + 1) % 16 == 0:
            print(f"  {i + 1}/{n_samples}...")
        
        sample = generate_instance(
            grid_size=GRID_SIZE,
            m_min=20,
            m_max=100,
            noise_min=1e-3,
            noise_max=5e-2,
            nu_choices=nu_choices,
            pde_family="diffusion",
            k_type="gp",
            noise_type="gaussian",
            fast_gp=False,
            device=DEVICE,
        )
        
        k_grid = sample["k_grid"].flatten()
        # RMS of k around the mean
        k_rms = torch.sqrt(((k_grid - k_grid.mean()) ** 2).mean()).item()
        k_rms_values.append(k_rms)
    
    # Stats
    k_rms_tensor = torch.tensor(k_rms_values)
    
    result = {
        "nu_choices": list(nu_choices),
        "n_samples": n_samples,
        "k_rms_mean": float(k_rms_tensor.mean().item()),
        "k_rms_std": float(k_rms_tensor.std(unbiased=False).item()),
        "k_rms_min": float(k_rms_tensor.min().item()),
        "k_rms_max": float(k_rms_tensor.max().item()),
    }
    
    print(f"  k_rms: {result['k_rms_mean']:.6f} ± {result['k_rms_std']:.6f}")
    return result


def main():
    """Run diagnostic of nu parameter variations."""
    
    print("="*70)
    print("Generator Parameter Diagnostic")
    print("="*70)
    print("Testing which nu_choices configuration generates k_grids")
    print("with statistics matching the paper's expected behavior")
    print()
    
    # Test configurations
    configs = [
        (0.5,),           # Very rough
        (1.5,),           # Rough
        (2.5,),           # Smooth
        (0.5, 1.5),       # Rough mix
        (1.5, 2.5),       # Current config
        (0.5, 1.5, 2.5),  # All three
    ]
    
    results = []
    for nu_choices in configs:
        result = test_nu_variation(nu_choices, n_samples=48)
        results.append(result)
    
    # Save results
    output_dir = Path(__file__).parent.parent / "outputs_recovery_gpu_fast_stable"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "generator_param_diagnostic.json"
    
    with open(output_file, "w") as f:
        json.dump({
            "diagnostic": "Generator nu_choices variations",
            "grid_size": GRID_SIZE,
            "results": results,
        }, f, indent=2)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for result in results:
        nu_str = str(result['nu_choices'])
        print(f"nu={nu_str:20s} -> k_rms={result['k_rms_mean']:.6f}±{result['k_rms_std']:.6f}")
    
    print(f"\nResults saved to: {output_file}")
    
    # Analysis
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)
    print("""
This diagnostic shows k_rms (RMS variation) of generated coefficient fields
for each nu_choices configuration. 

Key insight: Different nu values control the SMOOTHNESS of generated fields:
- nu=0.5 (exponential kernel): Very rough, high variation
- nu=1.5 (Matern 3/2 kernel): Medium smoothness
- nu=2.5 (Matern 5/2 kernel): Very smooth, low variation

The paper's 0.2958 RMSE claim might have used a configuration with:
- Different weighting of nu values (e.g., more nu=0.5 roughness)
- Different observation counts (m_min, m_max)
- Different noise levels

Next step: Use these k_rms values to correlate with model RMSE performance.
The configuration with k_rms closest to a target value might match the paper.
    """)


if __name__ == "__main__":
    main()
