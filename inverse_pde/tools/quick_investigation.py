#!/usr/bin/env python
"""
Quick Discrepancy Investigation - Phase 1

Focuses on the most critical issues:
1. ECE calibration: improved 0.258→0.0495 — is temperature scaling applied?
2. OOD few_observations: RMSE increased 0.330→0.372 — what M value is used?
3. Baseline ECE values: GP ECE improved dramatically — real or test set change?

This is a fast probe to identify the root causes before full controlled comparison.
"""

import os
import sys
import json
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.model import AmortizedInversePDEModel
from data.generator import generate_instance
from training.metrics import batch_metrics


def quick_ece_temperature_probe(
    checkpoint_path,
    config_path,
    n_instances=50,
    device=None,
):
    """
    Quick test: is ECE improvement due to temperature scaling?
    
    Checks sigma values and ECE calibration before any scaling.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("\n" + "="*70)
    print("QUICK ECE CALIBRATION INVESTIGATION")
    print("="*70)
    
    # Load model
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = config.get('model', {})
    model = AmortizedInversePDEModel(
        d_model=model_config.get('d_model', 96),
        n_layers=model_config.get('n_layers', 3),
        n_heads=model_config.get('n_heads', 4),
        dropout=model_config.get('dropout', 0.1),
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"\nTesting ECE over {n_instances} in-distribution test instances...")
    
    rmse_list = []
    ece_list = []
    coverage_list = []
    sigma_mean_list = []
    
    for idx in range(n_instances):
        np.random.seed(100 + idx)
        torch.manual_seed(100 + idx)
        
        sample = generate_instance(
            grid_size=32,
            k_type='gp',
            noise_type='gaussian',
            m_min=20,
            m_max=100,
        )
        
        obs_coords = sample['obs_coords'].to(device).float() if torch.is_tensor(sample['obs_coords']) else torch.from_numpy(sample['obs_coords']).to(device).float()
        obs_values = sample['obs_values'].to(device).float() if torch.is_tensor(sample['obs_values']) else torch.from_numpy(sample['obs_values']).to(device).float()
        k_true = sample['k_grid'].to(device).float() if torch.is_tensor(sample['k_grid']) else torch.from_numpy(sample['k_grid']).to(device).float()
        
        with torch.no_grad():
            obs_times_tensor = sample['obs_times'].to(device).float() if torch.is_tensor(sample['obs_times']) else torch.from_numpy(sample['obs_times']).to(device).float() if isinstance(sample['obs_times'], np.ndarray) else torch.zeros(obs_coords.shape[0]).to(device).float()
            # Model expects (batch, n_obs, feature_dim)
            mu, log_sigma = model(
                obs_coords.unsqueeze(0),  # (1, n_obs, 2)
                obs_times_tensor.unsqueeze(0).unsqueeze(-1),  # (1, n_obs, 1)
                obs_values.unsqueeze(0).unsqueeze(-1),  # (1, n_obs, 1)
            )
            sigma = torch.exp(log_sigma)
        
        mu_flat = mu.squeeze(0).ravel()
        sigma_flat = sigma.squeeze(0).ravel()
        k_true_flat = k_true.ravel()
        
        # Record metrics
        rmse_val = float(torch.sqrt(torch.mean((mu_flat - k_true_flat) ** 2)))
        rmse_list.append(rmse_val)
        
        sigma_mean_list.append(float(torch.mean(sigma_flat).item()))
        
        # ECE without any adjustments
        metrics = batch_metrics(mu_flat, sigma_flat, k_true_flat)
        ece_list.append(metrics['ece'])
        coverage_list.append(metrics['coverage'])
    
    # Results
    result = {
        'rmse_mean': float(np.mean(rmse_list)),
        'rmse_std': float(np.std(rmse_list)),
        'ece_mean': float(np.mean(ece_list)),
        'ece_std': float(np.std(ece_list)),
        'coverage_mean': float(np.mean(coverage_list)),
        'sigma_mean_across_instances': float(np.mean(sigma_mean_list)),
        'n_instances': n_instances,
        'note': 'No temperature scaling applied. If ECE>>0.258, scaling was applied in original paper baseline.',
    }
    
    print(f"\n{'Metric':<30} {'Value':>12}")
    print("-" * 42)
    print(f"{'RMSE':<30} {result['rmse_mean']:>12.4f}")
    print(f"{'ECE (no temp scale)':<30} {result['ece_mean']:>12.4f}")
    print(f"{'Coverage (90%)':<30} {result['coverage_mean']:>12.4f}")
    print(f"{'Mean predicted sigma':<30} {result['sigma_mean_across_instances']:>12.4f}")
    print("="*70)
    print("\nInterpretation:")
    print(f"- Paper reported ECE=0.258, we measure ECE={result['ece_mean']:.4f}")
    if result['ece_mean'] < 0.15:
        print("  → Model is better calibrated now (real improvement)")
    else:
        print("  → Similar to paper. Changes are in evaluation regime or test set.")
    
    return result


def quick_ood_few_observations_probe(
    checkpoint_path,
    config_path,
    device=None,
):
    """
    Quick test: check few_observations performance across M values.
    
    Paper said 0.330 RMSE. Recovery shows 0.372.
    Test which M value(s) match which result.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("\n" + "="*70)
    print("FEW_OBSERVATIONS OOD QUICK INVESTIGATION")
    print("="*70)
    
    # Load model
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = config.get('model', {})
    model = AmortizedInversePDEModel(
        d_model=model_config.get('d_model', 96),
        n_layers=model_config.get('n_layers', 3),
        n_heads=model_config.get('n_heads', 4),
        dropout=model_config.get('dropout', 0.1),
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    results_by_m = {}
    m_values = [5, 10, 15, 20, 25, 50, 100]
    
    print(f"\nTesting RMSE vs number of observations M...")
    print(f"Paper claimed: few_observations RMSE=0.330 (M=?)")
    print(f"Recovery measured: RMSE=0.372 (M=?)\n")
    
    for m_obs in m_values:
        rmse_list = []
        
        for idx in range(20):
            np.random.seed(200 + idx)
            torch.manual_seed(200 + idx)
            
            # Generate with exact M
            sample = generate_instance(
                grid_size=32,
                k_type='gp',
                noise_type='gaussian',
                m_min=m_obs,
                m_max=m_obs,
            )
            
            obs_coords = sample['obs_coords'].to(device).float() if torch.is_tensor(sample['obs_coords']) else torch.from_numpy(sample['obs_coords']).to(device).float()
            obs_values = sample['obs_values'].to(device).float() if torch.is_tensor(sample['obs_values']) else torch.from_numpy(sample['obs_values']).to(device).float()
            k_true = sample['k_grid'].to(device).float() if torch.is_tensor(sample['k_grid']) else torch.from_numpy(sample['k_grid']).to(device).float()
            
            with torch.no_grad():
                obs_times_tensor = sample['obs_times'].to(device).float() if torch.is_tensor(sample['obs_times']) else torch.from_numpy(sample['obs_times']).to(device).float() if isinstance(sample['obs_times'], np.ndarray) else torch.zeros(obs_coords.shape[0]).to(device).float()
                # Model expects (batch, n_obs, feature_dim)
                mu, _ = model(
                    obs_coords.unsqueeze(0),  # (1, n_obs, 2)
                    obs_times_tensor.unsqueeze(0).unsqueeze(-1),  # (1, n_obs, 1)
                    obs_values.unsqueeze(0).unsqueeze(-1),  # (1, n_obs, 1)
                )
            
            rmse_val = float(torch.sqrt(torch.mean((mu.squeeze(0).ravel() - k_true.ravel()) ** 2)))
            rmse_list.append(rmse_val)
        
        mean_rmse = float(np.mean(rmse_list))
        std_rmse = float(np.std(rmse_list))
        results_by_m[m_obs] = {'mean': mean_rmse, 'std': std_rmse}
        
        match_330 = "  ← matches paper claim (0.330)" if abs(mean_rmse - 0.330) < 0.02 else ""
        match_372 = "  ← matches recovery result (0.372)" if abs(mean_rmse - 0.372) < 0.02 else ""
        
        print(f"M={m_obs:3d}: RMSE={mean_rmse:.4f}±{std_rmse:.4f}{match_330}{match_372}")
    
    print("\n" + "="*70)
    print("Interpretation:")
    print("- RMSE improves as M increases (more observations)")
    print("- Find which M matches paper (0.330) vs recovery (0.372)")
    print("- Likely: test set uses different M distribution, or OOD test set changed")
    
    return results_by_m


def main():
    parser_args = [
        ('--checkpoint', dict(required=True, help="Checkpoint path")),
        ('--config', dict(required=True, help="Config YAML")),
        ('--output-dir', dict(required=True, help="Output directory")),
    ]
    
    import argparse
    parser = argparse.ArgumentParser(description="Quick Discrepancy Investigation")
    for arg_name, arg_opts in parser_args:
        parser.add_argument(arg_name, **arg_opts)
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # Test 1: ECE calibration
    try:
        ece_results = quick_ece_temperature_probe(args.checkpoint, args.config, device=device)
        results['ece_calibration'] = ece_results
    except Exception as e:
        print(f"\n[ERROR] ECE probe failed: {e}")
        import traceback
        traceback.print_exc()
        results['ece_calibration'] = {'error': str(e)}
    
    # Test 2: OOD few observations
    try:
        ood_results = quick_ood_few_observations_probe(args.checkpoint, args.config, device=device)
        results['ood_few_observations'] = ood_results
    except Exception as e:
        print(f"\n[ERROR] OOD probe failed: {e}")
        import traceback
        traceback.print_exc()
        results['ood_few_observations'] = {'error': str(e)}
    
    # Save results
    output_file = output_dir / 'quick_investigation_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[OK] Quick investigation saved to {output_file}\n")


if __name__ == '__main__':
    main()
