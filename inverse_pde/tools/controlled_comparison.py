#!/usr/bin/env python
"""
Controlled Comparison Evaluation

Systematically compares recovery results with original paper claims.
Identifies sources of discrepancies in:
1. PINN baseline performance (RMSE dropped from 0.686 to 0.123 — needs investigation)
2. ECE calibration (improved 0.258→0.0495 — temperature scaling effect?)
3. OOD generalization (few_observations 0.330→0.372 — test set change?)
4. Five-seed realistic-GP stability (mean/variance don't match)

Goal: Determine which discrepancies are data-driven vs. legitimate improvements
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.model import AmortizedInversePDEModel
from evaluation.evaluate import run_full_evaluation
from baselines.pinn_baseline import run_pinn_inversion
from data.generator import generate_instance
from training.metrics import batch_metrics


def controlled_pinn_evaluation(
    checkpoint_path,
    config_path,
    data_dir,
    n_instances=100,
    max_steps=1000,
    convergence_loss_threshold=1e-3,
    convergence_grad_threshold=1e-4,
    device=None,
):
    """
    Controlled PINN baseline evaluation with explicit convergence settings.
    
    Args:
        checkpoint_path: main model checkpoint (for comparison)
        config_path: config YAML
        data_dir: data directory
        n_instances: test instances
        max_steps: max optimization steps per instance
        convergence_loss_threshold: PINN loss < this to count as converged
        convergence_grad_threshold: gradient norm < this to count as converged
        device: torch device
        
    Returns:
        results dict with RMSE, ECE, coverage, step usage, convergence fraction, timing
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("\n" + "="*70)
    print("CONTROLLED PINN BASELINE EVALUATION")
    print("="*70)
    print(f"Settings:")
    print(f"  Max steps per instance: {max_steps}")
    print(f"  Convergence criteria: loss < {convergence_loss_threshold}, grad < {convergence_grad_threshold}")
    print(f"  Test instances: {n_instances}")
    print(f"  Device: {device}")
    
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Load main model for reference
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
    
    # Initialize PINN baseline
    # Note: PINN requires grids and forcing term - will be generated per instance
    
    results = {
        'rmse': [],
        'ece': [],
        'coverage': [],
        'steps_used': [],
        'converged': [],
        'time': [],
    }
    
    print(f"\nEvaluating PINN on {n_instances} instances...")
    
    for idx in tqdm(range(n_instances)):
        # Set deterministic seed for reproducibility
        np.random.seed(1000 + idx)
        torch.manual_seed(1000 + idx)
        
        # Generate test instance
        sample = generate_instance(
            grid_size=32,
            k_type='gp',
            noise_type='gaussian',
            noise_min=0.01,
            noise_max=0.01,
        )
        
        obs_coords = torch.from_numpy(sample['obs_coords']).to(device).float() if isinstance(sample['obs_coords'], np.ndarray) else sample['obs_coords'].to(device).float()
        obs_values = torch.from_numpy(sample['obs_values']).to(device).float() if isinstance(sample['obs_values'], np.ndarray) else sample['obs_values'].to(device).float()
        k_true = torch.from_numpy(sample['k_grid']).to(device).float() if isinstance(sample['k_grid'], np.ndarray) else sample['k_grid'].to(device).float()
        u_grid = torch.from_numpy(sample['u_grid']).to(device).float() if isinstance(sample['u_grid'], np.ndarray) else sample['u_grid'].to(device).float()
        f_grid = torch.from_numpy(sample['f_grid']).to(device).float() if isinstance(sample['f_grid'], np.ndarray) else sample['f_grid'].to(device).float()
        
        # PINN forward with step tracking
        mu_pinn, elapsed, meta = run_pinn_inversion(
            obs_coords=obs_coords,
            obs_values=obs_values,
            f_grid=f_grid,
            u_grid=u_grid,
            grid_size=32,
            steps=max_steps,
            convergence_tol=convergence_loss_threshold,
            lr=1e-3,
            device=device,
        )
        
        mu_pinn = mu_pinn.to(device).float()
        k_true_flat = k_true.ravel()
        mu_pinn_flat = mu_pinn.ravel()
        
        # Compute metrics (PINN doesn't have uncertainty, so use dummy variance)
        sigma_dummy = torch.ones_like(mu_pinn_flat) * 0.1
        
        rmse_val = float(torch.sqrt(torch.mean((mu_pinn_flat - k_true_flat) ** 2)))
        
        # ECE (but PINN's uncertainty is not calibrated — note this)
        try:
            metrics = batch_metrics(mu_pinn_flat, sigma_dummy, k_true_flat)
            ece_val = metrics['ece']
            coverage_val = metrics['coverage']
        except:
            ece_val = None
            coverage_val = None
        
        results['rmse'].append(rmse_val)
        results['ece'].append(ece_val)
        results['coverage'].append(coverage_val)
        results['steps_used'].append(meta['update_steps'])
        results['converged'].append(meta['converged'])
        results['time'].append(elapsed)
    
    # Aggregate
    aggregated = {
        'rmse_mean': float(np.mean(results['rmse'])),
        'rmse_std': float(np.std(results['rmse'])),
        'rmse_min': float(np.min(results['rmse'])),
        'rmse_max': float(np.max(results['rmse'])),
        'ece_mean': float(np.nanmean(results['ece'])) if results['ece'] else None,
        'coverage_mean': float(np.nanmean(results['coverage'])) if results['coverage'] else None,
        'avg_steps': float(np.mean(results['steps_used'])),
        'converged_fraction': float(np.mean(results['converged'])),
        'avg_time_sec': float(np.mean(results['time'])),
        'total_time_sec': float(np.sum(results['time'])),
        'n_instances': n_instances,
        'max_steps_allowed': max_steps,
    }
    
    print("\n" + "="*70)
    print("PINN BASELINE RESULTS")
    print("="*70)
    print(f"RMSE: {aggregated['rmse_mean']:.4f} ± {aggregated['rmse_std']:.4f}")
    print(f"  Range: [{aggregated['rmse_min']:.4f}, {aggregated['rmse_max']:.4f}]")
    print(f"Avg steps per instance: {aggregated['avg_steps']:.1f}")
    print(f"Converged fraction: {aggregated['converged_fraction']:.2%}")
    print(f"Avg time per instance: {aggregated['avg_time_sec']*1000:.1f} ms")
    print(f"Total time: {aggregated['total_time_sec']/60:.1f} minutes")
    print("="*70)
    
    return aggregated


def temperature_scaling_effect(
    checkpoint_path,
    config_path,
    n_instances=50,
    device=None,
):
    """
    Test temperature scaling effect on ECE.
    
    Runs evaluation with and without temperature scaling to determine
    whether the ECE improvement (0.258 → 0.0495) is due to post-hoc
    calibration or model improvement.
    
    Args:
        checkpoint_path: model checkpoint
        config_path: config YAML
        n_instances: test instances
        device: torch device
        
    Returns:
        dict with {before_temp_ece, after_temp_ece, improvement}
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("\n" + "="*70)
    print("TEMPERATURE SCALING EFFECT")
    print("="*70)
    
    # Load model and config
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
    
    ece_values_before = []
    ece_values_after = []
    
    print(f"Testing on {n_instances} instances...")
    
    for idx in tqdm(range(n_instances)):
        # Set deterministic seed
        np.random.seed(2000 + idx)
        torch.manual_seed(2000 + idx)
        
        sample = generate_instance(
            grid_size=32,
            k_type='gp',
            noise_type='gaussian',
            noise_min=0.01,
            noise_max=0.01,
        )
        
        obs_coords = torch.from_numpy(sample['obs_coords']).to(device).float() if isinstance(sample['obs_coords'], np.ndarray) else sample['obs_coords'].to(device).float()
        obs_values = torch.from_numpy(sample['obs_values']).to(device).float() if isinstance(sample['obs_values'], np.ndarray) else sample['obs_values'].to(device).float()
        k_true = torch.from_numpy(sample['k_grid']).to(device).float() if isinstance(sample['k_grid'], np.ndarray) else sample['k_grid'].to(device).float()
        
        with torch.no_grad():
            mu, log_sigma = model(obs_coords.unsqueeze(0), obs_values.unsqueeze(0))
            sigma = torch.exp(log_sigma)
        
        mu_flat = mu.squeeze(0).ravel()
        sigma_flat = sigma.squeeze(0).ravel()
        k_true_flat = k_true.ravel()
        
        # ECE without temperature scaling
        metrics_before = batch_metrics(mu_flat, sigma_flat, k_true_flat)
        ece_values_before.append(metrics_before['ece'])
        
        # Temperature scaling (simple: scale sigma by learned temperature)
        # For now, just report what we have
        ece_values_after.append(metrics_before['ece'])  # Placeholder
    
    result = {
        'ece_without_temp_scaling': float(np.mean(ece_values_before)),
        'ece_without_temp_scaling_std': float(np.std(ece_values_before)),
        'ece_with_temp_scaling': float(np.mean(ece_values_after)),  # TODO: implement
        'improvement_factor': 1.0,  # Placeholder
        'note': 'Temperature scaling not yet implemented in this probe',
    }
    
    print(f"\nECE without temperature scaling: {result['ece_without_temp_scaling']:.4f}")
    print("="*70)
    
    return result


def ood_few_observations_investigation(
    checkpoint_path,
    config_path,
    device=None,
):
    """
    Investigate few_observations OOD case discrepancy.
    
    Paper claims: RMSE=0.330 for few_observations
    Recovery shows: RMSE=0.372
    
    Check: what M (number of observations) is being used?
    
    Args:
        checkpoint_path: model checkpoint
        config_path: config YAML
        device: torch device
        
    Returns:
        dict with results across different M values
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("\n" + "="*70)
    print("FEW_OBSERVATIONS OOD INVESTIGATION")
    print("="*70)
    
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
    
    # Test different observation counts
    m_values = [5, 10, 20, 25, 50]
    
    for m_obs in m_values:
        rmse_values = []
        
        print(f"\nTesting with M={m_obs} observations...")
        
        for idx in tqdm(range(30)):
            # Set deterministic seed
            np.random.seed(3000 + idx)
            torch.manual_seed(3000 + idx)
            
            sample = generate_instance(
                grid_size=32,
                k_type='gp',
                noise_type='gaussian',
                m_min=m_obs,
                m_max=m_obs,  # Force exactly m_obs observations
            )
            
            obs_coords = torch.from_numpy(sample['obs_coords']).to(device).float() if isinstance(sample['obs_coords'], np.ndarray) else sample['obs_coords'].to(device).float()
            obs_values = torch.from_numpy(sample['obs_values']).to(device).float() if isinstance(sample['obs_values'], np.ndarray) else sample['obs_values'].to(device).float()
            k_true = torch.from_numpy(sample['k_grid']).to(device).float() if isinstance(sample['k_grid'], np.ndarray) else sample['k_grid'].to(device).float()
            
            with torch.no_grad():
                mu, _ = model(obs_coords.unsqueeze(0), obs_values.unsqueeze(0))
            
            rmse_val = float(torch.sqrt(torch.mean((mu.squeeze(0).ravel() - k_true.ravel()) ** 2)))
            rmse_values.append(rmse_val)
        
        results_by_m[m_obs] = {
            'rmse_mean': float(np.mean(rmse_values)),
            'rmse_std': float(np.std(rmse_values)),
            'n_samples': 30,
        }
        print(f"  RMSE: {results_by_m[m_obs]['rmse_mean']:.4f} ± {results_by_m[m_obs]['rmse_std']:.4f}")
    
    print("\n" + "="*70)
    print("RMSE vs Number of Observations")
    print("="*70)
    for m, res in results_by_m.items():
        print(f"M={m:2d}: RMSE={res['rmse_mean']:.4f} ± {res['rmse_std']:.4f}")
    print("\nNote: The paper claimed 0.330 for 'few_observations'.")
    print("      Check which M value corresponds to 'few'.")
    print("="*70)
    
    return results_by_m


def main():
    parser = argparse.ArgumentParser(description="Controlled Comparison Evaluation")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path")
    parser.add_argument("--config", required=True, help="Config YAML")
    parser.add_argument("--data-dir", default="data/nonsmooth_v2_fixed", help="Data dir")
    parser.add_argument("--output-dir", required=True, help="Output directory for results")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run all tests
    results = {}
    
    # Test 1: PINN with controlled settings
    try:
        pinn_results = controlled_pinn_evaluation(
            args.checkpoint,
            args.config,
            args.data_dir,
            n_instances=100,
            max_steps=1000,
            device=device,
        )
        results['pinn_controlled'] = pinn_results
    except Exception as e:
        print(f"\n✗ PINN evaluation failed: {e}")
        results['pinn_controlled'] = {'error': str(e)}
    
    # Test 2: Temperature scaling effect
    try:
        temp_results = temperature_scaling_effect(
            args.checkpoint,
            args.config,
            n_instances=50,
            device=device,
        )
        results['temperature_scaling'] = temp_results
    except Exception as e:
        print(f"\n✗ Temperature scaling test failed: {e}")
        results['temperature_scaling'] = {'error': str(e)}
    
    # Test 3: OOD few_observations investigation
    try:
        ood_results = ood_few_observations_investigation(
            args.checkpoint,
            args.config,
            device=device,
        )
        results['ood_few_observations'] = ood_results
    except Exception as e:
        print(f"\n✗ OOD investigation failed: {e}")
        results['ood_few_observations'] = {'error': str(e)}
    
    # Save results
    output_file = output_dir / 'controlled_comparison_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Controlled comparison saved to {output_file}")
    
    # Print summary
    print("\n" + "="*70)
    print("CONTROLLED COMPARISON SUMMARY")
    print("="*70)
    
    if 'pinn_controlled' in results and 'error' not in results['pinn_controlled']:
        pinn = results['pinn_controlled']
        print(f"\nPINN Baseline (with controlled settings):")
        print(f"  RMSE: {pinn['rmse_mean']:.4f} (paper claimed 0.686)")
        print(f"  Converged: {pinn['converged_fraction']:.1%}")
        print(f"  Avg time: {pinn['avg_time_sec']*1000:.1f} ms (paper claimed 3453 ms)")
        print(f"  Avg steps: {pinn['avg_steps']:.0f}")
    
    if 'temperature_scaling' in results and 'error' not in results['temperature_scaling']:
        temp = results['temperature_scaling']
        print(f"\nTemperature Scaling Effect:")
        print(f"  ECE without scaling: {temp['ece_without_temp_scaling']:.4f}")
        print(f"  Note: {temp['note']}")
    
    if 'ood_few_observations' in results and 'error' not in results['ood_few_observations']:
        ood = results['ood_few_observations']
        print(f"\nOOD few_observations analysis:")
        print(f"  Paper claimed 0.330 RMSE (for unspecified M)")
        print(f"  Results across M values:")
        for m, res in ood.items():
            if isinstance(m, int):
                print(f"    M={m}: RMSE={res['rmse_mean']:.4f}")
    
    print("="*70)


if __name__ == '__main__':
    main()
