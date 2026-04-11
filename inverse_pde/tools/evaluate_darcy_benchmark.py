#!/usr/bin/env python
"""
Darcy Flow Benchmark Evaluation

Downloads FNO paper dataset (Li et al. 2021) and evaluates the inverse PDE model
on standard Darcy flow benchmarks.

Dataset: ~1000 instances of (k, u) pairs on 64x64 grid
Task: Given sparse u observations, recover permeability k
Comparison: Direct inverse problem (vs FNO which solves the forward problem)
"""

import os
import sys
import json
import argparse
import urllib.request
import tempfile
from pathlib import Path

import numpy as np
import torch
import scipy.io
from scipy.ndimage import zoom
from tqdm import tqdm

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.model import AmortizedInversePDEModel
from evaluation.evaluate import batch_metrics
from data.generator import generate_instance


def download_darcy_dataset(output_path=None, use_mirror=False):
    """
    Download Darcy flow dataset from FNO paper.
    
    Primary source: https://github.com/neuraloperator/neuraloperator
    The dataset contains 1000 instances of (k, u) pairs on 64x64 grid.
    
    Args:
        output_path: where to save .mat file. If None, uses temp directory
        use_mirror: if True, try alternative download sources
        
    Returns:
        path to downloaded .mat file
    """
    print("\n" + "="*70)
    print("DARCY FLOW DATASET DOWNLOAD")
    print("="*70)
    
    if output_path is None:
        output_path = Path(tempfile.gettempdir()) / "darcy_flow_1000.mat"
    else:
        output_path = Path(output_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Primary source: FNO repository raw data
    urls = [
        # FNO repository (most reliable)
        "https://raw.githubusercontent.com/neuraloperator/neuraloperator/master/data/darcy_flow_1000.mat",
        # Alternative: Zenodo/OSF if available
    ]
    
    if use_mirror:
        urls.append(
            "https://zenodo.org/record/3957632/files/darcy_flow_1000.mat"
        )
    
    for url in urls:
        try:
            print(f"\nAttempting download from: {url}")
            print(f"Saving to: {output_path}")
            
            # Download with progress bar
            def show_progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                percent = min(downloaded * 100 / total_size, 100)
                print(f"\rProgress: {percent:.1f}% ({downloaded / 1e6:.1f} / {total_size / 1e6:.1f} MB)",
                      end="", flush=True)
            
            urllib.request.urlretrieve(url, output_path, show_progress)
            print(f"\n✓ Successfully downloaded from {url}")
            
            # Verify file
            if output_path.stat().st_size > 1e6:  # > 1MB
                print(f"✓ File size: {output_path.stat().st_size / 1e6:.1f} MB")
                return str(output_path)
            else:
                print(f"✗ Downloaded file too small ({output_path.stat().st_size} bytes)")
                output_path.unlink()
                continue
                
        except Exception as e:
            print(f"✗ Failed: {e}")
            if output_path.exists():
                output_path.unlink()
            continue
    
    print("\n" + "!"*70)
    print("DOWNLOAD FAILED - Dataset unavailable from remote sources")
    print("!"*70)
    print("\nTo manually acquire the Darcy dataset:")
    print("1. Visit: https://github.com/neuraloperator/neuraloperator")
    print("2. Download darcy_flow_1000.mat from data/ directory")
    print("3. Place in: inverse_pde/data/darcy/")
    print("\nAlternatively, use synthetic baseline (Realistic-GP):")
    print("  - Local zero-shot proxy suite already tested in missing_claim_checks.json")
    print("\n" + "!"*70)
    
    return None


def prepare_darcy_data(mat_file_path, target_size=32, n_samples=None):
    """
    Load Darcy MAT file and prepare for evaluation.
    
    Args:
        mat_file_path: path to darcy_flow_1000.mat
        target_size: downsampled grid size (default 32 to match model)
        n_samples: max instances to use (None = all)
        
    Returns:
        dict with keys:
            - k_true: (n, 32, 32) permeability field
            - u_true: (n, 32, 32) solution field
            - grid_coords: (32*32, 2) coordinates
    """
    print(f"\nLoading Darcy dataset from {mat_file_path}")
    
    mat = scipy.io.loadmat(mat_file_path)
    
    # FNO dataset keys are typically 'coeff' and 'sol'
    k_64 = mat.get('coeff', mat.get('k'))  # (1000, 64, 64)
    u_64 = mat.get('sol', mat.get('u'))    # (1000, 64, 64)
    
    if k_64 is None or u_64 is None:
        raise ValueError(f"Could not find coefficient/solution data in {mat_file_path}")
    
    print(f"  Original shapes: k {k_64.shape}, u {u_64.shape}")
    
    # Downsample to 32x32 to match model
    if target_size != 64:
        scale_factor = target_size / 64
        k_32 = zoom(k_64, (1, scale_factor, scale_factor), order=1)
        u_32 = zoom(u_64, (1, scale_factor, scale_factor), order=1)
        print(f"  Downsampled to: k {k_32.shape}, u {u_32.shape}")
    else:
        k_32 = k_64
        u_32 = u_64
    
    # Limit to n_samples
    if n_samples is not None:
        k_32 = k_32[:n_samples]
        u_32 = u_32[:n_samples]
        print(f"  Limited to {n_samples} samples")
    
    # Create grid coordinates for evaluation domain
    grid_size = target_size
    x = np.linspace(0, 1, grid_size)
    y = np.linspace(0, 1, grid_size)
    xx, yy = np.meshgrid(x, y)
    grid_coords = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)
    
    print(f"  Grid coordinates shape: {grid_coords.shape}")
    print(f"  Data shapes: k {k_32.shape}, u {u_32.shape}")
    
    return {
        'k_true': k_32,
        'u_true': u_32,
        'grid_coords': grid_coords,
        'grid_size': grid_size,
    }


def sample_observations(u_field, n_obs=50, seed=None):
    """
    Randomly subsample u observations from a grid.
    
    Args:
        u_field: (32, 32) solution field
        n_obs: number of observations to sample
        seed: random seed for reproducibility
        
    Returns:
        obs_coords: (n_obs, 2) observation locations in [0,1]^2
        obs_values: (n_obs,) observed values
    """
    if seed is not None:
        np.random.seed(seed)
    
    grid_size = u_field.shape[0]
    x = np.linspace(0, 1, grid_size)
    y = np.linspace(0, 1, grid_size)
    
    # Random indices
    idx = np.random.choice(grid_size * grid_size, size=n_obs, replace=False)
    grid_idx = np.unravel_index(idx, (grid_size, grid_size))
    
    # Coordinates in [0, 1]
    obs_x = x[grid_idx[0]]
    obs_y = y[grid_idx[1]]
    obs_coords = np.stack([obs_x, obs_y], axis=1).astype(np.float32)
    
    # Values
    obs_values = u_field[grid_idx].astype(np.float32)
    
    return obs_coords, obs_values


def evaluate_darcy(model, device, darcy_data, n_eval=100, n_obs=50, batch_size=32):
    """
    Evaluate model on Darcy benchmark instances.
    
    Args:
        model: inverse PDE model
        device: torch device
        darcy_data: dict from prepare_darcy_data()
        n_eval: number of instances to evaluate
        n_obs: number of observations per instance
        batch_size: batch size for evaluation
        
    Returns:
        results dict with metrics
    """
    k_true = darcy_data['k_true'][:n_eval]
    u_true = darcy_data['u_true'][:n_eval]
    grid_coords = darcy_data['grid_coords']
    
    print(f"\nEvaluating on {len(k_true)} Darcy instances")
    print(f"  Observations per instance: {n_obs}")
    print(f"  Grid size: {darcy_data['grid_size']}x{darcy_data['grid_size']}")
    
    results = {
        'rmse': [],
        'ece': [],
        'coverage': [],
        'inference_time': [],
    }
    
    grid_coords_tensor = torch.from_numpy(grid_coords).to(device)
    
    for idx in tqdm(range(0, len(k_true), batch_size), desc="Darcy eval"):
        batch_end = min(idx + batch_size, len(k_true))
        batch_k = k_true[idx:batch_end]
        batch_u = u_true[idx:batch_end]
        
        batch_obs_coords = []
        batch_obs_values = []
        
        for i, u in enumerate(batch_u):
            obs_x, obs_y = sample_observations(u, n_obs=n_obs, seed=42 + idx + i)
            batch_obs_coords.append(obs_x)
            batch_obs_values.append(obs_y)
        
        # Stack into batch format
        # Model expects: obs_coords (batch*n_obs, 2), obs_values (batch*n_obs,)
        # We'll process one at a time instead
        
        for i in range(batch_end - idx):
            obs_coords = torch.from_numpy(batch_obs_coords[i]).to(device)
            obs_values = torch.from_numpy(batch_obs_values[i]).to(device)
            
            # Forward pass with timing
            with torch.no_grad():
                start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                
                if start_time is not None:
                    start_time.record()
                    start_cpu = None
                else:
                    import time
                    start_cpu = time.time()
                
                mu, log_sigma = model(obs_coords.unsqueeze(0), obs_values.unsqueeze(0))
                
                if end_time is not None:
                    end_time.record()
                    torch.cuda.synchronize()
                    elapsed = start_time.elapsed_time(end_time) / 1000.0
                else:
                    elapsed = time.time() - start_cpu
            
            # Compute metrics
            sigma = torch.exp(log_sigma)
            k_pred = mu.squeeze(0).cpu().numpy()
            sigma_pred = sigma.squeeze(0).cpu().numpy()
            k_true_i = batch_k[i]
            
            # RMSE
            rmse = np.sqrt(np.mean((k_pred - k_true_i)**2))
            results['rmse'].append(rmse)
            
            # ECE, Coverage (need mu/sigma at all grid points)
            mu_all = mu.squeeze(0).cpu().numpy()
            sigma_all = sigma.squeeze(0).cpu().numpy()
            target_flat = k_true_i.ravel()
            
            metrics = batch_metrics(
                mu_all.ravel(),
                sigma_all.ravel(),
                target_flat
            )
            
            results['ece'].append(metrics['ece'])
            results['coverage'].append(metrics['coverage'])
            results['inference_time'].append(elapsed)
    
    # Aggregate
    return {
        'rmse_mean': float(np.mean(results['rmse'])),
        'rmse_std': float(np.std(results['rmse'])),
        'ece_mean': float(np.mean(results['ece'])),
        'ece_std': float(np.std(results['ece'])),
        'coverage_mean': float(np.mean(results['coverage'])),
        'coverage_std': float(np.std(results['coverage'])),
        'inference_time_ms': float(np.mean(results['inference_time']) * 1000),
        'n_instances': len(results['rmse']),
        'n_observations': n_obs,
    }


def main():
    parser = argparse.ArgumentParser(description="Darcy Flow Benchmark Evaluation")
    parser.add_argument("--checkpoint", required=True, help="Path to trained model checkpoint")
    parser.add_argument("--config", required=True, help="Config YAML file")
    parser.add_argument("--data-dir", default="data/nonsmooth_v2_fixed", help="Data directory")
    parser.add_argument("--darcy-path", default=None, help="Path to darcy_flow_1000.mat (if already downloaded)")
    parser.add_argument("--download-output", default="data/darcy/darcy_flow_1000.mat", 
                        help="Where to save downloaded dataset")
    parser.add_argument("--n-eval", type=int, default=100, help="Number of instances to evaluate")
    parser.add_argument("--n-obs", type=int, default=50, help="Number of observations per instance")
    parser.add_argument("--output", required=True, help="Output JSON file for results")
    
    args = parser.parse_args()
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Import config to get model params
    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
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
    print(f"✓ Model loaded")
    
    # Get Darcy dataset
    darcy_path = args.darcy_path
    if darcy_path is None or not Path(darcy_path).exists():
        print(f"\nDarcy dataset not found at {darcy_path}")
        darcy_path = download_darcy_dataset(args.download_output, use_mirror=True)
    
    if darcy_path is None:
        print("\nDarcy dataset unavailable. Results will show blocked status.")
        results = {
            'status': 'blocked',
            'reason': 'Darcy dataset not available for download',
            'note': 'Use --darcy-path to manually specify local dataset location',
        }
    else:
        try:
            # Prepare data
            darcy_data = prepare_darcy_data(darcy_path, target_size=32, n_samples=200)
            
            # Evaluate
            results = evaluate_darcy(
                model, device, darcy_data,
                n_eval=args.n_eval,
                n_obs=args.n_obs,
                batch_size=32
            )
            results['status'] = 'success'
            
        except Exception as e:
            print(f"\n✗ Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            results = {
                'status': 'error',
                'error': str(e),
            }
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_path}")
    print("\nResults:")
    print(json.dumps(results, indent=2))
    
    return results


if __name__ == '__main__':
    main()
