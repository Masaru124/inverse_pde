#!/usr/bin/env python
"""
Darcy Flow Benchmark Evaluation (PyTorch Format)

Evaluates the inverse PDE model on Darcy flow benchmark from .pt files.
Dataset: 1000 train + 50 test instances of (k, u) pairs on 16x16 grid
Task: Given sparse u observations, recover permeability k

Usage:
    python tools/evaluate_darcy_pt.py \
        --checkpoint outputs_recovery_gpu_fast_stable/checkpoints/epoch_006_val_nll_-0.618860.pt \
        --config configs/nonsmooth_v2_fast_stable_quick_eval.yaml \
        --train-data darcy_train_16.pt \
        --test-data darcy_test_16.pt \
        --output-dir results_recovery_gpu_fast_stable_quick_full_v3 \
        --n-eval 50
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import zoom
from tqdm import tqdm

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.model import AmortizedInversePDEModel
from evaluation.evaluate import batch_metrics
from utils import load_config


def load_darcy_data(train_path, test_path, target_size=32):
    """
    Load Darcy dataset from .pt files and upsample to target grid size.
    
    Args:
        train_path: path to training data
        test_path: path to test data
        target_size: target grid size (default 32 to match model)
    
    Returns:
        dict with keys:
            - k_train: (1000, target_size, target_size) training permeability
            - u_train: (1000, target_size, target_size) training solution
            - k_test: (50, target_size, target_size) test permeability
            - u_test: (50, target_size, target_size) test solution
            - grid_coords: (target_size^2, 2) grid coordinates
    """
    print(f"\nLoading Darcy dataset:")
    print(f"  Train from: {train_path}")
    print(f"  Test from: {test_path}")
    
    train_data = torch.load(train_path)
    test_data = torch.load(test_path)
    
    # 'x' is permeability field k, 'y' is solution u
    k_train = train_data['x'].numpy()
    u_train = train_data['y'].numpy()
    k_test = test_data['x'].numpy()
    u_test = test_data['y'].numpy()
    
    print(f"  Original k_train shape: {k_train.shape}")
    print(f"  Original u_train shape: {u_train.shape}")
    print(f"  Original k_test shape: {k_test.shape}")
    print(f"  Original u_test shape: {u_test.shape}")
    
    # Upsample to target size if needed
    source_size = k_test.shape[1]
    if source_size != target_size:
        scale_factor = target_size / source_size
        k_train = zoom(k_train, (1, scale_factor, scale_factor), order=1)
        u_train = zoom(u_train, (1, scale_factor, scale_factor), order=1)
        k_test = zoom(k_test, (1, scale_factor, scale_factor), order=1)
        u_test = zoom(u_test, (1, scale_factor, scale_factor), order=1)
        print(f"  Upsampled to: k {k_test.shape}, u {u_test.shape}")
    
    # Create grid coordinates for target grid
    grid_size = k_test.shape[1]
    x = np.linspace(0, 1, grid_size)
    y = np.linspace(0, 1, grid_size)
    xx, yy = np.meshgrid(x, y)
    grid_coords = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)
    
    return {
        'k_train': k_train,
        'u_train': u_train,
        'k_test': k_test,
        'u_test': u_test,
        'grid_coords': grid_coords,
        'grid_size': grid_size,
    }


def sample_observations(u_field, n_obs=16, seed=None):
    """
    Randomly subsample u observations from a grid.
    
    Args:
        u_field: (H, W) solution field
        n_obs: number of observations to sample
        seed: random seed
        
    Returns:
        obs_coords: (n_obs, 2) normalized coordinates
        obs_values: (n_obs,) observed values
    """
    if seed is not None:
        np.random.seed(seed)
    
    h, w = u_field.shape
    grid_size = h
    
    # Random indices
    idx = np.random.choice(h * w, size=min(n_obs, h * w), replace=False)
    grid_idx = np.unravel_index(idx, (h, w))
    
    # Coordinates in [0, 1]
    x = np.linspace(0, 1, grid_size)
    y = np.linspace(0, 1, grid_size)
    
    obs_x = x[grid_idx[0]]
    obs_y = y[grid_idx[1]]
    obs_coords = np.stack([obs_x, obs_y], axis=1).astype(np.float32)
    
    # Values
    obs_values = u_field[grid_idx].astype(np.float32)
    
    return obs_coords, obs_values


def evaluate_darcy(model, device, darcy_data, n_eval=50, n_obs=16, batch_size=8):
    """
    Evaluate model on Darcy test set.
    
    Args:
        model: inverse PDE model
        device: torch device
        darcy_data: dict from load_darcy_data()
        n_eval: number of test instances to evaluate
        n_obs: number of observations per instance
        batch_size: batch size for evaluation
        
    Returns:
        results dict with metrics
    """
    k_test = darcy_data['k_test'][:n_eval]
    u_test = darcy_data['u_test'][:n_eval]
    grid_coords = darcy_data['grid_coords']
    
    print(f"\nEvaluating on {len(k_test)} Darcy test instances")
    print(f"  Grid size: {darcy_data['grid_size']}x{darcy_data['grid_size']}")
    print(f"  Observations per instance: {n_obs}")
    print(f"  Total grid points: {len(grid_coords)}")
    
    results = {
        'rmse': [],
        'mae': [],
        'ece': [],
        'coverage': [],
        'inference_time': [],
        'instances': n_eval,
        'n_obs': n_obs,
        'grid_size': darcy_data['grid_size'],
    }
    
    grid_coords_tensor = torch.from_numpy(grid_coords).to(device)
    
    model.eval()
    with torch.no_grad():
        for idx in tqdm(range(0, len(k_test), batch_size), desc="Darcy evaluation"):
            batch_end = min(idx + batch_size, len(k_test))
            batch_size_actual = batch_end - idx
            batch_k_true = k_test[idx:batch_end]
            batch_u_true = u_test[idx:batch_end]
            
            # Sample observations for each instance
            batch_obs_coords = []
            batch_obs_values = []
            
            for i in range(batch_size_actual):
                # Use deterministic seed based on index for reproducibility
                obs_coords, obs_values = sample_observations(
                    batch_u_true[i], 
                    n_obs=n_obs, 
                    seed=idx + i
                )
                batch_obs_coords.append(obs_coords)
                batch_obs_values.append(obs_values)
            
            # Stack into batches
            obs_coords_batch = np.stack(batch_obs_coords, axis=0)  # (B, n_obs, 2)
            obs_values_batch = np.stack(batch_obs_values, axis=0)  # (B, n_obs)
            
            # Convert to tensors
            obs_coords_batch = torch.from_numpy(obs_coords_batch).to(device)
            obs_values_batch = torch.from_numpy(obs_values_batch).unsqueeze(-1).to(device)  # (B, n_obs, 1)
            
            # Evaluate
            t0 = time.time()
            pred_mean, pred_cov = model(
                obs_coords=obs_coords_batch,
                obs_times=None,
                obs_values=obs_values_batch,
            )
            inference_time = (time.time() - t0) / batch_size_actual
            
            # Compute metrics
            pred_mean_np = pred_mean.cpu().numpy()  # (B, 32, 32)
            pred_cov_np = pred_cov.cpu().numpy()  # (B, 32, 32)
            
            # Flatten to match grid_size^2
            batch_size_actual = pred_mean_np.shape[0]
            pred_mean_np = pred_mean_np.reshape(batch_size_actual, -1)  # (B, 1024)
            pred_cov_np = pred_cov_np.reshape(batch_size_actual, -1)  # (B, 1024)
            pred_std_np = np.sqrt(np.maximum(pred_cov_np, 1e-8))
            
            grid_size = darcy_data['grid_size']
            
            for i in range(batch_size_actual):
                k_true_flat = batch_k_true[i].flatten()
                k_pred_mean = pred_mean_np[i]
                k_pred_std = pred_std_np[i]
                
                # RMSE
                rmse = np.sqrt(np.mean((k_pred_mean - k_true_flat) ** 2))
                results['rmse'].append(float(rmse))
                
                # MAE
                mae = np.mean(np.abs(k_pred_mean - k_true_flat))
                results['mae'].append(float(mae))
                
                # ECE (expected calibration error)
                error = np.abs(k_pred_mean - k_true_flat)
                mask = k_pred_std > 1e-6
                if np.any(mask):
                    z_score = error[mask] / k_pred_std[mask]
                    # Approximate ECE: mean |coverage - confidence|
                    ece_val = float(np.mean(np.abs(error[mask] / (k_pred_std[mask] + 1e-6))))
                    results['ece'].append(ece_val)
                else:
                    results['ece'].append(float(np.mean(error)))
                
                # Coverage at 2-sigma
                coverage = np.mean(error <= 2 * k_pred_std)
                results['coverage'].append(float(coverage))
                
                # Inference time
                results['inference_time'].append(float(inference_time))
    
    return results


def main(args):
    print("\n" + "="*70)
    print("DARCY FLOW BENCHMARK EVALUATION (PT FORMAT)")
    print("="*70)
    
    # Load config
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Extract model config
    model_cfg = config.get('model', {})
    data_cfg = config.get('data', {})
    
    # Load model with correct dimensions
    print(f"\nLoading checkpoint: {args.checkpoint}")
    model = AmortizedInversePDEModel(
        grid_size=data_cfg.get('grid_size', 32),
        d_model=model_cfg.get('d_model', 96),
        n_heads=model_cfg.get('n_heads', 4),
        n_layers=model_cfg.get('n_layers', 3),
        dropout=model_cfg.get('dropout', 0.1),
        mc_samples=model_cfg.get('mc_samples', 50),
        include_time=model_cfg.get('include_time', False),
    )
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    print(f"[OK] Model loaded")
    
    # Load Darcy data
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Use model's grid size from config
    target_grid_size = data_cfg.get('grid_size', 32)
    darcy_data = load_darcy_data(
        args.train_data, 
        args.test_data,
        target_size=target_grid_size
    )
    
    # Evaluate
    results = evaluate_darcy(
        model,
        device,
        darcy_data,
        n_eval=args.n_eval,
        n_obs=args.n_obs,
        batch_size=args.batch_size,
    )
    
    # Summarize
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    summary = {
        'metric': {
            'rmse': {
                'mean': float(np.mean(results['rmse'])),
                'std': float(np.std(results['rmse'])),
                'min': float(np.min(results['rmse'])),
                'max': float(np.max(results['rmse'])),
            },
            'mae': {
                'mean': float(np.mean(results['mae'])),
                'std': float(np.std(results['mae'])),
            },
            'ece': {
                'mean': float(np.mean(results['ece'])),
                'std': float(np.std(results['ece'])),
            },
            'coverage_2sigma': {
                'mean': float(np.mean(results['coverage'])),
                'std': float(np.std(results['coverage'])),
            },
            'inference_time_ms': {
                'mean': float(np.mean(results['inference_time']) * 1000),
                'std': float(np.std(results['inference_time']) * 1000),
            },
        },
        'config': {
            'instances': results['instances'],
            'n_obs': results['n_obs'],
            'grid_size': results['grid_size'],
        },
    }
    
    for metric_name, metric_vals in summary['metric'].items():
        print(f"\n{metric_name}:")
        for key, val in metric_vals.items():
            if isinstance(val, float):
                print(f"  {key}: {val:.4f}")
    
    # Save results
    output_file = Path(args.output_dir) / 'darcy_benchmark_results.json'
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n[OK] Results saved to {output_file}")
    
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate model on Darcy benchmark')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--train-data', type=str, default='darcy_train_16.pt', help='Training data path')
    parser.add_argument('--test-data', type=str, default='darcy_test_16.pt', help='Test data path')
    parser.add_argument('--output-dir', type=str, default='results_darcy', help='Output directory')
    parser.add_argument('--n-eval', type=int, default=50, help='Number of test instances')
    parser.add_argument('--n-obs', type=int, default=16, help='Number of observations')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    
    args = parser.parse_args()
    main(args)
