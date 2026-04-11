import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from data.dataset import build_dataloaders
from evaluation.evaluate import evaluate_main_model, evaluate_ood, fit_temperature_scaling
from main import _build_model
from utils import load_config, get_device, set_seed

config = load_config('configs/nonsmooth_v2_fast_stable.yaml')
set_seed(int(config.get('seed', 42)), deterministic=bool(config.get('deterministic', False)))
device = get_device()

data_dir = Path('data/nonsmooth_v2_fixed')
train_loader, val_loader, test_loader = build_dataloaders(
    data_dir=data_dir,
    n_train=int(config['data']['n_train']),
    n_val=int(config['data']['n_val']),
    n_test=int(config['data']['n_test']),
    batch_size=int(config['training']['batch_size']),
    seed=int(config.get('seed', 42)),
    num_workers=0,
    pin_memory=bool(config.get('pin_memory', device.type == 'cuda')),
    persistent_workers=False,
    prefetch_factor=int(config.get('prefetch_factor', 2)),
)

model = _build_model(config).to(device)
ckpt_path = Path('outputs_recovery_gpu_fast_stable/checkpoints/epoch_006_val_nll_-0.618860.pt')
ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
model.load_state_dict(ckpt['model_state_dict'])

ece_bins = int(config['evaluation']['ece_bins'])
coverage = float(config['evaluation']['coverage_level'])
target_fields = list(config['training'].get('target_fields', ['k_grid']))
pde_family = str(config['data'].get('pde_family', 'diffusion'))

def run_case(name, use_mc, temp):
    main = evaluate_main_model(
        model=model,
        test_loader=test_loader,
        device=device,
        ece_bins=ece_bins,
        coverage_level=coverage,
        target_fields=target_fields,
        temperature=temp,
        use_mc_dropout=use_mc,
    )
    ood = evaluate_ood(
        model=model,
        device=device,
        ece_bins=ece_bins,
        coverage_level=coverage,
        grid_size=int(config['data']['grid_size']),
        target_fields=target_fields,
        pde_family=pde_family,
        n_samples=12,
        temperature=temp,
        use_mc_dropout=use_mc,
    )
    max_ood_rmse = max(v['rmse'] for v in ood.values())
    return {
        'name': name,
        'temperature': temp,
        'main_rmse': main['rmse'],
        'main_ece': main['ece'],
        'main_coverage': main['coverage'],
        'ood_checkerboard_rmse': ood['non_smooth_checkerboard']['rmse'],
        'ood_max_rmse': max_ood_rmse,
    }

fit_det = fit_temperature_scaling(
    model=model,
    val_loader=val_loader,
    device=device,
    target_fields=target_fields,
    objective='ece',
    ece_bins=ece_bins,
    coverage_level=coverage,
    coverage_min=0.85,
    t_min=0.8,
    t_max=2.5,
    n_steps=35,
    enforce_non_sharpening=True,
    use_mc_dropout=False,
)

fit_mc = fit_temperature_scaling(
    model=model,
    val_loader=val_loader,
    device=device,
    target_fields=target_fields,
    objective='ece',
    ece_bins=ece_bins,
    coverage_level=coverage,
    coverage_min=0.85,
    t_min=0.8,
    t_max=2.5,
    n_steps=35,
    enforce_non_sharpening=True,
    use_mc_dropout=True,
)

results = [
    run_case('det_no_temp', False, 1.0),
    run_case('det_with_temp', False, float(fit_det['temperature'])),
    run_case('mc_no_temp', True, 1.0),
    run_case('mc_with_temp', True, float(fit_mc['temperature'])),
]

print(json.dumps({'fit_det': fit_det, 'fit_mc': fit_mc, 'results': results}, indent=2))
