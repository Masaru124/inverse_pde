# Amortized Inverse PDE Solver

This repository trains a neural inverse solver that maps sparse noisy observations of $u(x)$ to a distribution over $k(x)$ for diffusion PDEs.

## Maintained Configuration Set

Only these configs are kept and supported:

- `configs/default.yaml` (CLI fallback)
- `configs/nonsmooth_v2.yaml` (main training config)
- `configs/nonsmooth_v2_phase1.yaml` (short debugging run)

## Setup

```bash
cd inverse_pde
python -m venv .venv311
# Windows PowerShell
.venv311\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Main Workflow

### Option A: Maintained local runner

```powershell
.\run_full_local.ps1
```

This script:

1. Trains with `configs/nonsmooth_v2.yaml`
2. Uses data directory `data/nonsmooth_v2_fixed`
3. Writes artifacts to `outputs_nonsmooth_ece_stop`
4. Runs evaluation into `results_eval_epoch004`

### Option B: Direct commands

```bash
python main.py --mode train --config configs/nonsmooth_v2.yaml --data-dir data/nonsmooth_v2_fixed --output-dir outputs_nonsmooth_ece_stop
python main.py --mode evaluate --config configs/nonsmooth_v2.yaml --checkpoint outputs_nonsmooth_ece_stop/checkpoints/<best>.pt --data-dir data/nonsmooth_v2_fixed --results-dir results_eval_epoch004
```

## Key Outputs

- Training log: `outputs_nonsmooth_ece_stop/training_log.csv`
- Best checkpoint(s): `outputs_nonsmooth_ece_stop/checkpoints/*.pt`
- Evaluation metrics: `results_eval_epoch004/metrics.json`

## Notes

- Early stopping is configured for ECE-focused behavior in `configs/nonsmooth_v2.yaml`.
- Dataset shards are auto-generated if missing.
- Global seed and deterministic controls are set in config files.
