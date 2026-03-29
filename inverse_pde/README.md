# Amortized Inverse PDE Solver

Research codebase for learning a mapping from sparse noisy observations of u(x) to a distribution over k(x) for:

div(k(x) grad(u(x))) = f(x)

## Setup

```bash
cd inverse_pde
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## One-line Commands

```bash
python main.py --mode train
python main.py --mode evaluate --checkpoint path/to/ckpt.pt
python main.py --mode baselines
```

## New Experiment Presets

- reaction-diffusion multi-target training:

```bash
python main.py --mode train --config configs/reaction_diffusion.yaml --data-dir data/reaction_diffusion --output-dir outputs_reaction_diffusion
python main.py --mode evaluate --config configs/reaction_diffusion.yaml --checkpoint outputs_reaction_diffusion/checkpoints/<best>.pt --data-dir data/reaction_diffusion --results-dir results_reaction_diffusion
```

- non-smooth coefficient mix (60% GP + mixed sharp regimes):

```bash
python main.py --mode train --config configs/diffusion_nonsmooth_mix.yaml --data-dir data/diffusion_nonsmooth --output-dir outputs_nonsmooth
```

- structured noise mix (gaussian/correlated/outlier):

```bash
python main.py --mode train --config configs/diffusion_structured_noise.yaml --data-dir data/diffusion_structured_noise --output-dir outputs_structured_noise
```

## Streamlit Frontend

Run an interactive dashboard to inspect final metrics, compare configurations, and visualize OOD/calibration behavior:

```bash
streamlit run app.py
```

The app automatically reads:

- summary table: `results_final_summary/comparison.csv`
- run metrics: `results*/metrics.json`

It also includes an Equation Playground tab where users can:

- choose PDE variants (diffusion, Poisson, reaction-diffusion, advection-diffusion),
- generate a synthetic test instance with configurable noise/observation count,
- optionally run a selected checkpoint and view predicted `k`, error maps, and RMSE/MAE.

## What Happens In Each Mode

- train: validates generator first, auto-generates dataset shards if missing, then trains with early stopping and top-3 checkpoint retention.
- evaluate: loads checkpoint, evaluates main model and baselines, runs OOD tests, saves metrics and figures.
- baselines: runs baseline pipeline and writes baseline metrics into results.

## Outputs

- training logs: outputs/training_log.csv
- checkpoints: outputs/checkpoints/\*.pt
- evaluation metrics: results/metrics.json
- figures: results/figures/instance_0.png ... instance_5.png

## Notes

- Synthetic data generation uses torch.autograd.grad for f computation.
- Variable-length observations are supported through key padding masks in cross-attention.
- Reaction-diffusion mode extends observation tokens from (x, u) to (x, t, u) and predicts both k(x) and r(x).
- Generator now supports non-smooth coefficient samplers (piecewise/inclusion/checkerboard) and structured noise (correlated/outlier).
- Evaluation now includes per-k-type metrics, expanded OOD conditions, PINN LR/optimizer sweeps, and 32->64 resolution transfer checks.
- Attention maps are exported during evaluation to results/figures/attention when enabled in config.
- MC dropout remains active at inference via explicit forced stochastic passes.
- Global seed and deterministic behavior are configurable in configs/default.yaml.
