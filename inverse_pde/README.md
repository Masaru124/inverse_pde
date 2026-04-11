# Amortized Inverse PDE Solver

This project trains and evaluates an amortized inverse PDE model that maps sparse noisy observations of $u(x)$ to a probabilistic reconstruction of coefficient fields $k(x)$.

It includes:

1. Training and evaluation pipelines for diffusion inverse problems.
2. A production-style Streamlit app (`app.py`) for interactive demos and results exploration.
3. Option A / Option B transfer-validation scripts, including real EIT DBAR-file evaluation.

## Environment Setup

```powershell
cd inverse_pde
python -m venv .venv313
.venv313\Scripts\Activate.ps1
pip install -r requirements.txt
```

Core dependencies are listed in `requirements.txt` (`torch`, `numpy`, `scipy`, `matplotlib`, `streamlit`, `pandas`, etc.).

## Training and Evaluation

### Maintained local runner

```powershell
.\run_full_local.ps1
```

### Direct commands

```powershell
python main.py --mode train --config configs/nonsmooth_v2.yaml --data-dir data/nonsmooth_v2_fixed --output-dir outputs_nonsmooth_ece_stop
python main.py --mode evaluate --config configs/nonsmooth_v2.yaml --checkpoint outputs_nonsmooth_ece_stop/checkpoints/<best>.pt --data-dir data/nonsmooth_v2_fixed --results-dir results_eval_epoch004
```

## Streamlit App

Run the interactive app:

```powershell
streamlit run app.py
```

The app includes:

1. Home and method walkthrough pages.
2. Benchmark and reliability visualizations.
3. Real-world application cards, including real DBAR-based EIT reporting.
4. Live demo and custom-data upload workflows.

## Option A / Option B Validation Scripts

### Synthetic boundary-shift proxy (EIT-style geometry)

```powershell
python tools/run_option_ab_validation.py --eval-cases 20 --finetune-steps 60 --batch-size 4
```

Output:

1. `results_option_ab_boundary.json`

### Real DBAR EIT files

```powershell
python tools/run_dbar_real_option_ab.py --finetune-steps 120 --lr 1e-4
```

Output:

1. `results_option_ab_dbar_real.json`

Current DBAR real-data summary (from `results_option_ab_dbar_real.json`):

1. Option A (zero-shot): RMSE 0.2201, MAE 0.1649, coverage 0.9657.
2. Option B (decoder fine-tune): RMSE 0.1735, MAE 0.1067, coverage 0.9605.

Important note: DBAR NtoD measurements are complex-valued; current scalar-input compatibility uses `real(NtoD)` in the real-data script.

## Key Paths

1. Main app: `app.py`
2. Training entrypoint: `main.py`
3. Configs: `configs/`
4. Model code: `model/`
5. Data generation/loading: `data/`
6. Real DBAR files: `dbar/data/`
7. Validation tools: `tools/`
8. Paper source: `research_paper/paper.tex`

## Notes

1. Checkpoints are auto-discovered under `outputs*/checkpoints/*.pt`.
2. If deploying on Streamlit Community Cloud, ensure required checkpoint and result artifacts are committed.
3. For paper claims, cite numbers directly from JSON artifacts in this repository.
