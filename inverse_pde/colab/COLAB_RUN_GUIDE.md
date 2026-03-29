# Google Colab Run Guide for Inverse PDE

This folder contains a ready-to-run Colab notebook for training and evaluation.

- Notebook: `colab/Inverse_PDE_Colab_Full_Run.ipynb`
- Main entrypoint used: `main.py`

## What This Notebook Covers

1. Clone your repository in Colab.
2. Install dependencies from `requirements.txt`.
3. Verify GPU availability.
4. Run training for:
   - baseline full run (`configs/full_run_d96.yaml`)
   - non-smooth run (`configs/nonsmooth.yaml`)
5. Run evaluation for each checkpoint.
6. Create `results_final_summary/comparison_colab.csv`.
7. Download a zip of outputs and results.

## How To Use

1. Upload this notebook to Colab or open it from your GitHub repo.
2. Switch runtime to GPU.
3. Edit `REPO_URL` in the first setup cell.
4. Run cells in order.

## Important Runtime Notes

- Full-size dataset settings are large (`n_train=40000`, `n_val=5000`, `n_test=5000`) and can take hours.
- Dataset generation is automatic if `data/generated/dataset_shard_*.pt` is missing.
- If you only want a quick validation run, set one of the smoke configs in the config cell.

## Optional Quick-Run Swap

In the config cell, replace:

- `BASELINE_CONFIG = 'configs/full_run_d96.yaml'` with `configs/smoke.yaml`
- `NONSMOOTH_CONFIG = 'configs/nonsmooth.yaml'` with `configs/smoke_nonsmooth.yaml`

Then rerun training and evaluation cells.

## Expected Artifacts

- Training outputs:
  - `outputs_colab_full_run_d96/`
  - `outputs_colab_nonsmooth/`
- Evaluation outputs:
  - `results_colab_full_run_d96/`
  - `results_colab_nonsmooth/`
- Summary CSV:
  - `results_final_summary/comparison_colab.csv`

## Troubleshooting

- GPU not visible:
  - Reconnect runtime with GPU enabled and rerun setup cells.
- Out of memory:
  - Reduce `training.batch_size` in the config.
- Long runtime:
  - Use smoke configs first to verify pipeline end-to-end.
