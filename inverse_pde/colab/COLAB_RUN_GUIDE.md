# Google Colab Run Guide for Inverse PDE

This folder contains a ready-to-run Colab notebook for training and evaluation.

- Notebook: `colab/Inverse_PDE_Colab_Full_Run.ipynb`
- Main entrypoint used: `main.py`

## What This Notebook Covers

1. Clone your repository in Colab.
2. Install dependencies from `requirements.txt`.
3. Verify GPU availability.
4. Run training for the maintained nonsmooth-v2 setup (`configs/nonsmooth_v2.yaml`).
5. Run evaluation for the selected checkpoint.
6. Download outputs and results.

## How To Use

1. Upload this notebook to Colab or open it from your GitHub repo.
2. Switch runtime to GPU.
3. Edit `REPO_URL` in the first setup cell.
4. Run cells in order.

## Important Runtime Notes

- Full-size dataset settings are large (`n_train=40000`, `n_val=5000`, `n_test=5000`) and can take hours.
- Dataset generation is automatic if required shards are missing.
- For quick validation, switch config to `configs/nonsmooth_v2_phase1.yaml`.

## Optional Quick-Run Swap

In the config cell, replace:

- `TRAIN_CONFIG = 'configs/nonsmooth_v2.yaml'` with `configs/nonsmooth_v2_phase1.yaml`

Then rerun training and evaluation cells.

## Expected Artifacts

- Training outputs:
  - `outputs_colab_nonsmooth_v2/`
- Evaluation outputs:
  - `results_colab_nonsmooth_v2/`

## Troubleshooting

- GPU not visible:
  - Reconnect runtime with GPU enabled and rerun setup cells.
- Out of memory:
  - Reduce `training.batch_size` in the config.
- Long runtime:
  - Use `configs/nonsmooth_v2_phase1.yaml` first to verify pipeline end-to-end.
