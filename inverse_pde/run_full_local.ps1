#!/usr/bin/env pwsh
<#
.SYNOPSIS
Run the maintained nonsmooth-v2 pipeline locally.

.DESCRIPTION
1) Trains with configs/nonsmooth_v2.yaml on data/nonsmooth_v2_fixed.
2) Picks the latest checkpoint from outputs_nonsmooth_ece_stop/checkpoints.
3) Runs evaluation into results_eval_epoch004 (or custom directory).
#>

param(
    [switch]$SkipEvaluation,
    [string]$Config = "configs/nonsmooth_v2.yaml",
    [string]$DataDir = "data/nonsmooth_v2_fixed",
    [string]$OutputDir = "outputs_nonsmooth_ece_stop",
    [string]$ResultsDir = "results_eval_epoch004"
)

$ErrorActionPreference = "Stop"

$ProjectDir = "c:\Users\User\Desktop\neural network for pde\inverse_pde"
$VenvPath = ".venv311\Scripts\Activate.ps1"
$PythonExe = ".venv311\Scripts\python.exe"

function Get-LatestCheckpoint {
    param([string]$Dir)
    $ckpts = Get-ChildItem "$Dir\checkpoints\*.pt" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime
    if (-not $ckpts -or $ckpts.Count -eq 0) {
        throw "No checkpoints found in $Dir\checkpoints"
    }
    return $ckpts[-1].FullName
}

Write-Host "Running maintained nonsmooth-v2 pipeline" -ForegroundColor Cyan
Set-Location $ProjectDir
& $VenvPath

$env:WANDB_DISABLED = "true"
$env:WANDB_MODE = "disabled"

if (-not (Test-Path $PythonExe)) {
    throw "Missing Python executable: $PythonExe"
}

& $PythonExe -u main.py --mode train --config $Config --data-dir $DataDir --output-dir $OutputDir
if ($LASTEXITCODE -ne 0) {
    throw "Training failed with exit code $LASTEXITCODE"
}

if (-not $SkipEvaluation) {
    $ckpt = Get-LatestCheckpoint -Dir $OutputDir
    Write-Host "Using checkpoint: $(Split-Path $ckpt -Leaf)" -ForegroundColor Gray
    & $PythonExe -u main.py --mode evaluate --config $Config --checkpoint $ckpt --data-dir $DataDir --results-dir $ResultsDir
    if ($LASTEXITCODE -ne 0) {
        throw "Evaluation failed with exit code $LASTEXITCODE"
    }
}

Write-Host "Done." -ForegroundColor Green
