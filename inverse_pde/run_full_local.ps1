#!/usr/bin/env pwsh
<#
.SYNOPSIS
Local Full Production Pipeline Runner for RTX 4050
Runs baseline and nonsmooth with full configs (40k/5k/5k split).
Expected total runtime: 1.5-2.5 hours on RTX 4050 with CUDA enabled.
#>

param(
    [switch]$BaselineOnly,
    [switch]$SkipEvaluation
)

$ErrorActionPreference = "Stop"
$InformationPreference = "Continue"

# Configuration
$ProjectDir = "c:\Users\User\Desktop\neural network for pde\inverse_pde"
$VenvPath = ".venv311\Scripts\Activate.ps1"
$PythonExe = ".venv311\Scripts\python.exe"
$DataDir = "data\generated"
$OutputBaseline = "outputs_full_run_d96"
$OutputNonsmooth = "outputs_nonsmooth"
$ResultsBaseline = "results_full_run_d96"
$ResultsNonsmooth = "results_nonsmooth"

# Helper functions
function Test-GPU {
    Write-Host "=== GPU Detection ===" -ForegroundColor Cyan
    $gpuCheck = & $PythonExe -c "import sys, torch; print('Python:', sys.executable); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
    Write-Host $gpuCheck
    
    $isCuda = $gpuCheck -like "*CUDA: True*"
    if (-not $isCuda) {
        Write-Host "WARNING: GPU not detected. Reinstall PyTorch with CUDA:" -ForegroundColor Yellow
        Write-Host "$PythonExe -m pip uninstall torch torchvision torchaudio -y" -ForegroundColor Yellow
        Write-Host "$PythonExe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121" -ForegroundColor Yellow
        Write-Host ""
    }
    return $isCuda
}

function Get-LatestCheckpoint {
    param([string]$OutputDir)
    $ckpts = Get-ChildItem "$OutputDir\checkpoints\*.pt" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime
    if ($ckpts.Count -eq 0) {
        throw "No checkpoints found in $OutputDir\checkpoints"
    }
    return $ckpts[-1].FullName
}

function Measure-Time {
    param([scriptblock]$ScriptBlock, [string]$Label)
    Write-Host ""
    Write-Host ">>> $Label" -ForegroundColor Green
    $start = Get-Date
    & $ScriptBlock
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE"
    }
    $elapsed = (Get-Date) - $start
    Write-Host "<<< $Label completed in $('{0:hh}h {0:mm}m {0:ss}s' -f $elapsed)" -ForegroundColor Green
}

# Main
Write-Host "
╔════════════════════════════════════════════════════════════╗
║   Local Full Production Pipeline - RTX 4050               ║
║   Baseline + Nonsmooth (40k/5k/5k) (~1.5-2.5h expected)   ║
╚════════════════════════════════════════════════════════════╝
" -ForegroundColor Cyan

# Navigate to project
Set-Location $ProjectDir
Write-Host "Working directory: $ProjectDir" -ForegroundColor Gray

# Activate venv
Write-Host ""
Write-Host "Activating .venv311..." -ForegroundColor Cyan
& $VenvPath

# Disable Weights & Biases logging for faster and quieter local runs.
$env:WANDB_DISABLED = "true"
$env:WANDB_MODE = "disabled"

if (-not (Test-Path $PythonExe)) {
    Write-Host "ERROR: Expected Python executable not found: $PythonExe" -ForegroundColor Red
    Write-Host "Aborting." -ForegroundColor Red
    exit 1
}

# Test GPU
$hasGpu = Test-GPU
Write-Host ""

if (-not $hasGpu) {
    Write-Host "ERROR: GPU not available. CUDA-enabled PyTorch is required for full runs." -ForegroundColor Red
    Write-Host "Aborting." -ForegroundColor Red
    exit 1
}

Write-Host "Run options:" -ForegroundColor Cyan
Write-Host "  BaselineOnly: $BaselineOnly" -ForegroundColor Gray
Write-Host "  SkipEvaluation: $SkipEvaluation" -ForegroundColor Gray
Write-Host ""

# Create output directories
New-Item -ItemType Directory -Path $OutputBaseline, $OutputNonsmooth, $ResultsBaseline, $ResultsNonsmooth -Force | Out-Null

# ============================================================================
# BASELINE TRAINING (Full)
# ============================================================================
Measure-Time {
    & $PythonExe -u main.py --mode train `
        --config configs/full_run_d96.yaml `
        --data-dir $DataDir `
        --output-dir $OutputBaseline `
        2>&1
} "Baseline Training (full_run_d96.yaml, 40k samples)"

# Resolve baseline checkpoint
Write-Host ""
Write-Host "Resolving baseline checkpoint..." -ForegroundColor Gray
$ckptBaseline = Get-LatestCheckpoint $OutputBaseline
Write-Host "Found: $(Split-Path $ckptBaseline -Leaf)" -ForegroundColor Gray

# ============================================================================
# BASELINE EVALUATION
# ============================================================================
if (-not $SkipEvaluation) {
    Measure-Time {
        & $PythonExe -u main.py --mode evaluate `
            --config configs/full_run_d96.yaml `
            --checkpoint $ckptBaseline `
            --data-dir $DataDir `
            --results-dir $ResultsBaseline `
            2>&1
    } "Baseline Evaluation (PINN, OOD, resolution transfer)"
}
else {
    Write-Host "Skipping baseline evaluation (--SkipEvaluation)." -ForegroundColor Yellow
}

if (-not $BaselineOnly) {
    # ============================================================================
    # NONSMOOTH TRAINING (Full)
    # ============================================================================
    Measure-Time {
        & $PythonExe -u main.py --mode train `
            --config configs/nonsmooth.yaml `
            --data-dir $DataDir `
            --output-dir $OutputNonsmooth `
            2>&1
    } "Nonsmooth Training (nonsmooth.yaml, 40k mixed samples)"

    # Resolve nonsmooth checkpoint
    Write-Host ""
    Write-Host "Resolving nonsmooth checkpoint..." -ForegroundColor Gray
    $ckptNonsmooth = Get-LatestCheckpoint $OutputNonsmooth
    Write-Host "Found: $(Split-Path $ckptNonsmooth -Leaf)" -ForegroundColor Gray

    # ============================================================================
    # NONSMOOTH EVALUATION
    # ============================================================================
    if (-not $SkipEvaluation) {
        Measure-Time {
            & $PythonExe -u main.py --mode evaluate `
                --config configs/nonsmooth.yaml `
                --checkpoint $ckptNonsmooth `
                --data-dir $DataDir `
                --results-dir $ResultsNonsmooth `
                2>&1
        } "Nonsmooth Evaluation (per-k-type metrics, OOD, resolution transfer)"
    }
    else {
        Write-Host "Skipping nonsmooth evaluation (--SkipEvaluation)." -ForegroundColor Yellow
    }
}
else {
    Write-Host "Skipping nonsmooth training/evaluation (--BaselineOnly)." -ForegroundColor Yellow
}

# ============================================================================
# SUMMARY
# ============================================================================
Write-Host ""
Write-Host "╔════════════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║   FULL PRODUCTION PIPELINE COMPLETE                       ║" -ForegroundColor Green
Write-Host "╚════════════════════════════════════════════════════════════╝" -ForegroundColor Green
Write-Host ""
Write-Host "Outputs saved to:" -ForegroundColor Cyan
Write-Host "  Baseline: $OutputBaseline" -ForegroundColor Gray
if (-not $BaselineOnly) {
    Write-Host "  Nonsmooth: $OutputNonsmooth" -ForegroundColor Gray
}
Write-Host ""
Write-Host "Results saved to:" -ForegroundColor Cyan
if (-not $SkipEvaluation) {
    Write-Host "  Baseline: $ResultsBaseline" -ForegroundColor Gray
    if (-not $BaselineOnly) {
        Write-Host "  Nonsmooth: $ResultsNonsmooth" -ForegroundColor Gray
    }
}
else {
    Write-Host "  Evaluation skipped (no new results directories)." -ForegroundColor Yellow
}
Write-Host ""
Write-Host "Metrics include:" -ForegroundColor Cyan
Write-Host "  - RMSE, ECE, coverage (ID and multi-OOD conditions)" -ForegroundColor Gray
Write-Host "  - Per-coefficient-type breakdown (k_type: gp, piecewise, inclusion, checkerboard)" -ForegroundColor Gray
Write-Host "  - PINN baseline (Adam, 1000 steps, convergence analysis)" -ForegroundColor Gray
Write-Host "  - 32→64 resolution transfer evaluation" -ForegroundColor Gray
Write-Host "  - Attention figures" -ForegroundColor Gray
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Review results_*/metrics.json for quantitative results" -ForegroundColor Gray
Write-Host "  2. Check results_*/figures/ for visualizations" -ForegroundColor Gray
Write-Host "  3. For paper: run streamlit app.py to generate comparison dashboard" -ForegroundColor Gray
Write-Host ""
