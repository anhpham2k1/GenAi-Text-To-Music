# Train both methods 10 epochs and prepare comparison CSVs
# Usage: powershell -ExecutionPolicy Bypass -File run_compare_train.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

Write-Host "=== 0. Shared split (data/) ===" -ForegroundColor Cyan
python -m compare.make_split --midi_dir data/processed

Write-Host "=== 1. Train Transformer (10 epochs) ===" -ForegroundColor Cyan
Set-Location "$Root\GenAI_Transformer"
# 4060/5060 8GB: batch 8 | 5060 Ti 16GB: co the batch 16
python train.py --epochs 10 --batch_size 8 --max_seq_len 1024 --no_early_stop --save_epochs "1,5,10"

Write-Host "=== 2. Train Diffusion (10 epochs) ===" -ForegroundColor Cyan
Set-Location "$Root\GenAI_Diffusion"
# 4060/5060: batch 2-4 | 5060 Ti: batch 8
python train.py --epochs 10 --batch_size 4

Write-Host "=== 3. Generate + Evaluate epoch 1/5/10 ===" -ForegroundColor Cyan
Set-Location $Root
python -m compare.run_comparison_eval --epochs 1 5 10

Write-Host "=== DONE. See compare\results\comparison_table.csv ===" -ForegroundColor Green
Write-Host "Docs: docs\TRAINING_GUIDE.md" -ForegroundColor Cyan
