#!/bin/bash
# Sequential MaxNorm sweep runner
# Waits for R19-C10 Run2 to finish, then runs all remaining sweeps

PYTHON="/home/kyccj/miniconda3/envs/venv_1/bin/python"
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

echo "=========================================="
echo "  MaxNorm Sweep Pipeline"
echo "=========================================="
echo ""

# 1. Spikformer CIFAR10 (alpha=7)
echo "[$(date '+%Y-%m-%d %H:%M')] Starting: Spikformer-C10 MaxNorm sweep"
$PYTHON run_spikformer_c10_maxnorm_sweep.py 2>&1 | tee _spikformer_c10_maxnorm_launcher.log
echo "[$(date '+%Y-%m-%d %H:%M')] Done: Spikformer-C10"
echo ""

# 2. VGG16 CIFAR10 (alpha=4)
echo "[$(date '+%Y-%m-%d %H:%M')] Starting: VGG16-C10 MaxNorm sweep"
$PYTHON run_vgg_c10_maxnorm_sweep.py 2>&1 | tee _vgg_c10_maxnorm_launcher.log
echo "[$(date '+%Y-%m-%d %H:%M')] Done: VGG16-C10"
echo ""

# 3. R19 CIFAR100 (alpha=7)
echo "[$(date '+%Y-%m-%d %H:%M')] Starting: R19-C100 MaxNorm sweep"
$PYTHON run_r19_c100_maxnorm_sweep.py 2>&1 | tee _r19_c100_maxnorm_launcher.log
echo "[$(date '+%Y-%m-%d %H:%M')] Done: R19-C100"
echo ""

# 4. VGG16 CIFAR100 (alpha=7)
echo "[$(date '+%Y-%m-%d %H:%M')] Starting: VGG16-C100 MaxNorm sweep"
$PYTHON run_vgg_c100_maxnorm_sweep.py 2>&1 | tee _vgg_c100_maxnorm_launcher.log
echo "[$(date '+%Y-%m-%d %H:%M')] Done: VGG16-C100"
echo ""

echo "=========================================="
echo "  ALL SWEEPS COMPLETED"
echo "  $(date '+%Y-%m-%d %H:%M')"
echo "=========================================="
