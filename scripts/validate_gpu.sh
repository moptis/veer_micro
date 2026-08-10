#!/bin/bash
# Interactive Phase 0 validation on a GPU node.
#
#   sbatch scripts/validate_gpu.sh configs/broadview_sites.yaml
#
# Runs all six Phase 0 checks including the dem_mode arbitration, and writes a
# JSON report. Nothing downstream should be trusted until this has been read.

#SBATCH --job-name=fuxicfd-val
#SBATCH --account=clusterwakes
#SBATCH --partition=debug-gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --output=logs/validate_%j.out
#SBATCH --error=logs/validate_%j.err

set -euo pipefail

CONFIG="${1:-configs/broadview_sites.yaml}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fuxicfd

mkdir -p logs reports
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

srun fuxicfd-wrf validate \
    --config "${CONFIG}" \
    --n-tiles 5 \
    --n-times 24 \
    --compare-modes \
    --out "reports/phase0_$(date +%Y%m%d_%H%M%S).json"
