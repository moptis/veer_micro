#!/bin/bash
# Full dem_mode x roughness_mode arbitration against the NLCD-forced WRF store.
#
#   sbatch scripts/compare_modes_nlcdwrf.sh
#
# `validate --compare-modes` loops over dem_mode only, holding roughness_mode at
# whatever the config says (cli.py:142), so roughness_mode has never actually
# been tested -- every config carries `absolute` as an untested default. Running
# the same command against the two twin configs, which differ only in
# terrain.roughness_mode, yields all four combinations without touching the
# validated code path.
#
# Writes two JSON reports; scripts/collate_modes.py turns them into the 2x2.

# Resource sizing follows scripts/submit_inference.sh: gpu-h100 nodes carry 4x
# H100 and 128 cores, so 32 CPUs and one GPU per task lets four tasks pack onto
# a node and the GPU count stay the limit. Measured peak RSS is single-digit GB,
# so a 120G request buys nothing and blocks backfill -- a 120G/8h array sat at
# 1 of 40 tasks running while 32G/4h got all 40 scheduled.
#SBATCH --job-name=fuxicfd-modes
#SBATCH --account=clusterwakes
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/modes_%j.out
#SBATCH --error=logs/modes_%j.err

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fuxicfd

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs reports
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

STAMP=$(date +%Y%m%d_%H%M%S)
N_TIMES=240        # stride 219 in the configs -> 219 x 240 = 52560, the whole record

for RMODE in absolute recentered; do
    echo "############ roughness_mode=${RMODE} (both dem_modes) ############"
    srun fuxicfd-wrf validate \
        --config "configs/wsw_nlcdwrf_modes_r-${RMODE}.yaml" \
        --n-tiles 1 \
        --n-times "${N_TIMES}" \
        --compare-modes \
        --out "reports/modes_nlcdwrf_r-${RMODE}_${STAMP}.json"
done

echo "############ collated result ############"
python scripts/collate_modes.py \
    "reports/modes_nlcdwrf_r-absolute_${STAMP}.json" \
    "reports/modes_nlcdwrf_r-recentered_${STAMP}.json"
