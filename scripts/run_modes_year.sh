#!/bin/bash
# Full-record 2x2 mode experiment on tile wsw_rough, then a WRG per cell.
#
#   sbatch scripts/run_modes_year.sh
#
# One array task per dem_mode x roughness_mode cell, on gpu-h100. All the GPU
# partitions share a single node pool, so this may sit behind a queue; the 2-day
# limit means that costs waiting, not a truncated run.
#
# Budget: ~21 min of inference per cell at ~42 tile-steps/s x 52560 steps, plus
# an 18.9 GB store to write. Expect 30-40 min per cell; the wall time is set far
# above that so a slow node cannot truncate a cell.
#
# RESUMABLE: a cell whose .wrg already exists is skipped, so a resubmit after any
# failure picks up only the cells that are still missing.
#
# AIR DENSITY: 0.9757 kg/m3, the mean of air_density_100m over this tile, not the
# 1.225 default. The tile sits at 1955 m; the default would overstate power
# density by 26%.

# Resource sizing follows scripts/submit_inference.sh: gpu-h100 nodes carry 4x
# H100 and 128 cores, so 32 CPUs and one GPU per task lets four tasks pack onto
# a node and the GPU count stay the limit. Measured peak RSS is single-digit GB,
# so a 120G request buys nothing and blocks backfill -- a 120G/8h array sat at
# 1 of 40 tasks running while 32G/4h got all 40 scheduled.
#SBATCH --job-name=fuxicfd-year2x2
#SBATCH --account=clusterwakes
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/year2x2_%A_%a.out
#SBATCH --error=logs/year2x2_%A_%a.err

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fuxicfd

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs reports wrg

AIR_DENSITY=0.9757
HUB=100.0

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

CELLS=(dabs_rabs dabs_rrec drec_rabs drec_rrec)
for CELL in "${CELLS[$SLURM_ARRAY_TASK_ID]}"; do
    CONFIG="configs/wsw_nlcdwrf_year_${CELL}.yaml"
    STORE="/scratch/moptis/fuxicfd/out/wsw_nlcdwrf_year_${CELL}.zarr"
    WRG="wrg/wsw_rough_${CELL}.wrg"

    if [[ -s "${WRG}" ]]; then
        echo "=== [${CELL}] already has ${WRG}, skipping   $(date)"
        continue
    fi

    echo "############ [${CELL}] inference over the full record   $(date)"
    srun fuxicfd-wrf run "${CONFIG}"

    echo "############ [${CELL}] WRG export at ${HUB} m           $(date)"
    fuxicfd-wrf wrg "${STORE}" \
        --out "${WRG}" \
        --hub-height "${HUB}" \
        --resolution 30 \
        --bin-size 30 \
        --air-density "${AIR_DENSITY}" \
        --tile wsw_rough

    echo "############ [${CELL}] done                             $(date)"
    ls -la "${WRG}"
done

echo "############ cell complete   $(date)"
