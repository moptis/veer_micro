#!/bin/bash
# Phase 1 of the downscale sweep: warm each project's raw terrain cache.
#
# Launched by scripts/launch_sweep.sh -- not usually run directly.
#
# One CPU-only array task per project, fetching that project's SRTM and
# land-cover rasters and building all four static tensors -- no model, no
# inference (see scripts/prefetch_terrain.py).
#
# This phase exists so phase 2 can run all 60 cells concurrently. Launched
# together against a cold cache, four cells of one project race to download the
# same tiles -- so the choice is one prefetch pass, or serialising the cells.
#
# No GPU: the work is almost entirely network wait, and holding an H100 idle
# while S3 responds is the most expensive possible way to wait.

#SBATCH --job-name=fuxicfd-prefetch
#SBATCH --account=clusterwakes
#SBATCH --partition=shared
#SBATCH --qos=high
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --array=0-15
#SBATCH --output=logs/prefetch_%A_%a.out
#SBATCH --error=logs/prefetch_%A_%a.err

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fuxicfd

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

PROJECTS=(arw azw grw gwr k2w lcw lfw lgw mkw msm oew ph1 ph2 prw sww wsw)
PROJ="${PROJECTS[$SLURM_ARRAY_TASK_ID]}"
OUT_ROOT=/scratch/moptis/fuxicfd/out/wrg_sweep
mkdir -p "${OUT_ROOT}"

echo "### [${PROJ}] terrain prefetch   $(date)"
srun "${CONDA_PREFIX}/bin/python" scripts/prefetch_terrain.py \
    "configs/sweep/${PROJ}_dabs_rabs.yaml"
echo "### [${PROJ}] prefetch complete   $(date)"
