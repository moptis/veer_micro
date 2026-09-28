#!/bin/bash
# Phase 2 of the downscale sweep: every project x mode cell, all in parallel.
#
# Launched by scripts/launch_sweep.sh -- not usually run directly, because it
# assumes phase 1 has already warmed the terrain cache.
#
# One array task per CELL (16 projects x 8 modes = 128), not per project. An
# earlier version ran a project's four cells in sequence so the first would warm
# the shared raw terrain cache; scripts/prefetch_terrain.sh now does that once up
# front, which frees all 60 to run concurrently -- about 4x on wall clock.
#
# Why not parallelise over tiles instead: GPU inference is ~16 s of a ~204 s
# cell (42 tiles x 16 sectors at the measured 42 tile-steps/s). The other ~92% is
# per-output-grid work -- apply_transfer, combine_sectors, and writing a 422 MB
# ASCII .wrg -- which runs after all tiles are blended and is inherently serial.
# Spreading tiles across GPUs would attack 8% of the runtime; spreading cells
# across nodes attacks all of it.
#
# RESUMABLE: a cell whose .wrg already exists is skipped, so a resubmit after any
# failure picks up only what is missing. Phase 1's cell is always already present.

#SBATCH --job-name=fuxicfd-sweep
#SBATCH --account=clusterwakes
#SBATCH --partition=gpu-h100
##SBATCH --qos=high
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --array=0-127
#SBATCH --output=logs/sweep_%A_%a.out
#SBATCH --error=logs/sweep_%A_%a.err

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fuxicfd

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

PROJECTS=(arw azw grw gwr k2w lcw lfw lgw mkw msm oew ph1 ph2 prw sww wsw)
CELLS=(dabs_rabs dabs_rrec drec_rabs drec_rrec dabs_rconst drec_rconst dabs_rcoarse drec_rcoarse)

# Task id -> (project, cell). Projects are the slow axis so a project's four
# cells land on different nodes rather than queueing behind each other.
PROJ="${PROJECTS[$((SLURM_ARRAY_TASK_ID / 8))]}"
CELL="${CELLS[$((SLURM_ARRAY_TASK_ID % 8))]}"

# Overridable so a variant sweep (e.g. speed_bins: 3) can run beside the
# main one without overwriting it: sbatch --export=ALL,CONFIG_DIR=...,OUT_ROOT=...
CONFIG_DIR="${CONFIG_DIR:-configs/sweep}"
OUT_ROOT="${OUT_ROOT:-/scratch/moptis/fuxicfd/out/wrg_sweep}"
mkdir -p "${OUT_ROOT}"
WRG="${OUT_ROOT}/${PROJ}_${CELL}_30m.wrg"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

if [[ -s "${WRG}" ]]; then
    echo "### [${PROJ}/${CELL}] already present, skipping"
    exit 0
fi

echo "### [${PROJ}/${CELL}] $(date)"
srun fuxicfd-wrf wrg-downscale "${CONFIG_DIR}/${PROJ}_${CELL}.yaml"
ls -la "${WRG}"

echo "### post-check"
# "${CONDA_PREFIX}/bin/python", not bare `python`: under sbatch the activated
# env's bin is not reliably first on PATH for non-srun steps, and a bare `python`
# picked up a system interpreter with no numpy -- failing the whole task after
# the cell had already succeeded.
"${CONDA_PREFIX}/bin/python" scripts/check_downscale_output.py "${WRG}"

echo "### [${PROJ}/${CELL}] complete   $(date)"
