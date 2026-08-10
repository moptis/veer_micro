#!/bin/bash
# Slurm array job for FuXi-CFD inference on Kestrel H100 nodes.
#
#   ./scripts/launch.sh configs/broadview_sites.yaml 16      <-- preferred
#   sbatch --array=0-15 scripts/submit_inference.sh configs/broadview_sites.yaml 16
#
# **Array index = time shard, NOT tile.** Every task runs all tiles for its slice
# of the time axis. A 5-tile config with --array=0-15 is 16 tasks, not 5.
#
# Shards split on Zarr chunk boundaries (see writers.shard_bounds), so tasks write
# disjoint chunks and need no locking.
#
# The second argument is the TOTAL shard count and must stay identical across
# re-runs. Do not fall back to SLURM_ARRAY_TASK_COUNT: for a partial re-run like
# `--array=0,1,2,7` that variable is 4, which would redefine every shard boundary
# and write the wrong timesteps into already-populated chunks.
#
# Node layout: gpu-h100 nodes have 4x H100 and 128 cores. This script requests one
# GPU per task, so four tasks can pack onto a node. For exclusive whole-node use,
# switch to --gres=gpu:h100:4 and launch four workers with different --shard.

#SBATCH --job-name=fuxicfd
#SBATCH --account=clusterwakes
#SBATCH --partition=gpu-h100
#SBATCH --qos=normal
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=logs/fuxicfd_%A_%a.out
#SBATCH --error=logs/fuxicfd_%A_%a.err

set -euo pipefail

CONFIG="${1:?usage: sbatch scripts/submit_inference.sh <config.yaml> <n_shards>}"
NSHARDS="${2:?missing n_shards -- must match the original launch exactly}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fuxicfd   # activation hook prepends $CONDA_PREFIX/lib to LD_LIBRARY_PATH

mkdir -p logs

# Unbuffered stdout. Without this, progress lines sit in an 8 KB block buffer and
# the .out file appears frozen for ~10 minutes at a time, which looks exactly like
# a hung job. stderr is unbuffered already, which is why only the h5py warning
# showed up during the first launch.
export PYTHONUNBUFFERED=1

SHARD="${SLURM_ARRAY_TASK_ID:-0}"

if [ "${SHARD}" -ge "${NSHARDS}" ]; then
    echo "shard ${SHARD} >= n_shards ${NSHARDS}: nothing to do"
    exit 0
fi

echo "host=$(hostname) shard=${SHARD}/${NSHARDS} config=${CONFIG}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

srun fuxicfd-wrf run "${CONFIG}" \
    --shard "${SHARD}" \
    --n-shards "${NSHARDS}" \
    --device cuda
