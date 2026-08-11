#!/bin/bash
# Two-phase launcher: initialise the store once, then fan out the array.
#
#   ./scripts/launch.sh configs/broadview_sites.yaml 16
#
# Use this rather than sbatch'ing submit_inference.sh directly.
#
# Why two phases: every array task calls store.create() and builds per-tile
# terrain. Launched together they all race to create the same Zarr arrays and to
# download the same SRTM/WorldCover tiles. Array-task creation is idempotent now,
# but an init job with an afterok dependency removes the contention entirely and
# fails loudly once instead of 16 times.
#
# To re-run only specific shards after a partial failure, keep --n-shards
# IDENTICAL to the original launch or the time ranges shift:
#
#   sbatch --array=0,1,2,7 scripts/submit_inference.sh configs/broadview_sites.yaml 16

set -euo pipefail

CONFIG="${1:?usage: ./scripts/launch.sh <config.yaml> [n_shards]}"
NSHARDS="${2:-16}"

mkdir -p logs

echo "== phase 1: init store + terrain cache =="
INIT_ID=$(sbatch --parsable \
    --job-name=fuxicfd-init \
    --account=clusterwakes \
    --partition=shared \
    --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=64G \
    --time=04:00:00 \
    --output=logs/fuxicfd_init_%j.out \
    --error=logs/fuxicfd_init_%j.err \
    --wrap "source \$(conda info --base)/etc/profile.d/conda.sh && conda activate fuxicfd && fuxicfd-wrf init ${CONFIG}")
echo "   init job: ${INIT_ID}"

echo "== phase 2: inference array 0-$((NSHARDS - 1)) =="
ARRAY_ID=$(sbatch --parsable \
    --dependency=afterok:"${INIT_ID}" \
    --array=0-$((NSHARDS - 1)) \
    scripts/submit_inference.sh "${CONFIG}" "${NSHARDS}")
echo "   array job: ${ARRAY_ID} (starts after ${INIT_ID} succeeds)"
echo
echo "watch:   squeue -j ${INIT_ID},${ARRAY_ID}"
echo "logs:    logs/fuxicfd_${ARRAY_ID}_*.err"
