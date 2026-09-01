#!/bin/bash
# One WRG per tile, as a Slurm array -- then concatenate.
#
#   sbatch --array=0-237 scripts/wrg_tiles.sh <store.zarr> <outdir> [hub] [res] [bins] [rho]
#   fuxicfd-wrf wrg-concat <outdir> --out domain.wrg
#
# Why per-tile rather than one domain-wide job: a 6x6 km trimmed core at 30 m is
# 200x200 points, so its histogram is 256 MB against the 77 GB a domain-wide 30 m
# grid would need, and each task reads only its own tile instead of all 18 TB. A
# single-job export was OOM-killed at 183 GB and projected ~9 hours; this is a few
# minutes per task, all in parallel, and a failed tile costs only that tile.
#
# That is also why the default resolution here is 30 m and not 100: per tile the
# model's native grid is affordable, and 100 m was only ever a concession to the
# single-job design.
#
# Each task trims 1500 m from its tile edges. At the 6 km tiling stride that
# leaves a 6x6 km core which tessellates exactly -- and it discards the margin
# measured at 1.9x the residual noise, since FuXi-CFD gets no buffer around its
# 9 km window. wrg-concat resolves any remaining duplicate points by keeping the
# sample from the tile holding it most interior.

#SBATCH --job-name=wrgtile
#SBATCH --account=clusterwakes
#SBATCH --partition=shared
#SBATCH --nodes=1 --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=logs/wrgtile_%A_%a.out
#SBATCH --error=logs/wrgtile_%A_%a.err

set -euo pipefail

STORE="${1:?usage: sbatch scripts/wrg_tiles.sh <store.zarr> <outdir> [hub] [res] [bins] [rho]}"
OUTDIR="${2:?missing output directory}"
HUB="${3:-100}"
RES="${4:-30}"
BINS="${5:-22.5}"
# Air density, kg/m3. Power density scales linearly with it, so a wrong value is a
# straight systematic error in every W/m2 figure. Take it from the source's own
# air_density_100m over the tiles being exported rather than assuming: the WSW box
# averages 0.9828 (1.01 would overstate power density by 2.8%), and the higher
# wsw_rough tile at 1955 m is 0.9757. The 1.01 default is kept only so existing
# invocations behave as before.
RHO="${6:-1.01}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fuxicfd
export PYTHONUNBUFFERED=1
mkdir -p logs "${OUTDIR}"

IDX="${SLURM_ARRAY_TASK_ID:-0}"
# Call the env's interpreter by path, not by name. `conda activate` sets
# CONDA_PREFIX correctly, but a uv-managed python on PATH shadows it, and a bare
# `python` here then runs an interpreter with none of our packages -- which fails
# as a bare ModuleNotFoundError with no hint that the wrong python was picked.
# Console scripts like fuxicfd-wrf are immune (their shebang is absolute), which
# is why only this inline snippet was affected.
TILE=$("${CONDA_PREFIX:?conda env not active}/bin/python" - "$STORE" "$IDX" <<'PY'
import sys, zarr
root = zarr.open_group(sys.argv[1], mode="r")
tiles = sorted(root.group_keys())
i = int(sys.argv[2])
print(tiles[i] if i < len(tiles) else "")
PY
)

if [ -z "${TILE}" ]; then
    echo "array index ${IDX} beyond the tile count: nothing to do"
    exit 0
fi

OUT="${OUTDIR}/${TILE}.wrg"
if [ -s "${OUT}" ]; then
    echo "${OUT} already exists, skipping (delete it to force a rebuild)"
    exit 0
fi

echo "host=$(hostname) tile=${TILE} (index ${IDX})"
srun fuxicfd-wrf wrg "${STORE}" --out "${OUT}" \
    --tile "${TILE}" --hub-height "${HUB}" --resolution "${RES}" \
    --bin-size "${BINS}" --air-density "${RHO}" --trim-edge 1500 \
    --workers "${SLURM_CPUS_PER_TASK:-8}" \
    --terrain-cache /scratch/moptis/fuxicfd/cache
