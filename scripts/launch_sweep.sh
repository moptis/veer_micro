#!/bin/bash
# Two-phase launcher for the sector-transfer sweep.
#
#   python scripts/gen_downscale_configs.py     # writes configs/sweep/
#   ./scripts/launch_sweep.sh
#
# Phase 1 warms each project's raw terrain cache (15 CPU tasks); phase 2 runs all
# 60 project x mode cells concurrently on GPUs, gated on phase 1 with afterok.
#
# Use this rather than sbatch'ing run_downscale_sweep.sh directly. The raw
# terrain cache is keyed on tile geometry, not on dem_mode/roughness_mode, so a
# project's four cells want identical rasters -- launched together against a cold
# cache they race to download the same files. Splitting the fetch into its own
# phase is what lets phase 2 be 60-wide instead of 15-wide.
#
# To re-run only some cells after a partial failure, skip the launcher: any cell
# whose .wrg already exists is skipped, so
#     sbatch --array=12,13,14,15 scripts/run_downscale_sweep.sh
# picks up just those (task id = project_index * 4 + cell_index).

set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs

if [[ ! -d configs/sweep ]] || [[ -z "$(ls -A configs/sweep 2>/dev/null)" ]]; then
    echo "configs/sweep is empty -- run: python scripts/gen_downscale_configs.py" >&2
    exit 1
fi

N_CFG=$(ls configs/sweep/*.yaml | wc -l)
echo "== found ${N_CFG} configs =="

echo "== phase 1: terrain prefetch (15 projects, CPU) =="
PREFETCH_ID=$(sbatch --parsable scripts/prefetch_terrain.sh)
echo "   prefetch array: ${PREFETCH_ID}"

echo "== phase 2: 60 cells (GPU), after phase 1 =="
SWEEP_ID=$(sbatch --parsable --dependency=afterok:"${PREFETCH_ID}" scripts/run_downscale_sweep.sh)
echo "   sweep array:    ${SWEEP_ID}"

cat <<EOF

Submitted. Watch with:
  squeue -u \$USER
  tail -f logs/prefetch_${PREFETCH_ID}_0.out
  tail -f logs/sweep_${SWEEP_ID}_0.out

Check everything once done:
  \$CONDA_PREFIX/bin/python scripts/check_downscale_output.py \\
      /scratch/moptis/fuxicfd/out/wrg_sweep/*_30m.wrg
EOF
