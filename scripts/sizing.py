"""Print cost and storage estimates for one or more run configs.

    python scripts/sizing.py configs/*.yaml

GPU time here is cheap and storage is not, which is the whole point of looking
before launching: a full-domain sweep over the entire time series costs under
three GPU-days but well over 100 TB.
"""

from __future__ import annotations

import sys
from pathlib import Path

from fuxicfd_wrf.config import RunConfig
from fuxicfd_wrf.pipeline import plan

#: Measured on one H100, batch 4-16 (see README).
TILE_STEPS_PER_SEC = 42.0


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 1
    for path in argv:
        cfg = RunConfig.from_yaml(path)
        d = plan(cfg)
        n_inf = d["n_tiles"] * d["n_times"]
        gpu_h = n_inf / TILE_STEPS_PER_SEC / 3600
        print(f"\n=== {Path(path).name} ===")
        print(f"  tiles           : {d['n_tiles']}")
        print(f"  timesteps       : {d['n_times']}  ({d['time_range'][0]} .. {d['time_range'][1]})")
        print(f"  levels          : {len(d['levels_m'])} of 27")
        print(f"  variables       : {','.join(d['variables'])}  ({d['dtype']})")
        print(f"  per tile-step   : {d['mb_per_tile_timestep']:.2f} MB")
        print(f"  inferences      : {n_inf:,}")
        print(f"  GPU time        : {gpu_h:.1f} H100-hours")
        total_gb = d["total_gb"]
        size = f"{total_gb / 1000:.1f} TB" if total_gb > 1000 else f"{total_gb:.1f} GB"
        print(f"  TOTAL SIZE      : {size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
