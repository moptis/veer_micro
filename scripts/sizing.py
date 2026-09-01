"""Print cost and storage estimates for one or more run configs.

    python scripts/sizing.py configs/*.yaml
    python scripts/sizing.py configs/wsw_domain.yaml --bbox -105.8 34.3 -105.2 34.8

GPU time here is cheap and storage is not, which is the whole point of looking
before launching: a full-domain sweep over the entire time series costs under
three GPU-days but well over 100 TB.

`--bbox MIN_LON MIN_LAT MAX_LON MAX_LAT` overrides `tiling.bbox` for the estimate
only, so you can price a few candidate areas against an existing config before
committing one to YAML. Cost is linear in tile count, so this is normally the
largest single lever available.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fuxicfd_wrf.config import BBox, RunConfig
from fuxicfd_wrf.pipeline import plan

#: Sustained rate of a *production array task*, measured on the 174-tile Panhandle
#: run. Not the 42 tile-steps/s an idle H100 reaches: gpu-h100 nodes pack four
#: tasks onto 128 cores, and the CPU stages (preprocess ~15%, postprocess ~27%)
#: then compete. Sizing off 42 is what under-quoted that run by 2.8x and put every
#: shard past its wall clock -- see scripts/bench_stages.py for the breakdown.
TILE_STEPS_PER_SEC = 15.0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("configs", nargs="*", type=Path)
    ap.add_argument("--bbox", nargs=4, type=float, default=None,
                    metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
                    help="override tiling.bbox for this estimate only")
    args = ap.parse_args(argv)
    if not args.configs:
        print(__doc__)
        return 1

    for path in args.configs:
        cfg = RunConfig.from_yaml(path)
        if args.bbox is not None:
            if cfg.tiling.mode != "domain":
                raise SystemExit(f"{path.name}: --bbox needs tiling.mode='domain'")
            cfg.tiling.bbox = BBox(min_lon=args.bbox[0], min_lat=args.bbox[1],
                                   max_lon=args.bbox[2], max_lat=args.bbox[3])
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
