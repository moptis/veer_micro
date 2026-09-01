#!/usr/bin/env python
"""Collate `validate --compare-modes` reports into the dem_mode x roughness_mode 2x2.

`--compare-modes` arbitrates dem_mode only, so a full 2x2 takes two runs that
differ in terrain.roughness_mode. Each report therefore supplies one column:
its `4_closure[<dem_mode>]` / `5_profile[<dem_mode>]` entries are both at that
report's roughness_mode, which is read back from the config name recorded in the
filename.

    python scripts/collate_modes.py reports/modes_..._r-absolute_*.json \
                                    reports/modes_..._r-recentered_*.json
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

DEM_MODES = ("absolute", "recentered")


def roughness_mode_of(path: Path) -> str:
    m = re.search(r"_r-(absolute|recentered)", path.name)
    if not m:
        raise SystemExit(f"cannot tell roughness_mode from filename: {path.name}")
    return m.group(1)


def main(argv: list[str]) -> int:
    if not argv:
        raise SystemExit(__doc__)

    cells: dict[tuple[str, str], dict] = {}
    for p in map(Path, argv):
        rep = json.loads(p.read_text())
        rmode = roughness_mode_of(p)
        for dmode in DEM_MODES:
            prof = rep.get(f"5_profile[{dmode}]")
            clos = rep.get(f"4_closure[{dmode}]")
            if prof is None or clos is None:
                continue
            tiles = prof.get("per_tile", {})
            ctiles = clos.get("per_tile", {})
            if not tiles:
                continue
            n = len(tiles)
            cells[(dmode, rmode)] = {
                "profile_rmse": sum(t["rmse_overall"] for t in tiles.values()) / n,
                "mean_abs_bias": sum(t["mean_abs_bias"] for t in tiles.values()) / n,
                "u_bias": sum(t["u_bias"] for t in ctiles.values()) / len(ctiles),
                "v_bias": sum(t["v_bias"] for t in ctiles.values()) / len(ctiles),
                "n_tiles": n,
            }

    if not cells:
        raise SystemExit("no usable mode results found in the given reports")

    print("\ndem_mode x roughness_mode  --  mean profile RMSE vs WRF (m/s), lower is better\n")
    hdr = f"{'dem_mode':<14}" + "".join(f"{'rough=' + r:>22}" for r in DEM_MODES)
    print(hdr)
    print("-" * len(hdr))
    for dmode in DEM_MODES:
        row = f"{dmode:<14}"
        for rmode in DEM_MODES:
            c = cells.get((dmode, rmode))
            row += f"{c['profile_rmse']:>22.4f}" if c else f"{'--':>22}"
        print(row)

    print("\nsupporting detail (mean |bias| through the profile; closure u/v bias)\n")
    print(f"{'dem_mode':<14}{'roughness_mode':<16}{'prof RMSE':>11}"
          f"{'|bias|':>10}{'u_bias':>10}{'v_bias':>10}")
    ranked = sorted(cells.items(), key=lambda kv: kv[1]["profile_rmse"])
    for (dmode, rmode), c in ranked:
        print(f"{dmode:<14}{rmode:<16}{c['profile_rmse']:>11.4f}"
              f"{c['mean_abs_bias']:>10.4f}{c['u_bias']:>10.4f}{c['v_bias']:>10.4f}")

    best, bc = ranked[0]
    worst, wc = ranked[-1]
    spread = wc["profile_rmse"] - bc["profile_rmse"]
    print(f"\nbest : dem_mode={best[0]}, roughness_mode={best[1]}  "
          f"(RMSE {bc['profile_rmse']:.4f})")
    print(f"spread across the 2x2: {spread:.4f} m/s "
          f"({100 * spread / bc['profile_rmse']:.1f}% of the best)")
    print("\nLower RMSE against the WRF profile wins. If two cells are within noise "
          "of each other,\nprefer 'absolute' as the faithful reading of the model card.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
