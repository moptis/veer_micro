"""Report run progress by inspecting the Zarr store, not the logs.

    python scripts/progress.py /scratch/$USER/fuxicfd/out/broadview_sites.zarr
    python scripts/progress.py <store> --watch 60

Reads the store rather than the job output because Slurm .out files are block
buffered: at a few timesteps per second the progress lines only flush every ~10
minutes, which is indistinguishable from a hung job. Chunk files on disk are the
ground truth.

A zarr chunk file exists only once written, so counting chunk files under each
variable gives exact completion. Chunks are also timestamped, so recent write
activity confirms the job is alive.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np


def chunk_stats(var_dir: Path) -> tuple[int, float]:
    """(chunks written, newest mtime) for one variable's chunk tree."""
    newest, count = 0.0, 0
    root = var_dir / "c"                       # zarr v3 chunk root
    if not root.exists():
        root = var_dir                          # zarr v2 lays chunks out flat
    for p in root.rglob("*"):
        if p.is_file():
            count += 1
            m = p.stat().st_mtime
            if m > newest:
                newest = m
    return count, newest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("store", type=Path)
    ap.add_argument("--watch", type=float, default=0.0, help="refresh every N seconds")
    args = ap.parse_args()

    while True:
        import zarr

        root = zarr.open_group(str(args.store), mode="r")
        n_times = root["time"].shape[0]
        tiles = sorted(root.group_keys())
        if tiles:
            first = root[tiles[0]]
            var = next(v for v in ("u", "v", "w", "k") if v in first.array_keys())
            time_chunk = first[var].chunks[0]
            groups = [(t, root[t]) for t in tiles]
        else:                                    # mosaic store
            var = next(v for v in ("u", "v", "w", "k") if v in root.array_keys())
            time_chunk = root[var].chunks[0]
            groups = [("<mosaic>", root)]

        expected = int(np.ceil(n_times / time_chunk))
        now = time.time()
        print(f"\n=== {args.store.name} ===")
        print(f"timesteps {n_times:,}  time_chunk {time_chunk}  -> {expected:,} chunks per variable")

        total_done = 0
        for name, g in groups:
            done, newest = chunk_stats(Path(args.store) / (name if name != "<mosaic>" else "") / var)
            total_done += done
            age = f"{now - newest:6.0f}s ago" if newest else "     never"
            bar_n = int(30 * min(done / expected, 1.0))
            bar = "#" * bar_n + "." * (30 - bar_n)
            print(f"  {name:<16} [{bar}] {done:>6,}/{expected:,}  {100*done/expected:5.1f}%  last write {age}")

        overall = total_done / (expected * len(groups))
        print(f"  {'OVERALL':<16} {100*overall:5.1f}%   (variable '{var}')")
        if not args.watch:
            return 0
        time.sleep(args.watch)


if __name__ == "__main__":
    raise SystemExit(main())
