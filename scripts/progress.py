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


def fast_check(store: Path) -> int:
    """Verify coverage by sampling the data, not by counting files.

    An unwritten zarr chunk reads back as fill_value (0), so an all-zero slice at
    a sampled timestep means that chunk was never written. Far cheaper than
    stat-ing every chunk file, and it checks the thing that actually matters.
    """
    import zarr

    root = zarr.open_group(str(store), mode="r")
    tiles = sorted(root.group_keys()) or [None]
    n_times = root["time"].shape[0]
    var = None
    probe = root if tiles == [None] else root[tiles[0]]
    for v in ("u", "v", "w", "k"):
        if v in probe.array_keys():
            var = v
            break
    step = max(1, n_times // 10)
    idx = list(range(0, n_times, step)) + [n_times - 1]
    print(f"{store.name}: {len(tiles)} tile(s), {n_times:,} timesteps, "
          f"sampling {len(idx)} of them in '{var}'")

    bad = []
    for t in tiles:
        g = root if t is None else root[t]
        arr = g[var]
        for i in idx:
            if not np.any(np.asarray(arr[i])):
                bad.append((t, i))
                break
    if bad:
        print(f"  INCOMPLETE: {len(bad)} tile(s) have an all-zero sampled timestep")
        for t, i in bad[:10]:
            print(f"    {t} empty at time index {i}")
        return 1
    print("  all sampled timesteps non-empty in every tile")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("store", type=Path)
    ap.add_argument("--watch", type=float, default=0.0, help="refresh every N seconds")
    ap.add_argument("--fast", action="store_true",
                    help="sample the data instead of counting chunk files. Counting "
                         "is O(tiles x chunks) stats -- 521k on a 238-tile domain, "
                         "minutes on Lustre. Sampling reads ~20 slices and detects "
                         "unwritten chunks because zarr returns fill_value (0) there.")
    args = ap.parse_args()

    if args.fast:
        return fast_check(args.store)

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
