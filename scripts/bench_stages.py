"""Time each stage of the inference loop, to see what actually limits throughput.

    python scripts/bench_stages.py configs/wsw_box_nlcd.yaml --tiles 8 --steps 48

The production array runs at ~15 tile-steps/s per task against a headline 42 on
an idle H100, and the packing arithmetic (4 tasks x 32 CPUs = a full 128-core
node) suggests the gap is CPU preprocessing starving the GPU rather than
anything about the model. That is a guess until measured, and the fix differs
completely depending on the answer: more GPUs if the model dominates, vectorised
preprocessing if it does not.

Stages timed, matching pipeline.run_shard:

  read        h5 -> wind fields for a block of timesteps      (I/O)
  stencil     sparse matvec onto the 9x9 coarse stencil       (CPU)
  preprocess  normalize + zoom 9->300 and assemble (4,300,300)(CPU)
  onnx        session.run                                     (GPU)
  postprocess denormalize + split                             (CPU)

Runs single-threaded-per-stage on one process, so the numbers are per-task rates
and directly comparable to the `tile-steps/s` the run log prints.
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from fuxicfd_wrf.config import RunConfig
from fuxicfd_wrf.model import postprocess, preprocess
from fuxicfd_wrf.pipeline import prepare
from fuxicfd_wrf.winds import stencil_winds


class Timer:
    def __init__(self):
        self.t = defaultdict(float)
        self.n = defaultdict(int)

    def __call__(self, key):
        self._key = key
        return self

    def __enter__(self):
        self._t0 = time.perf_counter()

    def __exit__(self, *exc):
        self.t[self._key] += time.perf_counter() - self._t0
        self.n[self._key] += 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    ap.add_argument("--tiles", type=int, default=8)
    ap.add_argument("--steps", type=int, default=48)
    ap.add_argument("--batch", type=int, default=None)
    args = ap.parse_args()

    cfg = RunConfig.from_yaml(args.config)
    prep = prepare(cfg, build_static=True)
    tiles = prep.tiles[: args.tiles]
    batch = args.batch or cfg.model.batch_size
    from fuxicfd_wrf.model import assets as assets_mod
    from fuxicfd_wrf.model.session import FuxiCFDSession
    session = FuxiCFDSession(
        assets_mod.resolve(cfg.model.assets_root).onnx, device=cfg.model.device
    )
    batch = session.max_batch(batch)
    print(f"[bench] {len(tiles)} tiles x {args.steps} steps, batch {batch}")
    print(f"[bench] {session.describe()}")

    T = Timer()
    rows = prep.rows[: args.steps]
    with T("read"):
        speed_all, dir_all = prep.source.read_wind_100m(rows)

    n_done = 0
    for i in range(0, len(rows), batch):
        sl = slice(i, min(i + batch, len(rows)))
        for tile in tiles:
            interp = prep.interpolators[tile.tile_id]
            with T("stencil"):
                u9, v9 = stencil_winds(interp, speed_all[sl], dir_all[sl])
            with T("preprocess"):
                winds = preprocess.normalize_winds(u9, v9, prep.in_stats)
                x = preprocess.assemble(winds, prep.statics[tile.tile_id])
            with T("onnx"):
                raw = session.run(x)
            with T("postprocess"):
                pred = postprocess.denormalize(raw, prep.out_stats)
                postprocess.split(pred)
            n_done += x.shape[0]

    total = sum(T.t.values())
    print(f"\n{'stage':<14}{'total s':>10}{'ms/tile-step':>15}{'share':>9}")
    print("-" * 48)
    for k in ("read", "stencil", "preprocess", "onnx", "postprocess"):
        if k not in T.t:
            continue
        print(f"{k:<14}{T.t[k]:10.2f}{1000 * T.t[k] / n_done:15.2f}"
              f"{100 * T.t[k] / total:8.1f}%")
    print("-" * 48)
    print(f"{'TOTAL':<14}{total:10.2f}{1000 * total / n_done:15.2f}"
          f"{'':>9}\n{'rate':<14}{n_done / total:10.2f} tile-steps/s")

    cpu = sum(T.t[k] for k in ("stencil", "preprocess", "postprocess") if k in T.t)
    gpu = T.t.get("onnx", 0.0)
    print(f"\nCPU stages {cpu:.2f}s vs GPU {gpu:.2f}s -- "
          f"{'CPU-bound' if cpu > gpu else 'GPU-bound'}, ratio {cpu / max(gpu, 1e-9):.2f}x")
    if cpu > gpu:
        print(f"Perfectly overlapping CPU and GPU would cap at "
              f"{n_done / cpu:.1f} tile-steps/s (CPU is the floor).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
