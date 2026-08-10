"""Phase 0 validation: prove the pipeline before spending a GPU-week on it.

Six checks, in increasing order of what they can tell you:

1. **Upstream parity** -- our port reproduces the shipped example. If this fails,
   nothing else is meaningful.
2. **Error floor** -- FuXi-CFD's own RMSE against the CFD it was trained on. The
   ceiling on any accuracy claim downstream.
3. **Distribution shift** -- how far Broadview's terrain sits from the training
   distribution, in sigma. Broadview is ~+4 sigma in elevation and ~-4 sigma in
   roughness, so this is expected to look bad; the point is to quantify it.
4. **Physical closure** -- does the tile-mean FuXi-CFD wind near 100 m come back to
   the WRF wind that was fed in? Reported per component, because a v-only sign
   error is the signature of a flipped y axis (see grids.py).
5. **Profile agreement** -- tile-mean FuXi-CFD vertical profile against WRF's own
   9-height profile. Independent physical evidence, using data already on disk.
6. **dem_mode arbitration** -- runs check 5 under both `absolute` and `recentered`
   and reports which is closer, so the choice is made on evidence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import RunConfig
from .constants import OUTPUT_HEIGHTS, OUTPUT_VARS, WRF_WIND_HEIGHTS
from .grids import DomainGeometry, Tile
from .model import assets as assets_mod
from .model import postprocess, preprocess
from .model.session import FuxiCFDSession
from .terrain.build import build_terrain, sigma_offsets
from .winds import speeddir_to_uv, stencil_winds, uv_to_speeddir

#: Model level closest to the 100 m WRF input, used for the closure check.
CLOSURE_LEVEL = int(np.argmin(np.abs(OUTPUT_HEIGHTS - 100.0)))  # 102.18 m


@dataclass
class Report:
    checks: dict = field(default_factory=dict)

    def add(self, name: str, passed: bool | None, detail: dict) -> None:
        self.checks[name] = {"passed": passed, **detail}

    def to_json(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(self.checks, fh, indent=2, default=float)

    def render(self) -> str:
        lines = []
        for name, c in self.checks.items():
            status = {True: "PASS", False: "FAIL", None: "INFO"}[c.get("passed")]
            lines.append(f"\n[{status}] {name}")
            for k, v in c.items():
                if k == "passed":
                    continue
                if isinstance(v, dict):
                    lines.append(f"    {k}:")
                    lines += [f"        {kk}: {vv}" for kk, vv in v.items()]
                else:
                    lines.append(f"    {k}: {v}")
        return "\n".join(lines)


# --- checks 1 & 2 ------------------------------------------------------------

def check_upstream_parity(assets_root=None, *, device: str = "cpu") -> tuple[dict, dict]:
    A = assets_mod.resolve(assets_root)
    A.require(A.onnx, A.example_inputs, A.example_prediction, A.example_outputs)
    in_stats, out_stats = A.load_input_scaler(), A.load_output_scaler()

    z = np.load(A.example_inputs)
    x = preprocess.build_model_input({k: z[k] for k in z.files}, in_stats)
    session = FuxiCFDSession(A.onnx, device=device)
    ours = postprocess.denormalize_and_split(session.run(x[None]), out_stats)

    theirs = np.load(A.example_prediction)
    parity = {
        v: {"max_abs_diff": float(np.abs(ours[v] - theirs[v]).max())}
        for v in OUTPUT_VARS
    }

    label = np.load(A.example_outputs)
    floor = {}
    for v in OUTPUT_VARS:
        truth = np.transpose(label[v], (2, 0, 1))[:, :300, :300]
        floor[v] = {
            "rmse": float(np.sqrt(np.mean((ours[v] - truth) ** 2))),
            "bias": float(np.mean(ours[v] - truth)),
            "label_std": float(truth.std()),
        }

    # Parity is good if our discrepancy is negligible against the model's own RMSE.
    ok = all(
        parity[v]["max_abs_diff"] < 0.01 * floor[v]["rmse"] for v in OUTPUT_VARS
    )
    return (
        {"passed": ok, "per_var": parity, "dynamic_batch": session.dynamic_batch,
         "provider": session.active_provider},
        {"passed": None, "per_var": floor,
         "note": "Upper bound on achievable accuracy; WRF-driven error cannot beat this."},
    )


# --- check 3 -----------------------------------------------------------------

def check_distribution(tiles: list[Tile], cache_root: Path, in_stats: dict) -> dict:
    per_tile, terrains = {}, {}
    for t in tiles:
        terrain = build_terrain(t, cache_root)
        terrains[t.tile_id] = terrain
        per_tile[t.tile_id] = {**sigma_offsets(terrain, in_stats), **terrain.stats()}

    worst = max(
        abs(v["dem_sigma_mean"]) for v in per_tile.values()
    ), max(abs(v["z0_sigma_mean"]) for v in per_tile.values())
    return {
        "passed": None,
        "per_tile": per_tile,
        "worst_abs_sigma_dem": worst[0],
        "worst_abs_sigma_z0": worst[1],
        "interpretation": (
            "Values beyond ~3 sigma mean the model is extrapolating. Broadview is "
            "expected to be ~+4 sigma in elevation and ~-4 sigma in roughness. This "
            "is why terrain.dem_mode exists -- see check 6."
        ),
    }, terrains


# --- checks 4 & 5 ------------------------------------------------------------

def _tile_mean_profile(fields: dict[str, np.ndarray]) -> np.ndarray:
    """Tile-mean horizontal wind speed per level, (n_time, 27).

    Speed is computed pointwise then averaged -- averaging u and v first would
    report the speed of the mean flow, which is smaller wherever direction varies.
    """
    speed = np.hypot(fields["u"], fields["v"])
    return speed.mean(axis=(-2, -1))


def run_inference_sample(
    prep, tiles: list[Tile], rows: np.ndarray, session: FuxiCFDSession,
    statics: dict[str, np.ndarray],
) -> dict[str, dict[str, np.ndarray]]:
    """Infer a small (tile x time) sample, returning per-tile field dicts."""
    speed, direction = prep.source.read_wind_100m(rows)
    # Respect the cuDNN batch cap: --n-times is user-supplied and routinely
    # exceeds it, and feeding it straight to session.run would fail at 32.
    batch = session.max_batch(len(rows))
    out = {}
    for t in tiles:
        interp = prep.interpolators[t.tile_id]
        u9, v9 = stencil_winds(interp, speed, direction)
        winds = preprocess.normalize_winds(u9, v9, prep.in_stats)
        x = preprocess.assemble(winds, statics[t.tile_id])
        pred = np.concatenate(
            [
                postprocess.denormalize(session.run(x[i : i + batch]), prep.out_stats)
                for i in range(0, x.shape[0], batch)
            ],
            axis=0,
        )
        out[t.tile_id] = {"fields": postprocess.split(pred), "u9": u9, "v9": v9}
    return out


def check_closure(sample: dict) -> dict:
    """Does the tile-mean wind at ~102 m recover the WRF input wind?

    u and v are reported separately on purpose: a flipped y axis shows up as a
    v-only sign error while u looks perfect.
    """
    per_tile = {}
    for tile_id, s in sample.items():
        f = s["fields"]
        pred_u = f["u"][:, CLOSURE_LEVEL].mean(axis=(-2, -1))
        pred_v = f["v"][:, CLOSURE_LEVEL].mean(axis=(-2, -1))
        in_u = s["u9"].mean(axis=(-2, -1))
        in_v = s["v9"].mean(axis=(-2, -1))
        per_tile[tile_id] = {
            "u_bias": float(np.mean(pred_u - in_u)),
            "v_bias": float(np.mean(pred_v - in_v)),
            "u_rmse": float(np.sqrt(np.mean((pred_u - in_u) ** 2))),
            "v_rmse": float(np.sqrt(np.mean((pred_v - in_v) ** 2))),
            "u_corr": float(np.corrcoef(pred_u, in_u)[0, 1]) if len(pred_u) > 2 else None,
            "v_corr": float(np.corrcoef(pred_v, in_v)[0, 1]) if len(pred_v) > 2 else None,
            "input_mean_speed": float(np.mean(np.hypot(in_u, in_v))),
        }
    v_corrs = [c["v_corr"] for c in per_tile.values() if c["v_corr"] is not None]
    sign_flip = bool(v_corrs) and float(np.mean(v_corrs)) < 0
    return {
        "passed": not sign_flip,
        "level_m": float(OUTPUT_HEIGHTS[CLOSURE_LEVEL]),
        "per_tile": per_tile,
        "suspected_y_axis_flip": sign_flip,
        "interpretation": (
            "A negative v correlation with positive u correlation means the terrain "
            "grid is upside down relative to the wind stencil -- check the north-up "
            "convention in grids.py."
        ),
    }


def check_profile(sample: dict, prep, rows: np.ndarray, tiles: list[Tile]) -> dict:
    """Compare tile-mean FuXi-CFD profile against WRF's own 9-height profile."""
    wrf_vars = [f"wind_speed_{h}m" for h in WRF_WIND_HEIGHTS]
    available = [v for v in wrf_vars if v in getattr(prep.source, "variables", [])]
    heights = [h for h, v in zip(WRF_WIND_HEIGHTS, wrf_vars) if v in available]
    if not available:
        return {"passed": None, "note": "no multi-height WRF wind speeds in source"}

    data = prep.source.read(available, rows)
    per_tile = {}
    for t in tiles:
        interp = prep.interpolators[t.tile_id]
        wrf_prof = np.stack(
            [interp.apply(data[v]).mean(axis=(-2, -1)) for v in available], axis=-1
        )  # (n_time, n_heights)
        fuxi_prof = _tile_mean_profile(sample[t.tile_id]["fields"])  # (n_time, 27)
        interp_prof = np.stack(
            [np.interp(heights, OUTPUT_HEIGHTS, fuxi_prof[i]) for i in range(fuxi_prof.shape[0])]
        )
        diff = interp_prof - wrf_prof
        per_tile[t.tile_id] = {
            "heights_m": heights,
            "wrf_mean": wrf_prof.mean(axis=0).round(4).tolist(),
            "fuxi_mean": interp_prof.mean(axis=0).round(4).tolist(),
            "bias_per_height": diff.mean(axis=0).round(4).tolist(),
            "rmse_overall": float(np.sqrt(np.mean(diff**2))),
            "mean_abs_bias": float(np.abs(diff.mean(axis=0)).mean()),
        }
    return {
        "passed": None,
        "per_tile": per_tile,
        "interpretation": (
            "FuXi-CFD should track the WRF profile shape while adding sub-km "
            "structure. Large monotonic bias suggests the static-channel "
            "distribution shift is degrading the model."
        ),
    }
