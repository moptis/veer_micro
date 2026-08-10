"""Zarr output stores.

Site mode writes one group per tile::

    <store>.zarr/<tile_id>/{u,v,w,k}   (time, level, y, x)

Mosaic mode writes a single domain-wide 30 m grid, compositing overlapping tiles
with a cosine-squared taper.

Why float16 is the default
--------------------------
27 levels x 4 vars x 300 x 300 is 38.9 MB per tile-timestep at float32, 19.4 MB at
float16. Across a long time series that difference is measured in terabytes.
float16 carries ~3 decimal digits, which is well below FuXi-CFD's own error
against the CFD it was trained on -- the precision lost is precision the model
never had. Wind speeds of O(10 m/s) resolve to ~0.005 m/s.

Chunk/shard alignment
---------------------
Slurm array tasks write disjoint time ranges into one store concurrently. Zarr has
no locking, so two tasks touching the same chunk would race and silently corrupt
it. `shard_bounds` splits work on chunk boundaries, and `TileStore.write_block`
rejects any write that straddles one.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

from .constants import MODEL_SHAPE, OUTPUT_HEIGHTS, OUTPUT_VARS, TILE_SPAN_M
from .levels import LevelSelection
from .grids import Tile


# --- zarr 2 / 3 compatibility -----------------------------------------------

_ZARR_MAJOR = int(zarr.__version__.split(".")[0])


#: Errors raised when a concurrent task created the node first.
def _conflict_errors() -> tuple[type[Exception], ...]:
    import zarr.errors as ze

    names = ("ContainsArrayError", "ContainsGroupError", "NodeExistsError")
    found = tuple(getattr(ze, n) for n in names if hasattr(ze, n))
    return found + (FileExistsError,)


_CONFLICT_ERRORS = _conflict_errors()
_CREATE_ATTEMPTS = 5


def _open_group(path: Path, mode: str):
    """Open (or create) the root group, tolerating concurrent creation.

    ``zarr.open_group(mode="a")`` is itself check-then-act at the root: several
    Slurm tasks starting together can all find the store absent and all try to
    write ``zarr.json``, and the losers get ``ContainsGroupError`` at path ''.
    This is the same race as `_create_array`, one level up, and it is the one the
    first production launch actually hit second.
    """
    if mode == "r":
        return zarr.open_group(str(path), mode=mode)

    last_exc = None
    for _ in range(_CREATE_ATTEMPTS):
        try:
            return zarr.open_group(str(path), mode=mode)
        except _CONFLICT_ERRORS as exc:
            last_exc = exc
            # Someone else just created it; re-opening is exactly what we want.
            try:
                return zarr.open_group(str(path), mode="r+")
            except Exception:  # noqa: BLE001 - still mid-write, back off
                time.sleep(0.2 + random.random() * 0.3)
    raise RuntimeError(f"could not open or create zarr group at {path}") from last_exc


def _create_array(group, name, *, shape, chunks, dtype, fill_value=0):
    """Create an array idempotently, spanning the zarr 2 and zarr 3 APIs.

    Must tolerate many Slurm array tasks entering at once. A bare
    ``if name in group`` check is check-then-act: several tasks can all see the
    node as absent and all try to create it, and every loser gets
    ``ContainsArrayError``. That took out 11 of 16 tasks in the first production
    launch. Losing the race is the *normal* outcome here, not an error -- the
    winner's array is exactly the one we want -- so conflicts are caught and the
    existing node reused.
    """
    last_exc = None
    for _ in range(_CREATE_ATTEMPTS):
        if name in group:
            return group[name]
        try:
            if _ZARR_MAJOR >= 3:
                return group.create_array(
                    name, shape=shape, chunks=chunks, dtype=dtype, fill_value=fill_value
                )
            return group.create_dataset(
                name, shape=shape, chunks=chunks, dtype=dtype,
                fill_value=fill_value, overwrite=False,
            )
        except _CONFLICT_ERRORS as exc:
            last_exc = exc
            # Another task is mid-create; give the metadata write a moment to land.
            time.sleep(0.2 + random.random() * 0.3)
    if name in group:
        return group[name]
    raise RuntimeError(
        f"could not create or open array {name!r} after {_CREATE_ATTEMPTS} attempts"
    ) from last_exc


def _require_group(root, name):
    """Idempotent group creation under the same concurrency conditions."""
    last_exc = None
    for _ in range(_CREATE_ATTEMPTS):
        try:
            return root.require_group(name)
        except _CONFLICT_ERRORS as exc:
            last_exc = exc
            if name in root:
                return root[name]
            time.sleep(0.2 + random.random() * 0.3)
    if name in root:
        return root[name]
    raise RuntimeError(f"could not create or open group {name!r}") from last_exc


def _set_attrs(node, **attrs) -> None:
    for k, v in attrs.items():
        node.attrs[k] = v


# --- time encoding -----------------------------------------------------------

def to_epoch_ns(times: pd.DatetimeIndex) -> np.ndarray:
    """Encode timestamps as epoch nanoseconds.

    Do not use ``DatetimeIndex.view("int64")`` here. pandas may hold the index at
    microsecond resolution, in which case ``view`` reinterprets the raw buffer and
    silently yields microsecond counts that later decode as 1970 -- an off-by-1000
    that produces plausible-looking dates rather than an error. Converting the unit
    explicitly is the only safe route.
    """
    idx = pd.DatetimeIndex(times)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    return idx.to_numpy(dtype="datetime64[ns]").astype("int64")


def from_epoch_ns(values) -> pd.DatetimeIndex:
    """Inverse of `to_epoch_ns`; returns a UTC-aware index."""
    return pd.to_datetime(np.asarray(values), unit="ns", utc=True)


# --- level selection ---------------------------------------------------------

def resolve_levels(requested: list[float] | None) -> "LevelSelection":
    """Build the level selection for `requested` heights (None = all 27).

    Requests that do not match a model level are interpolated rather than snapped:
    labelling an interpolated 80 m field as 80 m is honest, whereas snapping to
    83.578 m and calling it 80 m is not. See levels.py.
    """
    return LevelSelection.build(OUTPUT_HEIGHTS, requested)


def shard_bounds(n_times: int, time_chunk: int, n_shards: int) -> list[tuple[int, int]]:
    """Split [0, n_times) into `n_shards` ranges aligned to `time_chunk`.

    Alignment is what makes concurrent Slurm writes safe. Empty shards are
    dropped, so asking for more shards than there are chunks is harmless.
    """
    n_chunks = int(np.ceil(n_times / time_chunk))
    per = int(np.ceil(n_chunks / max(1, n_shards)))
    out = []
    for s in range(n_shards):
        c0 = s * per
        if c0 >= n_chunks:
            break
        c1 = min(c0 + per, n_chunks)
        out.append((c0 * time_chunk, min(c1 * time_chunk, n_times)))
    return out


class TileStore:
    """Per-tile Zarr group holding (time, level, y, x) fields."""

    def __init__(
        self,
        path: Path | str,
        *,
        times: pd.DatetimeIndex,
        tiles: list[Tile],
        levels: np.ndarray,
        variables: list[str] = list(OUTPUT_VARS),
        dtype: str = "float16",
        time_chunk: int = 6,
    ):
        self.path = Path(path)
        self.times = times
        self.tiles = {t.tile_id: t for t in tiles}
        self.levels = np.asarray(levels)
        self.variables = list(variables)
        self.dtype = dtype
        self.time_chunk = time_chunk

    def create(self, *, config_json: str | None = None) -> None:
        """Initialise the store. Safe to call from many tasks; existing arrays are kept."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        root = _open_group(self.path, mode="a")
        nt, nl = len(self.times), len(self.levels)
        h, w = MODEL_SHAPE

        _set_attrs(
            root,
            created_by="fuxicfd-wrf",
            model="FuXi-CFD v1.0 (10.5281/zenodo.18770845)",
            license="CC BY-NC 4.0 -- non-commercial use only",
            levels_m=self.levels.tolist(),
            variables=self.variables,
            time_units="UTC",
            orientation="north-up: axis -2 is y with row 0 northmost, axis -1 is x eastward",
            note=(
                "Times are the WRF timestamps in chronological order. The source "
                "control.h5 is NOT chronologically ordered; do not assume this "
                "index maps to its row order."
            ),
        )
        if config_json:
            root.attrs["run_config"] = config_json

        # Times as epoch nanoseconds -- zarr has no datetime64 dtype.
        if "time" not in root:
            t = _create_array(
                root, "time", shape=(nt,), chunks=(min(nt, 8192),), dtype="int64"
            )
            t[:] = to_epoch_ns(self.times)
            _set_attrs(t, units="nanoseconds since 1970-01-01T00:00:00Z")
            # Read back and compare. A unit mismatch decodes to plausible-looking
            # dates rather than failing, so it must be checked, not assumed.
            # Verify the unit encoding round-trips. Only meaningful if we are the
            # task that wrote it; a concurrent writer can leave a partial read.
            decoded = from_epoch_ns(t[:])
            expected = pd.DatetimeIndex(self.times).tz_convert("UTC")
            if not decoded.equals(expected) and decoded[0] != expected[0]:
                raise RuntimeError(
                    "time axis failed to round-trip through the store: wrote "
                    f"{self.times[0]}, read back {decoded[0]}"
                )
        if "level" not in root:
            lv = _create_array(root, "level", shape=(nl,), chunks=(nl,), dtype="float64")
            lv[:] = self.levels
            _set_attrs(lv, units="m", positive="up", description="height above ground level")

        for tile_id, tile in self.tiles.items():
            g = _require_group(root, tile_id)
            _set_attrs(
                g,
                tile_id=tile_id,
                crs=tile.crs,
                center_x=tile.x,
                center_y=tile.y,
                bounds=list(tile.bounds()),
            )
            xs, ys = tile.model_axes()
            for name, vals in (("x", xs), ("y", ys)):
                if name not in g:
                    a = _create_array(
                        g, name, shape=(len(vals),), chunks=(len(vals),), dtype="float64"
                    )
                    a[:] = vals
                    _set_attrs(a, units="m", crs=tile.crs)

            for var in self.variables:
                if var not in g:
                    a = _create_array(
                        g, var,
                        shape=(nt, nl, h, w),
                        chunks=(self.time_chunk, nl, h, w),
                        dtype=self.dtype,
                    )
                    _set_attrs(a, **_VAR_ATTRS[var])

    def write_block(self, tile_id: str, t0: int, fields: dict[str, np.ndarray]) -> None:
        """Write `fields` (each (n, n_level, 300, 300)) starting at time index `t0`.

        Refuses writes that straddle a chunk boundary at either end, since that is
        exactly the condition under which two Slurm tasks can corrupt each other.
        """
        n = next(iter(fields.values())).shape[0]
        t1 = t0 + n
        if t0 % self.time_chunk != 0:
            raise ValueError(
                f"write starts at t0={t0}, not a multiple of time_chunk="
                f"{self.time_chunk}; concurrent shards could race on that chunk"
            )
        if t1 % self.time_chunk != 0 and t1 != len(self.times):
            raise ValueError(
                f"write ends at t1={t1}, which is neither a chunk boundary nor the "
                f"end of the time axis ({len(self.times)})"
            )

        root = _open_group(self.path, mode="a")
        g = root[tile_id]
        for var, arr in fields.items():
            if var not in self.variables:
                continue
            g[var][t0:t1] = arr.astype(self.dtype)


_VAR_ATTRS = {
    "u": {"long_name": "eastward wind (tile grid frame)", "units": "m s-1"},
    "v": {"long_name": "northward wind (tile grid frame)", "units": "m s-1"},
    "w": {"long_name": "vertical wind", "units": "m s-1"},
    "k": {"long_name": "turbulent kinetic energy", "units": "m2 s-2"},
}


# --- mosaic ------------------------------------------------------------------

def cosine_taper(shape: tuple[int, int], overlap_px: int) -> np.ndarray:
    """Cosine-squared edge taper used to blend overlapping tiles.

    FuXi-CFD's training CFD had a 1.5 km buffer around the 9 km output window, but
    the model itself only ever sees the 9 km. Its tile edges are therefore the
    least trustworthy part of the field, and a hard mosaic seam shows it. Weights
    go to zero at the very edge, so seam pixels are carried by the neighbouring
    tile's interior.
    """
    h, w = shape
    if overlap_px <= 0:
        return np.ones(shape, dtype=np.float32)

    def ramp(n: int) -> np.ndarray:
        r = np.ones(n, dtype=np.float32)
        m = min(overlap_px, n // 2)
        if m <= 0:
            return r
        t = (np.arange(m) + 0.5) / m
        edge = np.sin(0.5 * np.pi * t) ** 2
        r[:m] = edge
        r[-m:] = edge[::-1]
        return r

    return np.outer(ramp(h), ramp(w)).astype(np.float32)


def _interp_matrix(dst_frac: np.ndarray, n_src: int) -> np.ndarray:
    """Dense 1-D linear interpolation matrix from source indices to `dst_frac`.

    Separable resampling: applying this along y then x is equivalent to a 2-D
    bilinear map, and at 300x300 the dense matmuls are free.
    """
    i0 = np.clip(np.floor(dst_frac).astype(int), 0, n_src - 2)
    f = (dst_frac - i0).astype(np.float32)
    M = np.zeros((len(dst_frac), n_src), dtype=np.float32)
    rows = np.arange(len(dst_frac))
    M[rows, i0] = 1.0 - f
    M[rows, i0 + 1] = f
    return M


@dataclass
class _TilePlacement:
    """Precomputed mapping of one tile onto the domain grid."""

    row: slice
    col: slice
    wy: np.ndarray   # (n_dst_row, 300)
    wx: np.ndarray   # (n_dst_col, 300)
    weight: np.ndarray  # (n_dst_row, n_dst_col) blend weight


class MosaicStore:
    """Single domain-wide 30 m store, compositing overlapping tiles.

    Why tiles need resampling rather than slotting into place
    ---------------------------------------------------------
    Each tile's output sits on its own 30.1003 m lattice centred on that tile (see
    `Tile.model_axes`). With a 6000 m stride, 6000 / 30.1003 = 199.3 pixels -- not
    an integer -- so no two tiles share a common lattice and none aligns with a
    round 30 m domain grid. Pasting tiles in by nearest pixel would scatter up to
    half a pixel of position error per tile.

    So the domain grid is defined at exactly `res_m`, and every tile is bilinearly
    resampled onto it via precomputed separable weights.

    Blending
    --------
    FuXi-CFD sees only its own 9 km window; its training CFD had a 1.5 km buffer
    that the model never gets. Tile edges are therefore its least reliable region,
    and a hard seam shows it. Overlapping tiles are combined with a cosine-squared
    taper that goes to zero at the edge, so seam pixels are carried by the
    neighbouring tile's interior. Weights are accumulated and divided out, so the
    result is a proper weighted mean even where three or four tiles overlap.
    """

    def __init__(
        self,
        path: Path | str,
        *,
        times: pd.DatetimeIndex,
        tiles: list[Tile],
        levels: np.ndarray,
        variables: list[str] = list(OUTPUT_VARS),
        dtype: str = "float16",
        time_chunk: int = 6,
        res_m: float = 30.0,
        stride_m: float | None = None,
        crs: str = None,
    ):
        if not tiles:
            raise ValueError("mosaic needs at least one tile")
        self.path = Path(path)
        self.times = times
        self.tiles = tiles
        self.levels = np.asarray(levels)
        self.variables = list(variables)
        self.dtype = dtype
        self.time_chunk = time_chunk
        self.res_m = float(res_m)
        self.crs = crs or tiles[0].crs

        half = TILE_SPAN_M / 2
        self.x0 = min(t.x for t in tiles) - half
        self.y1 = max(t.y for t in tiles) + half
        x1 = max(t.x for t in tiles) + half
        y0 = min(t.y for t in tiles) - half
        self.nx = int(round((x1 - self.x0) / self.res_m)) + 1
        self.ny = int(round((self.y1 - y0) / self.res_m)) + 1

        # Taper width in domain pixels, from the tile overlap.
        overlap_m = TILE_SPAN_M - float(stride_m or TILE_SPAN_M)
        self.overlap_px = max(0, int(round(overlap_m / self.res_m)))

        self._placements = {t.tile_id: self._place(t) for t in tiles}

    @property
    def x(self) -> np.ndarray:
        return self.x0 + np.arange(self.nx) * self.res_m

    @property
    def y(self) -> np.ndarray:
        return self.y1 - np.arange(self.ny) * self.res_m

    def _place(self, tile: Tile) -> _TilePlacement:
        tx, ty = tile.model_axes()          # ascending x, descending y
        minx, maxx = tx[0], tx[-1]
        miny, maxy = ty[-1], ty[0]

        c0 = int(np.ceil((minx - self.x0) / self.res_m - 1e-9))
        c1 = int(np.floor((maxx - self.x0) / self.res_m + 1e-9))
        r0 = int(np.ceil((self.y1 - maxy) / self.res_m - 1e-9))
        r1 = int(np.floor((self.y1 - miny) / self.res_m + 1e-9))
        c0, c1 = max(c0, 0), min(c1, self.nx - 1)
        r0, r1 = max(r0, 0), min(r1, self.ny - 1)

        dst_x = self.x0 + np.arange(c0, c1 + 1) * self.res_m
        dst_y = self.y1 - np.arange(r0, r1 + 1) * self.res_m
        res = tile.model_res_m
        wx = _interp_matrix((dst_x - tx[0]) / res, MODEL_SHAPE[1])
        wy = _interp_matrix((ty[0] - dst_y) / res, MODEL_SHAPE[0])

        weight = cosine_taper((len(dst_y), len(dst_x)), self.overlap_px)
        return _TilePlacement(
            row=slice(r0, r1 + 1), col=slice(c0, c1 + 1), wy=wy, wx=wx, weight=weight
        )

    def create(self, *, config_json: str | None = None) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        root = _open_group(self.path, mode="a")
        nt, nl = len(self.times), len(self.levels)

        _set_attrs(
            root,
            created_by="fuxicfd-wrf",
            model="FuXi-CFD v1.0 (10.5281/zenodo.18770845)",
            license="CC BY-NC 4.0 -- non-commercial use only",
            mosaic=True,
            crs=self.crs,
            resolution_m=self.res_m,
            overlap_px=self.overlap_px,
            n_tiles=len(self.tiles),
            levels_m=self.levels.tolist(),
            orientation="north-up: axis -2 is y with row 0 northmost, axis -1 is x eastward",
            note=(
                "Tiles were bilinearly resampled from their native 30.1003 m grids "
                "onto this exact-30 m grid and combined with a cosine-squared taper."
            ),
        )
        if config_json:
            root.attrs["run_config"] = config_json

        if "time" not in root:
            t = _create_array(root, "time", shape=(nt,), chunks=(min(nt, 8192),), dtype="int64")
            t[:] = to_epoch_ns(self.times)
            _set_attrs(t, units="nanoseconds since 1970-01-01T00:00:00Z")
        if "level" not in root:
            lv = _create_array(root, "level", shape=(nl,), chunks=(nl,), dtype="float64")
            lv[:] = self.levels
            _set_attrs(lv, units="m", positive="up")
        for name, vals in (("x", self.x), ("y", self.y)):
            if name not in root:
                a = _create_array(root, name, shape=(len(vals),), chunks=(len(vals),), dtype="float64")
                a[:] = vals
                _set_attrs(a, units="m", crs=self.crs)

        for var in self.variables:
            if var not in root:
                a = _create_array(
                    root, var,
                    shape=(nt, nl, self.ny, self.nx),
                    chunks=(self.time_chunk, nl, min(self.ny, 512), min(self.nx, 512)),
                    dtype=self.dtype,
                )
                _set_attrs(a, **_VAR_ATTRS[var])

    def blend(self, per_tile: dict[str, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
        """Composite per-tile fields into domain arrays.

        `per_tile[tile_id][var]` is (n_time, n_level, 300, 300). Returns
        {var: (n_time, n_level, ny, nx)} as float32.
        """
        any_tile = next(iter(per_tile.values()))
        nt, nl = next(iter(any_tile.values())).shape[:2]
        acc = {
            v: np.zeros((nt, nl, self.ny, self.nx), dtype=np.float32)
            for v in self.variables
        }
        wsum = np.zeros((self.ny, self.nx), dtype=np.float32)

        for tile_id, fields in per_tile.items():
            p = self._placements[tile_id]
            for var in self.variables:
                src = np.asarray(fields[var], dtype=np.float32)
                # separable bilinear: (..., 300, 300) -> (..., ndy, ndx)
                tmp = np.einsum("ij,ntjk->ntik", p.wy, src)
                dst = np.einsum("ntik,lk->ntil", tmp, p.wx)
                acc[var][:, :, p.row, p.col] += dst * p.weight
            wsum[p.row, p.col] += p.weight

        safe = np.where(wsum > 0, wsum, 1.0)
        for var in self.variables:
            acc[var] /= safe
            acc[var][:, :, wsum == 0] = np.nan
        return acc

    def write_block(self, t0: int, fields: dict[str, np.ndarray]) -> None:
        n = next(iter(fields.values())).shape[0]
        t1 = t0 + n
        if t0 % self.time_chunk != 0:
            raise ValueError(
                f"mosaic write starts at t0={t0}, not a multiple of time_chunk="
                f"{self.time_chunk}; concurrent shards could race on that chunk"
            )
        root = _open_group(self.path, mode="a")
        for var, arr in fields.items():
            if var in self.variables:
                root[var][t0:t1] = arr.astype(self.dtype)

    def coverage(self) -> np.ndarray:
        """Summed blend weight per domain pixel. Zeros mark uncovered gaps."""
        w = np.zeros((self.ny, self.nx), dtype=np.float32)
        for p in self._placements.values():
            w[p.row, p.col] += p.weight
        return w
