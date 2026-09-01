# fuxicfd-wrf

Drive the pre-trained [FuXi-CFD](https://doi.org/10.5281/zenodo.18770845) model with
WRF output to reconstruct 30 m, 27-level 3D wind fields (`u, v, w, k`).

Built for the Broadview, NM domain (1 km WRF d03), but the geometry layer is generic.

```bash
conda env create -f env/environment.yml
conda activate fuxicfd
pip install -e .

fuxicfd-wrf fetch-model                          # ~889 MB ONNX + scalers
fuxicfd-wrf inspect /path/to/control.h5          # grid + time structure
fuxicfd-wrf plan    configs/broadview_sites.yaml # tile/time counts, output size
sbatch scripts/validate_gpu.sh                   # Phase 0 -- read this before scaling
sbatch --array=0-15 scripts/submit_inference.sh configs/broadview_sites.yaml
```

## What the model actually requires

The model card, the paper, and the shipped inference code disagree in places. The
code is authoritative — it produced the shipped predictions. What it requires:

| Input | Shape | Units |
|---|---|---|
| `u_100m`, `v_100m` | `(9, 9)` @ 1 km | m/s |
| `dem` | `(301, 301)` @ 30 m | m (absolute elevation) |
| `roughness` | `(301, 301)` @ 30 m | m (z₀) |

Both grids describe the same 9 km × 9 km box. Output is `(1, 27, 4, 300, 300)` —
note the axis order is **(level, variable, y, x)**, not (variable, level, y, x).

**Output heights run 5 m → 214.29 m over 27 levels** — a uniform 5 m ladder from
5 to 100 m, then 106.5, 114.95, 125.94, 140.22, 158.78, 182.91, 214.29. Source of
truth is the paper's Supplementary Note 1.1, corroborated by the dataset README
for `outputs.npz`.

**Do not derive them from `utils/height.npy`.** That asset is the CFD solver's
full 56-level grid reaching 4 km, not the 27 saved output levels. An earlier
version of this file used `height.npy[2:29]`, giving 10 → 300.795 m — wrong by up
to 86 m, labelling the true 100 m level as 151.9 m. That came from upstream's
`inference_example/scripts/check.ipynb`, which slices it that way and looks up
45.0 m, a value that sits at a different index under each grid, so the shipped
example silently plots the wrong level and nothing in it reveals the error.
`constants.validate_heights` guards against the regression.

Two hub heights land on exact model levels and need no interpolation: **80 m**
(index 15) and **100 m** (index 19).

The model card says terrain is `(300, 300)`. It is `(301, 301)` — preprocessing
zooms 301→300.

## Verified against upstream

`tests/test_upstream_parity.py` replays the shipped example end-to-end. Measured on
this cluster:

| var | our diff vs shipped (CPU) | FuXi-CFD RMSE vs CFD truth | label σ |
|---|---|---|---|
| u | 3.8e-5 | **0.152** m/s | 1.264 |
| v | 6.5e-5 | **0.155** m/s | 1.382 |
| w | 1.4e-5 | **0.061** m/s | 0.407 |
| k | 2.4e-5 | **0.058** m²/s² | 0.273 |

That right-hand column is the **error floor**: WRF-driven output cannot be more
accurate than this. On GPU the discrepancy rises to ~0.013 m/s because H100 runs
convolutions on TF32 by default — still ~8% of the error floor, i.e. negligible in
quadrature. Tolerances in the test are provider-aware for this reason.

## ⚠️ Broadview is outside the training distribution

FuXi-CFD trained on `dem` 352 ± 228 m and `roughness` 0.82 ± 0.20 m — low-elevation,
forested/built-up complex terrain. Broadview is High Plains at 1180–1500 m,
grassland/cropland with z₀ ≈ 0.037 m. Measured per-tile offsets:

```
dem       +3.9 to +4.7 sigma
roughness -3.4 to -3.9 sigma
```

The model is extrapolating on both static channels. `terrain.dem_mode` exposes the
two defensible responses:

- `absolute` — physical values, faithful to the model card.
- `recentered` — preserve per-tile relief (what drives terrain-induced flow) but
  shift the tile mean onto the training mean.

**`recentered` is the default, chosen on evidence, not preference.** Phase 0 over 5
tiles × 24 timesteps:

| | mean profile RMSE vs WRF | mean 102 m closure bias (u, v) |
|---|---|---|
| `absolute` | 0.351 | +0.220, +0.225 m/s |
| `recentered` | **0.287** | **+0.129, +0.145** m/s |

Re-run `fuxicfd-wrf validate --compare-modes` if you change site or season — this
result is not guaranteed to transfer.

Two further caveats worth carrying:

- **The terrain here is nearly flat** — 49–228 m of relief per 9 km tile, σ ≈ 8–60 m.
  A *terrain*-informed model has little terrain signal to exploit at this site, so
  the gain over WRF may be modest. The most complex tile (`bv_northwest`, 228 m
  relief, 14% tree cover) is also where FuXi-CFD deviates most from WRF.
- **FuXi-CFD produces a more sheared profile than WRF**, biased low by ~0.1–0.3 m/s
  through 20–200 m and high by ~0.2–0.6 m/s at 250–300 m. Consistent across all
  tiles and both modes.

## Source data quirks

`control.h5` (113 GB, Broadview):

- **The time index is not chronological.** 52560 steps at 10-min cadence, stitched
  from ~12 month-long chunks scattered over 2000–2018, with 5 backward jumps.
  `control_status.csv` starts 2019-01-01 and matches neither. Always select by
  timestamp; `WrfSource.select` returns rows in chronological order.
- **The grid is ~800 m, not 1 km** (dx 802, dy 798; 78×132 = 106×64 km), so winds
  are bilinearly resampled onto the model's true 1 km stencil.
- Chunks are `(411, 81)`, so a single-timestep read costs ~17 MB of I/O to deliver
  41 KB. Reads are grouped into chunk-aligned blocks.

The per-timestep `.npz` files carry **no coordinates** — `NpzWrfSource` needs
`coordinates_from=` pointing at the h5.

## Design notes

**Orientation.** All 2-D arrays are north-up (row 0 = northmost). Upstream never
states its convention, so Phase 0 tests it: the closure check reports u and v bias
separately, and a flipped y axis appears as a v-only sign error. Measured
correlations are 0.9999 for both components — the convention is confirmed, not
assumed.

**Interpolation.** The WRF grid is affine in UTM 13N to within 10.25 m, so the 9×9
stencil is built as a sparse `(81, 10296)` bilinear operator once per tile and
reapplied as a matmul per timestep. `DomainGeometry` refuses the shortcut above a
25 m residual rather than degrading quietly — a regular *lat/lon* grid, for
contrast, misses affine by 203 m.

That residual grows with the square of the domain size, and it is what a big
domain runs into first:

| domain | extent | global fit | local fit |
|---|---|---|---|
| Broadview | 106 × 64 km | 10.25 m | — |
| WSW | 96 × 114 km | 13.9 m | — |
| **Pattern ph1** | **180 × 153 km** | **42.7 m — rejected** | **2.86 m** |

`tiling.affine_fit: local` refits the map around each query point over a 17 × 17
cell window instead of fitting one map to the whole domain. Lambert-vs-UTM misfit
is a smooth bowl, so locally it is ~0: Pattern ph1 comes down to 2.86 m, which is
the floor set by the float32 lon/lat in the store — the same number a fit in the
grid's own Lambert projection gives, i.e. as good as the coordinates allow.

Use it for any domain the guard rejects. It is not the default because it changes
the interpolation weights everywhere, so a store built under one setting is not
bit-comparable with one built under the other; on a domain the global fit already
handles the two agree to well within that fit's own residual. The alternative —
tiling in the WRF's native Lambert instead of UTM — fits just as well but puts the
WRG on different coordinates from the coarse deliverables, which is not worth it.

Winds are converted to components **before** interpolation. Averaging direction
across the 0/360 branch cut is a reliable way to manufacture nonsense.

**Storage.** float16, 19.44 MB per tile-timestep (float32 would be 38.9). float16
carries ~3 decimal digits — well below the model's own error floor, so the lost
precision is precision the model never had.

**Concurrency.** Slurm array tasks write disjoint time ranges to one Zarr store.
Zarr has no locking, so `shard_bounds` splits work on chunk boundaries and
`write_block` rejects any straddling write.

**Land cover → roughness.** `terrain.landcover_source` defaults to `auto`: NLCD
where it has coverage (CONUS/AK/HI, 30 m), ESA WorldCover elsewhere (global,
10 m). NLCD is preferred wherever it exists because WorldCover has a *single*
built-up class spanning everything from a farm track to a city centre, and our
lookup sends all of it to an urban z₀ of 0.8 m. Against grassland at 0.03 m that
is a 3.8σ step across one 30 m pixel, and it produced localised wind dipoles with
no terrain behind them. NLCD resolves four development intensities and gives the
same pixels a median z₀ of 0.25 m; domain-wide on WSW it yields **8× fewer
isolated z₀ spikes** (0.20% of pixels vs 1.60%).

The honest caveat: FuXi-CFD was *trained* on WorldCover-derived roughness, so
NLCD is simultaneously more accurate on the ground and further from the training
distribution. And the two disagree about far more than the speckle — on WSW,
WorldCover calls 90% of the domain grassland while NLCD calls 65% of it
shrub/scrub, a **47% shift in domain-mean z₀** affecting nearly every pixel.
`scripts/compare_sources_domain.py` measures both effects for any run; Phase 0
against the WRF profiles is what settles which input is better.

**Table provenance.** `NLCD_Z0` is UL's table — external reference values. The
repo originally shipped an in-house set assigned by analogy with WAsP roughness
classes, with no source recorded; it has been removed. It was not a small
difference: it put grassland at 0.03 against UL's 0.06 while both agree on
shrub/scrub at 0.10, and those two classes cover 92% of WSW — so it imposed
**roughly double the roughness contrast** across nearly the whole domain
(`std(ln z₀)` 0.79 vs 0.64). Flow response scales with log z₀, so that alone
changed how sensitive a run appeared to land cover, independently of the model.

`WORLDCOVER_Z0` has *not* had the same treatment and is still analogy-assigned.
It is kept because WorldCover is the global fallback and what FuXi-CFD trained
on, but it carries no more provenance than the table that was removed.

Two things to pin or key, both learned the hard way:

- **Pin `landcover_source` on any config whose store already exists.** `auto`
  resolves at run time, so a config left on `auto` can rebuild a store with
  different terrain than it originally had.
- **Derived terrain is cached, and the cache key carries a hash of the z₀ values**
  (`roughness.table_fingerprint`). A key naming only the *source* cannot see that
  the values behind it changed, and would serve tensors built with the old table.
  Editing any z₀ now invalidates the cache automatically.

## Performance

Measured on one H100 (batch 4–16 all saturate):

| | throughput |
|---|---|
| H100 | **~42 tile-steps/s** |
| 104 CPU cores | 0.91 tile-steps/s |

Batch is capped at 16: at 32 the final upconv fails with
`CUDNN_STATUS_NOT_SUPPORTED`, and throughput has already plateaued by batch 4.

The domain holds **57 tiles** edge-to-edge at 9 km, or **140** at 6 km stride
(3 km overlap, for mosaicking). Sizing at 19.44 MB per tile-timestep, all 27 levels,
52560 timesteps:

| scope | size | H100-hours |
|---|---|---|
| 1 tile, all times | 1.0 TB | 0.3 |
| 5 site tiles, all times | 5.1 TB | 1.7 |
| 57 tiles (9 km), all times | 58 TB | 20 |
| 140 tiles (6 km, mosaic), all times | 143 TB | 49 |
| 140 tiles, hourly, 7 levels (`broadview_domain.yaml`) | **6.2 TB** | 8.1 |

GPU time is cheap here; **storage is the binding constraint.** That last row is why
the shipped domain config subsamples to hourly and keeps only turbine-relevant
levels. Run `python scripts/sizing.py <config>` before launching anything.

## Output layouts

**Tile mode** (default) — one Zarr group per tile, nothing stitched:

```
<store>.zarr/<tile_id>/{u,v,w,k}    (time, level, y, x)
```

**Mosaic mode** (`output.mosaic: true`) — one domain-wide grid at exactly 30 m:

```
<store>.zarr/{u,v,w,k}              (time, level, y, x)
```

Each tile's native output sits on its own **30.1003 m** lattice (the 300 pixels
span the full 9000 m — see `Tile.model_axes`), and with a 6000 m stride
`6000 / 30.1003 = 199.3` is not an integer, so no two tiles share a lattice.
Tiles are therefore bilinearly resampled onto the common grid rather than pasted
in, then combined with a cosine-squared taper across the overlap.

Verified: mean adjacent-pixel gradient **at** tile seams is 0.0170 m/s versus
0.0182 m/s elsewhere (0.93×) — the blend leaves no seam signature.

## Reducing output size

The model computes all 27 levels unconditionally (its head is one `19200 → 108`
projection, `108 = 27×4`), so **restricting levels saves storage, not compute**.
Set `output.levels` to exact model heights:

| levels kept | per tile-timestep | 5 tiles × 52560 steps |
|---|---|---|
| all 27 | 19.44 MB | 5.1 TB |
| 7 (rotor span) | 5.04 MB | 1.3 TB |
| 1 | 0.72 MB | 189 GB |

Since the GPU cost is identical either way and you can't cheaply re-run, keeping
~5 rotor-spanning levels is usually the better trade than 1 — shear needs two
levels and turbulence intensity needs `k`.

### Restricting the area: `tiling.bbox`

Levels trade storage only. **Area trades both** — cost and storage are linear in
tile count — so if you don't need 30 m across the whole coarse domain, this is
the biggest lever there is:

```yaml
tiling:
  mode: domain
  stride_m: 6000
  bbox:                 # lon/lat degrees
    min_lon: -105.65
    min_lat:   34.40
    max_lon: -105.32
    max_lat:   34.67
```

```
wsw_domain.yaml, full        238 tiles   82.7 H100-hours   18.0 TB
wsw_domain.yaml + 30 km bbox  36 tiles   12.5 H100-hours    2.7 TB
```

Price candidate areas without editing YAML:

```bash
python scripts/sizing.py configs/wsw_domain.yaml --bbox -105.65 34.40 -105.32 34.67
```

Two properties worth knowing:

- **Tiles are filtered from the full-domain lattice, not re-originated on the
  box.** A subset run keeps the same tile centres and the same tile ids as the
  full run, so the two stores stay directly comparable and a subset can be
  dropped into an existing store without realigning anything.
- **Selection is by tile *core*, not by the 9 km footprint.** At `stride_m:
  6000` the core is the 6 km square that survives the 1.5 km edge trim, so the
  selected set is the minimal one that still fully covers the box. Selecting on
  the footprint would pull in a ring of tiles whose contribution is entirely
  discarded downstream.

The covered area comes out slightly *larger* than requested — tile cores are
rectangles in UTM and the box is in lon/lat, and the selection rounds outward.
Erring toward one extra tile is safe; erring inward puts a hole in the middle of
a deliverable, and nothing downstream reports it. A bbox that reaches outside the
domain fails at setup with the reachable lon/lat range in the error, rather than
silently returning fewer tiles.

## Sector transfer: coarse WRG in, fine WRG out

```bash
fuxicfd-wrf wrg-inspect  /path/to/coarse_100m.wrg --crs EPSG:32613
fuxicfd-wrf wrg-downscale configs/broadview_downscale.yaml
```

**FuXi-CFD is a steady-state RANS emulator.** Every training label is a converged
solution for one (terrain, inflow) pair, and nothing in the network carries a time
dimension — its 9×9 coarse input was a 1 km coarsening of those same mean fields.
Driving it with instantaneous WRF snapshots hands a mean-field operator a
turbulent realisation. A sector-conditional mean drawn from a WRG is much closer
to what it was trained on, and there are 12 of them instead of 52,560 timesteps.

So the axis being swept is not intrinsically time; it is a list of independent
**inflow states**. `sources/base.py` calls that seam `CoarseSource`. `WrfSource`
supplies one state per timestep, `WrgSectorSource` one per sector.

| | timestep path | sector path |
|---|---|---|
| inferences (57 tiles) | 3.0 M (hourly) | **684** |
| GPU time | ~8 H100-hours | **~16 s** |
| intermediate storage | 6.2 TB | **none** |

None of the sharding, resume, chunk-alignment, float16 or Zarr machinery applies:
`wrg_downscale.run` is one pass with no store in the middle.

### What the transfer is

Per sector, the model output is reduced to two fields on the tile's 300×300 grid:

```
S      = |U_fine| / |U_coarse_upsampled|          speedup
dtheta = theta_fine - theta_coarse_upsampled      deflection
```

The denominator is the 9×9 stencil zoomed to 300×300 — **not** the parent cell's
value. That is literally what the model ingested (preprocessing normalises then
zooms; both are affine, so zooming the raw winds recovers the de-normalised field
the network saw). Against the parent cell, S would absorb the coarse-scale
gradient across the tile, so it would carry structure the mesoscale model already
resolved and applying it would double-count.
`tests/test_transfer.py::test_upsample_matches_preprocessing` pins this.

Applying it is deliberately *not* `A_fine = A_coarse × S`:

```
k_fine = k_coarse                        carried over
A_fine = U_fine / Gamma(1 + 1/k_fine)    exact, no reliance on S
```

`zoom(A·Γ(1+1/k))` is not `Γ(1+1/k)·zoom(A)` unless k is constant across the tile,
so recovering A from the model's own speed avoids an error wherever k varies. S is
still computed and saved, because it is what the timestep-vs-sector comparison
tests — but it is a diagnostic here, not the mechanism.

### Two assumptions, both measurable

**k is carried over unchanged.** Terrain reshapes the speed distribution as well
as scaling it, and a neutral RANS emulator driven at one speed per sector cannot
report that. If `speed_bins: 3` shows S varying with inflow magnitude, that
dependence *is* the k adjustment, and it belongs in `transfer.apply_transfer`.
The run writes a `.diagnostics.json` with the spread of S across bins; below
~0.02 relative, one inference per sector suffices.

**Direction is uniform across each tile.** A WRG records a frequency per sector,
never a mean direction *within* one, so every coarse cell in a state gets the
sector centre while the speed field varies cell to cell. Over 9 km the directional
spread of a mesoscale field is small, but this is exactly what the
timestep-vs-sector speedup comparison is there to test.

Deflection is applied as a nearest-bin reassignment of sector frequency, with
sectors that collect flow from more than one source recombined by moment
matching. At 30° bins that is a coarse instrument — under half a bin moves
nothing, over half a bin moves everything — and it is the honest limit of what a
sector-binned representation can express. `deflect: false` isolates the speedup.

### Power density convention

This path is analytic throughout: the mixture's third moment is
`sum_s f_s A_s³ Γ(1+3/k_s)`, and the all-sector Weibull is fitted to match the
first and third moments, so `0.5ρA³Γ(1+3/k)` reproduces it by construction. That
matches `wrg_read.weibull_power_density` and **differs from `wrg.power_density`**,
which accumulates the third moment from a streamed time series. The two agree on
well-fitted data and diverge where the Weibull fit is poor — so a timestep-derived
WRG and a sector-derived WRG must be compared on one convention. Use this one.

### Validating the shortcut without masts

The timestep path is the ground truth for the sector path, and it needs no
observations. From an existing Zarr store, compute per-timestep
`S = |U_fine| / |U_coarse|`, bin by sector and average; then compare against
`<out>.transfer.npz` from a `wrg-downscale` run with `deflect: false`. Agreement
settles the shortcut internally. Divergence measures the input-distribution
mismatch directly — sector-mean 9×9 fields have far less spatial variance than
instantaneous ones, and the input scaler was fitted on the latter.

This is why the timestep path should not be removed until that comparison has
run.

## WRG export

```bash
fuxicfd-wrf wrg <store>.zarr --out site.wrg \
    --hub-height 100 \
    --terrain-cache /scratch/$USER/fuxicfd/cache
```

Reads either store layout. From a tile store it samples and blends tiles straight
onto the WRG grid, so no full mosaic has to be materialised — a WRG needs one
level and two variables, not a multi-TB intermediate.

Output matches `~/wakemap_processing/wakemap_processor/create_wrg.py`:

```
<nx> <ny> <xmin> <ymin> <cellsize>
GridPoint X Y elev hub A k power_density n_sectors [freq*1000 A*10 k*100]*n
```

Sector convention matches too: first sector centred on north, 12 sectors at 30°.
Elevation comes from the cached SRTM DEMs.

**Two deliberate deviations from that reference:**

1. **Power density is the standard W/m²** — `0.5·ρ·mean(U³)`, from the exact
   accumulated third moment. The reference computes
   `0.5·ρ·π(H/2)²·mean(U)³·0.33/1000`, a turbine power estimate in kW. That is not
   a power density, and it is easy to see why in the existing coarse deliverables:
   its value scales as hub height² through the rotor area (489 → 880 → 1419 → 2983
   at 80/100/120/160 m), which no density should do. Recomputed properly the same
   files give 406 → 477 → 545 → 671 W/m². **Do not compare that column across
   producers** — recompute it from A and k, which is what `wrg_read` does.
2. **No regridding interpolation.** FuXi-CFD output is already on a regular
   projected grid, so it is sampled directly rather than `griddata`-interpolated,
   and the reference's 20% edge buffer defaults to 0 here (`--buffer` to restore).

Two things to set deliberately:

- **Sector count.** The default is 12 sectors (30°), matching `create_wrg.py`. The
  existing Broadview coarse deliverables use **16** (22.5°) — pass
  `--bin-size 22.5` to compare against them directly.
- **Air density defaults to 1.225** kg/m³. Broadview is at ~1340 m where ρ ≈ 1.06,
  so the default overstates power density by ~13%. Pass `--air-density 1.06`.

## Vertical interpolation

FuXi-CFD's levels are a uniform 5 m ladder to 100 m, then widen: 106.5, 114.95,
125.94, 140.22, 158.78, 182.91, 214.29 m. So **80 m and 100 m are exact levels**,
while anything above 100 m is interpolated. `output.levels` accepts arbitrary
heights:

```yaml
output:
  levels: [80.0, 100.0, 120.0, 160.0]
```

```
80 m  (exact)
100 m (exact)
120 m (interp 114.95/125.94 m, w=0.463)
160 m (interp 158.78/182.91 m, w=0.086)
```

The run log prints exactly that. Interpolation is linear in log(z) — physically
motivated for a wind profile, though barely consequential in the dense lower
stack. It matters more above 140 m, where the levels are 15–17% apart.
Extrapolation beyond 5 m or 214.29 m raises rather than clamping.

The distinction that matters: an interpolated 120 m field is **labelled 120 m**,
which is honest. *Snapping* to 125.94 m and calling it 120 m would not be, and
the code refuses to do that anywhere.

`fuxicfd-wrf wrg --hub-height 80` interpolates within whatever levels the store
holds, by the same machinery.

## Comparing against the coarse 1 km WRGs

`notebooks/compare_wrg.ipynb` compares the FuXi-CFD WRG against the existing
1 km deliverables in
`/scratch/moptis/c2wind/broadview/control/deliverables/wrg/broadview`.

It separates the two questions that actually matter:

1. **Coarse-scale agreement** — averaged back up to 1 km cells, does the high-res
   field reproduce WRF? Large bias means a broken chain, not a discovery.
2. **Sub-kilometre structure** — the spread *within* each 1 km cell. That is the
   entire product. If it is negligible, the downscaling added nothing at this site,
   which is a live possibility given Broadview's 49–228 m of relief per tile.

Plus shear across the four heights and a directional-frequency rose.

## Where output goes

Nothing is written inside the repo except reports.

| what | where |
|---|---|
| Model weights + scalers | `/scratch/$USER/fuxicfd/assets/` (`$FUXICFD_ASSETS`) |
| SRTM / WorldCover / per-tile terrain cache | `/scratch/$USER/fuxicfd/cache/` (`$FUXICFD_CACHE`) |
| Zarr stores | `output.path` in the config — `/scratch/$USER/fuxicfd/out/*.zarr` |
| WRG files | wherever `--out` points; convention `/scratch/$USER/fuxicfd/out/wrg/` |
| Phase 0 reports | `reports/` in the repo |
| Slurm logs | `logs/` in the repo |

Weibull fitting uses a vectorised MLE (fixed-point on the shape parameter) rather
than looping `scipy.stats.weibull_min.fit` — validated against scipy to 0.2%
in `tests/test_wrg.py`, and fast enough to fit hundreds of thousands of points ×
13 sectors in seconds instead of hours. Fits that hit the `k` clip bounds are
reported as a warning; that normally means the record is too short to fit.

## Cluster notes (NREL Kestrel)

`onnxruntime-gpu` is pinned to **1.22.0**, a CUDA 12 build. The H100 nodes run
driver 550.54.15, which supports CUDA 12.4 at most, so current CUDA 13 wheels
cannot load. Its CUDA/cuDNN come from `nvidia-*-cu12` pip packages.

An `activate.d` hook prepends `$CONDA_PREFIX/lib` and the `nvidia/*/lib` directories
to `LD_LIBRARY_PATH` — needed because the system `/lib64/libstdc++.so.6` predates
`GLIBCXX_3.4.29`, which numpy and onnxruntime require.

Session thread count defaults to the Slurm allocation rather than the node's 128
cores; otherwise onnxruntime floods stderr with `pthread_setaffinity_np` failures.

## Licence

Code: see `LICENSE`. **The FuXi-CFD model weights and dataset are CC BY-NC 4.0 —
non-commercial use only.** They are downloaded at runtime, never vendored.
# fuxi-cfd-wrf
