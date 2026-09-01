"""Which roughness setting should ship: compare both FuXi WRGs against WRF's own.

All three are read on equal footing (sector count per row, power density recomputed
from A and k), restricted to the area the absolute-z0 run covers, and the two 30 m
fields are averaged onto WRF's 1 km cells so the comparison is like-for-like.
"""
import numpy as np, pandas as pd
from fuxicfd_wrf.wrg_read import read_wrg, weibull_mean_speed

RHO = 1.0823
COARSE = "/scratch/moptis/c2wind/pattern/ph1/control/deliverables/wrg/pattern/ph1/control_{H}m.wrg"
RCONST = "/scratch/moptis/c2wind/pattern/ph1/control/deliverables/wrg_fuxi/pattern/ph1/control_{H}m.wrg"
RABS   = "/scratch/moptis/fuxicfd/out/wrg/pattern_ph1_rabs_concat/control_{H}m.wrg"

rows = []
for H in (80, 100, 120, 160):
    wrf = read_wrg(COARSE.format(H=H), air_density=RHO)
    a   = read_wrg(RABS.format(H=H),   air_density=RHO)
    c   = read_wrg(RCONST.format(H=H), air_density=RHO)
    for d in (wrf, a, c):
        d["U"] = weibull_mean_speed(d.A.values, d.k.values)
    # Overlap = the absolute-z0 footprint, with a half-cell margin for binning.
    x0, x1 = a.x.min(), a.x.max(); y0, y1 = a.y.min(), a.y.max()
    def clip(d): return d[(d.x >= x0) & (d.x <= x1) & (d.y >= y0) & (d.y <= y1)]
    wrf_c = clip(wrf)
    # Average the 30 m fields onto WRF's 1 km cells: nearest-cell binning on the
    # coarse origin, so each fine point is assigned to the cell containing it.
    cs = 1000.0
    cx0, cy0 = wrf.x.min(), wrf.y.min()
    def upscale(d):
        d = clip(d).copy()
        d["cx"] = cx0 + np.round((d.x - cx0) / cs) * cs
        d["cy"] = cy0 + np.round((d.y - cy0) / cs) * cs
        return d.groupby(["cx", "cy"]).agg(U=("U", "mean"), A=("A", "mean"),
                                           k=("k", "mean"), n=("U", "size")).reset_index()
    ua, uc = upscale(a), upscale(c)
    w = wrf_c.rename(columns={"x": "cx", "y": "cy"})[["cx", "cy", "U", "A", "k"]]
    m = w.merge(ua, on=["cx", "cy"], suffixes=("_wrf", "_abs")).merge(
        uc.rename(columns={"U": "U_con", "A": "A_con", "k": "k_con"}), on=["cx", "cy"])
    rows.append(dict(H=H, n_cells=len(m),
                     U_wrf=m.U_wrf.mean(), U_con=m.U_con.mean(), U_abs=m.U_abs.mean(),
                     bias_con=(m.U_con - m.U_wrf).mean(), bias_abs=(m.U_abs - m.U_wrf).mean(),
                     rmse_con=np.sqrt(((m.U_con - m.U_wrf) ** 2).mean()),
                     rmse_abs=np.sqrt(((m.U_abs - m.U_wrf) ** 2).mean()),
                     sub_km_sd_con=None, sub_km_sd_abs=None))
    # Sub-km structure: spread of the 30 m field inside each 1 km cell.
    for tag, d in (("con", c), ("abs", a)):
        dd = clip(d).copy()
        dd["cx"] = cx0 + np.round((dd.x - cx0) / cs) * cs
        dd["cy"] = cy0 + np.round((dd.y - cy0) / cs) * cs
        rows[-1][f"sub_km_sd_{tag}"] = dd.groupby(["cx", "cy"]).U.std().mean()

df = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print("\nMean wind speed over the shared area, 30 m fields averaged onto WRF's 1 km cells\n")
print(df[["H", "n_cells", "U_wrf", "U_con", "U_abs"]].round(3).to_string(index=False))
print("\nBias vs WRF (m/s) and RMSE across cells\n")
print(df[["H", "bias_con", "bias_abs", "rmse_con", "rmse_abs"]].round(3).to_string(index=False))
print("\nSub-kilometre structure: mean within-1km-cell sd of the 30 m field (m/s)\n")
print(df[["H", "sub_km_sd_con", "sub_km_sd_abs"]].round(3).to_string(index=False))
