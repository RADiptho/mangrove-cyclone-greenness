"""IBTrACS loading and wind-swath geometry (quadrant radii, 6 h interpolation, radius climatology, azimuth interpolation).

Verbatim from the MODIS export kit used to build the dataset, so that exposure and contamination flags are computed
identically. storm_summary() derives the storm-level fields used by the Bay of Bengal screen.
"""
import math
from pathlib import Path
import numpy as np
import pandas as pd
from .config import IBTRACS_NI

R_EARTH, NM2KM = 6371.0088, 1.852
KM_PER_DEG = math.pi * R_EARTH / 180.0
THRESH = (34, 50, 64)

def load_ni_tracks(path=None):
    f = Path(path) if path else IBTRACS_NI
    cols = ["SID", "SEASON", "NAME", "ISO_TIME", "LAT", "LON", "USA_WIND", "TRACK_TYPE"] + \
           [f"USA_R{k}_{q}" for k in THRESH for q in ("NE", "SE", "SW", "NW")]
    ib = pd.read_csv(f, skiprows=[1], usecols=cols, low_memory=False, na_values=[" ", ""])
    for c in cols:
        if c not in ("SID", "NAME", "ISO_TIME", "TRACK_TYPE"):
            ib[c] = pd.to_numeric(ib[c], errors="coerce")
    ib["t"] = pd.to_datetime(ib["ISO_TIME"], errors="coerce")
    ib = ib[ib["TRACK_TYPE"].astype(str).str.lower().eq("main")].drop_duplicates(["SID", "t"])
    return ib.sort_values(["SID", "t"]).reset_index(drop=True)

def radius_climatology(ib):
    q = ("NE", "SE", "SW", "NW")
    r34 = ib[[f"USA_R34_{x}" for x in q]].mean(axis=1, skipna=False) * NM2KM
    bins = [34, 50, 64, 84, 96, 250]
    b = pd.cut(ib["USA_WIND"], bins, right=False)
    clim = {"bins": bins, "r34": {}, "r50": {}, "r64": {}}
    for iv, g in ib.assign(r34=r34, b=b).dropna(subset=["r34"]).groupby("b", observed=True):
        clim["r34"][int(iv.left)] = float(g["r34"][g["r34"] > 0].median())
        for k in (50, 64):
            rk = g[[f"USA_R{k}_{x}" for x in q]].mean(axis=1, skipna=False) * NM2KM
            okm = (g["r34"] > 0) & rk.notna() & (rk > 0)
            clim[f"r{k}"][int(iv.left)] = float((rk[okm] / g.loc[okm, "r34"]).median()) if okm.sum() > 20 else np.nan
    return clim

def _bin_left(w, bins):
    for lo, hi in zip(bins[:-1], bins[1:]):
        if lo <= w < hi:
            return lo
    return None

def prepare_track(g, clim, step_min=30):
    g = g.drop_duplicates("t").sort_values("t")
    g = g[g["LAT"].notna() & g["LON"].notna()]
    if len(g) < 2:
        return None
    ts = (g["t"] - pd.Timestamp("1970-01-01")).dt.total_seconds().to_numpy(float)
    wind = g["USA_WIND"].to_numpy(float)
    lon = np.degrees(np.unwrap(np.radians(g["LON"].to_numpy(float))))
    lat = g["LAT"].to_numpy(float)
    R = {}
    q = ("NE", "SE", "SW", "NW")
    for k in THRESH:
        raw = g[[f"USA_R{k}_{x}" for x in q]].to_numpy(float) * NM2KM
        rep = ~np.all(np.isnan(raw), axis=1)
        arr = np.where(np.isnan(raw), 0.0, raw)
        need = (~rep) & (np.nan_to_num(wind, nan=0) >= k)
        if need.any() and rep.any():
            for j in np.nonzero(need)[0]:
                if np.abs(ts[rep] - ts[j]).min() <= 6 * 3600:
                    for qi in range(4):
                        arr[j, qi] = np.interp(ts[j], ts[rep], arr[rep, qi])
                    need[j] = False
        for j in np.nonzero(need)[0]:
            lo = _bin_left(wind[j], clim["bins"])
            if k == 34:
                v = clim["r34"].get(lo, np.nan); arr[j, :] = v if np.isfinite(v) else 0.0
            else:
                r = clim[f"r{k}"].get(lo, np.nan); arr[j, :] = R[34][j, :] * r if np.isfinite(r) else 0.0
        arr[(~rep) & ~(np.nan_to_num(wind, nan=0) >= k)] = 0.0
        R[k] = arr
    R[50] = np.minimum(R[50], R[34]); R[64] = np.minimum(R[64], R[50])
    tn = np.arange(ts[0], ts[-1] + 1, step_min * 60.0)
    out = {"lat": np.interp(tn, ts, lat), "lon": np.interp(tn, ts, lon)}
    for k in THRESH:
        out[f"R{k}"] = np.column_stack([np.interp(tn, ts, R[k][:, qi]) for qi in range(4)])
    return out

def swath_class(trk, clat, clon):
    """Max wind class per cell (0 none, 1 >=34 kt, 2 >=50 kt, 3 >=64 kt); azimuth-interpolated quadrant radii."""
    n = len(clat); cls = np.zeros(n, np.int8)
    if trk is None:
        return cls
    rmax = trk["R34"].max(axis=1)
    for s in np.nonzero(rmax > 0)[0]:
        lim = rmax[s] + 5.0
        sel = np.nonzero((np.abs(clat - trk["lat"][s]) <= lim / KM_PER_DEG) &
                         (np.abs(clon - trk["lon"][s]) <= lim / (KM_PER_DEG * max(math.cos(math.radians(trk["lat"][s])), 0.2))))[0]
        if not len(sel):
            continue
        dx = (clon[sel] - trk["lon"][s]) * KM_PER_DEG * np.cos(np.radians((clat[sel] + trk["lat"][s]) / 2))
        dy = (clat[sel] - trk["lat"][s]) * KM_PER_DEG
        d = np.hypot(dx, dy)
        az = (np.degrees(np.arctan2(dx, dy)) + 360) % 360
        u = ((az - 45.0) % 360.0) / 90.0; i0 = np.floor(u).astype(int) % 4; f = u - np.floor(u)
        for j, k in enumerate(THRESH):
            Rk = trk[f"R{k}"][s]
            r = (1 - f) * Rk[i0] + f * Rk[(i0 + 1) % 4]
            hit = sel[d <= r]
            cls[hit] = np.maximum(cls[hit], j + 1)
    return cls

def hav_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

# =============================================================================
# prepare: events, windows, cell sample
# =============================================================================

def storm_summary(path=None, first_season=2000):
    """Storm-level summary: season, name, sub-basins, maximum USA wind and minimum IBTrACS LANDFALL distance."""
    f = Path(path) if path else IBTRACS_NI
    ib = pd.read_csv(f, skiprows=[1], usecols=["SID", "SEASON", "NAME", "SUBBASIN", "USA_WIND", "LANDFALL"], low_memory=False, na_values=[" ", ""])
    for c in ("SEASON", "USA_WIND", "LANDFALL"):
        ib[c] = pd.to_numeric(ib[c], errors="coerce")
    ib = ib[ib.SEASON >= first_season]
    g = ib.groupby("SID")
    return pd.DataFrame({"season": g.SEASON.first(), "name": g.NAME.first(),
                         "subbasins": g.SUBBASIN.agg(lambda s: "|".join(sorted(set(s.dropna().astype(str))))),
                         "max_usa_wind_kt": g.USA_WIND.max(), "min_landfall_km": g.LANDFALL.min()}).reset_index().rename(columns={"SID": "sid"})
