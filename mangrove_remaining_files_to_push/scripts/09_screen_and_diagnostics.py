"""Stage 09. Bay of Bengal storm screen (IBTrACS v04r01, 2000 to 2024), wind-radius provenance, within-storm spatial
designs, mask and purity sensitivity, bootstrap stability, Moran's I, missingness models, HAC series lengths,
valid observations by event and class.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from p4.config import W, O, T, FIG, FDT, SUP, data, STORM_FLAGS
import os, sys, math, time, importlib.util, warnings
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import NearestNeighbors
warnings.filterwarnings("ignore")
t0 = time.time(); PART = os.environ.get("PART", "all")
R2 = SUP
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "svg.fonttype": "none", "pdf.fonttype": 42})
MONTHS = np.arange(-36, 25); IDX = lambda m: m + 36; SEED = 20261004
import p4.ibtracs as kit
from p4.config import SCREEN_SEASONS

# ======================================================================= A. full Bay of Bengal screen + radius provenance
if PART in ("all", "screen"):
    ib = kit.load_ni_tracks(); clim = kit.radius_climatology(ib)
    SM = kit.storm_summary()                     # storm-level fields derived from the official IBTrACS file
    SM = SM[SM.season.between(*SCREEN_SEASONS)]   # screening frame 2000 to 2024 (storms after 2024 lack follow-up)
    EW = pd.read_csv(data("inputs/modis_event_windows.csv"))
    ADDED = {"2019116N02090", "2019302N11118", "2023129N08091"}   # Fani, Bulbul, Mocha (added after the screen)
    E8 = EW[~EW.sid.isin(ADDED)].rename(columns={"event_name": "name"})[["sid", "name", "season"]].assign(name=lambda d: d.name.str.upper())
    G = pd.read_csv(data("inputs/bay_of_bengal_mangrove_cell_exposure_grid.csv.gz"))
    latc = [c for c in G.columns if c.lower() in ("cell_lat", "lat")][0]; lonc = [c for c in G.columns if c.lower() in ("cell_lon", "lon")][0]
    areac = "a2020"
    G = G[G[areac] > 0]; clat, clon, carea = G[latc].to_numpy(float), G[lonc].to_numpy(float), G[areac].to_numpy(float)
    bob = SM[SM.subbasins.astype(str).str.contains("BB")].copy()
    rows = []
    for r in bob.itertuples():
        g = ib[ib.SID == r.sid]
        reported = int(g[[f"USA_R34_{q}" for q in ("NE", "SE", "SW", "NW")]].notna().any(axis=1).sum()) if len(g) else 0
        a34 = a50 = a64 = 0.0
        if len(g) and (r.max_usa_wind_kt or 0) >= 34:
            cl = kit.swath_class(kit.prepare_track(g, clim), clat, clon)
            a34, a50, a64 = carea[cl >= 1].sum(), carea[cl >= 2].sum(), carea[cl >= 3].sum()
        landfall = (r.min_landfall_km == 0) if pd.notna(r.min_landfall_km) else False
        reason = ("analysed" if r.sid in (set(E8.sid) | {"2019116N02090", "2019302N11118", "2023129N08091"}) else
                  "excluded: maximum wind below 34 kt" if not ((r.max_usa_wind_kt or 0) >= 34) else
                  "excluded: no landfall" if not landfall else
                  "excluded: no reported R34 radii" if reported == 0 else
                  "excluded: R34 mangrove overlap below 50 km2" if a34 < 50 else
                  "eligible, not analysed (no MODIS extraction)")
        rows.append({"season": r.season, "name": str(r.name).title(), "sid": r.sid, "max_usa_wind_kt": r.max_usa_wind_kt, "landfall": landfall,
                     "fixes_with_reported_r34": reported, "r34_mangrove_km2": round(a34, 1), "r50_mangrove_km2": round(a50, 1),
                     "r64_mangrove_km2": round(a64, 1), "decision": reason})
    SC = pd.DataFrame(rows).sort_values(["season", "name"]); SC.to_csv(R2 + "bay_of_bengal_cyclone_screen_2000_2024.csv", index=False)
    # radius provenance for the 8 analysed events (share of >=34 kt fixes with reported, interpolated or climatological R34)
    prov = []
    E11 = pd.concat([E8[["sid", "name", "season"]], pd.DataFrame({"sid": ["2019116N02090", "2019302N11118", "2023129N08091"], "name": ["FANI", "BULBUL", "MOCHA"], "season": [2019, 2019, 2023]})])
    for r in E11.itertuples():
        g = ib[ib.SID == r.sid].drop_duplicates("t").sort_values("t")
        if r.sid == "2019302N11118":
            g = g[g.t >= pd.Timestamp("2019-11-02 12:00")]
        w = g.USA_WIND.fillna(0).to_numpy()
        raw = g[[f"USA_R34_{q}" for q in ("NE", "SE", "SW", "NW")]].to_numpy(float); rep = ~np.all(np.isnan(raw), 1)
        ts = (g.t - pd.Timestamp("1970-01-01")).dt.total_seconds().to_numpy(); need = (~rep) & (w >= 34)
        interp = np.array([rep.any() and np.abs(ts[rep] - ts[j]).min() <= 6 * 3600 for j in range(len(g))]) & need
        n34 = int((w >= 34).sum())
        prov.append({"event": str(r.name).title() + " " + str(r.season), "fixes_ge34kt": n34, "reported": int((rep & (w >= 34)).sum()),
                     "interpolated_within_6h": int(interp.sum()), "climatology": int((need & ~interp).sum())})
    pd.DataFrame(prov).to_csv(R2 + "wind_radius_provenance.csv", index=False)
    # eligible storms with R64 winds over mangroves that were not analysed (Supplementary Table S2)
    raw = pd.read_csv(kit.IBTRACS_NI, skiprows=[1], usecols=["SID", "ISO_TIME", "LAT", "LON", "DIST2LAND", "TRACK_TYPE"], low_memory=False, na_values=[" ", ""])
    raw = raw[raw.TRACK_TYPE.astype(str).str.lower().eq("main")]
    for c in ("LAT", "LON", "DIST2LAND"): raw[c] = pd.to_numeric(raw[c], errors="coerce")
    REASON = "Not extracted: outside the a priori seed list and not among the three storms prioritised during review"
    NOTES = {"Giri 2010": "largest unanalysed R64 exposure; priority extension",
             "Unnamed 2004": "36-month baseline starts before Aqua MODIS (mid-2002), so part of the baseline would be Terra-only"}
    U = SC[SC.decision.str.startswith("eligible") & (SC.r64_mangrove_km2 > 0)].sort_values("r64_mangrove_km2", ascending=False)
    urows = []
    for r in U.itertuples():
        g = ib[ib.SID == r.sid]; cl = kit.swath_class(kit.prepare_track(g, clim), clat, clon)
        lf = raw[(raw.SID == r.sid) & (raw.DIST2LAND == 0)].sort_values("ISO_TIME").iloc[0]
        urows.append({"season": r.season, "name": r.name, "landfall_utc": str(lf.ISO_TIME)[:16], "landfall_lat": round(float(lf.LAT), 1), "landfall_lon": round(float(lf.LON), 1),
                      "max_usa_wind_kt": r.max_usa_wind_kt, "r64_mangrove_km2": r.r64_mangrove_km2, "r64_mangrove_cells": int((cl >= 3).sum()),
                      "reason": REASON, "note": NOTES.get(f"{r.name} {r.season}", np.nan)})
    pd.DataFrame(urows).to_csv(R2 + "unanalysed_eligible_r64_storms.csv", index=False)
    n_all = len(SC); n34 = int((SC.max_usa_wind_kt >= 34).sum()); nlf = int(((SC.max_usa_wind_kt >= 34) & SC.landfall).sum())
    nrad = int(((SC.max_usa_wind_kt >= 34) & SC.landfall & (SC.fixes_with_reported_r34 > 0)).sum())
    nel = int(SC.decision.str.startswith(("analysed", "eligible")).sum()); nan_ = int(SC.decision.str.startswith("analysed").sum()); nnot = nel - nan_
    print("screen", dict(n_all=n_all, n34=n34, nlf=nlf, nrad=nrad, nel=nel, analysed=nan_, eligible_not=nnot), f"{time.time()-t0:.0f}s", flush=True)
    print(SC[SC.decision.str.startswith(("analysed", "eligible"))][["season", "name", "max_usa_wind_kt", "r34_mangrove_km2", "r64_mangrove_km2", "decision"]].to_string(index=False))

# ======================================================================= B-I. spatial designs, mask/purity, statistical diagnostics
if PART in ("all", "stats"):
    A = pd.read_pickle(W + "anom.pkl"); R = pd.read_pickle(W + "stageB.pkl"); meta = R["meta"]; match = R["match"]; BC = R["bc_primary"]; DUMMY = R["DUMMY"]
    Mn = A[("ndvi", "vrelax", "main")]["M"].astype(float); Mn = np.vstack([Mn, np.full((1, Mn.shape[1]), np.nan)])
    mi = meta.set_index("pid"); EID = {e.split("_")[1].title(): e for e in meta.event_id.unique()}
    CAN = ["f_ndvi_level", "f_ndvi_amp", "f_ndvi_sd", "f_ndvi_pretrend", "f_evi_level", "f_frac340"]
    def blocks(pids, deg=0.25):
        return (np.floor(mi.loc[pids, "cell_lat"].to_numpy() / deg)).astype(int).astype(str) + "_" + (np.floor(mi.loc[pids, "cell_lon"].to_numpy() / deg)).astype(int).astype(str)
    def bootw(bl, Bn, rng):
        ub, inv = np.unique(bl, return_inverse=True); return rng.multinomial(len(ub), np.full(len(ub), 1 / len(ub)), size=Bn)[:, inv].astype(float)
    def m06(D, Bn=500, seed=SEED, bl=None):
        sl = slice(IDX(0), IDX(6) + 1); Dm = D[:, sl]; n = np.isfinite(Dm).sum(0); ok = n >= max(30, 0.2 * len(D))
        est = np.nanmean(np.nanmean(Dm, 0)[ok])
        Wt = bootw(bl, Bn, np.random.default_rng(seed)); f = np.isfinite(Dm)
        bb = (Wt @ np.where(f, Dm, 0)) / np.maximum(Wt @ f, 1e-12); v = np.nanmean(bb[:, ok], 1)
        return est, np.nanpercentile(v, 2.5), np.nanpercentile(v, 97.5), int(len(D))
    def primary_D(ev):
        eid = EID[ev]; mt = match[eid]; ep = mt["exp_pid"]; cls = mi.loc[ep, "exposure_class"].to_numpy()
        D = Mn[ep] - np.nanmean(Mn[mt["ctrl_pid"]], 1) - BC[(eid, "ndvi")]
        return ep, cls, D, mt
    rows = []
    for ev in ("Sidr", "Nargis", "Aila", "Fani", "Bulbul", "Amphan", "Mocha"):
        eid = EID[ev]; ep, cls, D, mt = primary_D(ev); s = cls == "R64"; pe = ep[s]
        est = m06(D[s], bl=blocks(pe)); rows.append({"event": ev, "analysis": "primary (revised matching, bias-corrected)", "est": est[0], "lo": est[1], "hi": est[2], "n": est[3]})
        # C. stable mangrove (GMW core / 2020 >= 0.9) and D. footprint purity
        core = (mi.loc[pe, "gmw_core_km2_cell"] / mi.loc[pe, "gmw2020_km2_cell"].replace(0, np.nan)).to_numpy()
        for lab, mask in (("stable mangrove 1996 to 2020 (core/2020 at least 0.9)", core >= 0.9),
                          ("footprint purity at least 0.75", mi.loc[pe, "gmw2020_frac_block220m"].to_numpy() >= 0.75),
                          ("footprint purity at least 0.90", mi.loc[pe, "gmw2020_frac_block220m"].to_numpy() >= 0.90)):
            if mask.sum() >= 30:
                e_ = m06(D[s][mask], bl=blocks(pe[mask])); rows.append({"event": ev, "analysis": lab, "est": e_[0], "lo": e_[1], "hi": e_[2], "n": e_[3]})
        # B1. within-storm wind gradient with 0.5 deg spatial fixed effects (R64 vs R34, R50 dummy; canopy covariates)
        g = meta[(meta.event_id == eid) & (meta.exposure_class != "control") & meta.baseline_ok].copy()
        Y = Mn[g.pid.values][:, IDX(0):IDX(6) + 1]; nv = np.isfinite(Y).sum(1); g["y"] = np.nanmean(Y, 1); g = g[nv >= 3]
        g["fe"] = blocks(g.pid.values, 0.5); g["cl"] = blocks(g.pid.values, 0.25)
        idb = g.groupby("fe").exposure_class.agg(lambda x: ("R64" in set(x)) and ("R34" in set(x) or "R50" in set(x)))
        gg = g[g.fe.isin(idb[idb].index)]
        if gg.exposure_class.eq("R64").sum() >= 30 and gg.exposure_class.isin(["R34", "R50"]).sum() >= 30:
            X = pd.get_dummies(gg.fe, drop_first=True, dtype=float); X["R64"] = (gg.exposure_class == "R64").astype(float); X["R50"] = (gg.exposure_class == "R50").astype(float)
            for c in CAN: X[c] = (gg[c] - gg[c].mean()) / (gg[c].std() or 1)
            X = sm.add_constant(X); fit = sm.OLS(gg.y.values, X).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(gg.cl)[0]})
            b, se = fit.params["R64"], fit.bse["R64"]
            rows.append({"event": ev, "analysis": "within-storm gradient, R64 vs R34 with 0.5 degree spatial fixed effects", "est": b, "lo": b - 1.96 * se, "hi": b + 1.96 * se,
                         "n": int(len(gg)), "n_blocks": int(gg.fe.nunique())})
        # B2. distance-restricted matched gradient: R64 cells vs R34/R50 cells of the same storm within 50 km (canopy Mahalanobis, k <= 5)
        r64 = g[g.exposure_class == "R64"]; low = g[g.exposure_class.isin(["R34", "R50"])]
        if len(r64) >= 30 and len(low) >= 30:
            Z = np.vstack([r64[CAN].values, low[CAN].values]); mu, sd = Z.mean(0), Z.std(0).clip(1e-9)
            nn = NearestNeighbors(n_neighbors=min(40, len(low))).fit((low[CAN].values - mu) / sd); d, ix = nn.kneighbors((r64[CAN].values - mu) / sd)
            la1, lo1 = np.radians(r64.cell_lat.values)[:, None], np.radians(r64.cell_lon.values)[:, None]
            la2, lo2 = np.radians(low.cell_lat.values[ix]), np.radians(low.cell_lon.values[ix])
            km = 6371 * 2 * np.arcsin(np.sqrt(np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2))
            Dg, keep = [], []
            for i in range(len(r64)):
                okk = ix[i][km[i] <= 50][:5]
                if len(okk):
                    Dg.append(Mn[r64.pid.values[i]] - np.nanmean(Mn[low.pid.values[okk]], 0)); keep.append(i)
            if len(keep) >= 30:
                Dg = np.array(Dg); e_ = m06(Dg, bl=blocks(r64.pid.values[keep]))
                rows.append({"event": ev, "analysis": "within-storm gradient, R64 vs R34/R50 matched within 50 km", "est": e_[0], "lo": e_[1], "hi": e_[2], "n": e_[3]})
    SP = pd.DataFrame(rows); SP.to_csv(R2 + "spatial_design_mask_purity_sensitivity.csv", index=False)
    print(SP.round(4).to_string(index=False), flush=True)
    # E. bootstrap stability (500 vs 2000 replicates, 3 seeds)
    bst = []
    for ev in ("Sidr", "Nargis"):
        ep, cls, D, mt = primary_D(ev); s = cls == "R64"; bl = blocks(ep[s])
        for Bn in (500, 2000):
            for sd_ in (1, 2, 3):
                e_ = m06(D[s], Bn=Bn, seed=SEED + sd_, bl=bl); bst.append({"event": ev, "replicates": Bn, "seed": sd_, "lo": e_[1], "hi": e_[2]})
    BST = pd.DataFrame(bst); BST.to_csv(R2 + "bootstrap_replicate_stability.csv", index=False)
    print(BST.groupby(["event", "replicates"])[["lo", "hi"]].agg(["min", "max"]).round(4).to_string(), flush=True)
    # F. spatial autocorrelation of cell-level months 0-6 differences (Moran's I by distance band)
    mor = []
    for ev in ("Sidr", "Nargis"):
        ep, cls, D, mt = primary_D(ev); s = cls == "R64"; y = np.nanmean(D[s][:, IDX(0):IDX(6) + 1], 1); ok = np.isfinite(y)
        la, lo = np.radians(mi.loc[ep[s], "cell_lat"].to_numpy()[ok]), np.radians(mi.loc[ep[s], "cell_lon"].to_numpy()[ok]); y = y[ok] - np.nanmean(y[ok])
        dd = 6371 * 2 * np.arcsin(np.sqrt(np.sin((la[:, None] - la) / 2) ** 2 + np.cos(la[:, None]) * np.cos(la) * np.sin((lo[:, None] - lo) / 2) ** 2))
        for a_, b_ in ((0, 5), (5, 10), (10, 20), (20, 30), (30, 50), (50, 100)):
            Wm = ((dd > a_) & (dd <= b_)).astype(float); np.fill_diagonal(Wm, 0)
            if Wm.sum() > 0:
                I = (len(y) / Wm.sum()) * (y @ Wm @ y) / (y @ y); mor.append({"event": ev, "band_km": f"{a_}-{b_}", "morans_I": I, "pairs": int(Wm.sum() / 2)})
    MOR = pd.DataFrame(mor); MOR.to_csv(R2 + "spatial_autocorrelation_morans_I.csv", index=False); print(MOR.round(3).to_string(index=False), flush=True)
    # G. missingness model diagnostics (AUC in-sample and 5-fold CV, weight range) and H. HAC series lengths
    BCOV = CAN + ["cell_lat", "cell_lon", "prior_max_class_window"]; mm = []
    for ev in ("Sidr", "Nargis"):
        ep, cls, D, mt = primary_D(ev); s = cls == "R64"; X = mi.loc[ep[s], BCOV].astype(float).to_numpy(); X = (X - X.mean(0)) / X.std(0).clip(1e-9)
        for m in range(0, 7):
            y = np.isfinite(D[s][:, IDX(m)]).astype(int)
            if 10 <= y.sum() <= len(y) - 10:
                lr = LogisticRegression(max_iter=3000); p_in = lr.fit(X, y).predict_proba(X)[:, 1]
                p_cv = cross_val_predict(LogisticRegression(max_iter=3000), X, y, cv=5, method="predict_proba")[:, 1]
                w = 1 / p_in.clip(0.05, 1); mm.append({"event": ev, "month": m, "share_observed": y.mean(), "auc_in_sample": roc_auc_score(y, p_in),
                                                       "auc_cv5": roc_auc_score(y, p_cv), "max_weight": w[y == 1].max(), "ess_share": (w[y == 1].sum() ** 2 / (w[y == 1] ** 2).sum()) / y.sum()})
            else:
                mm.append({"event": ev, "month": m, "share_observed": y.mean()})
    MM = pd.DataFrame(mm); MM.to_csv(R2 + "missingness_model_diagnostics.csv", index=False); print(MM.round(3).to_string(index=False), flush=True)
    ES = R["ES"]; hac = []
    for ev in ("Sidr", "Nargis"):
        d = ES[(ES.event_name == ev) & (ES.group == "R64") & (ES.spec == "primary_matched")]
        f = d.beta.notna(); hac.append({"event": ev, "pre_months": int((f & (d.rel_month < 0)).sum()), "post_months": int((f & (d.rel_month >= 0)).sum()),
                                         "post_months_0_6": int((f & d.rel_month.between(0, 6)).sum())})
    pd.DataFrame(hac).to_csv(R2 + "hac_series_lengths.csv", index=False); print(hac, flush=True)
    # I. valid observations per event and exposure class (analysed cells)
    MP = pd.read_csv(T + "final_event_cell_month_panel.csv.gz", usecols=["event_id", "cell_uid", "exposure_class", "n_valid_obs"])
    CP = pd.read_csv(T + "final_event_cell_panel.csv", usecols=["event_id", "event_name", "cell_uid", "in_analysis"])
    MP = MP.merge(CP, on=["event_id", "cell_uid"]); MP = MP[MP.in_analysis]
    VO = MP.groupby(["event_name", "exposure_class"]).agg(cells=("cell_uid", "nunique"), valid_observations=("n_valid_obs", "sum")).reset_index()
    VO.to_csv(R2 + "valid_observations_by_event_and_class.csv", index=False); print(VO.to_string(index=False))
print("DONE", f"{time.time()-t0:.0f}s")
