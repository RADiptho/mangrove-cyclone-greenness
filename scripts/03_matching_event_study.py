"""Stage 03. Storm contamination and censoring, cell features, matching (exact prior storm, propensity-score caliper,
Mahalanobis k<=5, regression bias correction), event-study estimates, spatial block bootstrap, HAC tests,
placebo in time and space, specifications and pooled estimates.

Strict-QA correction: a baseline is used only when at least 24 valid pre-landfall observations exist under the QA rule
in use. Strict-QA anomalies of cells with fewer than 24 strict-valid pre-landfall observations are therefore set to missing.

Optional parallelism: EVENTS=SIDR,NARGIS MODE=part runs selected events; MODE=merge combines the parts.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from p4.config import W, O, T, FIG, FDT, SUP, data, STORM_FLAGS
import pandas as pd, numpy as np, time, math, importlib.util, json
from scipy import stats
t0 = time.time()
import os
ONLY = os.environ.get("EVENTS", ""); MODE = os.environ.get("MODE", "full")   # MODE: part | merge | full
def _skip(eid):
    return MODE == "merge" or (ONLY and eid.split("_")[1] not in ONLY.split(","))
SEED = 20261004
rng = np.random.default_rng(SEED)
MONTHS = np.arange(-36, 25)
IDX = lambda m: m + 36
B = 500
BLOCK_DEG = 0.25
K_MATCH = 5
NPRE_MIN = 24

meta = pd.read_pickle(W + "meta.pkl")
A = pd.read_pickle(W + "anom.pkl")
# strict-QA correction: enforce the >= 24 valid pre-landfall observation rule for the strict-QA baselines
for _k in list(A):
    if _k[1] == "vstrict" and _k[2] == "main":
        A[_k]["M"] = A[_k]["M"].copy(); A[_k]["M"][A[_k]["npre"] < NPRE_MIN] = np.nan
# ---------------------------------------------------------------- storm contamination over the full window
import p4.ibtracs as kit
if STORM_FLAGS == "ibtracs":
    ib = kit.load_ni_tracks(); clim = kit.radius_climatology(ib)
    span = ib.groupby("SID")["t"].agg(["min", "max"]); mw = ib.groupby("SID")["USA_WIND"].max()
meta["landfall_utc"] = pd.to_datetime(meta["landfall_utc"])
meta["censor_m50"] = 99; meta["censor_m34"] = 99; meta["prior_max_class_window"] = 0; meta["prior_r50_month"] = np.nan
storm_rows = []
for eid, g in (meta.groupby("event_id", sort=False) if STORM_FLAGS == "ibtracs" else []):
    lf = g["landfall_utc"].iloc[0]; sid = g["sid"].iloc[0]
    cand = span[(span["max"] >= lf - pd.Timedelta(days=1100)) & (span["min"] <= lf + pd.Timedelta(days=740))].index
    la, lo = g["cell_lat"].to_numpy(), g["cell_lon"].to_numpy()
    c50 = np.full(len(g), 99); c34 = np.full(len(g), 99); pmax = np.zeros(len(g), int); p50m = np.full(len(g), np.nan)
    for s in cand:
        if s == sid or mw.get(s, 0) < 34:
            continue
        gs = ib[ib["SID"] == s]
        cl = kit.swath_class(kit.prepare_track(gs, clim), la, lo)
        if not cl.any():
            continue
        tpk = gs.loc[gs["USA_WIND"].idxmax(), "t"] if gs["USA_WIND"].notna().any() else gs["t"].iloc[0]
        rm = int(math.floor((tpk.normalize() - lf.normalize()).days / 30.4375))
        if rm < -36 or rm > 24:
            continue
        storm_rows.append({"event_id": eid, "other_sid": s, "other_name": str(gs["NAME"].iloc[0]).title(), "rel_month": rm,
                           "max_wind_kt": float(mw.get(s, np.nan)), "cells_r34plus": int((cl >= 1).sum()),
                           "cells_r50plus": int((cl >= 2).sum()), "cells_total": len(g)})
        if rm >= 0:
            c50 = np.where((cl >= 2) & (rm < c50), rm, c50); c34 = np.where((cl >= 1) & (rm < c34), rm, c34)
        else:
            pmax = np.maximum(pmax, cl); p50m = np.where((cl >= 2) & (np.isnan(p50m) | (rm > p50m)), rm, p50m)
    meta.loc[g.index, "censor_m50"] = c50; meta.loc[g.index, "censor_m34"] = c34
    meta.loc[g.index, "prior_max_class_window"] = pmax; meta.loc[g.index, "prior_r50_month"] = p50m
if STORM_FLAGS == "released":   # use the contamination and censoring flags released with the dataset
    _p = pd.read_csv(data("final_event_cell_panel.csv"), usecols=["event_id", "cell_uid", "censor_m50", "censor_m34", "prior_max_class_window", "prior_r50_month"])
    _m = meta[["event_id", "cell_uid"]].merge(_p, on=["event_id", "cell_uid"], how="left")
    for c in ("censor_m50", "censor_m34"):
        meta[c] = _m[c].fillna(99).astype(int).to_numpy()
    meta["prior_max_class_window"] = _m["prior_max_class_window"].fillna(0).astype(int).to_numpy(); meta["prior_r50_month"] = _m["prior_r50_month"].to_numpy()
    storm_rows = pd.read_csv(data("window_storm_contamination.csv")).to_dict("records")
storms = pd.DataFrame(storm_rows)
storms.to_csv(W + "window_storms.csv", index=False)
print("storm contamination done", f"{time.time()-t0:.0f}s", flush=True)

# ---------------------------------------------------------------- per-cell features
key = ("ndvi", "vrelax", "main")
npre = A[key]["npre"]; meta["npre_valid_relaxed"] = npre
meta["baseline_ok"] = npre >= NPRE_MIN
bn, be = A[("ndvi", "vrelax", "main")]["beta"], A[("evi", "vrelax", "main")]["beta"]
meta["f_ndvi_level"] = bn[:, 0]; meta["f_ndvi_amp"] = np.hypot(bn[:, 1], bn[:, 2])
meta["f_ndvi_sd"] = A[key]["sdpre"]; meta["f_evi_level"] = be[:, 0]
Mn = A[key]["M"]
pre = Mn[:, :36]
x = MONTHS[:36].astype(float)
okp = np.isfinite(pre)
xm = np.where(okp, x, np.nan)
xbar = np.nanmean(xm, 1); ybar = np.nanmean(pre, 1)
num = np.nansum((xm - xbar[:, None]) * (pre - ybar[:, None]), 1); den = np.nansum((xm - xbar[:, None]) ** 2, 1)
meta["f_ndvi_pretrend"] = np.where(den > 0, num / den, 0.0)
meta["f_frac340"] = meta["gmw2020_frac_window340m"]
FEATS = ["f_ndvi_level", "f_ndvi_amp", "f_ndvi_sd", "f_ndvi_pretrend", "f_evi_level", "f_frac340",
         "cell_lat", "cell_lon", "prior_max_class_window"]
meta["block"] = (np.floor(meta["cell_lat"] / BLOCK_DEG)).astype(int).astype(str) + "_" + (np.floor(meta["cell_lon"] / BLOCK_DEG)).astype(int).astype(str)
fu = {}  # follow-up months per event (min over sensors, from data)
for eid, g in meta.groupby("event_id"):
    cnt = A[key]["cnt"][g.pid.values]
    has = np.nonzero(cnt.sum(0) > 0)[0]
    fu[eid] = int(MONTHS[has.max()])
meta["followup_last_month"] = meta["event_id"].map(fu)

# ---------------------------------------------------------------- matching
# Primary (R1): exact matching on prior-storm exposure (none vs >= R34 in the 36-month baseline window),
# propensity-score caliper (0.2 SD of logit PS fitted on canopy + prior-storm covariates),
# k = 5 nearest controls by Mahalanobis distance on canopy covariates (with replacement),
# and Abadie-Imbens style regression bias correction for residual imbalance in all covariates incl. location.
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
CAN = ["f_ndvi_level", "f_ndvi_amp", "f_ndvi_sd", "f_ndvi_pretrend", "f_evi_level", "f_frac340"]
PSCOV = CAN + ["prior_max_class_window"]
BCOV = FEATS
PS_CAL_SD, N_CAND = 0.2, 25
NPID = int(meta.pid.max()) + 1
DUMMY = NPID  # index of an all-NaN row appended to every outcome matrix (padding for < k matches)
for _k in list(A.keys()):
    if "M" in A[_k] and A[_k]["M"].shape[0] == NPID:
        A[_k]["M"] = np.vstack([A[_k]["M"], np.full((1, A[_k]["M"].shape[1]), np.nan, dtype=A[_k]["M"].dtype)])

def mahal_match(E, C, k):
    Z = np.vstack([E, C]); mu = Z.mean(0); sd = Z.std(0); sd[sd == 0] = 1
    Ez, Cz = (E - mu) / sd, (C - mu) / sd
    S = np.cov(np.vstack([Ez, Cz]).T) + np.eye(E.shape[1]) * 1e-6
    L = np.linalg.cholesky(np.linalg.inv(S))
    nn = NearestNeighbors(n_neighbors=min(k, len(C))).fit(Cz @ L)
    d, ix = nn.kneighbors(Ez @ L)
    return ix, d

def revised_match(exp, pool, k=K_MATCH):
    pe = np.minimum(exp["prior_max_class_window"].to_numpy(), 1); pp = np.minimum(pool["prior_max_class_window"].to_numpy(), 1)
    X = np.vstack([exp[PSCOV].to_numpy(float), pool[PSCOV].to_numpy(float)]); y = np.r_[np.ones(len(exp)), np.zeros(len(pool))]
    Xs = (X - X.mean(0)) / X.std(0).clip(1e-9)
    ps = LogisticRegression(max_iter=5000).fit(Xs, y).predict_proba(Xs)[:, 1].clip(1e-6, 1 - 1e-6)
    lg = np.log(ps / (1 - ps)); le, lp = lg[:len(exp)], lg[len(exp):]; cal = PS_CAL_SD * lg.std()
    ctrl = np.full((len(exp), k), DUMMY, dtype=np.int64); nmatch = np.zeros(len(exp), int)
    ppid = pool.pid.to_numpy()
    for c in np.unique(pe):
        ei = np.nonzero(pe == c)[0]; ci = np.nonzero(pp == c)[0]
        if len(ci) < k:
            continue
        ix, _ = mahal_match(exp.iloc[ei][CAN].to_numpy(float), pool.iloc[ci][CAN].to_numpy(float), N_CAND)
        for r, i in enumerate(ei):
            ok = ci[ix[r]][np.abs(lp[ci[ix[r]]] - le[i]) <= cal][:k]
            ctrl[i, :len(ok)] = ppid[ok]; nmatch[i] = len(ok)
    keep = nmatch >= 1
    return {"exp_pid": exp.pid.to_numpy()[keep], "ctrl_pid": ctrl[keep], "n_match": nmatch[keep], "pool_pid": ppid,
            "n_exposed_eligible": len(exp), "caliper_logit": cal}

def old_match(exp, pool, k=K_MATCH):
    ix, d = mahal_match(exp[FEATS].to_numpy(float), pool[FEATS].to_numpy(float), k)
    return {"exp_pid": exp.pid.to_numpy(), "ctrl_pid": pool.pid.to_numpy()[ix], "n_match": np.full(len(exp), k), "pool_pid": pool.pid.to_numpy(),
            "n_exposed_eligible": len(exp)}

mi_all = meta.set_index("pid")
def balance_rows(eid, g, exp, pool, mt, label):
    rows = []
    cp = mt["ctrl_pid"].ravel(); cp = cp[cp != DUMMY]
    used = mi_all.loc[cp]; ex = mi_all.loc[mt["exp_pid"]]
    for f in FEATS:
        e_, p_, m_ = ex[f].astype(float), pool[f].astype(float), used[f].astype(float)
        e0 = exp[f].astype(float)
        sdp = math.sqrt((e0.var() + p_.var()) / 2) or 1e-9
        rows.append({"event_id": eid, "event_name": g.event_name.iloc[0], "control_set": label, "covariate": f.replace("f_", ""),
                     "mean_exposed": e_.mean(), "mean_controls_all": p_.mean(), "mean_controls_matched": m_.mean(),
                     "smd_before": (e0.mean() - p_.mean()) / sdp, "smd_after": (e_.mean() - m_.mean()) / sdp,
                     "var_ratio_after": e_.var() / m_.var() if m_.var() > 0 else np.nan,
                     "n_exposed": len(mt["exp_pid"]), "n_exposed_eligible": mt["n_exposed_eligible"], "n_control_pool": len(pool),
                     "n_unique_matched_controls": len(np.unique(cp)), "share_exposed_retained": len(mt["exp_pid"]) / mt["n_exposed_eligible"],
                     "mean_controls_per_exposed": float(np.mean(mt["n_match"]))})
    return rows

def hav_km(la1, lo1, la2, lo2):
    la1, lo1, la2, lo2 = map(np.radians, (la1, lo1, la2, lo2))
    a_ = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a_))

match, match_near, match_v1, match_geo = {}, {}, {}, {}; bal_rows = []; dist_rows = []
for eid, g in meta.groupby("event_id", sort=False):
    exp = g[(g.exposure_class != "control") & g.baseline_ok]
    pool = g[(g.exposure_class == "control") & g.baseline_ok & (g.censor_m50 > 24)]
    match[eid] = revised_match(exp, pool); bal_rows += balance_rows(eid, g, exp, pool, match[eid], "all_pool")
    match_v1[eid] = old_match(exp, pool); bal_rows += balance_rows(eid, g, exp, pool, match_v1[eid], "v1_original")
    near = pool[pool["dist_to_nearest_exposed_km"] <= 150]
    if len(near) >= 50:
        match_near[eid] = revised_match(exp, near); bal_rows += balance_rows(eid, g, exp, near, match_near[eid], "near_150km")
    geo = pool[pool["dist_to_nearest_exposed_km"] <= 300]
    if len(geo) >= 100:
        match_geo[eid] = revised_match(exp, geo); bal_rows += balance_rows(eid, g, exp, geo, match_geo[eid], "geo_300km")
    for lab, md in (("all_pool", match), ("v1_original", match_v1), ("near_150km", match_near), ("geo_300km", match_geo)):
        if eid not in md:
            continue
        mt = md[eid]; ex = mi_all.loc[mt["exp_pid"]]
        dd = []
        for i, row_ in enumerate(mt["ctrl_pid"]):
            cc = row_[row_ != DUMMY]
            if len(cc):
                cm = mi_all.loc[cc]; dd.extend(hav_km(ex.cell_lat.iloc[i], ex.cell_lon.iloc[i], cm.cell_lat.values, cm.cell_lon.values))
        dist_rows.append({"event_id": eid, "event_name": g.event_name.iloc[0], "control_set": lab, "median_exposed_control_km": float(np.median(dd)),
                          "q90_exposed_control_km": float(np.percentile(dd, 90))})
bal = pd.DataFrame(bal_rows); match_dist = pd.DataFrame(dist_rows)
print("matching done", f"{time.time()-t0:.0f}s", flush=True)

# ---------------------------------------------------------------- bias correction (regression adjustment, Abadie-Imbens)
XZ_mu = meta[BCOV].astype(float).mean(); XZ_sd = meta[BCOV].astype(float).std().replace(0, 1)
def Xstd(pids):
    return ((mi_all.loc[pids, BCOV].astype(float) - XZ_mu) / XZ_sd).to_numpy()
_bc_cache = {}
def bc_adjust(eid, key, mdict, label):
    """n_exp x 61 adjustment = (X_i - mean_j X_j) gamma_m, gamma_m from month-wise OLS of control-pool anomalies on covariates."""
    ck = (eid, key, label)
    if ck in _bc_cache:
        return _bc_cache[ck]
    mt = mdict[eid]; M = A[key]["M"]; pool = mt["pool_pid"]
    Xp = np.column_stack([np.ones(len(pool)), Xstd(pool)]); Yp = M[pool].astype(float)
    G = np.zeros((len(MONTHS), len(BCOV)))
    for j in range(len(MONTHS)):
        ok = np.isfinite(Yp[:, j])
        if ok.sum() >= 30:
            coef, *_ = np.linalg.lstsq(Xp[ok], Yp[ok, j], rcond=None); G[j] = coef[1:]
    Xe = Xstd(mt["exp_pid"])
    cp = mt["ctrl_pid"]; Xc = np.zeros_like(Xe); 
    for i in range(len(cp)):
        cc = cp[i][cp[i] != DUMMY]; Xc[i] = Xstd(cc).mean(0)
    adj = (Xe - Xc) @ G.T
    _bc_cache[ck] = adj
    return adj
# ---------------------------------------------------------------- estimation helpers
def smooth(b):
    b = np.asarray(b, float); f = np.isfinite(b)
    k = np.ones(3)
    num = np.convolve(np.where(f, b, 0.0), k, mode="same"); den = np.convolve(f.astype(float), k, mode="same")
    return np.where(den >= 2, num / np.maximum(den, 1), np.nan)

def boot_weights(blocks, Bn, rng):
    ub, inv = np.unique(blocks, return_inverse=True)
    cnt = rng.multinomial(len(ub), np.full(len(ub), 1 / len(ub)), size=Bn)  # Bn x nblocks
    return cnt[:, inv].astype(float)  # Bn x ncells

def wmean(Wt, Dm):
    fin = np.isfinite(Dm)
    return (Wt @ np.where(fin, Dm, 0.0)) / np.maximum(Wt @ fin.astype(float), 1e-12)

def metrics(b, fu_last, cens_m, band=None, prebase=None):
    s = smooth(b)
    out = {}
    out["immediate_m0_1"] = np.nanmean(b[[IDX(0), IDX(1)]])
    out["mean_m0_6"] = np.nanmean(b[IDX(0):IDX(6) + 1])
    out["mean_m7_12"] = np.nanmean(b[IDX(7):IDX(12) + 1])
    seg = s[IDX(0):IDX(12) + 1]
    if np.all(np.isnan(seg)):
        out.update(trough=np.nan, trough_month=np.nan)
    else:
        j = int(np.nanargmin(seg)); out["trough"] = seg[j]; out["trough_month"] = j
    for t in (3, 6, 12):
        out[f"eff_m{t}"] = s[IDX(t)]
    pre_s = s[:IDX(-1) + 1]
    bandv = 2 * np.nanstd(pre_s) if band is None else band
    out["noise_band"] = bandv
    out["pre_mean_m24_1"] = np.nanmean(b[IDX(-24):IDX(-1) + 1])
    end = min(fu_last, cens_m)
    rec = np.nan; censored = False
    if np.isfinite(out.get("trough", np.nan)) and out["trough"] < -bandv:
        tm = int(out["trough_month"])
        for m in range(tm + 1, end):
            if s[IDX(m)] >= -bandv and (m + 1 > end - 1 or s[IDX(m + 1)] >= -bandv):
                rec = m; break
        if np.isnan(rec):
            censored = True
    out["recovery_month"] = rec; out["recovery_censored"] = censored; out["recovery_window_end"] = end
    return out

def group_cells(eid, grp, mdict=None):
    g = meta[meta.event_id == eid].set_index("pid")
    ep = (match if mdict is None else mdict)[eid]["exp_pid"]; cls = g.loc[ep, "exposure_class"].to_numpy()
    sel = {"R64": cls == "R64", "R50": cls == "R50", "R34": cls == "R34", "R50plus": np.isin(cls, ["R50", "R64"]),
           "all_exposed": np.ones(len(cls), bool)}[grp]
    return sel

def unit_diffs(eid, M, censor=True, cens_col="censor_m50", mdict=None, bc=None):
    g = meta.set_index("pid")
    mdict = match if mdict is None else mdict
    ep = mdict[eid]["exp_pid"]; cp = mdict[eid]["ctrl_pid"]
    E = M[ep].astype(float).copy()
    if censor:
        cm = g.loc[ep, cens_col].to_numpy()
        for i, c in enumerate(cm):
            if c < 99:
                E[i, IDX(c):] = np.nan
    Cm = np.nanmean(M[cp].astype(float), axis=1)  # nexp x 61 (padding rows are NaN)
    D = E - Cm
    if bc is not None:
        D = D - bc
    return D, E

# ---------------------------------------------------------------- main loop
SPECS = {  # name: (index, qa, baseline, matched, censor, seasonal_diff)
    "primary_matched": ("ndvi", "vrelax", "main", True, True, False),
    "evi_matched": ("evi", "vrelax", "main", True, True, False),
    "strictqa_matched": ("ndvi", "vstrict", "main", True, True, False),
    "unmatched_allcontrols": ("ndvi", "vrelax", "main", False, True, False),
    "seasonal_diff_matched": ("ndvi", "vrelax", "main", True, True, True),
    "no_censoring_matched": ("ndvi", "vrelax", "main", True, False, False),
    "censor_r34_matched": ("ndvi", "vrelax", "main", True, "r34", False),
    "evi_seasonal_diff_matched": ("evi", "vrelax", "main", True, True, True),
    "near_controls_matched": ("ndvi", "vrelax", "main", "near", True, False),
    "original_matching_v1": ("ndvi", "vrelax", "main", "v1", True, False),
    "revised_no_biascorr": ("ndvi", "vrelax", "main", "nobc", True, False),
    "geo_300km_matched": ("ndvi", "vrelax", "main", "geo", True, False),
}
MDICT = {True: ("match", None), "near": ("match_near", None), "v1": ("match_v1", None), "nobc": ("match", None), "geo": ("match_geo", None)}
MIN_CELLS_ABS, MIN_CELLS_FRAC = 30, 0.20
import statsmodels.api as sm
def its_hac(b, fu_last):
    m = MONTHS[(MONTHS <= fu_last)]
    y = b[IDX(m[0]):IDX(m[-1]) + 1]
    ok = np.isfinite(y)
    X = np.column_stack([np.ones(len(m)), (m >= 0) & (m <= 6), (m >= 7) & (m <= 12), (m >= 13)]).astype(float)
    r = sm.OLS(y[ok], X[ok]).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
    pre = (m < 0) & ok
    xp = m[pre].astype(float)
    rs = sm.OLS(y[pre], sm.add_constant(xp)).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
    rm = sm.OLS(y[pre], np.ones((pre.sum(), 1))).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
    mu, se = rm.params[0], rm.bse[0]; marg = 0.01
    p_tost = max(1 - stats.norm.cdf((mu + marg) / se), stats.norm.cdf((mu - marg) / se))
    return {"its_post0_6": r.params[1], "its_post0_6_se_hac": r.bse[1], "its_post0_6_p": r.pvalues[1],
            "its_post7_12": r.params[2], "its_post7_12_p": r.pvalues[2], "its_post13_24": r.params[3], "its_post13_24_p": r.pvalues[3],
            "pretrend_slope_hac": rs.params[1], "pretrend_slope_hac_p": rs.pvalues[1], "pre_mean_hac": mu,
            "pre_equivalence_tost_p_margin0p01": p_tost}
GROUPS = ["R64", "R50", "R34", "R50plus", "all_exposed"]
es_rows, sum_rows, pre_rows, extra_rows = [], [], [], []
BLOCKS_ALT = (0.10, 0.25, 0.50, 1.00)
def sensitivity_extras(eid, ename, grp, Dg, blocks, pids, b, low):
    """Block-size sensitivity and missing-data sensitivity for the months 0-6 primary estimand."""
    rng2 = np.random.default_rng(SEED + 99)
    out = {"event_id": eid, "event_name": ename, "group": grp, "n_cells": len(Dg)}
    sl = slice(IDX(0), IDX(6) + 1); okm = ~low[sl]
    def m06(bmat):
        x = bmat[..., sl][..., okm]; return np.nanmean(x, -1)
    out["mean_m0_6"] = float(m06(b[None])[0])
    la, lo_ = mi_all.loc[pids, "cell_lat"].to_numpy(), mi_all.loc[pids, "cell_lon"].to_numpy()
    for bs in BLOCKS_ALT:
        blk = (np.floor(la / bs)).astype(int).astype(str) + "_" + (np.floor(lo_ / bs)).astype(int).astype(str)
        Wt = boot_weights(blk, B, rng2); bb = wmean(Wt, Dg); v = m06(bb)
        out[f"ci_lo_block{bs:.2f}"], out[f"ci_hi_block{bs:.2f}"] = np.nanpercentile(v, 2.5), np.nanpercentile(v, 97.5)
        out[f"n_blocks_{bs:.2f}"] = len(np.unique(blk))
    O = np.isfinite(Dg[:, sl])                      # observed cell-months in months 0-6
    nobs = O.sum(1); out["share_cell_months_observed_m0_6"] = float(O[:, okm].mean())
    Wt = boot_weights(blocks, B, rng2)
    cc = nobs >= 5                                   # complete-case: >= 5 of 7 months observed
    out["n_complete_case"] = int(cc.sum())
    if cc.sum() >= 10:
        out["cc_mean_m0_6"] = float(m06(np.nanmean(Dg[cc], 0)[None])[0]); v = m06(wmean(Wt[:, cc], Dg[cc]))
        out["cc_ci_lo"], out["cc_ci_hi"] = np.nanpercentile(v, 2.5), np.nanpercentile(v, 97.5)
    D0 = Dg.copy(); x = D0[:, sl]; x[~np.isfinite(x)] = 0.0; D0[:, sl] = x   # null imputation (missing = no effect)
    out["null_imp_mean_m0_6"] = float(np.nanmean(np.nanmean(D0[:, sl], 0))); v = np.nanmean(wmean(Wt, D0)[:, sl], 1)
    out["null_imp_ci_lo"], out["null_imp_ci_hi"] = np.nanpercentile(v, 2.5), np.nanpercentile(v, 97.5)
    Xc = Xstd(pids); Wipw = np.ones_like(Dg)          # inverse probability of observation weighting
    for j in range(IDX(0), IDX(6) + 1):
        y = np.isfinite(Dg[:, j]).astype(int)
        if 0 < y.sum() < len(y):
            p = LogisticRegression(max_iter=3000).fit(Xc, y).predict_proba(Xc)[:, 1].clip(0.05, 1)
            Wipw[:, j] = 1 / p
    Dw = np.where(np.isfinite(Dg), Dg, np.nan)
    num = np.nansum(Wipw[:, sl] * np.nan_to_num(Dw[:, sl]), 0); den = np.nansum(Wipw[:, sl] * np.isfinite(Dw[:, sl]), 0)
    bi = num / np.maximum(den, 1e-12); out["ipw_mean_m0_6"] = float(np.nanmean(bi[okm]))
    vb = []
    for r in range(B):
        w = Wt[r][:, None] * Wipw[:, sl]; nn_ = np.nansum(w * np.nan_to_num(Dw[:, sl]), 0); dd_ = np.nansum(w * np.isfinite(Dw[:, sl]), 0)
        vb.append(np.nanmean((nn_ / np.maximum(dd_, 1e-12))[okm]))
    out["ipw_ci_lo"], out["ipw_ci_hi"] = np.nanpercentile(vb, 2.5), np.nanpercentile(vb, 97.5)
    obs_hi = nobs >= 4                               # observed vs missing: cells observed in >= 4 of 7 months vs fewer
    X_ = mi_all.loc[pids, BCOV].astype(float)
    if 5 <= obs_hi.sum() <= len(obs_hi) - 5:
        smds = {}
        for f in BCOV:
            a_, b_ = X_[f].to_numpy()[obs_hi], X_[f].to_numpy()[~obs_hi]
            sdp = math.sqrt((a_.var() + b_.var()) / 2) or 1e-9; smds[f] = (a_.mean() - b_.mean()) / sdp
        k_ = max(smds, key=lambda z: abs(smds[z])); out["obs_vs_missing_max_abs_smd"] = abs(smds[k_]); out["obs_vs_missing_max_cov"] = k_.replace("f_", "")
        out["obs_vs_missing_canopy_max_abs_smd"] = max(abs(smds[f]) for f in CAN)
    out["n_cells_mostly_observed"] = int(obs_hi.sum())
    return out
boot_store = {}

def seasonal_diff(bmat):
    out = bmat.copy()
    for m in range(0, 25):
        lags = [m - 12 * k for k in (1, 2, 3) if -36 <= m - 12 * k < 0]
        out[..., IDX(m)] = bmat[..., IDX(m)] - np.nanmean(np.stack([bmat[..., IDX(l)] for l in lags], -1), -1)
    return out

ORIG8 = sorted(["2007_SIDR_2007314N10093", "2008_NARGIS_2008117N11090", "2009_AILA_2009143N17089", "2020_AMPHAN_2020136N10088", "2021_YAAS_2021143N15090", "2022_SITRANG_2022295N13093", "2024_REMAL_2024145N14087", "2024_DANA_2024295N15092"])
EV_SEED = {e: i for i, e in enumerate(ORIG8)}   # original seeds unchanged (results of the 8 published events reproduce exactly)
for e in sorted(set(meta.event_id.unique()) - set(ORIG8)):
    EV_SEED[e] = len(EV_SEED)
for eid, g in meta.groupby("event_id", sort=False):
    if _skip(eid):
        continue
    rng = np.random.default_rng(SEED + 1000 * EV_SEED[eid])
    ename = g.event_name.iloc[0]; fu_last = fu[eid]
    exp_meta = meta.set_index("pid").loc[match[eid]["exp_pid"]]
    blocks_e = exp_meta["block"].to_numpy()
    for sname, (ix_, qa, base, matched, censor, sd_) in SPECS.items():
        M = A[(ix_, qa, base)]["M"]
        cens_col = "censor_m34" if censor == "r34" else "censor_m50"
        if matched == "near" and eid not in match_near:
            continue
        if matched == "geo" and eid not in match_geo:
            continue
        md_ = {True: match, "near": match_near, "v1": match_v1, "nobc": match, "geo": match_geo, False: match}[matched]
        bcv = bc_adjust(eid, (ix_, qa, base), md_, str(matched)) if matched in (True, "near", "geo") else None
        D, E = unit_diffs(eid, M, censor=bool(censor), cens_col=cens_col, mdict=md_, bc=bcv)
        exp_meta = meta.set_index("pid").loc[md_[eid]["exp_pid"]]; blocks_e = exp_meta["block"].to_numpy()
        pool = match[eid]["pool_pid"]
        Cpool = M[pool].astype(float)
        blocks_c = meta.set_index("pid").loc[pool, "block"].to_numpy()
        for grp in GROUPS:
            sel = group_cells(eid, grp, md_)
            if sel.sum() < 10:
                continue
            ccut = exp_meta.loc[sel.nonzero()[0] if False else exp_meta.index[sel], cens_col].to_numpy()
            # month at which >=50% of group cells censored by a later storm
            cm_sorted = np.sort(ccut)
            cens_m = int(cm_sorted[len(cm_sorted) // 2]) if censor and cm_sorted[len(cm_sorted) // 2] < 99 else 99
            if matched:
                Dg = D[sel]
                b = np.nanmean(Dg, 0)
                Wt = boot_weights(blocks_e[sel], B, rng)
                bb = wmean(Wt, Dg)
            else:
                Eg = E[sel]
                b = np.nanmean(Eg, 0) - np.nanmean(Cpool, 0)
                We = boot_weights(blocks_e[sel], B, rng); Wc = boot_weights(blocks_c, B, rng)
                bb = wmean(We, Eg) - wmean(Wc, Cpool)
            nE = np.isfinite(D[sel] if matched else E[sel]).sum(0)
            if matched:
                cps = md_[eid]["ctrl_pid"][sel]; uc = np.setdiff1d(np.unique(cps), [DUMMY])
                nC = np.isfinite(M[uc].astype(float)).sum(0)
            else:
                nC = np.isfinite(Cpool).sum(0)
            need = max(MIN_CELLS_ABS, MIN_CELLS_FRAC * sel.sum())
            low = nE < need
            b = np.where(low, np.nan, b); bb[:, low] = np.nan
            if sd_:
                b = seasonal_diff(b[None])[0]; bb = seasonal_diff(bb)
            lo, hi = np.nanpercentile(bb, 2.5, 0), np.nanpercentile(bb, 97.5, 0)
            for m in MONTHS:
                es_rows.append({"event_id": eid, "event_name": ename, "spec": sname, "index": ix_.upper(), "qa": qa,
                                "group": grp, "rel_month": int(m), "beta": b[IDX(m)], "ci_lo": lo[IDX(m)], "ci_hi": hi[IDX(m)],
                                "n_exposed_cells_with_data": int(nE[IDX(m)]), "n_control_cells_with_data": int(nC[IDX(m)]), "month_insufficient_coverage": bool(low[IDX(m)])})
            mt = metrics(b, fu_last, cens_m)
            band = mt["noise_band"]
            mb = [metrics(bb[r], fu_last, cens_m, band=band) for r in range(B)]
            row = {"event_id": eid, "event_name": ename, "spec": sname, "index": ix_.upper(), "group": grp,
                   "n_exposed_cells": int(sel.sum()), "n_control_cells": int(len(np.setdiff1d(np.unique(md_[eid]["ctrl_pid"][sel]), [DUMMY]))) if matched else len(pool),
                   "followup_last_month": fu_last, "followup_censored_2024": fu_last < 24,
                   "storm_censor_month_median": cens_m if cens_m < 99 else np.nan}
            for k_, v in mt.items():
                row[k_] = v
            for k_ in ("immediate_m0_1", "mean_m0_6", "mean_m7_12", "trough", "eff_m3", "eff_m6", "eff_m12", "pre_mean_m24_1"):
                vals = np.array([x[k_] for x in mb], float)
                row[k_ + "_ci_lo"], row[k_ + "_ci_hi"] = np.nanpercentile(vals, 2.5), np.nanpercentile(vals, 97.5)
            m06b = np.array([x["mean_m0_6"] for x in mb], float); m06b = m06b[np.isfinite(m06b)]
            row["boot_p_two_sided_m0_6"] = float(max(2 * min(np.mean(m06b >= 0), np.mean(m06b <= 0)), 1 / len(m06b))) if len(m06b) else np.nan
            rm = np.array([x["recovery_month"] for x in mb], float)
            row["recovery_month_boot_median"] = np.nanmedian(rm) if np.isfinite(rm).any() else np.nan
            row["recovery_boot_share_censored_or_none"] = float(np.mean(~np.isfinite(rm)))
            for t in (3, 6, 12):
                row[f"recovery_pct_m{t}"] = (1 - mt[f"eff_m{t}"] / mt["trough"]) * 100 if (mt["trough"] < -band) else np.nan
            row["decline_detectable"] = bool(mt["trough"] < -band and row["trough_ci_hi"] < 0 and row["mean_m0_6_ci_hi"] < 0)
            # pretrend tests (6 bins of 6 months)
            bins = [np.nanmean(b[IDX(-36 + 6 * k):IDX(-36 + 6 * k + 5) + 1]) for k in range(6)]
            bbins = np.column_stack([np.nanmean(bb[:, IDX(-36 + 6 * k):IDX(-36 + 6 * k + 5) + 1], 1) for k in range(6)])
            S = np.cov(bbins.T) + np.eye(6) * 1e-10
            chi2 = float(np.array(bins) @ np.linalg.solve(S, np.array(bins)))
            xs = MONTHS[:36].astype(float); okx = np.isfinite(b[:36])
            slope = np.polyfit(xs[okx], b[:36][okx], 1)[0] if okx.sum() > 5 else np.nan
            bsl = [np.polyfit(xs[np.isfinite(bb[r, :36])], bb[r, :36][np.isfinite(bb[r, :36])], 1)[0] for r in range(0, B, 5) if np.isfinite(bb[r, :36]).sum() >= 3]
            row.update({"pretrend_wald_chi2_6bins": chi2, "pretrend_wald_p": float(1 - stats.chi2.cdf(chi2, 6)),
                        "pretrend_slope_per_month": slope, "pretrend_slope_ci_lo": np.nanpercentile(bsl, 2.5),
                        "pretrend_slope_ci_hi": np.nanpercentile(bsl, 97.5),
                        "pre_months_ci_excluding_0_share": float(np.mean((lo[:36] > 0) | (hi[:36] < 0)))})
            try:
                row.update(its_hac(b, fu_last))
            except Exception as ex:
                row["its_error"] = str(ex)[:80]
            row["n_months_insufficient_coverage_post"] = int(low[IDX(0):IDX(min(fu_last, 24)) + 1].sum())
            sum_rows.append(row)
            if sname in ("primary_matched", "evi_matched"):
                boot_store[(eid, grp, ix_)] = {"b": b, "lo": lo, "hi": hi, "band": band, "fu_last": fu_last, "cens_m": cens_m}
            if sname == "primary_matched" and grp in ("R64", "R50", "R34"):
                extra_rows.append(sensitivity_extras(eid, ename, grp, D[sel], blocks_e[sel], exp_meta.index[sel].to_numpy(), b, low))
    print(eid, "specs done", f"{time.time()-t0:.0f}s", flush=True)

ES = pd.DataFrame(es_rows); SUM = pd.DataFrame(sum_rows)

# ---------------------------------------------------------------- placebo-in-time (fake landfall at -12 months) and placebo-in-space
plac_rows = []
for eid, g in meta.groupby("event_id", sort=False):
    if _skip(eid):
        continue
    rng = np.random.default_rng(SEED + 1000 * EV_SEED[eid] + 500)
    ename = g.event_name.iloc[0]
    exp_meta = meta.set_index("pid").loc[match[eid]["exp_pid"]]
    blocks_e = exp_meta["block"].to_numpy()
    for ix_ in ("ndvi", "evi"):
        Mp = A[(ix_, "vrelax", "placebo12")]["M"]
        Dp, Ep = unit_diffs(eid, Mp, censor=False, bc=bc_adjust(eid, (ix_, "vrelax", "placebo12"), match, "True"))
        Mr = A[(ix_, "vrelax", "main")]["M"]
        pool = match[eid]["pool_pid"]; blocks_c = meta.set_index("pid").loc[pool, "block"].to_numpy()
        for grp in ["R64", "R50", "R34", "R50plus", "all_exposed"]:
            sel = group_cells(eid, grp)
            if sel.sum() < 10:
                continue
            b = np.nanmean(Dp[sel], 0)
            Wt = boot_weights(blocks_e[sel], B, rng); bb = wmean(Wt, Dp[sel])
            fake = lambda x: (np.nanmean(x[..., IDX(-12):IDX(-6) + 1], -1))
            fake_imm = lambda x: np.nanmean(x[..., [IDX(-12), IDX(-11)]], -1)
            def fake_trough(x):
                s = np.array([smooth(r) for r in np.atleast_2d(x)])
                return np.nanmin(s[:, IDX(-12):IDX(0)], 1)
            real = SUM[(SUM.event_id == eid) & (SUM.group == grp) & (SUM.spec == ("primary_matched" if ix_ == "ndvi" else "evi_matched"))].iloc[0]
            # placebo in space: random control blocks as pseudo-treated, unmatched DiD on real dates
            Cp = Mr[pool].astype(float)
            ub = np.unique(blocks_c); n_target = int(sel.sum())
            perm_m06, perm_tr = [], []
            Er = Mr[match[eid]["exp_pid"][sel]].astype(float)
            real_un = np.nanmean(Er, 0) - np.nanmean(Cp, 0)
            for r in range(1999):
                order = rng.permutation(ub); pick = []; n = 0
                for bk in order:
                    idx = np.nonzero(blocks_c == bk)[0]; pick.extend(idx); n += len(idx)
                    if n >= n_target:
                        break
                pick = np.array(pick); rest = np.setdiff1d(np.arange(len(pool)), pick)
                if len(rest) < 10:
                    continue
                d = np.nanmean(Cp[pick], 0) - np.nanmean(Cp[rest], 0)
                perm_m06.append(np.nanmean(d[IDX(0):IDX(6) + 1]))
                perm_tr.append(np.nanmin(smooth(d)[IDX(0):IDX(12) + 1]))
            perm_m06, perm_tr = np.array(perm_m06), np.array(perm_tr)
            r_m06 = np.nanmean(real_un[IDX(0):IDX(6) + 1]); r_tr = np.nanmin(smooth(real_un)[IDX(0):IDX(12) + 1])
            plac_rows.append({"event_id": eid, "event_name": ename, "index": ix_.upper(), "group": grp,
                              "real_mean_m0_6": real["mean_m0_6"], "real_trough": real["trough"],
                              "placebo_time_mean_m0_6": float(fake(b)), "placebo_time_mean_m0_6_ci_lo": np.nanpercentile(fake(bb), 2.5),
                              "placebo_time_mean_m0_6_ci_hi": np.nanpercentile(fake(bb), 97.5),
                              "placebo_time_immediate": float(fake_imm(b)), "placebo_time_trough": float(fake_trough(b)[0]),
                              "placebo_time_trough_ci_lo": np.nanpercentile(fake_trough(bb), 2.5),
                              "placebo_time_trough_ci_hi": np.nanpercentile(fake_trough(bb), 97.5),
                              "real_minus_placebo_trough": real["trough"] - float(fake_trough(b)[0]),
                              "unmatched_real_mean_m0_6": r_m06, "unmatched_real_trough": r_tr,
                              "placebo_space_n_perm": len(perm_m06),
                              "placebo_space_p_mean_m0_6_one_sided": float((1 + np.sum(perm_m06 <= r_m06)) / (len(perm_m06) + 1)),
                              "placebo_space_p_trough_one_sided": float((1 + np.sum(perm_tr <= r_tr)) / (len(perm_tr) + 1)),
                              "placebo_space_trough_q05": np.nanpercentile(perm_tr, 5) if len(perm_tr) else np.nan})
    print(eid, "placebos done", f"{time.time()-t0:.0f}s", flush=True)
PL = pd.DataFrame(plac_rows)
if MODE == "part":
    pd.to_pickle({"es": es_rows, "sum": sum_rows, "pl": plac_rows, "extra": extra_rows, "boot": boot_store}, W + f"stageB_part_{ONLY}.pkl")
    print("PART SAVED", ONLY, f"{time.time()-t0:.0f}s", flush=True); raise SystemExit
if MODE == "merge":
    import glob as _g
    for f in sorted(_g.glob(W + "stageB_part_*.pkl")):
        pp_ = pd.read_pickle(f); es_rows += pp_["es"]; sum_rows += pp_["sum"]; plac_rows += pp_["pl"]; extra_rows += pp_["extra"]; boot_store.update(pp_["boot"])
    ES = pd.DataFrame(es_rows); SUM = pd.DataFrame(sum_rows); PL = pd.DataFrame(plac_rows)
    print("MERGED parts:", ES.event_id.nunique(), "events", flush=True)

# ---------------------------------------------------------------- pooled dose-response across events (random effects)
pool_rows = []
for spec in ("primary_matched", "evi_matched"):
    for grp in ("R64", "R50", "R34", "R50plus"):
        d = SUM[(SUM.spec == spec) & (SUM.group == grp)].dropna(subset=["mean_m0_6"])
        if len(d) < 2:
            continue
        y = d["mean_m0_6"].to_numpy(); se = ((d["mean_m0_6_ci_hi"] - d["mean_m0_6_ci_lo"]) / 3.92).to_numpy()
        w = 1 / se ** 2; fe = np.sum(w * y) / np.sum(w); Q = np.sum(w * (y - fe) ** 2); dfq = len(y) - 1
        tau2 = max(0, (Q - dfq) / (np.sum(w) - np.sum(w ** 2) / np.sum(w)))
        wr = 1 / (se ** 2 + tau2); re = np.sum(wr * y) / np.sum(wr); sre = math.sqrt(1 / np.sum(wr))
        pool_rows.append({"spec": spec, "group": grp, "n_events": len(y), "events": ";".join(d.event_name),
                          "random_effects_mean_m0_6": re, "ci_lo": re - 1.96 * sre, "ci_hi": re + 1.96 * sre,
                          "tau2": tau2, "I2": max(0, (Q - dfq) / Q) if Q > 0 else 0, "Q": Q})
POOL = pd.DataFrame(pool_rows)
bc_primary = {(eid, ix_): bc_adjust(eid, (ix_, "vrelax", "main"), match, "True") for eid in match for ix_ in ("ndvi", "evi")}
EXTRA = pd.DataFrame(extra_rows)
pd.to_pickle({"meta": meta, "match": match, "match_near": match_near, "match_v1": match_v1, "match_geo": match_geo, "bal": bal, "match_dist": match_dist,
              "ES": ES, "SUM": SUM, "PL": PL, "POOL": POOL, "storms": storms, "fu": fu, "boot": boot_store, "bc_primary": bc_primary,
              "EXTRA": EXTRA, "DUMMY": DUMMY}, W + "stageB.pkl")
print("DONE", f"{time.time()-t0:.0f}s", flush=True)
