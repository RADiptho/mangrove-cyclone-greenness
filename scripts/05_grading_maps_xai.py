"""Stage 05. Evidence grading, study-area map, forest plots, pre-trend figure, exploratory interpretability
(ridge regression, gradient boosting, permutation importance, SHAP; leave-one-event-out), dataset audit.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from p4.config import W, O, T, FIG, FDT, SUP, data, STORM_FLAGS
import pandas as pd, numpy as np, os, json, time, math, glob, hashlib, zipfile, subprocess, sys, shutil, platform, warnings, importlib.util, re
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle
from matplotlib.lines import Line2D
t0 = time.time()
SEED = 20261004
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                     "savefig.bbox": "tight", "savefig.pad_inches": 0.04, "svg.fonttype": "none", "pdf.fonttype": 42})
def save(fig, name):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIG + f"{name}.{ext}", dpi=600, facecolor="white")
    plt.close(fig)
ORDER = ["Sidr", "Nargis", "Aila", "Fani", "Bulbul", "Amphan", "Yaas", "Sitrang", "Mocha", "Remal", "Dana"]
YEAR = {"Sidr": 2007, "Nargis": 2008, "Aila": 2009, "Fani": 2019, "Bulbul": 2019, "Amphan": 2020, "Yaas": 2021, "Sitrang": 2022, "Mocha": 2023, "Remal": 2024, "Dana": 2024}

# ============================================================== 1. evidence grading (user framework) and relabel
RELABEL = {"positive anomaly (not damage)": "positive anomaly (not interpretable)",
           "pretrend violation (not interpretable)": "pre-trend failure (not interpretable)",
           "no detectable decline": "no robust decline evidence"}
C = pd.read_pickle(W + "stageC.pkl"); EV = C["EV"].copy()
PT = pd.read_csv(T + "pretrend_placebo_test_summary.csv")
def crit(r):
    evi = EV[(EV.event_id == r.event_id) & (EV.group == r.group) & (EV["index"] == "EVI")]
    pt = PT[(PT.event_id == r.event_id) & (PT.group == r.group) & (PT["index"] == r["index"])].iloc[0]
    out = {"c1_ci_below_zero": bool(r.mean_m0_6_ci_hi < 0),
           "c2_hac_p_lt_0p05": bool(r.its_post0_6_p < 0.05 and r.its_post0_6 < 0),
           "c3_space_placebo_p_lt_0p05": bool(pt.placebo_space_p_mean_m0_6_one_sided < 0.05),
           "c4_strict_qa_same_sign": (np.nan if pd.isna(r.sens_strictqa_mean_m0_6) else bool(r.sens_strictqa_mean_m0_6 < 0)),
           "c5_near_controls_same_sign": (np.nan if pd.isna(r.sens_near_controls_mean_m0_6) else bool(r.sens_near_controls_mean_m0_6 < 0)),
           "c6_evi_same_sign": (bool(evi.mean_m0_6.iloc[0] < 0) if len(evi) else np.nan),
           "c7_no_pretrend_hac": bool(pt.pretrend_slope_hac_p >= 0.05),
           "c8_complete_24m_followup": bool(r.followup_last_month >= 24),
           "space_placebo_p": pt.placebo_space_p_mean_m0_6_one_sided, "pretrend_hac_p": pt.pretrend_slope_hac_p,
           "evi_mean_m0_6": evi.mean_m0_6.iloc[0] if len(evi) else np.nan}
    core = [out["c1_ci_below_zero"], out["c2_hac_p_lt_0p05"], out["c3_space_placebo_p_lt_0p05"], out["c4_strict_qa_same_sign"] is True,
            out["c5_near_controls_same_sign"] is True, out["c6_evi_same_sign"] is True]
    if not out["c7_no_pretrend_hac"]:
        g = "pre-trend failure (not interpretable)"
    elif all(core):
        g = "robust decline"
    elif out["c1_ci_below_zero"] and out["c2_hac_p_lt_0p05"]:
        g = "suggestive decline"
    elif r.mean_m0_6_ci_lo > 0 and r.its_post0_6_p < 0.05:
        g = "positive anomaly (not interpretable)"
    else:
        g = "no robust decline evidence"
    names = {"c1_ci_below_zero": "CI includes or exceeds 0", "c2_hac_p_lt_0p05": "HAC test not significant",
             "c3_space_placebo_p_lt_0p05": "space placebo p >= 0.05", "c4_strict_qa_same_sign": "strict QA sign differs",
             "c5_near_controls_same_sign": "nearby-control sign differs", "c6_evi_same_sign": "EVI sign differs",
             "c7_no_pretrend_hac": "pre-trend detected"}
    failed = [v for k, v in names.items() if out[k] is False]
    out["evidence_grade"] = g
    out["decisive_reasons"] = "all criteria met" if g == "robust decline" else ("; ".join(failed) if failed else "positive and significant: not a decline")
    out["censoring_status"] = "complete 24 months" if out["c8_complete_24m_followup"] else "22.6 months, censored before a full 24-month follow-up"
    return pd.Series(out)
NDV = EV[(EV["index"] == "NDVI") & EV.group.isin(["R64", "R50", "R34"])].copy()
G = pd.concat([NDV[["event_id", "event_name", "index", "group", "n_exposed_cells", "mean_m0_6", "mean_m0_6_ci_lo", "mean_m0_6_ci_hi",
                    "its_post0_6_p", "sens_strictqa_mean_m0_6", "sens_near_controls_mean_m0_6"]].reset_index(drop=True),
               NDV.apply(crit, axis=1).reset_index(drop=True)], axis=1)
# check grading equals the previous classification (relabelled)
prev = NDV.evidence_class.map(lambda s: RELABEL.get(s, s)).reset_index(drop=True)
G["matches_prespecified_rule"] = (G.evidence_grade == prev)
G.to_csv(T + "evidence_grading_table.csv", index=False)
print("grading mismatches:", int((~G.matches_prespecified_rule).sum()), flush=True)
EV["evidence_class"] = EV.evidence_class.map(lambda s: RELABEL.get(s, s))
EV["followup_status"] = np.where(EV.followup_last_month < 24, "22.6 months, censored before a full 24-month follow-up", "complete 24 months")
EV.to_csv(T + "event_level_canopy_impact_summary.csv", index=False)
C["EV"] = EV; pd.to_pickle(C, W + "stageC.pkl")

# ============================================================== 2. new core figures

import p4.ibtracs as kit
ib = kit.load_ni_tracks(); clim = kit.radius_climatology(ib)
meta = C["meta"]; bob = pd.read_csv(data("inputs/bay_of_bengal_mangrove_cell_exposure_grid.csv.gz"))
from mpl_toolkits.basemap import Basemap
def fig_study_area():
    box_ = (9.5, 24.5, 80.5, 98.5)
    fig = plt.figure(figsize=(7.2, 6.0)); ax = fig.add_axes([0.07, 0.08, 0.86, 0.86])
    m = Basemap(projection="cyl", llcrnrlat=box_[0], urcrnrlat=box_[1], llcrnrlon=box_[2], urcrnrlon=box_[3], resolution="i", ax=ax)
    m.fillcontinents(color="#f2f0ea", lake_color="white", zorder=0); m.drawcoastlines(linewidth=0.35, color="#4d4d4d"); m.drawcountries(linewidth=0.3, color="#8c8c8c")
    ax.scatter(bob.lon, bob.lat, s=0.6, c="#1b7837", marker="s", lw=0, zorder=2, rasterized=True)
    cols = plt.cm.tab20(np.arange(len(ORDER)) % 20); rows = []
    for i, en in enumerate(ORDER):
        sid = meta[meta.event_name == en].sid.iloc[0]; tk = kit.prepare_track(ib[ib.SID == sid], clim)
        ax.plot(tk["lon"], tk["lat"], color=cols[i], lw=1.2, zorder=3, label=f"{en} {YEAR[en]}")
        lf = pd.to_datetime(meta[meta.event_name == en].landfall_utc.iloc[0]); tr = ib[ib.SID == sid].copy(); tr["dt"] = (tr.t - lf).abs(); p = tr.loc[tr.dt.idxmin()]
        ax.scatter([p.LON], [p.LAT], marker="*", s=70, c=[cols[i]], edgecolors="black", linewidths=0.5, zorder=5)
        rows.append({"event": en, "year": YEAR[en], "landfall_utc": lf, "landfall_lon": p.LON, "landfall_lat": p.LAT})
    for name, lo, la in [("Sundarbans", 89.0, 22.75), ("Ayeyarwady delta", 94.0, 15.25), ("Bhitarkanika", 86.1, 20.25), ("Rakhine coast", 93.9, 19.6), ("Andaman Islands", 93.2, 12.3)]:
        ax.text(lo, la, name, fontsize=7, style="italic", color="#14532d", zorder=6)
    from matplotlib.ticker import FuncFormatter
    ax.set_axis_on(); ax.set_xticks(np.arange(82, 99, 4)); ax.set_yticks(np.arange(10, 25, 3))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:.0f}\u00b0E")); ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:.0f}\u00b0N"))
    ax.tick_params(labelbottom=True, labelleft=True, labelsize=7)
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linewidth(0.6)
    h, l = ax.get_legend_handles_labels()
    h += [Line2D([], [], marker="s", ls="", color="#1b7837", ms=4, label="GMW v3 mangrove (0.01\u00b0 cells)"), Line2D([], [], marker="*", ls="", mfc="white", mec="black", ms=8, label="landfall")]
    ax.legend(handles=h, loc="lower left", fontsize=6.5, ncol=2, title="Best tracks (IBTrACS v04r01)", title_fontsize=7)
    axi = fig.add_axes([0.70, 0.66, 0.22, 0.26])
    mi = Basemap(projection="cyl", llcrnrlat=-5, urcrnrlat=38, llcrnrlon=60, urcrnrlon=115, resolution="c", ax=axi)
    mi.fillcontinents(color="#e8e6df", lake_color="white"); mi.drawcoastlines(linewidth=0.3)
    axi.add_patch(Rectangle((box_[2], box_[0]), box_[3] - box_[2], box_[1] - box_[0], fill=False, ec="#b2182b", lw=1))
    for sp in axi.spines.values():
        sp.set_visible(True); sp.set_linewidth(0.6)
    ax.set_title("Study area: Bay of Bengal mangroves and the eleven analysed cyclone tracks", fontsize=8.5, loc="left")
    pd.DataFrame(rows).to_csv(FDT + "fig02_study_area_landfalls.csv", index=False)
    save(fig, "fig02_study_area_event_map")
fig_study_area()

CLASS_COL = {"robust decline": "#b2182b", "suggestive decline": "#ef8a62", "no robust decline evidence": "#7f7f7f",
             "positive anomaly (not interpretable)": "#2a8c7f", "pre-trend failure (not interpretable)": "#e08214"}
def forest(index, name):
    d = EV[(EV["index"] == index) & EV.group.isin(["R64", "R50", "R34"])].copy()
    d["o"] = d.event_name.map({e: i for i, e in enumerate(ORDER)}); d["g"] = d.group.map({"R64": 0, "R50": 1, "R34": 2})
    d = d.sort_values(["o", "g"]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    y = np.arange(len(d))[::-1]
    for yi, r in zip(y, d.itertuples()):
        c = CLASS_COL[r.evidence_class]
        ax.plot([r.mean_m0_6_ci_lo, r.mean_m0_6_ci_hi], [yi, yi], color=c, lw=1.2)
        ax.scatter([r.mean_m0_6], [yi], color=c, s=22 if r.evidence_class != "robust decline" else 34, zorder=3,
                   marker="D" if r.evidence_class == "robust decline" else "o")
        ax.text(0.105, yi, f"n = {r.n_exposed_cells}", fontsize=5.8, va="center")
        ax.text(0.135, yi, r.evidence_class + (" \u2020" if r.followup_last_month < 24 else ""), fontsize=5.8, va="center", color=c)
    ax.axvline(0, color="black", lw=0.6)
    ax.set_yticks(y); ax.set_yticklabels([f"{r.event_name} {YEAR[r.event_name]} \u00b7 {r.group}" for r in d.itertuples()], fontsize=6.5)
    ax.set_xlim(-0.11, 0.10); ax.set_xlabel(f"\u0394{index}, months 0\u20136 (exposed \u2212 matched controls), 95% spatial block bootstrap CI")
    for k in range(1, len(ORDER)):
        idx = d.index[d.o == k].min(); ax.axhline(y[idx] + 0.5, color="#dddddd", lw=0.5)
    fig.text(0.13, 0.005, "\u2020 22.6 months, censored before a full 24-month follow-up. Gaps in monsoon-month MODIS coverage reduce precision for May landfalls.", fontsize=5.8)
    d[["event_name", "group", "n_exposed_cells", "mean_m0_6", "mean_m0_6_ci_lo", "mean_m0_6_ci_hi", "evidence_class", "followup_status"]].to_csv(FDT + f"{name}.csv", index=False)
    save(fig, name)
forest("NDVI", "fig06_forest_plot_ndvi_m0_6"); forest("EVI", "figS3_forest_plot_evi_m0_6")


import statsmodels.api as sm
ES = pd.read_csv(T + "did_or_event_study_results.csv")
def pretrend_fig():
    rows = []
    for en in ORDER:
        for grp in ("R64", "R50", "R34"):
            d = ES[(ES.event_name == en) & (ES.spec == "primary_matched") & (ES.group == grp) & (ES.rel_month < 0)].dropna(subset=["beta"])
            if len(d) < 10:
                continue
            x = d.rel_month.to_numpy(float); yv = d.beta.to_numpy(float)
            rr = sm.OLS(yv, sm.add_constant(x)).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
            rows.append({"event_name": en, "group": grp, "slope_per_year": 12 * rr.params[1], "ci_lo": 12 * rr.conf_int()[1][0], "ci_hi": 12 * rr.conf_int()[1][1], "p": rr.pvalues[1],
                         "pre_mean": float(np.mean(yv))})
    d = pd.DataFrame(rows); d.to_csv(FDT + "fig10a_pretrend_hac_slopes.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.0, 4.6)); yy = np.arange(len(d))[::-1]
    for y_, r in zip(yy, d.itertuples()):
        c = "#e08214" if r.p < 0.05 else "#4d4d4d"
        ax.plot([r.ci_lo, r.ci_hi], [y_, y_], color=c, lw=1.1); ax.scatter([r.slope_per_year], [y_], color=c, s=18, zorder=3)
        if r.p < 0.05:
            ax.text(r.ci_hi + 0.002, y_, f"p = {r.p:.3f}", fontsize=5.8, va="center", color=c)
    ax.axvline(0, color="black", lw=0.6)
    ax.set_yticks(yy); ax.set_yticklabels([f"{r.event_name} {YEAR[r.event_name]} \u00b7 {r.group}" for r in d.itertuples()], fontsize=6.5)
    ax.set_xlabel("Pre-landfall trend in \u0394NDVI (per year), months \u221236 to \u22121, 95% HAC CI")
    ax.legend(handles=[Line2D([], [], color="#4d4d4d", marker="o", label="no pre-trend (p \u2265 0.05)"), Line2D([], [], color="#e08214", marker="o", label="pre-trend detected (p < 0.05)")], fontsize=6, loc="lower right")
    save(fig, "fig10a_pretrend_diagnostics")
pretrend_fig()
print("new figures", f"{time.time()-t0:.0f}s", flush=True)

# ============================================================== 3. exploratory interpretability (cell level, leave-one-event-out)
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold
from sklearn.inspection import permutation_importance
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
CP = pd.read_csv(T + "final_event_cell_panel.csv")
X = CP[(CP.exposure_class != "control") & CP.in_analysis & CP.did_ndvi_m0_6.notna() & (CP.n_valid_months_post0_6 >= 3)].copy()
mm = meta.set_index(["event_id", "cell_uid"])
X["hours34_event"] = mm.loc[list(zip(X.event_id, X.cell_uid)), "hours34_event"].to_numpy()
dist = np.full(len(X), np.nan)
for en in ORDER:
    sid = meta[meta.event_name == en].sid.iloc[0]; tk = kit.prepare_track(ib[ib.SID == sid], clim)
    ii = np.nonzero((X.event_name == en).to_numpy())[0]
    la, lo = X.cell_lat.to_numpy()[ii], X.cell_lon.to_numpy()[ii]
    best = np.full(len(ii), np.inf)
    for a, b in zip(tk["lat"], tk["lon"]):
        best = np.minimum(best, kit.hav_km(la, lo, a, b))
    dist[ii] = best
X["dist_to_track_km"] = dist
X["wind_class_ordinal"] = X.exposure_class.map({"R34": 1, "R50": 2, "R64": 3})
FEATS = ["wind_class_ordinal", "hours34_event", "dist_to_track_km", "prior_max_class_window", "f_ndvi_level", "f_ndvi_amp", "f_ndvi_sd",
         "f_ndvi_pretrend", "f_evi_level", "gmw2020_frac_window340m", "n_valid_months_post0_6"]
X = X.dropna(subset=FEATS + ["did_ndvi_m0_6"]).reset_index(drop=True)
Xm, y, grp_ev, grp_blk = X[FEATS].to_numpy(float), X.did_ndvi_m0_6.to_numpy(float), X.event_name.to_numpy(), X.block.to_numpy()
def gbm(): return GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8, min_samples_leaf=20, random_state=SEED)
def ridge(): return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 25)))
res, imp_rows, fold_rows = [], [], []
for cvname, splitter, groups in (("leave-one-event-out", LeaveOneGroupOut(), grp_ev), ("spatial-block 5-fold (within-event structure retained)", GroupKFold(5), grp_blk)):
    for mname, mk in (("ridge (inner CV alpha)", ridge), ("gradient boosting (fixed a priori)", gbm)):
        pred = np.full(len(y), np.nan); imps = []
        for tr, te in splitter.split(Xm, y, groups):
            mdl = mk().fit(Xm[tr], y[tr]); pred[te] = mdl.predict(Xm[te])
            if cvname.startswith("leave"):
                fold_rows.append({"cv": cvname, "model": mname, "held_out_event": groups[te][0], "n_test": len(te),
                                  "r2_fold": 1 - np.sum((y[te] - pred[te]) ** 2) / np.sum((y[te] - y[te].mean()) ** 2)})
                pi = permutation_importance(mdl, Xm[te], y[te], n_repeats=10, random_state=SEED); imps.append(pi.importances_mean)
        r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
        res.append({"cv": cvname, "model": mname, "n_cells": len(y), "n_events": len(set(grp_ev)), "r2_out_of_sample": r2,
                    "mae": float(np.mean(np.abs(y - pred))), "note": "exploratory association model; not causal evidence"})
        if imps:
            imps = np.array(imps)
            for j, f in enumerate(FEATS):
                imp_rows.append({"model": mname, "feature": f, "perm_importance_mean_heldout_events": imps[:, j].mean(), "sd_across_folds": imps[:, j].std()})
XR = pd.DataFrame(res); XI = pd.DataFrame(imp_rows); XF = pd.DataFrame(fold_rows)
shp_rows = []
try:
    import shap
    mdl = gbm().fit(Xm, y); ex = shap.TreeExplainer(mdl); sv = ex.shap_values(Xm)
    for j, f in enumerate(FEATS):
        shp_rows.append({"feature": f, "mean_abs_shap": float(np.abs(sv[:, j]).mean())})
    order = np.argsort(np.abs(sv).mean(0))
    fig, ax = plt.subplots(figsize=(5.6, 3.8)); rng_ = np.random.default_rng(0)
    for i, j in enumerate(order):
        xv = Xm[:, j]; q = (xv - np.nanpercentile(xv, 5)) / (np.nanpercentile(xv, 95) - np.nanpercentile(xv, 5) + 1e-12)
        ax.scatter(sv[:, j], i + rng_.uniform(-0.28, 0.28, len(xv)), c=np.clip(q, 0, 1), cmap="coolwarm", s=2, lw=0, rasterized=True)
    ax.set_yticks(range(len(order))); ax.set_yticklabels(np.array(FEATS)[order], fontsize=6.5); ax.axvline(0, color="black", lw=0.5)
    ax.set_xlabel("SHAP value (contribution to cell \u0394NDVI, months 0\u20136); exploratory")
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap="coolwarm"), ax=ax, fraction=0.04); cb.set_label("feature value (low to high)")
    save(fig, "figS5_exploratory_shap_cell_model")
except Exception as e:
    print("SHAP skipped", e)
XS = pd.DataFrame(shp_rows)
fig, ax = plt.subplots(figsize=(5.4, 3.4))
dd = XI[XI.model.str.startswith("gradient")].sort_values("perm_importance_mean_heldout_events")
ax.barh(dd.feature, dd.perm_importance_mean_heldout_events, xerr=dd.sd_across_folds, color="#7f7f7f", error_kw=dict(lw=0.6))
ax.axvline(0, color="black", lw=0.5); ax.set_xlabel("Permutation importance on held-out events (\u0394R\u00b2); exploratory")
save(fig, "figS4_exploratory_permutation_importance")
XR.to_csv(T + "exploratory_xai_model_performance.csv", index=False); XI.to_csv(T + "exploratory_xai_permutation_importance.csv", index=False)
XF.to_csv(T + "exploratory_xai_leave_one_event_out_folds.csv", index=False); XS.to_csv(T + "exploratory_xai_shap_summary.csv", index=False)
X[["event_name", "cell_uid", "exposure_class"] + FEATS + ["did_ndvi_m0_6"]].to_csv(FDT + "figS4_S5_xai_model_data.csv", index=False)
print("xai", XR[["cv", "model", "r2_out_of_sample"]].round(3).to_dict("records"), f"{time.time()-t0:.0f}s", flush=True)

# ============================================================== 4. dataset audit (Table 2), recomputed from the released data
import hashlib as _hl
from p4.config import IBTRACS_NI, IBTRACS_SHA256, STORM_FLAGS
Q = pd.read_csv(T + "modis_qa_valid_observation_summary.csv")
CP = pd.read_csv(T + "final_event_cell_panel.csv")
long = pd.read_csv(T + "final_event_cell_month_panel.csv.gz", usecols=["event_id"])
obs = pd.read_pickle(W + "obs.pkl")
nobs_raw, nobs_an = len(obs), int(Q.n_obs.sum())
dup = int(obs.duplicated(["pid", "sensor", "cday"]).sum())
an = CP[CP.in_analysis]
ibsha = _hl.sha256(open(IBTRACS_NI, "rb").read()).hexdigest() if (STORM_FLAGS == "ibtracs" and os.path.exists(IBTRACS_NI)) else "not used"
fu = CP.groupby("event_name").followup_last_month.max()
A = []
def add(item, check, value, source="recomputed"):
    A.append({"item": item, "check": check, "value": value, "source": source})
add("Source files", "IBTrACS v04r01 NI file SHA-256 matches the published analysis", str(ibsha == IBTRACS_SHA256) if ibsha != "not used" else "released storm flags used")
add("Cyclone events", "analysed events", f"{CP.event_id.nunique()} ({', '.join(ORDER)})")
add("Raw MODIS rows", "point-composite rows", f"{nobs_raw:,}")
add("Analysed MODIS rows", "rows in the relative-month window -36 to +24", f"{nobs_an:,} ({nobs_raw - nobs_an} rows outside the window)")
add("Event-cell panel", "records; in matched analysis", f"{len(CP):,}; {len(an):,}")
add("Event-cell-month panel", "rows with at least one valid observation", f"{len(long):,}")
add("Exposure classes", "R64; R50; R34; controls (matched analysis)", "; ".join(f"{c} {int((an.exposure_class == c).sum()):,}" for c in ("R64", "R50", "R34", "control")))
add("Exclusions", "no baseline; contaminated controls; outside caliper",
    f"{int((~CP.baseline_ok).sum()):,}; {int((CP.baseline_ok & (CP.exposure_class == 'control') & ~CP.in_analysis).sum()):,}; {int((CP.baseline_ok & (CP.exposure_class != 'control') & ~CP.in_analysis).sum()):,}")
add("QA valid share", "relaxed; strict; fill", f"{Q.n_valid_relaxed.sum() / nobs_an:.1%}; {Q.n_valid_strict.sum() / nobs_an:.1%}; {Q.n_fill.sum() / nobs_an:.1%}")
add("Duplicate check", "duplicated event, cell, sensor and composite records", f"{dup}")
add("Censoring check", "last follow-up month by event", "; ".join(f"{e} {int(fu[e])}" for e in ORDER))
add("Earth Engine jobs", "original events; added events", "436 of 436 jobs; 6 of 6 batch tasks; 0 failed", "provenance (extraction log)")
add("Archive integrity", "original extraction; added events", "12 of 12 parts, CRC passed; extension archive CRC passed, 9,540 of 9,540 cells", "provenance (extraction log)")
pd.DataFrame(A).to_csv(T + "dataset_audit_summary.csv", index=False)
# valid relaxed share in the seven calendar months from landfall (Table 4, last column)
qcal = pd.read_csv(FDT + "figS_qa_valid_share_by_calendar_month.csv"); _m = C["meta"]; cov = []
for en in ORDER:
    lf = pd.to_datetime(_m[_m.event_name == en].landfall_utc.iloc[0]); post_m = [((lf.month - 1 + k) % 12) + 1 for k in range(0, 7)]
    cov.append({"event_name": en, "landfall_month": lf.strftime("%b"), "valid_share_relaxed_months0_6_calendar": round(qcal[(qcal.event_name == en) & qcal.cal_m.isin(post_m)].vrelax.mean(), 2)})
pd.DataFrame(cov).to_csv(SUP + "valid_share_months0_6_by_event.csv", index=False)
print(pd.DataFrame(A).to_string(index=False))
print("stage 05 done", f"{time.time()-t0:.0f}s", flush=True)
