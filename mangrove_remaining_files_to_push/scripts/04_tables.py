"""Stage 04. Evidence classes, event-level summary, pre-trend and placebo table, balance, pooled estimates,
QA summary, final event-cell and event-cell-month panels, matched exposed-control pairs, figure data.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from p4.config import W, O, T, FIG, FDT, SUP, data, STORM_FLAGS
import pandas as pd, numpy as np, json, os, time
t0 = time.time()
TAB = T
R = pd.read_pickle(W + "stageB.pkl"); A = pd.read_pickle(W + "anom.pkl")
meta, match, bal, ES, SUM, PL, POOL, storms, fu = (R[k] for k in ("meta", "match", "bal", "ES", "SUM", "PL", "POOL", "storms", "fu"))
DUMMY = R["DUMMY"]; BC = R["bc_primary"]
MATCHED = set(np.concatenate([match[e]["exp_pid"] for e in match]).tolist())
MONTHS = np.arange(-36, 25); IDX = lambda m: m + 36
SEED = 20261004; rng = np.random.default_rng(SEED + 7)
geo = pd.read_csv(data("inputs/mangrove_exposure_area_by_event.csv"))
ORDER = ["Sidr", "Nargis", "Aila", "Fani", "Bulbul", "Amphan", "Yaas", "Sitrang", "Mocha", "Remal", "Dana"]
YEAR = {"Sidr": 2007, "Nargis": 2008, "Aila": 2009, "Fani": 2019, "Bulbul": 2019, "Amphan": 2020, "Yaas": 2021, "Sitrang": 2022, "Mocha": 2023, "Remal": 2024, "Dana": 2024}

# ---------------------------------------------------------------- classification of claims (pre-specified rule)
P = SUM[SUM.spec.isin(["primary_matched", "evi_matched"])].copy()
alt = SUM.pivot_table(index=["event_id", "group", "index"], columns="spec", values="mean_m0_6").reset_index()
P = P.merge(alt[["event_id", "group", "index"] + [c for c in ("strictqa_matched", "near_controls_matched", "seasonal_diff_matched",
                                                              "unmatched_allcontrols", "no_censoring_matched") if c in alt.columns]],
            on=["event_id", "group", "index"], how="left")
# EVI robustness columns come from the EVI seasonal diff spec only; strict/near are NDVI specs
pl = PL[["event_id", "group", "index", "placebo_space_p_mean_m0_6_one_sided", "placebo_space_p_trough_one_sided",
         "placebo_time_mean_m0_6", "placebo_time_mean_m0_6_ci_lo", "placebo_time_mean_m0_6_ci_hi", "placebo_time_trough"]]
P = P.merge(pl, on=["event_id", "group", "index"], how="left")
def classify(r):
    c1 = r.mean_m0_6_ci_hi < 0
    c2 = (r.its_post0_6_p < 0.05) and (r.its_post0_6 < 0)
    c3 = r.placebo_space_p_mean_m0_6_one_sided < 0.05
    nd = SUM[(SUM.event_id == r.event_id) & (SUM.group == r.group) & (SUM["index"] == "NDVI")].set_index("spec")["mean_m0_6"]
    c4 = all((nd.get(s, np.nan) < 0) or pd.isna(nd.get(s, np.nan)) for s in ("strictqa_matched", "near_controls_matched", "seasonal_diff_matched"))
    c5 = r.pretrend_slope_hac_p >= 0.05
    if not c5:
        return "pretrend violation (not interpretable)", c1, c2, c3, c4, c5
    if c1 and c2 and c3 and c4:
        return "robust decline", c1, c2, c3, c4, c5
    if c1 and c2:
        return "suggestive decline", c1, c2, c3, c4, c5
    if r.mean_m0_6_ci_lo > 0 and r.its_post0_6_p < 0.05:
        return "positive anomaly (not damage)", c1, c2, c3, c4, c5
    return "no detectable decline", c1, c2, c3, c4, c5
cl = P.apply(classify, axis=1, result_type="expand")
cl.columns = ["evidence_class", "crit_bootstrap_ci_below0", "crit_its_hac_p_lt_0p05", "crit_placebo_space_p_lt_0p05",
              "crit_same_sign_strict_near_seasonal", "crit_no_pretrend_hac"]
P = pd.concat([P, cl], axis=1)
lvl = meta[meta.pid.isin(MATCHED)].groupby(["event_id", "exposure_class"])["f_ndvi_level"].mean().rename("baseline_ndvi_level").reset_index()
lvl_e = meta[meta.pid.isin(MATCHED)].groupby(["event_id", "exposure_class"])["f_evi_level"].mean().rename("baseline_evi_level").reset_index()
gmap = {"R64": ["R64"], "R50": ["R50"], "R34": ["R34"], "R50plus": ["R50", "R64"], "all_exposed": ["R34", "R50", "R64"]}
def base_level(r):
    src = lvl if r["index"] == "NDVI" else lvl_e; col = "baseline_ndvi_level" if r["index"] == "NDVI" else "baseline_evi_level"
    return src[(src.event_id == r.event_id) & src.exposure_class.isin(gmap[r.group])][col].mean()
P["baseline_level"] = P.apply(base_level, axis=1)
P["mean_m0_6_pct_of_baseline"] = 100 * P["mean_m0_6"] / P["baseline_level"]
P["trough_pct_of_baseline"] = 100 * P["trough"] / P["baseline_level"]
ga = geo.set_index("event_id")
def class_area(r):
    g = ga.loc[r.event_id]
    return {"R64": g.A64_2020, "R50": g.A50_2020 - g.A64_2020, "R34": g.A34_2020 - g.A50_2020, "R50plus": g.A50_2020,
            "all_exposed": g.A34_2020}[r.group]
P["exposed_mangrove_km2_gmw2020"] = P.apply(class_area, axis=1)
P["followup_status"] = np.where(P.followup_last_month < 24, "censored: data end before 24 months (2024 event)", "complete 24 months")
contam = meta[meta.exposure_class != "control"].groupby(["event_id", "exposure_class"]).agg(
    share_prior_r50_in_baseline=("prior_r50_month", lambda s: float(s.notna().mean())),
    share_censored_by_later_r50=("censor_m50", lambda s: float((s < 99).mean())),
    share_exposed_by_other_eligible_event=("exposed_by_other_eligible_event", "mean")).reset_index()
def contam_for(r, col):
    d = contam[(contam.event_id == r.event_id) & contam.exposure_class.isin(gmap[r.group])]
    return d[col].mean()
for c in ("share_prior_r50_in_baseline", "share_censored_by_later_r50", "share_exposed_by_other_eligible_event"):
    P[c] = P.apply(lambda r: contam_for(r, c), axis=1)
P["recovery_status"] = np.where(~P.evidence_class.isin(["robust decline", "suggestive decline"]), "not applicable (no decline claimed)",
                                np.where(P.recovery_month.notna(), "recovered to pre-event noise band",
                                         np.where(P.recovery_censored, "censored before recovery", "not applicable")))
claimed = P.evidence_class.isin(["robust decline", "suggestive decline"])
for c in ("recovery_month", "recovery_month_boot_median", "recovery_pct_m3", "recovery_pct_m6", "recovery_pct_m12"):
    P.loc[~claimed, c] = np.nan
P["event_order"] = P.event_name.map({e: i for i, e in enumerate(ORDER)})
P = P.sort_values(["index", "event_order", "group"])
keep = ["event_id", "event_name", "index", "group", "evidence_class", "n_exposed_cells", "n_control_cells", "exposed_mangrove_km2_gmw2020",
        "baseline_level", "immediate_m0_1", "immediate_m0_1_ci_lo", "immediate_m0_1_ci_hi", "mean_m0_6", "mean_m0_6_ci_lo", "mean_m0_6_ci_hi",
        "mean_m0_6_pct_of_baseline", "its_post0_6", "its_post0_6_se_hac", "its_post0_6_p", "trough", "trough_ci_lo", "trough_ci_hi",
        "trough_month", "trough_pct_of_baseline", "eff_m3", "eff_m3_ci_lo", "eff_m3_ci_hi", "eff_m6", "eff_m6_ci_lo", "eff_m6_ci_hi",
        "eff_m12", "eff_m12_ci_lo", "eff_m12_ci_hi", "recovery_pct_m3", "recovery_pct_m6", "recovery_pct_m12", "noise_band",
        "recovery_month", "recovery_month_boot_median", "recovery_boot_share_censored_or_none", "recovery_status", "recovery_window_end",
        "followup_last_month", "followup_status", "storm_censor_month_median", "its_post7_12", "its_post7_12_p", "its_post13_24", "its_post13_24_p",
        "strictqa_matched", "near_controls_matched", "seasonal_diff_matched", "unmatched_allcontrols", "no_censoring_matched",
        "crit_bootstrap_ci_below0", "crit_its_hac_p_lt_0p05", "crit_placebo_space_p_lt_0p05", "crit_same_sign_strict_near_seasonal",
        "crit_no_pretrend_hac", "share_prior_r50_in_baseline", "share_censored_by_later_r50", "share_exposed_by_other_eligible_event",
        "n_months_insufficient_coverage_post"]
EV = P[keep].rename(columns={"strictqa_matched": "sens_strictqa_mean_m0_6", "near_controls_matched": "sens_near_controls_mean_m0_6",
                             "seasonal_diff_matched": "sens_seasonal_diff_mean_m0_6", "unmatched_allcontrols": "sens_unmatched_mean_m0_6",
                             "no_censoring_matched": "sens_no_censoring_mean_m0_6"})
EV.loc[EV["index"] == "EVI", ["sens_strictqa_mean_m0_6", "sens_near_controls_mean_m0_6", "sens_unmatched_mean_m0_6", "sens_no_censoring_mean_m0_6"]] = np.nan
EV.to_csv(TAB + "event_level_canopy_impact_summary.csv", index=False)

# ---------------------------------------------------------------- pretrend / placebo table
PT = SUM[SUM.spec.isin(["primary_matched", "evi_matched"])][["event_id", "event_name", "index", "group", "pre_mean_m24_1", "pre_mean_hac",
     "pretrend_slope_hac", "pretrend_slope_hac_p", "pre_equivalence_tost_p_margin0p01", "pretrend_slope_per_month", "pretrend_slope_ci_lo",
     "pretrend_slope_ci_hi", "pretrend_wald_chi2_6bins", "pretrend_wald_p", "pre_months_ci_excluding_0_share"]].merge(
    PL.drop(columns=["event_name"]), on=["event_id", "group", "index"], how="left")
PT["note"] = ("pretrend_wald_p uses a cell-block bootstrap covariance that ignores month-to-month regional shocks and is over-powered; "
              "the HAC slope test and the +/-0.01 equivalence (TOST) test are the primary pre-trend diagnostics")
PT.to_csv(TAB + "pretrend_placebo_test_summary.csv", index=False)
bal.to_csv(TAB + "matched_control_balance_summary.csv", index=False)
ES.to_csv(TAB + "did_or_event_study_results.csv", index=False)
SUM.to_csv(TAB + "robustness_specification_summary.csv", index=False)
POOL.to_csv(TAB + "pooled_random_effects_by_wind_class.csv", index=False)
storms.to_csv(TAB + "window_storm_contamination.csv", index=False)
qa = pd.read_pickle(W + "qa_summary.pkl"); qa.to_csv(TAB + "modis_qa_valid_observation_summary.csv", index=False)
pd.read_pickle(W + "qa_calmonth.pkl").to_csv(FDT + "figS_qa_valid_share_by_calendar_month.csv", index=False)

# ---------------------------------------------------------------- final event-cell panels
Mn, Me = A[("ndvi", "vrelax", "main")]["M"], A[("evi", "vrelax", "main")]["M"]
_nanrow = np.full((1, Mn.shape[1]), np.nan)
Mn = np.vstack([Mn, _nanrow]); Me = np.vstack([Me, _nanrow])   # row DUMMY pads exposed cells with < 5 matched controls
cnt = A[("ndvi", "vrelax", "main")]["cnt"]
lev = pd.read_pickle(W + "level_ndvi.pkl")
cell_rows = []
meta["did_ndvi_m0_6"] = np.nan; meta["did_evi_m0_6"] = np.nan; meta["anom_ndvi_m0_6"] = np.nan
for eid, g in meta.groupby("event_id"):
    mt = match[eid]
    cm_n = np.nanmean(Mn[mt["ctrl_pid"]], axis=1) + BC[(eid, "ndvi")]; cm_e = np.nanmean(Me[mt["ctrl_pid"]], axis=1) + BC[(eid, "evi")]
    sl = slice(IDX(0), IDX(6) + 1)
    with np.errstate(all="ignore"):
        meta.loc[meta.pid.isin(mt["exp_pid"]), "did_ndvi_m0_6"] = pd.Series(np.nanmean(Mn[mt["exp_pid"], sl] - cm_n[:, sl], 1), index=mt["exp_pid"]).reindex(meta.loc[meta.pid.isin(mt["exp_pid"]), "pid"]).values
        meta.loc[meta.pid.isin(mt["exp_pid"]), "did_evi_m0_6"] = pd.Series(np.nanmean(Me[mt["exp_pid"], sl] - cm_e[:, sl], 1), index=mt["exp_pid"]).reindex(meta.loc[meta.pid.isin(mt["exp_pid"]), "pid"]).values
        pool = mt["pool_pid"]; pm = np.nanmean(Mn[pool, sl], 0)
        meta.loc[meta.pid.isin(pool), "did_ndvi_m0_6"] = pd.Series(np.nanmean(Mn[pool, sl] - pm, 1), index=pool).reindex(meta.loc[meta.pid.isin(pool), "pid"]).values
    meta.loc[g.index, "anom_ndvi_m0_6"] = np.nanmean(Mn[g.pid.values, sl], 1)
bins = [-9, -0.10, -0.05, -0.02, 0.02, 0.05, 9]
labs = ["<= -0.10", "-0.10 to -0.05", "-0.05 to -0.02", "-0.02 to 0.02", "0.02 to 0.05", ">= 0.05"]
meta["ndvi_anomaly_class_m0_6"] = pd.cut(meta["did_ndvi_m0_6"], bins, labels=labs)
meta["matched_retained"] = meta.pid.isin(MATCHED)
meta["in_analysis"] = meta.baseline_ok & (((meta.exposure_class != "control") & meta.matched_retained) | ((meta.exposure_class == "control") & (meta.censor_m50 > 24)))
meta["n_valid_months_post0_6"] = (cnt[:, IDX(0):IDX(6) + 1] > 0).sum(1)
cell_cols = ["event_id", "event_name", "cell_uid", "lon", "lat", "cell_lon", "cell_lat", "exposure_class", "in_analysis", "matched_retained", "baseline_ok",
             "npre_valid_relaxed", "f_ndvi_level", "f_ndvi_amp", "f_ndvi_sd", "f_ndvi_pretrend", "f_evi_level", "gmw2020_frac_block220m",
             "gmw2020_frac_window340m", "dist_to_nearest_exposed_km", "dist_band", "prior_max_class_window", "prior_r50_month",
             "censor_m50", "censor_m34", "other_storm_max_class_prior365d", "other_storm_max_class_post365d", "control_contaminated_any365d",
             "exposed_by_other_eligible_event", "followup_last_month", "anom_ndvi_m0_6", "did_ndvi_m0_6", "did_evi_m0_6",
             "ndvi_anomaly_class_m0_6", "n_valid_months_post0_6", "block"]
CP = meta[cell_cols].copy()
CP["censor_m50"] = CP["censor_m50"].replace(99, np.nan); CP["censor_m34"] = CP["censor_m34"].replace(99, np.nan)
CP.to_csv(TAB + "final_event_cell_panel.csv", index=False)
P_ = len(meta)
long = pd.DataFrame({"event_id": np.repeat(meta.event_id.values, 61), "cell_uid": np.repeat(meta.cell_uid.values, 61),
                     "exposure_class": np.repeat(meta.exposure_class.values, 61), "rel_month": np.tile(MONTHS, P_),
                     "ndvi_anomaly": Mn[:P_].ravel().round(5), "evi_anomaly": Me[:P_].ravel().round(5), "ndvi_level": lev.ravel().round(5),
                     "n_valid_obs": cnt.ravel()})
cm50 = np.repeat(meta.censor_m50.values, 61)
long["after_later_r50_storm"] = long.rel_month.values >= cm50
long["after_data_end"] = long.rel_month.values > np.repeat(meta.followup_last_month.values, 61)
long = long[long.n_valid_obs > 0]
long.to_csv(TAB + "final_event_cell_month_panel.csv.gz", index=False, compression="gzip")
print("panels", len(CP), len(long), f"{time.time()-t0:.0f}s", flush=True)

# ---------------------------------------------------------------- figure data: event-study group lines (exposed vs matched controls)
def blocks_w(blocks, Bn):
    ub, inv = np.unique(blocks, return_inverse=True)
    return rng.multinomial(len(ub), np.full(len(ub), 1 / len(ub)), size=Bn)[:, inv].astype(float)
def wmean(Wt, Dm):
    f = np.isfinite(Dm); return (Wt @ np.where(f, Dm, 0.0)) / np.maximum(Wt @ f.astype(float), 1e-12)
lines = []
head = {"Sidr": "R64", "Nargis": "R64", "Aila": "R64", "Fani": "R64", "Bulbul": "R64", "Amphan": "R64", "Yaas": "R50", "Sitrang": "R34", "Mocha": "R64", "Remal": "R50", "Dana": "R50"}
mi = meta.set_index("pid")
for eid, g in meta.groupby("event_id"):
    en = g.event_name.iloc[0]; mt = match[eid]
    cls = mi.loc[mt["exp_pid"], "exposure_class"].to_numpy()
    for ixn, M in (("NDVI", Mn), ("EVI", Me)):
        for grp in ("R64", "R50", "R34"):
            sel = cls == grp
            if sel.sum() < 10:
                continue
            Ex = M[mt["exp_pid"][sel]].astype(float).copy()
            cm = mi.loc[mt["exp_pid"][sel], "censor_m50"].to_numpy()
            for i, c in enumerate(cm):
                if c < 99:
                    Ex[i, IDX(c):] = np.nan
            Cx = np.nanmean(M[mt["ctrl_pid"][sel]].astype(float), axis=1) + BC[(eid, ixn.lower())][sel]   # bias-corrected control mean
            need = max(30, 0.2 * sel.sum())
            low = (np.isfinite(Ex - Cx).sum(0) < need)
            Wt = blocks_w(mi.loc[mt["exp_pid"][sel], "block"].to_numpy(), 500)
            be, bc = wmean(Wt, Ex), wmean(Wt, Cx)
            for k, m in enumerate(MONTHS):
                lines.append({"event_id": eid, "event_name": en, "index": ixn, "group": grp, "headline_group": grp == head[en],
                              "rel_month": int(m), "exposed_mean": np.nan if low[k] else np.nanmean(Ex[:, k]),
                              "exposed_ci_lo": np.nan if low[k] else np.nanpercentile(be[:, k], 2.5),
                              "exposed_ci_hi": np.nan if low[k] else np.nanpercentile(be[:, k], 97.5),
                              "control_mean": np.nan if low[k] else np.nanmean(Cx[:, k]),
                              "control_ci_lo": np.nan if low[k] else np.nanpercentile(bc[:, k], 2.5),
                              "control_ci_hi": np.nan if low[k] else np.nanpercentile(bc[:, k], 97.5),
                              "insufficient_coverage": bool(low[k]), "beyond_followup": int(m) > fu[eid]})
LINES = pd.DataFrame(lines); LINES.to_csv(FDT + "fig02_event_study_exposed_vs_control_lines.csv", index=False)
pd.to_pickle({"EV": EV, "PT": PT, "meta": meta, "LINES": LINES, "head": head, "ORDER": ORDER, "YEAR": YEAR}, W + "stageC.pkl")
print("DONE", f"{time.time()-t0:.0f}s", flush=True)
