"""Stage 08. Recovery sensitivity, multiplicity (Holm, Benjamini-Hochberg) and equivalence tests, block-size and
missing-data sensitivity, valid cells per event-study month, balance comparison and entropy-balancing feasibility.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from p4.config import W, O, T, FIG, FDT, SUP, data, STORM_FLAGS
import pandas as pd, numpy as np, os, math, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none", "pdf.fonttype": 42})
RT = SUP
R = pd.read_pickle(W + "stageB.pkl"); SUM, PL, ES, bal, EX, POOL, boot, meta = (R[k] for k in ("SUM", "PL", "ES", "bal", "EXTRA", "POOL", "boot", "meta"))
ORDER = ["Sidr", "Nargis", "Aila", "Fani", "Bulbul", "Amphan", "Yaas", "Sitrang", "Mocha", "Remal", "Dana"]
YEAR = {"Sidr": 2007, "Nargis": 2008, "Aila": 2009, "Fani": 2019, "Bulbul": 2019, "Amphan": 2020, "Yaas": 2021, "Sitrang": 2022, "Mocha": 2023, "Remal": 2024, "Dana": 2024}
MONTHS = np.arange(-36, 25); IDX = lambda m: m + 36
EID = {e.split("_")[1].title(): e for e in meta.event_id.unique()}
# ------------------------------------------------------------- 1. recovery sensitivity
def smooth_w(b, w):
    if w == 1: return b.copy()
    f = np.isfinite(b); k = np.ones(w)
    num = np.convolve(np.where(f, b, 0.0), k, mode="same"); den = np.convolve(f.astype(float), k, mode="same")
    return np.where(den >= math.ceil(w / 2), num / np.maximum(den, 1), np.nan)
def recovery(b, fu_last, cens_m, w, kband, consec):
    s = smooth_w(b, w); seg = s[IDX(0):IDX(12) + 1]
    if np.all(np.isnan(seg)): return np.nan, np.nan, "no data"
    tm = int(np.nanargmin(seg)); tr = seg[tm]; band = kband * np.nanstd(s[:IDX(-1) + 1]); end = min(fu_last, cens_m)
    if not tr < -band: return np.nan, tm, "no decline beyond band"
    for m in range(tm + 1, end):
        ok = all((m + j > end - 1) or (np.isfinite(s[IDX(m + j)]) and s[IDX(m + j)] >= -band) for j in range(consec))
        if np.isfinite(s[IDX(m)]) and ok: return m, tm, "recovered"
    return np.nan, tm, "censored"
rec_rows = []
for ev in ("Sidr", "Nargis"):
    for ix in ("ndvi", "evi"):
        bs = boot[(EID[ev], "R64", ix)]
        for w in (1, 3, 5):
            for kb in (1.5, 2.0, 2.5):
                for c in (1, 2, 3):
                    m, tm, st = recovery(bs["b"], bs["fu_last"], bs["cens_m"], w, kb, c)
                    rec_rows.append({"event_name": ev, "index": ix.upper(), "smoothing_window_months": w, "band_sd_multiplier": kb, "consecutive_months": c,
                                     "trough_month": tm, "recovery_month": m, "status": st, "primary": (w == 3 and kb == 2.0 and c == 2)})
REC = pd.DataFrame(rec_rows); REC.to_csv(RT + "recovery_sensitivity.csv", index=False)
# ------------------------------------------------------------- 2. multiple testing and equivalence
def holm(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); adj = np.empty(n); run = 0
    for r, i in enumerate(o):
        run = max(run, (n - r) * p[i]); adj[i] = min(1, run)
    return adj
def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)[::-1]; adj = np.empty(n); run = 1
    for r, i in enumerate(o):
        run = min(run, p[i] * n / (n - r)); adj[i] = run
    return adj
P = SUM[(SUM.spec == "primary_matched") & SUM.group.isin(["R64", "R50", "R34"])].copy()
pl = PL[PL["index"] == "NDVI"].set_index(["event_id", "group"])
P["space_p"] = [pl.loc[(e, g), "placebo_space_p_mean_m0_6_one_sided"] for e, g in zip(P.event_id, P.group)]
P["space_n_perm"] = [pl.loc[(e, g), "placebo_space_n_perm"] for e, g in zip(P.event_id, P.group)]
for col, nm in (("its_post0_6_p", "hac"), ("boot_p_two_sided_m0_6", "boot"), ("space_p", "space")):
    P[f"{nm}_p_holm"] = holm(P[col].values); P[f"{nm}_p_bh"] = bh(P[col].values)
P["o"] = P.event_name.map({e: i for i, e in enumerate(ORDER)}); P["gg"] = P.group.map({"R64": 0, "R50": 1, "R34": 2}); P = P.sort_values(["o", "gg"])
MT = P[["event_name", "group", "mean_m0_6", "its_post0_6_p", "hac_p_holm", "hac_p_bh", "boot_p_two_sided_m0_6", "boot_p_holm", "boot_p_bh",
        "space_p", "space_p_holm", "space_p_bh", "space_n_perm", "pretrend_slope_hac_p", "pre_equivalence_tost_p_margin0p01"]].copy()
MT["survives_holm_all_three"] = (MT.hac_p_holm < 0.05) & (MT.boot_p_holm < 0.05) & (MT.space_p_holm < 0.05) & (MT.mean_m0_6 < 0)
MT["pretrend_equivalent_0p01"] = MT.pre_equivalence_tost_p_margin0p01 < 0.05
MT.to_csv(RT + "multiple_testing_and_equivalence.csv", index=False)
# ------------------------------------------------------------- 3. block size and 4. missing data
EX["o"] = EX.event_name.map({e: i for i, e in enumerate(ORDER)}); EX["gg"] = EX.group.map({"R64": 0, "R50": 1, "R34": 2})
EX = EX.sort_values(["o", "gg"]).drop(columns=["o", "gg"])
EX[["event_name", "group", "n_cells", "mean_m0_6"] + [c for c in EX.columns if "block" in c]].to_csv(RT + "block_size_sensitivity.csv", index=False)
mcols = ["event_name", "group", "n_cells", "mean_m0_6", "share_cell_months_observed_m0_6", "n_complete_case", "cc_mean_m0_6", "cc_ci_lo", "cc_ci_hi",
         "ipw_mean_m0_6", "ipw_ci_lo", "ipw_ci_hi", "null_imp_mean_m0_6", "null_imp_ci_lo", "null_imp_ci_hi", "n_cells_mostly_observed",
         "obs_vs_missing_max_abs_smd", "obs_vs_missing_max_cov", "obs_vs_missing_canopy_max_abs_smd"]
EX[[c for c in mcols if c in EX.columns]].to_csv(RT + "missing_data_sensitivity.csv", index=False)
# ------------------------------------------------------------- 5. valid cells per event-study month
head = {"Sidr": "R64", "Nargis": "R64", "Aila": "R64", "Fani": "R64", "Bulbul": "R64", "Amphan": "R64", "Yaas": "R50", "Sitrang": "R34", "Mocha": "R64", "Remal": "R50", "Dana": "R50"}
VC = ES[(ES.spec == "primary_matched") & ES.group.isin(["R64", "R50", "R34"])][["event_name", "group", "rel_month", "n_exposed_cells_with_data",
                                                                                    "n_control_cells_with_data", "month_insufficient_coverage"]]
VC.to_csv(RT + "valid_cells_by_event_study_month.csv", index=False)
fig, axs = plt.subplots(3, 4, figsize=(7.2, 5.3), sharey=False); axs.ravel()[11].set_visible(False)
for ax, en in zip(axs.ravel(), ORDER):
    d = VC[(VC.event_name == en) & (VC.group == head[en]) & VC.rel_month.between(-12, 24)]
    ax.bar(d.rel_month, d.n_exposed_cells_with_data, color="#b2182b", width=0.8, label="exposed cells")
    ax.plot(d.rel_month, d.n_control_cells_with_data, color="#2166ac", lw=1.0, label="matched controls")
    low = d[d.month_insufficient_coverage]
    ax.scatter(low.rel_month, np.zeros(len(low)), marker="x", color="black", s=10, lw=0.7, label="below coverage rule")
    ax.axvline(0, color="black", lw=0.6); ax.set_title(f"{en} {YEAR[en]} ({head[en]})", fontsize=7.5)
    ax.set_xlim(-12.5, 24.5); ax.tick_params(labelsize=6.5)
for ax in axs[2]: ax.set_xlabel("Months from landfall", fontsize=7)
for ax in axs[:, 0]: ax.set_ylabel("Cells with valid data", fontsize=7)
axs[0, 0].legend(fontsize=5.8, loc="lower left", frameon=False)
fig.tight_layout()
for ext in ("png", "pdf", "svg"):
    fig.savefig(FIG + f"figR1_valid_cells_by_month.{ext}", dpi=600, bbox_inches="tight", facecolor="white")
plt.close(fig)
# ------------------------------------------------------------- 6. balance comparison and entropy-balancing feasibility
CANP = ["ndvi_level", "ndvi_amp", "ndvi_sd", "ndvi_pretrend", "evi_level", "frac340", "prior_max_class_window"]
FEATS = ["f_ndvi_level", "f_ndvi_amp", "f_ndvi_sd", "f_ndvi_pretrend", "f_evi_level", "f_frac340", "cell_lat", "cell_lon", "prior_max_class_window"]
def eb(exp, pool):
    Z0 = pool[FEATS].to_numpy(float); mu, sd = Z0.mean(0), Z0.std(0).clip(1e-9); Z = (Z0 - mu) / sd; tz = (exp[FEATS].to_numpy(float).mean(0) - mu) / sd
    from scipy.special import logsumexp, softmax
    f = lambda l: logsumexp(Z @ l) - l @ tz
    g = lambda l: softmax(Z @ l) @ Z - tz
    r = minimize(f, np.zeros(Z.shape[1]), jac=g, method="BFGS", options={"maxiter": 3000}); w = softmax(Z @ r.x)
    return 1 / np.sum(w ** 2), float(np.abs(w @ Z - tz).max())
md = R["match_dist"]; rows = []
for en in ORDER:
    a = bal[(bal.control_set == "all_pool") & (bal.event_name == en)]; v = bal[(bal.control_set == "v1_original") & (bal.event_name == en)]
    g = meta[meta.event_name == en]; exp = g[(g.exposure_class != "control") & g.baseline_ok]; pool = g[(g.exposure_class == "control") & g.baseline_ok & (g.censor_m50 > 24)]
    ess, resid = eb(exp, pool)
    rows.append({"event_name": en, "n_exposed_eligible": int(a.n_exposed_eligible.iloc[0]), "n_exposed_retained": int(a.n_exposed.iloc[0]),
                 "share_retained": float(a.share_exposed_retained.iloc[0]), "mean_controls_per_exposed": float(a.mean_controls_per_exposed.iloc[0]),
                 "v1_max_abs_smd_canopy_prior": v[v.covariate.isin(CANP)].smd_after.abs().max(), "r1_max_abs_smd_canopy_prior": a[a.covariate.isin(CANP)].smd_after.abs().max(),
                 "v1_max_abs_smd_all": v.smd_after.abs().max(), "r1_max_abs_smd_all": a.smd_after.abs().max(),
                 "v1_share_below_0p25": (v.smd_after.abs() < 0.25).mean(), "r1_share_below_0p25": (a.smd_after.abs() < 0.25).mean(),
                 "r1_smd_lat": float(a[a.covariate == "cell_lat"].smd_after.iloc[0]), "r1_smd_lon": float(a[a.covariate == "cell_lon"].smd_after.iloc[0]),
                 "r1_median_exposed_control_km": float(md[(md.event_name == en) & (md.control_set == "all_pool")].median_exposed_control_km.iloc[0]),
                 "eb_all_covariates_ess": ess, "eb_max_residual_sd": resid, "control_pool": len(pool)})
BC = pd.DataFrame(rows); BC.to_csv(RT + "balance_comparison_original_vs_revised.csv", index=False)
# ------------------------------------------------------------- summary
print("recovery grid", REC[REC.primary][["event_name", "index", "recovery_month"]].to_dict("records"))
print("recovery range NDVI", REC[REC["index"] == "NDVI"].groupby("event_name").recovery_month.agg(["min", "max", lambda s: s.isna().sum()]).to_dict())
print(MT[["event_name", "group", "hac_p_holm", "boot_p_holm", "space_p_holm", "survives_holm_all_three", "pre_equivalence_tost_p_margin0p01"]].round(4).to_string(index=False))
print(BC.round(2).to_string(index=False))
print(pd.read_csv(RT + "missing_data_sensitivity.csv").query("group=='R64'").round(3).to_string(index=False))
print(pd.read_csv(RT + "block_size_sensitivity.csv").query("group=='R64'").round(3).to_string(index=False))
print(POOL[(POOL.spec == "primary_matched")].round(3).to_string(index=False))
