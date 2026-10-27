"""Stage 07. Event maps, event-study figures, recovery timeline, specification robustness, balance, QA coverage, placebo.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from p4.config import W, O, T, FIG, FDT, SUP, data, STORM_FLAGS
import pandas as pd, numpy as np, os, time, math, importlib.util, warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.basemap import Basemap
t0 = time.time()
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                     "savefig.bbox": "tight", "savefig.pad_inches": 0.04, "svg.fonttype": "none", "pdf.fonttype": 42})
C = pd.read_pickle(W + "stageC.pkl"); R = pd.read_pickle(W + "stageB.pkl")
EV, meta, LINES, head, ORDER, YEAR = C["EV"], C["meta"], C["LINES"], C["head"], C["ORDER"], C["YEAR"]
ES, SUM, PL, storms, fu, bal = R["ES"], R["SUM"], R["PL"], R["storms"], R["fu"], R["bal"]
EID = {e: meta[meta.event_name == e].event_id.iloc[0] for e in ORDER}
made = []

def save(fig, name):
    for ext, kw in (("png", {"dpi": 600}), ("pdf", {"dpi": 600}), ("svg", {"dpi": 600})):
        fig.savefig(FIG + f"{name}.{ext}", facecolor="white", **kw)
    plt.close(fig); made.append(name)

def coast(ax, box, res="i"):
    m = Basemap(projection="cyl", llcrnrlat=box[0], urcrnrlat=box[1], llcrnrlon=box[2], urcrnrlon=box[3],
                resolution=res, ax=ax, fix_aspect=True)
    m.drawcoastlines(linewidth=0.35, color="#4d4d4d"); m.drawcountries(linewidth=0.3, color="#8c8c8c")
    m.fillcontinents(color="#f2f0ea", lake_color="white", zorder=0)
    from matplotlib.ticker import MaxNLocator, FuncFormatter
    ax.set_axis_on(); ax.xaxis.set_major_locator(MaxNLocator(5)); ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:.1f}\u00b0E"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:.1f}\u00b0N"))
    ax.tick_params(labelbottom=True, labelleft=True, labelsize=6.5)
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linewidth(0.6)

# ---------------------------------------------------------------- tracks
import p4.ibtracs as kit
ib = kit.load_ni_tracks(); clim = kit.radius_climatology(ib)
bob = pd.read_csv(data("inputs/bay_of_bengal_mangrove_cell_exposure_grid.csv.gz"))
ANOM_LABS = ["<= -0.10", "-0.10 to -0.05", "-0.05 to -0.02", "-0.02 to 0.02", "0.02 to 0.05", ">= 0.05"]
ANOM_COLS = ["#7f3b08", "#d8833a", "#f7d39b", "#e6e6e6", "#9fd4c9", "#2a8c7f"]
SW_COLS = {1: "#fff1b8", 2: "#fdbb84", 3: "#e34a33"}

def event_map(en, fname):
    eid = EID[en]; g = meta[meta.event_id == eid]
    sid = g.sid.iloc[0]; tk = kit.prepare_track(ib[ib.SID == sid], clim)
    col = f"class_{en.upper()}_{YEAR[en]}"
    sw = bob[bob[col] > 0]
    s_ = g[g.in_analysis]
    pad = 0.6
    box = (max(5, s_.cell_lat.min() - pad), min(30, s_.cell_lat.max() + pad), max(78, s_.cell_lon.min() - pad), min(100, s_.cell_lon.max() + pad))
    core = sw[sw[col] >= 2] if (sw[col] >= 2).sum() > 20 else sw
    ibox = (core.lat.min() - 0.25, core.lat.max() + 0.25, core.lon.min() - 0.35, core.lon.max() + 0.35)
    fig = plt.figure(figsize=(7.2, 6.4))
    ax = fig.add_axes([0.07, 0.25, 0.60, 0.70])
    def layers(a, b, msize, swsize):
        coast(a, b)
        for k in (1, 2, 3):
            q = sw[(sw[col] == k) & sw.lat.between(b[0], b[1]) & sw.lon.between(b[2], b[3])]
            a.scatter(q.lon, q.lat, s=swsize, c=SW_COLS[k], marker="s", lw=0, zorder=2, rasterized=True)
        for cls, mk in (("control", "^"), ("exp", "o")):
            q = s_[(s_.exposure_class == "control") if cls == "control" else (s_.exposure_class != "control")]
            q = q[q.cell_lat.between(b[0], b[1]) & q.cell_lon.between(b[2], b[3])]
            cc = q["ndvi_anomaly_class_m0_6"].astype(str).map(dict(zip(ANOM_LABS, ANOM_COLS))).fillna("#ffffff")
            a.scatter(q.lon, q.lat, s=msize, c=cc, marker=mk, edgecolors="#333333", linewidths=0.15, zorder=4, rasterized=True)
        a.plot(tk["lon"], tk["lat"], color="black", lw=1.1, zorder=5)
        land = g.landfall_utc.iloc[0]
        a.set_xlim(b[2], b[3]); a.set_ylim(b[0], b[1])
    layers(ax, box, 9, 1.2)
    # landfall point = track position nearest the landfall time (first DIST2LAND=0 in old tracks ~ coast crossing)
    tr = ib[ib.SID == sid].copy(); tr["dt"] = (tr.t - g.landfall_utc.iloc[0]).abs()
    lf = tr.loc[tr.dt.idxmin()]
    ax.scatter([lf.LON], [lf.LAT], marker="*", s=90, c="white", edgecolors="black", linewidths=0.8, zorder=6)
    # scale bar and north arrow
    lat0 = box[0] + 0.07 * (box[1] - box[0]); lon0 = box[2] + 0.05 * (box[3] - box[2])
    km = 100 if (box[3] - box[2]) > 4 else 50
    dlon = km / (111.32 * math.cos(math.radians(lat0)))
    ax.plot([lon0, lon0 + dlon], [lat0, lat0], color="black", lw=2, zorder=7, solid_capstyle="butt")
    ax.text(lon0 + dlon / 2, lat0 + 0.02 * (box[1] - box[0]), f"{km} km", ha="center", fontsize=6.5, zorder=7)
    ax.annotate("N", xy=(box[2] + 0.06 * (box[3] - box[2]), box[0] + 0.93 * (box[1] - box[0])),
                xytext=(box[2] + 0.06 * (box[3] - box[2]), box[0] + 0.83 * (box[1] - box[0])),
                arrowprops=dict(arrowstyle="-|>", lw=1, color="black"), ha="center", fontsize=8, fontweight="bold", zorder=7)
    # inset
    axi = fig.add_axes([0.70, 0.52, 0.29, 0.43])
    layers(axi, ibox, 14, 3.0)
    axi.tick_params(labelsize=5.5)
    axi.set_title("Zoom: damaging-wind core", fontsize=7)
    ax.add_patch(plt.Rectangle((ibox[2], ibox[0]), ibox[3] - ibox[2], ibox[1] - ibox[0], fill=False, lw=0.8, ec="black", zorder=8))
    r = EV[(EV.event_name == en) & (EV["index"] == "NDVI") & (EV.group == head[en])].iloc[0]
    ax.set_title(f"{en} {YEAR[en]}: sampled mangrove cells, wind swath and NDVI response (months 0\u20136)", fontsize=8.5, loc="left")
    # legends
    h1 = [Patch(fc=SW_COLS[1], label="\u226534 kt swath"), Patch(fc=SW_COLS[2], label="\u226550 kt swath"),
          Patch(fc=SW_COLS[3], label="\u226564 kt swath"), Line2D([], [], color="black", lw=1.1, label="best track"),
          Line2D([], [], marker="*", ls="", mfc="white", mec="black", ms=9, label="landfall"),
          Line2D([], [], marker="o", ls="", mfc="white", mec="#333333", ms=5, label="exposed cell"),
          Line2D([], [], marker="^", ls="", mfc="white", mec="#333333", ms=5, label="control cell")]
    h2 = [Patch(fc=c, ec="#555555", lw=0.3, label=l) for l, c in zip(ANOM_LABS, ANOM_COLS)]
    lax = fig.add_axes([0.70, 0.25, 0.29, 0.24]); lax.axis("off")
    l1 = lax.legend(handles=h1, loc="upper left", fontsize=6.5, title="Exposure", title_fontsize=7)
    lax.add_artist(l1)
    lax.legend(handles=h2, loc="upper right", fontsize=6.3, title="\u0394NDVI vs matched\ncontrols (DiD)", title_fontsize=6.6)
    txt = (f"Headline class {head[en]}: mean \u0394NDVI months 0\u20136 = {r.mean_m0_6:+.3f} "
           f"[{r.mean_m0_6_ci_lo:+.3f}, {r.mean_m0_6_ci_hi:+.3f}]; evidence: {r.evidence_class}.")
    fig.text(0.07, 0.205, txt, fontsize=6.8)
    q = s_.assign(event_name=en)[["event_name", "cell_uid", "lon", "lat", "exposure_class", "did_ndvi_m0_6", "ndvi_anomaly_class_m0_6"]]
    q.to_csv(FDT + f"{fname}_cells.csv", index=False)
    pd.DataFrame({"lon": tk["lon"], "lat": tk["lat"]}).to_csv(FDT + f"{fname}_track.csv", index=False)
    save(fig, fname)

for en in ORDER:
    try:
        event_map(en, f"fig03_map_{en.lower()}_{YEAR[en]}")
    except Exception as ex:
        print("map failed", en, ex, flush=True)
print("maps", f"{time.time()-t0:.0f}s", flush=True)

# ---------------------------------------------------------------- event-study exposed vs control (small multiples)
def es_lines(index, name):
    fig, axes = plt.subplots(3, 4, figsize=(7.2, 5.7), sharex=True, sharey=True); axes.ravel()[11].set_visible(False)
    for ax, en in zip(axes.ravel(), ORDER):
        d = LINES[(LINES.event_name == en) & (LINES["index"] == index) & (LINES.group == head[en]) & LINES.rel_month.between(-24, 24)]
        x = d.rel_month
        ax.fill_between(x, d.control_ci_lo, d.control_ci_hi, color="#2166ac", alpha=0.18, lw=0)
        ax.plot(x, d.control_mean, color="#2166ac", lw=1.0, label="matched controls")
        ax.fill_between(x, d.exposed_ci_lo, d.exposed_ci_hi, color="#b2182b", alpha=0.18, lw=0)
        ax.plot(x, d.exposed_mean, color="#b2182b", lw=1.0, label="exposed")
        ax.axvline(0, color="black", lw=0.7); ax.axhline(0, color="#999999", lw=0.4)
        if fu[EID[en]] < 24:
            ax.axvspan(fu[EID[en]] + 0.5, 24.5, color="#bbbbbb", alpha=0.45, lw=0)
            ax.text(fu[EID[en]] + 0.8, 0.09, "no data", fontsize=5.5, rotation=90, va="top")
        st = storms[(storms.event_id == EID[en]) & (storms.rel_month > 0) & (storms.cells_r50plus >= 0.1 * storms.cells_total)]
        for s in st.itertuples():
            ax.axvline(s.rel_month, color="#666666", lw=0.6, ls=":")
            ax.text(s.rel_month + 0.3, -0.195, s.other_name, fontsize=5, rotation=90, va="bottom", color="#444444")
        n = int(EV[(EV.event_name == en) & (EV["index"] == index) & (EV.group == head[en])].n_exposed_cells.iloc[0])
        ax.set_title(f"{en} {YEAR[en]} ({head[en]}, n={n})", fontsize=7.5)
        ax.set_xlim(-24, 24); ax.set_ylim(-0.20, 0.10)
        ax.set_xticks([-24, -12, 0, 12, 24])
    for ax in list(axes[2, :3]) + [axes[1, 3]]:
        ax.set_xlabel("Months from landfall"); ax.tick_params(labelbottom=True)
    for ax in axes[:, 0]:
        ax.set_ylabel(f"{index} anomaly")
    axes[0, 0].legend(fontsize=6, loc="lower left")
    fig.tight_layout(h_pad=0.6, w_pad=0.4)
    save(fig, name)
es_lines("NDVI", "fig04_event_study_exposed_vs_control_ndvi")
es_lines("EVI", "fig05_event_study_exposed_vs_control_evi")

def did_classes(index, spec, name):
    fig, axes = plt.subplots(3, 4, figsize=(7.2, 5.7), sharex=True, sharey=True); axes.ravel()[11].set_visible(False)
    cols = {"R64": "#7f0000", "R50": "#e34a33", "R34": "#f0a830"}
    rows = []
    for ax, en in zip(axes.ravel(), ORDER):
        for grp in ("R34", "R50", "R64"):
            d = ES[(ES.event_name == en) & (ES.spec == spec) & (ES.group == grp) & ES.rel_month.between(-24, 24)]
            if d.empty:
                continue
            ax.fill_between(d.rel_month, d.ci_lo, d.ci_hi, color=cols[grp], alpha=0.15, lw=0)
            ax.plot(d.rel_month, d.beta, color=cols[grp], lw=0.9, label=f"\u2265{grp[1:]} kt" if grp != "R34" else "34\u201349 kt")
            rows.append(d)
        ax.axvline(0, color="black", lw=0.7); ax.axhline(0, color="#999999", lw=0.4)
        if fu[EID[en]] < 24:
            ax.axvspan(fu[EID[en]] + 0.5, 24.5, color="#bbbbbb", alpha=0.45, lw=0)
        ax.set_title(f"{en} {YEAR[en]}", fontsize=7.5); ax.set_xlim(-24, 24); ax.set_ylim(-0.20, 0.09); ax.set_xticks([-24, -12, 0, 12, 24])
    for ax in list(axes[2, :3]) + [axes[1, 3]]:
        ax.set_xlabel("Months from landfall"); ax.tick_params(labelbottom=True)
    for ax in axes[:, 0]:
        ax.set_ylabel(f"\u0394{index} (exposed \u2212 matched)")
    h = [Line2D([], [], color=cols["R64"], label="\u226564 kt (R64)"), Line2D([], [], color=cols["R50"], label="50\u201363 kt (R50)"),
         Line2D([], [], color=cols["R34"], label="34\u201349 kt (R34)")]
    axes[0, 0].legend(handles=h, fontsize=5.8, loc="lower left")
    fig.tight_layout(h_pad=0.6, w_pad=0.4)
    pd.concat(rows).to_csv(FDT + f"{name}.csv", index=False)
    save(fig, name)
did_classes("NDVI", "primary_matched", "figS1_event_study_did_by_wind_class_ndvi")
did_classes("EVI", "evi_matched", "figS2_event_study_did_by_wind_class_evi")
print("event study", f"{time.time()-t0:.0f}s", flush=True)

# ---------------------------------------------------------------- donut: greenness-loss-equivalent area for robust declines
def donut(name):
    d = EV[(EV["index"] == "NDVI") & EV.group.isin(["R34", "R50", "R64"]) & (EV.evidence_class == "robust decline")].copy()
    d["loss_equiv_km2"] = d.exposed_mangrove_km2_gmw2020 * (-d.mean_m0_6_pct_of_baseline / 100)
    d["loss_equiv_lo"] = d.exposed_mangrove_km2_gmw2020 * (-100 * d.mean_m0_6_ci_hi / d.baseline_level / 100)
    d["loss_equiv_hi"] = d.exposed_mangrove_km2_gmw2020 * (-100 * d.mean_m0_6_ci_lo / d.baseline_level / 100)
    d.to_csv(SUP + "greenness_loss_equivalent_index.csv", index=False)
    return   # greenness-loss-equivalent index table only (reported in the text)
    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    cols = ["#b2182b", "#ef8a62", "#fddbc7", "#67a9cf"][:len(d)]
    w, _ = ax.pie(d.loss_equiv_km2, colors=cols, startangle=90, counterclock=False, wedgeprops=dict(width=0.36, edgecolor="white", lw=1.2))
    for wi, r in zip(w, d.itertuples()):
        ang = np.deg2rad((wi.theta1 + wi.theta2) / 2)
        ax.text(1.28 * np.cos(ang), 1.2 * np.sin(ang),
                f"{r.event_name} {YEAR[r.event_name]} ({r.group})\n{r.loss_equiv_km2:,.0f} km\u00b2 [{r.loss_equiv_lo:,.0f}\u2013{r.loss_equiv_hi:,.0f}]\n{100 * r.loss_equiv_km2 / d.loss_equiv_km2.sum():.0f}%",
                ha="center", va="center", fontsize=6.6)
    ax.text(0, 0, f"{d.loss_equiv_km2.sum():,.0f} km\u00b2\ngreenness-loss\nequivalent", ha="center", va="center", fontsize=7.2, fontweight="bold")
    others = sorted(set(ORDER) - set(d.event_name))
    ax.set_title("Share of short-term canopy greenness loss (months 0\u20136)\namong robust declines", fontsize=8.5)
    fig.text(0.5, 0.02, "Exposed GMW 2020 area \u00d7 relative NDVI decline. Zero contribution (no robust decline): " + ", ".join(others),
             ha="center", fontsize=5.8, wrap=True)
    save(fig, name)
donut("fig06_donut_greenness_loss_equivalent_area")

# ---------------------------------------------------------------- recovery timeline
def timeline(name):
    rows = []
    for en in ORDER:
        for grp in ("R64", "R50", "R34"):
            r = EV[(EV.event_name == en) & (EV["index"] == "NDVI") & (EV.group == grp)]
            if len(r):
                rows.append(r.iloc[0])
    d = pd.DataFrame(rows).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    for i, r in d.iterrows():
        y = len(d) - 1 - i
        end = r.followup_last_month
        ax.plot([0, end], [y, y], color="#d0d0d0", lw=1.2, solid_capstyle="butt", zorder=1)
        if end < 24:
            ax.plot([end, 24], [y, y], color="#d0d0d0", lw=1.2, ls=":", zorder=1)
            ax.scatter([end], [y], marker=">", s=22, c="#555555", zorder=3)
            ax.text(end + 0.3, y + 0.28, f"censored at {end} mo (data end)", fontsize=5.5, color="#444444")
        st = storms[(storms.event_id == r.event_id) & (storms.rel_month > 0) & (storms.rel_month <= 24) & (storms.cells_r50plus >= 0.1 * storms.cells_total)]
        for s in st.itertuples():
            ax.scatter([s.rel_month], [y], marker="|", s=60, c="#1f78b4", zorder=3, lw=1.2)
        if r.evidence_class == "robust decline":
            ax.plot([0, r.recovery_month], [y, y], color="#b2182b", lw=5, solid_capstyle="butt", zorder=2)
            ax.scatter([r.trough_month], [y], marker="v", s=30, c="black", zorder=4)
            ax.text(r.recovery_month + 0.4, y - 0.12, f"recovered by month {int(r.recovery_month)} (trough {r.trough:+.3f} at month {int(r.trough_month)})",
                    fontsize=6, va="center")
        else:
            ax.text(0.4, y - 0.12, r.evidence_class, fontsize=6, style="italic", color="#666666", va="center")
    ax.set_yticks(range(len(d))); ax.set_yticklabels([f"{r.event_name} {YEAR[r.event_name]} \u00b7 {r.group}" for _, r in d.iloc[::-1].iterrows()], fontsize=6.5)
    ax.set_xlim(-0.5, 25.5); ax.set_xlabel("Months from landfall"); ax.axvline(0, color="black", lw=0.7)
    h = [Line2D([], [], color="#b2182b", lw=5, label="decline until recovery to pre-event noise band"),
         Line2D([], [], marker="v", ls="", color="black", label="trough"),
         Line2D([], [], marker="|", ls="", color="#1f78b4", ms=8, mew=1.2, label="later storm \u226550 kt over \u226510% of sample"),
         Line2D([], [], marker=">", ls="", color="#555555", label="follow-up censored (2024 event)")]
    ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=2, fontsize=6.3)
    ax.set_title("NDVI recovery timeline by event and wind class", fontsize=8.5, loc="left")
    d[["event_name", "group", "evidence_class", "trough", "trough_month", "recovery_month", "followup_last_month", "followup_status"]].to_csv(FDT + f"{name}.csv", index=False)
    save(fig, name)
timeline("fig11_recovery_timeline")

# ---------------------------------------------------------------- specification robustness
def specfig(name):
    specs = ["primary_matched", "strictqa_matched", "near_controls_matched", "geo_300km_matched", "seasonal_diff_matched", "unmatched_allcontrols",
             "no_censoring_matched", "censor_r34_matched", "original_matching_v1", "revised_no_biascorr", "evi_matched"]
    labels = ["primary (revised matching, bias-corrected)", "strict QA", "controls \u2264150 km", "controls \u2264300 km", "seasonal-differenced", "unmatched controls",
              "no storm censoring", "censor at later R34", "original matching (submitted)", "revised matching, no bias correction", "EVI"]
    cols = plt.cm.tab20(np.arange(len(specs)) % 20)
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    rows = []
    for i, en in enumerate(ORDER):
        for k, sp in enumerate(specs):
            r = SUM[(SUM.event_name == en) & (SUM.group == head[en]) & (SUM.spec == sp)]
            if r.empty:
                continue
            r = r.iloc[0]; y = len(ORDER) - 1 - i + (k - 3.5) * 0.09
            ax.errorbar(r.mean_m0_6, y, xerr=[[r.mean_m0_6 - r.mean_m0_6_ci_lo], [r.mean_m0_6_ci_hi - r.mean_m0_6]], fmt="o", ms=2.6,
                        color=cols[k], lw=0.7, capsize=0, label=labels[k] if i == 0 else None)
            rows.append({"event_name": en, "group": head[en], "spec": sp, "mean_m0_6": r.mean_m0_6, "ci_lo": r.mean_m0_6_ci_lo, "ci_hi": r.mean_m0_6_ci_hi})
    ax.axvline(0, color="black", lw=0.6)
    ax.set_yticks(range(len(ORDER))); ax.set_yticklabels([f"{e} {YEAR[e]} ({head[e]})" for e in ORDER[::-1]])
    ax.set_xlabel("Mean \u0394 index, months 0\u20136 (95% spatial block bootstrap)")
    ax.legend(fontsize=6, loc="center left", bbox_to_anchor=(1.0, 0.5))
    pd.DataFrame(rows).to_csv(FDT + f"{name}.csv", index=False)
    save(fig, name)
specfig("fig08_robustness_specifications")

def lovefig(name):
    b = bal[bal.control_set == "all_pool"]
    covs = list(dict.fromkeys(b.covariate)); cols = plt.cm.tab10(np.arange(len(ORDER)))
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    for i, en in enumerate(ORDER):
        d = b[b.event_name == en].set_index("covariate").reindex(covs)
        y = np.arange(len(covs)) + (i - 3.5) * 0.08
        ax.scatter(d.smd_before, y, marker="o", facecolors="none", edgecolors=cols[i], s=12, lw=0.7)
        ax.scatter(d.smd_after, y, marker="o", color=cols[i], s=12, label=f"{en} {YEAR[en]}")
    for v in (-0.25, 0.25):
        ax.axvline(v, color="#999999", lw=0.5, ls="--")
    ax.axvline(0, color="black", lw=0.5)
    ax.set_yticks(range(len(covs))); ax.set_yticklabels([c.replace("_", " ") for c in covs])
    ax.set_xlabel("Standardised mean difference, exposed \u2212 controls (open: all controls, filled: matched)")
    ax.legend(fontsize=6, loc="center left", bbox_to_anchor=(1.0, 0.5))
    b.to_csv(FDT + f"{name}.csv", index=False)
    save(fig, name)
lovefig("fig15_matched_control_balance")

def qafig(name):
    q = pd.read_pickle(W + "qa_calmonth.pkl").pivot(index="event_name", columns="cal_m", values="vrelax").reindex(ORDER)
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    im = ax.imshow(q.values, cmap="YlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(12)); ax.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
    ax.set_yticks(range(len(ORDER))); ax.set_yticklabels([f"{e} {YEAR[e]}" for e in ORDER])
    for i in range(q.shape[0]):
        for j in range(12):
            ax.text(j, i, f"{q.values[i, j]:.2f}", ha="center", va="center", fontsize=5.5, color="white" if q.values[i, j] > 0.65 else "black")
    cb = fig.colorbar(im, ax=ax, fraction=0.035); cb.set_label("share of valid observations (relaxed QA)")
    ax.set_xlabel("Calendar month")
    q.to_csv(FDT + f"{name}.csv")
    save(fig, name)
qafig("fig09_modis_valid_observation_coverage")

def placebofig(name):
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    rows = []
    for i, en in enumerate(ORDER):
        r = EV[(EV.event_name == en) & (EV["index"] == "NDVI") & (EV.group == head[en])].iloc[0]
        p = PL[(PL.event_name == en) & (PL["index"] == "NDVI") & (PL.group == head[en])].iloc[0]
        y = len(ORDER) - 1 - i
        ax.errorbar(r.mean_m0_6, y + 0.12, xerr=[[r.mean_m0_6 - r.mean_m0_6_ci_lo], [r.mean_m0_6_ci_hi - r.mean_m0_6]], fmt="o", color="#b2182b", ms=3.5, lw=0.8,
                    label="real landfall" if i == 0 else None)
        ax.errorbar(p.placebo_time_mean_m0_6, y - 0.12, xerr=[[p.placebo_time_mean_m0_6 - p.placebo_time_mean_m0_6_ci_lo], [p.placebo_time_mean_m0_6_ci_hi - p.placebo_time_mean_m0_6]],
                    fmt="s", color="#4d4d4d", ms=3, lw=0.8, label="placebo landfall 12 months earlier" if i == 0 else None)
        ax.text(0.075, y, f"space-placebo p = {p.placebo_space_p_mean_m0_6_one_sided:.3f}", fontsize=5.8, va="center")
        rows.append({"event_name": en, "group": head[en], "real_mean_m0_6": r.mean_m0_6, "placebo_time_mean_m0_6": p.placebo_time_mean_m0_6,
                     "placebo_space_p": p.placebo_space_p_mean_m0_6_one_sided})
    ax.axvline(0, color="black", lw=0.6); ax.set_xlim(-0.1, 0.13)
    ax.set_yticks(range(len(ORDER))); ax.set_yticklabels([f"{e} {YEAR[e]} ({head[e]})" for e in ORDER[::-1]])
    ax.set_xlabel("Mean \u0394NDVI, months 0\u20136 (exposed \u2212 matched controls)")
    ax.legend(fontsize=6, loc="lower left")
    pd.DataFrame(rows).to_csv(FDT + f"{name}.csv", index=False)
    save(fig, name)
placebofig("fig10b_placebo_diagnostics")
pd.Series(made).to_csv(FDT + "_figure_list.csv", index=False)
print("DONE", len(made), f"{time.time()-t0:.0f}s", flush=True)
