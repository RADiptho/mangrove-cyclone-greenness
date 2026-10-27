"""Stage 06. Evidence-grading matrix figure (criteria met by each event and wind class).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from p4.config import W, O, T, FIG, FDT, SUP, data, STORM_FLAGS
import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                     "savefig.bbox": "tight", "savefig.pad_inches": 0.04, "svg.fonttype": "none", "pdf.fonttype": 42})
def save(fig, name):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIG + f"{name}.{ext}", dpi=600, facecolor="white")
    plt.close(fig)
ORDER = ["Sidr", "Nargis", "Aila", "Fani", "Bulbul", "Amphan", "Yaas", "Sitrang", "Mocha", "Remal", "Dana"]
YEAR = {"Sidr": 2007, "Nargis": 2008, "Aila": 2009, "Fani": 2019, "Bulbul": 2019, "Amphan": 2020, "Yaas": 2021, "Sitrang": 2022, "Mocha": 2023, "Remal": 2024, "Dana": 2024}

CLASS_COL = {"robust decline": "#b2182b", "suggestive decline": "#ef8a62", "no robust decline evidence": "#7f7f7f",
             "positive anomaly (not interpretable)": "#2a8c7f", "pre-trend failure (not interpretable)": "#e08214"}
G = pd.read_csv(T + 'evidence_grading_table.csv')
def grading_fig():
    crit_cols = ["c1_ci_below_zero", "c2_hac_p_lt_0p05", "c3_space_placebo_p_lt_0p05", "c4_strict_qa_same_sign", "c5_near_controls_same_sign",
                 "c6_evi_same_sign", "c7_no_pretrend_hac", "c8_complete_24m_followup"]
    labels = ["CI\n< 0", "HAC\np < 0.05", "placebo\np < 0.05", "strict QA\nsign", "near ctrl\nsign", "EVI\nsign", "no\npre-trend", "24-month\nfollow-up"]
    d = G.copy(); d["o"] = d.event_name.map({e: i for i, e in enumerate(ORDER)}); d["g"] = d.group.map({"R64": 0, "R50": 1, "R34": 2})
    d = d.sort_values(["o", "g"]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    for i, r in d.iterrows():
        yy = len(d) - 1 - i
        for j, c in enumerate(crit_cols):
            v = r[c]
            fc = "#d9d9d9" if (v is None or (isinstance(v, float) and np.isnan(v))) else ("#4daf4a" if bool(v) else "#f4a3a3")
            if j == 7:
                fc = "#d9d9d9" if bool(v) else "#fdd49e"
            ax.add_patch(Rectangle((j, yy), 0.94, 0.86, fc=fc, ec="white"))
            sym = "\u2013" if fc == "#d9d9d9" and j != 7 else ("\u2713" if bool(v) else "\u2717")
            ax.text(j + 0.47, yy + 0.43, sym, ha="center", va="center", fontsize=7)
        ax.text(8.15, yy + 0.43, r.evidence_grade, va="center", fontsize=6.2, color=CLASS_COL[r.evidence_grade], fontweight="bold" if r.evidence_grade == "robust decline" else "normal")
    ax.set_xlim(0, 11.2); ax.set_ylim(0, len(d))
    ax.set_yticks(np.arange(len(d)) + 0.43); ax.set_yticklabels([f"{r.event_name} {YEAR[r.event_name]} \u00b7 {r.group}" for _, r in d.iloc[::-1].iterrows()], fontsize=6.5)
    ax.set_xticks(np.arange(8) + 0.47); ax.set_xticklabels(labels, fontsize=5.6); ax.xaxis.tick_top()
    ax.text(8.15, len(d) + 0.3, "evidence grade", fontsize=6.5, fontweight="bold")
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    h = [Patch(fc="#4daf4a", label="criterion met"), Patch(fc="#f4a3a3", label="criterion not met"), Patch(fc="#d9d9d9", label="not available / complete follow-up"),
         Patch(fc="#fdd49e", label="follow-up censored (22.6 months)")]
    ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.45, -0.01), ncol=4, fontsize=6)
    d.to_csv(FDT + "fig07_evidence_grading_matrix.csv", index=False)
    save(fig, "fig07_evidence_grading_matrix")
grading_fig()
print('ok')
