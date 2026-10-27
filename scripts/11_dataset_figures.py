"""Stage 11. Dataset construction counts and exposed cells by event and wind class (Supplementary Figs. S1 and S2).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from p4.config import W, O, T, FIG, FDT, SUP, data, STORM_FLAGS
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "svg.fonttype": "none", "pdf.fonttype": 42, "legend.frameon": False})
def save(fig, name):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIG + f"{name}.{ext}", dpi=600, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)
CP = pd.read_csv(T + "final_event_cell_panel.csv")
an = CP[CP.in_analysis]
CLS = ["R64", "R50", "R34", "control"]
COL = {"R64": "#d7301f", "R50": "#fc8d59", "R34": "#fdcc8a", "control": "#6baed6"}
NAME = {"R64": "R64 (≥64 kt)", "R50": "R50 (50–63 kt)", "R34": "R34 (34–49 kt)", "control": "Matched-control pool"}
cnt = an.exposure_class.value_counts().reindex(CLS)
# ---------------- stacked bars: analysed event-cells by event and class
ORDER = ["Sidr", "Nargis", "Aila", "Fani", "Bulbul", "Amphan", "Yaas", "Sitrang", "Mocha", "Remal", "Dana"]
YEAR = {"Sidr": 2007, "Nargis": 2008, "Aila": 2009, "Fani": 2019, "Bulbul": 2019, "Amphan": 2020, "Yaas": 2021, "Sitrang": 2022, "Mocha": 2023, "Remal": 2024, "Dana": 2024}
tab = an.groupby(["event_name", "exposure_class"]).size().unstack(fill_value=0).reindex(ORDER).reindex(columns=CLS, fill_value=0)
fig, ax = plt.subplots(figsize=(6.4, 3.4))
left = np.zeros(len(tab))
y = np.arange(len(tab))[::-1]
for c in CLS:
    ax.barh(y, tab[c].values, left=left, color=COL[c], edgecolor="white", lw=0.6, label=NAME[c], height=0.68)
    for yi, l, v in zip(y, left, tab[c].values):
        if v >= 180:
            ax.text(l + v / 2, yi, f"{v:,}", ha="center", va="center", fontsize=6.6, color="black" if c != "R64" else "white")
    left += tab[c].values
for yi, tot in zip(y, left):
    ax.text(tot + 40, yi, f"{int(tot):,}", va="center", fontsize=7)
ax.set_yticks(y); ax.set_yticklabels([f"{e} {YEAR[e]}" for e in ORDER]); ax.set_xlabel("Event-cells in the matched analysis panel")
ax.set_xlim(0, left.max() * 1.12)
ax.legend(ncol=4, fontsize=7, loc="upper center", bbox_to_anchor=(0.45, -0.17))
tab.reset_index().to_csv(T + "figure_data_tables/fig_cells_by_event_class.csv", index=False)
save(fig, "figD3_cells_by_event_and_class")
# ---------------- dataset construction counts (log scale, stage coloured)
counts = [("Stage 1", "IBTrACS candidate events", 24), ("Stage 1", "Priority Bay of Bengal events", 8),
          ("Stage 2", "GMW v3 GeoTIFF files extracted", 4563), ("Stage 2", "Candidate GMW tiles", 722),
          ("Stage 3", "Exposed cells sampled (R34, R50, R64)", int((CP.exposure_class != "control").sum())), ("Stage 3", "Control cells sampled", int((CP.exposure_class == "control").sum())),
          ("Stage 4", "Earth Engine export jobs (all completed)", 436), ("Stage 4", "Raw MODIS point-composite rows", 4989570), ("Stage 4", "Analysed MODIS rows", 4989550),
          ("Stage 5", "Event-cell panel records", len(CP)), ("Stage 5", "Matched analysis event-cells", int(CP.in_analysis.sum())), ("Stage 5", "Event-cell-month rows", 925675)]
SC = {"Stage 1": "#E69F00", "Stage 2": "#1b7837", "Stage 3": "#d7301f", "Stage 4": "#0072B2", "Stage 5": "#6a51a3"}
fig, ax = plt.subplots(figsize=(6.6, 4.0))
yy = np.arange(len(counts))[::-1]
for yi, (st, lab, v) in zip(yy, counts):
    ax.barh(yi, v, color=SC[st], height=0.62, alpha=0.9)
    ax.text(v * 1.15, yi, f"{v:,}", va="center", fontsize=7)
ax.set_xscale("log"); ax.set_xlim(3, 4e7)
ax.set_yticks(yy); ax.set_yticklabels([c[1] for c in counts], fontsize=7.4)
ax.set_xlabel("Count (log scale)")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(fc=c, label=s) for s, c in SC.items()], ncol=5, fontsize=7, loc="upper center", bbox_to_anchor=(0.35, -0.13))
pd.DataFrame(counts, columns=["stage", "item", "count"]).to_csv(T + "figure_data_tables/fig_dataset_construction_counts.csv", index=False)
save(fig, "figD1_dataset_construction_counts")
