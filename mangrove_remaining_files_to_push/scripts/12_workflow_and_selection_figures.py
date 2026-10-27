"""Stage 12. Workflow figure (Fig. 1) and event-selection figure (Fig. 2). Fig. 2 counts are read from the storm screen."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from p4.config import FIG, SUP
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, Ellipse, Rectangle, Arc
plt.rcParams.update({"font.family": "DejaVu Sans"})
OUT = FIG; K = "#1f2a36"; LW = 0.9

def rbox(ax, x, y, w, h, fc, ec=K, lw=LW, r=0.06, ls="-", z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, ls=ls, zorder=z))
def arr(ax, pts, ls="-", lw=LW, hw=0.16, hl=0.30):
    for (x0, y0), (x1, y1) in zip(pts[:-2], pts[1:-1]): ax.plot([x0, x1], [y0, y1], color=K, lw=lw, ls=ls, zorder=1)
    ax.annotate("", xy=pts[-1], xytext=pts[-2], arrowprops=dict(arrowstyle=f"-|>,head_length={hl},head_width={hw}", color=K, lw=lw, ls=ls, shrinkA=0, shrinkB=0), zorder=1)
def badge(ax, x, y, n):
    ax.add_patch(Ellipse((x, y), 0.20, 0.20, fc="white", ec=K, lw=0.8, zorder=6)); ax.text(x, y, str(n), ha="center", va="center", fontsize=7.5, fontweight="bold", zorder=7)

# ======================================================================= Figure 1: pipeline architecture style
W, H = 6.7, 2.75
fig = plt.figure(figsize=(W, H)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
T, S = 8.8, 7.9
# --- inputs with icons
def icon_track(x, y):
    t = np.linspace(0, 1, 40); ax.plot(x + 0.05 + 0.30 * t, y + 0.05 + 0.26 * t ** 1.6, color="#c0392b", lw=1.1, zorder=3)
    for tt in (0.15, 0.45, 0.75): ax.plot(x + 0.05 + 0.30 * tt, y + 0.05 + 0.26 * tt ** 1.6, "o", ms=2.2, color="#c0392b", zorder=3)
    th = np.linspace(0, 4 * np.pi, 80); rr = 0.009 * th; ax.plot(x + 0.30 + rr * np.cos(th), y + 0.30 + rr * np.sin(th), color="#7f8c8d", lw=0.7, zorder=3)
def icon_grid(x, y):
    cols = ["#2e7d32", "#66bb6a", "#a5d6a7", "#43a047", "#2e7d32", "#81c784", "#c8e6c9", "#388e3c", "#66bb6a"]
    for i in range(3):
        for j in range(3): ax.add_patch(Rectangle((x + 0.04 + 0.11 * i, y + 0.04 + 0.11 * j), 0.10, 0.10, fc=cols[3 * i + j], ec="white", lw=0.4, zorder=3))
def icon_frames(x, y):
    for k, c in enumerate(["#9ccc65", "#7cb342", "#558b2f"]):
        ax.add_patch(Rectangle((x + 0.04 + 0.05 * k, y + 0.03 + 0.04 * k), 0.24, 0.20, fc=c, ec="#33402a", lw=0.5, zorder=3 + k))
INP = [(icon_track, "IBTrACS v04r01", "tracks, quadrant radii"), (icon_grid, "Global Mangrove\nWatch v3", "0.01\u00b0 mangrove cells"), (icon_frames, "MODIS Terra\nand Aqua", "NDVI, EVI, QA (GEE)")]
iy = [1.98, 1.20, 0.42]
for (f, a, b), y in zip(INP, iy):
    f(0.06, y); nl = a.count("\n"); ax.text(0.50, y + 0.31 + 0.05 * nl, a, fontsize=S - 0.2, fontweight="bold", va="center", linespacing=1.0); ax.text(0.50, y + 0.11 - 0.02 * nl, b, fontsize=S - 0.6, va="center", color="#333333")
badge(ax, 0.10, 2.60, 1); ax.text(0.25, 2.60, "Data integration", fontsize=T, fontweight="bold", va="center")
# --- exposure and alignment (trapezoid, encoder style)
tx0, tx1, ty0, ty1, shr = 1.97, 2.70, 0.62, 1.92, 0.20
ax.add_patch(Polygon([(tx0, ty0), (tx1, ty0 + shr), (tx1, ty1 - shr), (tx0, ty1)], closed=True, fc="#8fbf4f", ec=K, lw=LW, zorder=2))
ax.text((tx0 + tx1) / 2 - 0.01, (ty0 + ty1) / 2, "Exposure\nand\nalignment", ha="center", va="center", fontsize=T - 1.0, fontweight="bold", zorder=3, linespacing=1.05)
badge(ax, tx0 + 0.15, ty1 - 0.20, 2)
ax.text((tx0 + tx1) / 2 - 0.10, 0.24, "R34, R50, R64 classes\nlandfall time axis, QA", ha="center", va="top", fontsize=S - 0.6, color="#333333", linespacing=1.0)
xs_in, xe_el = 1.74, 1.86
yc = [iy[0] + 0.21, iy[1] + 0.22, iy[2] + 0.22]
ye = [ty1 - 0.26, yc[1], ty0 + 0.26]
for c_, e_ in zip(yc, ye):
    if abs(c_ - e_) < 1e-6: arr(ax, [(xs_in, c_), (tx0, e_)])
    else: arr(ax, [(xs_in, c_), (xe_el, c_), (xe_el, e_), (tx0, e_)])
# --- counterfactual design: matched controls (top, memory style) and seasonal baseline (bottom)
cx0, cw = 2.98, 1.50
mty, mth = 2.02, 0.46
rbox(ax, cx0, mty, cw, mth, "#f8c9bd", r=0.08)
ax.text(cx0 + 0.20, mty + mth + 0.12, "Matched counterfactual", ha="left", va="center", fontsize=T, fontweight="bold"); badge(ax, cx0 + 0.06, mty + mth + 0.12, 3)
slots = [("exposed", "#e57373"), ("control", "#64b5f6"), ("control", "#64b5f6")]
sx = cx0 + 0.12
for lab, c in slots:
    rbox(ax, sx, mty + 0.16, 0.28, mth - 0.21, "#eeeeee", lw=0.6, r=0.04, z=3)
    for j in range(3): ax.add_patch(Rectangle((sx + 0.045 + 0.072 * j, mty + 0.20), 0.050, mth - 0.29, fc=c, ec="#333333", lw=0.3, zorder=4))
    sx += 0.31
ax.text(sx + 0.03, mty + mth / 2 + 0.03, "\u2026 k \u2264 5", ha="left", va="center", fontsize=S - 0.9, zorder=4)
ax.text(cx0 + 0.12 + 0.14, mty + 0.07, "exposed", ha="center", va="center", fontsize=S - 0.9, zorder=4)
ax.text(cx0 + 0.12 + 0.31 + 0.295, mty + 0.07, "controls", ha="center", va="center", fontsize=S - 0.9, zorder=4)
bby, bbh = 0.10, 0.44
rbox(ax, cx0, bby, cw, bbh, "#d9c7a3", r=0.08)
ax.text(cx0 + cw / 2, bby - 0.02 + bbh + 0.0 - bbh - 0.06, "", fontsize=1)
tt = np.linspace(0, 1, 100); ax.plot(cx0 + 0.32 + 0.46 * tt, bby + bbh / 2 + 0.11 * np.sin(2 * np.pi * 2 * tt), color="#5d4037", lw=1.0, zorder=3)
ax.text(cx0 + 0.86, bby + bbh / 2, "Seasonal\nbaseline", ha="left", va="center", fontsize=T - 0.8, fontweight="bold", zorder=3, linespacing=1.0)
badge(ax, cx0 + 0.15, bby + bbh / 2, 3)
# --- estimation module (centre)
ex0, ey0, ew, eh = 3.08, 1.10, 1.24, 0.40
rbox(ax, ex0, ey0, ew, eh, "#f6b13c", r=0.10)
ax.text(ex0 + ew / 2 + 0.11, ey0 + eh / 2, "Event-study\nestimation", ha="center", va="center", fontsize=T, fontweight="bold", zorder=3, linespacing=1.0); badge(ax, ex0 + 0.16, ey0 + eh / 2, 4)
ym = ey0 + eh / 2
arr(ax, [(tx1, ym), (ex0, ym)])
arr(ax, [((tx0 + tx1) / 2, ty1 - shr / 2), ((tx0 + tx1) / 2, mty + mth / 2), (cx0, mty + mth / 2)])
ax.text(2.56, mty + mth / 2 + 0.16, "PS caliper,\nMahalanobis", ha="center", va="bottom", fontsize=S - 0.8, linespacing=1.0)
arr(ax, [((tx0 + tx1) / 2, ty0 + shr / 2), ((tx0 + tx1) / 2, bby + bbh / 2), (cx0, bby + bbh / 2)])
arr(ax, [(ex0 + ew / 2, mty), (ex0 + ew / 2, ey0 + eh)]); ax.text(ex0 + ew / 2 - 0.06, (mty + ey0 + eh) / 2, "matched\ncontrols", ha="right", va="center", fontsize=S - 0.8, linespacing=1.0)
arr(ax, [(ex0 + ew / 2, bby + bbh), (ex0 + ew / 2, ey0)]); ax.text(ex0 + ew / 2 - 0.06, (bby + bbh + ey0) / 2, "monthly\nanomalies", ha="right", va="center", fontsize=S - 0.8, linespacing=1.0)
# --- monthly series bar, inference head, output
bx = 4.55; ax.add_patch(Rectangle((bx, ym - 0.48), 0.08, 0.96, fc="#f39c12", ec=K, lw=0.8, zorder=2))
ax.text(bx + 0.13, ym + 0.40, "monthly\n\u0394NDVI", ha="left", va="center", fontsize=S - 0.8, linespacing=1.0)
arr(ax, [(ex0 + ew, ym), (bx, ym)])
hx0, hx1 = 4.82, 5.50
ax.add_patch(Polygon([(hx0, ym - 0.18), (hx1, ym - 0.30), (hx1, ym + 0.30), (hx0, ym + 0.18)], closed=True, fc="#2a9d8f", ec=K, lw=LW, zorder=2))
ax.text((hx0 + hx1) / 2 + 0.02, ym, "Inference", ha="center", va="center", fontsize=T - 1.4, fontweight="bold", color="white", zorder=3)
ax.text((hx0 + hx1) / 2 - 0.04, ym - 0.36, "bootstrap, HAC,\nplacebo, Holm", ha="center", va="top", fontsize=S - 0.8, linespacing=1.0)
arr(ax, [(bx + 0.08, ym), (hx0, ym)])
# output icon: mini forest plot
ox, oy = 6.05, ym - 0.30
rbox(ax, ox, oy, 0.58, 0.60, "white", lw=0.7, r=0.04)
ax.plot([ox + 0.38, ox + 0.38], [oy + 0.06, oy + 0.54], color="#7f8c8d", lw=0.6, ls=(0, (2, 1)), zorder=3)
for k, (c, lo, hi, col) in enumerate([(0.18, 0.10, 0.27, "#c0392b"), (0.14, 0.08, 0.21, "#c0392b"), (0.40, 0.31, 0.49, "#555555"), (0.43, 0.33, 0.52, "#555555")]):
    yy = oy + 0.48 - 0.12 * k; ax.plot([ox + lo, ox + hi], [yy, yy], color=col, lw=0.9, zorder=4); ax.plot(ox + c, yy, "o", ms=2.6, color=col, zorder=5)
ax.text(ox + 0.29, oy + 0.70, "Output", ha="center", va="bottom", fontsize=T - 0.6, fontweight="bold")
arr(ax, [(hx1, ym), (ox, ym)])
# evidence grading (gating style) and recovery
gx0, gy0, gw, gh = 4.76, 0.20, 1.48, 0.40
rbox(ax, gx0, gy0, gw, gh, "#4f9de0", r=0.10)
ax.text(gx0 + gw / 2 + 0.11, gy0 + gh / 2, "Evidence grading\nand recovery", ha="center", va="center", fontsize=T - 0.6, fontweight="bold", color="white", zorder=3, linespacing=1.0); badge(ax, gx0 + 0.16, gy0 + gh / 2, 5)
jx = 5.72; arr(ax, [(jx, ym), (jx, gy0 + gh)]); ax.text(jx + 0.06, gy0 + gh + 0.20, "test\nresults", ha="left", va="center", fontsize=S - 0.8, linespacing=1.0)
arr(ax, [(gx0 + gw, gy0 + gh / 2), (ox + 0.29, gy0 + gh / 2), (ox + 0.29, oy)])
ax.plot([bx + 0.04, bx + 0.04], [ym - 0.48, gy0 + gh / 2], color=K, lw=LW, zorder=1); arr(ax, [(bx + 0.04, gy0 + gh / 2), (gx0, gy0 + gh / 2)])
# exploratory ML (dashed, secondary)
mx0, my0, mw, mh = 4.85, 2.06, 1.78, 0.46
rbox(ax, mx0, my0, mw, mh, "#ede4f6", ls=(0, (4, 2)), r=0.08)
ax.text(mx0 + mw / 2, my0 + mh / 2 + 0.07, "Exploratory ML", ha="center", va="center", fontsize=T - 0.6, fontweight="bold", zorder=3)
ax.text(mx0 + mw / 2, my0 + mh / 2 - 0.12, "ridge, gradient boosting, SHAP", ha="center", va="center", fontsize=S - 0.8, zorder=3)
arr(ax, [(bx + 0.04, ym + 0.48), (bx + 0.04, ym + 0.64), (mx0 + mw / 2, ym + 0.64), (mx0 + mw / 2, my0)], ls=(0, (4, 2)))
ax.text(mx0 + mw / 2, my0 + mh + 0.11, "secondary, not used for grading", ha="center", va="center", fontsize=S - 0.8, style="italic", color="#444444")
fig.savefig(OUT + "fig01_workflow_dataset_construction.png", dpi=600, facecolor="white"); plt.close(fig)
print(f"Fig 1 {W} x {H} in; print scale {5.17 / W:.2f}; smallest text {(S - 0.9) * 5.17 / W:.1f} pt")

# ======================================================================= Figure 2: phase banner, data cylinder, screening funnel
import pandas as pd
SC = pd.read_csv(SUP + "bay_of_bengal_cyclone_screen_2000_2024.csv")
n_all = len(SC); n34 = int((SC.max_usa_wind_kt >= 34).sum()); nlf = int(((SC.max_usa_wind_kt >= 34) & SC.landfall).sum())
nrad = int(((SC.max_usa_wind_kt >= 34) & SC.landfall & (SC.fixes_with_reported_r34 > 0)).sum())
EL = SC[SC.decision.str.startswith(("analysed", "eligible"))]; nel = len(EL); nan_ = int(EL.decision.str.startswith("analysed").sum()); nnot = nel - nan_
r64e = int((EL.r64_mangrove_km2 > 0).sum()); r64a = int(((EL.r64_mangrove_km2 > 0) & EL.decision.str.startswith("analysed")).sum())
W2, H2 = 5.4, 3.0
fig = plt.figure(figsize=(W2, H2)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W2); ax.set_ylim(0, H2); ax.axis("off")
F = 8.4
hy, hh, tip = 2.60, 0.34, 0.15
heads = [("Data source", "#cfe3f3", 0.05, 1.58), ("Screening criteria", "#d6ecc4", 1.58, 3.78), ("Selection outcome", "#f8d8b0", 3.78, 5.35)]
for k, (lab, c, x0, x1) in enumerate(heads):
    pts = [(x0, hy), (x1 - tip, hy), (x1, hy + hh / 2), (x1 - tip, hy + hh), (x0, hy + hh)] + ([(x0 + tip, hy + hh / 2)] if k else [])
    ax.add_patch(Polygon(pts, closed=True, fc=c, ec="#4a5a6a", lw=0.7, zorder=2))
    ax.text((x0 + x1) / 2 + (0.04 if k == 1 else (0.0 if k == 2 else -0.04)), hy + hh / 2, lab, ha="center", va="center", fontsize=F + 0.2, fontweight="bold", zorder=3)
# data cylinder
cxm, cw2, cy0, cy1, eh2 = 0.78, 1.34, 0.55, 2.28, 0.24
ax.add_patch(Rectangle((cxm - cw2 / 2, cy0), cw2, cy1 - cy0, fc="#eaf3fb", ec="none", zorder=2))
ax.add_patch(Ellipse((cxm, cy0), cw2, eh2, fc="#eaf3fb", ec="none", zorder=2))
ax.add_patch(Arc((cxm, cy0), cw2, eh2, theta1=180, theta2=360, ec="#2e6fa7", lw=0.9, ls=(0, (3, 2)), zorder=3))
ax.add_patch(Rectangle((cxm - cw2 / 2 + 0.005, cy0), cw2 - 0.01, 0.02, fc="#eaf3fb", ec="none", zorder=2))
for xx in (cxm - cw2 / 2, cxm + cw2 / 2): ax.plot([xx, xx], [cy0, cy1], color="#2e6fa7", lw=0.9, ls=(0, (3, 2)), zorder=3)
ax.add_patch(Ellipse((cxm, cy1), cw2, eh2, fc="#d4e6f6", ec="#2e6fa7", lw=0.9, ls=(0, (3, 2)), zorder=4))
ax.text(cxm, 1.86, "IBTrACS v04r01", ha="center", va="center", fontsize=F, fontweight="bold", zorder=5)
for k, t in enumerate(["\u2022 Bay of Bengal", "\u2022 2000 to 2024", f"\u2022 n = {n_all} storms"]):
    ax.text(cxm - cw2 / 2 + 0.12, 1.56 - 0.25 * k, t, ha="left", va="center", fontsize=F - 0.6, zorder=5)
# hollow block arrow
ay = 2.40 - 0.22
_xe = 2.68 - (0.92 + 0.80) / 2 - 0.012   # left edge of the top funnel tier at its mid-height, minus the edge half-width
ax.add_patch(Polygon([(_xe - 0.32, ay - 0.09), (_xe - 0.16, ay - 0.09), (_xe - 0.16, ay - 0.17), (_xe, ay), (_xe - 0.16, ay + 0.17), (_xe - 0.16, ay + 0.09), (_xe - 0.32, ay + 0.09)], closed=True, fc="white", ec="#4a5a6a", lw=0.9, zorder=3))
# screening funnel (counts inside each tier)
fx, top, gap2 = 2.68, 2.40, 0.05
tiers = [(["Maximum wind \u2265 34 kt"], n34, 0.92, 0.80, 0.44), (["Landfall"], nlf, 0.80, 0.68, 0.44), (["Reported R34 radii"], nrad, 0.68, 0.56, 0.44),
         (["Mangrove overlap", "\u2265 50 km\u00b2"], nel, 0.56, 0.44, 0.56)]
cols = ["#c5e1a5", "#aed581", "#9ccc65", "#7cb342"]; y1 = top
for k, (lines, n, w1, w0, th) in enumerate(tiers):
    y0 = y1 - th
    ax.add_patch(Polygon([(fx - w1, y1), (fx + w1, y1), (fx + w0, y0), (fx - w0, y0)], closed=True, fc=cols[k], ec="#33691e", lw=0.8, zorder=2))
    allt = lines + [f"n = {n}"]; step = 0.135; yc = (y0 + y1) / 2 + step * (len(allt) - 1) / 2
    for j, t in enumerate(allt):
        ax.text(fx, yc - step * j, t, ha="center", va="center", fontsize=F - 0.9 if j < len(lines) else F - 0.5, fontweight="bold" if j == len(lines) else "normal", zorder=3)
    y1 = y0 - gap2
fbot = y0
# outcome arrow boxes
def pent(x, y, w, h, c, lab, n, note=None):
    ax.add_patch(Polygon([(x, y), (x + w - 0.16, y), (x + w, y + h / 2), (x + w - 0.16, y + h), (x, y + h)], closed=True, fc=c, ec="#7a4a12", lw=0.8, zorder=2))
    off = 0.07 if note else 0.0
    ax.text(x + 0.12, y + h / 2 + 0.08 + off, lab, ha="left", va="center", fontsize=F - 0.3, fontweight="bold", zorder=3)
    ax.text(x + 0.12, y + h / 2 - 0.10 + off, f"n = {n}", ha="left", va="center", fontsize=F - 0.3, zorder=3)
    if note: ax.text(x + 0.12, y + 0.10, note, ha="left", va="center", fontsize=F - 1.6, style="italic", color="#3a3a3a", zorder=3)
px, pw, ph = 3.96, 1.36, 0.46
ya, yb, phb = 1.82, 1.08, 0.60
pent(px, ya, pw, ph, "#f4a261", "Analysed", nan_); pent(px, yb, pw, phb, "#f9c88b", "Not analysed", nnot, note="no MODIS extraction")
jx2, ylow = 3.72, fbot - 0.12
ax.plot([fx, fx], [fbot, ylow], color=K, lw=LW); ax.plot([fx, jx2], [ylow, ylow], color=K, lw=LW); ax.plot([jx2, jx2], [ylow, ya + ph / 2], color=K, lw=LW)
arr(ax, [(jx2, ya + ph / 2), (px, ya + ph / 2)]); arr(ax, [(jx2, yb + phb / 2), (px, yb + phb / 2)])
# callout
ex, ey, ew2, eh3 = 4.64, 0.40, 1.42, 0.66
ax.add_patch(Ellipse((ex, ey), ew2, eh3, fc="white", ec="#4a5a6a", lw=0.8, ls=(0, (3, 2)), zorder=2))
ax.text(ex, ey, f"{r64a} of {r64e} eligible storms\nwith R64 winds over\nmangroves analysed", ha="center", va="center", fontsize=F - 1.5, zorder=3, linespacing=1.15)
ax.plot([ex, ex], [ey + eh3 / 2, yb], color="#4a5a6a", lw=0.7, ls=(0, (3, 2)), zorder=1)
fig.savefig(OUT + "figR2_event_selection_flow.png", dpi=600, facecolor="white"); plt.close(fig)
print(f"Fig 2 {W2} x {H2} in; print scale {0.78 * 5.17 / W2:.2f}; smallest text {(F - 1.6) * 0.78 * 5.17 / W2:.1f} pt")
