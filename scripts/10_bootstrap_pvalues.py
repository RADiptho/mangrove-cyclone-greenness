"""Stage 10. Two-sided bootstrap p-values from 4,999 spatial block replicates for the multiplicity analysis.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from p4.config import W, O, T, FIG, FDT, SUP, data, STORM_FLAGS
# Bootstrap p-values with 4,999 replicates for the 27-estimate multiplicity analysis (CIs remain 500-replicate).
import numpy as np, pandas as pd
RT = SUP; SEED = 20261004; MONTHS = np.arange(-36, 25); IDX = lambda m: m + 36
A = pd.read_pickle(W + "anom.pkl"); R = pd.read_pickle(W + "stageB.pkl"); meta, match, BC = R["meta"], R["match"], R["bc_primary"]
Mn = A[("ndvi", "vrelax", "main")]["M"].astype(float); Mn = np.vstack([Mn, np.full((1, Mn.shape[1]), np.nan)]); mi = meta.set_index("pid")
CENS = mi["censor_m50"]
def holm(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); adj = np.empty(n); run = 0
    for r, i in enumerate(o): run = max(run, (n - r) * p[i]); adj[i] = min(1, run)
    return adj
def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)[::-1]; adj = np.empty(n); run = 1
    for r, i in enumerate(o): run = min(run, p[i] * n / (n - r)); adj[i] = run
    return adj
MT = pd.read_csv(RT + "multiple_testing_and_equivalence.csv"); EID = {e.split("_")[1].title(): e for e in meta.event_id.unique()}; out = []
for r in MT.itertuples():
    eid = EID[r.event_name]; mt = match[eid]; ep = mt["exp_pid"]; cls = mi.loc[ep, "exposure_class"].to_numpy(); s = cls == r.group
    D = Mn[ep] - np.nanmean(Mn[mt["ctrl_pid"]], 1) - BC[(eid, "ndvi")]
    cm = CENS.loc[ep].to_numpy(); D[MONTHS[None, :] >= cm[:, None]] = np.nan          # primary spec censors at the next >= 50 kt storm
    Dg = D[s]; sl = slice(IDX(0), IDX(6) + 1); Dm = Dg[:, sl]; f = np.isfinite(Dm); ok = f.sum(0) >= max(30, 0.2 * len(Dg))
    est = np.nanmean(np.nanmean(Dm, 0)[ok])
    bl = mi.loc[ep[s], "block"].to_numpy(); ub, inv = np.unique(bl, return_inverse=True)
    rng = np.random.default_rng(SEED + 4999 + len(out)); Wt = rng.multinomial(len(ub), np.full(len(ub), 1 / len(ub)), size=4999)[:, inv].astype(float)
    bb = (Wt @ np.where(f, Dm, 0)) / np.maximum(Wt @ f, 1e-12); v = np.nanmean(bb[:, ok], 1)
    p = max(2 * min(np.mean(v >= 0), np.mean(v <= 0)), 1 / 4999)
    out.append({"event_name": r.event_name, "group": r.group, "check_mean_m0_6": est, "boot4999_p_two_sided": p})
O = pd.DataFrame(out); MT = MT.merge(O, on=["event_name", "group"])
assert (MT.check_mean_m0_6 - MT.mean_m0_6).abs().max() < 1e-6, (MT.check_mean_m0_6 - MT.mean_m0_6).abs().max()
MT["boot4999_p_holm"] = holm(MT.boot4999_p_two_sided); MT["boot4999_p_bh"] = bh(MT.boot4999_p_two_sided)
MT["survives_holm_all_three"] = (MT.hac_p_holm < 0.05) & (MT.boot4999_p_holm < 0.05) & (MT.space_p_holm < 0.05) & (MT.mean_m0_6 < 0)
MT.drop(columns=["check_mean_m0_6"]).to_csv(RT + "multiple_testing_and_equivalence.csv", index=False)
print("family", len(MT)); print(MT[["event_name", "group", "mean_m0_6", "hac_p_holm", "boot4999_p_two_sided", "boot4999_p_holm", "space_p_holm", "survives_holm_all_three"]].round(4).to_string(index=False))
