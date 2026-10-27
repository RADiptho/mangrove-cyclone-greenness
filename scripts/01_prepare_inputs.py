"""Stage 01. Build the observation and cell caches from the released Mendeley dataset.

Reads the two observation-level QC files (7,183,770 MODIS point-composite rows) and the cell metadata
(31,263 event-cells), and applies the same conversions as the original pipeline (cell id = metadata row order).
"""
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from p4.config import W, data

t0 = time.time()
meta = pd.read_csv(data("inputs/modis_cell_metadata.csv"))
meta["pid"] = np.arange(len(meta), dtype=np.int32)
key = pd.Series(meta["pid"].values, index=meta["event_id"] + "|" + meta["cell_uid"])
meta.to_pickle(W + "meta.pkl")

FILES = ["observations/modis_observations_qc_sidr_nargis_aila_amphan_yaas_sitrang_remal_dana.csv.gz",
         "observations/modis_observations_qc_fani_bulbul_mocha.csv.gz"]
cols = ["event_id", "cell_uid", "sensor", "composite_start", "date", "date_source", "relative_day", "NDVI", "EVI", "SummaryQA",
        "valid_strict", "valid_relaxed", "flag_fill", "dqa_mixed_clouds", "dqa_possible_shadow", "dqa_adjacent_cloud", "dqa_vi_usefulness"]
ep = pd.Timestamp("1970-01-01")
out = []
for f in FILES:
    for ch in pd.read_csv(data(f), usecols=cols, chunksize=750_000, low_memory=False):
        pid = key.reindex((ch["event_id"] + "|" + ch["cell_uid"]).values).values
        cs = pd.to_datetime(ch["composite_start"]); dd = pd.to_datetime(ch["date"])
        out.append(pd.DataFrame({
            "pid": pid.astype("float64"), "sensor": (ch["sensor"] == "MYD13Q1").astype(np.int8),
            "cday": ((cs - ep).dt.days).astype(np.int32), "day": ((dd - ep).dt.days).astype(np.int32),
            "doy_ok": (ch["date_source"] == "pixel_DayOfYear").astype(bool), "rel_day": ch["relative_day"].astype(np.int16),
            "ndvi": ch["NDVI"].astype(np.float32), "evi": ch["EVI"].astype(np.float32),
            "sqa": pd.to_numeric(ch["SummaryQA"], errors="coerce").fillna(-9).astype(np.int16),
            "vstrict": ch["valid_strict"].astype(str).eq("True"), "vrelax": ch["valid_relaxed"].astype(str).eq("True"),
            "fill": ch["flag_fill"].astype(str).eq("True"),
            "mixcl": ch["dqa_mixed_clouds"].astype(np.int8), "shadow": ch["dqa_possible_shadow"].astype(np.int8),
            "adjcl": ch["dqa_adjacent_cloud"].astype(np.int8), "useful": ch["dqa_vi_usefulness"].astype(np.int8)}))
    print(f"read {f}: {sum(len(d) for d in out):,} rows, {time.time() - t0:.0f}s", flush=True)
df = pd.concat(out, ignore_index=True)
unmatched = int(df.pid.isna().sum())
df["pid"] = df["pid"].fillna(-1).astype(np.int32)
assert len(df) == 7_183_770, len(df)
assert len(meta) == 31_263 and unmatched == 0, (len(meta), unmatched)
df.to_pickle(W + "obs.pkl")
print(f"stage 01 done: {len(df):,} observations, {len(meta):,} event-cells, {time.time() - t0:.0f}s", flush=True)
