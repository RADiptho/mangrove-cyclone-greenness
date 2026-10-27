"""Paths and run-wide settings. All locations are configurable through environment variables; no local paths are hard-coded."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Folder "DATA" of the Mendeley dataset (https://doi.org/10.17632/r32hyd39dj.1) after extraction
DATA = Path(os.environ.get("P4_DATA", ROOT / "data" / "mendeley" / "DATA"))
# IBTrACS v04r01 North Indian basin file (public domain, NOAA NCEI)
IBTRACS_NI = Path(os.environ.get("P4_IBTRACS_NI", ROOT / "data" / "ibtracs" / "ibtracs.NI.list.v04r01.csv"))
IBTRACS_URL = "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.NI.list.v04r01.csv"
# SHA-256 of the IBTrACS file used for the published analysis (accessed 2026-10-03)
IBTRACS_SHA256 = "9f9734cf416f393649bd2a3ddeee82545470fe4d8c58b4a4a9aa1824818e9c6a"
WORK = Path(os.environ.get("P4_WORK", ROOT / "work"))            # intermediate caches
OUT = Path(os.environ.get("P4_OUT", ROOT / "outputs"))           # tables and figures
# "ibtracs" recomputes storm contamination and censoring from IBTrACS (default);
# "released" uses the flags released in the Mendeley panel (for IBTrACS files revised after 2026-10-03)
STORM_FLAGS = os.environ.get("P4_STORM_FLAGS", "ibtracs")
SCREEN_SEASONS = (2000, 2024)                                     # Bay of Bengal screening frame

W = str(WORK) + os.sep
O = str(OUT) + os.sep
T = O + "tables" + os.sep
FIG = O + "figures" + os.sep
FDT = T + "figure_data_tables" + os.sep
SUP = T + "supplementary" + os.sep
for d in (W, T, FIG, FDT, SUP):
    os.makedirs(d, exist_ok=True)

def data(rel):
    return str(DATA / rel)
