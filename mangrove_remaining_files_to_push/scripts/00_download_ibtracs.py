"""Stage 00. Download the IBTrACS v04r01 North Indian basin file from NOAA NCEI and compare its SHA-256 with the file
used for the published analysis (accessed 2026-10-03). IBTrACS is public domain (Gahtan et al., 2024, doi:10.25921/82ty-9e16).
"""
import hashlib, sys, urllib.request
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from p4.config import IBTRACS_NI, IBTRACS_URL, IBTRACS_SHA256

IBTRACS_NI.parent.mkdir(parents=True, exist_ok=True)
if not IBTRACS_NI.exists():
    print("downloading", IBTRACS_URL)
    urllib.request.urlretrieve(IBTRACS_URL, IBTRACS_NI)
sha = hashlib.sha256(IBTRACS_NI.read_bytes()).hexdigest()
if sha == IBTRACS_SHA256:
    print("IBTrACS file identical to the published analysis:", sha)
else:
    print("WARNING: IBTrACS file differs from the published analysis (NOAA revises IBTrACS in place).\n"
          f"  found    {sha}\n  expected {IBTRACS_SHA256}\n"
          "  Storm contamination flags may then differ. Run with P4_STORM_FLAGS=released to use the flags released\n"
          "  with the Mendeley dataset, which reproduces the published estimates exactly.")
