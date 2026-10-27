# Event-specific mangrove greenness responses to tropical cyclones in the Bay of Bengal

Analysis code for the manuscript

> Diptho, R. A. and Ghosh, P. *Event specific mangrove greenness responses to tropical cyclones across wind exposure classes in the Bay of Bengal.* Submitted to *Discover Earth Observation* (Springer Nature).

The code reproduces every table, figure and number in the manuscript and its Supplementary Information from the released dataset:

> Event Aligned MODIS Dataset of Mangrove Greenness Responses to Eleven Tropical Cyclones in the Bay of Bengal from 2007 to 2024. Mendeley Data. https://doi.org/10.17632/r32hyd39dj.1

The study combines IBTrACS quadrant wind radii, Global Mangrove Watch v3 mangrove cells and quality-decoded MODIS Terra and Aqua NDVI and EVI for eleven cyclones (Sidr 2007, Nargis 2008, Aila 2009, Fani 2019, Bulbul 2019, Amphan 2020, Yaas 2021, Sitrang 2022, Mocha 2023, Remal 2024, Dana 2024).

## Reproduce the results

```sh
git clone https://github.com/RADiptho/mangrove-cyclone-greenness.git
cd mangrove-cyclone-greenness
python3.12 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
# place the Mendeley DATA folder at data/mendeley/DATA, then:
sh run_all.sh
```

## Data

No raw data are stored in this repository. Download the released dataset from https://doi.org/10.17632/r32hyd39dj.1. IBTrACS v04r01 North Indian basin is downloaded by `scripts/00_download_ibtracs.py`.

## Verification

`scripts/13_verify.py` verifies manuscript tables, quoted values, eleven-event coverage, the 2000–2024 storm screen, and manuscript figures. The final verified run passed 262/262 checks.

## Corrections implemented

* Strict-QA harmonic baselines require at least 24 valid pre-landfall observations under the QA rule in use. This removes the spurious Sitrang strict-QA outlier without changing any evidence grade.
* The storm screening frame is restricted to 2000–2024; corrected initial screening counts are 185, 100 and 87.

## Repository layout

`p4/` core configuration/functions; `scripts/` numbered reproducibility pipeline; `reference/` manuscript verification values; `data/` local input placeholders.

## License

Code: MIT License. Data: see the Mendeley record and upstream source terms.
