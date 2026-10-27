# Mangrove–Cyclone Greenness Responses in the Bay of Bengal

A reproducible research workflow for quantifying event-specific mangrove greenness responses to tropical cyclones across wind-exposure classes in the Bay of Bengal.

The project integrates IBTrACS v04r01 quadrant wind radii, Global Mangrove Watch v3 mangrove extent, and quality-decoded MODIS Collection 6.1 Terra/Aqua NDVI and EVI observations for eleven tropical cyclones from 2007–2024.

## Cyclone events

Sidr (2007), Nargis (2008), Aila (2009), Fani (2019), Bulbul (2019), Amphan (2020), Yaas (2021), Sitrang (2022), Mocha (2023), Remal (2024), and Dana (2024).

## Workflow

The pipeline covers wind-exposure classification, MODIS quality control, seasonal baseline estimation, event-aligned greenness anomalies, exposed–control matching, recovery analysis, uncertainty and robustness checks, and exploratory explainability analyses.

Strict-QA harmonic baselines require at least 24 valid pre-landfall observations under the QA rule in use.

## Data

The released event-aligned dataset is available from Mendeley Data:

https://doi.org/10.17632/r32hyd39dj.1

Additional source data:
- IBTrACS v04r01 — NOAA/NCEI
- Global Mangrove Watch v3
- MOD13Q1 and MYD13Q1 Collection 6.1 — NASA MODIS

Large/raw datasets are intentionally not stored in this repository.

## Setup

```bash
git clone https://github.com/RADiptho/mangrove-cyclone-greenness.git
cd mangrove-cyclone-greenness
python -m venv .venv
pip install -r requirements.txt
```

Place the released dataset in the local `data/` directory following `data/README.md`, then run the numbered scripts in `scripts/`.

## Repository structure

- `p4/` — shared configuration and analysis utilities
- `scripts/` — reproducible processing and analysis stages
- `reference/` — compact verification/reference values
- `data/` — local-data instructions only
- `requirements.txt` — Python dependencies

## License

Code is released under the MIT License. Data remain subject to the licences and terms of their respective providers.
