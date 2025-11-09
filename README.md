# ClimateWiz

ClimateWiz is an interactive 3D globe (Streamlit + Globe.gl/Three.js) to explore yearly warming by country. It supports:
- Anomalies (ΔT) relative to the 1901–2029 country baseline
- Absolute annual temperatures (°C)
- Colorblind palette, PNG export, and a country info panel with a mini trend chart (1901–last year in the data)

Target: education & exploration—quick, intuitive views of climate change across countries and decades.

## Features
- Modes: Anomaly (ΔT) and Absolute (°C)
- Hover: country name; Click: info panel with
  - snapshot value for the currently selected year,
  - linear trend (°C/decade),
  - mini time-series chart (1901–last available year, e.g. 2029)
- Colorblind mode: alternate, daltonism-friendly palette
- Export PNG: save a screenshot of the globe
- Guide: in-app explanation (top-left button)

## Project Structure
```
EmissionWiz/
├─ src/
│ ├─ app/
│ │ └─ app.py # Streamlit app embedding Globe.gl (HTML/JS)
│ ├─ data/
│ │ └─ temperature/
│ │ └─ temp_per_country/
│ │ └─ yearly_temp_aggregated/
│ │ └─ country_year.csv # Annual per-country data (see schema)
└─ ...
```

## Data Schema
`src/data/temperature/temp_per_country/yearly_temp_aggregated/country_year.csv`

Required columns:
- `country` – country name (underscores or spaces are fine; normalized in-app)
- `year` – integer year
- `temp_c` – annual mean temperature (°C)
- `base` – country mean 1901–2029 (°C)
- `anom` – `temp_c - base` (°C)

Example:
```
country,year,temp_c,base,anom
Albania,1901,10.9,12.3,-1.4
Albania,1902,11.8,12.3,-0.5
...
```
The app harmonizes country names (e.g., `Bosnia and Herz.` → `Bosnia-Herzegovinia`) via a JS alias map.

---

## Installation (Conda)

### 1) Requirements
- Miniconda/Anaconda installed
- Optional: `git` if cloning the repo

### 2) Create & activate environment
Run in the project root (where `environment.yml` is located):
```bash
conda env create -f environment.yml
conda activate ClimateWiz
```

## Run
From the project root:
```
streamlit run src/app/app.py
```
Streamlit will open at `http://localhost:8501`. The app renders in fullscreen.

## How to Use

1) Choose Anomaly or Absolute in the top-right panel.

2) Drag the year slider to select a snapshot year.

3) Click a country to open the info panel (left):
   - snapshot value for the selected year,
   - linear trend (°C/decade) computed over 1901–last available year,
   - mini chart with axes and units.

4) Toggle Colorblind: ON/OFF for an alternative palette.

5) Export PNG saves a screenshot of the globe.

## Years & Slider Behavior

 - The app loads all years present in `country_year.csv` (e.g., 1901–2029 including projections).

- The slider default position is deliberately set to 2024 (or, if absent, the latest available year).
This is controlled in the JS section of app.py:

```js
const START_YEAR = '2024';
```
 - The maximum slider year is computed from the last year in the CSV.
If the slider stops at 2024, the CSV likely does not contain later years or the configured CSV path is wrong.


## Configuration

In src/app/app.py:
- CSV path: `DATA_CSV`
- Anomaly color range: `ANOM_CLIP = (-3.0, 3.0)`
- Slider default year: `const START_YEAR = '2024'` (JS)
- Country aliases: `ALIASES` (JS)
- Color schemes: handled in `colorScaleFactory` / `setGradient` (JS)

## Extending

- More years / projections: just extend `country_year.csv`; the app adapts automatically.
- Additional metrics: add new CSV columns and mirror the `anom/abs` pattern in the payload and JS.
- Search / autoplay: the panel design supports extra controls if you want to add them.

## Troubleshooting

- Slider capped at 2024
  - Ensure the CSV actually contains years > 2024; verify `DATA_CSV` path.

- “Missing columns”
  - CSV must include `country, year, temp_c, base, anom.`

- Country appears gray
  - No data for that country or name mismatch—update the `ALIASES` map if needed.

- Layout/frame issues
  - Clear Streamlit cache (`st.cache_data.clear()`), hard-refresh the browser.

## Methodology
- Source: country-aggregated annual means (e.g., CRU TS v4.x based preparation).
- Anomalies: `year_value − mean(1901–2029)` per country.
- Aggregation: monthly → annual means; countries require sufficient monthly coverage.
- Country harmonization: alias map; some small/disputed territories may be excluded.

## Data Source (CRU CY v4.09 – TMP)

We use **CRU CY v4.09 Country Averages: TMP** from the University of East Anglia’s Climatic Research Unit (CRU):
`https://crudata.uea.ac.uk/cru/data/hrg/cru_ts_4.09/crucy.2503061057.v4.09/countries/tmp/`

* **What it is:** Country-level, area-weighted **monthly mean temperature (°C)** series derived from CRU TS v4.09.
* **Coverage:** **1901–2024** (per CRU TS v4.09).
* **Format:** One `.per` time series file per country.
* **How we use it:** Parse monthly series, compute anomalies vs **1991–2020**, then aggregate to annual means for the app and model training.

**Licence & attribution:** Open Government Licence. Please acknowledge **CRU (University of East Anglia)** and **NCAS**.

## Architecture & Tools

### Overview

![ALT_TEXT](docs/figs/architecture_overview.png)


ClimateWiz transforms **monthly country temperatures** (~1901–latest data year) into **anomalies relative to the 1991–2020 baseline** and produces **ML forecasts** (monthly, then aggregated to **annual means**). The Streamlit app (embedding Globe.gl/Three.js) visualizes **Anomaly (ΔT)** and **Absolute (°C)**, with a country panel showing **trend (°C/decade)** and a mini time series. For stable multi-step forecasts we use **damping**, **clipping**, and **climatology blending**.

### Dataflow (folders & key scripts)

* **`src/data/temperature/dataset_temp/`** – Raw monthly, country-level data (~1901–…).
* **Phase 1–2: Cleaning & Baseline**

  * `scripts/compute_climatology_anomalies.py` – Monthly climatology & anomalies.
  * `scripts/define_reference_period.py` – Set **1991–2020** reference period.
  * **Output:** `data_clean/*.csv`.
* **Phase 3: Features & Folds**

  * `scripts/phase3_build_features.py` – Features (sin/cos seasonality, lags 1/12/24, rolling stats 3/12, optional climatology term).
  * `scripts/phase3_make_folds.py` – Time-aware CV folds (TimeSeriesSplit).
  * **Output:** `features/features_v*.csv`, `reports/phase3_folds.csv`.
* **Phase 4: Modeling & Metrics**

  * `scripts/phase4_train_ridge.py` – Ridge (α grid), standardization, **recursive** H-step forecasting; **damping**, **clipping**, **climatology blend** (horizon-dependent).
  * `scripts/phase4_metrics.py` – Country/global metrics.
  * **Baselines:** `baselines/*`.
* **Phase 4–5: Post-processing & App Payload**

  * `scripts/phase4_blend_with_baselines.py`, `scripts/phase5_apply_forecasts_to_country_files.py`.
  * **Final artifacts:** monthly forecasts + annual aggregates → `models/*`, `reports/*`; app tables such as `src/data/temperature/temp_per_country/yearly_temp_aggregated/country_year.csv`.

### App Architecture (UI)

![ALT_TEXT](docs/figs/app_architecture.png)

* **Frontend:** Streamlit (`src/app/app.py`) with `streamlit.components.v1.html` embedding **Globe.gl/Three.js**.
* **Interaction:** Mode toggle (ΔT/°C), year slider, hover tooltips, click → country panel (snapshot, °C/decade trend, mini chart), **colorblind palette**, PNG export, in-app Guide.
* **Data access:** Cached CSV payloads for fast first paint; versioned artifacts.

### Core Modeling

* **Targets:** monthly **anomalies**; **annual** values = mean of 12 monthly predictions.
* **Features (examples):**
  `sin/cos(2π·month/12)`, `anom_lag_1/12/24`, rolling mean/stdev (3/12), optional **climatology**.
* **Training:** `TimeSeriesSplit`, `StandardScaler`, Ridge α-grid.
* **Forecasting:** recursive 1..H with **damping** and **climatology blending** (horizon-weighted model↔climatology), optional **clipping**.
* **Outputs:** per-country monthly series, blended series, annual aggregates; consistent CSV payloads for the frontend.

### Tools & Environment

* **Python 3.11**, **Conda** (`environment.yml`)

  * Data: `pandas`, `numpy`
  * ML: `scikit-learn` (Ridge, StandardScaler, TimeSeriesSplit)
  * App: `streamlit` + Globe.gl/Three.js embed
  * Tests/CI: `pytest` (basic)

### Reproducibility & Project Layout

```
ClimateWiz/
├─ src/
│  ├─ app/                         # Streamlit + Globe.gl embed
│  └─ data/temperature/...         # Raw & app data (CSV)
├─ scripts/                        # Phase 1–5 pipeline
├─ data_clean/                     # Cleaned tables
├─ features/                       # Features & folds
├─ baselines/                      # Climatology/lag baselines
├─ models/                         # Forecasts & blends
├─ reports/                        # Folds, metrics, payload
└─ environment.yml                 # Reproducible environment
```

### Deployment

* **Local:**
  `conda env create -f environment.yml && conda activate ClimateWiz`
  `streamlit run src/app/app.py`
* **Server:** same setup; provide CSV artifacts (`src/data/...`, `features/`, `reports/`, `models/`).
* **Performance:** app-level caching, compact CSVs.
