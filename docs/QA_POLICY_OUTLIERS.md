# Outlier Policy (Phase 1 – Item 4)

## Purpose
Check outliers to prevent bias in climatology/anomalies and in lag/rolling features.

## Country identifier
We use the **English country name** in a column called `country`. If your data is stored as **one file per country** (e.g., in `src/data/tempPerCountry`), we will derive `country` from the filename.

## Inputs
Two modes are supported:
1) **Single file**: `--input` pointing to a CSV/Parquet/Feather with columns: `country`, `year`, `month`, `temp_c`.
2) **Directory of per‑country files**: `--input_dir` pointing to a folder (e.g., `src/data/tempPerCountry`) that contains one file per country. We infer `country` from the filename (without extension). Each file must contain at least `year`, `month`, `temp_c`.

## Definitions
- **Z-Score (per month type):** Standardization within the same calendar month (compare January only with January).
- **Robust Z:** Median and MAD (1.4826 * MAD as scale).

## Flags
- `flag_abs_range`: |temp_c| > 60 °C
- `flag_jump_gt15`: |temp_c - temp_{prev_month}| > 15 °C (within the same country)
- `flag_z_gt3`: |z| > 3 (classic)
- `flag_zrob_gt4`: |z_robust| > 4 (robust)
- `flag_any_outlier`: OR across all flags above

## Procedure
1. Ensure columns: `country`, `year`, `month`, `temp_c` (if directory mode is used, we add `country` from the filename).
2. Group by (`country`, `month`) and compute:
   - classic mean µ and std σ → `z`
   - median and MAD → `z_robust = (x - median) / (1.4826 * MAD)`
3. Sort per country chronologically and compute month-to-month differences → `flag_jump_gt15`.
4. Write enriched dataset + aggregation report.

## Removal vs. Retention
- **Do not delete automatically.** Only fix obvious typos.
- **Document** all flags and consider them during modelling (e.g., as weights or explicit exclusion rule depending on share).

## Parameters (recommended, adjustable)
- Thresholds: 60 °C, 15 °C, |z|>3, |z_robust|>4.
- Required columns: `country`, `year`, `month`, `temp_c`.
- Input formats: Parquet/CSV/Feather.

## Outputs
- `data_clean/monthly_with_outlier_flags.parquet|csv`
- `reports/outliers_summary.csv` (counts per country + share)
- `reports/outliers_summary.json` (metadata, parameters, timestamp)
## Usage

You can run the script either on a single consolidated file or on a directory that contains one file per country.

### Single file
Input file must contain these columns: `country, year, month, temp_c` (optionally `date`).

```
python scripts/qa_outliers.py   --input data_clean/monthly_clean.csv   --output data_clean/monthly_with_outlier_flags.csv   --summary_csv reports/outliers_summary.csv   --summary_json reports/outliers_summary.json
```

### Directory with one file per country
Each file must contain at least: `year, month, temp_c` and preferably `country`.  
If `country` is missing, it is derived from the **filename** (without extension).  
If `year`/`month` are missing but `date` exists, they are extracted from `date`.

```
python scripts/qa_outliers.py   --input_dir src/data/temperature/temp_per_country   --output data_clean/monthly_with_outlier_flags.csv   --summary_csv reports/outliers_summary.csv   --summary_json reports/outliers_summary.json
```

### Parameters (common)
- `--country_col`, `--year_col`, `--month_col`, `--temp_col` to override column names if needed.
- `--abs_temp_limit` (default 60.0), `--jump_threshold` (default 15.0),
  `--z_thresh` (default 3.0), `--zrob_thresh` (default 4.0).

### Outputs
- **Flagged dataset** (`--output`): original rows + z, z_robust and all boolean flags.
- **Summary CSV** (`--summary_csv`): per-country counts and percentages for each flag and overall.
- **Summary JSON** (`--summary_json`): metadata (parameters, timestamp, row counts).

### Notes
- Parquet requires `pyarrow` or `fastparquet`; CSV works out of the box.
- Run from the project root (`EmissionWiz/EmissionWiz`) so relative paths resolve.
- In PyCharm, set *Script path* to `scripts/qa_outliers.py` and *Working directory* to the project root.

## Reference period (Step 5)

We use a 30‑year **climate normal**. Default window: **1981–2010**. If a country lacks enough valid data in that window, choose the **closest 30‑year window** that maximizes completeness (per month).

**Run (per‑country directory input):**
```
python scripts/define_reference_period.py   --input_dir src/data/temperature/temp_per_country   --report_csv reports/reference_periods.csv   --report_json reports/reference_periods.json   --min_per_month 25   --default_start 1981 --default_end 2010
```
**Acceptance rule:** At least **25 valid years per month** (Jan..Dec) within the chosen 30‑year window. The script scans sliding windows if the default fails and picks the best one (tie‑break: closest center to 1981–2010).

## Climatology & anomalies (Step 6)

Compute monthly **climatology** per country/month using the chosen 30‑year reference, then subtract to get **anomalies**.

**Run:**
```
python scripts/compute_climatology_anomalies.py   --input_dir src/data/temperature/temp_per_country   --ref_csv reports/reference_periods.csv   --output_climatology data_clean/monthly_climatology_1981_2010.csv   --output_anomalies data_clean/monthly_anomalies.csv   --default_start 1981 --default_end 2010
```
- If `--ref_csv` is omitted, all countries use the default window (1981–2010).
- **Outputs**
  - Climatology: `country, month, clim_temp_c, ref_start, ref_end`
  - Anomalies: `country, year, month, temp_c, clim_temp_c, anomaly_c`
- **Checks**
  - Climatology has **12 rows per country** (months 1–12).
  - Mean of `anomaly_c` within the reference ≈ **0** per country×month.

## Month encoding policy (Step 7)

Treat months as **cyclical** so **December and January are neighbors**. Define two features for month `m ∈ {1..12}`:

- `mon_sin = sin(2π·m/12)`
- `mon_cos = cos(2π·m/12)`

**Guidelines**
- Do **not** use ordinal 1..12 or one‑hot alone for seasonality.
- Implementation happens during **Phase 3 (Feature Engineering)**. Step 7 is a **policy decision**, not a transform.

## Lags & rolling stats (Step 8 – policy)

- Compute **on anomalies**; not on raw °C.
- Lags: `anom_lag1`, `anom_lag12` (optional `anom_lag24`).
- Rolling (past-only): `roll_mean_3`, `roll_std_3` (optional `roll_mean_12`).
- **No leakage:** only past values of the same country; no symmetric windows including t or future.
- Edge handling: if history/window incomplete → set to NA (do not use partial windows).

## Sanity & persistence checks (Step 9)

Compute per-country diagnostics on `monthly_anomalies.*`:
- Lag‑12 autocorrelation of anomalies (persistence)
- Linear trend slope in °C/decade (on anomalies)
- Mean/Std of anomalies

**Run:**
```
python scripts/analyze_sanity_persistence.py   --anomalies data_clean/monthly_anomalies.csv   --report_csv reports/sanity_persistence.csv   --report_json reports/sanity_persistence.json
```

**Notes:**
- Positive lag‑12 autocorr is expected for seasonal persistence.
- Trend on anomalies indicates residual warming/cooling beyond the fixed climatology.
- Early months may lack history; this affects autocorr/trend length but not correctness.

## Final outputs & validation (Step 10)

Run a final consistency check across Phase‑1 outputs:

```
python scripts/validate_phase1_outputs.py   --monthly_clean data_clean/monthly_clean.csv   --climatology data_clean/monthly_climatology_1981_2010.csv   --anomalies data_clean/monthly_anomalies.csv   --reference_periods reports/reference_periods.csv   --outliers_summary reports/outliers_summary.csv   --sanity_persistence reports/sanity_persistence.csv   --report_csv reports/phase1_consistency.csv   --report_json reports/phase1_consistency.json
```

Checks per country:
- Climatology has **12 months**.
- Anomaly rows **match** monthly_clean rows (if monthly_clean provided).
- Mean anomaly within the reference window ≈ **0** for all 12 months (tolerance ±0.15 °C).
