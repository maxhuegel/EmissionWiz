# QA Notes – Phase 1 (Final)

_Generated: 2025-11-09T01:14:13.963695Z_

## Checklist (Steps 5–10)
- [x] **Step 5 – Reference period set** (`reports/reference_periods.csv`)
- [x] **Step 6 – Climatology written** (`data_clean/monthly_climatology_1981_2010.*`)
- [x] **Step 6 – Anomalies written** (`data_clean/monthly_anomalies.*`)
- [x] **Step 7 – Month encoding policy fixed** (sin/cos)
- [x] **Step 8 – Lags/Rollings policy fixed** (past-only, anomalies)
- [x] **Step 9 – Sanity & persistence report** (`reports/sanity_persistence.*`)
- [x] **Step 10 – Phase‑1 consistency check** (`reports/phase1_consistency.*`)

## Key results
- Countries evaluated: **291**
- Climatology 12 months OK: **100.0%**
- Mean anomaly ≈ 0 within reference (all 12 months): **100.0%**
- Median lag‑12 autocorr (anomalies): **0.309**
- Share with positive lag‑12 autocorr: **99.7%**
- Median anomaly trend: **+0.089 °C/decade**

## Files recap
- Step 5: `reports/reference_periods.csv` / `.json`
- Step 6: `data_clean/monthly_climatology_1981_2010.*`, `data_clean/monthly_anomalies.*`
- Step 7: Policy in `docs/FEATURE_POLICY.md` & `docs/QA_POLICY_OUTLIERS.md`
- Step 8: Policy in `docs/FEATURE_POLICY.md` & `docs/QA_POLICY_OUTLIERS.md`
- Step 9: `reports/sanity_persistence.*`
- Step 10: `reports/phase1_consistency.*`

## Open points / notes
- <add any country‑specific notes or data caveats here>
