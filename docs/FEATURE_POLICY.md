# Feature Policy (Phase 1 → Phase 3 handoff)

This file records feature decisions made during Phase 1.

## Monthly seasonality (Step 7)
Represent month `m ∈ {1..12}` as **cyclical**:
- `mon_sin = sin(2π·m/12)`
- `mon_cos = cos(2π·m/12)`

## Persistence (Step 8 – policy only)
- Lags on **anomalies**: `anom_lag1`, `anom_lag12` (optional `anom_lag24`).
- Rolling stats on **anomalies**: `roll_mean_3`, `roll_std_3` (optional `roll_mean_12`).
- **No leakage**: compute strictly from **past** values of the same country.

## Static geo/context (if available)
- `lat`, `lon`, `elevation`, `region` (categorical).

## Persistence features (Step 8 – final policy)

All persistence features are computed **on anomalies** (not raw °C) and **strictly from the past** of the same country (no leakage). Naming is fixed to keep training/eval consistent.

### Lags (per country, per monthly time index)
- `anom_lag1`  = anomaly at t-1 (previous month)
- `anom_lag12` = anomaly at t-12 (same month, previous year)
- `anom_lag24` = anomaly at t-24 (optional; switches on if data length allows)

**Edge handling:** If not enough history exists, the lag is **NA**. Downstream code may drop early rows or impute conservatively (median 0.0 for anomalies is acceptable for small shares, but prefer dropping where feasible).

### Rolling statistics (past-only windows, excluding current t)
- `roll_mean_3` = mean of anomalies over {t-1, t-2, t-3}
- `roll_std_3`  = std  of anomalies over {t-1, t-2, t-3}
- `roll_mean_12` (optional) = mean over {t-1 … t-12}

**Edge handling:** If the full window is not available, the rolling value is **NA** (no partial windows).

### Leakage rules (must)
1) **Past-only:** Lags/Rollings use **only** indices < t.
2) **Same country:** Never pool over future or other countries’ future values.
3) **Split-aware transforms:** Any scaler/encoder fitted later is trained **only** on the training fold in time, then applied to val/test.

### Implementation note (Phase 3)
- Features will be generated from `data_clean/monthly_anomalies.*` joined by `country, year, month`.
- Cyclical month encoding (`mon_sin`, `mon_cos`) from Step 7 is always included alongside persistence features.
