# Feature Schema – features_v1.csv

- Target: target_anom_t_plus_1 (anomaly at t+1)
- Core features: mon_sin, mon_cos, anom_lag1, anom_lag12, roll_mean_3, roll_std_3
- Optional: anom_lag24, roll_mean_12
- Warm-up removal: drop rows with NA in core features and target
