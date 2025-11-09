# Phase 4 – Model training (Ridge) & backtesting

- Train per-country Ridge regression on anomaly target (t+1) with Phase-3 features.
- Rolling-origin evaluation over cutoffs; recursive horizons 1..HMAX (from phase2_setup.json).
- Compare against climatology and lag12 baselines in identical buckets.
- Outputs: models/forecasts_model_ridge.csv, models/metrics_*.csv, reports/phase4_*.md