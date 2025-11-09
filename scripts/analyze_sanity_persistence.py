#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse, json
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

SUPPORTED = {".csv", ".parquet"}

def load_any(path: Path) -> pd.DataFrame:
    sfx = path.suffix.lower()
    if sfx == ".csv":
        return pd.read_csv(path)
    if sfx == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file: {path}")

def month_order_key(row):
    return int(row["year"]) * 12 + int(row["month"])

def lag_autocorr(series: pd.Series, lag: int) -> float | None:
    try:
        return float(series.autocorr(lag))
    except Exception:
        return None

def linear_trend_slope(series: pd.Series, months_to_decade: bool = True) -> float | None:
    # fit y = a + b*t on index t = 0..n-1; return b in °C/decade if months_to_decade
    y = series.values
    n = len(y)
    if n < 3 or np.isnan(y).all():
        return None
    t = np.arange(n, dtype=float)
    # mask NaNs
    mask = ~np.isnan(y)
    if mask.sum() < 3:
        return None
    t, y = t[mask], y[mask]
    b, a = np.polyfit(t, y, 1)  # slope, intercept (numpy returns [slope, intercept])
    if months_to_decade:
        b = b * 120.0  # months per decade
    return float(b)

def per_country_stats(df: pd.DataFrame, min_len: int = 24) -> pd.DataFrame:
    out_rows = []
    for country, g in df.groupby("country", sort=True):
        g = g.sort_values(["year","month"])
        s = g["anomaly_c"].astype(float)
        n = len(s)
        valid = int(s.notna().sum())
        ac_lag12 = lag_autocorr(s, 12) if n >= 13 else None
        slope_dec = linear_trend_slope(s) if n >= 12 else None
        mean_anom = float(np.nanmean(s)) if valid > 0 else None
        std_anom = float(np.nanstd(s, ddof=1)) if valid > 1 else None
        out_rows.append({
            "country": country,
            "n_rows": n,
            "n_valid_anomaly": valid,
            "autocorr_lag12": ac_lag12,
            "trend_decade_c": slope_dec,
            "mean_anomaly_c": mean_anom,
            "std_anomaly_c": std_anom
        })
    return pd.DataFrame(out_rows).sort_values("country")

def main():
    ap = argparse.ArgumentParser(description="Step 9: Sanity & persistence checks on monthly anomalies per country.")
    ap.add_argument("--anomalies", required=True, help="Path to data_clean/monthly_anomalies.(csv|parquet)")
    ap.add_argument("--report_csv", required=True, help="Output CSV with per-country stats")
    ap.add_argument("--report_json", required=True, help="Output JSON with meta and global summary")
    args = ap.parse_args()

    anomalies_path = Path(args.anomalies)
    df = load_any(anomalies_path)

    # required columns
    req = {"country","year","month","anomaly_c"}
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns in anomalies file: {missing}")

    # compute stats
    stats = per_country_stats(df)

    # write CSV
    Path(args.report_csv).parent.mkdir(parents=True, exist_ok=True)
    stats.to_csv(args.report_csv, index=False)

    # meta + global summary
    meta = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "anomalies": str(anomalies_path),
        "countries": int(stats["country"].nunique()),
        "rows_anomalies": int(len(df)),
        "global_summary": {
            "median_autocorr_lag12": float(stats["autocorr_lag12"].median(skipna=True)) if "autocorr_lag12" in stats else None,
            "median_trend_decade_c": float(stats["trend_decade_c"].median(skipna=True)) if "trend_decade_c" in stats else None,
            "share_autocorr_lag12_pos": float((stats["autocorr_lag12"] > 0).mean())
        }
    }
    with open(args.report_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("[OK] Step 9 report written:", args.report_csv)
    print(json.dumps(meta, indent=2))

if __name__ == "__main__":
    main()
