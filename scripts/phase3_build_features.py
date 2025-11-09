#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

def load_df(path: Path)->pd.DataFrame:
    if path.suffix.lower()==".csv": return pd.read_csv(path)
    if path.suffix.lower()==".parquet": return pd.read_parquet(path)
    raise SystemExit(f"Unsupported file: {path}")

def add_calendar(df: pd.DataFrame)->pd.DataFrame:
    d = df.copy()
    d["mon_sin"] = np.sin(2*np.pi*d["month"].astype(float)/12.0)
    d["mon_cos"] = np.cos(2*np.pi*d["month"].astype(float)/12.0)
    return d

def add_persistence(df: pd.DataFrame)->pd.DataFrame:
    def f(g: pd.DataFrame)->pd.DataFrame:
        g = g.sort_values(["year","month"]).copy()
        g["anom_lag1"] = g["anomaly_c"].shift(1)
        g["anom_lag12"] = g["anomaly_c"].shift(12)
        g["anom_lag24"] = g["anomaly_c"].shift(24)
        g["roll_mean_3"] = g["anomaly_c"].shift(1).rolling(3, min_periods=3).mean()
        g["roll_std_3"]  = g["anomaly_c"].shift(1).rolling(3, min_periods=3).std(ddof=0)
        g["roll_mean_12"] = g["anomaly_c"].shift(1).rolling(12, min_periods=12).mean()
        return g
    return df.groupby("country", group_keys=False).apply(f)

def add_target(df: pd.DataFrame)->pd.DataFrame:
    def f(g: pd.DataFrame)->pd.DataFrame:
        g = g.sort_values(["year","month"]).copy()
        g["target_anom_t_plus_1"] = g["anomaly_c"].shift(-1)
        return g
    return df.groupby("country", group_keys=False).apply(f)

def main():
    ap = argparse.ArgumentParser(description="Phase 3 – Build features_v1 (leakage-free).")
    ap.add_argument("--anomalies", required=True)
    ap.add_argument("--out_features", required=True)
    ap.add_argument("--drop_optional", action="store_true")
    args = ap.parse_args()

    df = load_df(Path(args.anomalies))
    req = {"country","year","month","temp_c","clim_temp_c","anomaly_c"}
    miss = [c for c in req if c not in df.columns]
    if miss: raise SystemExit(f"Missing columns: {miss}")

    d = add_calendar(df)
    d = add_persistence(d)
    d = add_target(d)

    core = ["anom_lag1","anom_lag12","roll_mean_3","roll_std_3","target_anom_t_plus_1"]
    base_cols = ["country","year","month","temp_c","clim_temp_c","anomaly_c","mon_sin","mon_cos","anom_lag1","anom_lag12","roll_mean_3","roll_std_3","target_anom_t_plus_1"]
    opt_cols = ["anom_lag24","roll_mean_12"]
    cols = base_cols if args.drop_optional else base_cols + opt_cols

    out = d[cols].dropna(subset=core, how="any").copy()
    Path(args.out_features).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_features, index=False)
    print("[OK] features_v1 written:", args.out_features, "rows:", len(out))

if __name__ == "__main__":
    main()
