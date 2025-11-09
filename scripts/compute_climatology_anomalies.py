#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse, json
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

SUPPORTED = {".csv", ".parquet", ".feather"}

def load_any(path: Path) -> pd.DataFrame:
    sfx = path.suffix.lower()
    if sfx == ".csv":
        return pd.read_csv(path)
    if sfx == ".parquet":
        return pd.read_parquet(path)
    if sfx == ".feather":
        return pd.read_feather(path)
    raise ValueError(f"Unsupported file: {path}")

def ensure_cols(df: pd.DataFrame, src: Path) -> pd.DataFrame:
    df = df.copy()
    if "country" not in df.columns:
        df["country"] = src.stem
    if ("year" not in df.columns or "month" not in df.columns) and "date" in df.columns:
        dt = pd.to_datetime(df["date"], errors="coerce")
        if "year" not in df.columns: df["year"] = dt.dt.year
        if "month" not in df.columns: df["month"] = dt.dt.month
    for req in ["year","month","temp_c"]:
        if req not in df.columns:
            raise SystemExit(f"{src.name}: missing required column '{req}'")
    # sanitize
    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)
    return df[["country","year","month","temp_c"]]

def read_per_country(input_dir: Path) -> pd.DataFrame:
    files = [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED]
    if not files:
        raise SystemExit(f"No data files found in {input_dir}")
    frames = []
    for p in sorted(files):
        frames.append(ensure_cols(load_any(p), p))
    return pd.concat(frames, ignore_index=True)

def read_reference(ref_csv: Path, default_start: int, default_end: int) -> pd.DataFrame:
    if ref_csv and ref_csv.exists():
        ref = pd.read_csv(ref_csv)
        # columns expected: country, chosen_start, chosen_end, fallback_used, ok_full_coverage, default_start, default_end
        # fallback: if chosen_* missing, use default_*
        ref["ref_start"] = ref["chosen_start"].fillna(ref.get("default_start", default_start))
        ref["ref_end"] = ref["chosen_end"].fillna(ref.get("default_end", default_end))
        return ref[["country","ref_start","ref_end"]]
    # no ref file -> single default for all
    return None

def compute_climatology(df: pd.DataFrame, ref: pd.DataFrame|None, default_start: int, default_end: int) -> pd.DataFrame:
    if ref is not None:
        dfm = df.merge(ref, on="country", how="left")
        dfm["ref_start"] = dfm["ref_start"].fillna(default_start)
        dfm["ref_end"] = dfm["ref_end"].fillna(default_end)
    else:
        dfm = df.copy()
        dfm["ref_start"] = default_start
        dfm["ref_end"] = default_end
    in_ref = (dfm["year"] >= dfm["ref_start"]) & (dfm["year"] <= dfm["ref_end"])
    ref_df = dfm.loc[in_ref, ["country","month","temp_c","ref_start","ref_end"]]
    clim = (ref_df
            .groupby(["country","month","ref_start","ref_end"], as_index=False)["temp_c"]
            .mean()
            .rename(columns={"temp_c":"clim_temp_c"}))
    return clim

def compute_anomalies(df: pd.DataFrame, clim: pd.DataFrame) -> pd.DataFrame:
    out = df.merge(clim[["country","month","clim_temp_c"]], on=["country","month"], how="left")
    out["anomaly_c"] = out["temp_c"] - out["clim_temp_c"]
    return out

def main():
    ap = argparse.ArgumentParser(description="Compute monthly climatology and anomalies (Step 6).")
    ap.add_argument("--input_dir", required=True, help="Folder with one file per country; files must contain country/year/month/temp_c (date optional).")
    ap.add_argument("--output_climatology", required=True, help="Output file (.csv or .parquet).")
    ap.add_argument("--output_anomalies", required=True, help="Output file (.csv or .parquet).")
    ap.add_argument("--ref_csv", default=None, help="CSV from Step 5 with chosen reference periods (reports/reference_periods.csv). If not provided, uses default window for all countries.")
    ap.add_argument("--default_start", type=int, default=1981, help="Default reference start year (inclusive).")
    ap.add_argument("--default_end", type=int, default=2010, help="Default reference end year (inclusive).")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    df = read_per_country(input_dir)

    ref = Path(args.ref_csv) if args.ref_csv else None
    ref_df = read_reference(ref, args.default_start, args.default_end) if ref else None

    clim = compute_climatology(df, ref_df, args.default_start, args.default_end)
    # Save climatology
    Path(args.output_climatology).parent.mkdir(parents=True, exist_ok=True)
    if args.output_climatology.lower().endswith(".parquet"):
        clim.to_parquet(args.output_climatology, index=False)
    else:
        clim.to_csv(args.output_climatology, index=False)

    # Anomalies
    anomalies = compute_anomalies(df, clim)
    Path(args.output_anomalies).parent.mkdir(parents=True, exist_ok=True)
    if args.output_anomalies.lower().endswith(".parquet"):
        anomalies.to_parquet(args.output_anomalies, index=False)
    else:
        anomalies.to_csv(args.output_anomalies, index=False)

    meta = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "input_dir": str(input_dir),
        "ref_source": str(ref) if ref else None,
        "default_window": [args.default_start, args.default_end],
        "rows_input": int(len(df)),
        "rows_anomalies": int(len(anomalies)),
        "countries": int(df["country"].nunique()),
    }
    print(json.dumps(meta, indent=2))

if __name__ == "__main__":
    main()
