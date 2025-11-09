#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Append the next 60 future months per country (immediately after the last month in each file)
using the latest cutoff from a forecasts CSV.

Input schema expected in forecasts:
  country, year, month, cutoff_ym, horizon, pred_c

Country file schema (as in your repo):
  date, year, month, temp_c, country
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

REQ = ["country","year","month","cutoff_ym","horizon","pred_c"]

def norm(s:str)->str: return str(s).strip()
def ym_key(y:int,m:int)->int: return int(y)*12 + (int(m)-1)
def key_to_ym(k:int)->tuple[int,int]: return k//12, (k%12)+1
def midmonth(y:int,m:int)->str: return f"{int(y):04d}-{int(m):02d}-15"

def latest_cutoff(df: pd.DataFrame)->str:
    def k(s): y,m = s.split("-"); return int(y)*12 + int(m) - 1
    return sorted(df["cutoff_ym"].unique(), key=k)[-1]

def main():
    ap = argparse.ArgumentParser(description="Append next 60 months per country after the last date in each file.")
    ap.add_argument("--country_dir", required=True, help="src/data/temperature/temp_per_country")
    ap.add_argument("--forecasts", required=True, help="models/forecasts_model_ridge_PROD_full.csv (or similar)")
    ap.add_argument("--out_dir", required=True, help="output directory for updated country CSVs")
    ap.add_argument("--allow_overwrite", action="store_true",
                    help="if set, replace existing months; otherwise only append truly new months")
    args = ap.parse_args()

    cdir = Path(args.country_dir)
    outdir = Path(args.out_dir); outdir.mkdir(parents=True, exist_ok=True)

    F = pd.read_csv(args.forecasts)
    miss = [c for c in REQ if c not in F.columns]
    if miss:
        raise ValueError(f"Forecasts missing columns: {miss}")

    # normalize forecasts
    F["country"] = F["country"].map(norm)
    F["year"]    = F["year"].astype(int)
    F["month"]   = F["month"].astype(int)

    # use latest cutoff, horizons 1..60 (falls dein File >60 Horizonte enthält, ist das ok)
    lc = latest_cutoff(F)
    F = F[F["cutoff_ym"] == lc].copy()
    if F.empty:
        raise ValueError("No rows at latest cutoff in forecasts.")
    F["k"] = F.apply(lambda r: ym_key(r["year"], r["month"]), axis=1)

    F_idx = F.set_index(["country","k"]).sort_index()

    total_added = 0
    for p in sorted(cdir.glob("*.csv")):
        try:
            df = pd.read_csv(p)
        except Exception as e:
            print(f"[WARN] cannot read {p.name}: {e}")
            continue

        if not set(["date","year","month","temp_c","country"]).issubset(df.columns):
            print(f"[WARN] {p.name}: unexpected schema; skipping.")
            continue

        file_country = norm(df["country"].iloc[0]) if not df.empty else p.stem
        # last existing (y,m) in this country file
        if df.empty:
            last_k = -1
        else:
            df["year"] = df["year"].astype(int)
            df["month"] = df["month"].astype(int)
            last_k = int(df.apply(lambda r: ym_key(r["year"], r["month"]), axis=1).max())

        # get all forecast months strictly after last_k
        try:
            sub = F_idx.loc[file_country]
            if isinstance(sub, pd.Series):
                sub = sub.to_frame().T
        except KeyError:
            print(f"[INFO] {p.name}: no forecasts for country='{file_country}' at cutoff {lc}")
            sub = pd.DataFrame(columns=F_idx.columns)

        if sub.empty:
            added = 0
            out = df
        else:
            sub = sub.reset_index()  # has columns: k, pred_c
            sub = sub[sub["k"] > last_k].sort_values("k")

            # optional: if we only want the next 60 calendar months, take head(60)
            sub = sub.head(60).copy()

            if sub.empty:
                added = 0
                out = df
            else:
                # build target rows
                rows = []
                for _, r in sub.iterrows():
                    y, m = key_to_ym(int(r["k"]))
                    rows.append({
                        "date":   midmonth(y,m),
                        "year":   int(y),
                        "month":  int(m),
                        "temp_c": float(r["pred_c"]),
                        "country": file_country
                    })
                add = pd.DataFrame(rows)

                if args.allow_overwrite:
                    # drop existing (y,m) to replace with forecast
                    idx_existing = set(zip(df["year"].astype(int), df["month"].astype(int)))
                    idx_add = set(zip(add["year"], add["month"]))
                    to_drop = list(idx_existing & idx_add)
                    if to_drop:
                        mask = ~df.set_index(["year","month"]).index.isin(to_drop)
                        df = df[mask].reset_index(drop=True)

                out = pd.concat([df, add], ignore_index=True)
                added = len(add)

        out.to_csv(outdir / p.name, index=False)
        total_added += added
        print(f"[OK] {p.name}: last_k={last_k} +{added} rows (cutoff {lc})")

    print(f"[DONE] total added rows: {total_added}")

if __name__ == "__main__":
    main()