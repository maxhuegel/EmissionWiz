#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import pandas as pd
import json

REQUIRED = ["country","year","month","temp_c","clim_temp_c","anomaly_c"]

def parse_map(kvs: list[str]):
    mapping = {}
    for kv in kvs or []:
        if "=" not in kv:
            raise SystemExit(f"Bad --map entry: {kv}. Expected form dst=src")
        dst, src = kv.split("=", 1)
        mapping[dst.strip()] = src.strip()
    return mapping

def main():
    ap = argparse.ArgumentParser(description="Normalize anomalies CSV to required schema for Phase 3/4.")
    ap.add_argument("--in_csv", required=True, help="input anomalies CSV")
    ap.add_argument("--out_csv", required=True, help="output normalized CSV")
    ap.add_argument("--map", nargs="*", help="column mappings in form dst=src, e.g., country=Country month=Month year=Year temp_c=TempC")
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    mapping = parse_map(args.map)

    # default: identity mapping
    for col in REQUIRED:
        if col not in df.columns:
            src = mapping.get(col)
            if not src or src not in df.columns:
                raise SystemExit(f"Missing column '{col}'. Provide mapping with --map {col}=<existing_col>")
            df[col] = df[src]

    out = df[REQUIRED + [c for c in df.columns if c not in REQUIRED]]
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print("[OK] Wrote normalized anomalies:", args.out_csv)

if __name__ == "__main__":
    main()
