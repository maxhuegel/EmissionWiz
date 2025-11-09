#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

KEYS = ["country","year","month","cutoff_ym","horizon"]

def load_buckets(p):
    return json.load(open(p, "r", encoding="utf-8"))["buckets"]

def bucket_name(h, buckets):
    for b in buckets:
        if b["h_start"] <= h <= b["h_end"]:
            return b["name"]
    return "h_na"

def main():
    ap = argparse.ArgumentParser(description="Blend model forecasts with baselines (safe left-join + fallback).")
    ap.add_argument("--setup_json", required=True)
    ap.add_argument("--model_forecasts", required=True)
    ap.add_argument("--baseline_clim", required=True)
    ap.add_argument("--baseline_lag12", required=True)
    ap.add_argument("--out_forecasts", required=True)
    ap.add_argument("--buckets_to_opt", nargs="*", default=["h07_12","h13_24"])
    ap.add_argument("--w_min", type=float, default=0.0)
    ap.add_argument("--w_max", type=float, default=0.8)
    ap.add_argument("--grid_steps", type=int, default=41)
    args = ap.parse_args()

    buckets = load_buckets(args.setup_json)

    m = pd.read_csv(args.model_forecasts)
    c = pd.read_csv(args.baseline_clim)[KEYS+["pred_c"]].rename(columns={"pred_c":"pred_c_clim"})
    l = pd.read_csv(args.baseline_lag12)[KEYS+["pred_c"]].rename(columns={"pred_c":"pred_c_lag12"})

    # *** WICHTIG: LEFT JOIN auf das Modell, damit KEINE Modellzeilen verloren gehen ***
    df = (m
          .merge(c, on=KEYS, how="left")
          .merge(l, on=KEYS, how="left"))

    # Beste Baseline pro Zeile (nur, wenn vorhanden); sonst NaN
    if "truth_c" in df.columns:
        df["ae_clim"]  = (df["pred_c_clim"]  - df["truth_c"]).abs()
        df["ae_lag12"] = (df["pred_c_lag12"] - df["truth_c"]).abs()
        df["pred_c_base"] = np.where(
            df["ae_clim"].notna() & df["ae_lag12"].notna(),
            np.where(df["ae_clim"] <= df["ae_lag12"], df["pred_c_clim"], df["pred_c_lag12"]),
            np.where(df["pred_c_clim"].notna(), df["pred_c_clim"], df["pred_c_lag12"])
        )
    else:
        # Fallback ohne truth: nimm vorhandene Baseline (clim bevorzugt)
        df["pred_c_base"] = np.where(df["pred_c_clim"].notna(), df["pred_c_clim"], df["pred_c_lag12"])

    df["bucket"] = df["horizon"].apply(lambda h: bucket_name(int(h), buckets))

    # Gewichte pro Ziel-Bucket aus RMSE minimieren (nur dort, wo Base vorhanden ist)
    grid = np.linspace(args.w_min, args.w_max, num=args.grid_steps)
    best_w = {}
    for name in args.buckets_to_opt:
        sub = df[(df["bucket"]==name) & (df["pred_c_base"].notna())]
        if sub.empty:
            best_w[name] = 0.0
            continue
        y, mm, bb = sub["truth_c"].values, sub["pred_c"].values, sub["pred_c_base"].values
        best = (0.0, float("inf"))
        for w in grid:
            p = (1.0 - w)*mm + w*bb
            rmse = float(np.sqrt(np.mean((p - y)**2)))
            if rmse < best[1]:
                best = (w, rmse)
        best_w[name] = best[0]

    # Blend anwenden: nur dort, wo Base vorhanden; sonst bleibt Model-Pred unverändert
    df["blend_w"] = df["bucket"].map(lambda n: best_w.get(n, 0.0))
    df.loc[df["pred_c_base"].isna(), "blend_w"] = 0.0
    df["pred_c"] = (1.0 - df["blend_w"]) * df["pred_c"] + df["blend_w"] * df["pred_c_base"].fillna(0.0)

    # In ursprünglichem Modellschema speichern
    out_cols = list(m.columns)
    Path(args.out_forecasts).parent.mkdir(parents=True, exist_ok=True)
    df[out_cols].to_csv(args.out_forecasts, index=False)
    print("Optimized weights per bucket:", best_w)
    print("[OK] Blended forecasts written:", args.out_forecasts, "rows=", len(df))

if __name__ == "__main__":
    main()