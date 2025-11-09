#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
import numpy as np

def bucket_name(h: int, buckets: list[dict])->str:
    for b in buckets:
        if b["h_start"] <= h <= b["h_end"]:
            return b["name"]
    return "h_na"

def main():
    ap = argparse.ArgumentParser(description="Phase 4 – Metrics for model forecasts + comparison to baselines.")
    ap.add_argument("--setup_json", required=True)
    ap.add_argument("--model_forecasts", required=True)
    ap.add_argument("--baseline_clim", required=True)
    ap.add_argument("--baseline_lag12", required=True)
    ap.add_argument("--out_by_country", required=True)
    ap.add_argument("--out_global", required=True)
    ap.add_argument("--out_summary_md", required=True)
    ap.add_argument("--out_decision_md", required=True)
    args = ap.parse_args()

    with open(args.setup_json, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    buckets = cfg["buckets"]

    m = pd.read_csv(args.model_forecasts)
    b1 = pd.read_csv(args.baseline_clim).assign(baseline="climatology")
    b2 = pd.read_csv(args.baseline_lag12).assign(baseline="lag12")

    m["ae"] = (m["pred_c"] - m["truth_c"]).abs()
    m["se"] = (m["pred_c"] - m["truth_c"])**2
    m["bucket"] = m["horizon"].apply(lambda h: bucket_name(int(h), buckets))

    by_country = (m.groupby(["country","bucket"])
                    .agg(n=("ae","count"),
                         MAE=("ae","mean"),
                         RMSE=("se", lambda s: float(np.sqrt(s.mean()))))
                    .reset_index())
    by_country["who"] = "model_ridge"

    global_m = (by_country.groupby(["who","bucket"])
                  .agg(countries=("country","nunique"),
                       MAE=("MAE","mean"),
                       RMSE=("RMSE","mean"))
                  .reset_index())

    b = pd.concat([b1,b2], ignore_index=True)
    b["ae"] = (b["pred_c"] - b["truth_c"]).abs()
    b["se"] = (b["pred_c"] - b["truth_c"])**2
    b["bucket"] = b["horizon"].apply(lambda h: bucket_name(int(h), buckets))

    b_by_country = (b.groupby(["country","baseline","bucket"])
                      .agg(n=("ae","count"),
                           MAE=("ae","mean"),
                           RMSE=("se", lambda s: float(np.sqrt(s.mean()))))
                      .reset_index())

    b_global = (b_by_country.groupby(["baseline","bucket"])
                  .agg(countries=("country","nunique"),
                       MAE=("MAE","mean"),
                       RMSE=("RMSE","mean"))
                  .reset_index())

    Path(args.out_by_country).parent.mkdir(parents=True, exist_ok=True)
    by_country.to_csv(args.out_by_country, index=False)
    global_m.to_csv(args.out_global, index=False)

    # Summary MD
    topline = pd.concat([
        global_m.rename(columns={"who":"who"})[["who","bucket","countries","MAE","RMSE"]],
        b_global.rename(columns={"baseline":"who"})[["who","bucket","countries","MAE","RMSE"]]
    ], ignore_index=True).sort_values(["bucket","who"])

    lines = ["# Phase 4 – Summary", "", "## Global RMSE/MAE by bucket", "",
             "| bucket | who | countries | MAE | RMSE |", "|---|---|---:|---:|---:|"]
    for _, r in topline.iterrows():
        lines.append(f"| {r['bucket']} | {r['who']} | {int(r['countries'])} | {r['MAE']:.3f} | {r['RMSE']:.3f} |")
    Path(args.out_summary_md).write_text("\n".join(lines), encoding="utf-8")

    # Decision MD (wins vs. best baseline)
    model_c = by_country.rename(columns={"MAE":"MAE_model","RMSE":"RMSE_model"})[["country","bucket","MAE_model","RMSE_model"]]
    best_b = (b_by_country.sort_values(["country","bucket","RMSE"])
                .groupby(["country","bucket"]).head(1)
                .rename(columns={"baseline":"best_baseline","RMSE":"RMSE_best","MAE":"MAE_best"})
                [["country","bucket","best_baseline","RMSE_best","MAE_best"]])
    cmp = model_c.merge(best_b, on=["country","bucket"], how="inner")
    cmp["improvement_pct"] = (cmp["RMSE_best"] - cmp["RMSE_model"]) / cmp["RMSE_best"]
    wins = (cmp.groupby("bucket")["improvement_pct"].apply(lambda s: float((s>0).mean()))).reset_index()

    dec = ["# Phase 4 – Decision", "", "## Wins by country (model better than best baseline)", "",
           "| bucket | share_model_better |", "|---|---:|"]
    for _, r in wins.iterrows():
        dec.append(f"| {r['bucket']} | {r['improvement_pct']:.1%} |")
    dec += ["", "## Criteria (from Phase 2)",
            "- ≥70% countries better than both baselines in 1–12 and 13–24 months.",
            "- Global RMSE improvement ≥10–15% in 1–12 and 13–24 months.",
            "- No >10% regressions in >10% of countries."]

    Path(args.out_decision_md).write_text("\n".join(dec), encoding="utf-8")
    print("[OK] Wrote metrics & reports.")

if __name__ == "__main__":
    main()