#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

def main():
    ap = argparse.ArgumentParser(description="Phase 3 – QA checks on features_v1")
    ap.add_argument("--features_csv", required=True)
    ap.add_argument("--out_md", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.features_csv)
    req = {"country","anomaly_c","target_anom_t_plus_1","anom_lag1","anom_lag12","roll_mean_3","roll_std_3","mon_sin","mon_cos"}
    miss = [c for c in req if c not in df.columns]
    if miss: raise SystemExit(f"Missing columns: {miss}")

    # correlations
    rows = []
    for country, g in df.groupby("country"):
        g = g[["target_anom_t_plus_1","anom_lag1"]].dropna()
        if len(g) >= 12:
            rows.append({"country": country, "corr_target_vs_lag1": float(g.corr().iloc[0,1])})
    corr_df = pd.DataFrame(rows).sort_values("corr_target_vs_lag1", ascending=False)

    # na shares
    na = df.isna().mean().sort_values(ascending=False).reset_index()
    na.columns = ["column","share_na"]

    # write md
    lines = []
    lines.append(f"# Phase 3 – QA checks")
    lines.append(f"Date: {datetime.utcnow().isoformat()}Z")
    lines.append("")
    lines.append(f"- Rows: **{len(df)}**, Countries: **{df['country'].nunique()}**")
    lines.append("")
    lines.append("## Correlation target vs anom_lag1 (top 10)")
    lines.append("| country | corr |")
    lines.append("|---|---:|")
    for _, r in corr_df.head(10).iterrows():
        lines.append(f"| {r['country']} | {r['corr_target_vs_lag1']:.3f} |")
    lines.append("")
    lines.append("## Missing values (top 10)")
    lines.append("| column | share_na |")
    lines.append("|---|---:|")
    for _, r in na.head(10).iterrows():
        lines.append(f"| {r['column']} | {r['share_na']:.3%} |")
    Path(args.out_md).write_text("\n".join(lines), encoding="utf-8")
    print("[OK] QA md written:", args.out_md)

if __name__ == "__main__":
    main()
