#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse, json
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

def load_any(path: Path) -> pd.DataFrame:
    sfx = path.suffix.lower()
    if sfx == ".csv":
        return pd.read_csv(path)
    if sfx == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file: {path}")

def main():
    ap = argparse.ArgumentParser(description="Phase 1 – Step 10: Validate final outputs and basic consistency.")
    ap.add_argument("--monthly_clean", required=False, help="data_clean/monthly_clean.(csv|parquet) with outlier flags")
    ap.add_argument("--climatology", required=True, help="data_clean/monthly_climatology_1981_2010.(csv|parquet)")
    ap.add_argument("--anomalies", required=True, help="data_clean/monthly_anomalies.(csv|parquet)")
    ap.add_argument("--reference_periods", required=False, help="reports/reference_periods.csv")
    ap.add_argument("--outliers_summary", required=False, help="reports/outliers_summary.csv")
    ap.add_argument("--sanity_persistence", required=False, help="reports/sanity_persistence.csv")
    ap.add_argument("--report_csv", required=True, help="Output CSV with per-country checks")
    ap.add_argument("--report_json", required=True, help="Output JSON with global summary")
    args = ap.parse_args()

    # Load required files
    clim = load_any(Path(args.climatology))
    anom = load_any(Path(args.anomalies))

    # Optional files
    ref_df = pd.read_csv(args.reference_periods) if args.reference_periods and Path(args.reference_periods).exists() else None
    outliers_df = pd.read_csv(args.outliers_summary) if args.outliers_summary and Path(args.outliers_summary).exists() else None
    sp_df = pd.read_csv(args.sanity_persistence) if args.sanity_persistence and Path(args.sanity_persistence).exists() else None
    clean_df = load_any(Path(args.monthly_clean)) if args.monthly_clean and Path(args.monthly_clean).exists() else None

    # Checks per country
    # 1) climatology has 12 rows per country
    c12 = clim.groupby("country")["month"].nunique().rename("clim_unique_months")
    # extract ref window per country from climatology if present
    if "ref_start" in clim.columns and "ref_end" in clim.columns:
        ref_info = clim.groupby("country")[["ref_start","ref_end"]].first()
    else:
        ref_info = pd.DataFrame(index=c12.index)
        ref_info["ref_start"] = np.nan
        ref_info["ref_end"] = np.nan

    # 2) anomalies rowcount per country (and mean≈0 inside ref window if ref info available)
    anom_counts = anom.groupby("country").size().rename("anomaly_rows")
    # join for per-country frame
    per_country = pd.concat([c12, anom_counts], axis=1)
    per_country = per_country.merge(ref_info, left_index=True, right_index=True, how="left").reset_index().rename(columns={"index":"country"})

    # mean≈0 within reference period by (country, month)
    def mean_zero_within_ref(anom: pd.DataFrame, clim: pd.DataFrame):
        if "ref_start" in clim.columns and "ref_end" in clim.columns:
            # build lookup per country
            ref_map = clim.groupby("country")[["ref_start","ref_end"]].first().to_dict(orient="index")
            flags = []
            for country, g in anom.groupby("country"):
                ref = ref_map.get(country, None)
                if not ref or pd.isna(ref["ref_start"]) or pd.isna(ref["ref_end"]):
                    flags.append({"country": country, "mean_anom_in_ref_ok": np.nan})
                    continue
                y0, y1 = int(ref["ref_start"]), int(ref["ref_end"])
                within = g[(g["year"]>=y0)&(g["year"]<=y1)]
                # per-month means
                pm = within.groupby("month")["anomaly_c"].mean()
                # accept if all months have |mean| < 0.15°C (tolerance)
                ok = bool((pm.abs() < 0.15).all()) if len(pm)==12 else False
                flags.append({"country": country, "mean_anom_in_ref_ok": ok})
            return pd.DataFrame(flags)
        else:
            return pd.DataFrame({"country": anom["country"].unique(), "mean_anom_in_ref_ok": np.nan})

    ref_ok = mean_zero_within_ref(anom, clim)
    per_country = per_country.merge(ref_ok, on="country", how="left")

    # 3) equality of anomalies rows vs monthly_clean rows (if available)
    if clean_df is not None:
        clean_counts = clean_df.groupby("country").size().rename("clean_rows")
        per_country = per_country.merge(clean_counts, left_on="country", right_index=True, how="left")
        per_country["rows_match_clean"] = per_country["clean_rows"].notna() & (per_country["clean_rows"] == per_country["anomaly_rows"])
    else:
        per_country["clean_rows"] = np.nan
        per_country["rows_match_clean"] = np.nan

    # Pass/Fail flags
    per_country["climatology_12_months_ok"] = per_country["clim_unique_months"] == 12

    # Write CSV
    out_csv = Path(args.report_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    per_country.to_csv(out_csv, index=False)

    # Global summary
    meta = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "climatology_file": args.climatology,
        "anomalies_file": args.anomalies,
        "monthly_clean_file": args.monthly_clean if args.monthly_clean else None,
        "reference_periods_file": args.reference_periods if args.reference_periods else None,
        "outliers_summary_file": args.outliers_summary if args.outliers_summary else None,
        "sanity_persistence_file": args.sanity_persistence if args.sanity_persistence else None,
        "countries": int(per_country["country"].nunique()),
        "share_climatology_12_ok": float((per_country["climatology_12_months_ok"]==True).mean()),
        "share_rows_match_clean": float((per_country["rows_match_clean"]==True).mean()) if "rows_match_clean" in per_country else None,
        "share_mean_anom_ref_ok": float((per_country["mean_anom_in_ref_ok"]==True).mean()) if "mean_anom_in_ref_ok" in per_country else None
    }
    out_json = Path(args.report_json)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("[OK] Step 10 validation written:", str(out_csv))
    print(json.dumps(meta, indent=2))

if __name__ == "__main__":
    main()
