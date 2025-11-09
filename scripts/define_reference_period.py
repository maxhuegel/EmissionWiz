#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

DEFAULT_MIN_PER_MONTH = 25
DEFAULT_WINDOW = (1981, 2010)
WINDOW_LEN = 30
SUPPORTED = {".csv", ".parquet", ".feather"}

def load_any(path: Path) -> pd.DataFrame:
    sfx = path.suffix.lower()
    if sfx == ".csv": return pd.read_csv(path)
    if sfx == ".parquet": return pd.read_parquet(path)
    if sfx == ".feather": return pd.read_feather(path)
    raise ValueError(f"Unsupported file: {path}")

def ensure_cols(df: pd.DataFrame, src: Path) -> pd.DataFrame:
    df = df.copy()
    if "country" not in df.columns: df["country"] = src.stem
    if ("year" not in df.columns or "month" not in df.columns) and "date" in df.columns:
        dt = pd.to_datetime(df["date"], errors="coerce")
        if "year" not in df.columns: df["year"] = dt.dt.year
        if "month" not in df.columns: df["month"] = dt.dt.month
    for req in ["year","month","temp_c"]:
        if req not in df.columns:
            raise SystemExit(f"{src.name}: missing required column '{req}'")
    return df[["country","year","month","temp_c"]]

def years_in_window(df, y0, y1):
    return df[(df["year"]>=y0) & (df["year"]<=y1)]

def window_score(dfw, n_min):
    counts = dfw.groupby("month")["year"].nunique().reindex(range(1,13), fill_value=0)
    months_meeting = int((counts >= n_min).sum())
    total_obs = int(counts.sum())
    return months_meeting, total_obs

def choose_window(df, default=(1981,2010), n_min=25):
    years = df["year"].dropna().astype(int)
    if years.empty:
        return {"ok": False, "reason": "no_years"}
    ymin, ymax = int(years.min()), int(years.max())
    d0, d1 = default
    df_def = years_in_window(df, d0, d1)
    mmeet, tobs = window_score(df_def, n_min)
    if mmeet == 12:
        return {"ok": True, "start": d0, "end": d1, "months_meeting": mmeet, "total_obs": tobs, "fallback": False}
    best = None
    default_center = (d0 + d1)/2.0
    for start in range(ymin, ymax - WINDOW_LEN + 2):
        end = start + WINDOW_LEN - 1
        dfw = years_in_window(df, start, end)
        mmeet, tobs = window_score(dfw, n_min)
        candidate = {"start": start, "end": end, "months_meeting": mmeet, "total_obs": tobs}
        if best is None or (mmeet, tobs) > (best["months_meeting"], best["total_obs"]) or (
            (mmeet, tobs) == (best["months_meeting"], best["total_obs"]) and abs((start+end)/2.0 - default_center) < abs((best["start"]+best["end"])/2.0 - default_center)
        ):
            best = candidate
    if best is None:
        return {"ok": False, "reason": "no_window"}
    ok = best["months_meeting"] == 12
    best["fallback"] = True
    best["ok"] = ok
    return best

def main():
    ap = argparse.ArgumentParser(description="Define per-country 30y reference periods for monthly climatology.")
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--report_csv", required=True)
    ap.add_argument("--report_json", required=True)
    ap.add_argument("--min_per_month", type=int, default=DEFAULT_MIN_PER_MONTH)
    ap.add_argument("--default_start", type=int, default=DEFAULT_WINDOW[0])
    ap.add_argument("--default_end", type=int, default=DEFAULT_WINDOW[1])
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    files = [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED]
    if not files:
        raise SystemExit(f"No data files found in {input_dir}")
    rows = []
    for p in sorted(files):
        df = ensure_cols(load_any(p), p)
        choice = choose_window(df, default=(args.default_start, args.default_end), n_min=args.min_per_month)
        country = str(df["country"].iloc[0])
        row = {
            "country": country,
            "default_start": args.default_start,
            "default_end": args.default_end,
            "chosen_start": choice.get("start"),
            "chosen_end": choice.get("end"),
            "months_meeting_min": choice.get("months_meeting"),
            "total_month_counts": choice.get("total_obs"),
            "fallback_used": bool(choice.get("fallback", False)),
            "ok_full_coverage": bool(choice.get("ok", False)),
            "reason_if_not_ok": choice.get("reason")
        }
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("country")
    Path(args.report_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.report_csv, index=False)
    meta = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "input_dir": str(input_dir),
        "min_per_month": args.min_per_month,
        "default_window": [args.default_start, args.default_end],
        "rowcount": int(len(out)),
        "n_ok_full": int(out["ok_full_coverage"].sum()),
        "n_fallback": int(out["fallback_used"].sum()),
    }
    with open(args.report_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print("[OK] Reference periods written:", args.report_csv)
    print(json.dumps(meta, indent=2))

if __name__ == "__main__":
    main()
