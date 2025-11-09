#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse, json, math
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

@dataclass
class Cfg:
    features_csv: Path
    anomalies_csv: Path
    cutoffs_csv: Path
    setup_json: Path
    in_forecasts: Path
    out_forecasts: Path
    h_start: int
    h_end: int
    alphas: list[float]
    min_train_rows: int

def ym_to_key(y:int, m:int)->int: return y*12 + (m-1)
def key_to_ym(k:int)->tuple[int,int]: return k//12, (k%12)+1

def load_cfg(args)->Cfg:
    with open(args.setup_json, "r", encoding="utf-8") as f:
        meta = json.load(f)
    # sanity
    _ = int(meta["horizons_max"])
    return Cfg(
        features_csv=Path(args.features),
        anomalies_csv=Path(args.anomalies),
        cutoffs_csv=Path(args.cutoffs_csv),
        setup_json=Path(args.setup_json),
        in_forecasts=Path(args.in_forecasts),
        out_forecasts=Path(args.out_forecasts),
        h_start=int(args.h_start),
        h_end=int(args.h_end),
        alphas=[float(a) for a in (args.alphas or [10.0, 100.0, 300.0])],
        min_train_rows=int(args.min_train_rows),
    )

def build_lookup(anom: pd.DataFrame):
    a = anom.copy()
    a["k"] = a["year"].astype(int)*12 + (a["month"].astype(int)-1)
    return a.set_index(["country","k"]).sort_index()

def select_features(df: pd.DataFrame)->list[str]:
    cols = ["mon_sin","mon_cos","anom_lag1","anom_lag12","roll_mean_3","roll_std_3"]
    for c in ["anom_lag24","roll_mean_12"]:
        if c in df.columns: cols.append(c)
    return cols

def fit_ridge_timeaware(X: np.ndarray, y: np.ndarray, alphas: list[float]):
    if X.shape[0] < 60:
        s = StandardScaler().fit(X)
        m = Ridge(alpha=alphas[0]).fit(s.transform(X), y)
        m._scaler, m._alpha = s, alphas[0]
        return m
    n_splits = 3 if X.shape[0] >= 100 else 2
    tscv = TimeSeriesSplit(n_splits=n_splits)
    best = (alphas[0], float("inf"))
    for a in alphas:
        rmses=[]
        for tr,va in tscv.split(X):
            s = StandardScaler().fit(X[tr])
            m = Ridge(alpha=a).fit(s.transform(X[tr]), y[tr])
            pred = m.predict(s.transform(X[va]))
            rmses.append(float(np.sqrt(np.mean((pred - y[va])**2))))
        avg = float(np.mean(rmses))
        if avg < best[1]: best = (a, avg)
    s = StandardScaler().fit(X)
    m = Ridge(alpha=best[0]).fit(s.transform(X), y)
    m._scaler, m._alpha = s, best[0]
    return m

def main():
    ap = argparse.ArgumentParser(description="Direct mid-horizon training (replace h7..24) and merge into existing forecasts.")
    ap.add_argument("--features", required=True)
    ap.add_argument("--anomalies", required=True)
    ap.add_argument("--cutoffs_csv", required=True)
    ap.add_argument("--setup_json", required=True)
    ap.add_argument("--in_forecasts", required=True, help="existing forecasts (recursive or blended)")
    ap.add_argument("--out_forecasts", required=True, help="merged forecasts with direct h7..24")
    ap.add_argument("--h_start", type=int, default=7)
    ap.add_argument("--h_end", type=int, default=24)
    ap.add_argument("--alphas", nargs="*", type=float, default=[30.0,100.0,300.0])
    ap.add_argument("--min_train_rows", type=int, default=120)
    args = ap.parse_args()
    cfg = load_cfg(args)

    feat = pd.read_csv(cfg.features_csv)
    anom = pd.read_csv(cfg.anomalies_csv)
    cuts = pd.read_csv(cfg.cutoffs_csv)
    with open(cfg.setup_json, "r", encoding="utf-8") as f:
        setup = json.load(f)
    HMAX = int(setup["horizons_max"])

    # keys
    feat["k"] = feat["year"].astype(int)*12 + (feat["month"].astype(int)-1)
    A = build_lookup(anom)

    # ensure cutoff_key
    if "cutoff_key" not in cuts.columns:
        def parse_ym(s:str)->int:
            y,m = s.split("-"); return ym_to_key(int(y), int(m))
        cuts = cuts.copy()
        cuts["cutoff_key"] = cuts["cutoff_ym"].apply(parse_ym)

    # select usable feature columns (global superset)
    base_cols = select_features(feat)

    rows = []
    for _, crow in cuts.iterrows():
        k_cut = int(crow["cutoff_key"])
        cutoff_ym = str(crow.get("cutoff_ym",""))

        for country, dfc in feat.groupby("country"):
            dfc = dfc.sort_values("k")
            # Precompute truth/clim per horizon h for fast access
            for h in range(cfg.h_start, min(cfg.h_end, HMAX)+1):
                # TRAIN: only rows with k <= k_cut - h (so that target at k+h exists after cutoff)
                train = dfc[dfc["k"] <= (k_cut - h)].copy()
                if len(train) < cfg.min_train_rows:
                    continue

                # dynamic feature selection per country/cutoff (no NaN columns)
                use_cols = [c for c in base_cols if c in train.columns and not train[c].isna().any()]
                if not use_cols:
                    continue

                # build direct target: anomaly at k+h
                def targ_at(k):
                    try:
                        return float(A.loc[(country, int(k+h)), "anomaly_c"])
                    except KeyError:
                        return np.nan
                train["y_target"] = train["k"].apply(targ_at)
                train = train.dropna(subset=["y_target"])
                if len(train) < cfg.min_train_rows:
                    continue

                X = train[use_cols].values
                y = train["y_target"].values
                if np.isnan(X).any() or np.isnan(y).any():
                    continue

                model = fit_ridge_timeaware(X, y, cfg.alphas)

                # PREDICT at the single origin k_cut for horizon h
                k_tgt = k_cut + h
                y_tgt, m_tgt = key_to_ym(k_tgt)
                try:
                    clim = float(A.loc[(country, k_tgt), "clim_temp_c"])
                    truth_c = float(A.loc[(country, k_tgt), "temp_c"])
                except KeyError:
                    continue

                # construct predictor row for that single (country, cutoff, h)
                # We reuse features from dfc at k = k_cut (features are already lagged/seasonal)
                xrow = dfc[dfc["k"] == k_cut]
                if xrow.empty:
                    continue
                x = xrow.iloc[0:1][use_cols].copy()
                if x.isna().any(axis=None):
                    continue

                x_s = model._scaler.transform(x.values)
                pred_anom = float(model.predict(x_s)[0])
                pred_c = pred_anom + clim

                rows.append({
                    "country": country,
                    "year": int(y_tgt),
                    "month": int(m_tgt),
                    "cutoff_ym": cutoff_ym,
                    "horizon": int(h),
                    "pred_anom": pred_anom,
                    "pred_c": pred_c,
                    "truth_c": truth_c,
                    "model": "ridge_direct"
                })

    # assemble direct mid-horizon frame
    mid = pd.DataFrame(rows)
    # merge into existing forecasts: replace only h in [h_start, h_end]
    base = pd.read_csv(cfg.in_forecasts)
    keycols = ["country","year","month","cutoff_ym","horizon"]
    # drop old mid-range
    keep = base[(base["horizon"] < cfg.h_start) | (base["horizon"] > cfg.h_end)].copy()
    merged = pd.concat([keep, mid], ignore_index=True).sort_values(keycols)
    Path(cfg.out_forecasts).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(cfg.out_forecasts, index=False)
    print(f"[OK] Wrote merged forecasts to {cfg.out_forecasts} "
          f"(replaced horizons {cfg.h_start}..{cfg.h_end})")

if __name__ == "__main__":
    main()