#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse, json, math
from pathlib import Path
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit

@dataclass
class Config:
    features_csv: Path
    anomalies_csv: Path
    cutoffs_csv: Path
    setup_json: Path
    out_forecasts: Path
    alphas: list[float]
    min_train_rows: int
    damping: float
    clip_anom: float
    blend_start: int
    blend_end: int
    blend_max: float

def ym_to_key(y:int,m:int)->int: return y*12 + (m-1)
def key_to_ym(k:int)->tuple[int,int]: return k//12, (k%12)+1

def load_cfg(args)->Config:
    with open(args.setup_json, "r", encoding="utf-8") as f:
        meta = json.load(f)
    _ = int(meta["horizons_max"])
    return Config(
        features_csv=Path(args.features),
        anomalies_csv=Path(args.anomalies),
        cutoffs_csv=Path(args.cutoffs_csv),
        setup_json=Path(args.setup_json),
        out_forecasts=Path(args.out_forecasts),
        alphas=[float(a) for a in (args.alphas or [0.1,1.0,10.0])],
        min_train_rows=int(args.min_train_rows),
        damping=float(args.damping),
        clip_anom=float(args.clip_anom),
        blend_start=int(args.blend_start),
        blend_end=int(args.blend_end),
        blend_max=float(args.blend_max),
    )

def build_lookup(df: pd.DataFrame):
    d = df.copy()
    d["k"] = d["year"].astype(int)*12 + (d["month"].astype(int)-1)
    return d.set_index(["country","k"]).sort_index()

def select_features(df: pd.DataFrame)->list[str]:
    cols = ["mon_sin","mon_cos","anom_lag1","anom_lag12","roll_mean_3","roll_std_3"]
    for c in ["anom_lag24","roll_mean_12"]:
        if c in df.columns: cols.append(c)
    return cols

def fit_ridge_timeaware(X: np.ndarray, y: np.ndarray, alphas: list[float]):
    if X.shape[0] < 60:
        scaler = StandardScaler().fit(X)
        m = Ridge(alpha=1.0).fit(scaler.transform(X), y)
        m._scaler = scaler
        m._alpha = 1.0
        return m
    n_splits = 3 if X.shape[0] >= 100 else 2
    tscv = TimeSeriesSplit(n_splits=n_splits)
    best_alpha, best_rmse = None, float("inf")
    for a in alphas:
        rmses = []
        for tr, va in tscv.split(X):
            s = StandardScaler().fit(X[tr])
            Xtr, Xva = s.transform(X[tr]), s.transform(X[va])
            m = Ridge(alpha=a).fit(Xtr, y[tr])
            pred = m.predict(Xva)
            rmse = float(np.sqrt(np.mean((pred - y[va])**2)))
            rmses.append(rmse)
        avg = float(np.mean(rmses))
        if avg < best_rmse:
            best_rmse, best_alpha = avg, a
    scaler = StandardScaler().fit(X)
    model = Ridge(alpha=best_alpha).fit(scaler.transform(X), y)
    model._scaler = scaler
    model._alpha = best_alpha
    return model

def blend_weight(h:int, start:int, end:int, wmax:float)->float:
    if end <= start or wmax <= 0: return 0.0
    if h <= start: return 0.0
    if h >= end: return wmax
    # linear ramp
    return wmax * (h - start) / float(end - start)

def main():
    ap = argparse.ArgumentParser(description="Phase 4 – Ridge per country, rolling-origin, recursive 1..HMAX with damping & climatology blend.")
    ap.add_argument("--features", required=True, help="features/features_v1.csv")
    ap.add_argument("--anomalies", required=True, help="data_clean/monthly_anomalies.csv")
    ap.add_argument("--cutoffs_csv", required=True, help="reports/phase3_folds.csv or phase2_cutoffs.csv")
    ap.add_argument("--setup_json", required=True, help="reports/phase2_setup.json")
    ap.add_argument("--out_forecasts", required=True, help="models/forecasts_model_ridge.csv")
    ap.add_argument("--alphas", nargs="*", type=float, default=[0.1,1.0,10.0])
    ap.add_argument("--min_train_rows", type=int, default=120)
    # Stabilitäts-Flags
    ap.add_argument("--damping", type=float, default=1.0, help="Mean-reversion factor for recursive anomaly (0..1). 1.0 = no damping.")
    ap.add_argument("--clip_anom", type=float, default=0.0, help="Clip absolute anomaly prediction to this value. 0 = off.")
    ap.add_argument("--blend_start", type=int, default=0, help="Horizon where climatology blending starts (0=off).")
    ap.add_argument("--blend_end", type=int, default=0, help="Horizon where blending reaches max.")
    ap.add_argument("--blend_max", type=float, default=0.0, help="Max blend weight with climatology at blend_end (0..1).")
    args = ap.parse_args()
    cfg = load_cfg(args)

    feat = pd.read_csv(cfg.features_csv)
    anom = pd.read_csv(cfg.anomalies_csv)
    cuts = pd.read_csv(cfg.cutoffs_csv)
    with open(cfg.setup_json, "r", encoding="utf-8") as f:
        setup = json.load(f)
    HMAX = int(setup["horizons_max"])

    feat["k"] = feat["year"].astype(int)*12 + (feat["month"].astype(int)-1)
    anom["k"] = anom["year"].astype(int)*12 + (anom["month"].astype(int)-1)
    L = build_lookup(anom)

    if "cutoff_key" not in cuts.columns:
        def parse_ym(s: str)->int:
            y, m = s.split("-"); return ym_to_key(int(y), int(m))
        cuts = cuts.copy(); cuts["cutoff_key"] = cuts["cutoff_ym"].apply(parse_ym)

    rows = []
    base_feature_list = select_features(feat)

    for _, crow in cuts.iterrows():
        k_cut = int(crow["cutoff_key"]); cutoff_ym = str(crow.get("cutoff_ym", ""))
        for country, dfc in feat.groupby("country"):
            dfc = dfc.sort_values("k")
            train = dfc[dfc["k"] <= k_cut].copy()
            if len(train) < cfg.min_train_rows:
                continue

            # Nur vollständige Spalten im jeweiligen Train-Set
            use_cols = [c for c in base_feature_list if c in train.columns and not train[c].isna().any()]
            if not use_cols:
                continue

            train = train.dropna(subset=["target_anom_t_plus_1"])
            if len(train) < cfg.min_train_rows:
                continue

            X = train[use_cols].values
            y = train["target_anom_t_plus_1"].values
            if np.isnan(X).any() or np.isnan(y).any():
                continue

            model = fit_ridge_timeaware(X, y, cfg.alphas)

            # Historie (Anomalien) bis Cutoff für Rekursion
            hist = []
            for kk in range(k_cut-59, k_cut+1):
                try: hist.append(float(L.loc[(country, kk), "anomaly_c"]))
                except KeyError: hist.append(np.nan)
            s = pd.Series(hist).fillna(method="ffill").fillna(method="bfill")
            hist = list(s.values)

            for h in range(1, HMAX+1):
                k_tgt = k_cut + h
                y_tgt, m_tgt = key_to_ym(k_tgt)
                try:
                    clim = float(L.loc[(country, k_tgt), "clim_temp_c"])
                    truth_c = float(L.loc[(country, k_tgt), "temp_c"])
                except KeyError:
                    break

                # Feature-Vektor aus State
                mon_sin = math.sin(2*math.pi*m_tgt/12.0)
                mon_cos = math.cos(2*math.pi*m_tgt/12.0)
                anom_lag1 = hist[-1] if len(hist)>=1 else np.nan
                anom_lag12 = hist[-12] if len(hist)>=12 else np.nan
                last3 = [v for v in hist[-3:] if pd.notna(v)]
                roll_mean_3 = float(np.mean(last3)) if len(last3)==3 else np.nan
                roll_std_3  = float(np.std(last3, ddof=0)) if len(last3)==3 else np.nan
                last12 = [v for v in hist[-12:] if pd.notna(v)]
                roll_mean_12 = float(np.mean(last12)) if len(last12)==12 else np.nan

                x = {"mon_sin":mon_sin, "mon_cos":mon_cos,
                     "anom_lag1":anom_lag1, "anom_lag12":anom_lag12,
                     "roll_mean_3":roll_mean_3, "roll_std_3":roll_std_3}
                if "anom_lag24" in use_cols:
                    x["anom_lag24"] = (hist[-24] if len(hist)>=24 else np.nan)
                if "roll_mean_12" in use_cols:
                    x["roll_mean_12"] = roll_mean_12

                x_df = pd.DataFrame([x])[use_cols]
                if x_df.isna().any(axis=None):
                    break

                x_s = model._scaler.transform(x_df.values)
                pred_anom = float(model.predict(x_s)[0])

                # Optional: Clip der Anomalie
                if cfg.clip_anom > 0:
                    pred_anom = float(np.clip(pred_anom, -cfg.clip_anom, cfg.clip_anom))

                # Climatology-Blend (auf °C)
                w = blend_weight(h, cfg.blend_start, cfg.blend_end, cfg.blend_max)
                pred_c = clim + (1.0 - w) * pred_anom

                rows.append({
                    "country": country, "year": int(y_tgt), "month": int(m_tgt),
                    "cutoff_ym": cutoff_ym, "horizon": int(h),
                    "pred_anom": pred_anom, "pred_c": pred_c, "truth_c": truth_c,
                    "model": "ridge"
                })

                # Rekursives Update mit Dämpfung (Mean-Reversion Richtung 0)
                damp = max(0.0, min(1.0, cfg.damping))
                hist.append(damp * pred_anom)
                if len(hist) > 120:
                    hist = hist[-120:]

    Path(cfg.out_forecasts).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(cfg.out_forecasts, index=False)
    print("[OK] Forecasts written:", cfg.out_forecasts)

if __name__ == "__main__":
    main()