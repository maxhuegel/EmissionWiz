#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

def main():
    ap = argparse.ArgumentParser(description="Phase 3 – Make rolling-origin folds from phase2_cutoffs.csv")
    ap.add_argument("--cutoffs_csv", required=True)
    ap.add_argument("--out_folds", required=True)
    args = ap.parse_args()

    c = pd.read_csv(args.cutoffs_csv)
    if "cutoff_ym" not in c.columns: raise SystemExit("cutoffs CSV missing 'cutoff_ym'")
    c = c.sort_values("cutoff_ym").reset_index(drop=True)
    c["fold_id"] = range(1, len(c)+1)
    out = c[["fold_id","cutoff_ym","share_with_both_ok","share_with_history_ok","share_with_future_ok"]].copy()
    out.to_csv(args.out_folds, index=False)
    print("[OK] folds written:", args.out_folds)

if __name__ == "__main__":
    main()
