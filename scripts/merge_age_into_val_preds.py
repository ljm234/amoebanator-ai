# scripts/merge_age_into_val_preds.py
from __future__ import annotations
import pandas as pd
from pathlib import Path

VAL = Path("outputs/metrics/val_preds.csv")
LOG = Path("outputs/diagnosis_log_pro.csv")

def main():
    if not VAL.exists() or not LOG.exists():
        print("missing files"); return
    v = pd.read_csv(VAL)
    d = pd.read_csv(LOG)
    if "age" not in d.columns:
        print("no age in log"); return
    if len(d) < len(v):
        print("log shorter than val"); return
    v["age"] = d["age"].tail(len(v)).to_list()
    v.to_csv(VAL, index=False)
    print("age merged into val_preds.csv")

if __name__ == "__main__":
    main()
