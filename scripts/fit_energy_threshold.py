from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from ml.robust.ood import fit_energy_threshold

VAL = Path("outputs/metrics/val_preds.csv")

def main():
    logits = None
    if VAL.exists():
        df = pd.read_csv(VAL)
        if {"logit_high","logit_low"}.issubset(df.columns):
            logits = df[["logit_low","logit_high"]].to_numpy(dtype=float)
    out = fit_energy_threshold(logits)
    print(f"Saved outputs/metrics/energy_threshold.json; tau={out['tau']:+.6f}")

if __name__ == "__main__":
    main()
