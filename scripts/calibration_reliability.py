#!/usr/bin/env python3
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

METRICS_DIR = Path("outputs/metrics")
VAL_PREDS   = METRICS_DIR / "val_preds.csv"
OUT_PNG     = METRICS_DIR / "reliability.png"
ECE_JSON    = METRICS_DIR / "ece.json"

def ece_score(y_true, p, bins=10):
    y = y_true.astype(int)
    p = p.astype(float)
    bin_edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for i in range(bins):
        lo, hi = bin_edges[i], bin_edges[i+1]
        idx = (p >= lo) & (p < hi) if i < bins-1 else (p >= lo) & (p <= hi)
        if not np.any(idx): 
            continue
        conf = p[idx].mean()
        acc  = y[idx].mean()
        ece += (idx.mean()) * abs(acc - conf)
    return float(ece)

def main():
    if not VAL_PREDS.exists():
        raise SystemExit("Missing outputs/metrics/val_preds.csv")

    df = pd.read_csv(VAL_PREDS)
    y  = df["y_true"].to_numpy()
    p  = df["p_high_cal"].to_numpy()

    # plot
    bins = np.linspace(0,1,11)
    digit = np.digitize(p, bins) - 1
    centers, accs = [], []
    for b in range(10):
        mask = (digit == b)
        if mask.any():
            centers.append((bins[b]+bins[b+1])/2)
            accs.append(y[mask].mean())
    plt.figure(figsize=(5,5))
    plt.plot([0,1],[0,1], linestyle="--")
    plt.plot(centers, accs, marker="o")
    plt.xlabel("Predicted probability (High)")
    plt.ylabel("Empirical frequency (High)")
    plt.title("Reliability diagram (calibrated)")
    plt.grid(alpha=0.3)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(OUT_PNG, dpi=160)

    ece = ece_score(y, p, bins=10)
    ECE_JSON.write_text(json.dumps({"ece_10bin": ece}, indent=2))
    print(json.dumps({"wrote": str(OUT_PNG), "ece_10bin": ece}, indent=2))

if __name__ == "__main__":
    main()
