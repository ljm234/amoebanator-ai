#!/usr/bin/env python3
import math, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

METRICS_DIR = Path("outputs/metrics")
VAL_PREDS = METRICS_DIR / "val_preds.csv"
OUT_PNG = METRICS_DIR / "coverage_abstain_vs_alpha.png"

def coverage_abstain_at_alpha(y, p, alpha):
    p_true = np.where(y == 1, p, 1.0 - p)
    scores = 1.0 - p_true
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    k = min(max(k, 1), n)
    qhat = float(np.partition(scores, k - 1)[k - 1])
    include_high = p >= (1.0 - qhat)
    include_low = p <= qhat
    both = include_high & include_low
    true_is_high = (y == 1)
    contained = (true_is_high & include_high) | (~true_is_high & include_low)
    coverage = contained.mean() if n else np.nan
    abstain = both.mean() if n else np.nan
    return coverage, abstain

def main():
    if not VAL_PREDS.exists():
        raise SystemExit("Missing outputs/metrics/val_preds.csv")
    df = pd.read_csv(VAL_PREDS)
    y = df["y_true"].astype(int).to_numpy()
    p = df["p_high_cal"].astype(float).to_numpy()

    alphas = np.arange(0.01, 0.201, 0.01)
    covs, absts = [], []
    for a in alphas:
        c, ab = coverage_abstain_at_alpha(y, p, float(a))
        covs.append(c)
        absts.append(ab)

    plt.figure(figsize=(6,4))
    plt.plot(alphas, covs, label="coverage (target = 1−α)")
    plt.plot(alphas, absts, label="abstain rate")
    plt.xlabel("alpha (α)")
    plt.ylabel("fraction")
    plt.title("Conformal coverage and abstain vs α")
    plt.grid(True, alpha=0.3)
    plt.legend()
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=160)
    print(f"Wrote {OUT_PNG}")

if __name__ == "__main__":
    main()
