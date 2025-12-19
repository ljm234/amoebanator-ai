# scripts/fit_feature_stats.py
import json
import numpy as np
import pandas as pd
from pathlib import Path

LOG = Path("outputs/diagnosis_log_pro.csv")
OUT = Path("outputs/metrics/feature_stats.json")
OUT.parent.mkdir(parents=True, exist_ok=True)

NUM_COLS = ["age","csf_glucose","csf_protein","csf_wbc"]

def main():
    if not LOG.exists():
        raise SystemExit(f"Missing {LOG}")

    df = pd.read_csv(LOG)
    for c in NUM_COLS:
        if c not in df.columns:
            raise SystemExit(f"Missing column {c} in {LOG}")

    X = df[NUM_COLS].apply(pd.to_numeric, errors="coerce").dropna().to_numpy(dtype=np.float64)
    if X.shape[0] < 10:
        print(f"Warning: low sample size (n={X.shape[0]}). Stats will be noisy.")

    mu = X.mean(axis=0)
    cov = np.cov(X, rowvar=False)
    cov += 1e-6 * np.eye(cov.shape[0])

    # simple physiological ranges to catch gross errors
    ranges = {
        "age": [0, 100],
        "csf_glucose": [5, 150],     # mg/dL
        "csf_protein": [5, 1000],    # mg/dL
        "csf_wbc": [0, 50000]        # cells/µL
    }

    out = {
        "numeric_cols": NUM_COLS,
        "mu": mu.tolist(),
        "cov": cov.tolist(),
        "ranges": ranges,
        "note": "Mahalanobis gate in feature space; chi2 threshold with df=len(NUM_COLS)."
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(f"Wrote {OUT} (n={X.shape[0]})")

if __name__ == "__main__":
    main()
