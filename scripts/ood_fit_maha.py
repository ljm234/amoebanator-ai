# scripts/ood_fit_maha.py
import json
from pathlib import Path
import numpy as np
import pandas as pd
from ml.ood_maha import fit_mahalanobis, save_maha

METRICS_DIR = Path("outputs/metrics")
VAL_PREDS   = METRICS_DIR / "val_preds.csv"

# Columns to use as raw clinical features (adjust if your CSV differs)
FEATS = ["age","csf_glucose","csf_protein","csf_wbc","pcr","microscopy","exposure","risk_score"]

def main():
    df = pd.read_csv(VAL_PREDS)
    y  = df["y_true"].astype(int).to_numpy()
    X  = df[FEATS].astype(float).to_numpy()

    mu0, mu1, inv = fit_mahalanobis(X, y)

    # threshold tau = 95th percentile of min-distance on validation set
    from ml.ood_maha import maha_score
    d = np.array([maha_score(x, mu0, mu1, inv) for x in X])
    tau = float(np.quantile(d, 0.95))

    save_maha(mu0, mu1, inv, tau)
    out = {"n": int(len(y)), "tau": tau, "features": FEATS}
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
