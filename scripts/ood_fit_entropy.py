#!/usr/bin/env python3
import json, math
from pathlib import Path
import numpy as np
import pandas as pd

METRICS_DIR = Path("outputs/metrics")
VAL_PREDS = METRICS_DIR / "val_preds.csv"
OOD_JSON = METRICS_DIR / "ood_gate.json"

def entropy_from_p(p):
    p = float(p)
    p = min(max(p, 1e-8), 1 - 1e-8)
    return -(p * math.log(p) + (1 - p) * math.log(1 - p))

def main(q=0.98):
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    if not VAL_PREDS.exists():
        raise SystemExit("Missing outputs/metrics/val_preds.csv — run ml.training_calib_dca first.")
    df = pd.read_csv(VAL_PREDS)
    if "p_high_cal" not in df.columns:
        raise SystemExit("val_preds.csv must contain column p_high_cal")
    H = df["p_high_cal"].astype(float).map(entropy_from_p).to_numpy()
    tau = float(np.quantile(H, q))
    out = {"method": "entropy", "tau": tau, "q": float(q), "n": int(H.shape[0])}
    OOD_JSON.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
