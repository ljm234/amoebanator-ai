import json
from pathlib import Path
import numpy as np
import pandas as pd
import argparse

MET = Path("outputs/metrics")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--val_preds", type=str, default=str(MET / "val_preds.csv"))
    args = ap.parse_args()

    df = pd.read_csv(args.val_preds)
    if not {"y_true","p_high_cal"}.issubset(df.columns):
        raise ValueError("val_preds.csv must have columns: y_true, p_high_cal")

    y = df["y_true"].astype(int).to_numpy()
    p_high = df["p_high_cal"].astype(float).to_numpy()

    p_true = np.where(y == 1, p_high, 1.0 - p_high)
    scores = 1.0 - p_true
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - args.alpha)))
    k = min(max(k, 1), n)
    qhat = float(np.partition(scores, k - 1)[k - 1])

    out = {"alpha": float(args.alpha), "qhat": qhat, "n": int(n), "source": "val_preds.csv"}
    (MET / "conformal.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
