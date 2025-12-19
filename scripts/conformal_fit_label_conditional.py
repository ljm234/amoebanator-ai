# scripts/conformal_fit_label_conditional.py
import json, argparse
from pathlib import Path
import numpy as np
import pandas as pd

METRICS_DIR = Path("outputs/metrics")
VAL_PREDS = METRICS_DIR / "val_preds.csv"
OUT = METRICS_DIR / "conformal_label_conditional.json"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.10)
    args = ap.parse_args()

    df = pd.read_csv(VAL_PREDS)
    y = df["y_true"].astype(int).to_numpy()
    p = df["p_high_cal"].astype(float).to_numpy()

    # class-conditional nonconformity scores
    s_pos = 1.0 - p[y == 1]   # want high p on positives
    s_neg = p[y == 0]         # want low p on negatives

    def qhat_of(scores, alpha):
        if len(scores) == 0:
            return float("nan")
        k = int(np.ceil((len(scores) + 1) * (1 - alpha)))
        k = min(max(k, 1), len(scores))
        return float(np.partition(scores, k-1)[k-1])

    q_pos = qhat_of(s_pos, args.alpha)
    q_neg = qhat_of(s_neg, args.alpha)

    out = {
        "alpha": float(args.alpha),
        "qhat_pos": q_pos,   # threshold for including "High"
        "qhat_neg": q_neg,   # threshold for including "Low"
        "n_pos": int((y==1).sum()),
        "n_neg": int((y==0).sum()),
        "source": "val_preds.csv",
        "method": "label-conditional split conformal"
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
