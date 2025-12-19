# scripts/conformal_eval_label_conditional.py
import json
from pathlib import Path
import numpy as np
import pandas as pd

METRICS_DIR = Path("outputs/metrics")
VAL_PREDS = METRICS_DIR / "val_preds.csv"
LC_JSON   = METRICS_DIR / "conformal_label_conditional.json"

def main():
    df = pd.read_csv(VAL_PREDS)
    y = df["y_true"].astype(int).to_numpy()
    p = df["p_high_cal"].astype(float).to_numpy()

    conf = json.loads(Path(LC_JSON).read_text())
    q_pos, q_neg = conf["qhat_pos"], conf["qhat_neg"]
    alpha = conf["alpha"]

    include_high = p >= (1.0 - q_pos)           # class-conditional rule
    include_low  = p <= q_neg
    abstain      = include_high & include_low
    singletons   = include_high ^ include_low
    empty        = (~include_high) & (~include_low)

    covered = ((y==1) & include_high) | ((y==0) & include_low)

    res = {
        "alpha": alpha,
        "target_coverage": 1.0 - alpha,
        "empirical_coverage": float(covered.mean()),
        "abstain_rate": float(abstain.mean()),
        "singleton_rate": float(singletons.mean()),
        "empty_rate": float(empty.mean()),
        "n": int(len(y)),
        "pos_coverage": float(covered[y==1].mean()) if (y==1).any() else float("nan"),
        "neg_coverage": float(covered[y==0].mean()) if (y==0).any() else float("nan")
    }
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()
