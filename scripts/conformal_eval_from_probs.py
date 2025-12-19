import json
from pathlib import Path
import numpy as np
import pandas as pd

MET = Path("outputs/metrics")

def main():
    conf = json.loads((MET/"conformal.json").read_text())
    alpha = float(conf["alpha"]); qhat = float(conf["qhat"])

    df = pd.read_csv(MET / "val_preds.csv")
    y = df["y_true"].astype(int).to_numpy()
    p_high = df["p_high_cal"].astype(float).to_numpy()

    include_high = p_high >= (1.0 - qhat)
    include_low  = p_high <= qhat
    set_size = include_high.astype(int) + include_low.astype(int)

    true_is_high = (y == 1)
    contained = (true_is_high & include_high) | (~true_is_high & include_low)

    out = {
        "alpha": alpha,
        "target_coverage": 1.0 - alpha,
        "empirical_coverage": float(contained.mean()),
        "abstain_rate": float((set_size == 2).mean()),
        "singleton_rate": float((set_size == 1).mean()),
        "empty_rate": float((set_size == 0).mean()),
        "n": int(len(y))
    }
    (MET/"conformal_eval.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
