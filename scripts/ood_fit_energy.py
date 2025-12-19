#!/usr/bin/env python3
import json, math
from pathlib import Path
import numpy as np
import pandas as pd

METRICS_DIR = Path("outputs/metrics")
VAL_PREDS    = METRICS_DIR / "val_preds.csv"
ENERGY_JSON  = METRICS_DIR / "ood_energy.json"

def neg_energy_from_p(p: float) -> float:
    p = float(min(max(p, 1e-8), 1 - 1e-8))
    z = math.log(p / (1.0 - p))
    return -math.log(1.0 + math.exp(z))

def main(q=0.95):
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    if not VAL_PREDS.exists():
        raise SystemExit("Missing outputs/metrics/val_preds.csv. Run Phase 1 training first.")

    df = pd.read_csv(VAL_PREDS)
    if "p_high_cal" not in df.columns:
        raise SystemExit("val_preds.csv must contain p_high_cal")

    e = df["p_high_cal"].astype(float).map(neg_energy_from_p).to_numpy()
    tau = float(np.quantile(e, q))  # 95th pct of in-distribution energy
    out = {"method": "energy_neg", "tau": tau, "q": float(q), "n": int(e.shape[0])}
    ENERGY_JSON.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
