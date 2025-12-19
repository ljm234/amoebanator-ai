# scripts/conformal_fit_grouped.py
from __future__ import annotations
import json, math
import numpy as np
import pandas as pd
from pathlib import Path

VAL = Path("outputs/metrics/val_preds.csv")
OUT = Path("outputs/metrics/conformal_grouped.json")

def qhat_from_probs(y_true: np.ndarray, p_high: np.ndarray, alpha: float) -> float:
    p_true = np.where(y_true == 1, p_high, 1.0 - p_high)
    scores = 1.0 - p_true
    n = len(scores)
    if n <= 0:
        return float(alpha)
    k = int(math.ceil((n + 1) * (1 - alpha)))
    k = min(max(k, 1), n)
    return float(np.partition(scores, k - 1)[k - 1])

def main():
    if not VAL.exists():
        print("missing val_preds.csv"); return
    df = pd.read_csv(VAL)
    if not {"y_true","p_high_cal"}.issubset(df.columns):
        print("missing columns"); return
    if "age" not in df.columns:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps({"alpha": 0.10, "groups": {}}, indent=2))
        print("no age in val_preds; saved empty groups"); return
    alpha = 0.10
    child = df[df["age"] < 18]
    adult = df[df["age"] >= 18]
    groups = {}
    for name, part in [("child", child), ("adult", adult)]:
        if len(part) >= 1:
            qh = qhat_from_probs(part["y_true"].to_numpy(int), part["p_high_cal"].to_numpy(float), alpha)
            groups[name] = {"n": int(len(part)), "qhat": float(qh)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"alpha": alpha, "groups": groups}, indent=2))
    print("saved conformal_grouped.json")

if __name__ == "__main__":
    main()
