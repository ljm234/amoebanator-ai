# scripts/debug_lastrow_ood.py
import json
import numpy as np
import pandas as pd
from pathlib import Path
from ml.robust.ood import STATS_JSON, LOG_CSV, check_ood_row

def main():
    if not Path(LOG_CSV).exists():
        print("Missing diagnosis_log_pro.csv"); return
    if not Path(STATS_JSON).exists():
        print("Missing feature_stats.json"); return
    df = pd.read_csv(LOG_CSV)
    row = df.iloc[-1]
    stats = json.loads(Path(STATS_JSON).read_text())
    out = check_ood_row(row, stats)
    cols = stats.get("numeric_cols") or stats.get("cols") or []
    d2 = float(out["d2"])
    tau = float(stats.get("tau", float("inf")))
    print(f"Mahalanobis d2={d2:.3f} vs tau={tau:.3f} (cols={len(cols)})")
    contrib = out.get("contrib")
    if contrib is not None and len(cols) == len(contrib):
        contrib = np.array(contrib)
        idx = np.argsort(contrib)[::-1]
        for k in idx[:5]:
            print(f"{cols[k]} contrib={contrib[k]:.3f}")

if __name__ == "__main__":
    main()
