# scripts/bootstrap_metrics.py
# 95% CI para AUC y Recall(High) con bootstrap sobre outputs/metrics/val_preds.csv
import json, os, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, recall_score

SRC = "outputs/metrics/val_preds.csv"
OUT = "outputs/metrics/ci.json"
N_BOOT = 2000
RNG = np.random.default_rng(42)

if not os.path.exists(SRC):
    raise FileNotFoundError(f"Missing {SRC}. Run python -m ml.training_calib_dca first.")

df = pd.read_csv(SRC)
y = df["y_true"].astype(int).to_numpy()
p = df["p_high_cal"].astype(float).to_numpy()  # usar calibradas

def boot_ci(stat_fn):
    stats = []
    n = len(y)
    for _ in range(N_BOOT):
        idx = RNG.integers(0, n, size=n)
        yy, pp = y[idx], p[idx]
        try:
            stats.append(stat_fn(yy, pp))
        except Exception:
            pass
    arr = np.array(stats, dtype=float)
    return float(np.nanpercentile(arr, 2.5)), float(np.nanpercentile(arr, 97.5)), float(np.nanmean(arr))

auc_ci = boot_ci(lambda yy, pp: roc_auc_score(yy, pp))
rec_ci = boot_ci(lambda yy, pp: recall_score(yy, (pp >= 0.5).astype(int), pos_label=1))

os.makedirs("outputs/metrics", exist_ok=True)
with open(OUT, "w") as f:
    json.dump({
        "auc_calibrated_CI95": {"lo": auc_ci[0], "hi": auc_ci[1], "mean": auc_ci[2]},
        "recall_high@0.5_CI95": {"lo": rec_ci[0], "hi": rec_ci[1], "mean": rec_ci[2]}
    }, f, indent=2)

print(f"Saved {OUT}")