"""
plot_calibration_and_dca.py
Generates a calibration curve (uncalibrated vs. temperature-scaled)
and a Decision Curve Analysis (DCA) plot from validation predictions.
Inputs:
  - outputs/metrics/val_preds.csv with: y_true, p_high_uncal, p_high_cal
Outputs:
  - outputs/metrics/calibration_curve.png
  - outputs/metrics/dca_curve.png
"""
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

METRICS_DIR = os.path.join("outputs", "metrics")
os.makedirs(METRICS_DIR, exist_ok=True)
csv_path = os.path.join(METRICS_DIR, "val_preds.csv")
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Missing {csv_path}. Run: python -m ml.training_calib_dca")

df = pd.read_csv(csv_path)
y = df["y_true"].astype(int).values
p_uncal = df["p_high_uncal"].astype(float).values
p_cal = df["p_high_cal"].astype(float).values

# Calibration
prob_true_u, prob_pred_u = calibration_curve(y, p_uncal, n_bins=10, strategy="uniform")
prob_true_c, prob_pred_c = calibration_curve(y, p_cal, n_bins=10, strategy="uniform")
plt.figure()
plt.plot([0,1],[0,1], linestyle="--")
plt.plot(prob_pred_u, prob_true_u, marker="o", label="Uncalibrated")
plt.plot(prob_pred_c, prob_true_c, marker="o", label="Temp-scaled")
plt.xlabel("Mean predicted probability (P[High])")
plt.ylabel("Fraction of positives")
plt.title("Calibration (Reliability) Curve")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(METRICS_DIR, "calibration_curve.png"), dpi=180)
plt.close()

# DCA
def net_benefit(y_true, prob, thresholds):
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob).astype(float)
    N = len(y_true); prev = y_true.mean()
    rows = []
    for t in thresholds:
        pred = (prob >= t).astype(int)
        tp = ((pred==1) & (y_true==1)).sum()
        fp = ((pred==1) & (y_true==0)).sum()
        nb_model = (tp/N) - (fp/N) * (t/(1-t))
        nb_all = prev - (1 - prev) * (t/(1-t))
        nb_none = 0.0
        rows.append((t, nb_model, nb_all, nb_none))
    return np.array(rows)

ts = np.linspace(0.01, 0.99, 99)
dca = net_benefit(y, p_cal, ts)
plt.figure()
plt.plot(dca[:,0], dca[:,1], label="Model (calibrated)")  # single chart, multiple lines
plt.plot(dca[:,0], dca[:,2], linestyle="--", label="Treat all")
plt.plot(dca[:,0], dca[:,3], linestyle="--", label="Treat none")
plt.xlabel("Threshold probability")
plt.ylabel("Net benefit")
plt.title("Decision Curve Analysis")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(METRICS_DIR, "dca_curve.png"), dpi=180)
plt.close()

print("Saved calibration_curve.png and dca_curve.png in outputs/metrics/")
