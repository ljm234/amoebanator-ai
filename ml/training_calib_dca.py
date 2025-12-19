# training_calib_dca.py
# Trains an MLP, fits temperature scaling, saves validation predictions,
# and writes val_preds.csv for plotting (calibration + DCA).
# Run:  python -m ml.training_calib_dca

import os, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, recall_score

from ml.calibration import fit_temperature  # uses CPU

# ---------- Model ----------
class MLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
        )
    def forward(self, x):
        return self.net(x)

# ---------- Data ----------
def load_tabular(csv_path="outputs/diagnosis_log_pro.csv"):
    df = pd.read_csv(csv_path)
    # one-hot for "symptoms" stored as "a;b;c"
    all_symptoms = set()
    for s in df["symptoms"].astype(str):
        for token in [t for t in s.split(";") if t]:
            all_symptoms.add(token)
    for sym in sorted(all_symptoms):
        df[f"sym_{sym}"] = df["symptoms"].astype(str).apply(
            lambda x: 1 if sym in x.split(";") else 0
        )

    feats = [
        "age", "csf_glucose", "csf_protein", "csf_wbc",
        "pcr", "microscopy", "exposure",
        *[c for c in df.columns if c.startswith("sym_")]
    ]
    X = df[feats].fillna(0).astype(float).values
    y = (df["risk_label"].astype(str).str.lower() == "high").astype(int).values
    return X, y, feats

# ---------- Numerically stable softmax ----------
def stable_softmax(logits: np.ndarray) -> np.ndarray:
    # Subtract row-wise max (same probabilities, prevents overflow)
    z = logits - np.max(logits, axis=1, keepdims=True)
    z = np.exp(z)
    z_sum = np.sum(z, axis=1, keepdims=True)
    return (z / z_sum).astype(np.float32)

# ---------- Main ----------
def main():
    os.makedirs("outputs/model", exist_ok=True)
    os.makedirs("outputs/metrics", exist_ok=True)

    X, y, feats = load_tabular()
    Xtr, Xva, ytr, yva = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    torch.set_default_dtype(torch.float32)  # MPS is float32-only

    model = MLP(X.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Class weight (clamped so tiny datasets don’t explode loss)
    pos = int((ytr == 1).sum())
    neg = int((ytr == 0).sum())
    w_pos = float(neg / max(pos, 1))
    w_pos = float(min(max(w_pos, 1.0), 10.0))
    weights = torch.tensor([1.0, w_pos], dtype=torch.float32, device=device)
    crit = nn.CrossEntropyLoss(weight=weights)

    xb = torch.tensor(Xtr, dtype=torch.float32, device=device)
    yb = torch.tensor(ytr, dtype=torch.long, device=device)

    for _ in range(60):
        model.train()
        opt.zero_grad()
        logits = model(xb)
        loss = crit(logits, yb)
        loss.backward()
        opt.step()

    # Validation logits → CPU numpy
    model.eval()
    with torch.no_grad():
        logits_val = model(torch.tensor(Xva, dtype=torch.float32, device=device)).cpu().numpy()

    # Uncalibrated & temperature-scaled probabilities (stable softmax)
    p_uncal = stable_softmax(logits_val)[:, 1]

    T = fit_temperature(model, logits_val, yva, device="cpu")
    T = float(np.clip(T, 0.1, 10.0))   # clamp temperature to sensible range
    logits_scaled = logits_val / T
    p_cal = stable_softmax(logits_scaled)[:, 1]

    # Guarantee finite numbers for sklearn
    p_uncal = np.nan_to_num(p_uncal, nan=0.5, posinf=1.0, neginf=0.0)
    p_cal   = np.nan_to_num(p_cal,   nan=0.5, posinf=1.0, neginf=0.0)

    auc = roc_auc_score(yva, p_cal)
    rec_high = recall_score(yva, (p_cal >= 0.5).astype(int), pos_label=1)

    print("Validation AUC (calibrated):", round(float(auc), 4))
    print("Recall(High) @0.5:", round(float(rec_high), 4))
    print("T (temperature, clamped [0.1,10.0]):", round(float(T), 4))

    # Artifacts
    torch.save(model.state_dict(), os.path.join("outputs/model", "model.pt"))
    with open(os.path.join("outputs/model", "features.json"), "w") as f:
        json.dump(feats, f, indent=2)
    with open(os.path.join("outputs/model", "temperature_scale.json"), "w") as f:
        json.dump({"T": float(T)}, f, indent=2)

    # Validation predictions for plots
    dfv = pd.DataFrame({
        "y_true": yva.astype(int),
        "p_high_uncal": p_uncal.astype(float),
        "p_high_cal": p_cal.astype(float),
    })
    dfv.to_csv(os.path.join("outputs/metrics", "val_preds.csv"), index=False)
    with open(os.path.join("outputs/metrics", "metrics.json"), "w") as f:
        json.dump(
            {"auc_calibrated": float(auc), "recall_high@0.5": float(rec_high), "T": float(T)},
            f, indent=2
        )
    print("Saved outputs/metrics/val_preds.csv and metrics.json, plus model artifacts.")

if __name__ == "__main__":
    main()