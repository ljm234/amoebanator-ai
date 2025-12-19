import os, json, numpy as np, pandas as pd

SRC = "outputs/metrics/val_preds.csv"
OUT_TABLE = "outputs/metrics/threshold_sweep.csv"
OUT_PICK = "outputs/metrics/threshold_pick.json"

if not os.path.exists(SRC):
    raise FileNotFoundError(f"Missing {SRC}. Run python -m ml.training_calib_dca first.")

df = pd.read_csv(SRC)
y = df["y_true"].astype(int).to_numpy()
p = df["p_high_cal"].astype(float).to_numpy()
N = len(y)
prev = y.mean()

def net_benefit(y_true, prob, t):
    pred = (prob >= t).astype(int)
    tp = ((pred==1) & (y_true==1)).sum()
    fp = ((pred==1) & (y_true==0)).sum()
    nb_model = (tp/N) - (fp/N) * (t/(1-t))
    nb_all   = prev - (1-prev) * (t/(1-t))
    return nb_model, nb_all

rows = []
for t in np.round(np.arange(0.01, 0.51, 0.01), 2):
    nb_model, nb_all = net_benefit(y, p, t)
    pred = (p >= t).astype(int)
    tp = ((pred==1) & (y==1)).sum()
    fp = ((pred==1) & (y==0)).sum()
    tn = ((pred==0) & (y==0)).sum()
    fn = ((pred==0) & (y==1)).sum()
    sens = tp / max(tp+fn, 1)
    spec = tn / max(tn+fp, 1)
    ppv  = tp / max(tp+fp, 1)
    f1   = 2*tp / max(2*tp + fp + fn, 1)
    rows.append([t, nb_model, nb_all, sens, spec, ppv, f1])

tab = pd.DataFrame(rows, columns=["threshold","net_benefit","net_benefit_all","sensitivity","specificity","PPV","F1"])
tab.to_csv(OUT_TABLE, index=False)
cand = tab[(tab["threshold"]>=0.05) & (tab["threshold"]<=0.30)].copy()
pick = cand.sort_values("net_benefit", ascending=False).head(1).to_dict(orient="records")[0]
with open(OUT_PICK, "w") as f:
    json.dump(pick, f, indent=2)

print(f"Saved {OUT_TABLE} & {OUT_PICK}")
print("Suggested threshold =", pick["threshold"], "with net_benefit =", pick["net_benefit"])