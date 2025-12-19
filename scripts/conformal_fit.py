import json, argparse
from pathlib import Path
import numpy as np

MET = Path("outputs/metrics")
MOD = Path("outputs/model")

def _load_T():
    for p in [MOD/"temperature_scale.json", MET/"temperature_scale.json"]:
        if p.exists():
            return float(json.loads(p.read_text())["T"])
    raise FileNotFoundError("temperature_scale.json not found in outputs/model or outputs/metrics")

def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / ez.sum(axis=1, keepdims=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.10)
    args = ap.parse_args()

    logits_p = MET / "val_logits.npy"
    y_p      = MET / "val_y.npy"
    if not (logits_p.exists() and y_p.exists()):
        raise FileNotFoundError("Missing outputs/metrics/val_logits.npy or val_y.npy. Re-run your calib script to export them.")

    T = _load_T()
    logits = np.load(logits_p)
    y = np.load(y_p).astype(int)
    p = _softmax(logits / T)

    s = 1.0 - p[np.arange(len(y)), y]  # nonconformity
    n = len(s)
    k = int(np.ceil((n + 1) * (1 - args.alpha)))
    s_sorted = np.sort(s)
    k = min(max(k, 1), n)
    qhat = float(s_sorted[k - 1])

    out = {"alpha": float(args.alpha), "qhat": qhat, "n_calib": int(n)}
    (MET / "conformal.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
