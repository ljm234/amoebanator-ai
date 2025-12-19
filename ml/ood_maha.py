# ml/ood_maha.py
import json
from pathlib import Path
import numpy as np

METRICS_DIR = Path("outputs/metrics")
MAHA_STATS  = METRICS_DIR / "maha_stats.npz"
MAHA_JSON   = METRICS_DIR / "maha.json"

def fit_mahalanobis(X: np.ndarray, y: np.ndarray):
    X0 = X[y==0]; X1 = X[y==1]
    mu0 = X0.mean(axis=0); mu1 = X1.mean(axis=0)
    # shared covariance (shrink if needed)
    cov = np.cov(X.T)
    # small ridge to stabilize inverse
    cov += 1e-6 * np.eye(cov.shape[0])
    inv = np.linalg.inv(cov)
    return mu0, mu1, inv

def maha_score(x: np.ndarray, mu0: np.ndarray, mu1: np.ndarray, inv: np.ndarray):
    # take min distance to either class centroid
    d0 = float(np.sqrt((x - mu0).T @ inv @ (x - mu0)))
    d1 = float(np.sqrt((x - mu1).T @ inv @ (x - mu1)))
    return min(d0, d1)

def load_maha():
    stats = np.load(MAHA_STATS)
    conf  = json.loads(Path(MAHA_JSON).read_text())
    return stats["mu0"], stats["mu1"], stats["inv"], conf["tau"]

def save_maha(mu0, mu1, inv, tau):
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(MAHA_STATS, mu0=mu0, mu1=mu1, inv=inv)
    Path(MAHA_JSON).write_text(json.dumps({"tau": float(tau)}, indent=2))
