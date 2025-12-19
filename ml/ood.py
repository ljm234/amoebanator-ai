from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

NUMERIC_COLS = [
    "age","csf_glucose","csf_protein","csf_wbc",
    "pcr","microscopy","exposure","risk_score"
]

LOG_CSV = Path("outputs/diagnosis_log_pro.csv")
METRICS_DIR = Path("outputs/metrics")
STATS_JSON = METRICS_DIR / "feature_stats.json"
ENERGY_JSON = METRICS_DIR / "energy_threshold.json"

def _load_log(path: Path = LOG_CSV) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _pick_cols(df: pd.DataFrame, drop_cols: list[str] | None) -> list[str]:
    cols = [c for c in NUMERIC_COLS if c in df.columns]
    if drop_cols:
        cols = [c for c in cols if c not in drop_cols]
    return cols

def _robust_z(x: np.ndarray, median: np.ndarray, mad: np.ndarray) -> np.ndarray:
    z = (x - median) / mad
    z = np.where(np.isfinite(z), z, 0.0)
    return z

def mahalanobis_d2(z: np.ndarray, mu: np.ndarray, S: np.ndarray, use_diagonal: bool = True):
    d = z - mu
    if use_diagonal:
        invd = 1.0 / (np.diag(S) + 1e-12)
        contrib = d * d * invd
        return float(contrib.sum()), contrib
    inv = np.linalg.inv(S + 1e-6 * np.eye(S.shape[0]))
    return float(d @ inv @ d), None

def fit_tabular_stats(
    csv: Path = LOG_CSV,
    drop_cols: list[str] | None = None,
    quantile: float = 0.999,
    use_diagonal: bool = True
) -> dict:
    df = _load_log(csv)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    if df.empty:
        out = {
            "cols": [],
            "numeric_cols": [],
            "median": [],
            "mad": [],
            "mu": [],
            "S": [],
            "use_diagonal": use_diagonal,
            "tau": float("inf"),
            "quantile": float(quantile)
        }
        STATS_JSON.write_text(json.dumps(out, indent=2))
        return out
    cols = _pick_cols(df, drop_cols)
    X = df[cols].to_numpy(dtype=float)
    X = np.where(np.isfinite(X), X, np.nan)
    col_median = np.nanmedian(X, axis=0)
    col_mad = np.nanmedian(np.abs(X - col_median), axis=0)
    col_mad = np.where(col_mad > 0, col_mad, 1.0)
    X_filled = np.where(np.isnan(X), col_median, X)
    Z = _robust_z(X_filled, col_median, col_mad)
    mu = Z.mean(axis=0)
    S = np.cov(Z, rowvar=False)
    if use_diagonal:
        S = np.diag(np.clip(np.diag(S), 1e-6, None))
    d2_list = []
    for i in range(Z.shape[0]):
        val, _ = mahalanobis_d2(Z[i], mu, S, use_diagonal)
        d2_list.append(val)
    tau = float(np.quantile(d2_list, quantile))
    out = {
        "cols": cols,
        "numeric_cols": cols,
        "median": col_median.tolist(),
        "mad": col_mad.tolist(),
        "mu": mu.tolist(),
        "S": S.tolist(),
        "use_diagonal": use_diagonal,
        "tau": tau,
        "quantile": float(quantile)
    }
    STATS_JSON.write_text(json.dumps(out, indent=2))
    return out

def load_stats(path: Path = STATS_JSON) -> dict:
    if Path(path).exists():
        return json.loads(Path(path).read_text())
    return {}

def score_tabular(row: pd.Series, stats: dict | None = None) -> dict:
    if stats is None:
        stats = load_stats()
    cols = stats.get("cols", [])
    if not cols:
        return {"d2": float("inf"), "tau": float("inf"), "in_dist": False, "reason": "NoStats"}
    x = row[cols].to_numpy(dtype=float)
    x = np.where(np.isfinite(x), x, np.nan)
    med = np.array(stats["median"], dtype=float)
    mad = np.array(stats["mad"], dtype=float)
    z = _robust_z(np.where(np.isnan(x), med, x), med, mad)
    mu = np.array(stats["mu"], dtype=float)
    S = np.array(stats["S"], dtype=float)
    d2, _ = mahalanobis_d2(z, mu, S, bool(stats.get("use_diagonal", True)))
    tau = float(stats.get("tau", float("inf")))
    return {"d2": float(d2), "tau": tau, "in_dist": bool(d2 <= tau)}

def score_energy(logits: np.ndarray) -> float:
    return -float(np.logaddexp.reduce(logits))

def fit_energy_threshold(val_logits: np.ndarray | None) -> dict:
    if val_logits is None or len(val_logits) == 0:
        tau = -0.0
    else:
        E = np.array([score_energy(v) for v in val_logits], dtype=float)
        tau = float(np.quantile(E, 0.01))
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    ENERGY_JSON.write_text(json.dumps({"tau": tau}, indent=2))
    return {"tau": tau}

def check_ood_row(row: pd.Series, stats: dict) -> dict:
    cols = stats.get("cols", [])
    if not cols:
        return {"d2": float("inf"), "contrib": None}
    x = row[cols].to_numpy(dtype=float)
    x = np.where(np.isfinite(x), x, np.nan)
    med = np.array(stats["median"], dtype=float)
    mad = np.array(stats["mad"], dtype=float)
    z = _robust_z(np.where(np.isnan(x), med, x), med, mad)
    mu = np.array(stats["mu"], dtype=float)
    S = np.array(stats["S"], dtype=float)
    d2, contrib = mahalanobis_d2(z, mu, S, bool(stats.get("use_diagonal", True)))
    return {"d2": float(d2), "contrib": (contrib.tolist() if contrib is not None else None)}
