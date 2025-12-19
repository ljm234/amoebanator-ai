from pathlib import Path
import json
import numpy as np

STATS_JSON = "outputs/metrics/feature_stats.json"
ENERGY_JSON = "outputs/metrics/energy_threshold.json"

def _load_json(path):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return {}

def _as_float_list(x):
    return [float(v) for v in x]

def _ensure_cov(stats):
    cols = stats.get("numeric_cols") or stats.get("cols") or []
    n = len(cols)
    cov = stats.get("cov")
    if isinstance(cov, list) and cov and isinstance(cov[0], list) and len(cov) == n and len(cov[0]) == n:
        return np.array(cov, dtype=float)
    S = stats.get("S", [])
    if isinstance(S, list) and S:
        if isinstance(S[0], list):
            try:
                M = np.array(S, dtype=float)
                if M.shape == (n, n):
                    return M
            except Exception:
                pass
        d = np.zeros((n, n), dtype=float)
        for i, v in enumerate(S[:n]):
            try:
                d[i, i] = float(v)
            except Exception:
                d[i, i] = float(np.asarray(v).ravel()[0])
        return d
    return np.eye(n, dtype=float)

def _vectorize(row, cols):
    out = []
    for c in cols:
        v = row.get(c, np.nan)
        try:
            out.append(float(v))
        except Exception:
            out.append(np.nan)
    return np.array(out, dtype=float)

def ood_score(row):
    stats = _load_json(STATS_JSON)
    energy = _load_json(ENERGY_JSON)
    cols = stats.get("numeric_cols") or stats.get("cols") or []
    mu = np.array(_as_float_list(stats.get("mu", [0] * len(cols))), dtype=float)
    med = np.array(_as_float_list(stats.get("median", [0] * len(cols))), dtype=float)
    mad = np.array(_as_float_list(stats.get("mad", [1] * len(cols))), dtype=float)
    cov = _ensure_cov(stats)
    x = _vectorize(row, cols)
    eps = 1e-8
    if cov.ndim == 2 and cov.shape[0] == cov.shape[1] and cov.shape[0] == len(cols):
        try:
            inv = np.linalg.pinv(cov + eps * np.eye(cov.shape[0]))
            d = x - mu
            d2 = float(d @ inv @ d)
        except Exception:
            s = np.sqrt(np.clip(np.diag(cov), eps, None))
            z = (x - mu) / s
            d2 = float(np.nansum(z * z))
    else:
        s = np.sqrt(np.clip(np.diag(cov), eps, None))
        z = (x - mu) / s
        d2 = float(np.nansum(z * z))
    tau = float(stats.get("tau", float("inf")))
    is_ood = bool(d2 > tau)
    zrob = (x - med) / np.clip(mad, eps, None)
    rng = {}
    for i, c in enumerate(cols):
        try:
            rng[c] = bool(abs(zrob[i]) > 6.0)
        except Exception:
            rng[c] = False
    e_tau = float(energy.get("tau", 0.0))
    return {"mahal": d2, "cutoff": tau, "is_ood": is_ood, "range_violations": rng, "energy_tau": e_tau}
