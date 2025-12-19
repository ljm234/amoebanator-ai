from __future__ import annotations
import json, math
import numpy as np
import pandas as pd
from pathlib import Path
from ml.robust.ood import score_tabular, load_stats, score_energy, ENERGY_JSON

METRICS_DIR = Path("outputs/metrics")
CONF_JSON = METRICS_DIR / "conformal.json"
CONF_G_JSON = METRICS_DIR / "conformal_grouped.json"
THRESH_PICK_JSON = METRICS_DIR / "threshold_pick.json"
DEFAULT_THRESHOLD = 0.15

def _read_json(p: Path) -> dict:
    try:
        if p.exists():
            return json.loads(p.read_text())
    except Exception:
        return {}
    return {}

def _toy_logits(row: pd.Series) -> tuple[float,float]:
    return 0.0, 0.3

def _choose_qhat(age_val: float | None) -> tuple[float,float,str]:
    g = None
    if CONF_G_JSON.exists() and age_val is not None:
        data = _read_json(CONF_G_JSON)
        alpha = float(data.get("alpha", 0.10))
        groups = data.get("groups", {})
        g = "child" if age_val < 18 else "adult"
        if g in groups:
            return float(groups[g].get("qhat", 0.10)), alpha, g
    data = _read_json(CONF_JSON)
    return float(data.get("qhat", 0.10)), float(data.get("alpha", 0.10)), g or "global"

def _energy_tau() -> float:
    d = _read_json(ENERGY_JSON)
    return float(d.get("tau", -2.0))

def infer_one(row_in: dict | pd.Series) -> dict:
    row = pd.Series(row_in) if not isinstance(row_in, pd.Series) else row_in
    stats = load_stats()
    ood = score_tabular(row=row, stats=stats)
    d2 = ood["d2"]
    tau_d2 = ood["tau"]
    if not ood["in_dist"]:
        return {
            "prediction": "ABSTAIN",
            "reason": "OOD",
            "p_high": 0.0,
            "mahalanobis_d2": float(d2),
            "d2_tau": float(tau_d2),
            "energy": None,
            "energy_tau": None,
            "include_low": False,
            "include_high": False,
            "contrib": ood.get("contrib")
        }
    lo, hi = _toy_logits(row)
    energy = score_energy(np.array([lo, hi], dtype=float))
    tau_e = _energy_tau()
    if energy < tau_e:
        return {
            "prediction": "ABSTAIN",
            "reason": "LowEnergy",
            "p_high": float(1.0 / (1.0 + math.exp(-hi))),
            "mahalanobis_d2": float(d2),
            "d2_tau": float(tau_d2),
            "energy": float(energy),
            "energy_tau": float(tau_e),
            "include_low": False,
            "include_high": False,
            "contrib": ood.get("contrib")
        }
    p_high = float(1.0 / (1.0 + math.exp(-hi)))
    age_val = None
    if "age" in row.index:
        try:
            age_val = float(row["age"])
        except Exception:
            age_val = None
    qhat, alpha, group = _choose_qhat(age_val)
    include_high = bool(p_high >= (1.0 - qhat))
    include_low = bool(p_high <= qhat)
    if include_high and include_low:
        return {
            "prediction": "ABSTAIN",
            "reason": "ConformalAmbiguity",
            "p_high": p_high,
            "mahalanobis_d2": float(d2),
            "d2_tau": float(tau_d2),
            "energy": float(energy),
            "energy_tau": float(tau_e),
            "include_low": True,
            "include_high": True,
            "qhat": float(qhat),
            "alpha": float(alpha),
            "group": group,
            "contrib": ood.get("contrib")
        }
    thresh_pick = _read_json(THRESH_PICK_JSON)
    threshold = float(thresh_pick.get("threshold", DEFAULT_THRESHOLD))
    if include_high and not include_low:
        label = "High"
    elif include_low and not include_high:
        label = "Low"
    else:
        label = "Moderate"
    out = {
        "prediction": label,
        "p_high": p_high,
        "threshold": threshold,
        "qhat": float(qhat),
        "alpha": float(alpha),
        "include_low": include_low,
        "include_high": include_high,
        "mahalanobis_d2": float(d2),
        "d2_tau": float(tau_d2),
        "energy": float(energy),
        "energy_tau": float(tau_e),
        "contrib": ood.get("contrib")
    }
    if group is not None:
        out["group"] = group
    return out
