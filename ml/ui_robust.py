# ml/ui_robust.py
import json
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st

STATS_JSON = Path("outputs/metrics/feature_stats.json")
ENERGY_JSON = Path("outputs/metrics/energy_threshold.json")
LOG_CSV = Path("outputs/diagnosis_log_pro.csv")

def render_robust_panel():
    st.subheader("Robustness")
    if not LOG_CSV.exists():
        st.info("No cases yet.")
        return
    df = pd.read_csv(LOG_CSV)
    if df.empty:
        st.info("No cases yet.")
        return
    row = df.iloc[-1].to_dict()
    stats = json.loads(STATS_JSON.read_text()) if STATS_JSON.exists() else {}
    energy = json.loads(ENERGY_JSON.read_text()) if ENERGY_JSON.exists() else {"tau": 0.0}
    cols = stats.get("numeric_cols") or stats.get("cols") or []
    ok = bool(cols)
    if not ok:
        st.warning("OOD not ready: numeric_cols")
        return
    med = pd.Series(stats["median"], index=cols).astype(float)
    mad = pd.Series(stats["mad"], index=cols).astype(float)
    mu  = pd.Series(stats["mu"], index=cols).astype(float)
    Sd  = pd.Series([v for v in stats["S"]], index=range(len(cols)))
    xv = pd.Series(row)[cols].astype(float)
    xv = xv.where(xv.notna(), med)
    z = (xv - med) / mad.replace(0.0, 1.0)
    invd = 1.0 / (pd.Series([Sd[i][i] if isinstance(Sd[i], list) else Sd[i] for i in range(len(cols))]).astype(float) + 1e-12)
    d = z - mu
    d2 = float((d*d*invd).sum())
    tau_d2 = float(stats.get("tau", float("inf")))
    lo = -1.0
    hi = 0.0
    e_tau = float(energy.get("tau", 0.0))
    e_val = float(row.get("logit_low", 0.0))
    h_val = float(row.get("logit_high", 0.0))
    from math import log, exp
    E = -float(np.logaddexp.reduce([lo, hi])) if False else -float(np.log(exp(e_val)+exp(h_val))) if (e_val or h_val) else float(row.get("energy", 0.0))
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Mahalanobis d²", f"{d2:.2f}")
        st.caption(f"τd² = {tau_d2:.2f} • {'in-distribution' if d2<=tau_d2 else 'OOD'}")
    with c2:
        st.metric("Energy", f"{E:.2f}")
        st.caption(f"τE = {e_tau:.2f} • {'confident' if E>=e_tau else 'low confidence'}")
