from __future__ import annotations
import json
import numpy as np
import pandas as pd
from pathlib import Path
import streamlit as st
from ml.infer import infer_one
from ml.robust.ood import STATS_JSON, ENERGY_JSON, fit_tabular_stats

LOG = Path("outputs/diagnosis_log_pro.csv")

def _read_json(p: Path) -> dict:
    try:
        if Path(p).exists():
            return json.loads(Path(p).read_text())
    except Exception:
        return {}
    return {}

def _write_json(p: Path, d: dict):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(d, indent=2))

def render_phase3_panel():
    with st.expander("Robustness (Phase 3)", expanded=True):
        if not LOG.exists():
            st.warning("No diagnosis_log_pro.csv"); return
        df = pd.read_csv(LOG)
        if df.empty:
            st.warning("Log is empty"); return
        row = df.iloc[-1]
        out = infer_one(row)
        stats = _read_json(STATS_JSON)
        ejson = _read_json(ENERGY_JSON)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("d²", f"{float(out.get('mahalanobis_d2', float('nan'))):.3f}")
        c2.metric("d² τ", f"{float(stats.get('tau', float('inf'))):.3f}")
        c3.metric("Energy", f"{float(out.get('energy', float('nan'))):.3f}")
        c4.metric("Energy τ", f"{float(ejson.get('tau', -2.0)):.3f}")

        c5, c6, c7 = st.columns(3)
        c5.metric("p(High)", f"{float(out.get('p_high', float('nan'))):.3f}")
        c6.metric("q̂", f"{float(out.get('qhat', float('nan'))):.3f}" if "qhat" in out else "—")
        c7.metric("α", f"{float(out.get('alpha', float('nan'))):.2f}" if "alpha" in out else "—")

        st.write(f"Decision: {out.get('prediction','?')}" + (f" ({out.get('reason')})" if out.get('prediction')=='ABSTAIN' and out.get('reason') else ""))

        cols = stats.get("cols", [])
        contrib = out.get("contrib", None)
        if contrib is not None and cols and len(cols) == len(contrib):
            order = np.argsort(np.array(contrib))[::-1]
            top = pd.DataFrame({"feature":[cols[i] for i in order], "contrib":np.array(contrib)[order]})
            st.dataframe(top.head(8), use_container_width=True)

        st.divider()
        st.subheader("Tune Gates")

        etau = float(ejson.get("tau", -2.0))
        new_etau = st.slider("Energy τ", -5.0, 1.0, value=float(etau), step=0.05)
        if st.button("Save Energy τ"):
            _write_json(ENERGY_JSON, {"tau": float(new_etau)})
            st.success(f"Saved Energy τ = {float(new_etau):.2f}")
            st.cache_data.clear()

        oq = float(stats.get("quantile", 0.999)) if stats else 0.999
        new_oq = st.slider("OOD quantile for d² τ", 0.90, 0.9999, value=float(oq), step=0.0001)
        drop = st.multiselect("Drop columns for OOD fit", cols if cols else ["age","csf_glucose","csf_protein","csf_wbc","pcr","microscopy","exposure","risk_score"], default=[])
        if st.button("Refit OOD τ"):
            out_stats = fit_tabular_stats(quantile=float(new_oq), drop_cols=list(drop), use_diagonal=True)
            st.success(f"Refit complete. τ={float(out_stats['tau']):.3f} on {len(out_stats['cols'])} cols")
            st.cache_data.clear()
