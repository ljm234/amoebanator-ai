from pathlib import Path
import json
import pandas as pd
import streamlit as st
from ml.infer import infer_one

def render_phase2_panel():
    st.subheader("Phase 2 — Conformal prediction (q̂) & ABSTAIN")

    met = Path("outputs/metrics")
    conf_p = met / "conformal.json"
    thr_p  = met / "threshold_pick.json"

    conf = json.loads(conf_p.read_text()) if conf_p.exists() else {"alpha": None, "qhat": None}
    thr  = json.loads(thr_p.read_text())  if thr_p.exists()  else {"threshold": None}

    st.write({
        "alpha (target error)": conf.get("alpha"),
        "qhat (cutoff)": conf.get("qhat"),
        "decision threshold": thr.get("threshold")
    })

    st.caption("Rule: include High if p ≥ 1−q̂; include Low if p ≤ q̂; if both → ABSTAIN.")
    if st.button("Infer on last logged case"):
        log = Path("outputs/diagnosis_log_pro.csv")
        if not log.exists():
            st.warning("Missing outputs/diagnosis_log_pro.csv")
            return
        df = pd.read_csv(log)
        if df.empty:
            st.warning("Log is empty.")
            return
        keep = {"age","csf_glucose","csf_protein","csf_wbc","pcr","microscopy","exposure","symptoms"}
        row = df.iloc[-1].to_dict()
        row = {k: row.get(k) for k in keep}
        out = infer_one(row)
        st.json(out)
