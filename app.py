import os, json, platform
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
from ml.ui_phase2 import render_phase2_panel
from ml.ui_phase3 import render_phase3_panel
from ml.ui_robust import render_robust_panel
from ml.ood_simple import ood_score

LOG_PATH = Path("outputs/diagnosis_log_pro.csv")
METRICS_DIR = Path("outputs/metrics")
CALIB_PNG = METRICS_DIR / "calibration_curve.png"
DCA_PNG = METRICS_DIR / "dca_curve.png"
METRICS_JSON = METRICS_DIR / "metrics.json"
CI_JSON = METRICS_DIR / "ci.json"
THRESH_PICK_JSON = METRICS_DIR / "threshold_pick.json"
VAL_PREDS = METRICS_DIR / "val_preds.csv"
CONFORMAL_JSON = METRICS_DIR / "conformal.json"
DEFAULT_RECOMMENDED_THRESHOLD = 0.15

def header():
    st.set_page_config(page_title="Amoebanator Pro", layout="centered")
    st.title("Amoebanator Pro — Phase 1 · 2 · 3")
    st.caption("Phase 1: calibrated probabilities + DCA · Phase 2: conformal coverage + ABSTAIN · Phase 3: OOD gate")

@st.cache_data
def load_log(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    for col in ["age","csf_glucose","csf_protein","csf_wbc","pcr","microscopy","exposure","risk_score"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "risk_label" in df.columns:
        df["risk_label"] = df["risk_label"].astype(str)
    if "symptoms" in df.columns:
        df["symptoms"] = df["symptoms"].fillna("").astype(str)
    return df

@st.cache_data
def load_json(path: Path):
    try:
        if path.exists():
            with open(path, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return {}

@st.cache_data
def load_val_preds(path: Path):
    if path.exists():
        try:
            df = pd.read_csv(path)
            y = df["y_true"].astype(int).to_numpy()
            p = df["p_high_cal"].astype(float).to_numpy()
            return y, p
        except Exception:
            return None, None
    return None, None

def symptom_counts(df: pd.DataFrame) -> pd.Series:
    if "symptoms" not in df.columns:
        return pd.Series(dtype=int)
    counts = {}
    for s in df["symptoms"].astype(str):
        for tok in [t for t in s.split(";") if t]:
            counts[tok] = counts.get(tok, 0) + 1
    return pd.Series(counts).sort_values(ascending=False)

def net_benefit(y_true: np.ndarray, prob: np.ndarray, t: float):
    if y_true is None or prob is None or len(y_true) == 0:
        return None, None, None, None, None
    y = (y_true.astype(int)).copy()
    p = (prob.astype(float)).copy()
    N = len(y)
    prev = y.mean()
    pred = (p >= t).astype(int)
    tp = ((pred==1) & (y==1)).sum()
    fp = ((pred==1) & (y==0)).sum()
    fn = ((pred==0) & (y==1)).sum()
    nb_model = (tp/N) - (fp/N) * (t/(1-t))
    nb_all = prev - (1 - prev) * (t/(1-t))
    return float(nb_model), float(nb_all), int(tp), int(fp), int(fn)

def sweep_best_threshold(y_true: np.ndarray, prob: np.ndarray, lo=0.05, hi=0.30, step=0.01):
    if y_true is None or prob is None:
        return None
    best = None
    t = lo
    while t <= hi + 1e-9:
        nb_model, nb_all, tp, fp, fn = net_benefit(y_true, prob, t)
        if nb_model is None:
            t += step
            continue
        if (best is None) or (nb_model > best["net_benefit"]):
            best = {"threshold": round(float(t), 2),
                    "net_benefit": nb_model,
                    "tp": tp, "fp": fp, "fn": fn}
        t += step
    return best

def conformal_metrics(y_true: np.ndarray, p_high: np.ndarray, qhat: float):
    if y_true is None or p_high is None:
        return None
    include_high = p_high >= (1.0 - qhat)
    include_low  = p_high <= qhat
    set_size2 = include_high & include_low
    true_is_high = (y_true == 1)
    contained = (true_is_high & include_high) | (~true_is_high & include_low)
    N = len(y_true)
    cov = float(contained.mean()) if N else float("nan")
    abst = float(set_size2.mean()) if N else float("nan")
    return {"coverage": cov, "abstain_rate": abst, "n": N}

def charts_block(df: pd.DataFrame):
    with st.expander("📊 Charts & Global Stats", expanded=True):
        if df.empty:
            st.warning("No data yet. Make sure outputs/diagnosis_log_pro.csv exists and has rows.")
            return
        c1, c2, c3 = st.columns(3)
        with c1:
            if "risk_label" in df.columns:
                order = ["High","Moderate","Low"]
                st.bar_chart(df["risk_label"].value_counts().reindex(order).fillna(0))
                st.caption("Risk label distribution")
        with c2:
            sc = symptom_counts(df)
            if not sc.empty:
                st.bar_chart(sc)
                st.caption("Symptom frequency")
        with c3:
            if "risk_score" in df.columns:
                st.bar_chart(df["risk_score"].astype(int).value_counts().sort_index())
                st.caption("Score distribution")

def figures_block():
    with st.expander("🧪 Metrics & Figures (Phase 1)", expanded=True):
        m = load_json(METRICS_JSON)
        c = load_json(CI_JSON)
        y, p = load_val_preds(VAL_PREDS)
        thresh_pick = load_json(THRESH_PICK_JSON)
        if not thresh_pick:
            best = sweep_best_threshold(y, p, lo=0.05, hi=0.30, step=0.01)
            thresh_pick = {"threshold": (best["threshold"] if best else DEFAULT_RECOMMENDED_THRESHOLD),
                           "net_benefit": (best["net_benefit"] if best else None)}
        st.subheader("Validation metrics (calibrated)")
        cols = st.columns(3)
        auc_val = m.get("auc_calibrated", float("nan"))
        rec_val = m.get("recall_high@0.5", float("nan"))
        temp_val = m.get("T", float("nan"))
        cols[0].metric("AUC", f"{auc_val:.3f}")
        cols[1].metric("Recall (High) @0.5", f"{rec_val:.3f}")
        cols[2].metric("Temperature (T)", f"{temp_val:.3f}")
        if c:
            c1, c2 = st.columns(2)
            a = c.get("auc_calibrated_CI95", {})
            r = c.get("recall_high@0.5_CI95", {})
            c1.write(f"**AUC 95% CI:** {a.get('lo', float('nan')):.3f} – {a.get('hi', float('nan')):.3f} (mean {a.get('mean', float('nan')):.3f})")
            c2.write(f"**Recall(High)@0.5 95% CI:** {r.get('lo', float('nan')):.3f} – {r.get('hi', float('nan')):.3f} (mean {r.get('mean', float('nan')):.3f})")
        st.divider()
        st.subheader("Recommended threshold (from Decision Curve Analysis)")
        t_pick = float(thresh_pick.get("threshold", DEFAULT_RECOMMENDED_THRESHOLD))
        nb = thresh_pick.get("net_benefit", None)
        cl1, cl2 = st.columns(2)
        cl1.metric("Recommended threshold", f"{t_pick:.2f}")
        cl2.metric("Net benefit @ threshold", f"{(nb if nb is not None else float('nan')):.4f}")
        st.caption("Chosen as the argmax net benefit within 0.05–0.30.")
        st.divider()
        st.subheader("Figures")
        c1, c2 = st.columns(2)
        with c1:
            st.image(str(CALIB_PNG)) if CALIB_PNG.exists() else st.info("Missing calibration_curve.png")
        with c2:
            st.image(str(DCA_PNG)) if DCA_PNG.exists() else st.info("Missing dca_curve.png")
        st.divider()
        if st.button("Clear cached data (reload CSV/metrics)"):
            st.cache_data.clear()
            st.success("Cache cleared. Use 'Rerun' in the app menu.")

def conformal_block():
    with st.expander("🛡️ Conformal (Phase 2) — coverage & ABSTAIN", expanded=True):
        y, p = load_val_preds(VAL_PREDS)
        conf = load_json(CONFORMAL_JSON)
        alpha_default = float(conf.get("alpha", 0.10)) if conf else 0.10
        alpha = st.slider("Target error rate α (coverage = 1−α)", 0.01, 0.20, value=alpha_default, step=0.01)
        if conf and abs(conf.get("alpha", 0.10) - alpha) < 1e-9:
            qhat = float(conf.get("qhat", 0.0))
            st.caption(f"Loaded qhat from conformal.json (n={conf.get('n','?')}, method={conf.get('method','?')})")
        else:
            if (y is None) or (p is None):
                st.warning("Need outputs/metrics/val_preds.csv to compute qhat.")
                return
            p_true = np.where(y==1, p, 1.0-p)
            scores = 1.0 - p_true
            n = len(scores)
            k = int(np.ceil((n + 1) * (1 - alpha)))
            k = min(max(k, 1), n)
            qhat = float(np.partition(scores, k-1)[k-1])
        m = conformal_metrics(y, p, qhat) if (y is not None and p is not None) else None
        c1, c2, c3 = st.columns(3)
        c1.metric("qhat", f"{qhat:.3f}")
        c2.metric("Target coverage", f"{1.0 - alpha:.2f}")
        c3.metric("Empirical coverage (val)", f"{(m['coverage'] if m else float('nan')):.2f}")
        if m:
            s1, s2 = st.columns(2)
            s1.metric("Abstain rate (val)", f"{m['abstain_rate']:.2f}")
            s2.write(f"n={m['n']}")
        st.write("Rule: include High if p≥1−qhat; include Low if p≤qhat; if both → ABSTAIN.")

def ood_panel(df: pd.DataFrame):
    st.subheader("Out-of-Distribution check (Phase 3)")
    if df.empty:
        st.info("No cases yet.")
        return
    last = df.iloc[-1].to_dict()
    try:
        s = ood_score(last)
    except Exception as e:
        st.warning(f"OOD not ready: {e}")
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("Mahalanobis", f"{s['mahal']:.2f}")
    c2.metric("Cutoff", f"{s['cutoff']:.2f}")
    c3.metric("Flag", "OOD" if s["is_ood"] else "In-dist")
    if any(s["range_violations"].values()):
        bad = [k for k,v in s["range_violations"].items() if v]
        st.error(f"Range issue in: {', '.join(bad)}")
    if s["is_ood"]:
        st.warning("Recommendation: ABSTAIN. Inputs sit outside the data cloud.")

def about_env_block():
    with st.expander("About & Environment", expanded=False):
        st.write("Phase 1 complete; Phase 2 adds conformal coverage + ABSTAIN; Phase 3 adds an OOD safety gate.")
        st.write(f"Streamlit: {st.__version__}")
        st.write(f"Python: {platform.python_version()}")
        st.write(f"Platform: {platform.platform()}")

def main():
    header()
    df = load_log(LOG_PATH)
    charts_block(df)
    figures_block()
    render_phase2_panel()
    conformal_block()
    render_phase3_panel()
    render_robust_panel()
    ood_panel(df)
    about_env_block()

if __name__ == "__main__":
    main()
