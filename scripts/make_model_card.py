# scripts/make_model_card.py
from pathlib import Path
import json, platform, datetime, sys
import importlib.metadata as md

ROOT = Path(".")
OUT_DIR = ROOT / "outputs" / "docs"
MET = ROOT / "outputs" / "metrics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def _read_json(p):
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}

def main():
    metrics = _read_json(MET / "metrics.json")
    ci = _read_json(MET / "ci.json")
    thresh = _read_json(MET / "threshold_pick.json")
    conf = _read_json(MET / "conformal.json")
    feat = _read_json(MET / "feature_stats.json")

    auc = metrics.get("auc_calibrated", float("nan"))
    rec = metrics.get("recall_high@0.5", float("nan"))
    T = metrics.get("T", float("nan"))
    t_rec = float(thresh.get("threshold", 0.15)) if thresh else 0.15
    nb = thresh.get("net_benefit", None)

    cov = conf.get("coverage", None) if conf else None
    alpha = conf.get("alpha", None) if conf else None
    abst = conf.get("abstain_rate", None) if conf else None

    calib_png = "outputs/metrics/calibration_curve.png"
    dca_png   = "outputs/metrics/dca_curve.png"

    deps = []
    for pkg in ["python","streamlit","torch","scikit-learn","pandas","numpy"]:
        try:
            if pkg == "python":
                deps.append(f"- Python: {platform.python_version()}")
            else:
                deps.append(f"- {pkg}: {md.version(pkg)}")
        except Exception:
            pass

    lines = []
    lines += [
        "# Amoebanator Pro — Model Card",
        "",
        f"_Generated: {datetime.datetime.now().isoformat(timespec='seconds')}_",
        "",
        "## Overview",
        "- **Task:** Early triage for primary amoebic meningoencephalitis (PAM) from clinical inputs.",
        "- **Audience:** Clinicians for rapid risk assessment and documentation.",
        "- **Interface:** Streamlit app with risk tiers, exports, dashboards, and abstain behavior.",
        "",
        "## Intended Use",
        "- **Use:** Support early suspicion, not a standalone diagnosis.",
        "- **Population:** Patients with suspected CNS infection where PAM is on the differential.",
        "- **Inputs:** Age, CSF glucose/protein/WBC, exposure, PCR/microscopy flags, symptom tokens.",
        "- **Output:** Calibrated probability of “High risk” plus a decision threshold chosen via decision-curve analysis.",
        "",
        "## Performance (Validation)",
        f"- **AUC (calibrated):** {auc:.3f}",
        f"- **Recall (High) @0.5:** {rec:.3f}",
        f"- **Temperature (T):** {T:.3f}",
    ]

    if ci:
        a = ci.get("auc_calibrated_CI95", {})
        r = ci.get("recall_high@0.5_CI95", {})
        lines += [
            f"- **AUC 95% CI:** {a.get('lo','?'):.3f}–{a.get('hi','?'):.3f} (mean {a.get('mean','?'):.3f})",
            f"- **Recall@0.5 95% CI:** {r.get('lo','?'):.3f}–{r.get('hi','?'):.3f} (mean {r.get('mean','?'):.3f})",
        ]

    lines += [
        "",
        "## Threshold (Decision-Curve Analysis)",
        f"- **Recommended threshold:** {t_rec:.2f}",
        f"- **Net benefit @ threshold:** {('%.4f' % nb) if nb is not None else 'NA'}",
        f"- **Figure:** ![]({dca_png})",
        "",
        "## Calibration",
        f"- **Figure:** ![]({calib_png})",
        "",
        "## Uncertainty (Conformal Prediction)",
        f"- **Target error (α):** {alpha if alpha is not None else 'NA'}",
        f"- **Empirical coverage:** {('%.3f' % cov) if cov is not None else 'NA'}",
        f"- **Abstain rate:** {('%.3f' % abst) if abst is not None else 'NA'}",
        "- **Rule:** include High if p ≥ 1 − q̂; include Low if p ≤ q̂; both ⇒ ABSTAIN.",
        "",
        "## Robustness (Out-of-Distribution Gate)",
        f"- **Numeric features:** {', '.join(feat.get('numeric_cols', [])) if feat else 'NA'}",
        "- **Gate:** Mahalanobis distance vs in-distribution cutoff; gross range checks.",
        "",
        "## Safety, Limitations, and Ethics",
        "- Performance can shift if case-mix changes (distribution shift).",
        "- Abstain signals uncertainty/OOD; treat as a cue for expert review and tests.",
        "",
        "## Environment & Versions",
        *deps,
        f"- Platform: {platform.platform()}",
        "",
        "## References",
        "- Calibration with temperature scaling (Guo et al., 2017).",
        "- Decision-curve analysis for clinical utility (Vickers & Elkin, 2006).",
        "- Conformal prediction for coverage with abstain.",
        "- OOD detection baselines (Mahalanobis, Energy).",
    ]
    (OUT_DIR / "MODEL_CARD.md").write_text("\n".join(lines))
    print(f"Wrote {OUT_DIR / 'MODEL_CARD.md'}")

if __name__ == "__main__":
    sys.exit(main())
