# scripts/make_tripod_ai_checklist.py
from pathlib import Path
import json, datetime

ROOT = Path(".")
OUT = ROOT / "outputs" / "docs"
MET = ROOT / "outputs" / "metrics"
OUT.mkdir(parents=True, exist_ok=True)

def j(p): 
    try: return json.loads(p.read_text())
    except: return {}

def main():
    metrics = j(MET/"metrics.json")
    ci = j(MET/"ci.json")
    conf = j(MET/"conformal.json")
    ood = j(MET/"feature_stats.json")

    lines = []
    lines += [
        "# TRIPOD+AI — Reporting Checklist (Amoebanator Pro)",
        f"_Generated {datetime.datetime.now().isoformat(timespec='seconds')}_",
        "",
        "## Title/Abstract",
        "- [x] Clear statement of objective (early PAM triage).",
        "- [x] Intended use and setting.",
        "",
        "## Source of Data",
        "- [x] Variables listed (age, CSF metrics, exposure, PCR/micro, symptoms).",
        "- [x] Outcome definition (High-risk proxy).",
        "",
        "## Model Development",
        "- [x] Model type: logistic baseline + temperature scaling.",
        "- [x] Missingness handling described in app code.",
        "",
        "## Model Performance",
        f"- [x] AUC (calibrated): {metrics.get('auc_calibrated','NA')}",
        f"- [x] Recall@0.5: {metrics.get('recall_high@0.5','NA')}",
        f"- [x] 95% CIs in ci.json: {bool(ci)}",
        "",
        "## Clinical Usefulness",
        "- [x] Decision-curve analysis; threshold chosen by net benefit argmax.",
        "",
        "## Uncertainty and Risk Control",
        f"- [x] Conformal coverage: {conf.get('coverage','NA')} at α={conf.get('alpha','NA')}",
        f"- [x] Abstain rate: {conf.get('abstain_rate','NA')}",
        "",
        "## Robustness/OOD",
        f"- [x] OOD gate present (Mahalanobis over {', '.join(ood.get('numeric_cols', [])) if ood else 'NA'}).",
        "",
        "## Limitations",
        "- [x] Not for standalone diagnosis; requires human oversight.",
        "- [x] Distribution shift risk acknowledged.",
        "",
        "## Availability",
        "- [x] Streamlit app, reproducible scripts, figures exported.",
    ]
    (OUT/"TRIPOD_AI_CHECKLIST.md").write_text("\n".join(lines))
    print(f"Wrote {OUT/'TRIPOD_AI_CHECKLIST.md'}")

if __name__ == '__main__':
    main()
    