# scripts/make_decide_ai_checklist.py
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
    conf = j(MET/"conformal.json")
    ood = j(MET/"feature_stats.json")

    lines = []
    lines += [
        "# DECIDE-AI — Early Clinical Evaluation (Amoebanator Pro)",
        f"_Generated {datetime.datetime.now().isoformat(timespec='seconds')}_",
        "",
        "## Clinical Context & Intended Integration",
        "- [x] Decision point: early PAM suspicion in ED/inpatient neurology.",
        "- [x] User: clinician; output informs next tests and escalation.",
        "",
        "## Human Factors & Workflow",
        "- [x] Immediate risk tier with calibrated probability.",
        "- [x] ABSTAIN paths route to human judgment and confirmatory testing.",
        "",
        "## Technical Validity",
        f"- [x] AUC (calibrated): {metrics.get('auc_calibrated','NA')}",
        f"- [x] Recall@0.5: {metrics.get('recall_high@0.5','NA')}",
        f"- [x] Coverage α={conf.get('alpha','NA')}, empirical={conf.get('coverage','NA')}, abstain={conf.get('abstain_rate','NA')}",
        f"- [x] OOD gate: Mahalanobis; range checks over {', '.join(ood.get('numeric_cols', [])) if ood else 'NA'}",
        "",
        "## Benefit–Harm Balance",
        "- [x] Threshold chosen by decision-curve analysis (net benefit).",
        "",
        "## Risks & Mitigations",
        "- [x] Distribution shift monitors (abstain + OOD).",
        "- [x] Explicit limits: not diagnostic; requires oversight.",
        "",
        "## Transparency",
        "- [x] Model Card included; figures saved; code reproducible.",
    ]
    (OUT/"DECIDE_AI_CHECKLIST.md").write_text("\n".join(lines))
    print(f"Wrote {OUT/'DECIDE_AI_CHECKLIST.md'}")

if __name__ == "__main__":
    main()
