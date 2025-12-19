# Model Card — Amoebanator Pro

**Intended use:** Early PAM triage support. Surface urgent steps (wet-mount, PCR, drug bundle) and risk tiering. Not a diagnostic.

**Inputs:** Age, exposure, symptom checklist, CSF (glucose, protein, WBC), PCR/microscopy flags.

**Outputs:** Calibrated probability of “High” tier, decision curve–chosen threshold, conformal coverage with ABSTAIN, OOD flags.

**Performance (validation):** AUC, Recall(High) at policy threshold, calibration temperature **T**. Add 95% CIs.

**Safety behaviors:** OOD gates (Mahalanobis + energy) → ABSTAIN; conformal **q̂** guarantees coverage; abstain on microscopy blur.

**Ethical limits:** Rare disease, tiny datasets, site drift; not for unsupervised use; human verification required.

**Training data:** Describe source/scope; pre-processing; splits.

**Calibration & thresholds:** Temperature scaling; DCA for operating point.

**Change log:** Versioned artifacts in `outputs/metrics/`.
