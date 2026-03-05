# Amoebanator Pro

A safety-critical machine learning system for early triage of primary amoebic meningoencephalitis (PAM). Built from scratch in Python with PyTorch, Streamlit, and a from-the-ground-up data governance layer.

PAM is one of the deadliest infections on record — over 97% fatality — and the window for intervention is brutally short. Every hour matters. This project exists because faster screening could actually change outcomes, and because the tooling around rare-disease ML is almost nonexistent. We're building it ourselves.

---

## What this actually does

The system takes a handful of bedside inputs (age, exposure history, symptoms, CSF lab values, PCR and microscopy flags) and produces:

1. **A calibrated probability** of high-risk PAM — not just a class label, but a number you can reason about.
2. **A decision-curve-optimized threshold** so the operating point isn't arbitrary. We pick the cutoff that maximizes clinical net benefit over a realistic range.
3. **Conformal prediction sets** with a formal coverage guarantee. If the model isn't confident enough, it says **ABSTAIN** instead of guessing.
4. **Out-of-distribution detection** (Mahalanobis distance + energy scoring) that flags inputs the model has never seen anything like — and refuses to score them.

The whole idea is: **if the model doesn't know, it should say so.**

---

## Project structure

```text
app.py                      ← Streamlit dashboard (Phases 1 + 2 + 3)
ml/
  training.py               ← PyTorch MLP with class weighting
  calibration.py            ← Temperature scaling (post-hoc)
  conformal.py              ← Split conformal prediction + ABSTAIN logic
  dca.py                    ← Decision Curve Analysis
  infer.py                  ← Inference pipeline
  robust.py                 ← Robust OOD scoring (Mahalanobis + energy)
  ood_simple.py             ← Lightweight OOD gate for the dashboard
  ood_maha.py               ← Mahalanobis distance utilities
  ood_energy.py             ← Energy-based OOD scoring
  ood.py                    ← Combined OOD orchestration
  ui_phase2.py              ← Conformal coverage UI panel
  ui_phase3.py              ← OOD gate UI panel
  ui_robust.py              ← Robustness dashboard panel
  training_calib_dca.py     ← End-to-end train → calibrate → DCA pipeline
  data/                     ← HIPAA-compliant data infrastructure
    acquisition.py          ← Resilient CDC data transfer with circuit breakers
    clinical.py             ← Clinical record parsing and validation
    microscopy.py           ← Microscopy image loading and preprocessing
    deidentification.py     ← Safe Harbor + k-anonymity + differential privacy
    compliance.py           ← IRB state machine, ORCID validation, CDC Form 0.1310
    audit_trail.py          ← Hash-chained audit log with Merkle tree checkpoints
    annotation_protocol.py  ← Expert annotation workflows and inter-rater calibration
    quality_assurance.py    ← Data quality validation pipeline
    versioning.py           ← Dataset lineage tracking
    dvc_versioning.py       ← DVC-based version control for datasets
    literature.py           ← PubMed mining and figure extraction
    synthetic.py            ← Synthetic data generation
    who_database.py         ← WHO Global Health Observatory access
    pathology_atlas.py      ← Licensed pathology atlas integration
    negative_collection.py  ← Negative-class sample collection
    labeling.py             ← Label Studio integration
scripts/                    ← Reproducible experiment scripts
  train.sh                  ← One-command training
  bootstrap_metrics.py      ← 95% confidence intervals via bootstrap
  threshold_sweep.py        ← Net-benefit threshold sweep + auto-pick
  conformal_fit.py          ← Fit conformal quantile q̂
  conformal_eval.py         ← Evaluate conformal coverage
  fit_feature_stats.py      ← Fit Mahalanobis reference statistics
  fit_energy_threshold.py   ← Fit energy-based OOD threshold
  plot_calibration_and_dca.py
  plot_coverage_abstain_vs_alpha.py
  ...and more
tests/
  test_phase1_1_audit_trail.py
  test_phase1_1_compliance.py
  test_phase1_1_deidentification.py
  test_scoring.py
outputs/
  model/                    ← Trained model artifact + temperature scale
  metrics/                  ← All evaluation JSONs, CSVs, and plots
  diagnosis_log_pro.csv     ← Case log
docs/
  model_card.md             ← Model card (intended use, limitations, safety)
  decide-ai.md              ← DECIDE-AI early clinical evaluation notes
```

---

## The ML pipeline, step by step

**Training.** A two-layer MLP in PyTorch with class weighting to handle severe imbalance (PAM is extremely rare). Nothing exotic on purpose — the architecture is simple because the safety layers around it are where the real work is.

**Calibration.** Post-hoc temperature scaling fitted via L-BFGS on a held-out calibration split. This makes the predicted probabilities actually mean what they say — a 0.30 prediction should be right about 30% of the time.

**Decision Curve Analysis.** Instead of picking a threshold by gut feel, we sweep 0.01–0.50 and pick the operating point that maximizes net clinical benefit within 0.05–0.30. The threshold, along with TP/FP/FN counts, is saved to `outputs/metrics/threshold_pick.json`.

**Conformal prediction.** Split conformal with a nonconformity score of `1 − p(true class)`. For a given error rate α, we compute q̂ and produce set-valued predictions. When both classes are included in the set, the system outputs **ABSTAIN** — it's telling you it genuinely doesn't know. Coverage is guaranteed at 1−α under exchangeability.

**OOD detection.** Two independent gates:

- **Mahalanobis distance** on the feature vector, with a 95th-percentile cutoff fitted on validation data.
- **Energy scoring** on the model logits, with a percentile-based threshold.

If either gate fires, the system flags the input as out-of-distribution and recommends abstaining.

**Bootstrap CIs.** All reported metrics (AUC, recall, etc.) come with 95% confidence intervals from 2000 bootstrap resamples.

---

## Data governance

This isn't an afterthought — it's foundational. The `ml/data/` package implements:

- **Safe Harbor de-identification** covering all 18 HIPAA identifiers, with age capping at 89, ZIP truncation for small-population prefixes, date generalization, and regex-based free-text scrubbing.
- **k-Anonymity** enforcement with configurable quasi-identifiers, generalization hierarchies, and suppression. Information loss is tracked.
- **Differential privacy** via Laplace, Gaussian, and Exponential mechanisms with formal budget tracking.
- **Hash-chained audit logs** where every data event is recorded with a SHA-256 chain. Merkle tree checkpoints let you verify integrity of any individual entry without replaying the whole log.
- **IRB lifecycle tracking** as a full state machine (DRAFT → SUBMITTED → UNDER_REVIEW → APPROVED, with branches for REVISIONS_REQUESTED, CONDITIONALLY_APPROVED, SUSPENDED, TERMINATED, EXPIRED, RENEWAL_PENDING).
- **ORCID validation** with ISO 7064 Mod 11-2 checksum verification.
- **CDC Form 0.1310** field-level validation including security plan depth checks and NIST SP 800-88 destruction plan verification.
- **Signed security attestations** (HMAC-SHA256) covering 12 safeguard checks across physical, technical, and administrative categories.

All of this is tested. The test suite covers tamper detection, chain verification, state machine transitions, checksum edge cases, and round-trip serialization.

---

## Quickstart

```bash
git clone https://github.com/ljm234/amoebanator-ai.git
cd amoebanator-ai
python3 -m venv venv && source venv/bin/activate
pip install torch numpy pandas scikit-learn streamlit matplotlib

# Train the model
python -m ml.training

# Generate plots and threshold picks
python scripts/plot_calibration_and_dca.py
python scripts/threshold_sweep.py
python scripts/bootstrap_metrics.py

# Fit OOD reference statistics
python scripts/fit_feature_stats.py
python scripts/fit_energy_threshold.py

# Fit conformal quantile
python scripts/conformal_fit.py

# Launch the dashboard
streamlit run app.py
```

---

## Running the tests

```bash
pip install pytest
pytest tests/ -v
```

The test suite includes ~120 test cases covering the audit trail, compliance gates, de-identification pipeline, and scoring logic.

---

## What this is not

This is a research and education project. It is not a cleared medical device. It is not validated for unsupervised clinical use. PAM is a rare and lethal condition — any tool in this space needs prospective clinical evaluation, site-level threshold tuning, and human oversight at every step. The model card in `docs/model_card.md` spells this out clearly.

---

## Why we built it this way

Most ML projects in medicine skip the hard parts. They train a model, report an AUC, and call it done. We wanted to actually think about what happens when the model is wrong — especially for a disease where being wrong means someone dies.

So we built calibration in from the start. We added conformal sets so the system can say "I don't know." We added OOD detection so it can say "I've never seen anything like this." And we built a full data governance stack because if you're working with clinical data and you don't have audit trails, de-identification, and compliance tracking, you're not being serious.

There's still a lot to do. The dataset is tiny. The model architecture is intentionally simple. We haven't done prospective evaluation yet. But the bones are right, and every piece is tested and reproducible.

---

## Roadmap

- Microscopy module (trophozoite detection via video / micrograph analysis)
- Few-shot augmentation for small-sample regimes
- Federated learning simulation for multi-site deployment
- Prospective pilot study with coverage and abstention tracking
- TRIPOD-AI and DECIDE-AI checklist completion

---

## License

Research and educational use. See the model card for limitations and intended use.
