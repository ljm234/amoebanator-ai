# Amoebanator Pro — Recovery+ (Safety‑Critical ML kit for PAM triage)

Clinical support app that estimates early risk of primary amoebic meningoencephalitis from simple bedside data. This repo adds a full ML pipeline with calibration, conformal prediction, and safety tooling.

**Goal:** fast rebuild with a serious ML core: calibrated probabilities, conformal sets, abstention under uncertainty, decision‑curve analysis, and an MPS‑friendly PyTorch pipeline. All de‑identified.

## Quickstart
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
python -m ml.training
```

## Highlights
- Rule engine with exposure escalation.
- PyTorch MLP with class weighting and temperature scaling (calibration).
- Conformal prediction (split) for set‑valued outputs / abstain.
- Decision Curve Analysis (DCA) for clinical utility.
- OOD hooks (Mahalanobis on penultimate features).
- Safe CSV logging with UUID case_id and timezone‑aware timestamps.

## Roadmap (research‑grade)
- Video/micrograph module (ConvLSTM or 3D‑CNN) for trophozoite detection.
- Few‑shot augmentation (VAE/mixup) for small-n.
- Federated fine‑tuning (simulated).
- Prospective pilot with coverage tracking.

## Disclaimer
Research/education only. Not a medical device.

