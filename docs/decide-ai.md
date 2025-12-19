# DECIDE-AI (early clinical evaluation) — Readiness Notes

**Context & users:** ED/inpatient clinicians during suspected PAM.

**Human factors:** Clear “what to do now” prompts; ABSTAIN routing; contact lines; audit trail.

**Data flow:** Inputs → calibrated prob → DCA threshold → conformal set (ABSTAIN if needed) → clinician-facing recommendation.

**Safety:** OOD detection gates; abstain on unfamiliar CSF profiles; microscopy module abstains on blur/low motion.

**Governance:** Model Card present; versioned artifacts; site-level threshold tuning via DCA.

**Next:** Plan prospective early-stage evaluation capturing usability errors/time-to-action, aligned to DECIDE-AI items.
