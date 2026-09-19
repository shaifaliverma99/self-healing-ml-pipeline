# Self-Healing ML Pipeline

Cost-aware drift detection and automated retraining for production ML. An
MTech final-year project scoped to double as (a) a deployable applied-ML/MLOps
system for a resume/interview demo, and (b) a small empirical study comparing
drift-detection algorithms, submittable as a paper.

**[Live dashboard](https://claude.ai/artifact/J95YiowGk2gu7ZEh9JkNMr)** — run
the drift-detection demo against the live API and see the validated detector
comparison, ablation study, and significance tests, all in one page.

## What's here

- `src/stream_generator.py` — synthetic concept-drift streams (SEA, SINE,
  rotating hyperplane) with known ground-truth drift points. Used instead of a
  real-world set like Elec2 because real streams have no labeled drift point,
  making detection-delay evaluation impossible to do rigorously. Swap in a real
  CSV later (see below) once you have one — the pipeline is dataset-agnostic.
- `src/drift_detectors.py` — DDM, EDDM, ADWIN, KSWIN implemented from scratch
  (published algorithms; no internet access was available in this environment
  to install `river`).
- `src/model.py` — the served classifier (SGDClassifier). `retrain()` is the
  original unconditional hot-swap (used by the already-published detector
  comparison, untouched so those results stay reproducible). `propose_retrain()`
  + `commit()` are the newer validation-gated path: fit a candidate on the
  older part of the buffer, score it against the current model on a held-out
  recent slice, and only commit if it actually wins.
- `src/diagnosis.py` — on a drift signal, fits a shallow decision tree to
  predict "was this prediction wrong" from the raw features; its
  `feature_importances_` give an interpretable "which feature looks
  implicated" signal, since these streams shift the label function P(Y|X),
  not the input distribution P(X), so comparing feature distributions
  before/after finds nothing.
- `src/cost_tracker.py` — simulated cloud cost (per-inference + per-retrain),
  calibrated to AWS Lambda / small training-instance list pricing.
- `src/pipeline.py` — `run_pipeline()` is the original ungated loop.
  `run_pipeline_with_diagnosis()` adds the diagnosis + validation-gate layers
  on top, tracking retrain attempts/commits/rejections separately.
- `src/api.py` — FastAPI service exposing `/predict`, `/status`, `/reset`.
  The live deployment runs the diagnosis + validation-gated path: every
  `/predict` response includes `recovery` ("committed"/"rejected"/null) and
  `diagnosis_feature` when a retrain is attempted.
- `experiments/run_comparison.py` — runs every detector against every stream,
  writes `results/detector_comparison.csv`. This table is the paper's core
  result.
- `experiments/run_diagnosis_validation.py` — runs the diagnosis +
  validation-gated pipeline (ADWIN, 10 seeds/stream), writes
  `results/diagnosis_validation.csv` (commit/reject rates) and
  `results/diagnosis_accuracy.csv` (per-event diagnosis correctness on SEA,
  which has a known decoy feature to check against).
- `deploy/README.md` — how to actually put the API on AWS Lambda / GCP Cloud
  Run / Render once you have an account to deploy to.
- `dashboard/` — the live dashboard (`index.html`, self-contained, no build
  step). `template.html` + `build_dashboard.py` regenerate `index.html` from
  the real `results/*.csv` files, same pattern as `paper/build_paper.py` — run
  it after any experiment changes so the dashboard's numbers stay in sync.

## Run it

```
pip install -r requirements.txt
python experiments/run_comparison.py             # single-run illustrative comparison -> results/detector_comparison.csv
python experiments/run_statistical_validation.py  # 10-seed validation + Wilcoxon significance tests
python experiments/run_ablation.py                # no-retrain vs periodic vs detector-triggered retraining
python experiments/run_timing.py                  # measured per-detector wall-clock cost
uvicorn src.api:app --reload                       # serves the pipeline locally at :8000
```

All five `results/*.csv` files are committed, so the paper's tables are
reproducible without re-running anything — but every script is deterministic
(seeded), so re-running reproduces them byte-for-byte.

**The validated finding** (see `paper/Self_Healing_ML_Pipeline_IEEE.docx` for
full detail): the single-run comparison suggested ADWIN detects every drift;
under 10-seed statistical testing that doesn't hold. What does hold: ADWIN's
accuracy is never significantly different from EDDM's, but it costs
significantly less in most streams — a cost advantage, not a detection
advantage. Against DDM/KSWIN, ADWIN detects significantly better, but not
always for free. A separate ablation study found naive periodic retraining is
statistically competitive with detector-triggered retraining under abrupt
drift; the detector's value is concentrated in gradual drift.

## Swapping in a real dataset later

Replace a call to `STREAMS[...]` with a loader that returns `(X, y,
drift_points)`. If you can get a real drift benchmark (Elec2 from OpenML, or
a Kaggle fraud/credit dataset with a known policy-change date you treat as
`drift_points`), everything downstream — detectors, pipeline, API, cost
tracker — works unchanged.

## Status

- **Paper**: done — `paper/Self_Healing_ML_Pipeline_IEEE.docx`, IEEE format,
  with a real (verified, CrossRef/arXiv-checked) literature review, multi-seed
  statistical validation, an ablation study, and honest limitations.
- **Deployment**: live at the URL in `deploy/README.md`; verified end-to-end,
  including the diagnosis + validation-gated retrain path (a candidate that
  doesn't beat the current model on held-out data is rejected, not committed).
- **Dashboard**: `dashboard/` (also served publicly via GitHub Pages from
  `docs/`) — live API stats, an in-browser drift simulation, and the full
  validated results (detector comparison, ablation, diagnosis accuracy).
- **Demo**: `demo/live_demo.py` streams a seeded drift event against the live
  API, printing the diagnosis and commit/reject decision as they happen.

### What "self-healing" means here, precisely

This project implements two of the stages a full autonomous-recovery
framework would need — **detect** (4 drift detectors) and a narrow
**diagnose -> decide -> repair -> validate** loop scoped to one failure mode
(concept drift -> which feature looks implicated -> retrain -> keep only if
it measurably helps). It does **not** implement: detection of non-drift
failures (schema violations, missing data, resource/infra failures, training
instability), a general decision engine with multiple recovery actions
(retry/rollback/config-fix/resource-scaling), root-cause analysis beyond
single-feature attribution, or continuous policy learning. Those are listed
as future work in the paper, not claimed as implemented.

- **Next, if continuing**: real-world dataset validation, a formal
  hyperparameter search, ablation across all four detectors' diagnosis/
  validation behavior (not just ADWIN), replacing simulated cost with
  measured cloud billing, and extending detection beyond concept drift to
  the other failure modes listed above.
