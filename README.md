# Self-Healing ML Pipeline

Cost-aware drift detection and automated retraining for production ML. An
MTech final-year project scoped to double as (a) a deployable applied-ML/MLOps
system for a resume/interview demo, and (b) a small empirical study comparing
drift-detection algorithms, submittable as a paper.

## What's here

- `src/stream_generator.py` — synthetic concept-drift streams (SEA, SINE,
  rotating hyperplane) with known ground-truth drift points. Used instead of a
  real-world set like Elec2 because real streams have no labeled drift point,
  making detection-delay evaluation impossible to do rigorously. Swap in a real
  CSV later (see below) once you have one — the pipeline is dataset-agnostic.
- `src/drift_detectors.py` — DDM, EDDM, ADWIN, KSWIN implemented from scratch
  (published algorithms; no internet access was available in this environment
  to install `river`).
- `src/model.py` — the served classifier (SGDClassifier), with a `retrain()`
  hot-swap method.
- `src/cost_tracker.py` — simulated cloud cost (per-inference + per-retrain),
  calibrated to AWS Lambda / small training-instance list pricing.
- `src/pipeline.py` — the online simulation loop: predict, monitor, retrain on
  drift, log metrics (accuracy, detection delay, false alarms, cost).
- `src/api.py` — FastAPI service exposing the pipeline as `/predict`,
  `/status`, `/reset`. This is the "applied system" for the job-demo side.
- `experiments/run_comparison.py` — runs every detector against every stream,
  writes `results/detector_comparison.csv`. This table is the paper's core
  result.
- `deploy/README.md` — how to actually put the API on AWS Lambda / GCP Cloud
  Run / Render once you have an account to deploy to.

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
- **Deployment**: live at the URL in `deploy/README.md`; verified end-to-end
  (predict/status/reset, input validation, drift-triggered retrain).
- **Demo**: `demo/live_demo.py` streams a seeded drift event against the live
  API for a recorded walkthrough.
- **Next, if continuing**: real-world dataset validation, a formal
  hyperparameter search, ablation across all four detectors (not just ADWIN),
  and replacing simulated cost with measured cloud billing.
