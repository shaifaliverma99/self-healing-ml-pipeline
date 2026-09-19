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
python experiments/run_comparison.py        # produces results/detector_comparison.csv
uvicorn src.api:app --reload                 # serves the pipeline locally at :8000
```

Already run once in this environment — `results/detector_comparison.csv` has a
first-pass table showing the real trade-off the paper argues: EDDM detects
fastest but with far more false alarms (higher retrain cost) than DDM/ADWIN,
while KSWIN is cheapest but misses some drifts. That trade-off, quantified in
dollars via `cost_tracker.py`, is the paper's contribution.

## Swapping in a real dataset later

Replace a call to `STREAMS[...]` with a loader that returns `(X, y,
drift_points)`. If you can get a real drift benchmark (Elec2 from OpenML, or
a Kaggle fraud/credit dataset with a known policy-change date you treat as
`drift_points`), everything downstream — detectors, pipeline, API, cost
tracker — works unchanged.

## Next steps (paper)

1. Add 1-2 more real-world-flavored synthetic streams (e.g. STAGGER) for
   breadth.
2. Report detection-delay / false-alarm / cost as a 3-way scatter per
   detector — that's your headline figure.
3. Write up as "Cost-aware drift detection and automated retraining for
   production ML" — target an applied-ML/systems workshop track
   (IEEE Big Data / Cloud workshops, or a student symposium) given the
   timeline.

## Next steps (resume/job)

1. Deploy `src/api.py` per `deploy/README.md`, get a public URL.
2. Record a 60-90s demo: hit `/predict` with a drifting stream, show
   `/status` reporting a version bump + drift event + cost.
3. Resume line: "Built and deployed an MLOps pipeline with automated
   concept-drift detection (DDM/EDDM/ADWIN/KSWIN) and self-healing
   retraining; benchmarked detection latency vs. false-alarm rate vs.
   cloud cost across 4 algorithms."
