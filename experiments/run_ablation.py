"""Ablation study: isolates the contribution of detector-triggered retraining
against two simpler alternatives, across N_SEEDS seeds per stream:

  A0 no_retrain        -- fit once, never adapt (lower bound)
  A1 periodic_retrain  -- retrain every fixed interval regardless of drift (naive)
  A2 full_system       -- the proposed system: ADWIN-triggered retraining

This answers the question a single detector-vs-detector comparison cannot:
does *detecting* drift before retraining actually buy anything over just
retraining on a schedule, at matched or lower cost?
"""
import csv
import os
import sys
from collections import deque

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.stream_generator import STREAMS
from src.drift_detectors import ADWIN
from src.model import OnlineModel
from src.cost_tracker import CostTracker
from src.pipeline import run_pipeline

N_SEEDS = 10
INIT_SIZE = 500
BUFFER_SIZE = 500
PERIODIC_INTERVAL = 5000  # comparable order of magnitude to ADWIN's typical retrain count


def run_no_retrain(X, y):
    n_samples, n_features = X.shape
    model = OnlineModel(n_features=n_features)
    model.fit_initial(X[:INIT_SIZE], y[:INIT_SIZE])
    cost = CostTracker()
    correct = 0
    for i in range(INIT_SIZE, n_samples):
        pred = model.predict(X[i])
        correct += int(pred == y[i])
        cost.record_inference()
    return correct / (n_samples - INIT_SIZE), cost.summary()["total_cost_usd"], 0


def run_periodic_retrain(X, y, interval=PERIODIC_INTERVAL):
    n_samples, n_features = X.shape
    model = OnlineModel(n_features=n_features)
    model.fit_initial(X[:INIT_SIZE], y[:INIT_SIZE])
    cost = CostTracker()
    buffer_X, buffer_y = deque(maxlen=BUFFER_SIZE), deque(maxlen=BUFFER_SIZE)
    correct, n_retrains = 0, 0
    for i in range(INIT_SIZE, n_samples):
        pred = model.predict(X[i])
        correct += int(pred == y[i])
        cost.record_inference()
        buffer_X.append(X[i]); buffer_y.append(y[i])
        if (i - INIT_SIZE) > 0 and (i - INIT_SIZE) % interval == 0:
            if model.retrain(np.array(buffer_X), np.array(buffer_y)):
                cost.record_retrain()
                n_retrains += 1
    return correct / (n_samples - INIT_SIZE), cost.summary()["total_cost_usd"], n_retrains


def run_full_system(X, y, drift_points):
    metrics = run_pipeline(X, y, drift_points, ADWIN())
    return metrics["overall_accuracy"], metrics["cost"]["total_cost_usd"], metrics["model_versions"]


def main():
    rows = []
    for stream_name, gen_fn in STREAMS.items():
        conditions = {"A0_no_retrain": [], "A1_periodic_retrain": [], "A2_full_system_ADWIN": []}
        costs = {k: [] for k in conditions}
        retrains = {k: [] for k in conditions}

        for seed in range(N_SEEDS):
            X, y, drift_points = gen_fn(seed=seed)

            acc, cst, nr = run_no_retrain(X, y)
            conditions["A0_no_retrain"].append(acc); costs["A0_no_retrain"].append(cst); retrains["A0_no_retrain"].append(nr)

            acc, cst, nr = run_periodic_retrain(X, y)
            conditions["A1_periodic_retrain"].append(acc); costs["A1_periodic_retrain"].append(cst); retrains["A1_periodic_retrain"].append(nr)

            acc, cst, nr = run_full_system(X, y, drift_points)
            conditions["A2_full_system_ADWIN"].append(acc); costs["A2_full_system_ADWIN"].append(cst); retrains["A2_full_system_ADWIN"].append(nr)

        for cond_name, accs in conditions.items():
            rows.append({
                "stream": stream_name, "condition": cond_name, "n_seeds": N_SEEDS,
                "acc_mean": np.mean(accs), "acc_std": np.std(accs, ddof=1),
                "cost_mean": np.mean(costs[cond_name]), "cost_std": np.std(costs[cond_name], ddof=1),
                "n_retrains_mean": np.mean(retrains[cond_name]),
            })
            print(f"{stream_name:20s} {cond_name:22s} "
                  f"acc={np.mean(accs):.3f}+/-{np.std(accs, ddof=1):.3f}  "
                  f"cost=${np.mean(costs[cond_name]):.3f}+/-{np.std(costs[cond_name], ddof=1):.3f}  "
                  f"retrains={np.mean(retrains[cond_name]):.1f}")

    out_path = os.path.join(os.path.dirname(__file__), "..", "results", "ablation_study.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved ablation results to {out_path}")


if __name__ == "__main__":
    main()
