"""Evaluates the two new layers absent from the original pipeline:

1. Diagnosis accuracy: on SEA specifically, feature index 2 is a pure decoy
   (not part of the labelling rule, unlike features 0-1) -- so a correct
   diagnosis should rank it last, not first. This is checked against ground
   truth we control by construction, not assumed.
2. Validation gate: how many retrain attempts are committed vs. rejected,
   and whether gating changes accuracy/cost relative to the original
   ungated ADWIN results already in results/statistical_validation.csv.

Runs ADWIN only (the detector Section VIII already establishes as the best
completeness/cost operating point), across the same 10 seeds as the rest of
the statistical validation, so results are directly comparable.
"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.stream_generator import STREAMS
from src.drift_detectors import ADWIN
from src.pipeline import run_pipeline_with_diagnosis

N_SEEDS = 10
SEA_DECOY_FEATURE = 2  # sea_stream's label rule uses only features 0 and 1


def main():
    rows = []
    sea_diag_rows = []

    for stream_name, gen_fn in STREAMS.items():
        accs, costs, attempts, committed, rejected = [], [], [], [], []
        for seed in range(N_SEEDS):
            X, y, drift_points = gen_fn(seed=seed)
            m = run_pipeline_with_diagnosis(X, y, drift_points, ADWIN())
            accs.append(m["overall_accuracy"])
            costs.append(m["cost"]["total_cost_usd"])
            attempts.append(m["retrain_attempts"])
            committed.append(m["retrains_committed"])
            rejected.append(m["retrains_rejected"])

            if stream_name == "sea_abrupt":
                for entry in m["diagnosis_log"]:
                    diag = entry["diagnosis"]
                    if diag is None:
                        continue
                    sea_diag_rows.append({
                        "seed": seed, "step": entry["step"],
                        "top_feature": diag["top_feature"],
                        "decoy_ranked_last": diag["ranking"][-1] == SEA_DECOY_FEATURE,
                        "decoy_is_top": diag["top_feature"] == SEA_DECOY_FEATURE,
                    })

        rows.append({
            "stream": stream_name, "n_seeds": N_SEEDS,
            "acc_mean": np.mean(accs), "acc_std": np.std(accs, ddof=1),
            "cost_mean": np.mean(costs), "cost_std": np.std(costs, ddof=1),
            "retrain_attempts_mean": np.mean(attempts),
            "retrains_committed_mean": np.mean(committed),
            "retrains_rejected_mean": np.mean(rejected),
            "commit_rate": np.sum(committed) / max(np.sum(attempts), 1),
        })
        print(f"{stream_name:20s} acc={np.mean(accs):.3f}+/-{np.std(accs,ddof=1):.3f} "
              f"cost=${np.mean(costs):.3f}+/-{np.std(costs,ddof=1):.3f} "
              f"attempts={np.mean(attempts):.1f} committed={np.mean(committed):.1f} "
              f"rejected={np.mean(rejected):.1f} commit_rate={rows[-1]['commit_rate']:.2f}")

    out_path = os.path.join(os.path.dirname(__file__), "..", "results", "diagnosis_validation.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print(f"\nSaved {out_path}")

    if sea_diag_rows:
        n = len(sea_diag_rows)
        decoy_is_top_rate = sum(r["decoy_is_top"] for r in sea_diag_rows) / n
        decoy_ranked_last_rate = sum(r["decoy_ranked_last"] for r in sea_diag_rows) / n
        print(f"\nSEA diagnosis events: {n}")
        print(f"  decoy feature (2) incorrectly ranked TOP: {decoy_is_top_rate:.2f}")
        print(f"  decoy feature (2) correctly ranked LAST: {decoy_ranked_last_rate:.2f}")

        diag_path = os.path.join(os.path.dirname(__file__), "..", "results", "diagnosis_accuracy.csv")
        with open(diag_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=sea_diag_rows[0].keys())
            w.writeheader(); w.writerows(sea_diag_rows)
        print(f"Saved {diag_path}")


if __name__ == "__main__":
    main()
