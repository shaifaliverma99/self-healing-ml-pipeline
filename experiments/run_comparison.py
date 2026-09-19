"""Runs every drift detector against every synthetic stream and writes a results
table -- this is the core empirical study for the paper: detection delay,
false-alarm rate, accuracy, and simulated cloud cost, per (stream, detector) pair.
"""
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.stream_generator import STREAMS
from src.drift_detectors import DETECTORS
from src.pipeline import run_pipeline


def main():
    rows = []
    for stream_name, gen_fn in STREAMS.items():
        X, y, drift_points = gen_fn()
        for detector_name, detector_cls in DETECTORS.items():
            detector = detector_cls()
            metrics = run_pipeline(X, y, drift_points, detector)
            rows.append({
                "stream": stream_name,
                "detector": detector_name,
                "true_drifts": metrics["true_drifts"],
                "detected_drifts": metrics["detected_drifts"],
                "mean_detection_delay": metrics["mean_detection_delay"],
                "false_alarms": metrics["false_alarms"],
                "overall_accuracy": metrics["overall_accuracy"],
                "model_versions": metrics["model_versions"],
                "total_cost_usd": metrics["cost"]["total_cost_usd"],
            })
            print(f"{stream_name:20s} {detector_name:6s} "
                  f"acc={metrics['overall_accuracy']:.3f} "
                  f"detected={metrics['detected_drifts']}/{metrics['true_drifts']} "
                  f"delay={metrics['mean_detection_delay']} "
                  f"false_alarms={metrics['false_alarms']} "
                  f"cost=${metrics['cost']['total_cost_usd']}")

    out_path = os.path.join(os.path.dirname(__file__), "..", "results", "detector_comparison.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
