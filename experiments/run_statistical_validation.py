"""Multi-seed statistical validation: repeats every (stream, detector) run across
independent seeds and reports mean +/- std, plus a paired significance test
(Wilcoxon signed-rank) between the best detector (ADWIN) and each alternative
on matched seeds. A single run (as in run_comparison.py) cannot support a
statistical-significance claim; this can.
"""
import csv
import os
import sys

import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.stream_generator import STREAMS
from src.drift_detectors import DETECTORS
from src.pipeline import run_pipeline

N_SEEDS = 10


def main():
    rows = []
    raw_rows = []  # long-format per-seed raw data, for full transparency/reproducibility
    raw_acc = {}   # (stream, detector) -> list of accuracy per seed, for the significance test
    raw_cost = {}  # (stream, detector) -> list of cost per seed, for the significance test

    for stream_name, gen_fn in STREAMS.items():
        for detector_name, detector_cls in DETECTORS.items():
            accs, delays, false_alarms_list, costs, detected_fracs = [], [], [], [], []
            for seed in range(N_SEEDS):
                X, y, drift_points = gen_fn(seed=seed)
                detector = detector_cls()
                metrics = run_pipeline(X, y, drift_points, detector)
                accs.append(metrics["overall_accuracy"])
                false_alarms_list.append(metrics["false_alarms"])
                costs.append(metrics["cost"]["total_cost_usd"])
                detected_fracs.append(metrics["detected_drifts"] / metrics["true_drifts"])
                if metrics["mean_detection_delay"] is not None:
                    delays.append(metrics["mean_detection_delay"])
                raw_rows.append({
                    "stream": stream_name, "detector": detector_name, "seed": seed,
                    "accuracy": metrics["overall_accuracy"],
                    "detected_drifts": metrics["detected_drifts"], "true_drifts": metrics["true_drifts"],
                    "false_alarms": metrics["false_alarms"], "cost_usd": metrics["cost"]["total_cost_usd"],
                })

            raw_acc[(stream_name, detector_name)] = accs
            raw_cost[(stream_name, detector_name)] = costs
            rows.append({
                "stream": stream_name,
                "detector": detector_name,
                "n_seeds": N_SEEDS,
                "acc_mean": np.mean(accs), "acc_std": np.std(accs, ddof=1),
                "detected_frac_mean": np.mean(detected_fracs), "detected_frac_std": np.std(detected_fracs, ddof=1),
                "false_alarms_mean": np.mean(false_alarms_list), "false_alarms_std": np.std(false_alarms_list, ddof=1),
                "delay_mean": np.mean(delays) if delays else None,
                "cost_mean": np.mean(costs), "cost_std": np.std(costs, ddof=1),
            })
            print(f"{stream_name:20s} {detector_name:6s} "
                  f"acc={np.mean(accs):.3f}+/-{np.std(accs, ddof=1):.3f}  "
                  f"detected={np.mean(detected_fracs):.2f}  "
                  f"cost=${np.mean(costs):.3f}+/-{np.std(costs, ddof=1):.3f}")

    out_path = os.path.join(os.path.dirname(__file__), "..", "results", "statistical_validation.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved per-condition stats to {out_path}")

    raw_path = os.path.join(os.path.dirname(__file__), "..", "results", "statistical_validation_raw.csv")
    with open(raw_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=raw_rows[0].keys())
        writer.writeheader()
        writer.writerows(raw_rows)
    print(f"Saved raw per-seed data to {raw_path}")

    def _paired_test(a, b):
        if np.all(np.array(a) == np.array(b)):
            return float("nan"), 1.0
        return wilcoxon(a, b)

    # Paired significance: ADWIN vs each other detector, matched by seed, per stream,
    # on both accuracy (does it detect better?) and cost (does it cost less?)
    sig_rows = []
    for stream_name in STREAMS:
        adwin_accs = raw_acc[(stream_name, "ADWIN")]
        adwin_costs = raw_cost[(stream_name, "ADWIN")]
        for other in ["DDM", "EDDM", "KSWIN"]:
            other_accs = raw_acc[(stream_name, other)]
            other_costs = raw_cost[(stream_name, other)]
            acc_stat, acc_p = _paired_test(adwin_accs, other_accs)
            cost_stat, cost_p = _paired_test(adwin_costs, other_costs)
            sig_rows.append({
                "stream": stream_name, "comparison": f"ADWIN vs {other}",
                "adwin_mean_acc": np.mean(adwin_accs), "other_mean_acc": np.mean(other_accs),
                "acc_wilcoxon_p": acc_p, "acc_significant_at_0.05": acc_p < 0.05,
                "adwin_mean_cost": np.mean(adwin_costs), "other_mean_cost": np.mean(other_costs),
                "cost_wilcoxon_p": cost_p, "cost_significant_at_0.05": cost_p < 0.05,
            })

    sig_path = os.path.join(os.path.dirname(__file__), "..", "results", "significance_tests.csv")
    with open(sig_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sig_rows[0].keys())
        writer.writeheader()
        writer.writerows(sig_rows)
    print(f"Saved significance tests to {sig_path}")
    for r in sig_rows:
        print(f"{r['stream']:20s} {r['comparison']:16s} "
              f"acc_p={r['acc_wilcoxon_p']:.4f} (sig={r['acc_significant_at_0.05']})  "
              f"cost_p={r['cost_wilcoxon_p']:.4f} (sig={r['cost_significant_at_0.05']})")


if __name__ == "__main__":
    main()
