"""Measures actual per-sample wall-clock cost of each detector's update() call,
for the computational-complexity section. Real measurements, not Big-O guesses.
"""
import csv
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.drift_detectors import DETECTORS

N = 20000
N_REPEATS = 5


def main():
    rows = []
    for name, cls in DETECTORS.items():
        times = []
        for _ in range(N_REPEATS):
            detector = cls()
            signal = [1] * N
            for i in range(N // 2, N):
                signal[i] = 0 if i % 3 == 0 else 1
            start = time.perf_counter()
            for v in signal:
                detector.update(v)
            times.append(time.perf_counter() - start)
        mean_total = sum(times) / len(times)
        rows.append({
            "detector": name,
            "n_samples": N,
            "mean_total_seconds": mean_total,
            "mean_us_per_sample": (mean_total / N) * 1e6,
        })
        print(f"{name:6s}  total={mean_total*1000:.1f}ms over {N} samples  "
              f"({(mean_total/N)*1e6:.2f} us/sample)")

    out_path = os.path.join(os.path.dirname(__file__), "..", "results", "timing.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
