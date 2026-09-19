"""Four published drift-detection algorithms, implemented from scratch (no `river`
dependency available in this environment). Each detector consumes a stream of
binary correctness indicators (1 = correct prediction, 0 = error) one at a time via
`update(value)` and reports whether it is in a warning or drift state.
"""
from collections import deque
import math
import numpy as np
from scipy.stats import ks_2samp


class DDM:
    """Drift Detection Method (Gama et al., 2004)."""

    def __init__(self, warning_level=2.0, drift_level=3.0, min_instances=30):
        self.warning_level = warning_level
        self.drift_level = drift_level
        self.min_instances = min_instances
        self.reset()

    def reset(self):
        self.n = 0
        self.p = 1.0
        self.s = 0.0
        self.p_min = float("inf")
        self.s_min = float("inf")

    def update(self, correct: int):
        error = 1 - correct
        self.n += 1
        self.p += (error - self.p) / self.n
        self.s = math.sqrt(self.p * (1 - self.p) / self.n) if self.n > 0 else 0.0

        warning, drift = False, False
        if self.n < self.min_instances:
            return warning, drift

        if self.p + self.s < self.p_min + self.s_min:
            self.p_min, self.s_min = self.p, self.s

        if self.p + self.s > self.p_min + self.drift_level * self.s_min:
            drift = True
            self.reset()
        elif self.p + self.s > self.p_min + self.warning_level * self.s_min:
            warning = True
        return warning, drift


class EDDM:
    """Early Drift Detection Method (Baena-Garcia et al., 2006)."""

    def __init__(self, alpha=0.95, beta=0.90, min_errors=30):
        self.alpha = alpha
        self.beta = beta
        self.min_errors = min_errors
        self.reset()

    def reset(self):
        self.n = 0
        self.n_errors = 0
        self.last_error_pos = -1
        self.mean_dist = 0.0
        self.std_dist = 0.0
        self.m2 = 0.0
        self.max_mean_plus_2std = 0.0

    def update(self, correct: int):
        self.n += 1
        warning, drift = False, False
        if correct == 0:
            self.n_errors += 1
            if self.last_error_pos >= 0:
                dist = self.n - self.last_error_pos
                delta = dist - self.mean_dist
                self.mean_dist += delta / self.n_errors
                self.m2 += delta * (dist - self.mean_dist)
                self.std_dist = math.sqrt(self.m2 / self.n_errors) if self.n_errors > 0 else 0.0
            self.last_error_pos = self.n

            if self.n_errors >= self.min_errors:
                cur = self.mean_dist + 2 * self.std_dist
                if cur > self.max_mean_plus_2std:
                    self.max_mean_plus_2std = cur
                ratio = cur / self.max_mean_plus_2std if self.max_mean_plus_2std > 0 else 1.0
                if ratio < self.beta:
                    drift = True
                    self.reset()
                elif ratio < self.alpha:
                    warning = True
        return warning, drift


class ADWIN:
    """Adaptive Windowing (Bifet & Gavalda, 2007), simplified exponential-histogram
    implementation: buckets are merged once more than `max_buckets` share a capacity,
    keeping memory sub-linear while approximating the full algorithm's cut-point search.
    """

    def __init__(self, delta=0.002, max_buckets=5):
        self.delta = delta
        self.max_buckets = max_buckets
        self.reset()

    def reset(self):
        # each bucket: [capacity, count_filled, sum, sum_sq]
        self.buckets = []  # list of lists per capacity-exponent level
        self.total = 0.0
        self.total_sq = 0.0
        self.width = 0

    def _insert_bucket(self, value):
        self.buckets.insert(0, {"cap": 1, "n": 1, "sum": value, "sumsq": value * value})
        self.total += value
        self.total_sq += value * value
        self.width += 1
        self._compress()

    def _compress(self):
        i = 0
        while i < len(self.buckets):
            same_cap = [j for j, b in enumerate(self.buckets) if b["cap"] == self.buckets[i]["cap"]]
            if len(same_cap) > self.max_buckets:
                j1, j2 = same_cap[-2], same_cap[-1]
                b1, b2 = self.buckets[j1], self.buckets[j2]
                merged = {"cap": b1["cap"] * 2, "n": b1["n"] + b2["n"],
                          "sum": b1["sum"] + b2["sum"], "sumsq": b1["sumsq"] + b2["sumsq"]}
                for idx in sorted([j1, j2], reverse=True):
                    del self.buckets[idx]
                self.buckets.append(merged)
                self.buckets.sort(key=lambda b: b["cap"])
                i = 0
            else:
                i += 1

    def _drop_oldest(self, n_drop):
        removed_sum, removed_sumsq, removed_n = 0.0, 0.0, 0
        while n_drop > 0 and self.buckets:
            b = self.buckets[-1]
            if b["n"] <= n_drop:
                removed_sum += b["sum"]
                removed_sumsq += b["sumsq"]
                removed_n += b["n"]
                n_drop -= b["n"]
                self.buckets.pop()
            else:
                break
        self.total -= removed_sum
        self.total_sq -= removed_sumsq
        self.width -= removed_n

    def update(self, value):
        self._insert_bucket(value)
        drift = False
        if self.width < 10:
            return False, drift

        n0 = 0
        sum0 = 0.0
        for b in reversed(self.buckets):
            n0 += b["n"]
            sum0 += b["sum"]
            n1 = self.width - n0
            if n1 <= 5 or n0 <= 5:
                continue
            sum1 = self.total - sum0
            mean0, mean1 = sum0 / n0, sum1 / n1
            m = 1.0 / (1.0 / n0 + 1.0 / n1)
            delta_prime = self.delta / self.width
            epsilon = math.sqrt((1.0 / (2 * m)) * math.log(4 / max(delta_prime, 1e-12)))
            if abs(mean0 - mean1) > epsilon:
                self._drop_oldest(n0)
                drift = True
                break
        return False, drift

    @property
    def estimated_mean(self):
        return self.total / self.width if self.width else 0.0


class KSWIN:
    """Kolmogorov-Smirnov Windowing (Raab et al., 2020)."""

    def __init__(self, alpha=0.005, window_size=200, stat_size=60):
        self.alpha = alpha
        self.window_size = window_size
        self.stat_size = stat_size
        self.window = deque(maxlen=window_size)

    def update(self, value):
        self.window.append(value)
        warning, drift = False, False
        if len(self.window) < self.window_size:
            return warning, drift
        w = list(self.window)
        reference = w[: self.stat_size]
        recent = w[-self.stat_size:]
        stat, p_value = ks_2samp(reference, recent, method="asymp")
        if p_value < self.alpha and stat > 0.1:
            drift = True
            for _ in range(self.window_size - self.stat_size):
                self.window.popleft()
        return warning, drift


DETECTORS = {
    "DDM": DDM,
    "EDDM": EDDM,
    "ADWIN": ADWIN,
    "KSWIN": KSWIN,
}
