"""Synthetic concept-drift stream generators with known ground-truth drift points.

These are the standard benchmarks used throughout the drift-detection literature
(Bifet & Gavalda 2007, Gama et al. 2004, MOA/scikit-multiflow), chosen because they
give a labeled ground-truth drift point -- something real-world sets like Elec2
lack -- which is required to measure detection delay and false-alarm rate.
"""
import numpy as np


def sea_stream(n_samples=20000, drift_points=(5000, 10000, 15000), noise=0.1, seed=0):
    """SEA generator: 3 features in [0,10], label = 1 if f1+f2 > threshold.
    Threshold cycles through classic SEA values at each drift point (abrupt drift).
    """
    rng = np.random.default_rng(seed)
    thresholds = [8.0, 9.0, 7.0, 9.5]
    X = rng.uniform(0, 10, size=(n_samples, 3))
    y = np.zeros(n_samples, dtype=int)
    seg_idx = 0
    boundaries = list(drift_points) + [n_samples]
    start = 0
    for end in boundaries:
        thr = thresholds[seg_idx % len(thresholds)]
        y[start:end] = (X[start:end, 0] + X[start:end, 1] > thr).astype(int)
        start = end
        seg_idx += 1
    flip = rng.uniform(0, 1, size=n_samples) < noise
    y[flip] = 1 - y[flip]
    return X, y, list(drift_points)


def sine_stream(n_samples=20000, drift_points=(5000, 10000, 15000), noise=0.1, seed=0):
    """SINE1 generator: 2 features in [0,1], label = 1 if f2 > sin(f1).
    Decision function reverses at each drift point (abrupt drift).
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, size=(n_samples, 2))
    y = np.zeros(n_samples, dtype=int)
    boundaries = list(drift_points) + [n_samples]
    start = 0
    reversed_ = False
    for end in boundaries:
        base = (X[start:end, 1] > np.sin(X[start:end, 0] * np.pi)).astype(int)
        y[start:end] = 1 - base if reversed_ else base
        start = end
        reversed_ = not reversed_
    flip = rng.uniform(0, 1, size=n_samples) < noise
    y[flip] = 1 - y[flip]
    return X, y, list(drift_points)


def rotating_hyperplane_stream(n_samples=20000, drift_points=(7000, 14000), noise=0.1, seed=0,
                                n_features=4, drift_width=1500):
    """Rotating hyperplane: weights slowly rotate around each drift point (gradual drift),
    unlike the abrupt SEA/SINE streams above.
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, size=(n_samples, n_features))
    w = rng.uniform(-1, 1, size=n_features)
    y = np.zeros(n_samples, dtype=int)
    w_cur = w.copy()
    drift_targets = [rng.uniform(-1, 1, size=n_features) for _ in drift_points]
    dp_sorted = sorted(drift_points)
    for i in range(n_samples):
        active = None
        for dp, target in zip(dp_sorted, drift_targets):
            if dp - drift_width // 2 <= i <= dp + drift_width // 2:
                active = (dp, target)
                break
        if active is not None:
            dp, target = active
            t = (i - (dp - drift_width // 2)) / drift_width
            t = min(max(t, 0.0), 1.0)
            w_cur = (1 - t) * w + t * target
        score = np.dot(X[i], w_cur)
        y[i] = int(score > np.sum(w_cur) / 2)
    flip = rng.uniform(0, 1, size=n_samples) < noise
    y[flip] = 1 - y[flip]
    return X, y, list(dp_sorted)


STREAMS = {
    "sea_abrupt": sea_stream,
    "sine_abrupt": sine_stream,
    "hyperplane_gradual": rotating_hyperplane_stream,
}
