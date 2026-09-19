"""Online simulation: feed a stream through the model one sample at a time,
run a drift detector on the correctness signal, and auto-retrain + hot-swap the
model on a drift signal. This is the core "self-healing" loop the API also runs.
"""
from collections import deque
import numpy as np

from src.model import OnlineModel
from src.cost_tracker import CostTracker


def run_pipeline(X, y, drift_points, detector, init_size=500, buffer_size=500,
                  cooldown=200, eval_window=250):
    n_samples, n_features = X.shape
    model = OnlineModel(n_features=n_features)
    model.fit_initial(X[:init_size], y[:init_size])

    cost = CostTracker()
    buffer_X, buffer_y = deque(maxlen=buffer_size), deque(maxlen=buffer_size)
    correctness_log = np.zeros(n_samples, dtype=int)
    flagged_drifts = []
    last_retrain_step = -cooldown

    for i in range(init_size, n_samples):
        x_i, y_i = X[i], y[i]
        pred = model.predict(x_i)
        correct = int(pred == y_i)
        correctness_log[i] = correct
        cost.record_inference()

        buffer_X.append(x_i)
        buffer_y.append(y_i)

        _, drift = detector.update(correct)
        if drift and (i - last_retrain_step) > cooldown:
            flagged_drifts.append(i)
            if model.retrain(np.array(buffer_X), np.array(buffer_y)):
                cost.record_retrain()
                last_retrain_step = i

    metrics = _compute_metrics(correctness_log, drift_points, flagged_drifts,
                                init_size, n_samples, eval_window)
    metrics["cost"] = cost.summary()
    metrics["flagged_drifts"] = flagged_drifts
    metrics["model_versions"] = model.version
    return metrics


def _compute_metrics(correctness_log, drift_points, flagged_drifts, init_size, n_samples, eval_window):
    overall_acc = correctness_log[init_size:].mean()

    delays = []
    matched_flags = set()
    for dp in drift_points:
        candidates = [f for f in flagged_drifts if f >= dp and f not in matched_flags]
        if candidates:
            f = min(candidates)
            delays.append(f - dp)
            matched_flags.add(f)
        else:
            delays.append(None)
    detected = sum(1 for d in delays if d is not None)
    mean_delay = np.mean([d for d in delays if d is not None]) if detected else None

    false_alarms = len(flagged_drifts) - detected

    windowed_acc = []
    for start in range(init_size, n_samples, eval_window):
        end = min(start + eval_window, n_samples)
        windowed_acc.append(round(float(correctness_log[start:end].mean()), 4))

    return {
        "overall_accuracy": round(float(overall_acc), 4),
        "true_drifts": len(drift_points),
        "detected_drifts": detected,
        "mean_detection_delay": None if mean_delay is None else round(float(mean_delay), 1),
        "false_alarms": max(false_alarms, 0),
        "n_flagged": len(flagged_drifts),
        "windowed_accuracy": windowed_acc,
    }
