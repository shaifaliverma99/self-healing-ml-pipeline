"""Online simulation: feed a stream through the model one sample at a time,
run a drift detector on the correctness signal, and auto-retrain + hot-swap the
model on a drift signal. This is the core "self-healing" loop the API also runs.
"""
from collections import deque
import numpy as np

from src.model import OnlineModel
from src.cost_tracker import CostTracker
from src.diagnosis import diagnose


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


def run_pipeline_with_diagnosis(X, y, drift_points, detector, init_size=500, buffer_size=500,
                                 cooldown=200, eval_window=250, val_frac=0.2, min_buffer=50):
    """Extends run_pipeline with two additional layers absent from it:

    Diagnosis: on every retrain attempt, fits a shallow decision tree to the
    buffer's (features -> was this prediction wrong) relationship and reports
    the most-implicated feature, an interpretable proxy for "where" the drift
    shows up given that these streams shift P(Y|X) rather than P(X) itself.

    Validation gate: a candidate retrain is only committed if it scores at
    least as well as the currently-served model on a held-out slice of the
    buffer; otherwise it is discarded and the old model keeps serving. This
    is the "does recovery actually help before we commit to it" check that
    run_pipeline's unconditional retrain() does not perform.
    """
    n_samples, n_features = X.shape
    model = OnlineModel(n_features=n_features)
    model.fit_initial(X[:init_size], y[:init_size])

    cost = CostTracker()
    buffer_X, buffer_y, buffer_err = (deque(maxlen=buffer_size), deque(maxlen=buffer_size),
                                       deque(maxlen=buffer_size))
    correctness_log = np.zeros(n_samples, dtype=int)
    flagged_drifts = []
    last_retrain_step = -cooldown
    diagnosis_log = []
    n_attempts = 0
    n_committed = 0

    for i in range(init_size, n_samples):
        x_i, y_i = X[i], y[i]
        pred = model.predict(x_i)
        correct = int(pred == y_i)
        correctness_log[i] = correct
        cost.record_inference()

        buffer_X.append(x_i); buffer_y.append(y_i); buffer_err.append(1 - correct)

        _, drift = detector.update(correct)
        if drift and (i - last_retrain_step) > cooldown and len(buffer_X) >= min_buffer:
            flagged_drifts.append(i)
            n_attempts += 1
            last_retrain_step = i

            diag = diagnose(np.array(buffer_X), np.array(buffer_err))
            diagnosis_log.append({"step": i, "diagnosis": diag})

            proposal = model.propose_retrain(np.array(buffer_X), np.array(buffer_y), val_frac=val_frac)
            cost.record_retrain()  # the candidate fit + validation pass is real compute, spent whether or not we keep it
            if proposal is not None and proposal["val_acc_new"] >= proposal["val_acc_old"]:
                model.commit(proposal["candidate"])
                n_committed += 1

    metrics = _compute_metrics(correctness_log, drift_points, flagged_drifts,
                                init_size, n_samples, eval_window)
    metrics["cost"] = cost.summary()
    metrics["flagged_drifts"] = flagged_drifts
    metrics["model_versions"] = model.version
    metrics["retrain_attempts"] = n_attempts
    metrics["retrains_committed"] = n_committed
    metrics["retrains_rejected"] = n_attempts - n_committed
    metrics["diagnosis_log"] = diagnosis_log
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
