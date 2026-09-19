"""Incrementally-servable classifier used by the pipeline and the API."""
from sklearn.linear_model import SGDClassifier
import numpy as np


class OnlineModel:
    def __init__(self, n_features: int, random_state: int = 0):
        self.n_features = n_features
        self.random_state = random_state
        self.clf = SGDClassifier(loss="log_loss", random_state=random_state)
        self.classes = np.array([0, 1])
        self._fitted = False
        self.version = 0

    def fit_initial(self, X, y):
        self.clf.fit(X, y)
        self._fitted = True

    def predict(self, x):
        if not self._fitted:
            return 0
        return int(self.clf.predict(x.reshape(1, -1))[0])

    def retrain(self, X, y):
        """Fresh fit on a buffer of recent samples, then hot-swap in as the new version.
        Skipped if the buffer has collapsed to a single class (SGDClassifier requires >= 2)."""
        if len(np.unique(y)) < 2:
            return False
        new_clf = SGDClassifier(loss="log_loss", random_state=self.random_state)
        new_clf.fit(X, y)
        self.clf = new_clf
        self._fitted = True
        self.version += 1
        return True

    def propose_retrain(self, X, y, val_frac=0.2):
        """Fit a candidate on the older part of the buffer and score both the
        candidate and the currently-served model on the most recent slice
        (held out from training, chronologically last so it represents the
        post-drift distribution). Returns None if there isn't enough data or
        class variety to do this safely; otherwise a dict the caller uses to
        decide whether to commit() or discard the candidate.
        """
        n = len(y)
        n_val = max(int(n * val_frac), 1)
        if n - n_val < 2 or n_val < 1:
            return None
        X_train, y_train = X[:-n_val], y[:-n_val]
        X_val, y_val = X[-n_val:], y[-n_val:]
        if len(np.unique(y_train)) < 2:
            return None

        candidate = SGDClassifier(loss="log_loss", random_state=self.random_state)
        candidate.fit(X_train, y_train)

        val_acc_new = float((candidate.predict(X_val) == y_val).mean())
        val_acc_old = (float((self.clf.predict(X_val) == y_val).mean())
                       if self._fitted else 0.0)
        return {
            "candidate": candidate,
            "val_acc_new": val_acc_new,
            "val_acc_old": val_acc_old,
            "n_val": n_val,
        }

    def commit(self, candidate):
        self.clf = candidate
        self._fitted = True
        self.version += 1
