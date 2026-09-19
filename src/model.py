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
