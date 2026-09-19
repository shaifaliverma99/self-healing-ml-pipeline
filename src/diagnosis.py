"""Failure diagnosis: identifies which input feature is most associated with
the model's recent errors.

Concept-drift streams (SEA, SINE, the rotating hyperplane) change the label
function P(Y|X), not the input distribution P(X) -- so a feature-drift
detector that compares P(X) before/after finds nothing, even though the
model is clearly failing. What *is* detectable is that errors become
concentrated in particular regions of feature space once the labelling rule
changes. A shallow decision tree fit to predict "was this prediction wrong"
from the raw features gives an interpretable proxy for which feature that
concentration lives in, via its feature_importances_.
"""
import numpy as np
from sklearn.tree import DecisionTreeClassifier


def diagnose(X, errors, max_depth=3, random_state=0):
    """X: (n, d) feature buffer. errors: (n,) 1 if the prediction on that
    sample was wrong, 0 if correct. Returns None if there is no variation to
    split on (all correct or all wrong), otherwise a ranking of feature
    indices by how strongly they separate correct from incorrect predictions.
    """
    errors = np.asarray(errors)
    if len(np.unique(errors)) < 2:
        return None
    tree = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    tree.fit(X, errors)
    importances = tree.feature_importances_
    ranking = np.argsort(importances)[::-1]
    top = int(ranking[0])
    if importances[top] <= 0:
        return None
    return {
        "ranking": ranking.tolist(),
        "importances": importances.tolist(),
        "top_feature": top,
    }
