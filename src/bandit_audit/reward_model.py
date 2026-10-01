"""Cross-fitting a deliberately misspecified, action-specific linear reward model."""

import numpy as np

from .estimators import importance_weights
from .policies import Log


def folds(n: int, count: int, seed: int) -> np.ndarray:
    """Outcome-independent, balanced fold assignment."""
    if not isinstance(count, int) or count < 2 or count > n:
        raise ValueError("fold count must be an integer between 2 and sample size")
    rng = np.random.default_rng(seed)
    assignment = np.empty(n, dtype=int)
    assignment[rng.permutation(n)] = np.arange(n) % count
    return assignment


def cross_fit(log: Log, context_features: np.ndarray, *, count: int = 3,
              seed: int = 71, ridge: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Never use a held-out row's reward to construct that row's predictions.

    Fit a separate intercept and slope per action, penalizing slopes only.
    The nonlinear reward surfaces are intentionally absent from these features.
    """
    importance_weights(log)
    if not np.isfinite(ridge) or ridge <= 0:
        raise ValueError("ridge must be positive and finite")
    features = np.asarray(context_features, dtype=float)
    if features.ndim != 1 or not np.isfinite(features).all():
        raise ValueError("context_features must be a finite vector")
    ids = np.asarray(log.contexts)
    if ids.dtype.kind not in "iu" or np.any(ids < 0) or np.any(ids >= len(features)):
        raise ValueError("context indices out of range")
    assignment = folds(len(ids), count, seed)
    design = np.column_stack([np.ones(len(ids)), features[ids]])
    predictions = np.empty_like(log.target)
    penalty = np.diag([0.0, ridge])
    for fold in range(count):
        heldout = assignment == fold
        for action in range(log.target.shape[1]):
            train = (~heldout) & (log.actions == action)
            if train.sum() < 2:
                raise ValueError("fewer than two training rewards for an action in a fold")
            x_train = design[train]
            beta = np.linalg.solve(x_train.T @ x_train + penalty,
                                   x_train.T @ log.rewards[train])
            predictions[heldout, action] = np.clip(design[heldout] @ beta, 0, 1)
    return predictions, assignment
