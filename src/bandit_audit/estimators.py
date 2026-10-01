"""Value estimators with explicit support and numerical validation."""

import numpy as np

from .policies import Log, check_support


def importance_weights(log: Log) -> np.ndarray:
    check_support(log.target, log.logging)
    n, n_actions = log.target.shape
    actions = np.asarray(log.actions)
    rewards = np.asarray(log.rewards)
    contexts = np.asarray(log.contexts)
    if actions.shape != (n,) or rewards.shape != (n,) or contexts.shape != (n,):
        raise ValueError("all logged columns must match the policy row count")
    if actions.dtype.kind not in "iu" or np.any(actions < 0) or np.any(actions >= n_actions):
        raise ValueError("actions must be valid integer column indices")
    if not np.isfinite(rewards).all() or np.any(rewards < 0) or np.any(rewards > 1):
        raise ValueError("this Bernoulli-reward lab expects finite rewards in [0, 1]")
    logged = log.logging[np.arange(n), actions]
    if np.any(logged <= 0):
        raise ValueError("an observed action has zero declared logging probability")
    return log.target[np.arange(n), actions] / logged


def evaluate(log: Log, reward_predictions: np.ndarray, clip: float = 10.0) -> dict:
    """Compute fixed-policy values; predictions must be trained independently/OOF."""
    if not np.isfinite(clip) or clip <= 0:
        raise ValueError("clip must be positive and finite")
    weights = importance_weights(log)
    q = np.asarray(reward_predictions, dtype=float)
    if q.shape != log.target.shape or not np.isfinite(q).all():
        raise ValueError("reward predictions must be finite and match policy shape")
    if weights.sum() <= 0:
        raise ValueError("no observed target-policy mass; SNIPS is undefined")
    baseline = np.sum(log.target * q, axis=1)
    observed_predictions = q[np.arange(len(weights)), log.actions]
    ips_scores = weights * log.rewards
    dr_scores = baseline + weights * (log.rewards - observed_predictions)
    return {
        "estimates": {
            "logged_mean": float(np.mean(log.rewards)),
            "direct": float(np.mean(baseline)),
            "ips": float(np.mean(ips_scores)),
            "snips": float(np.sum(ips_scores) / weights.sum()),
            "dr": float(np.mean(dr_scores)),
            "clipped_ips": float(np.mean(np.minimum(weights, clip) * log.rewards)),
        },
        "diagnostics": {
            "ess": float(weights.sum() ** 2 / np.sum(weights**2)),
            "max_weight": float(weights.max()),
            "mean_weight": float(weights.mean()),
            "clipped_row_fraction": float(np.mean(weights > clip)),
        },
    }
