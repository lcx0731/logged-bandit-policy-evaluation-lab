"""Policy validation, support checks, and a fully specified synthetic world."""

from dataclasses import dataclass

import numpy as np


def probabilities(matrix: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or min(matrix.shape) < 1:
        raise ValueError(f"{name} must be a nonempty probability matrix")
    if not np.isfinite(matrix).all() or np.any(matrix < 0) or np.any(matrix > 1):
        raise ValueError(f"{name} entries must be finite and in [0, 1]")
    if not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-12, rtol=0):
        raise ValueError(f"{name} rows must sum to one")
    return matrix


def check_support(target: np.ndarray, logging: np.ndarray) -> None:
    """Reject any target-positive action with exactly zero logging probability."""
    target = probabilities(target, "target policy")
    logging = probabilities(logging, "logging policy")
    if target.shape != logging.shape:
        raise ValueError("target and logging policies must have identical shapes")
    unsupported = (target > 0) & (logging == 0)
    if unsupported.any():
        raise ValueError(
            "Unsupported target policy: positive target mass at zero logging "
            f"probability in {int(unsupported.sum())} context-action cells"
        )


@dataclass(frozen=True)
class World:
    context_mass: np.ndarray
    features: np.ndarray
    reward_means: np.ndarray
    target: np.ndarray

    @property
    def value(self) -> float:
        """Exact population value, not an estimate from a Monte Carlo holdout."""
        return float(np.sum(self.context_mass[:, None] * self.target * self.reward_means))

    def logger(self, exploration: float) -> np.ndarray:
        if not np.isfinite(exploration) or not 0 <= exploration <= 1:
            raise ValueError("exploration must be in [0, 1]")
        policy = np.full_like(self.target, exploration / self.target.shape[1])
        policy[:, 0] += 1 - exploration
        return policy


def make_world() -> World:
    """Eight contexts and three actions; only the chosen Bernoulli reward is logged."""
    x = np.linspace(-1, 1, 8)
    mass = np.array([0.06, 0.10, 0.14, 0.20, 0.20, 0.14, 0.10, 0.06])
    means = np.column_stack([
        0.30 + 0.15 * x,
        0.35 + 0.35 * (1 - x**2),
        0.50 + 0.25 * np.sin(np.pi * x),
    ])
    # Prespecified, not selected using evaluation rewards: prefer action 1 on
    # negative contexts and action 2 on positive contexts, with 10% exploration.
    target = np.full((8, 3), 0.10 / 3)
    target[np.arange(8), np.where(x < 0, 1, 2)] += 0.90
    return World(mass, x, means, target)


@dataclass(frozen=True)
class Log:
    contexts: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    logging: np.ndarray
    target: np.ndarray


def sample_log(world: World, exploration: float, size: int, seed: int) -> Log:
    if not isinstance(size, int) or size < 1:
        raise ValueError("size must be a positive integer")
    logger = world.logger(exploration)
    # Check the whole known context universe, not just contexts appearing in a log.
    check_support(world.target, logger)
    rng = np.random.default_rng(seed)
    contexts = rng.choice(len(world.context_mass), size=size, p=world.context_mass)
    propensities = logger[contexts]
    cumulative = propensities.cumsum(axis=1)
    cumulative[:, -1] = 1.0
    actions = (rng.random(size)[:, None] > cumulative).sum(axis=1)
    rewards = rng.binomial(1, world.reward_means[contexts, actions]).astype(float)
    return Log(contexts, actions, rewards, propensities, world.target[contexts])
