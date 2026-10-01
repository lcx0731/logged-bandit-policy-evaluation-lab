from dataclasses import replace

import numpy as np
import pytest

from bandit_audit.policies import make_world, sample_log
from bandit_audit.reward_model import cross_fit, folds


def test_heldout_rewards_cannot_change_their_own_predictions():
    world = make_world()
    log = sample_log(world, .6, 800, 12)
    q, assignment = cross_fit(log, world.features, seed=18)
    modified = log.rewards.copy()
    modified[assignment == 0] = 1 - modified[assignment == 0]
    q_changed, same_assignment = cross_fit(replace(log, rewards=modified),
                                          world.features, seed=18)
    np.testing.assert_array_equal(assignment, same_assignment)
    np.testing.assert_array_equal(q[assignment == 0], q_changed[assignment == 0])
    assert np.isfinite(q).all() and np.all((q >= 0) & (q <= 1))


def test_folds_cover_every_row_and_are_balanced():
    assignment = folds(101, 3, 42)
    counts = np.bincount(assignment)
    assert len(assignment) == 101
    assert set(assignment) == {0, 1, 2}
    assert counts.max() - counts.min() <= 1
    np.testing.assert_array_equal(assignment, folds(101, 3, 42))


def test_missing_action_training_data_fails_explicitly():
    world = make_world()
    log = sample_log(world, .6, 100, 5)
    log = replace(log, actions=np.zeros(100, dtype=int))
    with pytest.raises(ValueError, match="fewer than two"):
        cross_fit(log, world.features)


@pytest.mark.parametrize("count", [0, 1, 102, 2.5])
def test_invalid_folds_rejected(count):
    with pytest.raises(ValueError, match="fold count"):
        folds(101, count, 1)
