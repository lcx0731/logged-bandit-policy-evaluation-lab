from dataclasses import replace

import numpy as np
import pytest

from bandit_audit.estimators import evaluate, importance_weights
from bandit_audit.policies import Log, check_support


def example():
    return Log(np.zeros(4, dtype=int), np.array([0, 1, 0, 1]),
               np.array([1., 0., 0., 1.]), np.tile([.5, .5], (4, 1)),
               np.tile([.75, .25], (4, 1)))


def test_hand_calculated_estimators_and_ess():
    result = evaluate(example(), np.tile([.4, .6], (4, 1)), clip=1)
    assert result["estimates"] == pytest.approx({
        "logged_mean": .5, "direct": .45, "ips": .5,
        "snips": .5, "dr": .5, "clipped_ips": .375,
    })
    assert result["diagnostics"] == pytest.approx({
        "ess": 3.2, "max_weight": 1.5, "mean_weight": 1,
        "clipped_row_fraction": .5,
    })


def test_exact_enumerated_ips_and_dr_expectation_with_wrong_reward_model():
    # The eight rows exactly enumerate logger probabilities and context masses.
    contexts = np.repeat([0, 1], 4)
    actions = np.array([0, 0, 0, 1, 0, 1, 1, 1])
    mu = np.array([[.75, .25], [.25, .75]])
    pi = np.array([[.2, .8], [.6, .4]])
    reward_means = np.array([[.1, .9], [.7, .3]])
    rewards = reward_means[contexts, actions]
    log = Log(contexts, actions, rewards, mu[contexts], pi[contexts])
    wrong_q = np.tile([.2, .1], (8, 1))
    result = evaluate(log, wrong_q)
    truth = .5 * (.2 * .1 + .8 * .9) + .5 * (.6 * .7 + .4 * .3)
    assert truth == pytest.approx(.64)
    assert result["estimates"]["ips"] == pytest.approx(truth)
    assert result["estimates"]["dr"] == pytest.approx(truth)
    assert abs(result["estimates"]["direct"] - truth) > .1


def test_oracle_reward_model_cancels_residuals():
    log = example()
    q = np.array([[1., .2], [.3, 0.], [0., .4], [.5, 1.]])
    result = evaluate(log, q)
    assert result["estimates"]["dr"] == pytest.approx(np.mean((log.target * q).sum(1)))


def test_identical_policies_reduce_ips_to_logged_mean():
    log = example()
    log = replace(log, target=log.logging.copy())
    result = evaluate(log, np.zeros_like(log.target))
    assert result["estimates"]["ips"] == pytest.approx(np.mean(log.rewards))
    assert result["diagnostics"]["ess"] == pytest.approx(4)


def test_support_is_checked_for_unobserved_actions_too():
    with pytest.raises(ValueError, match="Unsupported"):
        check_support(np.array([[.5, .5]]), np.array([[1., 0.]]))
    check_support(np.array([[1., 0.]]), np.array([[1., 0.]]))


@pytest.mark.parametrize("bad", [np.array([[.8, .8]]), np.array([[-.1, 1.1]]),
                               np.array([[np.nan, .5]]), np.array([[np.inf, .5]])])
def test_invalid_probability_tables_are_rejected(bad):
    with pytest.raises(ValueError):
        check_support(bad, np.array([[.5, .5]]))


@pytest.mark.parametrize("actions", [np.array([0., 1., 0., 1.]),
                                   np.array([0, 2, 0, 1]), np.array([0, -1, 0, 1])])
def test_malformed_actions_are_rejected(actions):
    with pytest.raises(ValueError, match="actions"):
        importance_weights(replace(example(), actions=actions))


def test_zero_weight_snips_is_not_silently_reported():
    log = example()
    log = replace(log, actions=np.ones(4, dtype=int), target=np.tile([1., 0.], (4, 1)))
    with pytest.raises(ValueError, match="SNIPS is undefined"):
        evaluate(log, np.zeros_like(log.target))


def test_impossible_observed_action_is_rejected_even_if_target_mass_is_zero():
    log = example()
    log = replace(log, logging=np.tile([1., 0.], (4, 1)),
                  target=np.tile([1., 0.], (4, 1)))
    with pytest.raises(ValueError, match="observed action"):
        importance_weights(log)


@pytest.mark.parametrize("clip", [0, -2, float("nan"), float("inf")])
def test_invalid_clip_is_rejected(clip):
    with pytest.raises(ValueError, match="clip"):
        evaluate(example(), np.zeros((4, 2)), clip=clip)
