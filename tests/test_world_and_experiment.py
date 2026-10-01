import json

import numpy as np
import pytest

from bandit_audit.experiment import run, save, summarize
from bandit_audit.policies import make_world, sample_log


def test_world_value_is_exact_weighted_sum_and_sampling_is_reproducible():
    world = make_world()
    total = sum(world.context_mass[c] * world.target[c, a] * world.reward_means[c, a]
                for c in range(8) for a in range(3))
    assert world.value == pytest.approx(total)
    np.testing.assert_allclose(world.target.sum(1), 1)
    first, second = sample_log(world, .6, 500, 10), sample_log(world, .6, 500, 10)
    for name in ("contexts", "actions", "rewards", "logging", "target"):
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name))


def test_zero_exploration_rejected_before_sampling():
    with pytest.raises(ValueError, match="Unsupported"):
        sample_log(make_world(), 0, 3000, 1)


def test_summary_has_correct_mcse_and_keeps_out_of_range_estimates():
    summary = summarize(np.array([-.2, .4, 1.3]), .5)
    assert summary["bias"] == pytest.approx(0)
    assert summary["rmse"] == pytest.approx(np.sqrt(.38))
    assert summary["bias_mcse"] == pytest.approx(np.sqrt(.19))
    assert summary["outside_reward_range_fraction"] == pytest.approx(2 / 3)


def test_end_to_end_artifacts_and_repeatability(tmp_path):
    first = run(repetitions=2, size=3000, seed=21)
    second = run(repetitions=2, size=3000, seed=21)
    assert first == second
    save(first, tmp_path)
    result = json.loads((tmp_path / "summary.json").read_text())
    assert len(result["summary"]) == 3
    assert "Unsupported" in result["zero_support_negative_control"]
    assert len(json.loads((tmp_path / "replications.json").read_text())) == 6
    for filename in ("estimator_audit.png", "support_diagnostics.png"):
        assert (tmp_path / filename).stat().st_size > 10000
    for scenario in result["summary"]:
        assert 0 < scenario["diagnostics"]["ess"] <= 3000
        if scenario["exploration"] == .6:
            assert scenario["exact_clipped_ips_estimand"] == pytest.approx(scenario["true_value"])
        else:
            assert scenario["exact_clipped_ips_estimand"] < scenario["true_value"]


def test_nonfinite_report_is_not_written(tmp_path):
    bad = {"design": {"value": float("nan")}, "replications": []}
    with pytest.raises(ValueError, match="JSON"):
        save(bad, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize("repetitions,size", [(1, 3000), (2, 10), (2.5, 3000)])
def test_invalid_experiment_parameters_rejected(repetitions, size):
    with pytest.raises(ValueError):
        run(repetitions=repetitions, size=size)
