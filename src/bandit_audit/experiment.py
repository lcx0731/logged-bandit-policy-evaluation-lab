"""Reproducible Monte Carlo audit with an exact finite-context benchmark."""

import argparse
import json
import platform
from importlib.metadata import version
from pathlib import Path

import numpy as np

from .estimators import evaluate
from .policies import check_support, make_world, sample_log
from .reward_model import cross_fit

EXPLORATION = (0.60, 0.15, 0.03)


def run(*, repetitions: int = 200, size: int = 3000, seed: int = 20261001) -> dict:
    if not isinstance(repetitions, int) or repetitions < 2:
        raise ValueError("at least two repetitions are needed for Monte Carlo uncertainty")
    if not isinstance(size, int) or size < 100:
        raise ValueError("size must be an integer of at least 100")
    world = make_world()
    rows = []
    all_runs = []
    seeds = np.random.SeedSequence(seed).spawn(len(EXPLORATION) * repetitions)
    seed_index = 0
    for exploration in EXPLORATION:
        replications = []
        for replicate in range(repetitions):
            data_seed, fold_seed = map(int, seeds[seed_index].generate_state(2))
            seed_index += 1
            log = sample_log(world, exploration, size, data_seed)
            q, _ = cross_fit(log, world.features, seed=fold_seed)
            result = evaluate(log, q)
            replications.append(result)
            all_runs.append({"exploration": exploration, "replicate": replicate,
                             **result})
        logger = world.logger(exploration)
        # This is the exact limiting estimand of UNNORMALIZED weight clipping.
        clipped_estimand = float(np.sum(
            world.context_mass[:, None] * logger
            * np.minimum(world.target / logger, 10.0) * world.reward_means
        ))
        rows.append({
            "exploration": exploration,
            "true_value": world.value,
            "exact_logged_policy_value": float(np.sum(
                world.context_mass[:, None] * logger * world.reward_means)),
            "exact_clipped_ips_estimand": clipped_estimand,
            "theoretical_max_weight": float(np.max(world.target / logger)),
            "diagnostics": {
                name: float(np.mean([r["diagnostics"][name] for r in replications]))
                for name in replications[0]["diagnostics"]
            },
            "methods": {
                name: summarize(np.array([r["estimates"][name] for r in replications]),
                                world.value)
                for name in replications[0]["estimates"]
            },
        })
    try:
        check_support(world.target, world.logger(0.0))
    except ValueError as error:
        support_failure = str(error)
    else:
        raise AssertionError("zero-exploration negative control was not rejected")
    return {
        "design": {"seed": seed, "repetitions": repetitions, "log_size": size,
                   "folds": 3, "ridge": 1.0, "weight_clip": 10.0,
                   "exploration": list(EXPLORATION), "source": "synthetic only",
                   "context_mass": world.context_mass.tolist(),
                   "features": world.features.tolist(),
                   "reward_means": world.reward_means.tolist(),
                   "target_policy": world.target.tolist()},
        "environment": {"python": platform.python_version(),
                        "numpy": version("numpy"), "matplotlib": version("matplotlib")},
        "zero_support_negative_control": support_failure,
        "summary": rows,
        "replications": all_runs,
    }


def summarize(estimates: np.ndarray, truth: float) -> dict:
    errors = estimates - truth
    return {
        "mean": float(estimates.mean()),
        "bias": float(errors.mean()),
        "bias_mcse": float(errors.std(ddof=1) / np.sqrt(len(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "p05": float(np.quantile(estimates, 0.05)),
        "p95": float(np.quantile(estimates, 0.95)),
        "outside_reward_range_fraction": float(np.mean((estimates < 0) | (estimates > 1))),
    }


def save(report: dict, output: Path) -> None:
    from .plotting import plot

    # Validate serialization before creating any output files.
    summary = {k: v for k, v in report.items() if k != "replications"}
    summary_json = json.dumps(summary, indent=2, allow_nan=False) + "\n"
    runs_json = json.dumps(report["replications"], indent=2, allow_nan=False) + "\n"
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(summary_json, encoding="utf-8")
    (output / "replications.json").write_text(runs_json, encoding="utf-8")
    plot(report, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=200)
    parser.add_argument("--size", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    try:
        report = run(repetitions=args.repetitions, size=args.size, seed=args.seed)
        save(report, args.output)
    except ValueError as error:
        parser.error(str(error))
    print(f"Exact target-policy value: {report['summary'][0]['true_value']:.6f}")
    for row in report["summary"]:
        print(f"exploration={row['exploration']:.2f}: "
              f"mean ESS={row['diagnostics']['ess']:.1f}; "
              f"IPS RMSE={row['methods']['ips']['rmse']:.4f}; "
              f"DR RMSE={row['methods']['dr']['rmse']:.4f}")
    print(f"Artifacts saved to {args.output}")
