"""Small, static figures; illustrative results are not production guarantees."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

COLORS = {"logged_mean": "#94a3b8", "direct": "#a78bfa", "ips": "#d97706",
          "snips": "#0891b2", "dr": "#166534", "clipped_ips": "#dc2626"}
LABELS = {"logged_mean": "Logged mean (wrong policy)", "direct": "Direct model",
          "ips": "IPS", "snips": "SNIPS", "dr": "Cross-fitted DR",
          "clipped_ips": "IPS clipped at 10"}


def plot(report: dict, output: Path) -> None:
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False})
    rows = report["summary"]
    positions = np.arange(len(rows))
    labels = [str(row["exploration"]) for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    for i, method in enumerate(COLORS):
        offset = positions + (i - 2.5) * 0.125
        biases = [row["methods"][method]["bias"] for row in rows]
        mcse = [1.96 * row["methods"][method]["bias_mcse"] for row in rows]
        axes[0].errorbar(offset, biases, yerr=mcse, fmt="o", capsize=3,
                         color=COLORS[method], label=LABELS[method], markersize=5)
        axes[1].plot(positions, [row["methods"][method]["rmse"] for row in rows],
                     "o-", color=COLORS[method], label=LABELS[method])
    axes[0].axhline(0, color="#475569", linewidth=1)
    axes[0].set(title="Bias ± 1.96 Monte Carlo standard errors", ylabel="Value bias")
    axes[1].set(title="RMSE across independent logged datasets", ylabel="Value RMSE")
    for axis in axes:
        axis.set_xticks(positions, labels)
        axis.set_xlabel("Logging exploration ε (less exploration →)")
        axis.grid(axis="y", alpha=0.2)
    handles, legend_labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="outside lower center", ncol=3, frameon=False)
    fig.suptitle("Fixed-policy evaluation: finite-sample bias and variance", fontsize=14)
    fig.savefig(output / "estimator_audit.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout="constrained")
    values = [row["diagnostics"]["ess"] / report["design"]["log_size"] for row in rows]
    axes[0].bar(labels, values, color="#0f766e")
    axes[0].set(title="Mean effective sample fraction", ylabel="ESS / logged rows")
    axes[1].bar(labels, [row["theoretical_max_weight"] for row in rows], color="#d97706")
    axes[1].set(title="Largest possible importance weight", ylabel="Target / logging probability")
    axes[2].plot(labels, [row["true_value"] for row in rows], "o-", label="Target truth", color="#166534")
    axes[2].plot(labels, [row["exact_clipped_ips_estimand"] for row in rows],
                 "o-", label="Clipped IPS limit", color="#dc2626")
    axes[2].set(title="Weight clipping changes the estimand", ylabel="Expected value")
    axes[2].legend(frameon=False)
    for axis in axes:
        axis.set_xlabel("Logging exploration ε")
        axis.grid(axis="y", alpha=0.2)
    fig.savefig(output / "support_diagnostics.png", dpi=160)
    plt.close(fig)
