"""Token-level utility and decision-track visualizations."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .selection import SelectionResult
from .signals import ModelSignals
from .utils import atomic_write_json


def save_decision_json(
    tokens: list[str],
    signals: ModelSignals,
    result: SelectionResult,
    output: str | Path,
    row: int = 0,
) -> None:
    records = []
    for index, token in enumerate(tokens):
        records.append(
            {
                "position": index,
                "token": token,
                "is_response": bool(signals.response_mask[row, index]),
                "eligible": bool(signals.eligible_mask[row, index]),
                "contextual_influence": float(signals.contextual_influence[row, index]),
                "predictive_uncertainty": float(signals.predictive_uncertainty[row, index]),
                "task_domain_alignment": float(signals.task_domain_alignment[row, index]),
                "gradient_utility": float(signals.gradient_utility[row, index]),
                "combined_utility": (
                    None
                    if not np.isfinite(result.utility[row, index])
                    else float(result.utility[row, index])
                ),
                "filtered_ci": bool(result.ci_noise_mask[row, index]),
                "filtered_pu": bool(result.pu_noise_mask[row, index]),
                "filtered_tda": bool(result.tda_noise_mask[row, index]),
                "selected": bool(result.selected_mask[row, index]),
            }
        )
    atomic_write_json(
        {
            "summary": result.summary(),
            "budget_ratio": float(result.budget_ratio[row]),
            "tokens": records,
        },
        output,
    )


def save_decision_plot(
    tokens: list[str],
    signals: ModelSignals,
    result: SelectionResult,
    output: str | Path,
    row: int = 0,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Install the `viz` extra to render token decisions") from exc
    response_indices = np.flatnonzero(signals.response_mask[row])
    if response_indices.size == 0:
        raise ValueError("the selected row has no response tokens")
    labels = [tokens[index].replace("▁", "_").replace("Ġ", "_") for index in response_indices]
    tracks = np.vstack(
        [
            signals.contextual_influence[row, response_indices],
            signals.predictive_uncertainty[row, response_indices],
            signals.task_domain_alignment[row, response_indices],
            signals.gradient_utility[row, response_indices],
            np.where(
                np.isfinite(result.utility[row, response_indices]),
                result.utility[row, response_indices],
                np.nan,
            ),
            result.selected_mask[row, response_indices].astype(float),
        ]
    )
    names = ["CI", "PU", "TDA", "Gradient", "Utility", "Selected"]
    width = max(10.0, min(30.0, len(labels) * 0.32))
    figure, axes = plt.subplots(6, 1, figsize=(width, 10), sharex=True, constrained_layout=True)
    x = np.arange(len(labels))
    for axis, values, name in zip(axes, tracks, names, strict=True):
        if name == "Selected":
            colors = ["#2463EB" if value else "#D1D5DB" for value in values]
            axis.bar(x, values, color=colors, width=0.9)
            axis.set_ylim(0, 1.1)
        else:
            axis.plot(x, values, marker="o", markersize=2, linewidth=1.1)
            filtered = result.noise_mask[row, response_indices]
            axis.scatter(x[filtered], values[filtered], color="#DC2626", s=16, zorder=3)
        axis.set_ylabel(name)
        axis.grid(alpha=0.2)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    figure.suptitle("NAST token utility and decision tracks")
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)
