"""Distribution-aware thresholding used by the NAST noise gates."""

from __future__ import annotations

from itertools import combinations

import numpy as np


def iqr_lower_threshold(values: np.ndarray, multiplier: float = 1.5) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("-inf")
    q1, q3 = np.quantile(values, [0.25, 0.75])
    return float(q1 - multiplier * (q3 - q1))


def _between_class_variance(hist: np.ndarray, boundaries: tuple[int, ...]) -> float:
    probability = hist.astype(np.float64)
    total = probability.sum()
    if total <= 0:
        return float("-inf")
    probability /= total
    indices = np.arange(probability.size, dtype=np.float64)
    global_mean = float(np.sum(indices * probability))
    score = 0.0
    start = 0
    for end in (*boundaries, probability.size):
        class_probability = float(probability[start:end].sum())
        if class_probability <= 0:
            return float("-inf")
        class_mean = float(np.sum(indices[start:end] * probability[start:end]) / class_probability)
        score += class_probability * (class_mean - global_mean) ** 2
        start = end
    return score


def multi_otsu_thresholds(
    values: np.ndarray,
    classes: int = 3,
    bins: int = 64,
) -> tuple[float, ...]:
    """Return 1 or 2 Otsu thresholds without requiring scikit-image.

    The routine is intended for short, per-instance token-score vectors, so an
    exhaustive search over histogram boundaries is both deterministic and fast.
    """

    if classes not in (2, 3):
        raise ValueError("classes must be 2 or 3")
    if bins < 4:
        raise ValueError("bins must be at least 4")
    clean = np.asarray(values, dtype=np.float64)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        raise ValueError("at least one finite value is required")
    low, high = float(clean.min()), float(clean.max())
    if np.isclose(low, high):
        return tuple(low for _ in range(classes - 1))
    unique_count = int(np.unique(clean).size)
    effective_bins = min(bins, max(classes + 1, unique_count * 2))
    hist, edges = np.histogram(clean, bins=effective_bins, range=(low, high))
    best_boundaries: tuple[int, ...] | None = None
    best_score = float("-inf")
    for candidate in combinations(range(1, effective_bins), classes - 1):
        score = _between_class_variance(hist, candidate)
        if score > best_score:
            best_score = score
            best_boundaries = candidate
    if best_boundaries is None:
        quantiles = np.linspace(0, 1, classes + 1)[1:-1]
        return tuple(float(value) for value in np.quantile(clean, quantiles))
    return tuple(float(edges[index]) for index in best_boundaries)


def low_alignment_threshold(
    values: np.ndarray,
    classes: int = 3,
    bins: int = 64,
    small_sample_quantile: float = 0.25,
) -> float:
    """Threshold separating the lowest TDA cluster from the other clusters."""

    clean = np.asarray(values, dtype=np.float64)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return float("-inf")
    if clean.size < max(8, classes * 3) or np.unique(clean).size < classes:
        return float(np.quantile(clean, small_sample_quantile))
    return multi_otsu_thresholds(clean, classes=classes, bins=bins)[0]
