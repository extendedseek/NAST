"""Dependency-light NAST filter, ranker, and adaptive budget allocator."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .config import SelectorConfig
from .thresholds import iqr_lower_threshold, low_alignment_threshold


def _masked_minmax(values: np.ndarray, mask: np.ndarray, epsilon: float) -> np.ndarray:
    result = np.zeros_like(values, dtype=np.float64)
    selected = values[mask]
    if selected.size == 0:
        return result
    low = float(np.nanmin(selected))
    high = float(np.nanmax(selected))
    if high - low <= epsilon:
        result[mask] = 1.0
    else:
        result[mask] = (selected - low) / (high - low + epsilon)
    return result


def _normalized_entropy(utilities: np.ndarray, epsilon: float) -> float:
    if utilities.size <= 1:
        return 0.0
    shifted = utilities - np.max(utilities)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum() + epsilon
    entropy = -float(np.sum(probabilities * np.log(probabilities + epsilon)))
    return float(np.clip(entropy / np.log(utilities.size), 0.0, 1.0))


@dataclass
class SelectionResult:
    selected_mask: np.ndarray
    candidate_mask: np.ndarray
    noise_mask: np.ndarray
    ci_noise_mask: np.ndarray
    pu_noise_mask: np.ndarray
    tda_noise_mask: np.ndarray
    recovered_mask: np.ndarray
    utility: np.ndarray
    budget_ratio: np.ndarray
    selected_count: np.ndarray
    response_count: np.ndarray
    candidate_count: np.ndarray

    def summary(self) -> dict[str, Any]:
        response = max(int(self.response_count.sum()), 1)
        candidates = max(int(self.candidate_count.sum()), 1)
        return {
            "selected_tokens": int(self.selected_count.sum()),
            "response_tokens": int(self.response_count.sum()),
            "candidate_tokens": int(self.candidate_count.sum()),
            "selected_ratio_response": float(self.selected_count.sum() / response),
            "selected_ratio_candidate": float(self.selected_count.sum() / candidates),
            "noise_ratio": float(self.noise_mask.sum() / response),
            "ci_filtered_ratio": float(self.ci_noise_mask.sum() / response),
            "pu_filtered_ratio": float(self.pu_noise_mask.sum() / response),
            "tda_filtered_ratio": float(self.tda_noise_mask.sum() / response),
            "recovered_tokens": int(self.recovered_mask.sum()),
            "mean_budget_ratio": float(np.mean(self.budget_ratio)),
        }

    def to_serializable(self) -> dict[str, Any]:
        payload = asdict(self)
        return {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in payload.items()
        }


class NASTSelector:
    """Implements the manuscript's filter-then-rank selection pipeline."""

    def __init__(self, config: SelectorConfig | None = None):
        self.config = config or SelectorConfig()

    def select(
        self,
        contextual_influence: np.ndarray,
        predictive_uncertainty: np.ndarray,
        task_domain_alignment: np.ndarray,
        gradient_utility: np.ndarray,
        response_mask: np.ndarray,
        normalized_instance_loss: np.ndarray | None = None,
        eligible_mask: np.ndarray | None = None,
        tda_population_mask: np.ndarray | None = None,
    ) -> SelectionResult:
        arrays = [
            np.asarray(contextual_influence, dtype=np.float64),
            np.asarray(predictive_uncertainty, dtype=np.float64),
            np.asarray(task_domain_alignment, dtype=np.float64),
            np.asarray(gradient_utility, dtype=np.float64),
        ]
        shape = arrays[0].shape
        if len(shape) != 2 or any(item.shape != shape for item in arrays):
            raise ValueError("all score arrays must have the same [batch, sequence] shape")
        response = np.asarray(response_mask, dtype=bool)
        if response.shape != shape:
            raise ValueError("response_mask must match score shape")
        eligible = response.copy()
        if eligible_mask is not None:
            candidate_eligibility = np.asarray(eligible_mask, dtype=bool)
            if candidate_eligibility.shape != shape:
                raise ValueError("eligible_mask must match score shape")
            eligible &= candidate_eligibility
        tda_population = eligible
        if tda_population_mask is not None:
            tda_population = np.asarray(tda_population_mask, dtype=bool)
            if tda_population.shape != shape:
                raise ValueError("tda_population_mask must match score shape")
        batch_size = shape[0]
        losses = (
            np.zeros(batch_size, dtype=np.float64)
            if normalized_instance_loss is None
            else np.asarray(normalized_instance_loss, dtype=np.float64).reshape(-1)
        )
        if losses.size != batch_size:
            raise ValueError("normalized_instance_loss must contain one value per instance")
        losses = np.clip(np.nan_to_num(losses, nan=0.0), 0.0, 1.0)

        ci, pu, tda, grad = arrays
        ci_noise = np.zeros(shape, dtype=bool)
        pu_noise = np.zeros(shape, dtype=bool)
        tda_noise = np.zeros(shape, dtype=bool)
        recovered = np.zeros(shape, dtype=bool)
        selected = np.zeros(shape, dtype=bool)
        candidate = np.zeros(shape, dtype=bool)
        utility = np.full(shape, -np.inf, dtype=np.float64)
        budget_ratio = np.zeros(batch_size, dtype=np.float64)
        selected_count = np.zeros(batch_size, dtype=np.int64)
        response_count = eligible.sum(axis=1).astype(np.int64)
        candidate_count = np.zeros(batch_size, dtype=np.int64)

        if not self.config.enabled:
            selected = eligible.copy()
            counts = eligible.sum(axis=1).astype(np.int64)
            return SelectionResult(
                selected_mask=selected,
                candidate_mask=selected.copy(),
                noise_mask=np.zeros(shape, dtype=bool),
                ci_noise_mask=np.zeros(shape, dtype=bool),
                pu_noise_mask=np.zeros(shape, dtype=bool),
                tda_noise_mask=np.zeros(shape, dtype=bool),
                recovered_mask=np.zeros(shape, dtype=bool),
                utility=np.where(eligible, 0.0, -np.inf),
                budget_ratio=np.where(counts > 0, 1.0, 0.0),
                selected_count=counts.copy(),
                response_count=counts.copy(),
                candidate_count=counts.copy(),
            )

        for row in range(batch_size):
            valid = eligible[row]
            if not np.any(valid):
                continue
            if self.config.use_ci:
                ci_cutoff = iqr_lower_threshold(ci[row, valid], self.config.ci_lambda)
                ci_noise[row, valid] = ci[row, valid] < ci_cutoff
            if self.config.use_pu:
                pu_noise[row, valid] = pu[row, valid] < self.config.pu_threshold
            if self.config.use_tda:
                tda_cutoff = low_alignment_threshold(
                    tda[row, tda_population[row]],
                    classes=self.config.tda_classes,
                    bins=self.config.tda_bins,
                    small_sample_quantile=self.config.tda_small_sample_quantile,
                )
                tda_noise[row, valid] = tda[row, valid] < tda_cutoff
            row_noise = (ci_noise[row] | pu_noise[row] | tda_noise[row]) & valid
            row_candidates = valid & ~row_noise

            minimum = min(self.config.min_candidate_tokens, int(valid.sum()))
            if int(row_candidates.sum()) < minimum:
                ci_norm = _masked_minmax(ci[row], valid, self.config.epsilon)
                pu_norm = _masked_minmax(pu[row], valid, self.config.epsilon)
                tda_norm = _masked_minmax(tda[row], valid, self.config.epsilon)
                recovery_score = ci_norm + pu_norm + tda_norm
                valid_indices = np.flatnonzero(valid)
                order = valid_indices[np.argsort(-recovery_score[valid_indices], kind="stable")]
                to_recover = order[:minimum]
                row_candidates[to_recover] = True
                recovered[row, to_recover] = True
                ci_noise[row, to_recover] = False
                pu_noise[row, to_recover] = False
                tda_noise[row, to_recover] = False
                row_noise[to_recover] = False

            candidate[row] = row_candidates
            candidate_count[row] = int(row_candidates.sum())
            if candidate_count[row] == 0:
                continue
            if self.config.ranking_strategy == "random":
                digest = hashlib.blake2b(
                    ci[row].tobytes() + pu[row].tobytes() + tda[row].tobytes(),
                    digest_size=8,
                    person=b"NAST-rng",
                ).digest()
                seed = int.from_bytes(digest, "little") ^ self.config.random_seed
                row_utility = np.zeros(shape[1], dtype=np.float64)
                row_utility[row_candidates] = np.random.default_rng(seed).random(
                    candidate_count[row]
                )
            else:
                grad_norm = _masked_minmax(grad[row], row_candidates, self.config.epsilon)
                ci_nonnegative = np.maximum(ci[row], 0.0)
                row_utility = (
                    self.config.utility_alpha
                    * np.log(self.config.epsilon + ci_nonnegative)
                    + self.config.utility_beta * grad_norm
                )
            utility[row, row_candidates] = row_utility[row_candidates]
            entropy = _normalized_entropy(row_utility[row_candidates], self.config.epsilon)
            ratio = self.config.r_min + self.config.entropy_weight * entropy + self.config.loss_weight * losses[row]
            ratio = float(np.clip(ratio, self.config.r_min, self.config.r_max))
            budget_ratio[row] = ratio
            denominator = (
                candidate_count[row]
                if self.config.budget_denominator == "candidate"
                else int(response_count[row])
            )
            k = int(np.ceil(ratio * denominator))
            k = max(self.config.min_selected_tokens, k)
            k = min(k, candidate_count[row])
            indices = np.flatnonzero(row_candidates)
            ranked = indices[np.argsort(-row_utility[indices], kind="stable")]
            selected[row, ranked[:k]] = True
            selected_count[row] = k

        noise = (ci_noise | pu_noise | tda_noise) & eligible
        if np.any(selected & ~candidate):
            raise RuntimeError("internal invariant failed: selected tokens must be candidates")
        return SelectionResult(
            selected_mask=selected,
            candidate_mask=candidate,
            noise_mask=noise,
            ci_noise_mask=ci_noise,
            pu_noise_mask=pu_noise,
            tda_noise_mask=tda_noise,
            recovered_mask=recovered,
            utility=utility,
            budget_ratio=budget_ratio,
            selected_count=selected_count,
            response_count=response_count,
            candidate_count=candidate_count,
        )
