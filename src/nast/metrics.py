"""Task, selection-quality, and interpretability metrics."""

from __future__ import annotations

import math
import re
from typing import Any, Iterable

import numpy as np


def gradient_mass_retention(
    gradient_scores: np.ndarray, selected_mask: np.ndarray, response_mask: np.ndarray
) -> float:
    scores = np.asarray(gradient_scores, dtype=np.float64)
    selected = np.asarray(selected_mask, dtype=bool)
    response = np.asarray(response_mask, dtype=bool)
    denominator = float(scores[response].sum())
    if denominator <= 0:
        return 1.0 if not np.any(response) else 0.0
    return float(scores[selected & response].sum() / denominator)


def topk_gradient_overlap(
    gradient_scores: np.ndarray,
    selected_mask: np.ndarray,
    response_mask: np.ndarray,
    k: int | float = 0.5,
) -> float:
    scores = np.asarray(gradient_scores, dtype=np.float64)
    selected = np.asarray(selected_mask, dtype=bool)
    response = np.asarray(response_mask, dtype=bool)
    overlaps: list[float] = []
    for row in range(scores.shape[0]):
        indices = np.flatnonzero(response[row])
        if indices.size == 0:
            continue
        count = int(math.ceil(k * indices.size)) if isinstance(k, float) and k <= 1 else int(k)
        count = min(max(count, 1), indices.size)
        ranked = indices[np.argsort(-scores[row, indices], kind="stable")[:count]]
        overlaps.append(float(selected[row, ranked].sum() / count))
    return float(np.mean(overlaps)) if overlaps else 0.0


def perplexity(total_negative_log_likelihood: float, token_count: int) -> float:
    if token_count <= 0:
        raise ValueError("token_count must be positive")
    return float(math.exp(min(total_negative_log_likelihood / token_count, 50.0)))


def normalize_answer(value: Any) -> str:
    text = str(value).lower().strip()
    text = text.replace(",", "")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^(the answer is|answer:)\s*", "", text)
    return text.strip(" .\n\t")


def extract_final_answer(text: str) -> str:
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if boxed:
        return normalize_answer(boxed[-1])
    gsm = re.findall(r"####\s*([^\n]+)", text)
    if gsm:
        return normalize_answer(gsm[-1])
    explicit = re.findall(r"(?:final answer|answer)\s*(?:is|:)?\s*([^\n.]+)", text, flags=re.I)
    if explicit:
        return normalize_answer(explicit[-1])
    numbers = re.findall(r"[-+]?\d+(?:\.\d+)?(?:/\d+)?", text)
    lines = text.splitlines()
    fallback = lines[-1] if lines else ""
    return normalize_answer(numbers[-1]) if numbers else normalize_answer(fallback)


def exact_match(predictions: Iterable[str], references: Iterable[str]) -> float:
    pairs = list(zip(predictions, references, strict=True))
    if not pairs:
        return 0.0
    correct = sum(
        extract_final_answer(prediction) == extract_final_answer(reference)
        for prediction, reference in pairs
    )
    return correct / len(pairs)


def accuracy(predictions: Iterable[int], references: Iterable[int]) -> float:
    pairs = list(zip(predictions, references, strict=True))
    return sum(int(prediction == reference) for prediction, reference in pairs) / max(len(pairs), 1)
