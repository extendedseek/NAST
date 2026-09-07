#!/usr/bin/env python3
"""Dependency-light end-to-end smoke test for the NAST selector."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nast.config import SelectorConfig  # noqa: E402
from nast.metrics import gradient_mass_retention, topk_gradient_overlap  # noqa: E402
from nast.selection import NASTSelector  # noqa: E402


def main() -> None:
    rng = np.random.default_rng(42)
    shape = (3, 24)
    response = np.zeros(shape, dtype=bool)
    response[:, 8:22] = True
    ci = rng.uniform(0.0, 1.0, shape)
    pu = rng.uniform(0.0, 0.4, shape)
    tda = rng.uniform(0.0, 1.0, shape)
    gradient = rng.lognormal(mean=0.0, sigma=0.8, size=shape)
    result = NASTSelector(SelectorConfig()).select(
        ci,
        pu,
        tda,
        gradient,
        response,
        normalized_instance_loss=np.array([0.1, 0.5, 0.9]),
    )
    assert np.all(result.selected_mask <= result.candidate_mask)
    assert np.all(result.selected_count >= 1)
    payload = result.summary()
    payload["synthetic_gradient_mass_retention"] = gradient_mass_retention(
        gradient, result.selected_mask, response
    )
    payload["synthetic_overlap_at_50"] = topk_gradient_overlap(
        gradient, result.selected_mask, response, 0.5
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
