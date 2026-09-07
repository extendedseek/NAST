"""Selective and full-response causal language-model objectives."""

from __future__ import annotations

from typing import Any


def selected_labels(labels: Any, selected_mask: Any):
    import torch

    if labels.shape != selected_mask.shape:
        raise ValueError("labels and selected_mask must have the same shape")
    response = labels.ne(-100)
    selected = selected_mask.bool() & response
    if torch.any(response.sum(dim=1).gt(0) & selected.sum(dim=1).eq(0)):
        raise ValueError("each non-empty response must retain at least one selected token")
    return labels.masked_fill(~selected, -100)


def selective_causal_lm_loss(logits: Any, labels: Any, selected_mask: Any):
    import torch.nn.functional as functional

    masked_labels = selected_labels(labels, selected_mask)
    shifted_logits = logits[:, :-1, :].contiguous()
    shifted_labels = masked_labels[:, 1:].contiguous()
    return functional.cross_entropy(
        shifted_logits.float().view(-1, shifted_logits.size(-1)),
        shifted_labels.view(-1),
        ignore_index=-100,
        reduction="mean",
    )


def full_response_causal_lm_loss(logits: Any, labels: Any):
    import torch.nn.functional as functional

    return functional.cross_entropy(
        logits[:, :-1, :].float().contiguous().view(-1, logits.size(-1)),
        labels[:, 1:].contiguous().view(-1),
        ignore_index=-100,
        reduction="mean",
    )
