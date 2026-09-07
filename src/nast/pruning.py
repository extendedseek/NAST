"""Portable token-path detachment for the backward-graph ablation."""

from __future__ import annotations

import contextlib
from typing import Any

from .model import find_decoder_layers


class TokenDetachContext:
    """Detach unselected token outputs at each decoder-layer boundary.

    This preserves their numerical values for later-layer context. It is a
    portable approximation: dense kernels inside each layer can still allocate
    dense intermediate tensors, so measured CUDA memory is the authority.
    """

    def __init__(self, model: Any, selected_mask: Any):
        self.model = model
        self.selected_mask = selected_mask
        self.handles: list[Any] = []

    def _replace_hidden(self, hidden: Any):
        import torch

        mask = self.selected_mask.to(device=hidden.device, dtype=torch.bool).unsqueeze(-1)
        if hidden.shape[:2] != mask.shape[:2]:
            raise RuntimeError(
                f"decoder hidden shape {tuple(hidden.shape)} does not match selection mask "
                f"{tuple(self.selected_mask.shape)}"
            )
        return torch.where(mask, hidden, hidden.detach())

    def _hook(self, _module, _inputs, output):
        if isinstance(output, tuple):
            return (self._replace_hidden(output[0]), *output[1:])
        if isinstance(output, list):
            return [self._replace_hidden(output[0]), *output[1:]]
        if hasattr(output, "last_hidden_state"):
            output.last_hidden_state = self._replace_hidden(output.last_hidden_state)
            return output
        return self._replace_hidden(output)

    def __enter__(self):
        self.handles = [
            layer.register_forward_hook(self._hook) for layer in find_decoder_layers(self.model)
        ]
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        return False


def backward_context(model: Any, selected_mask: Any, mode: str):
    if mode == "loss_only":
        return contextlib.nullcontext()
    if mode == "detach":
        return TokenDetachContext(model, selected_mask)
    raise ValueError(f"unknown backward mode: {mode}")
