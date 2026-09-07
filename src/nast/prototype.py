"""Task-domain prototype construction and persistence."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _model_device(model: Any):
    return next(model.parameters()).device


def compute_task_prototype(
    model: Any,
    dataloader: Any,
    max_samples: int = 1024,
    representation: str = "embedding",
):
    """Average one pooled vector per training example, as specified in NAST."""

    import torch

    previous_training = model.training
    model.eval()
    device = _model_device(model)
    vector_sum = None
    example_count = 0
    try:
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                if representation == "last_hidden":
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                        use_cache=False,
                        return_dict=True,
                    )
                    vectors = outputs.hidden_states[-1]
                else:
                    vectors = model.get_input_embeddings()(input_ids)
                weights = attention_mask.to(vectors.dtype).unsqueeze(-1)
                pooled = (vectors * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
                remaining = max_samples - example_count
                pooled = pooled[:remaining].float()
                batch_sum = pooled.sum(dim=0)
                vector_sum = batch_sum if vector_sum is None else vector_sum + batch_sum
                example_count += pooled.shape[0]
                if example_count >= max_samples:
                    break
    finally:
        model.train(previous_training)
    if vector_sum is None or example_count == 0:
        raise ValueError("cannot compute a prototype from an empty dataloader")
    return (vector_sum / example_count).detach().cpu()


def save_prototype(prototype: Any, path: str | Path, metadata: dict[str, Any] | None = None) -> None:
    import torch

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"prototype": prototype.detach().cpu(), "metadata": metadata or {}}, destination)


def load_prototype(path: str | Path):
    import torch

    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "prototype" not in payload:
        raise ValueError("invalid NAST prototype file")
    return payload["prototype"], payload.get("metadata", {})
