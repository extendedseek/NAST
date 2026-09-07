"""Torch implementations of the four token-level NAST signals."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import SelectorConfig
from .model import find_decoder_layers, unwrap_base_model
from .utils import temporary_requires_grad


@dataclass
class ModelSignals:
    contextual_influence: np.ndarray
    predictive_uncertainty: np.ndarray
    task_domain_alignment: np.ndarray
    gradient_utility: np.ndarray
    response_mask: np.ndarray
    eligible_mask: np.ndarray
    sequence_mask: np.ndarray
    instance_loss: np.ndarray


def causal_token_nll(logits: Any, labels: Any):
    import torch.nn.functional as functional

    batch, sequence, vocabulary = logits.shape
    shifted_labels = labels[:, 1:]
    valid = shifted_labels.ne(-100)
    safe_labels = shifted_labels.masked_fill(~valid, 0)
    token_loss = functional.cross_entropy(
        logits[:, :-1, :].float().reshape(-1, vocabulary),
        safe_labels.reshape(-1),
        reduction="none",
    ).reshape(batch, sequence - 1)
    return functional.pad(token_loss * valid, (1, 0), value=0.0)


def predictive_uncertainty(logits: Any, labels: Any, transform: str = "one_minus_p"):
    import torch

    nll = causal_token_nll(logits, labels)
    valid = labels.ne(-100)
    if transform == "nll":
        return nll * valid
    uncertainty = 1.0 - torch.exp(-nll)
    return uncertainty * valid


def contextual_influence(
    attentions: tuple[Any, ...] | list[Any],
    attention_mask: Any,
    last_n_layers: int = 4,
    position_normalize: bool = True,
):
    import torch

    usable = [attention for attention in attentions if attention is not None]
    if not usable:
        raise RuntimeError(
            "The model returned no attention tensors. Use model.attention_implementation=eager."
        )
    layers = usable[-min(last_n_layers, len(usable)) :]
    query_mask = attention_mask[:, None, :, None].to(layers[0].dtype)
    key_mask = attention_mask.to(layers[0].dtype)
    received = []
    for attention in layers:
        # [B, heads, query, key] -> received attention per key token.
        score = (attention * query_mask).sum(dim=2).mean(dim=1)
        received.append(score)
    influence = torch.stack(received, dim=0).mean(dim=0) * key_mask
    if position_normalize:
        # Number of valid causal query positions that can attend to each key.
        opportunity = torch.flip(
            torch.cumsum(torch.flip(attention_mask.float(), dims=[1]), dim=1), dims=[1]
        ).clamp_min(1.0)
        influence = influence / opportunity
    return influence


def task_domain_alignment(hidden_states: Any, prototype: Any, attention_mask: Any):
    import torch.nn.functional as functional

    prototype = prototype.to(device=hidden_states.device, dtype=hidden_states.dtype)
    prototype = functional.normalize(prototype.reshape(1, 1, -1), dim=-1)
    token_vectors = functional.normalize(hidden_states, dim=-1)
    cosine = (token_vectors * prototype).sum(dim=-1)
    return ((cosine + 1.0) * 0.5) * attention_mask


def _adapter_disabled_context(model: Any):
    disable = getattr(model, "disable_adapter", None)
    if callable(disable):
        try:
            return disable()
        except TypeError:
            pass
    return contextlib.nullcontext()


@contextlib.contextmanager
def _suspend_input_require_grads(model: Any):
    """Temporarily remove HF's embedding-output grad hook during scoring.

    Gradient checkpointing enables this hook for PEFT training. Leaving it
    active would construct an unnecessary graph through blocks preceding the
    explicit saliency leaf.
    """

    candidates = [model, getattr(model, "base_model", None), unwrap_base_model(model)]
    suspended: list[Any] = []
    seen: set[int] = set()
    for candidate in candidates:
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))
        handle = getattr(candidate, "_require_grads_hook", None)
        if handle is not None and hasattr(handle, "remove"):
            handle.remove()
            candidate._require_grads_hook = None
            suspended.append(candidate)
    try:
        yield
    finally:
        for candidate in suspended:
            enable = getattr(candidate, "enable_input_require_grads", None)
            if callable(enable):
                enable()


def _block_gradient_forward(
    model: Any, forward_kwargs: dict[str, Any], labels: Any, retained_blocks: int | None
):
    import torch

    layers = find_decoder_layers(model)
    target_layer = (
        layers[0]
        if retained_blocks is None
        else layers[-min(retained_blocks, len(layers))]
    )
    holder: dict[str, Any] = {}

    def pre_hook(_module, args, kwargs):
        if args:
            leaf = args[0].detach().requires_grad_(True)
            holder["leaf"] = leaf
            return (leaf, *args[1:]), kwargs
        hidden = kwargs.get("hidden_states")
        if hidden is None:
            raise RuntimeError("final decoder block exposes no hidden-state input")
        leaf = hidden.detach().requires_grad_(True)
        holder["leaf"] = leaf
        updated = dict(kwargs)
        updated["hidden_states"] = leaf
        return args, updated

    handle = target_layer.register_forward_pre_hook(pre_hook, with_kwargs=True)
    try:
        with torch.enable_grad():
            outputs = model(**forward_kwargs)
            if "leaf" not in holder:
                raise RuntimeError("last-block saliency hook was not invoked")
            token_nll = causal_token_nll(outputs.logits, labels)
            objective = token_nll.sum()
            gradient = torch.autograd.grad(objective, holder["leaf"], retain_graph=False)[0]
            gradient_score = gradient.float().abs().sum(dim=-1)
        return outputs, token_nll.detach(), gradient_score.detach()
    finally:
        handle.remove()


def _output_head_gradient(model: Any, hidden_states: Any, labels: Any):
    import torch

    leaf = hidden_states.detach().requires_grad_(True)
    output_head = model.get_output_embeddings()
    logits = output_head(leaf)
    nll = causal_token_nll(logits, labels)
    gradient = torch.autograd.grad(nll.sum(), leaf, retain_graph=False)[0]
    return gradient.float().abs().sum(dim=-1).detach()


def _eligible_mask(input_ids: Any, response_mask: Any, tokenizer: Any, exclude_punctuation: bool):
    import re
    import torch

    eligible = response_mask.clone().bool()
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    for token_id in special_ids:
        eligible &= input_ids.ne(int(token_id))
    if exclude_punctuation:
        for row in range(input_ids.shape[0]):
            tokens = tokenizer.convert_ids_to_tokens(input_ids[row].tolist())
            keep = [bool(re.search(r"[\w\d]", str(token), flags=re.UNICODE)) for token in tokens]
            eligible[row] &= torch.tensor(keep, device=eligible.device, dtype=torch.bool)
    return eligible


def collect_model_signals(
    model: Any,
    batch: dict[str, Any],
    prototype: Any,
    config: SelectorConfig,
    tokenizer: Any,
) -> ModelSignals:
    """Run a frozen scoring pass and return CPU score arrays.

    For `last_block`, a pre-hook inserts a stop-gradient boundary immediately
    before the final decoder block. Earlier blocks run without autograd, while
    the final block and LM head provide the saliency approximation described in
    the manuscript.
    """

    import torch

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]
    response_mask = batch["response_mask"].bool()
    need_ci = config.use_ci or (
        config.ranking_strategy == "utility" and config.utility_alpha > 0
    )
    need_tda = config.use_tda
    need_gradient = (
        config.ranking_strategy == "utility"
        and config.utility_beta > 0
        and config.gradient_proxy != "none"
    )
    kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "output_attentions": need_ci,
        "output_hidden_states": (
            (need_tda and config.tda_representation == "last_hidden")
            or (need_gradient and config.gradient_proxy == "output_head")
        ),
        "use_cache": False,
        "return_dict": True,
    }
    previous_training = model.training
    model.eval()
    try:
        with (
            _adapter_disabled_context(model),
            temporary_requires_grad(model, False),
            _suspend_input_require_grads(model),
        ):
            block_proxies = {"last_block", "last_two_blocks", "full_dense"}
            if need_gradient and config.gradient_proxy in block_proxies:
                retained_blocks = {
                    "last_block": 1,
                    "last_two_blocks": 2,
                    "full_dense": None,
                }[config.gradient_proxy]
                outputs, token_nll, gradient_score = _block_gradient_forward(
                    model, kwargs, labels, retained_blocks
                )
            else:
                with torch.no_grad():
                    outputs = model(**kwargs)
                    token_nll = causal_token_nll(outputs.logits, labels)
                if need_gradient and config.gradient_proxy == "output_head":
                    with torch.enable_grad():
                        gradient_score = _output_head_gradient(
                            model, outputs.hidden_states[-1], labels
                        )
                elif need_gradient and config.gradient_proxy == "nll":
                    gradient_score = token_nll.detach()
                else:
                    gradient_score = torch.zeros_like(token_nll)

            with torch.no_grad():
                ci = (
                    contextual_influence(
                        outputs.attentions,
                        attention_mask,
                        last_n_layers=config.ci_last_n_layers,
                        position_normalize=config.ci_position_normalize,
                    )
                    if need_ci
                    else torch.zeros_like(token_nll)
                )
                pu = predictive_uncertainty(
                    outputs.logits.detach(), labels, config.pu_transform
                )
                if need_tda:
                    if prototype is None:
                        raise ValueError("TDA scoring requires a task-domain prototype")
                    if config.tda_representation == "last_hidden":
                        tda_hidden = outputs.hidden_states[-1].detach()
                    else:
                        tda_hidden = model.get_input_embeddings()(input_ids)
                    tda = task_domain_alignment(tda_hidden, prototype, attention_mask)
                else:
                    tda = torch.zeros_like(token_nll)
                response_count = response_mask.sum(dim=1).clamp_min(1)
                instance_loss = (token_nll * response_mask).sum(dim=1) / response_count
                eligible = _eligible_mask(
                    input_ids, response_mask, tokenizer, config.exclude_punctuation_only
                )
    finally:
        model.train(previous_training)
    return ModelSignals(
        contextual_influence=ci.detach().float().cpu().numpy(),
        predictive_uncertainty=pu.detach().float().cpu().numpy(),
        task_domain_alignment=tda.detach().float().cpu().numpy(),
        gradient_utility=gradient_score.detach().float().cpu().numpy(),
        response_mask=response_mask.detach().cpu().numpy(),
        eligible_mask=eligible.detach().cpu().numpy(),
        sequence_mask=attention_mask.detach().bool().cpu().numpy(),
        instance_loss=instance_loss.detach().float().cpu().numpy(),
    )


def collect_dense_gradient_scores(model: Any, batch: dict[str, Any]) -> np.ndarray:
    """Return the full-backbone activation-gradient reference for GMR audits.

    This intentionally expensive path places the saliency leaf before the
    first decoder block. It is kept out of ordinary NAST training so the audit
    cannot be confused with the lightweight selection proxy.
    """

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]
    kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "output_attentions": False,
        "output_hidden_states": False,
        "use_cache": False,
        "return_dict": True,
    }
    previous_training = model.training
    model.eval()
    try:
        with temporary_requires_grad(model, False), _suspend_input_require_grads(model):
            _, _, gradient_score = _block_gradient_forward(
                model, kwargs, labels, retained_blocks=None
            )
        return gradient_score.detach().float().cpu().numpy()
    finally:
        model.train(previous_training)
