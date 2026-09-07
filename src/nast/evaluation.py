"""Full-response PPL, exact-match, and multiple-choice evaluation."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from .config import DatasetSpec, ExperimentConfig
from .data import NASTDataCollator, take_examples
from .formatting import NormalizedExample, build_prompt_ids, build_tokenized_example
from .metrics import accuracy, exact_match, perplexity
from .model import load_model, load_tokenizer
from .signals import causal_token_nll
from .utils import atomic_write_json, environment_metadata, set_seed


def _device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _move_model(model: Any, device: Any, quantized: bool) -> Any:
    if not quantized:
        model.to(device)
    model.eval()
    return model


def _limited_spec(spec: DatasetSpec, global_limit: int | None) -> DatasetSpec:
    if global_limit is None:
        return spec
    limit = global_limit if spec.max_samples is None else min(spec.max_samples, global_limit)
    return replace(spec, max_samples=limit)


def evaluate_perplexity(
    model: Any,
    tokenizer: Any,
    examples: list[NormalizedExample],
    config: ExperimentConfig,
    device: Any,
) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    tokenized = [
        build_tokenized_example(
            example,
            tokenizer,
            config.training.max_sequence_length,
            config.model.use_chat_template,
        )
        for example in examples
    ]
    loader = DataLoader(
        tokenized,
        batch_size=config.evaluation.per_device_batch_size,
        shuffle=False,
        collate_fn=NASTDataCollator(tokenizer),
    )
    total_nll = 0.0
    token_count = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
            nll = causal_token_nll(outputs.logits, labels)
            valid = labels.ne(-100)
            total_nll += float(nll[valid].sum().item())
            token_count += int(valid.sum().item())
    return {
        "metric": "full_response_perplexity",
        "negative_log_likelihood": total_nll / max(token_count, 1),
        "perplexity": perplexity(total_nll, max(token_count, 1)),
        "evaluated_tokens": token_count,
        "examples": len(examples),
    }


def _prompt_tensor(
    example: NormalizedExample,
    tokenizer: Any,
    config: ExperimentConfig,
    device: Any,
):
    import torch

    prompt_ids = build_prompt_ids(example, tokenizer, config.model.use_chat_template)
    prompt_ids = prompt_ids[-config.training.max_sequence_length :]
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    mask = torch.ones_like(ids)
    return ids, mask


def evaluate_exact_match(
    model: Any,
    tokenizer: Any,
    examples: list[NormalizedExample],
    config: ExperimentConfig,
    device: Any,
) -> dict[str, Any]:
    import torch

    predictions: list[str] = []
    references: list[str] = []
    with torch.no_grad():
        for example in examples:
            input_ids, attention_mask = _prompt_tensor(example, tokenizer, config, device)
            generation_kwargs: dict[str, Any] = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "max_new_tokens": config.evaluation.max_new_tokens,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "do_sample": config.evaluation.temperature > 0,
            }
            if config.evaluation.temperature > 0:
                generation_kwargs["temperature"] = config.evaluation.temperature
            output = model.generate(**generation_kwargs)
            generated = output[0, input_ids.shape[1] :]
            predictions.append(tokenizer.decode(generated, skip_special_tokens=True))
            references.append(example.response)
    return {
        "metric": "exact_match",
        "exact_match": exact_match(predictions, references),
        "examples": len(examples),
        "predictions": predictions,
    }


def _choice_nll(
    model: Any,
    tokenizer: Any,
    example: NormalizedExample,
    choice: str,
    config: ExperimentConfig,
    device: Any,
) -> float:
    import torch

    candidate = replace(example, response=choice)
    feature = build_tokenized_example(
        candidate,
        tokenizer,
        config.training.max_sequence_length,
        config.model.use_chat_template,
    )
    collated = NASTDataCollator(tokenizer, pad_to_multiple_of=None)([feature])
    labels = collated["labels"].to(device)
    with torch.no_grad():
        outputs = model(
            input_ids=collated["input_ids"].to(device),
            attention_mask=collated["attention_mask"].to(device),
            use_cache=False,
            return_dict=True,
        )
        losses = causal_token_nll(outputs.logits, labels)
    valid = labels.ne(-100)
    return float(losses[valid].mean().item())


def evaluate_multiple_choice(
    model: Any,
    tokenizer: Any,
    examples: list[NormalizedExample],
    config: ExperimentConfig,
    device: Any,
) -> dict[str, Any]:
    predictions: list[int] = []
    references: list[int] = []
    skipped = 0
    for example in examples:
        if not example.choices or example.answer_index is None:
            skipped += 1
            continue
        losses = [
            _choice_nll(model, tokenizer, example, choice, config, device)
            for choice in example.choices
        ]
        predictions.append(min(range(len(losses)), key=losses.__getitem__))
        references.append(example.answer_index)
    return {
        "metric": "normalized_conditional_likelihood_accuracy",
        "accuracy": accuracy(predictions, references),
        "examples": len(references),
        "skipped": skipped,
        "predictions": predictions,
    }


def evaluate_checkpoint(
    config: ExperimentConfig,
    checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    config.validate()
    set_seed(config.training.seed)
    tokenizer = load_tokenizer(config)
    model = load_model(config, adapter_path=checkpoint) if checkpoint else load_model(config)
    device = _device()
    model = _move_model(model, device, config.model.quantization != "none")
    results: dict[str, Any] = {
        "experiment_name": config.experiment_name,
        "checkpoint": str(checkpoint) if checkpoint else config.model.name_or_path,
        "environment": environment_metadata(),
        "datasets": {},
    }
    for spec in config.evaluation.datasets:
        limited = _limited_spec(spec, config.evaluation.max_samples)
        examples = take_examples(limited, limited.max_samples)
        if not examples:
            raise ValueError(f"no evaluation examples loaded from {spec.name}")
        if all(example.choices and example.answer_index is not None for example in examples):
            result = evaluate_multiple_choice(model, tokenizer, examples, config, device)
        elif all(example.task == "math" for example in examples):
            result = evaluate_exact_match(model, tokenizer, examples, config, device)
        else:
            result = evaluate_perplexity(model, tokenizer, examples, config, device)
        results["datasets"][spec.name] = result
    atomic_write_json(results, config.evaluation.output_file)
    return results
