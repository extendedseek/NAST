"""End-to-end NAST task-adaptation loop."""

from __future__ import annotations

import gc
import math
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .config import ExperimentConfig, save_resolved_config
from .data import NASTDataCollator, load_normalized_mixture, tokenize_examples
from .loss import full_response_causal_lm_loss, selective_causal_lm_loss
from .metrics import gradient_mass_retention, topk_gradient_overlap
from .model import count_parameters, load_model, load_tokenizer, save_model
from .profiling import StepMeasurement, StepTimer, aggregate_measurements, memory_peaks
from .prototype import compute_task_prototype, save_prototype
from .pruning import backward_context
from .selection import NASTSelector, SelectionResult
from .signals import ModelSignals, collect_dense_gradient_scores, collect_model_signals
from .utils import append_jsonl, atomic_write_json, environment_metadata, set_seed
from .visualization import save_decision_json


@dataclass
class LossNormalizer:
    momentum: float = 0.95
    mean: float | None = None
    variance: float = 1.0

    def normalize(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        batch_mean = float(values.mean())
        batch_variance = float(values.var())
        if self.mean is None:
            self.mean = batch_mean
            self.variance = max(batch_variance, 1.0e-4)
        else:
            difference = batch_mean - self.mean
            self.mean = self.momentum * self.mean + (1.0 - self.momentum) * batch_mean
            self.variance = self.momentum * self.variance + (1.0 - self.momentum) * (
                batch_variance + difference**2
            )
        z_score = (values - self.mean) / math.sqrt(self.variance + 1.0e-8)
        return 1.0 / (1.0 + np.exp(-np.clip(z_score, -20.0, 20.0)))


def _build_accelerator(config: ExperimentConfig):
    try:
        from accelerate import Accelerator
    except ImportError as exc:
        raise RuntimeError("Install `accelerate` to run NAST training") from exc
    kwargs: dict[str, Any] = {
        "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
        "mixed_precision": config.training.mixed_precision,
    }
    if config.tracking.report_to != "none":
        kwargs["log_with"] = config.tracking.report_to
        kwargs["project_dir"] = config.training.output_dir
    return Accelerator(**kwargs)


def _make_dataloader(config: ExperimentConfig, tokenizer: Any, shuffle: bool = True):
    try:
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise RuntimeError("Install PyTorch to create the dataloader") from exc
    normalized = load_normalized_mixture(config.train_datasets, config.training.seed)
    tokenized = tokenize_examples(
        normalized,
        tokenizer,
        config.training.max_sequence_length,
        config.model.use_chat_template,
    )
    generator = None
    if shuffle:
        import torch

        generator = torch.Generator().manual_seed(config.training.seed)
    return DataLoader(
        tokenized,
        batch_size=config.training.per_device_batch_size,
        shuffle=shuffle,
        collate_fn=NASTDataCollator(tokenizer),
        num_workers=config.training.num_workers,
        pin_memory=True,
        generator=generator,
    )


def _tensor_batch(batch: dict[str, Any]) -> dict[str, Any]:
    return {
        key: batch[key]
        for key in ("input_ids", "attention_mask", "labels", "response_mask")
    }


def _choose_tokens(
    selector: NASTSelector,
    signals: ModelSignals,
    normalizer: LossNormalizer,
) -> SelectionResult:
    normalized_loss = normalizer.normalize(signals.instance_loss)
    return selector.select(
        contextual_influence=signals.contextual_influence,
        predictive_uncertainty=signals.predictive_uncertainty,
        task_domain_alignment=signals.task_domain_alignment,
        gradient_utility=signals.gradient_utility,
        response_mask=signals.response_mask,
        normalized_instance_loss=normalized_loss,
        eligible_mask=signals.eligible_mask,
        tda_population_mask=signals.sequence_mask,
    )


def _select_all_response_tokens(selector: NASTSelector, batch: dict[str, Any]) -> SelectionResult:
    response = batch["response_mask"].detach().cpu().numpy().astype(bool)
    zeros = np.zeros(response.shape, dtype=np.float64)
    return selector.select(zeros, zeros, zeros, zeros, response)


def _save_checkpoint(
    accelerator: Any,
    model: Any,
    tokenizer: Any,
    output_dir: Path,
    step: int,
    keep: int,
) -> None:
    if not accelerator.is_main_process:
        return
    checkpoint = output_dir / f"checkpoint-{step:08d}"
    save_model(accelerator.unwrap_model(model), tokenizer, checkpoint)
    checkpoints = sorted(output_dir.glob("checkpoint-*"))
    for stale in checkpoints[:-keep]:
        shutil.rmtree(stale)


def run_training(config: ExperimentConfig) -> dict[str, Any]:
    """Train a model and persist all run metadata and measured metrics."""

    import torch
    from transformers import get_cosine_schedule_with_warmup

    config.validate()
    set_seed(config.training.seed)
    accelerator = _build_accelerator(config)
    output_dir = Path(config.training.output_dir).expanduser().resolve()
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        save_resolved_config(config, output_dir / "resolved_config.yaml")
        atomic_write_json(environment_metadata(), output_dir / "environment.json")
    accelerator.wait_for_everyone()

    tokenizer = load_tokenizer(config)
    model = load_model(config)
    train_loader = _make_dataloader(config, tokenizer, shuffle=True)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("model has no trainable parameters")
    optimizer = torch.optim.AdamW(
        trainable,
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    updates_per_epoch = math.ceil(len(train_loader) / config.training.gradient_accumulation_steps)
    planned_steps = (
        config.training.max_steps
        if config.training.max_steps > 0
        else math.ceil(config.training.epochs * updates_per_epoch)
    )
    warmup_steps = int(round(planned_steps * config.training.warmup_ratio))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, planned_steps)
    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )

    prototype = None
    if config.selector.enabled and config.selector.use_tda:
        prototype = compute_task_prototype(
            accelerator.unwrap_model(model),
            train_loader,
            max_samples=config.training.prototype_max_samples,
            representation=config.selector.tda_representation,
        )
    if accelerator.is_main_process and prototype is not None:
        save_prototype(
            prototype,
            output_dir / "task_prototype.pt",
            metadata={
                "representation": config.selector.tda_representation,
                "max_samples": config.training.prototype_max_samples,
                "training_datasets": [spec.name for spec in config.train_datasets],
            },
        )
    if accelerator.is_main_process:
        atomic_write_json(count_parameters(accelerator.unwrap_model(model)), output_dir / "model.json")

    selector = NASTSelector(config.selector)
    loss_normalizer = LossNormalizer()
    global_step = 0
    micro_step = 0
    measurements: list[StepMeasurement] = []
    aggregate_selected = 0
    aggregate_response = 0
    aggregate_candidate = 0
    aggregate_noise = 0
    aggregate_gmr: list[float] = []
    aggregate_overlap: list[float] = []
    started_at = time.time()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    if accelerator.device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(accelerator.device)

    while global_step < planned_steps:
        for raw_batch in train_loader:
            batch = _tensor_batch(raw_batch)
            processed_tokens = int(batch["attention_mask"].sum().item())
            device = batch["input_ids"].device
            with StepTimer(device) as timer:
                unwrapped = accelerator.unwrap_model(model)
                signals = None
                if config.selector.enabled:
                    signals = collect_model_signals(
                        unwrapped, batch, prototype, config.selector, tokenizer
                    )
                    result = _choose_tokens(selector, signals, loss_normalizer)
                else:
                    result = _select_all_response_tokens(selector, batch)
                selected_mask = torch.from_numpy(result.selected_mask).to(
                    device=device, dtype=torch.bool
                )
                with accelerator.accumulate(model):
                    backward_mode = (
                        config.training.backward_mode if config.selector.enabled else "loss_only"
                    )
                    with backward_context(model, selected_mask, backward_mode):
                        outputs = model(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            use_cache=False,
                            return_dict=True,
                        )
                        loss = selective_causal_lm_loss(
                            outputs.logits, batch["labels"], selected_mask
                        )
                        if config.training.fail_on_nonfinite and not torch.isfinite(loss):
                            raise FloatingPointError(f"non-finite training loss at step {global_step}")
                        accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(trainable, config.training.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
            micro_step += 1

            summary = result.summary()
            aggregate_selected += summary["selected_tokens"]
            aggregate_response += summary["response_tokens"]
            aggregate_candidate += summary["candidate_tokens"]
            aggregate_noise += int(result.noise_mask.sum())
            if signals is None:
                aggregate_gmr.append(1.0)
                aggregate_overlap.append(1.0)
            else:
                aggregate_gmr.append(
                    gradient_mass_retention(
                        signals.gradient_utility, result.selected_mask, signals.eligible_mask
                    )
                )
                aggregate_overlap.append(
                    topk_gradient_overlap(
                        signals.gradient_utility, result.selected_mask, signals.eligible_mask, 0.5
                    )
                )

            if accelerator.sync_gradients:
                global_step += 1
                allocated, reserved = memory_peaks(device)
                measurement = StepMeasurement(
                    elapsed_seconds=timer.elapsed_seconds,
                    processed_tokens=processed_tokens,
                    peak_allocated_bytes=allocated,
                    peak_reserved_bytes=reserved,
                )
                measurements.append(measurement)
                if accelerator.is_main_process and global_step % config.training.log_every == 0:
                    payload = {
                        "step": global_step,
                        "micro_step": micro_step,
                        "loss": float(loss.detach().float().item()),
                        "learning_rate": float(scheduler.get_last_lr()[0]),
                        "proxy_gradient_mass_retention": aggregate_gmr[-1],
                        "proxy_overlap_at_50": aggregate_overlap[-1],
                        **summary,
                        **measurement.to_dict(),
                    }
                    append_jsonl(payload, output_dir / "train_metrics.jsonl")
                decision_interval = config.tracking.log_token_decisions_every
                if (
                    accelerator.is_main_process
                    and signals is not None
                    and decision_interval > 0
                    and global_step % decision_interval == 0
                ):
                    token_ids = batch["input_ids"][0].detach().cpu().tolist()
                    tokens = tokenizer.convert_ids_to_tokens(token_ids)
                    save_decision_json(
                        tokens,
                        signals,
                        result,
                        output_dir
                        / "token_decisions"
                        / f"step-{global_step:08d}.json",
                    )
                if global_step % config.training.save_every == 0:
                    accelerator.wait_for_everyone()
                    _save_checkpoint(
                        accelerator,
                        model,
                        tokenizer,
                        output_dir,
                        global_step,
                        config.training.keep_last_checkpoints,
                    )
                if global_step >= planned_steps:
                    break
        if global_step >= planned_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_dir = output_dir / "final"
        save_model(accelerator.unwrap_model(model), tokenizer, final_dir)
        response_denominator = max(aggregate_response, 1)
        candidate_denominator = max(aggregate_candidate, 1)
        run_summary = {
            "experiment_name": config.experiment_name,
            "completed_steps": global_step,
            "micro_steps": micro_step,
            "wall_time_seconds": time.time() - started_at,
            "selected_tokens": aggregate_selected,
            "response_tokens": aggregate_response,
            "candidate_tokens": aggregate_candidate,
            "selected_ratio_response": aggregate_selected / response_denominator,
            "selected_ratio_candidate": aggregate_selected / candidate_denominator,
            "noise_ratio": aggregate_noise / response_denominator,
            "proxy_gradient_mass_retention": float(np.mean(aggregate_gmr)),
            "proxy_overlap_at_50": float(np.mean(aggregate_overlap)),
            "budget_denominator": config.selector.budget_denominator,
            "backward_mode": (
                config.training.backward_mode if config.selector.enabled else "loss_only"
            ),
            "profile": aggregate_measurements(measurements),
            "final_checkpoint": str(final_dir),
        }
        atomic_write_json(run_summary, output_dir / "run_summary.json")
    else:
        run_summary = {}
    accelerator.end_training()
    return run_summary


def _cycle(loader: Any) -> Iterator[dict[str, Any]]:
    while True:
        yield from loader


def profile_steps(
    config: ExperimentConfig,
    steps: int,
    dense: bool = False,
) -> dict[str, Any]:
    """Profile measured updates on a throwaway model instance."""

    import torch

    if steps < 1:
        raise ValueError("steps must be positive")
    set_seed(config.training.seed)
    accelerator = _build_accelerator(config)
    tokenizer = load_tokenizer(config)
    model = load_model(config)
    loader = _make_dataloader(config, tokenizer, shuffle=False)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("model has no trainable parameters")
    optimizer = torch.optim.AdamW(
        trainable,
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    run_dense = dense or not config.selector.enabled
    prototype = None
    if not run_dense and config.selector.use_tda:
        prototype = compute_task_prototype(
            accelerator.unwrap_model(model),
            loader,
            max_samples=min(config.training.prototype_max_samples, 128),
            representation=config.selector.tda_representation,
        )
    selector = NASTSelector(config.selector)
    normalizer = LossNormalizer()
    iterator = _cycle(loader)
    warmup = config.training.profile_warmup_steps
    measurements: list[StepMeasurement] = []
    selections: list[dict[str, Any]] = []
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for index in range(warmup + steps):
        raw_batch = next(iterator)
        batch = _tensor_batch(raw_batch)
        device = batch["input_ids"].device
        if device.type == "cuda" and index >= warmup:
            if index == warmup:
                torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        processed = int(batch["attention_mask"].sum().item())
        scoring_seconds = 0.0
        if run_dense:
            selected_mask = batch["response_mask"].bool()
            result = None
        else:
            with StepTimer(device) as scoring_timer:
                signals = collect_model_signals(
                    accelerator.unwrap_model(model), batch, prototype, config.selector, tokenizer
                )
                result = _choose_tokens(selector, signals, normalizer)
                selected_mask = torch.from_numpy(result.selected_mask).to(device=device)
            scoring_seconds = scoring_timer.elapsed_seconds
        mode = "loss_only" if run_dense else config.training.backward_mode
        with StepTimer(device) as forward_timer:
            with backward_context(model, selected_mask, mode):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    use_cache=False,
                    return_dict=True,
                )
                loss = (
                    full_response_causal_lm_loss(outputs.logits, batch["labels"])
                    if run_dense
                    else selective_causal_lm_loss(outputs.logits, batch["labels"], selected_mask)
                )
                with StepTimer(device) as backward_timer:
                    accelerator.backward(loss)
        # The nested backward timer is subtracted from the enclosing model/loss
        # interval so the reported phases remain non-overlapping.
        backward_seconds = backward_timer.elapsed_seconds
        forward_seconds = max(forward_timer.elapsed_seconds - backward_seconds, 0.0)
        with StepTimer(device) as optimizer_timer:
            accelerator.clip_grad_norm_(trainable, config.training.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        optimizer_seconds = optimizer_timer.elapsed_seconds
        if index >= warmup:
            allocated, reserved = memory_peaks(device)
            measurements.append(
                StepMeasurement(
                    elapsed_seconds=(
                        scoring_seconds
                        + forward_seconds
                        + backward_seconds
                        + optimizer_seconds
                    ),
                    processed_tokens=processed,
                    peak_allocated_bytes=allocated,
                    peak_reserved_bytes=reserved,
                    scoring_seconds=scoring_seconds,
                    forward_seconds=forward_seconds,
                    backward_seconds=backward_seconds,
                    optimizer_seconds=optimizer_seconds,
                )
            )
            if result is not None:
                selections.append(result.summary())
    profile = aggregate_measurements(measurements)
    profile.update(
        {
            "mode": "dense" if run_dense else "nast",
            "backward_mode": "loss_only" if run_dense else config.training.backward_mode,
            "scoring_time_included": not run_dense,
            "device": str(accelerator.device),
        }
    )
    if selections:
        profile["selection"] = {
            key: float(np.mean([item[key] for item in selections]))
            for key in selections[0]
        }
    accelerator.end_training()
    if hasattr(accelerator, "free_memory"):
        accelerator.free_memory()
    del model, optimizer, loader, iterator, prototype
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return profile


def audit_selection(
    config: ExperimentConfig,
    batches: int = 10,
    checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    """Compare NAST decisions with an expensive full-dense gradient reference."""

    if batches < 1:
        raise ValueError("batches must be positive")
    if not config.selector.enabled:
        raise ValueError("selection audit requires selector.enabled=true")
    set_seed(config.training.seed)
    accelerator = _build_accelerator(config)
    tokenizer = load_tokenizer(config)
    model = load_model(config, adapter_path=checkpoint) if checkpoint else load_model(config)
    loader = _make_dataloader(config, tokenizer, shuffle=False)
    model, loader = accelerator.prepare(model, loader)
    unwrapped = accelerator.unwrap_model(model)
    prototype = None
    if config.selector.use_tda:
        prototype = compute_task_prototype(
            unwrapped,
            loader,
            max_samples=config.training.prototype_max_samples,
            representation=config.selector.tda_representation,
        )
    selector = NASTSelector(config.selector)
    normalizer = LossNormalizer()
    gmr_values: list[float] = []
    overlap_values: list[float] = []
    selection_summaries: list[dict[str, Any]] = []
    evaluated_batches = 0
    for raw_batch in loader:
        batch = _tensor_batch(raw_batch)
        signals = collect_model_signals(
            unwrapped, batch, prototype, config.selector, tokenizer
        )
        result = _choose_tokens(selector, signals, normalizer)
        dense_scores = collect_dense_gradient_scores(unwrapped, batch)
        for row in range(dense_scores.shape[0]):
            row_slice = slice(row, row + 1)
            gmr_values.append(
                gradient_mass_retention(
                    dense_scores[row_slice],
                    result.selected_mask[row_slice],
                    signals.eligible_mask[row_slice],
                )
            )
            overlap_values.append(
                topk_gradient_overlap(
                    dense_scores[row_slice],
                    result.selected_mask[row_slice],
                    signals.eligible_mask[row_slice],
                    0.5,
                )
            )
        selection_summaries.append(result.summary())
        evaluated_batches += 1
        if evaluated_batches >= batches:
            break
    accelerator.end_training()
    if hasattr(accelerator, "free_memory"):
        accelerator.free_memory()
    del model, loader, unwrapped, prototype
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    if not selection_summaries:
        raise ValueError("selection audit received no training batches")
    return {
        "batches": evaluated_batches,
        "examples": len(gmr_values),
        "reference_gradient": "full_dense_first_block_input",
        "gradient_mass_retention": float(np.mean(gmr_values)),
        "overlap_at_50": float(np.mean(overlap_values)),
        "selection": {
            key: float(np.mean([item[key] for item in selection_summaries]))
            for key in selection_summaries[0]
        },
    }
