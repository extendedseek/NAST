"""Command-line interface for training, evaluation, profiling, and inspection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .config import load_config
from .data import DATASET_REGISTRY, NASTDataCollator, take_examples
from .evaluation import evaluate_checkpoint
from .formatting import build_tokenized_example
from .model import load_model, load_tokenizer
from .profiling import memory_reduction
from .prototype import compute_task_prototype
from .selection import NASTSelector
from .signals import collect_model_signals
from .trainer import LossNormalizer, audit_selection, profile_steps, run_training
from .utils import atomic_write_json
from .visualization import save_decision_json, save_decision_plot


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nast",
        description="Noise-Aware Selective Token Backpropagation",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-config", help="validate and summarize YAML")
    validate.add_argument("--config", required=True)

    train = subparsers.add_parser("train", help="run NAST task adaptation")
    train.add_argument("--config", required=True)

    evaluate = subparsers.add_parser("evaluate", help="evaluate a base model or adapter")
    evaluate.add_argument("--config", required=True)
    evaluate.add_argument("--checkpoint")

    profile = subparsers.add_parser("profile", help="measure memory and throughput")
    profile.add_argument("--config", required=True)
    profile.add_argument("--steps", type=int, default=10)
    profile.add_argument("--output", default="results/profile.json")
    profile.add_argument(
        "--dense",
        action="store_true",
        help="profile only the full-response, loss-only baseline",
    )
    profile.add_argument(
        "--compare-dense",
        action="store_true",
        help="also run a full-response, loss-only baseline",
    )

    inspect = subparsers.add_parser("inspect", help="visualize one token-selection decision")
    inspect.add_argument("--config", required=True)
    inspect.add_argument("--dataset-index", type=int, default=0)
    inspect.add_argument("--example-index", type=int, default=0)
    inspect.add_argument("--output-dir", default="results/token_inspection")

    audit = subparsers.add_parser(
        "audit-selection",
        help="compare selected tokens with a full-dense gradient reference",
    )
    audit.add_argument("--config", required=True)
    audit.add_argument("--checkpoint")
    audit.add_argument("--batches", type=int, default=10)
    audit.add_argument("--output", default="results/selection_audit.json")

    subparsers.add_parser("list-datasets", help="print built-in dataset identifiers")
    return parser


def _print(payload) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


def _inspect(args: argparse.Namespace) -> dict:
    import torch

    config = load_config(args.config)
    if not 0 <= args.dataset_index < len(config.train_datasets):
        raise IndexError("dataset-index is outside train_datasets")
    spec = config.train_datasets[args.dataset_index]
    examples = take_examples(spec, args.example_index + 1)
    if args.example_index >= len(examples):
        raise IndexError("example-index exceeds the loaded dataset")
    tokenizer = load_tokenizer(config)
    model = load_model(config)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    if config.model.quantization == "none":
        model.to(device)
    feature = build_tokenized_example(
        examples[args.example_index],
        tokenizer,
        config.training.max_sequence_length,
        config.model.use_chat_template,
    )
    batch = NASTDataCollator(tokenizer, pad_to_multiple_of=None)([feature])
    for key in ("input_ids", "attention_mask", "labels", "response_mask"):
        batch[key] = batch[key].to(device)
    prototype = None
    if config.selector.use_tda:
        prototype = compute_task_prototype(
            model,
            [batch],
            max_samples=1,
            representation=config.selector.tda_representation,
        )
    signals = collect_model_signals(model, batch, prototype, config.selector, tokenizer)
    normalizer = LossNormalizer()
    result = NASTSelector(config.selector).select(
        signals.contextual_influence,
        signals.predictive_uncertainty,
        signals.task_domain_alignment,
        signals.gradient_utility,
        signals.response_mask,
        normalizer.normalize(signals.instance_loss),
        signals.eligible_mask,
        signals.sequence_mask,
    )
    tokens = tokenizer.convert_ids_to_tokens(batch["input_ids"][0].detach().cpu().tolist())
    output_dir = Path(args.output_dir)
    json_path = output_dir / "token_decisions.json"
    plot_path = output_dir / "token_decisions.png"
    save_decision_json(tokens, signals, result, json_path)
    try:
        save_decision_plot(tokens, signals, result, plot_path)
        plot = str(plot_path.resolve())
    except RuntimeError:
        plot = None
    return {"summary": result.summary(), "json": str(json_path.resolve()), "plot": plot}


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "list-datasets":
        _print(DATASET_REGISTRY)
        return
    if args.command == "validate-config":
        config = load_config(args.config)
        _print(
            {
                "valid": True,
                "experiment_name": config.experiment_name,
                "model": config.model.name_or_path,
                "training_datasets": [item.name for item in config.train_datasets],
                "evaluation_datasets": [item.name for item in config.evaluation.datasets],
                "budget_denominator": config.selector.budget_denominator,
                "backward_mode": config.training.backward_mode,
            }
        )
        return
    if args.command == "train":
        _print(run_training(load_config(args.config)))
        return
    if args.command == "evaluate":
        _print(evaluate_checkpoint(load_config(args.config), args.checkpoint))
        return
    if args.command == "profile":
        config = load_config(args.config)
        if args.dense and args.compare_dense:
            raise ValueError("--dense and --compare-dense are mutually exclusive")
        if args.dense:
            payload = {"dense": profile_steps(config, args.steps, dense=True)}
        else:
            payload = {"nast": profile_steps(config, args.steps, dense=False)}
        if args.compare_dense:
            payload["dense"] = profile_steps(config, args.steps, dense=True)
            payload["memory_reduction_allocated"] = memory_reduction(
                payload["dense"]["peak_allocated_bytes"],
                payload["nast"]["peak_allocated_bytes"],
            )
        atomic_write_json(payload, args.output)
        _print(payload)
        return
    if args.command == "inspect":
        _print(_inspect(args))
        return
    if args.command == "audit-selection":
        payload = audit_selection(
            load_config(args.config),
            batches=args.batches,
            checkpoint=args.checkpoint,
        )
        atomic_write_json(payload, args.output)
        _print(payload)
        return
    raise RuntimeError(f"unhandled command: {args.command}")
