"""Dataset registry, normalization, mixing, tokenization, and collation."""

from __future__ import annotations

import itertools
import random
from pathlib import Path
from typing import Any, Iterable, Iterator

from .config import DatasetSpec, ExperimentConfig
from .formatting import NormalizedExample, build_tokenized_example, normalize_record


DATASET_REGISTRY: dict[str, dict[str, Any]] = {
    "open_platypus": {"path": "garage-bAInd/Open-Platypus", "split": "train"},
    "dolly_15k": {"path": "databricks/databricks-dolly-15k", "split": "train"},
    "oasst1": {"path": "OpenAssistant/oasst1", "split": "train"},
    "gsm8k": {"path": "openai/gsm8k", "config": "main", "split": "train"},
    "math": {"path": "DigitalLearningGmbH/MATH-lighteval", "split": "train"},
    "scienceqa": {"path": "derek-thomas/ScienceQA", "split": "train"},
    "commonsenseqa": {"path": "tau/commonsense_qa", "split": "train"},
    "mmlu": {"path": "cais/mmlu", "config": "all", "split": "test"},
}


def resolve_dataset_spec(spec: DatasetSpec) -> tuple[str, str | None, str]:
    canonical = spec.name.lower().replace("-", "_")
    entry = DATASET_REGISTRY.get(canonical)
    if entry:
        return (
            entry["path"],
            spec.config if spec.config is not None else entry.get("config"),
            spec.split or entry["split"],
        )
    return spec.name, spec.config, spec.split


def _load_raw(spec: DatasetSpec):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install `datasets` to load training or evaluation data") from exc
    path, subset, split = resolve_dataset_spec(spec)
    kwargs: dict[str, Any] = {
        "split": split,
        "streaming": spec.streaming,
    }
    if spec.revision:
        kwargs["revision"] = spec.revision
    if spec.data_files is not None:
        kwargs["data_files"] = spec.data_files
    local_path = Path(path).expanduser()
    if local_path.exists() or path.endswith((".json", ".jsonl")):
        kwargs["data_files"] = spec.data_files or str(local_path)
        return load_dataset("json", **kwargs)
    if subset:
        return load_dataset(path, subset, **kwargs)
    return load_dataset(path, **kwargs)


def _matches_language(row: dict[str, Any], language: str | None) -> bool:
    if language is None:
        return True
    observed = row.get("lang", row.get("language", row.get("locale")))
    return observed is None or str(observed).lower() == language.lower()


def _oasst_pairs(
    rows: Iterable[dict[str, Any]], max_samples: int | None, language: str | None
) -> Iterator[NormalizedExample]:
    messages: dict[str, dict[str, Any]] = {}
    emitted = 0
    deferred: list[dict[str, Any]] = []
    for row in rows:
        message_id = str(row.get("message_id", ""))
        if message_id:
            messages[message_id] = row
        if (
            row.get("role") != "assistant"
            or row.get("deleted")
            or not _matches_language(row, language)
        ):
            continue
        parent = messages.get(str(row.get("parent_id", "")))
        if parent is None:
            deferred.append(row)
            continue
        if (
            parent.get("role") != "prompter"
            or parent.get("deleted")
            or not _matches_language(parent, language)
        ):
            continue
        example = NormalizedExample(
            instruction=str(parent.get("text", "")).strip(),
            response=str(row.get("text", "")).strip(),
            task="conversation",
            source="oasst1",
            example_id=message_id or None,
        )
        if example.instruction and example.response:
            yield example
            emitted += 1
            if max_samples is not None and emitted >= max_samples:
                return
    for row in deferred:
        parent = messages.get(str(row.get("parent_id", "")))
        if (
            parent is None
            or parent.get("role") != "prompter"
            or parent.get("deleted")
            or not _matches_language(parent, language)
        ):
            continue
        example = NormalizedExample(
            instruction=str(parent.get("text", "")).strip(),
            response=str(row.get("text", "")).strip(),
            task="conversation",
            source="oasst1",
            example_id=str(row.get("message_id", "")) or None,
        )
        if example.instruction and example.response:
            yield example
            emitted += 1
            if max_samples is not None and emitted >= max_samples:
                return


def iter_normalized(spec: DatasetSpec) -> Iterator[NormalizedExample]:
    raw = _load_raw(spec)
    canonical = spec.name.lower().replace("-", "_")
    if canonical == "oasst1" or "oasst1" in spec.name.lower():
        yield from _oasst_pairs(raw, spec.max_samples, spec.language)
        return
    emitted = 0
    for row in raw:
        row = dict(row)
        if not _matches_language(row, spec.language):
            continue
        example = normalize_record(spec.name, row)
        if spec.task:
            example.task = spec.task
        if example.instruction and example.response:
            yield example
            emitted += 1
            if spec.max_samples is not None and emitted >= spec.max_samples:
                break


def _weighted_round_robin(
    streams: list[list[NormalizedExample]], weights: list[float], seed: int
) -> list[NormalizedExample]:
    """Deterministically interleave finite datasets without duplicating records."""

    rng = random.Random(seed)
    queues = [list(items) for items in streams]
    for queue in queues:
        rng.shuffle(queue)
    cursors = [0] * len(queues)
    output: list[NormalizedExample] = []
    while True:
        active = [index for index, queue in enumerate(queues) if cursors[index] < len(queue)]
        if not active:
            break
        active_weights = [weights[index] for index in active]
        chosen = rng.choices(active, weights=active_weights, k=1)[0]
        output.append(queues[chosen][cursors[chosen]])
        cursors[chosen] += 1
    return output


def load_normalized_mixture(specs: list[DatasetSpec], seed: int) -> list[NormalizedExample]:
    if not specs:
        raise ValueError("at least one training dataset is required")
    if any("mmlu" in spec.name.lower() for spec in specs):
        raise ValueError("MMLU is evaluation-only and cannot appear in train_datasets")
    streams = [list(iter_normalized(spec)) for spec in specs]
    if any(not stream for stream in streams):
        empty = [spec.name for spec, stream in zip(specs, streams, strict=True) if not stream]
        raise ValueError(f"no usable records loaded from: {', '.join(empty)}")
    return _weighted_round_robin(streams, [spec.weight for spec in specs], seed)


def tokenize_examples(
    examples: Iterable[NormalizedExample],
    tokenizer: Any,
    max_length: int,
    use_chat_template: bool,
) -> list[dict[str, Any]]:
    return [
        build_tokenized_example(example, tokenizer, max_length, use_chat_template)
        for example in examples
    ]


class NASTDataCollator:
    def __init__(self, tokenizer: Any, pad_to_multiple_of: int | None = 8):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch is required for tensor collation") from exc
        max_length = max(len(feature["input_ids"]) for feature in features)
        if self.pad_to_multiple_of:
            multiple = self.pad_to_multiple_of
            max_length = ((max_length + multiple - 1) // multiple) * multiple
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        batch: dict[str, list[Any]] = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
            "response_mask": [],
        }
        metadata = {"source": [], "task": [], "example_id": []}
        for feature in features:
            padding = max_length - len(feature["input_ids"])
            batch["input_ids"].append(feature["input_ids"] + [pad_id] * padding)
            batch["attention_mask"].append(feature["attention_mask"] + [0] * padding)
            batch["labels"].append(feature["labels"] + [-100] * padding)
            batch["response_mask"].append(feature["response_mask"] + [False] * padding)
            for key in metadata:
                metadata[key].append(feature.get(key))
        result = {
            "input_ids": torch.tensor(batch["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(batch["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(batch["labels"], dtype=torch.long),
            "response_mask": torch.tensor(batch["response_mask"], dtype=torch.bool),
        }
        result.update(metadata)
        return result


class TokenizedListDataset:
    def __init__(self, items: list[dict[str, Any]]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.items[index]


def build_train_dataset(config: ExperimentConfig, tokenizer: Any) -> TokenizedListDataset:
    examples = load_normalized_mixture(config.train_datasets, config.training.seed)
    tokenized = tokenize_examples(
        examples,
        tokenizer,
        config.training.max_sequence_length,
        config.model.use_chat_template,
    )
    return TokenizedListDataset(tokenized)


def take_examples(spec: DatasetSpec, limit: int | None) -> list[NormalizedExample]:
    iterator = iter_normalized(spec)
    return list(itertools.islice(iterator, limit)) if limit is not None else list(iterator)
