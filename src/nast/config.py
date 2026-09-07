"""Typed configuration loading and validation for NAST experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Literal, TypeVar, get_args, get_origin, get_type_hints

import yaml


@dataclass
class DatasetSpec:
    name: str
    split: str = "train"
    config: str | None = None
    revision: str | None = None
    weight: float = 1.0
    max_samples: int | None = None
    task: str | None = None
    language: str | None = None
    streaming: bool = False
    data_files: str | dict[str, str] | None = None


@dataclass
class ModelConfig:
    name_or_path: str = "HuggingFaceTB/SmolLM2-135M-Instruct"
    revision: str | None = None
    tokenizer_name_or_path: str | None = None
    trust_remote_code: bool = False
    dtype: Literal["auto", "float32", "float16", "bfloat16"] = "bfloat16"
    attention_implementation: Literal["eager", "sdpa", "flash_attention_2"] = "eager"
    quantization: Literal["none", "4bit", "8bit"] = "none"
    use_chat_template: bool = True


@dataclass
class LoRAConfig:
    enabled: bool = True
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    bias: Literal["none", "all", "lora_only"] = "none"
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )


@dataclass
class SelectorConfig:
    enabled: bool = True
    ranking_strategy: Literal["utility", "random"] = "utility"
    random_seed: int = 42
    use_ci: bool = True
    use_pu: bool = True
    use_tda: bool = True
    ci_lambda: float = 1.5
    ci_last_n_layers: int = 4
    ci_position_normalize: bool = True
    pu_threshold: float = 0.05
    pu_transform: Literal["one_minus_p", "nll"] = "one_minus_p"
    tda_classes: int = 3
    tda_bins: int = 64
    tda_small_sample_quantile: float = 0.25
    tda_representation: Literal["embedding", "last_hidden"] = "embedding"
    utility_alpha: float = 0.4
    utility_beta: float = 0.6
    gradient_proxy: Literal[
        "last_block", "last_two_blocks", "full_dense", "output_head", "nll", "none"
    ] = "last_block"
    r_min: float = 0.15
    r_max: float = 0.50
    entropy_weight: float = 0.20
    loss_weight: float = 0.15
    budget_denominator: Literal["candidate", "response"] = "candidate"
    min_selected_tokens: int = 1
    min_candidate_tokens: int = 1
    epsilon: float = 1.0e-8
    exclude_punctuation_only: bool = False


@dataclass
class TrainingConfig:
    output_dir: str = "outputs/smoke"
    seed: int = 42
    max_sequence_length: int = 512
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    epochs: float = 1.0
    max_steps: int = 20
    learning_rate: float = 2.0e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    mixed_precision: Literal["no", "fp16", "bf16"] = "bf16"
    gradient_checkpointing: bool = True
    backward_mode: Literal["loss_only", "detach"] = "detach"
    log_every: int = 1
    save_every: int = 100
    keep_last_checkpoints: int = 2
    num_workers: int = 0
    prototype_max_samples: int = 1024
    profile_warmup_steps: int = 3
    profile_active_steps: int = 10
    fail_on_nonfinite: bool = True


@dataclass
class EvaluationConfig:
    datasets: list[DatasetSpec] = field(default_factory=list)
    per_device_batch_size: int = 1
    max_samples: int | None = None
    max_new_tokens: int = 256
    temperature: float = 0.0
    output_file: str = "results/evaluation.json"


@dataclass
class TrackingConfig:
    report_to: Literal["none", "tensorboard", "wandb"] = "none"
    project: str = "nast"
    run_name: str | None = None
    log_token_decisions_every: int = 0


@dataclass
class ExperimentConfig:
    experiment_name: str = "nast-smoke"
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    selector: SelectorConfig = field(default_factory=SelectorConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    train_datasets: list[DatasetSpec] = field(default_factory=list)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)

    def validate(self) -> None:
        errors: list[str] = []
        if not self.experiment_name.strip():
            errors.append("experiment_name must not be empty")
        if not self.model.name_or_path.strip():
            errors.append("model.name_or_path must not be empty")
        if self.lora.rank <= 0:
            errors.append("lora.rank must be positive")
        if self.lora.alpha <= 0:
            errors.append("lora.alpha must be positive")
        if not 0.0 <= self.lora.dropout < 1.0:
            errors.append("lora.dropout must be in [0, 1)")
        if not 0.0 <= self.selector.pu_threshold <= 1.0:
            errors.append("selector.pu_threshold must be in [0, 1]")
        if self.selector.ci_last_n_layers < 1:
            errors.append("selector.ci_last_n_layers must be positive")
        if self.selector.ci_lambda < 0:
            errors.append("selector.ci_lambda must be non-negative")
        if not 0.0 < self.selector.r_min <= self.selector.r_max <= 1.0:
            errors.append("selector ratios must satisfy 0 < r_min <= r_max <= 1")
        if self.selector.utility_alpha < 0 or self.selector.utility_beta < 0:
            errors.append("selector utility weights must be non-negative")
        if self.selector.utility_alpha + self.selector.utility_beta <= 0:
            errors.append("at least one selector utility weight must be positive")
        if self.selector.random_seed < 0:
            errors.append("selector.random_seed must be non-negative")
        if self.selector.tda_classes not in (2, 3):
            errors.append("selector.tda_classes must be 2 or 3")
        if self.selector.tda_bins < 8:
            errors.append("selector.tda_bins must be at least 8")
        if not 0.0 <= self.selector.tda_small_sample_quantile <= 1.0:
            errors.append("selector.tda_small_sample_quantile must be in [0, 1]")
        if self.selector.entropy_weight < 0 or self.selector.loss_weight < 0:
            errors.append("selector adaptive-budget weights must be non-negative")
        if self.selector.min_selected_tokens < 1:
            errors.append("selector.min_selected_tokens must be at least 1")
        if self.selector.min_candidate_tokens < 1:
            errors.append("selector.min_candidate_tokens must be at least 1")
        if self.selector.epsilon <= 0:
            errors.append("selector.epsilon must be positive")
        if (
            self.selector.enabled
            and (self.selector.use_ci or self.selector.utility_alpha > 0)
            and self.model.attention_implementation != "eager"
        ):
            errors.append("CI scoring requires model.attention_implementation=eager")
        if self.training.max_sequence_length < 8:
            errors.append("training.max_sequence_length must be at least 8")
        if self.training.per_device_batch_size < 1:
            errors.append("training.per_device_batch_size must be positive")
        if self.training.gradient_accumulation_steps < 1:
            errors.append("training.gradient_accumulation_steps must be positive")
        if self.training.max_steps == 0 or self.training.max_steps < -1:
            errors.append("training.max_steps must be -1 (epoch based) or positive")
        if self.training.epochs <= 0:
            errors.append("training.epochs must be positive")
        if self.training.learning_rate <= 0:
            errors.append("training.learning_rate must be positive")
        if not 0.0 <= self.training.warmup_ratio < 1.0:
            errors.append("training.warmup_ratio must be in [0, 1)")
        if self.training.max_grad_norm <= 0:
            errors.append("training.max_grad_norm must be positive")
        if self.training.log_every < 1 or self.training.save_every < 1:
            errors.append("training log/save intervals must be positive")
        if self.training.keep_last_checkpoints < 1:
            errors.append("training.keep_last_checkpoints must be positive")
        if self.training.prototype_max_samples < 1:
            errors.append("training.prototype_max_samples must be positive")
        if self.training.profile_warmup_steps < 0 or self.training.profile_active_steps < 1:
            errors.append("training profile step counts are invalid")
        if self.evaluation.per_device_batch_size < 1:
            errors.append("evaluation.per_device_batch_size must be positive")
        if self.tracking.log_token_decisions_every < 0:
            errors.append("tracking.log_token_decisions_every must be non-negative")
        if self.model.quantization != "none" and not self.lora.enabled:
            errors.append("quantized training requires LoRA in this implementation")
        for spec in self.train_datasets:
            if spec.weight <= 0:
                errors.append(f"dataset {spec.name!r} has non-positive weight")
        for spec in [*self.train_datasets, *self.evaluation.datasets]:
            if spec.max_samples is not None and spec.max_samples < 1:
                errors.append(f"dataset {spec.name!r} has invalid max_samples")
            if spec.language is not None and not spec.language.strip():
                errors.append(f"dataset {spec.name!r} has an empty language filter")
        if errors:
            raise ValueError("Invalid NAST configuration:\n- " + "\n- ".join(errors))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


T = TypeVar("T")


def _strip_optional(annotation: Any) -> Any:
    origin = get_origin(annotation)
    if origin is None:
        return annotation
    args = get_args(annotation)
    if origin is list:
        return annotation
    if type(None) in args and len(args) == 2:
        return next(arg for arg in args if arg is not type(None))
    return annotation


def _construct(cls: type[T], values: dict[str, Any], path: str = "config") -> T:
    if not isinstance(values, dict):
        raise TypeError(f"{path} must be a mapping, got {type(values).__name__}")
    known = {item.name: item for item in fields(cls)}
    unknown = sorted(set(values) - set(known))
    if unknown:
        raise ValueError(f"Unknown keys in {path}: {', '.join(unknown)}")
    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for name, value in values.items():
        annotation = _strip_optional(hints.get(name, known[name].type))
        origin = get_origin(annotation)
        args = get_args(annotation)
        if is_dataclass(annotation) and isinstance(value, dict):
            kwargs[name] = _construct(annotation, value, f"{path}.{name}")
        elif origin is list and args and is_dataclass(args[0]):
            if not isinstance(value, list):
                raise TypeError(f"{path}.{name} must be a list")
            kwargs[name] = [
                _construct(args[0], item, f"{path}.{name}[{index}]")
                for index, item in enumerate(value)
            ]
        elif origin is Literal:
            if value not in args:
                allowed = ", ".join(repr(item) for item in args)
                raise ValueError(f"{path}.{name} must be one of: {allowed}")
            kwargs[name] = value
        else:
            kwargs[name] = value
    return cls(**kwargs)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_raw_config(path: Path, seen: set[Path]) -> dict[str, Any]:
    if path in seen:
        raise ValueError(f"cyclic configuration inheritance involving {path}")
    seen.add(path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise TypeError(f"configuration root must be a mapping: {path}")
    parent = raw.pop("extends", None)
    if parent is None:
        return raw
    parent_path = (path.parent / str(parent)).resolve()
    return _deep_merge(_load_raw_config(parent_path, seen), raw)


def load_config(path: str | Path) -> ExperimentConfig:
    """Load YAML, resolve optional `extends`, and reject unknown fields."""

    config_path = Path(path).expanduser().resolve()
    raw = _load_raw_config(config_path, set())
    config = _construct(ExperimentConfig, raw)
    config.validate()
    return config


def save_resolved_config(config: ExperimentConfig, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.to_dict(), handle, sort_keys=False, allow_unicode=True)
