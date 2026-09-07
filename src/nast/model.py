"""Model/tokenizer construction and architecture discovery."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .config import ExperimentConfig


def _torch_dtype(name: str):
    import torch

    mapping = {
        "auto": "auto",
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping[name]


def load_tokenizer(config: ExperimentConfig):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install `transformers` to load the tokenizer") from exc
    name = config.model.tokenizer_name_or_path or config.model.name_or_path
    tokenizer = AutoTokenizer.from_pretrained(
        name,
        revision=config.model.revision,
        trust_remote_code=config.model.trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer has neither a pad token nor an EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def _quantization_config(config: ExperimentConfig):
    if config.model.quantization == "none":
        return None
    try:
        import torch
        from transformers import BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError("Install the qlora extra for quantized training") from exc
    if config.model.quantization == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=(
                torch.bfloat16 if config.model.dtype == "bfloat16" else torch.float16
            ),
        )
    return BitsAndBytesConfig(load_in_8bit=True)


def load_model(config: ExperimentConfig, adapter_path: str | Path | None = None):
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise RuntimeError("Install `transformers` to load the language model") from exc
    quantization_config = _quantization_config(config)
    kwargs: dict[str, Any] = {
        "revision": config.model.revision,
        "trust_remote_code": config.model.trust_remote_code,
        "torch_dtype": _torch_dtype(config.model.dtype),
        "attn_implementation": config.model.attention_implementation,
        "low_cpu_mem_usage": True,
    }
    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        kwargs["device_map"] = {"": local_rank}
    checkpoint = Path(adapter_path).expanduser() if adapter_path is not None else None
    is_adapter = checkpoint is not None and (checkpoint / "adapter_config.json").exists()
    model_source = config.model.name_or_path if is_adapter or checkpoint is None else str(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(model_source, **kwargs)
    model.config.use_cache = False

    if is_adapter:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError("Install `peft` to load an adapter checkpoint") from exc
        model = PeftModel.from_pretrained(model, str(checkpoint), is_trainable=False)
        return model

    if checkpoint is not None:
        return model

    if config.lora.enabled:
        try:
            from peft import LoraConfig as PeftLoraConfig
            from peft import get_peft_model, prepare_model_for_kbit_training
        except ImportError as exc:
            raise RuntimeError("Install `peft` to enable LoRA") from exc
        if quantization_config is not None:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=config.training.gradient_checkpointing
            )
        peft_config = PeftLoraConfig(
            task_type="CAUSAL_LM",
            r=config.lora.rank,
            lora_alpha=config.lora.alpha,
            lora_dropout=config.lora.dropout,
            bias=config.lora.bias,
            target_modules=config.lora.target_modules,
        )
        model = get_peft_model(model, peft_config)

    if config.training.gradient_checkpointing:
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    return model


def unwrap_base_model(model: Any) -> Any:
    current = model
    if hasattr(current, "module"):
        current = current.module
    if hasattr(current, "get_base_model"):
        current = current.get_base_model()
    return current


def find_decoder_layers(model: Any) -> list[Any]:
    """Find the decoder block list for common Hugging Face causal LMs."""

    base = unwrap_base_model(model)
    candidate_paths = [
        "model.layers",
        "transformer.h",
        "gpt_neox.layers",
        "model.decoder.layers",
        "transformer.blocks",
    ]
    for path in candidate_paths:
        value = base
        try:
            for part in path.split("."):
                value = getattr(value, part)
        except AttributeError:
            continue
        if hasattr(value, "__len__") and len(value) > 0:
            return list(value)
    candidates: list[tuple[int, Any]] = []
    for name, module in base.named_modules():
        if name.endswith((".layers", ".h", ".blocks")) and hasattr(module, "__len__"):
            try:
                size = len(module)
            except TypeError:
                continue
            if size > 0:
                candidates.append((size, module))
    if candidates:
        return list(max(candidates, key=lambda pair: pair[0])[1])
    raise RuntimeError(
        "Could not locate decoder layers. Add the model's layer path to `find_decoder_layers`."
    )


def count_parameters(model: Any) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {"total": total, "trainable": trainable}


def save_model(model: Any, tokenizer: Any, output_dir: str | Path) -> None:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    unwrapped = model.module if hasattr(model, "module") else model
    unwrapped.save_pretrained(destination, safe_serialization=True)
    tokenizer.save_pretrained(destination)
