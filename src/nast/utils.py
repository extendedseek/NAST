"""Small runtime utilities kept independent from the training stack."""

from __future__ import annotations

import contextlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np


def optional_import(name: str, purpose: str):
    try:
        return __import__(name)
    except ImportError as exc:
        raise RuntimeError(
            f"{name!r} is required for {purpose}. Install the project with `pip install -e .`."
        ) from exc


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def atomic_write_json(payload: Any, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=_json_default)
        handle.write("\n")
    temporary.replace(destination)


def append_jsonl(payload: Any, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=_json_default) + "\n")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "detach"):
        tensor = value.detach().cpu()
        return tensor.item() if tensor.numel() == 1 else tensor.tolist()
    raise TypeError(f"Cannot JSON serialize {type(value).__name__}")


def package_versions(names: list[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def git_state() -> dict[str, str | bool | None]:
    def run(*args: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", *args], stderr=subprocess.DEVNULL, text=True
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(run("status", "--porcelain")),
    }


def environment_metadata() -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "hostname": platform.node(),
        "packages": package_versions(
            ["torch", "transformers", "datasets", "peft", "accelerate", "numpy", "PyYAML"]
        ),
        "git": git_state(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    try:
        import torch

        metadata["torch_cuda_available"] = torch.cuda.is_available()
        metadata["torch_cuda_version"] = torch.version.cuda
        metadata["cudnn_version"] = torch.backends.cudnn.version()
        metadata["gpu_names"] = [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ]
    except ImportError:
        metadata["torch_cuda_available"] = False
    return metadata


@contextlib.contextmanager
def temporary_requires_grad(module: Any, enabled: bool) -> Iterator[None]:
    parameters = list(module.parameters())
    states = [parameter.requires_grad for parameter in parameters]
    try:
        for parameter in parameters:
            parameter.requires_grad_(enabled)
        yield
    finally:
        for parameter, state in zip(parameters, states, strict=True):
            parameter.requires_grad_(state)
