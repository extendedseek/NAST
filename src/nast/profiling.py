"""Measured memory and step-time profiling helpers."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class StepMeasurement:
    elapsed_seconds: float
    processed_tokens: int
    peak_allocated_bytes: int
    peak_reserved_bytes: int
    scoring_seconds: float = 0.0
    forward_seconds: float = 0.0
    backward_seconds: float = 0.0
    optimizer_seconds: float = 0.0

    @property
    def tokens_per_second(self) -> float:
        return self.processed_tokens / max(self.elapsed_seconds, 1.0e-12)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["tokens_per_second"] = self.tokens_per_second
        payload["peak_allocated_gib"] = self.peak_allocated_bytes / 1024**3
        payload["peak_reserved_gib"] = self.peak_reserved_bytes / 1024**3
        return payload


class StepTimer:
    def __init__(self, device: Any):
        self.device = device
        self.started_at = 0.0
        self.cuda_start = None
        self.cuda_end = None

    def __enter__(self):
        import torch

        if self.device.type == "cuda":
            self.cuda_start = torch.cuda.Event(enable_timing=True)
            self.cuda_end = torch.cuda.Event(enable_timing=True)
            self.cuda_start.record()
        else:
            self.started_at = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        import torch

        if self.device.type == "cuda":
            self.cuda_end.record()
            torch.cuda.synchronize(self.device)
        return False

    @property
    def elapsed_seconds(self) -> float:
        if self.cuda_start is not None and self.cuda_end is not None:
            return float(self.cuda_start.elapsed_time(self.cuda_end) / 1000.0)
        return time.perf_counter() - self.started_at


def reset_peak_memory(device: Any) -> None:
    import torch

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)


def memory_peaks(device: Any) -> tuple[int, int]:
    import torch

    if device.type != "cuda":
        return 0, 0
    torch.cuda.synchronize(device)
    return (
        int(torch.cuda.max_memory_allocated(device)),
        int(torch.cuda.max_memory_reserved(device)),
    )


def aggregate_measurements(measurements: list[StepMeasurement]) -> dict[str, Any]:
    if not measurements:
        return {
            "steps": 0,
            "elapsed_seconds": 0.0,
            "processed_tokens": 0,
            "tokens_per_second": 0.0,
            "scoring_seconds": 0.0,
            "mean_scoring_seconds": 0.0,
            "forward_seconds": 0.0,
            "mean_forward_seconds": 0.0,
            "backward_seconds": 0.0,
            "mean_backward_seconds": 0.0,
            "optimizer_seconds": 0.0,
            "mean_optimizer_seconds": 0.0,
            "peak_allocated_bytes": 0,
            "peak_reserved_bytes": 0,
            "peak_allocated_gib": 0.0,
            "peak_reserved_gib": 0.0,
        }
    elapsed = sum(item.elapsed_seconds for item in measurements)
    scoring = sum(item.scoring_seconds for item in measurements)
    forward = sum(item.forward_seconds for item in measurements)
    backward = sum(item.backward_seconds for item in measurements)
    optimizer = sum(item.optimizer_seconds for item in measurements)
    tokens = sum(item.processed_tokens for item in measurements)
    allocated = max(item.peak_allocated_bytes for item in measurements)
    reserved = max(item.peak_reserved_bytes for item in measurements)
    return {
        "steps": len(measurements),
        "elapsed_seconds": elapsed,
        "mean_step_seconds": elapsed / len(measurements),
        "scoring_seconds": scoring,
        "mean_scoring_seconds": scoring / len(measurements),
        "forward_seconds": forward,
        "mean_forward_seconds": forward / len(measurements),
        "backward_seconds": backward,
        "mean_backward_seconds": backward / len(measurements),
        "optimizer_seconds": optimizer,
        "mean_optimizer_seconds": optimizer / len(measurements),
        "processed_tokens": tokens,
        "tokens_per_second": tokens / max(elapsed, 1.0e-12),
        "peak_allocated_bytes": allocated,
        "peak_reserved_bytes": reserved,
        "peak_allocated_gib": allocated / 1024**3,
        "peak_reserved_gib": reserved / 1024**3,
    }


def memory_reduction(dense_peak_bytes: int, method_peak_bytes: int) -> float | None:
    if dense_peak_bytes <= 0:
        return None
    return 1.0 - method_peak_bytes / dense_peak_bytes
