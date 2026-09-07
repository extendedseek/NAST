#!/usr/bin/env python3
"""Validate all YAML configurations and key repository invariants."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nast.config import load_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT))
    args = parser.parse_args()
    root = Path(args.root).resolve()
    configs = sorted((root / "configs").rglob("*.yaml"))
    failures = []
    for path in configs:
        try:
            config = load_config(path)
            if any("mmlu" in spec.name.lower() for spec in config.train_datasets):
                raise ValueError("MMLU appears in training data")
        except Exception as exc:  # validation tool should report all failures
            failures.append((path, exc))
    if failures:
        for path, exc in failures:
            print(f"FAIL {path}: {exc}", file=sys.stderr)
        return 1
    print(f"Validated {len(configs)} configurations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
