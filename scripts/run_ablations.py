#!/usr/bin/env python3
"""Run one or more ablation YAML files without shell-specific assumptions."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "configs",
        nargs="*",
        help="YAML files; defaults to every configs/ablations/*.yaml file",
    )
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    configs = [Path(value) for value in args.configs]
    if not configs:
        configs = sorted((root / "configs" / "ablations").glob("*.yaml"))
    failures = 0
    for config in configs:
        command = [sys.executable, "-m", "nast", "train", "--config", str(config)]
        print(" ".join(command), flush=True)
        if args.dry_run:
            continue
        result = subprocess.run(command, cwd=root, check=False)
        if result.returncode:
            failures += 1
            if not args.keep_going:
                return result.returncode
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
