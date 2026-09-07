#!/usr/bin/env python3
"""Flatten run summaries into an analysis-ready CSV table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def flatten(payload: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in payload.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            result.update(flatten(value, full_key))
        elif isinstance(value, (str, int, float, bool)) or value is None:
            result[full_key] = value
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs")
    parser.add_argument("--output", default="results/run_table.csv")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = sorted(Path(args.root).rglob("run_summary.json"))
    if not paths:
        raise SystemExit(f"No run_summary.json files found under {args.root}")
    rows = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            row = flatten(json.load(handle))
        row["run_directory"] = str(path.parent)
        rows.append(row)
    columns = sorted({key for row in rows for key in row})
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(destination.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
