#!/usr/bin/env python3
"""Create deterministic local train/test splits from verified MLZY records."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "bianzheng_mlzy_verified.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not valid JSON") from exc
            if row.get("status", "success") == "success":
                rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not 0 < args.test_ratio < 1:
        raise ValueError("--test-ratio must be between 0 and 1")

    rows = load_jsonl(args.input)
    if len(rows) < 2:
        raise ValueError("At least two successful records are required to split the dataset")

    shuffled = list(rows)
    random.Random(args.seed).shuffle(shuffled)
    test_count = max(1, min(len(shuffled) - 1, round(len(shuffled) * args.test_ratio)))
    test_rows = shuffled[:test_count]
    train_rows = shuffled[test_count:]

    train_path = args.output_dir / "bianzheng_mlzy_train.jsonl"
    test_path = args.output_dir / "bianzheng_mlzy_test.jsonl"
    write_jsonl(train_path, train_rows)
    write_jsonl(test_path, test_rows)

    print(f"train: {len(train_rows)} -> {train_path}")
    print(f"test: {len(test_rows)} -> {test_path}")


if __name__ == "__main__":
    main()
