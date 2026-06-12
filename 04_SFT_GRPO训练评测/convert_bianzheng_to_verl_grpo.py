#!/usr/bin/env python3
"""
Convert normalized TCM bianzheng records to verl GRPO prompt-only data.

verl GRPO/RL data should not include a gold assistant response in the prompt.
This script stores the gold syndromes and structured bianzheng labels in
reward_model.ground_truth / extra_info for a rule-based reward function.
"""

from __future__ import annotations

import argparse
import html
import json
import re
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any


DEFAULT_TRAIN = Path("grpodata/bianzheng_merged_train.jsonl")
DEFAULT_TEST = Path("grpodata/bianzheng_merged_test.jsonl")
DEFAULT_TREE = Path("zhenghou_btree.json")
DEFAULT_OUTPUT_DIR = Path("grpodata/verl")


PROMPT_TEMPLATE = """你是一位经验丰富的国医大师，请根据以下病案信息进行辨证分析，判断证型。
要求：
1. 先在 <think>...</think> 中进行推理，包含：四诊提取、辨证参数推导（八纲/脏腑/气血津液/六淫）、证型判定三个步骤
2. 然后输出证型JSON数组，如 ["证型1", "证型2"]

病案：
{case_json}"""


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} is not valid JSON") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def maybe_write_parquet(path: Path, rows: list[dict[str, Any]]) -> str:
    """Write parquet when common RL data dependencies are available."""
    try:
        import pandas as pd  # type: ignore
        import pyarrow  # noqa: F401  # type: ignore
    except Exception as exc:
        pandas_error = exc
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(path, index=False)
        return "written"

    try:
        from datasets import Dataset  # type: ignore
    except Exception as exc:
        return f"skipped: parquet requires pandas+pyarrow or datasets (pandas path: {pandas_error}; datasets path: {exc})"

    path.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(rows).to_parquet(str(path))
    return "written"


def load_tree_names(path: Path) -> set[str]:
    if not path.exists():
        return set()
    data = json.loads(path.read_text(encoding="utf-8"))
    names = data.get("name_to_code", {})
    if isinstance(names, dict):
        return set(names)
    return set()


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return " ".join(text.split())


def raw_text(value: Any) -> str:
    if value is None:
        return ""
    return clean_text(value)


def strip_field_prefix(value: Any, field_name: str) -> str:
    text = raw_text(value)
    if not text:
        return ""
    if field_name == "主诉":
        return re.sub(r"^主\s*诉\s*[：:。.]?\s*", "", text)
    return re.sub(rf"^{re.escape(field_name)}\s*[：:。.]?\s*", "", text)


def format_age(value: Any) -> str:
    text = clean_text(value)
    if not text:
        return ""
    if text.isdigit():
        return f"{text}岁"
    return text


def split_syndrome_text(value: Any) -> list[str]:
    text = clean_text(value)
    if not text:
        return []
    for sep in ["|", "、", "，", ",", ";", "；", "/", " "]:
        text = text.replace(sep, "|")
    return [item.strip() for item in text.split("|") if item.strip()]


def extract_syndromes(record: dict[str, Any]) -> tuple[list[str], str]:
    converted = record.get("converted", {})
    converted_syndromes = converted.get("syndrome")
    if isinstance(converted_syndromes, list):
        syndromes = [item for item in converted_syndromes if isinstance(item, str) and item.strip()]
        if syndromes:
            return syndromes, "converted.syndrome"

    raw_syndromes = split_syndrome_text(record.get("raw_data", {}).get("证型"))
    if raw_syndromes:
        return raw_syndromes, "raw_data.证型"

    return [], "missing"


def make_case_json(record: dict[str, Any]) -> dict[str, Any]:
    converted = record["converted"]
    case_info = converted.get("case_info", {})
    four = converted.get("four_diagnosis", {})
    raw = record.get("raw_data", {})

    return {
        "ID": raw_text(raw.get("ID")) or clean_text(record.get("original_id") or record.get("case_id") or converted.get("case_id")),
        "性别": raw_text(raw.get("性别")) or clean_text(case_info.get("gender")),
        "年龄": raw_text(raw.get("年龄")) or format_age(case_info.get("age")),
        "主诉": strip_field_prefix(raw.get("主诉"), "主诉") or clean_text(case_info.get("chief_complaint")),
        "症状": raw_text(raw.get("症状")) or clean_text(four.get("wen_zhen")),
        "望闻问切": strip_field_prefix(raw.get("中医望闻切诊"), "中医望闻切诊") or clean_text(" ".join(
            text for text in [four.get("wang"), four.get("wen"), four.get("qie")] if text
        )),
    }


def make_prompt(record: dict[str, Any]) -> str:
    case_json = json.dumps(make_case_json(record), ensure_ascii=False, indent=2)
    return PROMPT_TEMPLATE.format(case_json=case_json)


def build_verl_row(record: dict[str, Any], split: str, index: int, data_source: str, ability: str) -> dict[str, Any]:
    converted = record.get("converted")
    if not isinstance(converted, dict):
        raise ValueError(f"record {index} has no converted object")

    syndromes, syndrome_source = extract_syndromes(record)
    if not syndromes:
        raise ValueError(f"record {index} has invalid converted.syndrome")

    bianzheng = converted.get("bianzheng", {})
    prompt = make_prompt(record)
    ground_truth = {
        "syndrome": syndromes,
        "syndromes": syndromes,
        "bianzheng": bianzheng,
    }

    return {
        "data_source": data_source,
        "prompt": [{"role": "user", "content": prompt}],
        "ability": ability,
        "reward_model": {
            "style": "rule",
            "ground_truth": ground_truth,
        },
        "extra_info": {
            "split": split,
            "index": index,
            "case_id": record.get("case_id") or converted.get("case_id"),
            "record_index": record.get("record_index"),
            "original_id": record.get("original_id"),
            "raw_syndrome": record.get("raw_data", {}).get("证型"),
            "syndrome_source": syndrome_source,
            "syndromes": syndromes,
            "label_count": len(syndromes),
            "bianzheng": bianzheng,
        },
    }


def percentile(values: list[int], pct: float) -> int:
    if not values:
        return 0
    values = sorted(values)
    pos = min(len(values) - 1, max(0, round((len(values) - 1) * pct)))
    return values[pos]


def summarize(rows: list[dict[str, Any]], tree_names: set[str]) -> dict[str, Any]:
    labels: Counter[str] = Counter()
    label_counts: Counter[int] = Counter()
    prompt_lengths: list[int] = []
    bad_labels: Counter[str] = Counter()

    for row in rows:
        syndromes = row["extra_info"]["syndromes"]
        label_counts[len(syndromes)] += 1
        labels.update(syndromes)
        prompt_lengths.append(len(row["prompt"][0]["content"]))
        if tree_names:
            for syndrome in syndromes:
                if syndrome not in tree_names:
                    bad_labels[syndrome] += 1

    return {
        "records": len(rows),
        "unique_syndromes": len(labels),
        "label_count_distribution": dict(sorted(label_counts.items())),
        "bad_tree_label_count": sum(bad_labels.values()),
        "bad_tree_labels": dict(bad_labels.most_common()),
        "prompt_chars": {
            "min": min(prompt_lengths) if prompt_lengths else 0,
            "median": int(median(prompt_lengths)) if prompt_lengths else 0,
            "p95": percentile(prompt_lengths, 0.95),
            "max": max(prompt_lengths) if prompt_lengths else 0,
        },
        "top_syndromes": labels.most_common(30),
    }


def render_report(path: Path, stats: dict[str, Any], sample: dict[str, Any]) -> None:
    sample_prompt = sample["prompt"][0]["content"] if sample else ""
    sample_gt = sample.get("reward_model", {}).get("ground_truth", {}) if sample else {}
    rows = []
    for split, split_stats in stats["splits"].items():
        rows.append(
            "<tr>"
            f"<td>{html.escape(split)}</td>"
            f"<td>{split_stats['records']}</td>"
            f"<td>{split_stats['unique_syndromes']}</td>"
            f"<td>{html.escape(json.dumps(split_stats['label_count_distribution'], ensure_ascii=False))}</td>"
            f"<td>{split_stats['bad_tree_label_count']}</td>"
            f"<td>{html.escape(json.dumps(split_stats['prompt_chars'], ensure_ascii=False))}</td>"
            "</tr>"
        )

    report = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>verl GRPO 辨证数据转换报告</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; line-height: 1.55; color: #1f2933; }}
    h1 {{ font-size: 24px; margin-bottom: 8px; }}
    h2 {{ font-size: 18px; margin-top: 28px; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 12px; }}
    th, td {{ border: 1px solid #d7dde5; padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ background: #f3f6f8; }}
    code, pre {{ background: #f6f8fa; border: 1px solid #d7dde5; border-radius: 6px; }}
    pre {{ padding: 12px; white-space: pre-wrap; overflow-wrap: anywhere; }}
    .muted {{ color: #667085; }}
  </style>
</head>
<body>
  <h1>verl GRPO 辨证数据转换报告</h1>
  <p class="muted">数据为 prompt-only 格式，标准答案存放在 <code>reward_model.ground_truth</code>，不包含 gold assistant response。</p>
  <h2>输出文件</h2>
  <pre>{html.escape(json.dumps(stats["outputs"], ensure_ascii=False, indent=2))}</pre>
  <h2>统计</h2>
  <table>
    <thead><tr><th>split</th><th>records</th><th>unique syndromes</th><th>label count</th><th>bad tree labels</th><th>prompt chars</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
  <h2>样例 Prompt</h2>
  <pre>{html.escape(sample_prompt)}</pre>
  <h2>样例 Ground Truth</h2>
  <pre>{html.escape(json.dumps(sample_gt, ensure_ascii=False, indent=2))}</pre>
</body>
</html>
"""
    path.write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert normalized bianzheng data to verl GRPO format.")
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--tree", type=Path, default=DEFAULT_TREE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--data-source", default="tcm_bianzheng")
    parser.add_argument("--ability", default="bianzheng")
    parser.add_argument("--no-parquet", action="store_true", help="Only write JSONL and skip parquet output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tree_names = load_tree_names(args.tree)
    datasets = {
        "train": load_jsonl(args.train),
        "test": load_jsonl(args.test),
    }

    outputs: dict[str, Any] = {}
    stats = {"splits": {}, "outputs": outputs}
    first_sample: dict[str, Any] | None = None

    for split, source_rows in datasets.items():
        rows = [
            build_verl_row(record, split, index, args.data_source, args.ability)
            for index, record in enumerate(source_rows)
        ]
        if first_sample is None and rows:
            first_sample = rows[0]

        jsonl_path = args.output_dir / f"bianzheng_grpo_{split}.jsonl"
        write_jsonl(jsonl_path, rows)
        split_outputs = {"jsonl": str(jsonl_path)}

        if args.no_parquet:
            split_outputs["parquet"] = "skipped by --no-parquet"
        else:
            parquet_path = args.output_dir / f"bianzheng_grpo_{split}.parquet"
            parquet_status = maybe_write_parquet(parquet_path, rows)
            split_outputs["parquet"] = str(parquet_path) if parquet_status == "written" else parquet_status

        outputs[split] = split_outputs
        stats["splits"][split] = summarize(rows, tree_names)

    stats_path = args.output_dir / "bianzheng_grpo_data_stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    outputs["stats"] = str(stats_path)

    report_path = args.output_dir / "bianzheng_grpo_data_report.html"
    render_report(report_path, stats, first_sample or {})
    outputs["report"] = str(report_path)

    # Re-write stats after report path is known.
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"train records: {stats['splits']['train']['records']}")
    print(f"test records: {stats['splits']['test']['records']}")
    print(f"train jsonl: {outputs['train']['jsonl']}")
    print(f"test jsonl: {outputs['test']['jsonl']}")
    print(f"train parquet: {outputs['train']['parquet']}")
    print(f"test parquet: {outputs['test']['parquet']}")
    print(f"report: {report_path}")
    print(f"bad tree labels train/test: {stats['splits']['train']['bad_tree_label_count']} / {stats['splits']['test']['bad_tree_label_count']}")


if __name__ == "__main__":
    main()
