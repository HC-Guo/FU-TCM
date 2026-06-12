#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 qa_output 转换为 tcm_sft_data.json 格式
"""

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def convert_qa_to_sft(input_dir: str, output_file: str):
    """
    将 qa_output 目录下的所有 JSON 文件转换为 SFT 训练格式
    """
    base_path = Path(input_dir)
    sft_data = []

    # 查找所有 JSON 文件
    json_files = sorted(base_path.rglob("*.json"))
    print(f"找到 {len(json_files)} 个 JSON 文件")

    for filepath in json_files:
        if filepath.suffix != '.json' or filepath.name.endswith('.bak'):
            continue

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"  读取失败: {filepath} - {e}")
            continue

        # 提取 QA 对
        for pian in data.get('qa_data', []):
            for qa in pian.get('qa_pairs', []):
                q = qa.get('q', '').strip()
                a = qa.get('a', '').strip()

                # 跳过空的问题或答案
                if not q or not a:
                    continue

                sft_item = {
                    "instruction": "你是一位国医大师，请回答以下中医知识",
                    "input": q,
                    "output": a,
                    "type": qa.get('type', '')
                }
                sft_data.append(sft_item)

        # 显示进度
        if len(sft_data) % 10000 == 0:
            print(f"  已处理 {len(sft_data)} 条数据...")

    # 保存结果
    print(f"\n总共转换 {len(sft_data)} 条 QA 对")

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sft_data, f, ensure_ascii=False, indent=2)

    print(f"已保存到: {output_file}")

    return len(sft_data)

if __name__ == '__main__':
    INPUT_DIR = PROJECT_ROOT / 'qa_output'
    OUTPUT_FILE = PROJECT_ROOT / 'sft_data' / 'tcm_sft_data.json'

    convert_qa_to_sft(INPUT_DIR, OUTPUT_FILE)
