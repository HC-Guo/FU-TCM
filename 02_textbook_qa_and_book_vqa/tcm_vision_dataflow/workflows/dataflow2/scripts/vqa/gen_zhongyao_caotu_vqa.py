"""
gen_zhongyao_caotu_vqa.py
为《中国中药草图谱》（上册+下册）的 split_samples.jsonl 生成 VQA。

输入：zhongyao_caotu_workdir/output/{上册,下册}/split_samples.jsonl
输出：zhongyao_caotu_workdir/output/{上册,下册}/vqa_dataset.jsonl

每条 split_sample → 调用 LLM → 解析 XML → 写入扁平 QA pair
支持断点续跑：已有 vqa_dataset.jsonl 中的 image_path 不重复生成。
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from xml.etree import ElementTree as ET

# ── 路径配置 ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
WORK_DIR = PROJECT_ROOT / "zhongyao_caotu_workdir"
OUT_ROOT = WORK_DIR / "output"

sys.path.insert(0, str(PROJECT_ROOT.parent / "Dataflow" / "run_dataflow" / "api_pipelines"))
sys.path.insert(0, str(PROJECT_ROOT.parent / "Dataflow" / "run_dataflow"))

# ── Prompt ────────────────────────────────────────────────────────────────────
PROMPT_PATH = (
    PROJECT_ROOT.parent.parent
    / "面诊及脉诊_vqa_prompt工程"
    / "zhongyao_caotu"
    / "vqa_prompt.md"
)
SYSTEM_PROMPT = PROMPT_PATH.read_text(encoding="utf-8") if PROMPT_PATH.exists() else ""

# ── LLM 配置 ──────────────────────────────────────────────────────────────────
API_URL = os.getenv("MINIMAX_API_URL", "https://api.minimaxi.com/v1/chat/completions")
MODEL_NAME = os.getenv("MINIMAX_MODEL", "MiniMax-M1")
MAX_WORKERS = int(os.getenv("ZHONGYAO_VQA_MAX_WORKERS", "5"))
BATCH_SIZE = int(os.getenv("ZHONGYAO_VQA_BATCH_SIZE", "10"))  # 每批发送给 LLM 的样本数

VOLUMES = ["上册", "下册"]


# ── XML 解析 ──────────────────────────────────────────────────────────────────

def _clean_llm_output(raw: str) -> str:
    """去掉 <think>、markdown 代码块等干扰，提取 XML 内容"""
    # 优先：如果原文已含 <drug> 标签，直接提取 <drug>...</drug> 块
    # （MiniMax-M1 把思维链放在 <think> 里，XML 直接跟在后面，不用 <answer> 包裹）
    drug_match = re.search(r"(<drug>[\s\S]*?</drug>)", raw, flags=re.IGNORECASE)
    if drug_match:
        return drug_match.group(1).strip()

    # 次选：外层有 <answer>...</answer> 包裹整个 XML（部分模型格式）
    # 注意：不能用 findall，否则会把 <qa_pair> 里的 <answer> 也匹配进来
    outer_answer = re.search(r"<answer>([\s\S]*?)</answer>\s*$", raw, flags=re.IGNORECASE)
    if outer_answer:
        text = outer_answer.group(1)
    else:
        # 兜底：去掉 <think> 块
        text = re.sub(r"<think>[\s\S]*?</think>", "", raw, flags=re.IGNORECASE)

    # 去掉 markdown 代码块包裹（```xml ... ``` 或 ``` ... ```）
    text = re.sub(r"```(?:xml)?\s*([\s\S]*?)```", r"\1", text, flags=re.IGNORECASE)
    return text.strip()


def parse_xml_to_qa_pairs(xml_text: str, sample: dict) -> list[dict]:
    """
    解析 LLM 输出的 XML，返回扁平 QA pair 列表。
    每条包含：book_name, volume, drug_id, title, image_path, dimension, question, answer
    """
    if not xml_text or "<empty>" in xml_text.lower():
        return []

    # 包一层根节点，防止多个 <drug> 块解析失败
    wrapped = f"<root>{xml_text}</root>"
    try:
        root = ET.fromstring(wrapped)
    except ET.ParseError:
        # 尝试修复常见问题：去掉非法字符
        cleaned = re.sub(r"&(?!amp;|lt;|gt;|quot;|apos;)", "&amp;", xml_text)
        try:
            root = ET.fromstring(f"<root>{cleaned}</root>")
        except ET.ParseError:
            return []

    pairs = []
    # 先找 <drug> 元素；若没有则直接在根节点找 <qa_pair>
    drug_els = root.findall(".//drug")
    if not drug_els:
        drug_els = [root]
    for drug_el in drug_els:
        for qa in drug_el.findall(".//qa_pair"):
            question = (qa.findtext("question") or "").strip()
            answer = (qa.findtext("answer") or "").strip()
            dimension = (qa.findtext("dimension") or "").strip()
            if not question or not answer:
                continue
            pairs.append({
                "book_name": sample["book_name"],
                "volume": sample["volume"],
                "drug_id": sample.get("drug_id"),
                "title": sample["title"],
                "image_path": sample["image_path"],
                "dimension": dimension,
                "question": question,
                "answer": answer,
            })
    return pairs


# ── LLM 调用（单条，带重试）────────────────────────────────────────────────────

def call_llm_single(sample: dict, llm_serving) -> str:
    """将单条 split_sample 发给 LLM，返回原始文本"""
    user_content = json.dumps(
        {
            "title": sample["title"],
            "image_path": sample["image_path"],
            "context": sample["context"],
        },
        ensure_ascii=False,
    )
    responses = llm_serving.generate_from_input([user_content], system_prompt=SYSTEM_PROMPT)
    return responses[0] if responses else ""


def call_llm_batch(samples: list[dict], llm_serving) -> list[str]:
    """批量发送，返回与 samples 等长的响应列表"""
    inputs = [
        json.dumps(
            {
                "title": s["title"],
                "image_path": s["image_path"],
                "context": s["context"],
            },
            ensure_ascii=False,
        )
        for s in samples
    ]
    return llm_serving.generate_from_input(inputs, system_prompt=SYSTEM_PROMPT)


# ── 主处理逻辑 ────────────────────────────────────────────────────────────────

def process_volume(volume: str, llm_serving, debug_n: int = 0) -> None:
    vol_dir = OUT_ROOT / volume
    split_path = vol_dir / "split_samples.jsonl"
    vqa_path = vol_dir / "vqa_dataset.jsonl"

    if not split_path.exists():
        print(f"[WARN] 找不到 {split_path}，跳过")
        return

    samples = [json.loads(l) for l in split_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if debug_n > 0:
        samples = samples[:debug_n]
        print(f"[调试] {volume}: 仅处理前 {debug_n} 条")

    # 读取已有结果，用于断点续跑
    done_images: set[str] = set()
    existing_rows: list[dict] = []
    if vqa_path.exists():
        for line in vqa_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                existing_rows.append(row)
                done_images.add(row["image_path"])
        print(f"[续跑] {volume}: 已有 {len(done_images)} 张图片的 QA，跳过")

    todo = [s for s in samples if s["image_path"] not in done_images]
    print(f"[{volume}] 待处理: {len(todo)} / {len(samples)} 条")

    if not todo:
        print(f"[{volume}] 全部已完成，无需重跑")
        return

    new_rows: list[dict] = []
    total_batches = (len(todo) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx in range(0, len(todo), BATCH_SIZE):
        batch = todo[batch_idx: batch_idx + BATCH_SIZE]
        batch_num = batch_idx // BATCH_SIZE + 1
        print(f"  [{volume}] batch {batch_num}/{total_batches} ({len(batch)} 条)...")

        try:
            responses = call_llm_batch(batch, llm_serving)
        except Exception as e:
            print(f"  [ERROR] batch {batch_num} 调用失败: {e}，逐条重试...")
            responses = []
            for s in batch:
                try:
                    r = call_llm_single(s, llm_serving)
                    responses.append(r)
                    time.sleep(0.5)
                except Exception as e2:
                    print(f"    [ERROR] {s['title']} 失败: {e2}")
                    responses.append("")

        batch_new_rows: list[dict] = []
        for sample, raw in zip(batch, responses):
            cleaned = _clean_llm_output(raw or "")
            pairs = parse_xml_to_qa_pairs(cleaned, sample)
            if pairs:
                batch_new_rows.extend(pairs)
            else:
                print(f"    [WARN] {sample['title']} 无有效 QA（raw 长度={len(raw or '')}）")
                # debug 模式下保存原始输出供排查（用 drug_id 避免中文文件名问题）
                if debug_n > 0:
                    drug_id = sample.get("drug_id") or "unknown"
                    dbg_path = vol_dir / f"debug_raw_{drug_id}.txt"
                    dbg_path.write_text(
                        f"=== RAW ===\n{raw or 'NONE'}\n\n=== CLEANED ===\n{cleaned}",
                        encoding="utf-8",
                    )
                    print(f"    [DEBUG] 原始输出已保存到 {dbg_path}")

        new_rows.extend(batch_new_rows)

        # 每批完成后追加写入，防止中途崩溃丢失
        with vqa_path.open("a", encoding="utf-8") as f:
            for row in batch_new_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  [{volume}] batch {batch_num} 写入 {len(batch_new_rows)} 条 QA")

    total_new = len(new_rows)
    total_all = len(existing_rows) + total_new
    print(f"[{volume}] 完成: 新增 {total_new} 条 QA，累计 {total_all} 条 -> {vqa_path}")


# ── 入口 ──────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="为中国中药草图谱生成 VQA")
    parser.add_argument("--volumes", nargs="+", default=VOLUMES, help="处理哪些册（默认上册+下册）")
    parser.add_argument("--debug", type=int, default=0, help="调试模式：只处理前N条（0=全量）")
    args = parser.parse_args()

    if not SYSTEM_PROMPT:
        print(f"[ERROR] 找不到提示词文件: {PROMPT_PATH}")
        sys.exit(1)

    from dataflow.serving import APILLMServing_request

    llm_serving = APILLMServing_request(
        api_url=API_URL,
        model_name=MODEL_NAME,
        key_name_of_api_key="DF_API_KEY",
        max_workers=MAX_WORKERS,
        read_timeout=300,
    )

    for vol in args.volumes:
        process_volume(vol, llm_serving, debug_n=args.debug)

    print("\n全部完成。")


if __name__ == "__main__":
    main()
