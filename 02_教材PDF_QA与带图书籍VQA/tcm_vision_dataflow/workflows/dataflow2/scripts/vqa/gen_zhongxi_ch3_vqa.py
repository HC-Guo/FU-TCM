"""
《中西医结合望诊启迪》第3章 皮肤全息与毛细血管理论图 VQA 生成脚本。

核心机制：理论图解问答（Theory-Diagram-QA）
  - 第3章为皮肤望诊理论基础，图片为皮肤全息应答图、毛细血管变化图、皮肤颜色图等
  - 非临床病例图，无"主诉/诊断/辅助检查"结构
  - 生成围绕"皮肤视觉特征识别"和"全息理论应用"的 5 题组

输入：
  maizhen_vqa_workdir/output/manual_split/中西医结合望诊启迪/split_samples.jsonl
  （仅处理 figure_id 前缀为 "3-" 的条目）

输出（临时中间文件）：
  maizhen_vqa_workdir/output/zhongxi_vqa/ch3/vqa_dataset.jsonl
  maizhen_vqa_workdir/output/zhongxi_vqa/ch3/raw_debug.txt

最终合并后写入：
  maizhen_vqa_workdir/output/unified_output/中西医结合望诊启迪/vqa_dataset.jsonl
  （由 merge_zhongxi_vqa.py 完成合并）
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from gen_shezhen_vqa_final import ShezhenVQAProcessor

# ── 路径常量 ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
WORK_DIR = PROJECT_ROOT / "maizhen_vqa_workdir"
DEFAULT_INPUT = WORK_DIR / "output" / "manual_split" / "中西医结合望诊启迪"
DEFAULT_OUTPUT = WORK_DIR / "output" / "zhongxi_vqa" / "ch3"
BOOK_NAME = "中西医结合望诊启迪"
CHAPTER = 3

# ── 第3章 系统提示词（皮肤全息理论图版）──────────────────────────────────────
ZHONGXI_CH3_PROMPT = """你是一名中西医结合临床望诊专家，擅长解读皮肤望诊理论图示，并将图示内容转化为教学问答。

【输入格式】
每条样本包含以下字段：
- `image_caption`：图片编号（如"图3-2"）
- `section_title`：图片标题（如"皮肤全息应答图"）
- `context_text`：图片对应的理论说明文字

【任务说明】
本章图片为皮肤望诊理论示意图（皮肤全息应答图、毛细血管变化图、皮肤颜色变化图等），非临床病例图。
请根据 context_text 的理论内容，生成一套 5 题教学问答组。

【每套题组结构】
每套题组为一个 JSON 数组，包含 5 个题目对象：

Q1（visual_feature_recognition）：
  问题：该图展示的皮肤/毛细血管有哪些可观察到的视觉特征？
  答案：根据 context_text 描述图示中皮肤或毛细血管的形态、颜色、大小等视觉特征

Q2（holographic_theory）：
  问题：该图所体现的皮肤全息理论核心观点是什么？
  答案：从 context_text 中提取皮肤全息理论的核心内容（如皮肤反映内脏状态的机制）

Q3（pathological_significance）：
  问题：图示中的皮肤/毛细血管变化在临床上提示哪些病理意义？
  答案：从 context_text 中提取皮肤变化与疾病的对应关系

Q4（tcm_western_integration）：
  问题：该图如何体现中西医结合的望诊思路？
  答案：从 context_text 中提取中医理论与现代医学（如微循环、组织学）的结合点

Q5（diagnostic_guidance）：
  问题：临床医生如何利用该图示指导皮肤望诊的实际操作？
  答案：从 context_text 中提取操作方法、观察要点或注意事项；若文本未提及，则注明"文本未提供相关说明"

【输出格式】
输出一个包含 5 个题目对象的 JSON 数组：

[
  {
    "question": "该图展示的皮肤/毛细血管有哪些可观察到的视觉特征？",
    "answer": "（根据context_text描述）",
    "type": "visual_feature_recognition",
    "hit_region": "皮肤/毛细血管"
  },
  {
    "question": "该图所体现的皮肤全息理论核心观点是什么？",
    "answer": "（全息理论核心内容）",
    "type": "holographic_theory",
    "hit_region": "皮肤/毛细血管"
  },
  {
    "question": "图示中的皮肤/毛细血管变化在临床上提示哪些病理意义？",
    "answer": "（病理对应关系）",
    "type": "pathological_significance",
    "hit_region": "皮肤/毛细血管"
  },
  {
    "question": "该图如何体现中西医结合的望诊思路？",
    "answer": "（中西医结合点）",
    "type": "tcm_western_integration",
    "hit_region": "皮肤/毛细血管"
  },
  {
    "question": "临床医生如何利用该图示指导皮肤望诊的实际操作？",
    "answer": "（操作方法或注意事项）",
    "type": "diagnostic_guidance",
    "hit_region": "皮肤/毛细血管"
  }
]

【硬性约束】
- 答案必须严格来源于 context_text，不得凭空推断或使用预训练知识扩充
- 若 context_text 中未涉及某题所需内容，答案中注明"文本未提供相关说明"
- 严禁输出 <think> 标签或任何内心独白
- 严禁输出 Markdown 代码块标记（如 ```json）
- 必须返回可被 json.loads 解析的 JSON 数组"""


# ── 工具函数 ──────────────────────────────────────────────────────────────────
def _zhongxi_chapter(figure_id: str) -> int:
    """从 figure_id（如 '3-2'）提取章节号。"""
    m = re.match(r"^(\d+)-", figure_id.strip())
    if not m:
        return 0
    return int(m.group(1))


def _extract_qa_list(text: str) -> list[dict]:
    """从 LLM 响应中解析单套题组（一个包含5个对象的数组）。"""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"```(?:json)?", "", cleaned).replace("```", "").strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        bracket_match = re.search(r"(\[[\s\S]*\]|\{[\s\S]*\})", cleaned)
        if not bracket_match:
            return []
        try:
            parsed = json.loads(bracket_match.group(1))
        except json.JSONDecodeError:
            return []

    if isinstance(parsed, list):
        if not parsed:
            return []
        if isinstance(parsed[0], list):
            return parsed[0]
        elif isinstance(parsed[0], dict):
            return parsed
    elif isinstance(parsed, dict):
        qa_pairs = parsed.get("qa_pairs", [])
        if qa_pairs:
            return qa_pairs

    return []


# ── 核心处理器 ────────────────────────────────────────────────────────────────
class ZhongxiCh3VQAProcessor(ShezhenVQAProcessor):
    """《中西医结合望诊启迪》第3章 皮肤全息理论图 VQA 处理器。"""

    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        model_name: str,
        max_workers: int,
        temperature: float,
    ):
        self.input_dir = input_dir
        super().__init__(
            model_name=model_name,
            max_workers=max_workers,
            temperature=temperature,
            mode="enriched",
            output_dir=output_dir,
        )

    def _load_samples(
        self,
        limit: int | None,
        sample_rate: float,
        seed: int,
    ) -> list[dict]:
        """加载 split_samples.jsonl，仅保留第3章样本（figure_id 前缀为 '3-'）。"""
        import random

        sample_file = self.input_dir / "split_samples.jsonl"
        if not sample_file.exists():
            raise FileNotFoundError(f"未找到样本文件: {sample_file}")

        rng = random.Random(seed)
        items: list[dict] = []
        skipped = 0
        with sample_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue

                figure_id = row.get("figure_id", "")
                if _zhongxi_chapter(figure_id) != CHAPTER:
                    skipped += 1
                    continue

                if sample_rate < 1.0 and rng.random() > sample_rate:
                    continue

                items.append(row)
                if limit and len(items) >= limit:
                    break

        print(f"[加载] 第{CHAPTER}章样本 {len(items)} 条，跳过其他章节 {skipped} 条")
        return items

    def _build_user_prompt(self, sample: dict) -> str:
        return json.dumps(
            {
                "image_caption": sample.get("image_caption", ""),
                "section_title": sample.get("section_title", ""),
                "context_text": sample.get("context_text", ""),
            },
            ensure_ascii=False,
        )

    def run(
        self,
        limit: int | None = None,
        sample_rate: float = 1.0,
        seed: int = 42,
    ) -> Path:
        from tqdm import tqdm

        samples = self._load_samples(limit=limit, sample_rate=sample_rate, seed=seed)
        if not samples:
            raise ValueError(f"第{CHAPTER}章没有可处理的样本")

        user_inputs = [self._build_user_prompt(s) for s in samples]

        conversations = [
            [
                {"role": "system", "content": ZHONGXI_CH3_PROMPT},
                {"role": "user", "content": u},
            ]
            for u in user_inputs
        ]
        responses = self.llm.generate_from_conversations(conversations)

        out_dir = self.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "vqa_dataset.jsonl"
        debug_file = out_dir / "raw_debug.txt"

        written = 0
        empty_resp = 0
        no_qa = 0
        entry_id = 0

        with debug_file.open("w", encoding="utf-8") as dbg, out_file.open(
            "w", encoding="utf-8"
        ) as out:
            for idx, (sample, resp) in enumerate(
                tqdm(
                    zip(samples, responses),
                    total=len(samples),
                    desc=f"解析 {BOOK_NAME} 第{CHAPTER}章",
                )
            ):
                raw_text = (
                    resp
                    if isinstance(resp, str)
                    else (resp.get("content", "") if resp else "")
                )
                caption = sample.get("image_caption", f"图{CHAPTER}-{idx}")
                title = sample.get("section_title", "")

                dbg.write(f"=== [{idx}] {caption} | {title} ===\n")
                dbg.write(raw_text + "\n\n")

                if not raw_text.strip():
                    empty_resp += 1
                    continue

                qa_list = _extract_qa_list(raw_text)
                if not qa_list:
                    no_qa += 1
                    continue

                for qa in qa_list:
                    q = str(qa.get("question", "")).strip()
                    a = str(qa.get("answer", "")).strip()
                    qa_type = str(qa.get("type", "")).strip()
                    hit_region = str(qa.get("hit_region", "皮肤/毛细血管")).strip()

                    if not q or not a:
                        continue

                    entry_id += 1
                    row = {
                        "book_name": BOOK_NAME,
                        "section_title": title,
                        "section_title_raw": title,
                        "section_title_is_noise": False,
                        "image_path": sample.get("image_path", ""),
                        "image_caption": caption,
                        "context_text": sample.get("context_text", ""),
                        "question": q,
                        "answer": a,
                        "hit_region": hit_region,
                        "qa_type": qa_type,
                        "generation_mode": f"zhongxi_ch{CHAPTER}_theory_diagram",
                    }
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    written += 1

        print(
            f"[完成] 写入 {written} 条 | 空响应 {empty_resp} 条 | 无QA {no_qa} 条"
        )
        print(f"[输出] {out_file}")
        return out_file


# ── CLI ───────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"《中西医结合望诊启迪》第{CHAPTER}章 皮肤全息理论图 VQA 生成脚本"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(DEFAULT_INPUT),
        help="包含 split_samples.jsonl 的目录（默认：%(default)s）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help="输出目录（默认：%(default)s）",
    )
    parser.add_argument("--limit", type=int, default=None, help="最多处理多少条样本（调试用）")
    parser.add_argument("--sample-rate", type=float, default=1.0, help="随机抽样比例 (0,1]")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--max-workers", type=int, default=5, help="LLM 并发数")
    parser.add_argument("--temperature", type=float, default=0.1, help="LLM 温度")
    parser.add_argument("--model-name", type=str, default="MiniMax-M1", help="模型名")
    args = parser.parse_args()

    processor = ZhongxiCh3VQAProcessor(
        input_dir=Path(args.input_dir).expanduser(),
        output_dir=Path(args.output_dir).expanduser(),
        model_name=args.model_name,
        max_workers=args.max_workers,
        temperature=args.temperature,
    )
    processor.run(
        limit=args.limit,
        sample_rate=args.sample_rate,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
