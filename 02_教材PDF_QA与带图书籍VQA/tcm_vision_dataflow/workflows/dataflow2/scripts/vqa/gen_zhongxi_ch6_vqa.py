"""
《中西医结合望诊启迪》第6章 鼻部异常 VQA 生成脚本。

核心机制：雷达扫描裂变（Radar-Scan-Fission）
  - 读取 context_text，识别命中部位
  - 候选池：鼻头/鼻尖、鼻梁、鼻三角区、眉间/双眉、口周、整体面色、舌体
  - 对每个命中部位独立生成 5 题 CoT 题组（强化生化指标提取）
  - Q4 专属：ALT/GGT/TG/GLU/腹部超声/胃镜分级/HP 检测等生化循证

输入：
  maizhen_vqa_workdir/output/manual_split/中西医结合望诊启迪/split_samples.jsonl
  （仅处理 figure_id 前缀为 "6-" 的条目）

输出：
  maizhen_vqa_workdir/output/zhongxi_vqa/ch6/vqa_dataset.jsonl
  maizhen_vqa_workdir/output/zhongxi_vqa/ch6/raw_debug.txt
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
DEFAULT_OUTPUT = WORK_DIR / "output" / "zhongxi_vqa" / "ch6"
BOOK_NAME = "中西医结合望诊启迪"
CHAPTER = 6

# ── 第6章 系统提示词（生化循证升级版）────────────────────────────────────────
ZHONGXI_CH6_PROMPT = """你是一名中西医结合临床望诊专家，专注于鼻部及鼻周异常与腹腔内脏病变的关联判断，并结合现代生化检查进行验证。

【输入格式】
每条样本包含以下字段：
- `image_caption`：图片编号（如"图6-3"）
- `section_title`：病例标题（如"鼻头静脉曲张——脂肪肝"）
- `context_text`：病例文本，包含主诉/既往史/中医望诊/辅助检查/诊断/分析

【雷达扫描规则】
1. 读取 `context_text`，识别本图所展示的**命中部位**（可能有多个）
2. 命中部位候选池：
   - `鼻头/鼻尖`
   - `鼻梁`
   - `鼻三角区`
   - `眉间/双眉`
   - `口周`
   - `整体面色`
   - `舌体`
3. 判断方式：若 context_text 中的"中医望诊"或"查体"字段描述了某部位的异常体征，则视为命中
4. 对每一个命中部位，独立生成一套 5 题 CoT 题组
5. 若无明确命中，则生成一套以"鼻头/鼻尖"为部位的默认题组

【每套题组结构】
Q1（visual_feature）：
  问题：图中患者的[命中部位]有什么客观体征？
  答案：根据 context_text 中"中医望诊"内容，描述该部位的客观视觉特征

Q2（clinical_tcm）：
  问题：结合既往史，该体征提示了什么中医病理机制或脏腑异常？
  答案：结合"既往史"与中医理论，说明中医病理机制（如脾胃湿热、肝胆瘀滞等）

Q3（clinical_diagnosis）：
  问题：结合该体征与患者主诉，最终的临床确诊疾病是什么？
  答案：给出"诊断"字段的完整结论

Q4（gold_standard_labs）：
  问题：辅助检查中涉及哪些生化或影像指标？具体数值或分级结果如何？
  答案：精确提取以下指标（如有，含数值和单位）：
    - 肝功能：ALT（谷丙转氨酶，U/L）、GGT（谷氨酰转肽酶，U/L）
    - 血脂：甘油三酯 TG（mmol/L）、载脂蛋白 B、总胆固醇
    - 血糖：GLU（葡萄糖，mmol/L）、HbA1c
    - 腹部超声：脂肪肝分级（轻/中/重）、囊肿大小（cm）、胆囊息肉直径（mm）
    - 胃镜：炎症分级（浅表性/萎缩性/糜烂性）、HP 检测结果（+/-/++/+++）
    - 肿瘤标志物：CEA（ng/ml）等

Q5（chain_of_thought_mapping）：
  问题：医生是如何将[命中部位]的空间位置与最终诊断联系起来的？
  答案：引用"分析"字段，说明鼻部/面部体征区域与腹腔脏器的中医投射对应关系；
        若无"分析"字段，从其他字段推理并注明"文本未提供明确分析"

【输出格式】
严格输出 JSON，不含任何多余说明。
若有 N 个命中部位，输出包含 N 个内层数组的外层数组：

[
  [
    {
      "question": "图中患者的[命中部位]有什么客观体征？",
      "answer": "（中医望诊体征描述）",
      "type": "visual_feature",
      "hit_region": "[命中部位]"
    },
    {
      "question": "结合既往史，该体征提示了什么中医病理机制或脏腑异常？",
      "answer": "（中医病理机制说明）",
      "type": "clinical_tcm",
      "hit_region": "[命中部位]"
    },
    {
      "question": "结合该体征与患者主诉，最终的临床确诊疾病是什么？",
      "answer": "（诊断结论）",
      "type": "clinical_diagnosis",
      "hit_region": "[命中部位]"
    },
    {
      "question": "辅助检查中涉及哪些生化或影像指标？具体数值或分级结果如何？",
      "answer": "（所有生化指标含数值和单位）",
      "type": "gold_standard_labs",
      "hit_region": "[命中部位]"
    },
    {
      "question": "医生是如何将[命中部位]的空间位置与最终诊断联系起来的？",
      "answer": "（体表投射区与脏腑对应关系分析）",
      "type": "chain_of_thought_mapping",
      "hit_region": "[命中部位]"
    }
  ]
]

【硬性约束】
- 生化数值必须与 context_text 完全一致，不得虚构任何数值
- 若某指标 context_text 未提及，直接跳过，不得补充
- 每套题组的 hit_region 必须与 Q1/Q5 中的 [命中部位] 保持一致
- 严禁输出 <think> 标签或任何内心独白
- 严禁输出 Markdown 代码块标记（如 ```json）
- 必须返回可被 json.loads 解析的 JSON 数组"""


# ── 工具函数 ──────────────────────────────────────────────────────────────────
def _zhongxi_chapter(figure_id: str) -> int:
    """从 figure_id 提取章节号，返回 5-10 或 0（跳过）。"""
    m = re.match(r"^(\d+)-", figure_id.strip())
    if not m:
        return 0
    ch = int(m.group(1))
    return ch if 5 <= ch <= 10 else 0


def _extract_multi_qa_groups(text: str) -> list[list[dict]]:
    """
    从 LLM 响应中解析多套题组。
    兼容格式：
      1. [[{...},...], [{...},...]]  ← 标准多组
      2. [{...},...]                 ← 单组
      3. {"qa_pairs": [...]}        ← 旧格式
    """
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
            return [group for group in parsed if isinstance(group, list) and group]
        elif isinstance(parsed[0], dict):
            return [parsed]
    elif isinstance(parsed, dict):
        qa_pairs = parsed.get("qa_pairs", [])
        if qa_pairs:
            return [qa_pairs]

    return []


# ── 核心处理器 ────────────────────────────────────────────────────────────────
class ZhongxiCh6VQAProcessor(ShezhenVQAProcessor):
    """《中西医结合望诊启迪》第6章 鼻部异常 VQA 处理器（生化循证升级版）。"""

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
        """加载 split_samples.jsonl，仅保留第6章样本（figure_id 前缀为 '6-'）。"""
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
                ch = _zhongxi_chapter(figure_id)
                if ch != CHAPTER:
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
                {"role": "system", "content": ZHONGXI_CH6_PROMPT},
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
                figure_id = sample.get("figure_id", "")

                dbg.write(f"=== [{idx}] {caption} | {title} ===\n")
                dbg.write(raw_text + "\n\n")

                if not raw_text.strip():
                    empty_resp += 1
                    continue

                qa_groups = _extract_multi_qa_groups(raw_text)
                if not qa_groups:
                    no_qa += 1
                    continue

                for group in qa_groups:
                    if not group:
                        continue

                    hit_region = str(group[0].get("hit_region", "鼻头/鼻尖")).strip()
                    region_slug = re.sub(r"[^\w\u4e00-\u9fff]", "_", hit_region)

                    for qa in group:
                        q = str(qa.get("question", "")).strip()
                        a = str(qa.get("answer", "")).strip()
                        qa_type = str(qa.get("type", "")).strip()
                        hit_region_qa = str(qa.get("hit_region", hit_region)).strip()

                        if not q or not a:
                            continue

                        entry_id += 1
                        id_str = f"zhongxi_ch{CHAPTER}_{entry_id:04d}_{region_slug}"

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
                            "hit_region": hit_region_qa,
                            "qa_type": qa_type,
                            "generation_mode": f"zhongxi_ch{CHAPTER}_radar_fission",
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
        description=f"《中西医结合望诊启迪》第{CHAPTER}章 鼻部异常 雷达扫描裂变 VQA 生成脚本"
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
    parser.add_argument(
        "--limit", type=int, default=None, help="最多处理多少条样本（调试用）"
    )
    parser.add_argument(
        "--sample-rate", type=float, default=1.0, help="随机抽样比例 (0,1]"
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--max-workers", type=int, default=5, help="LLM 并发数")
    parser.add_argument("--temperature", type=float, default=0.1, help="LLM 温度")
    parser.add_argument(
        "--model-name", type=str, default="MiniMax-M1", help="模型名"
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()

    processor = ZhongxiCh6VQAProcessor(
        input_dir=input_dir,
        output_dir=output_dir,
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
