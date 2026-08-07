"""
《舌诊辨证图谱_周幸来》专用 VQA 生成脚本。

使用新版 10维（第二章）/ 6维（第三章）提示词，
替代 gen_shezhen_vqa_from_split.py 中的旧版固定4问/2问提示词。

输入：
  shezhen_vqa_workdir/output/rerun_20260402/rerun_20260402/舌诊辨证图谱_周幸来/split_samples.jsonl

输出：
  shezhen_vqa_workdir/output/rerun_20260402/rerun_20260402/舌诊辨证图谱_周幸来/vqa_dataset.jsonl
  shezhen_vqa_workdir/output/rerun_20260402/rerun_20260402/舌诊辨证图谱_周幸来/raw_debug.txt

章节路由规则：
  image_caption 匹配 "图2-" → 第二章 10维题库
  image_caption 匹配 "图3-" → 第三章 6维题库
  其他 → 默认使用第二章提示词
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from gen_shezhen_vqa_final import ShezhenVQAProcessor

# ── 路径常量 ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
WORK_DIR = PROJECT_ROOT / "shezhen_vqa_workdir"
DEFAULT_INPUT = (
    WORK_DIR / "output" / "rerun_20260402" / "rerun_20260402" / "舌诊辨证图谱_周幸来"
)
BOOK_NAME = "舌诊辨证图谱_周幸来"

# ── 第二章：10维全属性专属题库 ─────────────────────────────────────────────────
BIANZHEN_CH2_SYSTEM_PROMPT = """你是一位专业的中医舌诊数据标注专家。你的任务是从《舌诊辨证图谱》第二章（基础理论）的图文数据中，提取高质量的视觉问答（VQA）对，用于训练医疗大模型。

【输入数据结构】
每条输入包含以下字段：
- book_name: 书名
- section_title: 章节标题
- image_path: 图片路径
- image_caption: 图片编号（如"图2-1-1"）
- anchor_sentence: 原文锚定句（直接描述该图的核心句子）
- context_text: 上下文段落（包含更丰富的理论阐述）

【核心纪律 - 违反即失败】
1. 答案必须 100% 来自 anchor_sentence 和 context_text，绝对禁止用预训练知识脑补。
2. 原文没有提到的维度，必须静默跳过，不得输出该题。
3. 所有维度（1-10）的问题均使用"图中的舌象"作为主语，禁止在问题中出现具体舌象名称（如"肿胀舌"、"苍老舌"），让模型通过观察图像和文本来回答。
4. 每条问答必须独立完整，不依赖上下文即可理解。
5. 答案严禁出现"本章"、"本书"、"上文"、"如前所述"等词。

【10维检测题库】
请扫描 anchor_sentence 和 context_text，针对当前舌象，按以下顺序逐一检索。有则提取，无则静默跳过。

---

### 第一部分：客观视觉全维度拆解（纯看图）

【重要规则】维度1-4 为纯视觉题，问题中禁止出现舌象名称（如"肿胀舌"、"苍老舌"），必须使用"图中的舌象"作为主语，让模型通过观察图像来回答。

**维度1：津液干湿属性（type: moisture）**
Question 模板（固定，不得修改）：
"图中舌体的干湿润燥程度如何？"

触发条件：原文出现"润"、"滑"、"燥"、"干"、"湿"、"津"、"涸"、"滴"等描述水分的词汇。
典型适用：润苔、滑苔、燥苔、糙苔等苔质类舌象。

---

**维度2：纯颜色属性（type: color）**
Question 模板（固定，不得修改）：
"图中是什么舌色？舌色有什么特征？"

触发条件：原文出现颜色描述词（红、绛、淡白、紫、青、蓝、黄、灰、黑等）。
几乎所有舌象均触发此维度。

---

**维度3：形体与肌理属性（type: shape_texture）**
Question 模板（固定，不得修改）：
"图中舌体的形态和质地有什么特征？"

触发条件：原文出现"胖"、"瘦"、"苍老"、"娇嫩"、"裂纹"、"芒刺"、"粗糙"、"细腻"、"坚敛"等词。

---

**维度4：动态与体态属性（type: mobility）**
Question 模板（固定，不得修改）：
"图中舌体的动态和姿态有什么异常表现？"

触发条件：原文出现"强硬"、"痿软"、"颤动"、"歪斜"、"短缩"、"吐弄"、"屈伸"等动态描述词。
典型适用：第四节"舌态辨证"的所有舌象。

---

### 第二部分：病理、空间与临床推演（找逻辑）

**维度5：核心定性（type: core_pathogenesis）**
Question 模板（固定，不得修改）：
"该舌象总体对应什么中医病机或主证？"

触发条件：原文出现"主"、"属"、"提示"、"为……之征兆"等定性表述。
几乎所有舌象均触发此维度。

---

**维度6：局部空间拓扑（type: spatial_mapping）**
Question 模板（固定，不得修改）：
"该特征出现在舌体不同部位时，分别提示什么病理意义？"

触发条件：原文出现"舌尖"、"舌根"、"舌边"、"舌中"、"舌两侧"等部位词，且与不同病理意义绑定。
典型适用：点刺舌、芒刺舌、偏全苔、瘀斑舌等。

---

**维度7：舌苔/舌质叠加演变（type: interaction_evolution）**
Question 模板（固定，不得修改）：
"该舌象与其他舌苔或舌色兼见时，提示什么证候或病理演变？"

触发条件：原文出现"若见……兼……"、"若……与……并见"、"若……同时……"等叠加描述。

---

**维度8：临床兼见症（type: associated_symptoms）**
Question 模板（固定，不得修改）：
"出现此类舌象的患者，通常还伴随哪些临床症状？"

触发条件：原文出现"伴见"、"常见于"（后接症状而非疾病名）、"兼见"等词，且后接具体症状描述。

---

**维度9：病程预后（type: prognosis）**
Question 模板（固定，不得修改）：
"该舌象反映了怎样的病程阶段或疾病预后？"

触发条件：原文出现"极期"、"后期"、"转愈"、"预后"、"危候"、"佳兆"、"病进"、"病退"等词。

---

### 第三部分：现代医学（找映射）

**维度10：西医微观（type: western_mechanism）**
Question 模板（固定，不得修改）：
"该舌象在现代医学中对应哪些疾病或形成机制？"

触发条件：原文出现"西医学认为"、"现代医学"、"神经"、"血液"、"细胞"、"蛋白"等西医术语段落。

---

## 输出格式

严格输出 JSON，不含任何多余说明：

{
  "qa_pairs": [
    {
      "question": "具体问题（含舌象名，不含模糊指代）",
      "answer": "严格来自原文的答案",
      "type": "维度类型标识"
    }
  ]
}

若当前条目无任何可提取内容（如纯目录、纯页码），返回：
{"qa_pairs": []}

【硬性指令】
- 严禁输出 <think> 标签或任何内心独白。
- 严禁输出 Markdown 代码块标记（如 ```json）。
- 必须返回可被 json.loads 解析的 JSON 对象。"""

# ── 第三章：6维临床辨证专属题库 ──────────────────────────────────────────────
BIANZHEN_CH3_SYSTEM_PROMPT = """你是一位专业的中医舌诊数据标注专家。你的任务是从《舌诊辨证图谱》第三章（临床辨证图谱）的图文数据中，提取高质量的视觉问答（VQA）对，用于训练医疗大模型。

【输入数据结构】
每条输入包含以下字段：
- book_name: 书名
- section_title: 章节标题
- image_path: 图片路径
- image_caption: 图片编号（如"图3-4-11"）
- anchor_sentence: 原文锚定句（极简，通常为一句话）
- context_text: 上下文（包含前后图注，可能含病程分期标题和治疗段落）

【核心纪律 - 违反即失败】
1. 答案必须 100% 来自 anchor_sentence 和 context_text，绝对禁止脑补。
2. 原文没有提到的维度，必须静默跳过，不得输出该题。
3. 答案极简时，就输出极简答案，不得扩充。
4. 每条问答必须独立完整，不依赖上下文即可理解。

【6维临床诊断题库】
请扫描 anchor_sentence 和 context_text，按以下顺序逐一检索。有则提取，无则静默跳过。

---

### 第一部分：诊断前提（临床阶段与特征提取）

**维度1：病程与疾病分期（type: clinical_stage）**
Question 模板（固定，不得修改）：
"图中的舌象处于疾病的哪个阶段？"

触发条件：context_text 中存在带"●"的分期标题（如"●急性期"、"●慢性期"），且当前图片位于该标题之后、下一个"●"标题之前。
提取规则：直接提取"●"后的阶段名称（如"急性期"、"慢性期"）。
若无"●"分期标题，静默跳过此维度。

---

**维度2：舌质独立特征（type: visual_feature_body）**
Question 模板（固定，不得修改）：
"图中的舌质有什么特征？"

触发条件：anchor_sentence 中含有描述舌体本身的词（舌质、舌色、舌体、舌红、舌淡、舌绛、舌紫、舌胖、舌瘦、齿痕、瘀斑、瘀点等）。
提取规则：从 anchor_sentence 中只提取属于"舌肉/舌体"的描述部分，不包含苔的描述。

---

**维度3：舌苔独立特征（type: visual_feature_coating）**
Question 模板（固定，不得修改）：
"图中的舌苔有什么特征？"

触发条件：anchor_sentence 中含有描述舌苔的词（苔、苔黄、苔白、苔腻、苔薄、苔厚、苔干、苔润、苔剥、无苔、少苔等）。
提取规则：从 anchor_sentence 中只提取属于"舌苔"的描述部分，不包含舌质的描述。

---

### 第二部分：诊断结论与干预（临床映射）

**维度4：综合辨证定性（type: syndrome_differentiation）**
Question 模板（固定，不得修改）：
"图中的舌象对应什么中医证型？"

触发条件：anchor_sentence 中含有"属"字后接证型名称（如"属肝胆湿热"、"属热毒犯心"、"属气血两虚"）。
提取规则：提取"属"字及其后的完整证型描述。
此维度几乎每条图注均触发，是第三章最核心的维度。

---

**维度5：局部特异性指征与意义（type: specific_indicator）**
Question 模板（固定，不得修改）：
"图中舌体有什么局部特异性体征？提示什么临床意义？"

触发条件：context_text 中出现对当前图片的补充说明，描述了某个局部特异性体征及其独立意义（如"舌脉怒张，提示病程较久"、"见瘀点瘀斑"）。
若无此类补充说明，静默跳过。

---

**维度6：临床治疗与处方（type: treatment）**
Question 模板（固定，不得修改）：
"针对图中舌象对应的证型，有哪些治疗方案或处方？"

触发条件：context_text 中出现"【中医简易疗法】"或"效验方"等治疗段落。
提取规则：提取完整的治疗方案或处方内容。
若当前图注下无具体治疗段落，静默跳过。

---

## 输出格式

严格输出 JSON，不含任何多余说明：

{
  "qa_pairs": [
    {
      "question": "具体问题",
      "answer": "严格来自原文的答案（极简原文就输出极简答案）",
      "type": "维度类型标识"
    }
  ]
}

若当前条目无任何可提取内容，返回：
{"qa_pairs": []}

【硬性指令】
- 严禁输出 <think> 标签或任何内心独白。
- 严禁输出 Markdown 代码块标记（如 ```json）。
- 必须返回可被 json.loads 解析的 JSON 对象。"""

# ── JSON Schema（带 type 字段）────────────────────────────────────────────────
CH2_TYPE_ENUM = [
    "moisture", "color", "shape_texture", "mobility",
    "core_pathogenesis", "spatial_mapping", "interaction_evolution",
    "associated_symptoms", "prognosis", "western_mechanism",
]

CH3_TYPE_ENUM = [
    "clinical_stage", "visual_feature_body", "visual_feature_coating",
    "syndrome_differentiation", "specific_indicator", "treatment",
]

ALL_TYPE_ENUM = sorted(set(CH2_TYPE_ENUM + CH3_TYPE_ENUM))

BIANZHEN_QA_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "qa_pairs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "answer": {"type": "string"},
                    "type": {"type": "string", "enum": ALL_TYPE_ENUM},
                },
                "required": ["question", "answer", "type"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["qa_pairs"],
    "additionalProperties": False,
}


# ── 章节路由 ──────────────────────────────────────────────────────────────────
def _chapter(image_caption: str) -> int:
    """从 image_caption 提取章节号，默认返回 2。"""
    m = re.match(r"图(\d+)-", image_caption or "")
    return int(m.group(1)) if m else 2


def _select_prompt(sample: dict) -> str:
    ch = _chapter(str(sample.get("image_caption", "") or ""))
    return BIANZHEN_CH3_SYSTEM_PROMPT if ch == 3 else BIANZHEN_CH2_SYSTEM_PROMPT


# ── 核心处理器 ────────────────────────────────────────────────────────────────
class BianZhenVQAProcessor(ShezhenVQAProcessor):
    """《舌诊辨证图谱_周幸来》专用处理器，使用新版 10维/6维提示词。"""

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

    def _load_samples(self, limit: int | None, sample_rate: float, seed: int) -> list[dict]:
        import random

        sample_file = self.input_dir / "split_samples.jsonl"
        if not sample_file.exists():
            raise FileNotFoundError(f"未找到样本文件: {sample_file}")

        rng = random.Random(seed)
        items: list[dict] = []
        with sample_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if sample_rate < 1.0 and rng.random() > sample_rate:
                    continue
                items.append(row)
                if limit and len(items) >= limit:
                    break
        return items

    def run(
        self,
        limit: int | None = None,
        sample_rate: float = 1.0,
        seed: int = 42,
    ) -> Path:
        from tqdm import tqdm

        samples = self._load_samples(limit=limit, sample_rate=sample_rate, seed=seed)
        if not samples:
            raise ValueError("没有可处理的样本")

        print(f"[加载] {BOOK_NAME}: 样本 {len(samples)} 条")

        ch2_count = sum(
            1 for s in samples
            if _chapter(str(s.get("image_caption", "") or "")) == 2
        )
        ch3_count = len(samples) - ch2_count
        print(f"[路由] 第二章(10维): {ch2_count} 条  第三章(6维): {ch3_count} 条")

        user_inputs = [self._build_user_prompt(s) for s in samples]

        conversations = [
            [
                {"role": "system", "content": _select_prompt(s)},
                {"role": "user", "content": u},
            ]
            for s, u in zip(samples, user_inputs)
        ]
        responses = self.llm.generate_from_conversations(conversations)

        out_dir = self.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "vqa_dataset.jsonl"
        debug_file = out_dir / "raw_debug.txt"

        written = 0
        empty_after_clean = 0
        no_qa = 0

        with debug_file.open("w", encoding="utf-8") as dbg, out_file.open("w", encoding="utf-8") as out:
            for idx, (sample, resp) in enumerate(
                tqdm(zip(samples, responses), total=len(samples), desc=f"解析 {BOOK_NAME}")
            ):
                raw = resp or ""
                cleaned = self._clean_response(raw)

                dbg.write(f"# INDEX {idx}\n# RAW\n{raw}\n# CLEANED\n{cleaned}\n\n")

                if idx < 3:
                    ch = _chapter(str(sample.get("image_caption", "") or ""))
                    preview = raw[:200].replace("\n", " ")
                    print(f"[调试] #{idx + 1} ch={ch} caption={sample.get('image_caption', '')[:20]} resp={preview}")

                if not cleaned:
                    empty_after_clean += 1
                    continue

                qa_pairs = self._extract_qa_pairs(cleaned)
                if not qa_pairs:
                    no_qa += 1
                    continue

                section_title_raw = str(sample.get("section_title", "") or "")
                section_title_clean, section_noise = self._normalize_section_title(
                    section_title_raw,
                    str(sample.get("image_caption", "") or ""),
                )

                for qa in qa_pairs:
                    record = {
                        "book_name": sample.get("book_name", BOOK_NAME),
                        "section_title": section_title_clean,
                        "section_title_raw": section_title_raw,
                        "section_title_is_noise": section_noise,
                        "image_path": sample.get("image_path", ""),
                        "image_caption": self._caption_for_prompt(sample),
                        "context_text": sample.get("context_text", ""),
                        "question": qa["question"],
                        "answer": qa["answer"],
                        "generation_mode": "bianzhen_new",
                    }
                    if qa.get("type"):
                        record["type"] = qa["type"]
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    written += 1

        print(
            f"[完成] {BOOK_NAME}: 写入 {written} 条 QA -> {out_file}\n"
            f"       (清洗后为空 {empty_after_clean}, 无 qa_pair {no_qa})"
        )
        print(f"[调试] 原始响应日志: {debug_file}")
        return out_file


# ── CLI ───────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="《舌诊辨证图谱_周幸来》新版 10维/6维 VQA 生成脚本"
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
        default="",
        help="输出目录，默认与 --input-dir 相同",
    )
    parser.add_argument("--limit", type=int, default=None, help="最多处理多少条样本（调试用）")
    parser.add_argument("--sample-rate", type=float, default=1.0, help="随机抽样比例 (0,1]")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--max-workers", type=int, default=5, help="LLM 并发数")
    parser.add_argument("--temperature", type=float, default=0.1, help="LLM 温度")
    parser.add_argument("--model-name", type=str, default="MiniMax-M1", help="模型名")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser() if args.output_dir.strip() else input_dir

    processor = BianZhenVQAProcessor(
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
