"""
《实用中医舌诊彩色图谱》专用 VQA 生成脚本。

使用 8维原子化提示词，对舌质颜色、形貌、局灶、舌苔颜色、厚薄、润燥、
主病、分支辨证进行细粒度拆解。

输入：
  shezhen_vqa_workdir/output/rerun_20260402/rerun_20260402/实用中医舌诊彩色图谱/split_samples.jsonl

输出：
  shezhen_vqa_workdir/output/rerun_20260402/rerun_20260402/实用中医舌诊彩色图谱/vqa_dataset.jsonl
  shezhen_vqa_workdir/output/rerun_20260402/rerun_20260402/实用中医舌诊彩色图谱/raw_debug.txt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from gen_shezhen_vqa_final import ShezhenVQAProcessor

# ── 路径常量 ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
WORK_DIR = PROJECT_ROOT / "shezhen_vqa_workdir"
DEFAULT_INPUT = (
    WORK_DIR / "output" / "rerun_20260402" / "rerun_20260402" / "实用中医舌诊彩色图谱"
)
BOOK_NAME = "实用中医舌诊彩色图谱"

# ── 8维原子化系统提示词 ────────────────────────────────────────────────────────
SHITU_SYSTEM_PROMPT = """你是一位专业的中医舌诊数据标注专家。你的任务是从《实用中医舌诊彩色图谱》的图文数据中，提取高质量的视觉问答（VQA）对，用于训练医疗大模型。

【输入数据结构】
每条输入包含以下字段：
- book_name: 书名
- section_title: 章节标题（含舌象编号与名称，如"9 绛紫少苔津润舌"）
- image_path: 图片路径
- context_text: 完整文本（含"舌象"描述段与"主病"段）

【核心纪律 - 违反即失败】
1. 答案必须 100% 来自 context_text，绝对禁止用预训练知识脑补。
2. 原文没有提到的维度，必须静默跳过，不得输出该题。
3. 所有问题均使用"图中的舌象"作为主语，禁止在问题中出现具体舌象名称（如"绛紫舌"、"淡白舌"）。
4. 答案极简时，就输出极简答案，不得扩充。
5. 每条问答必须独立完整，不依赖上下文即可理解。

【8维原子化题库】
请扫描 context_text，按以下顺序逐一检索。有则提取，无则静默跳过。

---

### 第一部分：舌质（舌体肉）的原子化拆解

**维度1：舌体纯颜色属性（type: visual_feature_color）**
Question 模板（固定，不得修改）：
"图中是什么舌色？舌色有什么特征？"

触发条件：context_text 的"舌象"段中出现颜色描述词（红、绛、淡白、紫、青、蓝等）。
提取规则：只提取颜色词，切断其他描述。

---

**维度2：舌体三维形貌属性（type: visual_feature_shape）**
Question 模板（固定，不得修改）：
"图中舌体的形态有什么特征？"

触发条件：context_text 的"舌象"段中出现"胖"、"瘦"、"嫩"、"老"、"齿痕"等形体词。
提取规则：只提取形体与轮廓描述，不包含颜色或苔的描述。

---

**维度3：舌体表面特异性局灶（type: visual_feature_lesion）**
Question 模板（固定，不得修改）：
"图中舌体表面有什么局部病变？分别在哪个部位？"

触发条件：context_text 的"舌象"段中出现"裂纹"、"点刺"、"芒刺"、"红点"、"瘀斑"、"瘀点"等局灶词。
提取规则：将"局部病变"与"空间坐标"死绑输出，如"尖部有突起红点；中部有少数裂纹"。
若无空间位置描述，仅输出病变类型。

---

### 第二部分：舌苔的原子化拆解

**维度4：舌苔纯颜色属性（type: visual_feature_coating_color）**
Question 模板（固定，不得修改）：
"图中是什么苔色？"

触发条件：context_text 的"舌象"段中出现苔色描述词（白苔、黄苔、灰苔、黑苔等）。
提取规则：只提取苔的颜色，切断其他描述。

---

**维度5：舌苔厚薄与分布形态（type: visual_feature_coating_thickness）**
Question 模板（固定，不得修改）：
"图中舌苔的厚薄和分布有什么特征？"

触发条件：context_text 的"舌象"段中出现"薄"、"厚"、"无苔"、"少苔"、"剥"、"花剥"、"中部"、"根部"等厚薄或分布词。
提取规则：将厚薄程度与分布位置绑定输出，如"中部有极薄白苔，余处光莹无苔"。

---

**维度6：舌苔津液干湿属性（type: visual_feature_coating_moisture）**
Question 模板（固定，不得修改）：
"图中舌苔的润燥程度如何？"

触发条件：context_text 的"舌象"段中出现"润"、"湿"、"滑"、"燥"、"干"、"涸"等描述苔面水分的词。
若无此类描述，静默跳过。

---

### 第三部分：中医定性与逻辑

**维度7：核心主病（type: clinical_reasoning）**
Question 模板（固定，不得修改）：
"图中的舌象主要提示什么中医病机或证候？"

触发条件：context_text 中出现"主病"段落。
提取规则：提取"主病"段的完整内容，保留编号与分条结构。
此维度几乎每条样本均触发。

---

**维度8：发病条件与分支辨证（type: conditional_differentiation）**
Question 模板（固定，不得修改）：
"该舌象在不同发病条件下（如外感与内伤）分别对应什么证候？"

触发条件：context_text 的"主病"段中出现"在外感"、"在内伤"、"若……则"、"或……或"等分支辨证描述。
提取规则：提取各分支条件及其对应证候，保持原文逻辑结构。
若主病段无分支辨证，静默跳过。

---

## 输出格式

严格输出 JSON，不含任何多余说明：

{
  "qa_pairs": [
    {
      "question": "具体问题（使用'图中的舌象'，不含具体舌象名）",
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

# ── JSON Schema ───────────────────────────────────────────────────────────────
SHITU_TYPE_ENUM = [
    "visual_feature_color",
    "visual_feature_shape",
    "visual_feature_lesion",
    "visual_feature_coating_color",
    "visual_feature_coating_thickness",
    "visual_feature_coating_moisture",
    "clinical_reasoning",
    "conditional_differentiation",
]

SHITU_QA_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "qa_pairs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "answer": {"type": "string"},
                    "type": {"type": "string", "enum": SHITU_TYPE_ENUM},
                },
                "required": ["question", "answer", "type"],
            },
        }
    },
    "required": ["qa_pairs"],
}


# ── 核心处理器 ────────────────────────────────────────────────────────────────
class ShituVQAProcessor(ShezhenVQAProcessor):
    """《实用中医舌诊彩色图谱》专用处理器，使用 8维原子化提示词。"""

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

    def _build_user_prompt(self, sample: dict) -> str:
        """构建用户侧输入，直接传入 context_text 和 section_title。"""
        return json.dumps(
            {
                "book_name": sample.get("book_name", BOOK_NAME),
                "section_title": sample.get("section_title", ""),
                "image_path": sample.get("image_path", ""),
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
            raise ValueError("没有可处理的样本")

        print(f"[加载] {BOOK_NAME}: 样本 {len(samples)} 条")

        user_inputs = [self._build_user_prompt(s) for s in samples]

        conversations = [
            [
                {"role": "system", "content": SHITU_SYSTEM_PROMPT},
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
        empty_after_clean = 0
        no_qa = 0

        with debug_file.open("w", encoding="utf-8") as dbg, out_file.open("w", encoding="utf-8") as out:
            for idx, (sample, resp) in enumerate(
                tqdm(zip(samples, responses), total=len(samples), desc=f"解析 {BOOK_NAME}")
            ):
                raw_text = resp if isinstance(resp, str) else (resp.get("content", "") if resp else "")
                dbg.write(f"=== [{idx}] {sample.get('section_title', '')} ===\n")
                dbg.write(raw_text + "\n\n")

                payload = self._extract_json_payload(raw_text)
                if payload is None:
                    no_qa += 1
                    continue

                qa_pairs = payload.get("qa_pairs", [])
                if not qa_pairs:
                    empty_after_clean += 1
                    continue

                for qa in qa_pairs:
                    q = str(qa.get("question", "")).strip()
                    a = str(qa.get("answer", "")).strip()
                    t = str(qa.get("type", "")).strip()
                    if not q or not a:
                        continue
                    row = {
                        **{k: v for k, v in sample.items() if k != "anchor_sentence"},
                        "question": q,
                        "answer": a,
                        "generation_mode": "shitu_atomic",
                        "type": t,
                    }
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    written += 1

        print(f"[完成] 写入 {written} 条 | 空响应 {no_qa} 条 | 空QA {empty_after_clean} 条")
        print(f"[输出] {out_file}")
        return out_file


# ── CLI ───────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="《实用中医舌诊彩色图谱》8维原子化 VQA 生成脚本"
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

    processor = ShituVQAProcessor(
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
