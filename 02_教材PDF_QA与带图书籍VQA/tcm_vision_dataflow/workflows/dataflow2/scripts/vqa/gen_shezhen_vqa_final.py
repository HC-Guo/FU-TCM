from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

from tqdm import tqdm

from dataflow.serving import APILLMServing_request
from stratify_filter_vqa import process_one_file

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent
SPLIT_SAMPLES_DIR = PROJECT_ROOT / "shezhen_vqa_workdir" / "split_samples"
VQA_OUTPUT_DIR = PROJECT_ROOT / "shezhen_vqa_workdir" / "vqa_results"
VQA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Prompt
SHEZHEN_VQA_SYSTEM_PROMPT_ENRICHED = """你是一个中医舌诊专家。请根据提供的舌象图文片段提取视觉问答（VQA）对。

【定位规则 - 最优先执行】
上下文中可能包含多张图片的描述。你必须先在上下文中找到与"图片标注"（image_caption，如"图2-1-6"）编号完全一致的那句引用，然后只基于该引用所在的句子或段落生成问答。如果上下文中找不到与该编号直接对应的描述，只生成 1 个保守问答，答案仅描述图注标题本身。

【任务要求】
1. 答案必须严格来自上下文中与该图编号对应的描述，禁止引入任何外部知识或预训练记忆。
2. 如果对应段落描述详细，生成 2-3 个问答对，优先覆盖：视觉特征、病理意义、辨证要点。
3. 如果对应段落描述简短，只生成 1-2 个问答对，答案保持保守。
4. 问题必须清晰可独立理解，答案需简洁且医学表达规范。
5. 不要把原始图片标注机械拼进问题，尤其不要直接保留"图中（1）""（4）"这类编号前缀，也不要把判断性结论整段塞进问题。
6. 回答视觉问题时，优先描述可见部位与外观，如舌尖、舌中部、舌边、舌根、舌苔颜色、厚薄、腻腐、脱落等。
7. 视觉类问题要尽量自然，优先写成"观察图中舌象……""请描述图中舌象……"这类问法，不要把标题式短语或症状串直接当作问题主语。

【输出格式 - 严格 JSON】
只输出 JSON，不要输出任何解释、前后缀或代码块，格式如下：
{
  "qa_pairs": [
    {"question": "...", "answer": "..."},
    {"question": "...", "answer": "..."}
  ]
}

【硬性指令】
- 严禁输出 <think> 标签或任何内心独白。
- 严禁输出 Markdown 代码块标记（如 ```json）。
- 必须返回可被 json.loads 解析的 JSON 对象。"""

SHEZHEN_VQA_SYSTEM_PROMPT_STRICT = """你是一个中医舌诊专家。请根据提供的舌象图文片段提取视觉问答（VQA）对。

【定位规则 - 最优先执行】
上下文中可能包含多张图片的描述。你必须先在上下文中找到与"图片标注"（image_caption，如"图2-1-6"）编号完全一致的那句引用，然后只基于该引用所在的句子或段落生成问答。如果上下文中找不到与该编号直接对应的描述，只生成 1 个保守问答，答案仅描述图注标题本身。

【任务要求】
1. 答案必须严格来自上下文中与该图编号对应的描述，绝对禁止引入任何外部知识或预训练记忆。
2. 如果对应段落描述详细，生成 2-3 个问答对；如果描述简短，只生成 1-2 个问答对。
3. 答案必须保持保守，不得扩展到上下文未明确给出的病机细节。
4. 不要把原始图片标注机械拼进问题，尤其不要直接保留"图中（1）""（4）"这类编号前缀，也不要把判断性结论整段塞进问题。
5. 回答视觉问题时，优先描述可见部位与外观，如舌尖、舌中部、舌边、舌根、舌苔颜色、厚薄、腻腐、脱落等。
6. 视觉类问题要尽量自然，优先写成"观察图中舌象……""请描述图中舌象……"这类问法，不要把标题式短语或症状串直接当作问题主语。
7. 【视觉引导规则】至少有 1 个问题必须采用"视觉引导式"提问，即先描述图中可见的外观特征（如颜色、质地、分布），再询问其中医意义，而不是直接用舌象名称（如"绛舌""青舌"）开头提问。
   - 正确示例：「图中舌质呈深红偏紫、舌面干燥的舌象，在中医辨证中提示什么病理变化？」
   - 错误示例：「绛舌在中医辨证中的意义是什么？」（直接使用术语名称，模型无需视觉理解）

【输出格式 - 严格 JSON】
只输出 JSON，不要输出任何解释、前后缀或代码块，格式如下：
{
  "qa_pairs": [
    {"question": "...", "answer": "..."},
    {"question": "...", "answer": "..."}
  ]
}

【硬性指令】
- 严禁输出 <think> 标签或任何内心独白。
- 严禁输出 Markdown 代码块标记（如 ```json）。
- 必须返回可被 json.loads 解析的 JSON 对象。"""

USER_PROMPT_TEMPLATE = """【输入信息】
- 书籍: {book_name}
- 章节: {section_title}
- 图片路径: {image_path}
- 图片标注: {image_caption}
- 样本类型: {sample_hint}
- 原文锚定句（含图标引用的原句，最高优先级）:
{anchor_sentence}
- 上下文描述（供补充参考）:
{context_text}

【重要提示】生成问答时，必须以"原文锚定句"为第一依据，该句直接描述了"{image_caption}"所对应的舌象特征。上下文仅供补充说明，禁止从上下文中引入与本图标注无关的其他图片描述。

请按要求返回 JSON。"""

QA_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "qa_pairs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "answer": {"type": "string"},
                },
                "required": ["question", "answer"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["qa_pairs"],
    "additionalProperties": False,
}

NOISE_SECTION_PATTERNS = [
    "图书在版编目",
    "CIP",
    "版权",
    "前言",
    "目录",
    "封面",
    "扉页",
]

QUALITY_RISK_FLAGS = {
    "very_short_context",
    "high_expansion_ratio",
    "treatment_term_with_short_context",
}

QUALITY_AUDIT_FIELDS = {
    "_quality_score",
    "_tier",
    "_flags",
    "_meta",
}

VISUAL_QUESTION_KEYWORDS = {
    "视觉",
    "外观",
    "特征",
    "表现",
    "形态",
    "颜色",
    "色泽",
    "状态",
    "舌色",
    "舌质",
    "舌苔",
    "描述",
}

NON_VISUAL_QUESTION_KEYWORDS = {
    "主病",
    "病机",
    "病理",
    "辨证",
    "提示",
    "意义",
    "诊断",
    "如何理解",
    "治疗",
    "预后",
}


EXPLICIT_TREATMENT_QUESTION_KEYWORDS = {
    "治疗",
    "调理",
    "治则",
    "治法",
    "原则",
    "怎么办",
    "如何处理",
    "如何调整",
    "日常养护",
    "饮食",
    "护理",
}

TREATMENT_ANSWER_CUES = [
    "治疗应",
    "治则",
    "治法",
    "调理应",
    "调护应",
    "日常养护",
    "饮食方面",
    "饮食调理",
    "护理上",
    "建议",
]

VISUAL_SUBJECT_CUES = {
    "舌",
    "舌象",
    "舌苔",
    "舌质",
    "舌色",
    "舌尖",
    "舌边",
    "舌根",
    "红点",
    "厚苔",
    "薄苔",
    "黄苔",
    "白苔",
    "黑苔",
}

SPARSE_SAMPLE_HINTS = {
    "悬针", "来蛇", "去蛇", "曲虫", "齿痕", "裂纹", "瘀点", "瘀斑", "点纹", "舌",
    "苔", "红", "白", "黄", "紫", "淡", "暗", "滑", "腻", "剥",
}


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _strip_quality_fields(row: dict) -> dict:
    out = dict(row)
    for field in QUALITY_AUDIT_FIELDS:
        out.pop(field, None)
    return out


def _emit_quality_ready_file(
    input_path: Path,
    min_context_gold: int,
    allow_noise_gold: bool,
    dedupe: bool,
    drop_risk_short_context: bool,
    drop_noise_sections: bool,
) -> tuple[Path, dict]:
    summary = process_one_file(
        input_path=input_path,
        min_context_gold=max(1, int(min_context_gold)),
        allow_noise_gold=allow_noise_gold,
        dedupe=dedupe,
    )
    audit_path = Path(summary["outputs"]["audit"])
    audit_rows = _read_jsonl(audit_path)

    kept_rows: list[dict] = []
    dropped_reject = 0
    dropped_risk = 0
    dropped_noise = 0

    for row in audit_rows:
        tier = str(row.get("_tier", "")).strip().lower()
        if tier == "reject":
            dropped_reject += 1
            continue
        if drop_noise_sections and bool(row.get("section_title_is_noise", False)):
            dropped_noise += 1
            continue
        flags = {str(v) for v in (row.get("_flags") or [])}
        if drop_risk_short_context and (flags & QUALITY_RISK_FLAGS):
            dropped_risk += 1
            continue
        kept_rows.append(_strip_quality_fields(row))

    quality_ready_path = input_path.with_name(f"{input_path.stem}_quality_ready.jsonl")
    _write_jsonl(quality_ready_path, kept_rows)

    quality_summary = {
        "input": str(input_path),
        "quality_ready": str(quality_ready_path),
        "source_summary": summary,
        "drop_risk_short_context": drop_risk_short_context,
        "drop_noise_sections": drop_noise_sections,
        "kept": len(kept_rows),
        "dropped_reject": dropped_reject,
        "dropped_noise_sections": dropped_noise,
        "dropped_risk_short_context": dropped_risk,
    }
    quality_summary_path = input_path.with_name(f"{input_path.stem}_quality_ready_summary.json")
    quality_summary_path.write_text(json.dumps(quality_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return quality_ready_path, quality_summary


def _senior_compat_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_senior_compat.jsonl")


def _to_senior_compat_row(row: dict, row_id: int, source_name: str) -> dict:
    image_path = str(row.get("image_path", "") or "").strip()
    image_name = Path(image_path).name if image_path else ""
    compat_image_path = f"images/{image_name}" if image_name else image_path

    section_title = str(row.get("section_title", "") or "").strip()
    chapter_title = section_title
    if section_title.startswith("图注章节:"):
        chapter_title = section_title.replace("图注章节:", "", 1).strip() or section_title

    label = str(row.get("image_caption", "") or "").strip() or chapter_title or f"图 {row_id}"
    source = source_name.strip() if source_name.strip() else str(row.get("book_name", "") or "").strip()

    return {
        "id": row_id,
        "label": label,
        "chapter_title": chapter_title,
        "question": str(row.get("question", "") or "").strip(),
        "answer": str(row.get("answer", "") or "").strip(),
        "image_path": compat_image_path,
        "source": source,
    }


def _export_senior_compat_file(input_path: Path, source_name: str = "") -> tuple[Path, int]:
    rows = _read_jsonl(input_path)
    compat_rows = [_to_senior_compat_row(row, idx + 1, source_name) for idx, row in enumerate(rows)]
    out_path = _senior_compat_output_path(input_path)
    _write_jsonl(out_path, compat_rows)
    return out_path, len(compat_rows)


class ShezhenVQAProcessor:
    def __init__(
        self,
        model_name: str,
        max_workers: int,
        temperature: float,
        mode: str,
        drop_noise_sections: bool = False,
        max_context_chars: int = 1500,
        output_dir: Path | None = None,
    ):
        api_key = os.getenv("DF_API_KEY") or os.getenv("MINIMAX_API_KEY")
        if not api_key:
            raise ValueError("未检测到 API Key，请设置 DF_API_KEY 或 MINIMAX_API_KEY")
        os.environ["DF_API_KEY"] = api_key
        api_url = os.getenv("MINIMAX_API_URL", "https://api.minimaxi.com/v1/chat/completions")
        key_prefix = api_key[:10] + ("..." if len(api_key) > 10 else "")
        print(f"[配置] api_url={api_url}")
        print(f"[配置] model={model_name}")
        print(f"[配置] key_prefix={key_prefix} len={len(api_key)}")
        print(f"[配置] mode={mode}")
        print(f"[配置] drop_noise_sections={drop_noise_sections}")
        print(f"[配置] max_context_chars={max_context_chars}")
        self.mode = mode
        self.drop_noise_sections = drop_noise_sections
        self.max_context_chars = max(200, int(max_context_chars))
        self.output_dir = output_dir or VQA_OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"[配置] output_dir={self.output_dir}")

        self.llm = APILLMServing_request(
            api_url=api_url,
            model_name=model_name,
            key_name_of_api_key="DF_API_KEY",
            max_workers=max_workers,
            read_timeout=float(os.getenv("SHEZHEN_VQA_READ_TIMEOUT", "180")),
            temperature=temperature,
        )

    @staticmethod
    def _normalize_section_title(raw_title: str, image_caption: str) -> tuple[str, bool]:
        title = (raw_title or "").strip()
        if not title:
            return "未标注章节", True
        for token in NOISE_SECTION_PATTERNS:
            if token in title:
                fallback = (image_caption or "").strip()
                if fallback:
                    return f"图注章节:{fallback[:20]}", True
                return "未标注章节", True
        return title, False

    @staticmethod
    def _is_sparse_visual_sample(sample: dict) -> bool:
        caption = str(sample.get("image_caption", "") or "").strip()
        context = str(sample.get("context_text", "") or "").strip()
        if len(context) >= 24:
            return False
        if "图" not in caption:
            return False
        return any(token in caption for token in SPARSE_SAMPLE_HINTS)

    @staticmethod
    def _clean_response(text: str) -> str:
        """Remove reasoning and wrappers; keep payload body."""
        if not text:
            return ""
        cleaned = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)
        cleaned = re.sub(r"```(?:json|xml)?\s*([\s\S]*?)```", r"\1", cleaned, flags=re.IGNORECASE).strip()

        # Some providers wrap final answer in <answer>...</answer>.
        answer_parts = re.findall(r"<answer>([\s\S]*?)</answer>", cleaned, flags=re.IGNORECASE)
        if answer_parts:
            cleaned = "\n".join(answer_parts).strip()
        return cleaned.strip()

    @staticmethod
    def _extract_json_payload(text: str) -> dict | None:
        """Parse JSON payload from raw text; tolerant to wrappers."""
        if not text:
            return None
        body = text.strip()
        # Prefer fenced json content.
        m = re.search(r"```json\s*([\s\S]*?)```", body, flags=re.IGNORECASE)
        if m:
            body = m.group(1).strip()
        else:
            # Fallback to first {...} block.
            brace = re.search(r"\{[\s\S]*\}", body)
            if brace:
                body = brace.group(0).strip()
        try:
            payload = json.loads(body)
            return payload if isinstance(payload, dict) else None
        except Exception:
            return None

    @staticmethod
    def _extract_qa_pairs(text: str) -> list[dict[str, str]]:
        """Extract QA pairs from JSON first, then XML fallback."""
        qa_pairs: list[dict[str, str]] = []
        if not text:
            return qa_pairs

        payload = ShezhenVQAProcessor._extract_json_payload(text)
        if payload is not None:
            arr = payload.get("qa_pairs", [])
            if isinstance(arr, list):
                for item in arr:
                    if not isinstance(item, dict):
                        continue
                    question = str(item.get("question", "")).strip()
                    answer = str(item.get("answer", "")).strip()
                    if question and answer:
                        pair: dict[str, str] = {"question": question, "answer": answer}
                        qa_type = str(item.get("type", "")).strip()
                        if qa_type:
                            pair["type"] = qa_type
                        qa_pairs.append(pair)
            if qa_pairs:
                return qa_pairs

        # XML fallback for compatibility with earlier prompt style.
        chapter_blocks = re.findall(r"<chapter>[\s\S]*?</chapter>", text, flags=re.IGNORECASE)
        if not chapter_blocks:
            chapter_blocks = [text]
        for chapter in chapter_blocks:
            pair_blocks = re.findall(r"<qa_pair>[\s\S]*?</qa_pair>", chapter, flags=re.IGNORECASE)
            for block in pair_blocks:
                q_match = re.search(r"<question>([\s\S]*?)</question>", block, flags=re.IGNORECASE)
                a_match = re.search(r"<answer>([\s\S]*?)</answer>", block, flags=re.IGNORECASE)
                if not (q_match and a_match):
                    continue
                question = q_match.group(1).strip()
                answer = a_match.group(1).strip()
                if question and answer:
                    qa_pairs.append({"question": question, "answer": answer})
        return qa_pairs

    @staticmethod
    def _compress_caption(text: str, max_len: int = 24) -> str:
        text = re.sub(r"\s+", " ", str(text or "")).strip()
        if not text:
            return ""
        if "：" in text:
            prefix = text.split("：", 1)[0].strip()
            if prefix:
                text = prefix
        if len(text) <= max_len:
            return text
        clipped = text[:max_len]
        last_break = max(clipped.rfind(sep) for sep in "，、：:；;。！？!?（）() ")
        if last_break >= max_len // 2:
            clipped = clipped[:last_break]
        return clipped.rstrip("，、：:；;。！？!?（）() 的了而且并 ")

    @classmethod
    def _caption_for_prompt(cls, sample: dict) -> str:
        image_caption = cls._compress_caption(sample.get("image_caption", ""), max_len=24)
        if image_caption:
            return image_caption
        section_title = str(sample.get("section_title", "") or "").strip()
        if section_title.startswith("图注章节:"):
            section_title = section_title.replace("图注章节:", "", 1).strip()
        return cls._compress_caption(section_title, max_len=20)

    @staticmethod
    def _question_asks_treatment(question: str) -> bool:
        question = str(question or "").strip()
        return any(token in question for token in EXPLICIT_TREATMENT_QUESTION_KEYWORDS)

    @staticmethod
    def _strip_caption_prefix(text: str) -> str:
        text = str(text or "").strip()
        text = re.sub(r"^[（(]?\d+[)）]\s*", "", text)
        text = re.sub(r"^图中[（(]?\d+[)）]?", "", text)
        text = re.sub(r"^观察这张图片[，,]?", "", text)
        text = re.sub(r"^观察图片[，,]?", "", text)
        return text.strip("，,、：:；; ")

    @staticmethod
    def _collapse_repeated_tail(text: str) -> str:
        text = str(text or "").strip()
        max_size = min(len(text) // 2, 14)
        for size in range(max_size, 1, -1):
            frag = text[-size:]
            if text[:-size].endswith(frag):
                return text[:-size]
        return text

    @staticmethod
    def _ensure_sentence_punct(text: str) -> str:
        text = str(text or "").strip()
        if not text:
            return text
        text = text.rstrip("，,、；;：: ")
        if text[-1] not in "。！？!?":
            text += "。"
        return text

    @staticmethod
    def _is_title_like_visual_subject(text: str) -> bool:
        text = str(text or "").strip()
        if not text:
            return False
        noisy_tokens = {
            "怎么办",
            "说明",
            "提示",
            "导致",
            "引起",
            "可能",
            "如果",
            "为什么",
            "如何",
            "的是",
            "积食了",
            "咳嗽不停",
            "有时干咳",
            "有痰",
            "发热",
        }
        if len(text) >= 12:
            return True
        if any(token in text for token in noisy_tokens):
            return True
        return any(sep in text for sep in "，、：:；;")

    @classmethod
    def _natural_visual_anchor(cls, sample: dict) -> str:
        pool = " ".join(
            [
                str(sample.get("image_caption", "") or ""),
                str(sample.get("section_title", "") or ""),
                str(sample.get("context_text", "") or "")[:200],
            ]
        )
        if any(token in pool for token in {"孩子", "小朋友", "儿童", "婴儿"}):
            subject = "图中孩子"
        elif any(token in pool for token in {"患者", "病人"}):
            subject = "图中患者"
        else:
            subject = "图中"

        if any(token in pool for token in {"嘴唇", "唇色", "面色", "嘴唇颜色"}):
            return f"{subject}的舌象和面色"
        if subject == "图中":
            return "图中的舌象"
        return f"{subject}的舌象"

    @classmethod
    def _naturalize_visual_question(cls, sample: dict, question: str) -> str:
        question = str(question or "").strip()
        if not question or not cls._looks_like_visual_question(question):
            return question

        anchor = cls._natural_visual_anchor(sample)
        if "颜色和状态" in question:
            return f"请描述{anchor}的颜色和状态。"
        if "显著" in question and "表现" in question:
            return f"观察{anchor}，有哪些显著表现？"
        if "最突出的特征" in question:
            return f"{anchor}最突出的特征是什么？"
        if "外观" in question and ("值得注意" in question or "有哪些" in question):
            return f"观察{anchor}，有哪些值得注意的外观特征？"
        if "舌色" in question and "舌苔" in question:
            return f"观察{anchor}，请描述其舌色、舌体和舌苔特征。"
        return question

    @classmethod
    def _sanitize_question(cls, sample: dict, question: str) -> str:
        question = str(question or "").strip()
        if not question:
            return question

        short_caption = cls._caption_for_prompt(sample)
        safe_subject = short_caption if any(token in short_caption for token in VISUAL_SUBJECT_CUES) else "该舌象"
        raw_caption = str(sample.get("image_caption", "") or "").strip()
        raw_caption_clean = cls._strip_caption_prefix(raw_caption)
        if raw_caption:
            if raw_caption in question:
                question = question.replace(raw_caption, safe_subject)
            else:
                raw_prefix = raw_caption[: min(18, len(raw_caption))]
                if raw_prefix and raw_prefix in question and len(question) > 34:
                    question = question.replace(raw_prefix, safe_subject)

        if raw_caption_clean and len(raw_caption_clean) >= 6 and raw_caption_clean in question:
            question = question.replace(raw_caption_clean, safe_subject)

        question = cls._strip_caption_prefix(question)
        question = question.replace("这张图片展示的该舌象", "这张图片展示的舌象")
        question = question.replace("图中的该舌象", "图中的舌象")
        question = question.replace("图中该舌象", "图中舌象")
        question = question.replace("积食积食", "积食")
        question = cls._collapse_repeated_tail(question)
        question = re.sub(r"(变白|变黄|变黑|增厚|脱落)(\1)+", r"\1", question)
        question = re.sub(r"([、，,]){2,}", r"\1", question)
        question = question.strip("，,、：:；; ")

        if cls._looks_like_visual_question(question) and (
            re.search(r"[（(]\d+[)）]", question)
            or cls._is_title_like_visual_subject(raw_caption_clean)
            or "图中的（" in question
            or "图中（" in question
        ):
            question = cls._naturalize_visual_question(sample, question)

        if len(question) > 42:
            if "颜色和状态" in question:
                return cls._naturalize_visual_question(sample, question)
            if "显著" in question and "表现" in question:
                return cls._naturalize_visual_question(sample, question)
            if "最突出的特征" in question:
                return cls._naturalize_visual_question(sample, question)
        return question

    @classmethod
    def _trim_treatment_answer(cls, question: str, answer: str) -> str:
        answer = str(answer or "").strip()
        if not answer or cls._question_asks_treatment(question):
            return cls._ensure_sentence_punct(answer)

        for cue in TREATMENT_ANSWER_CUES:
            idx = answer.find(cue)
            if idx > 0:
                answer = answer[:idx].rstrip("，、；;：:。 ")
                break

        sentences = re.split(r"(?<=[。！？!?])", answer)
        compact = "".join(sentences[:2]).strip()
        answer = compact or answer
        answer = answer.replace("舌头中间部位", "舌中部")
        answer = answer.replace("舌头中间", "舌中部")
        answer = answer.replace("舌头中部", "舌中部")
        answer = answer.replace("舌头前半部分", "舌前半部")
        answer = answer.replace("舌头前边", "舌前部")
        return cls._ensure_sentence_punct(answer)

    @staticmethod
    def _looks_like_visual_question(question: str) -> bool:
        question = (question or "").strip()
        if not question:
            return False
        if any(token in question for token in NON_VISUAL_QUESTION_KEYWORDS):
            return False
        return any(token in question for token in VISUAL_QUESTION_KEYWORDS)

    @staticmethod
    def _visual_question_subject(sample: dict) -> str:
        image_caption = ShezhenVQAProcessor._caption_for_prompt(sample)
        if (
            image_caption
            and any(token in image_caption for token in VISUAL_SUBJECT_CUES)
            and not ShezhenVQAProcessor._is_title_like_visual_subject(image_caption)
        ):
            return image_caption

        section_title = str(sample.get("section_title", "") or "").strip()
        if section_title.startswith("图注章节:"):
            section_title = section_title.replace("图注章节:", "", 1).strip()
        if section_title and section_title not in {"形色舌诊"}:
            return section_title
        return "舌象"

    def _rewrite_visual_question(self, sample: dict, question: str, index: int) -> str:
        if index != 0 or not self._looks_like_visual_question(question):
            return question

        natural_subject = self._natural_visual_anchor(sample)
        subject = self._visual_question_subject(sample)
        generic_subject = subject if subject and subject != "舌象" else ""
        if not generic_subject or self._is_title_like_visual_subject(generic_subject):
            generic_subject = natural_subject

        templates: list[str] = []
        if generic_subject:
            templates.extend(
                [
                    f"观察{generic_subject}，最突出的视觉特征是什么？",
                    f"请描述{generic_subject}的颜色和状态。",
                    f"观察{generic_subject}，有哪些显著的舌象表现？",
                    f"观察{generic_subject}，请描述其舌色、舌体和舌苔特征。",
                    f"观察{generic_subject}，有哪些值得注意的外观特征？",
                ]
            )

        templates.extend(
            [
                "请描述图中舌象的颜色和状态。",
                "观察图中的舌象，有哪些显著表现？",
                "图中的舌象最突出的特征是什么？",
                "观察图中的舌象，请描述其舌色、舌质和舌苔特征。",
                "观察图中的舌象，有哪些值得注意的外观特征？",
            ]
        )

        candidates = [item.strip() for item in templates if item and item.strip()]
        if not candidates:
            return question

        rng = random.Random(
            f"{sample.get('book_name', '')}|{sample.get('image_path', '')}|{sample.get('image_caption', '')}|{question}"
        )
        rewritten = rng.choice(candidates)
        return rewritten if rewritten else question

    def _postprocess_qa_pairs(self, sample: dict, qa_pairs: list[dict[str, str]]) -> list[dict[str, str]]:
        processed: list[dict[str, str]] = []
        for index, qa in enumerate(qa_pairs):
            question = self._rewrite_visual_question(sample, qa.get("question", ""), index)
            question = self._sanitize_question(sample, question)
            answer = self._trim_treatment_answer(question, qa.get("answer", ""))
            processed.append(
                {
                    "question": question,
                    "answer": answer,
                }
            )
        return processed

    def _load_samples(self, book_name: str, limit: int | None, sample_rate: float, seed: int) -> list[dict]:
        sample_file = SPLIT_SAMPLES_DIR / f"{book_name}_samples.jsonl"
        if not sample_file.exists():
            raise FileNotFoundError(f"未找到样本文件: {sample_file}")
        rng = random.Random(seed)
        items: list[dict] = []
        skipped_noise_sections = 0
        with sample_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if self.drop_noise_sections:
                    _, is_noise = self._normalize_section_title(
                        str(row.get("section_title", "") or ""),
                        str(row.get("image_caption", "") or ""),
                    )
                    if is_noise:
                        skipped_noise_sections += 1
                        continue
                if sample_rate < 1.0 and rng.random() > sample_rate:
                    continue
                items.append(row)
                if limit and len(items) >= limit:
                    break
        if self.drop_noise_sections:
            print(f"[过滤] {book_name}: 跳过噪声章节样本 {skipped_noise_sections} 条")
        return items

    def _build_user_prompt(self, sample: dict) -> str:
        context_text = str(sample.get("context_text", "") or "").strip()
        if len(context_text) > self.max_context_chars:
            context_text = context_text[: self.max_context_chars] + "..."
        # anchor_sentence：切分时提取的含图标引用的原句；若无则回退到 context_text 首句
        anchor_sentence = str(sample.get("anchor_sentence", "") or "").strip()
        if not anchor_sentence:
            # 回退：取 context_text 中含 image_caption 编号的第一句
            cap = str(sample.get("image_caption", "") or "")
            for sent in context_text.replace("。", "。\n").splitlines():
                if cap.replace("图", "") in sent:
                    anchor_sentence = sent.strip()
                    break
            if not anchor_sentence:
                anchor_sentence = context_text[:200]  # 最终 fallback
        sample_hint = "普通图文样本"
        if self._is_sparse_visual_sample(sample):
            sample_hint = "短图注图谱样本（允许保守生成 1-2 个 QA，优先依据题注和可见舌象）"
        section_title_clean, _ = ShezhenVQAProcessor._normalize_section_title(
            str(sample.get("section_title", "") or ""),
            str(sample.get("image_caption", "") or ""),
        )
        return USER_PROMPT_TEMPLATE.format(
            book_name=sample.get("book_name", ""),
            section_title=section_title_clean,
            image_path=sample.get("image_path", ""),
            image_caption=self._caption_for_prompt(sample),
            sample_hint=sample_hint,
            anchor_sentence=anchor_sentence,
            context_text=context_text,
        )

    def process_book(
        self,
        book_name: str,
        limit: int | None,
        sample_rate: float,
        seed: int,
        output_tag: str = "",
    ) -> Path:
        samples = self._load_samples(book_name, limit=limit, sample_rate=sample_rate, seed=seed)
        if not samples:
            raise ValueError(f"{book_name} 没有可处理样本")

        user_inputs = [self._build_user_prompt(s) for s in samples]
        print(f"[加载] {book_name}: 样本 {len(samples)} 条")
        system_prompt = (
            SHEZHEN_VQA_SYSTEM_PROMPT_STRICT
            if self.mode == "strict"
            else SHEZHEN_VQA_SYSTEM_PROMPT_ENRICHED
        )
        responses = self.llm.generate_from_input(
            user_inputs=user_inputs,
            system_prompt=system_prompt,
            json_schema=QA_JSON_SCHEMA,
        )

        safe_tag = re.sub(r"[^\w\-]+", "_", output_tag or "").strip("_")
        suffix = f"_{safe_tag}" if safe_tag else ""
        out_file = self.output_dir / f"{book_name}_vqa{suffix}.jsonl"
        debug_file = self.output_dir / f"{book_name}_raw_debug{suffix}.txt"
        written = 0
        empty_after_clean = 0
        no_qa = 0

        with debug_file.open("w", encoding="utf-8") as dbg, out_file.open("w", encoding="utf-8") as out:
            for sample, resp in tqdm(zip(samples, responses), total=len(samples), desc=f"解析 {book_name}"):
                raw = resp or ""
                cleaned = self._clean_response(raw)

                dbg.write("# RAW\n")
                dbg.write(raw + "\n")
                dbg.write("# CLEANED\n")
                dbg.write(cleaned + "\n\n")

                if not cleaned:
                    empty_after_clean += 1
                    continue

                qa_pairs = self._extract_qa_pairs(cleaned)
                if not qa_pairs:
                    no_qa += 1
                    continue
                qa_pairs = self._postprocess_qa_pairs(sample, qa_pairs)

                section_title_raw = str(sample.get("section_title", "") or "")
                section_title_clean, section_noise = self._normalize_section_title(
                    section_title_raw,
                    str(sample.get("image_caption", "") or ""),
                )
                for qa in qa_pairs:
                    record = {
                        "book_name": sample.get("book_name", ""),
                        "section_title": section_title_clean,
                        "section_title_raw": section_title_raw,
                        "section_title_is_noise": section_noise,
                        "image_path": sample.get("image_path", ""),
                        "image_caption": self._caption_for_prompt(sample),
                        "context_text": sample.get("context_text", ""),
                        "question": qa["question"],
                        "answer": qa["answer"],
                        "generation_mode": self.mode,
                    }
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    written += 1

        print(
            f"[完成] {book_name}: 写入 {written} 条 QA -> {out_file} "
            f"(清洗后为空 {empty_after_clean}, 无 qa_pair {no_qa})"
        )
        print(f"[调试] 原始响应日志: {debug_file}")
        return out_file


def _discover_books() -> list[str]:
    return sorted(p.stem.replace("_samples", "") for p in SPLIT_SAMPLES_DIR.glob("*_samples.jsonl"))


def main() -> None:
    parser = argparse.ArgumentParser(description="基于 split_samples 生成舌诊 VQA（简化版）")
    parser.add_argument("--book", type=str, help="指定书名（如 形色舌诊_阎金海）")
    parser.add_argument("--all", action="store_true", help="处理 split_samples 下全部书籍")
    parser.add_argument("--limit", type=int, default=None, help="每本书最多处理样本数")
    parser.add_argument("--sample-rate", type=float, default=1.0, help="随机抽样比例 (0,1]")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--model", type=str, default="MiniMax-M2.5-highspeed", help="模型名")
    parser.add_argument("--max-workers", type=int, default=5, help="并发请求数")
    parser.add_argument("--temperature", type=float, default=0.1, help="采样温度")
    parser.add_argument("--mode", type=str, choices=["strict", "enriched"], default="enriched",
                        help="strict=仅依赖原文；enriched=允许中医常识补全")
    parser.add_argument("--drop-noise-sections", action="store_true",
                        help="过滤目录/CIP/版权等噪声章节样本")
    parser.add_argument("--max-context-chars", type=int, default=1500,
                        help="每条样本保留的上下文最大长度")
    parser.add_argument("--output-tag", type=str, default="",
                        help="输出文件后缀标签（用于 strict/enriched 对比，避免覆盖）")
    parser.add_argument("--output-subdir", type=str, default="",
                        help="输出到 shezhen_vqa_workdir 下的新子目录，例如 rerun_20260402")
    parser.add_argument("--quality-gate", action="store_true",
                        help="开启后在生成原始 VQA 文件后，自动运行分层质控并产出 quality_ready")
    parser.add_argument("--quality-min-context-gold", type=int, default=20,
                        help="quality gate 分层时 gold 需要的最小 context 长度")
    parser.add_argument("--quality-allow-noise-gold", action="store_true",
                        help="quality gate 时允许 section_title_is_noise 进入 gold")
    parser.add_argument("--quality-no-dedupe", action="store_true",
                        help="quality gate 时关闭去重")
    parser.add_argument("--quality-keep-risk-short-context", action="store_true",
                        help="quality gate 时保留高风险短上下文 QA（默认会丢弃）")
    parser.add_argument("--quality-drop-noise", dest="quality_drop_noise", action="store_true", default=True,
                        help="quality gate 时默认剔除 section_title_is_noise=true 的样本")
    parser.add_argument("--quality-keep-noise", dest="quality_drop_noise", action="store_false",
                        help="quality gate 时保留 section_title_is_noise=true 的样本")
    parser.add_argument("--export-senior-format", action="store_true",
                        help="导出师兄同款字段版（id/label/chapter_title/question/answer/image_path/source）")
    parser.add_argument("--senior-source-name", type=str, default="",
                        help="导出师兄同款字段版时，覆盖 source 字段")
    parser.add_argument("--skip-existing", action="store_true",
                        help="断点续跑：已存在输出文件时自动跳过该书")
    args = parser.parse_args()

    if not args.book and not args.all:
        parser.print_help()
        sys.exit(0)

    if args.sample_rate <= 0 or args.sample_rate > 1:
        raise ValueError("--sample-rate 必须在 (0,1] 范围内")

    output_dir = VQA_OUTPUT_DIR
    if args.output_subdir.strip():
        safe_subdir = re.sub(r"[^\w\-.]+", "_", args.output_subdir.strip()).strip("._")
        if not safe_subdir:
            raise ValueError("--output-subdir 经过清洗后为空，请换一个名字")
        output_dir = PROJECT_ROOT / "shezhen_vqa_workdir" / safe_subdir

    processor = ShezhenVQAProcessor(
        model_name=args.model,
        max_workers=args.max_workers,
        temperature=args.temperature,
        mode=args.mode,
        drop_noise_sections=args.drop_noise_sections,
        max_context_chars=args.max_context_chars,
        output_dir=output_dir,
    )

    if args.book:
        targets = [args.book]
    else:
        targets = _discover_books()
        if not targets:
            raise FileNotFoundError(f"{SPLIT_SAMPLES_DIR} 下未发现 *_samples.jsonl")

    for book_name in targets:
        try:
            safe_tag = re.sub(r"[^\w\-]+", "_", args.output_tag or "").strip("_")
            suffix = f"_{safe_tag}" if safe_tag else ""
            out_file_expected = output_dir / f"{book_name}_vqa{suffix}.jsonl"
            quality_ready_expected = out_file_expected.with_name(f"{out_file_expected.stem}_quality_ready.jsonl")
            senior_compat_from_vqa = _senior_compat_output_path(out_file_expected)
            senior_compat_from_quality = _senior_compat_output_path(quality_ready_expected)

            if args.skip_existing:
                if args.quality_gate and quality_ready_expected.exists():
                    if args.export_senior_format and (not senior_compat_from_quality.exists()):
                        export_path, export_count = _export_senior_compat_file(
                            quality_ready_expected,
                            source_name=args.senior_source_name,
                        )
                        print(f"[resume-export] {book_name}: rows={export_count} -> {export_path}")
                    else:
                        print(f"[skip] {book_name}: quality_ready 已存在 -> {quality_ready_expected}")
                    continue
                if (not args.quality_gate) and out_file_expected.exists():
                    if args.export_senior_format and (not senior_compat_from_vqa.exists()):
                        export_path, export_count = _export_senior_compat_file(
                            out_file_expected,
                            source_name=args.senior_source_name,
                        )
                        print(f"[resume-export] {book_name}: rows={export_count} -> {export_path}")
                    else:
                        print(f"[skip] {book_name}: vqa 输出已存在 -> {out_file_expected}")
                    continue
                if args.quality_gate and out_file_expected.exists() and (not quality_ready_expected.exists()):
                    quality_ready_path, quality_summary = _emit_quality_ready_file(
                        input_path=out_file_expected,
                        min_context_gold=args.quality_min_context_gold,
                        allow_noise_gold=args.quality_allow_noise_gold,
                        dedupe=(not args.quality_no_dedupe),
                        drop_risk_short_context=(not args.quality_keep_risk_short_context),
                        drop_noise_sections=args.quality_drop_noise,
                    )
                    print(
                        f"[resume-quality] {book_name}: kept={quality_summary['kept']} "
                        f"drop_noise={quality_summary['dropped_noise_sections']} "
                        f"drop_reject={quality_summary['dropped_reject']} "
                        f"drop_risk={quality_summary['dropped_risk_short_context']} -> {quality_ready_path}"
                    )
                    if args.export_senior_format:
                        export_path, export_count = _export_senior_compat_file(
                            quality_ready_path,
                            source_name=args.senior_source_name,
                        )
                        print(f"[export] {book_name}: rows={export_count} -> {export_path}")
                    continue

            out_file = processor.process_book(
                book_name=book_name,
                limit=args.limit,
                sample_rate=args.sample_rate,
                seed=args.seed,
                output_tag=args.output_tag,
            )

            export_input_path = out_file
            if args.quality_gate:
                quality_ready_path, quality_summary = _emit_quality_ready_file(
                    input_path=out_file,
                    min_context_gold=args.quality_min_context_gold,
                    allow_noise_gold=args.quality_allow_noise_gold,
                    dedupe=(not args.quality_no_dedupe),
                    drop_risk_short_context=(not args.quality_keep_risk_short_context),
                    drop_noise_sections=args.quality_drop_noise,
                )
                print(
                    f"[quality] {book_name}: kept={quality_summary['kept']} "
                    f"drop_noise={quality_summary['dropped_noise_sections']} "
                    f"drop_reject={quality_summary['dropped_reject']} "
                    f"drop_risk={quality_summary['dropped_risk_short_context']} -> {quality_ready_path}"
                )
                export_input_path = quality_ready_path

            if args.export_senior_format:
                export_path, export_count = _export_senior_compat_file(
                    export_input_path,
                    source_name=args.senior_source_name,
                )
                print(f"[export] {book_name}: rows={export_count} -> {export_path}")
        except Exception as exc:
            print(f"[异常] {book_name}: {exc}")


if __name__ == "__main__":
    main()
