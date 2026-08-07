from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
PROMPT_ROOT = PROJECT_ROOT / "maizhen_vqa_workdir" / "prompt_engineering"
UNIFIED_ROOT = PROJECT_ROOT / "maizhen_vqa_workdir" / "output" / "unified_output"
DEFAULT_SAVE_ROOT = PROMPT_ROOT / "debug_runs_mianzhen_regen"
VALID_QA_TYPES = {
    "visual_recognition",
    "visual_grounding",
    "text_grounded_explanation",
    "diagnostic_mapping",
    "visual_feature",
    "visual_mapping",
    "clinical_reasoning",
    "comprehensive_application",
    "western_disease",
    "tcm_pathogenesis",
    "clinical_advice",
    "treatment_prescription",
    "spatial_location",
    "dynamic_fluid",
    "systemic_validation",
    "wangzhen_finding",
    "clinical_analysis",
    "auxiliary_exam",
    "pulse_waveform_feature",
    "pulse_type_identification",
    "clinical_significance",
}

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataflow.serving import APILLMServing_request  # noqa: E402

from maizhen_vqa_workdir.prompt_engineering.debug_render_mianzhen_prompts import (  # noqa: E402
    BOOK_PROFILES_PATH,
    BOOK_RULES_PATH,
    SYSTEM_PROMPT_PATH,
    USER_TEMPLATE_PATH,
    build_prompt,
    load_json,
    load_text,
    read_jsonl,
)


SCHEMA_PATH = PROMPT_ROOT / "configs" / "vqa_output_schema.json"

BOOK_SPECIFIC_PROMPTS: dict[str, tuple[Path, Path]] = {
    "中西医结合望诊启迪": (
        PROMPT_ROOT / "prompts" / "system" / "zhongxi_wangzhen_system.md",
        PROMPT_ROOT / "prompts" / "user" / "zhongxi_wangzhen_user_template.md",
    ),
    "中医脉诊临床图解": (
        PROMPT_ROOT / "prompts" / "system" / "maizhen_clinical_system.md",
        PROMPT_ROOT / "prompts" / "user" / "maizhen_clinical_user_template.md",
    ),
}
AMBIGUOUS_ANSWER_CUES = (
    "需要观察原图",
    "需观察原图",
    "图中标注",
    "请参考图示",
    "请结合图示",
    "无法确定",
    "不能确定",
)
LEFT_RIGHT_QUESTION_CUES = (
    "左侧",
    "右侧",
    "哪一侧",
    "哪侧",
    "左右",
)


def _strip_think_and_fences(text: str) -> str:
    think_end = text.find("</think>")
    if think_end != -1:
        text = text[think_end + len("</think>"):]
    text = re.sub(r"```(?:json)?\s*", "", text)
    return text.strip()


def _fix_inner_quotes(snippet: str) -> str:
    """尝试修复 JSON 字符串值中未转义的 ASCII 双引号，如 即"来盛" → 即“来盛”。"""
    for _ in range(30):
        try:
            json.loads(snippet)
            return snippet
        except json.JSONDecodeError as e:
            pos = e.pos
            if pos is None or pos >= len(snippet):
                break
            quote_pos = None
            for look in range(min(pos, 3)):
                p = pos - 1 - look
                if p >= 0 and snippet[p] == '"':
                    quote_pos = p
                    break
            if quote_pos is None:
                break
            snippet = snippet[:quote_pos] + "“" + snippet[quote_pos + 1 :]
    return snippet


def parse_payload(raw_text: str) -> dict | None:
    text = (raw_text or "").strip()
    if not text:
        return None

    try:
        return json.loads(text)
    except Exception:
        pass

    text = _strip_think_and_fences(text)

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    snippet = text[start : end + 1]
    try:
        return json.loads(snippet)
    except json.JSONDecodeError:
        snippet = _fix_inner_quotes(snippet)
    try:
        return json.loads(snippet)
    except Exception:
        return None


def _is_supported_type(qa_type: str) -> bool:
    return qa_type in VALID_QA_TYPES


def _question_requires_side_info(question: str) -> bool:
    return any(cue in question for cue in LEFT_RIGHT_QUESTION_CUES)


def _answer_is_ambiguous(answer: str) -> bool:
    return any(cue in answer for cue in AMBIGUOUS_ANSWER_CUES)


def _sample_supports_side_info(sample: dict) -> bool:
    support_text = "\n".join(
        [
            str(sample.get("image_caption", "") or ""),
            str(sample.get("context_text", "") or ""),
        ]
    )
    return any(cue in support_text for cue in LEFT_RIGHT_QUESTION_CUES)


def _sanitize_item(item: dict, sample: dict) -> dict | None:
    question = str(item.get("question", "") or "").strip()
    answer = str(item.get("answer", "") or "").strip()
    qa_type = str(item.get("type", "") or "").strip()
    if not question or not answer:
        return None
    if not _is_supported_type(qa_type):
        return None
    if _answer_is_ambiguous(answer):
        return None
    if _question_requires_side_info(question) and not _sample_supports_side_info(sample):
        return None
    return {
        "question": question,
        "answer": answer,
        "type": qa_type,
    }


def filter_payload(payload: dict | None, sample: dict) -> dict | None:
    if not isinstance(payload, dict):
        return None

    qa_pairs = payload.get("qa_pairs")
    if not isinstance(qa_pairs, list):
        return None

    book_name = str(sample.get("book_name", "") or "")
    section_title = str(sample.get("section_title", "") or "")
    if "望诊之钥" in book_name and "病例" in section_title:
        max_qa = 12
    elif "望面诊病图解" in book_name:
        max_qa = 20
    elif "中医望诊彩色图谱" in book_name:
        max_qa = 30
    else:
        max_qa = 4

    kept: list[dict] = []
    for item in qa_pairs:
        if not isinstance(item, dict):
            continue
        normalized = _sanitize_item(item, sample)
        if normalized is None:
            continue
        kept.append(normalized)
        if len(kept) >= max_qa:
            break

    if not kept:
        return None
    return {"qa_pairs": kept}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows),
        encoding="utf-8",
    )


def expand_parsed_rows(parsed_rows: list[dict]) -> list[dict]:
    flat_rows: list[dict] = []
    for row in parsed_rows:
        meta = row.get("meta") or {}
        sample = row.get("sample") or {}
        parsed = row.get("parsed_response") or {}
        qa_pairs = parsed.get("qa_pairs") or []

        for qa in qa_pairs:
            flat_rows.append(
                {
                    "book_name": sample.get("book_name", meta.get("book_name", "")),
                    "section_title": sample.get("section_title", ""),
                    "section_title_raw": sample.get("section_title", ""),
                    "section_title_is_noise": False,
                    "image_path": sample.get("image_path", ""),
                    "image_caption": sample.get("image_caption", ""),
                    "context_text": sample.get("context_text", ""),
                    "question": qa.get("question", ""),
                    "answer": qa.get("answer", ""),
                    "qa_type": qa.get("type", ""),
                    "generation_mode": meta.get("generation_mode", ""),
                }
            )
    return flat_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="基于 unified_output/<书名>/split_samples.jsonl 重新生成并覆盖 VQA")
    parser.add_argument("--book", required=True, help="书名，必须与 unified_output 下目录一致")
    parser.add_argument("--index", type=int, default=0, help="起始样本序号")
    parser.add_argument("--count", type=int, default=0, help="处理样本数，0 表示直到末尾")
    parser.add_argument("--max-workers", type=int, default=3, help="并发数")
    parser.add_argument("--temperature", type=float, default=0.1, help="采样温度")
    parser.add_argument("--model-name", type=str, default="", help="模型名，默认读环境变量 MINIMAX_MODEL")
    parser.add_argument("--save-dir", type=str, default=str(DEFAULT_SAVE_ROOT), help="原始/解析结果保存目录")
    parser.add_argument("--dry-run", action="store_true", help="只检查输入并打印计划，不调用模型")
    args = parser.parse_args()

    book_dir = UNIFIED_ROOT / args.book
    split_path = book_dir / "split_samples.jsonl"
    if not split_path.exists():
        raise FileNotFoundError(f"未找到切分文件: {split_path}")

    samples = read_jsonl(split_path)
    start = max(0, args.index)
    end = len(samples) if args.count <= 0 else min(len(samples), start + args.count)
    selected = samples[start:end]
    if not selected:
        raise ValueError("没有选中任何样本")

    if args.dry_run:
        print(f"[检查] 书名={args.book} 样本数={len(samples)} 计划处理区间=[{start}, {end})")
        print(f"[输入] {split_path}")
        print(f"[输出] {(book_dir / 'vqa_dataset.jsonl')}")
        return

    system_prompt = load_text(SYSTEM_PROMPT_PATH)
    user_template = load_text(USER_TEMPLATE_PATH)
    schema = load_json(SCHEMA_PATH)
    profiles = load_json(BOOK_PROFILES_PATH)
    raw_rules = load_text(BOOK_RULES_PATH)
    profile = profiles.get(args.book, {})

    if args.book in BOOK_SPECIFIC_PROMPTS:
        sys_path, usr_path = BOOK_SPECIFIC_PROMPTS[args.book]
        if sys_path.exists():
            system_prompt = load_text(sys_path)
        if usr_path.exists():
            user_template = load_text(usr_path)

    user_inputs: list[str] = []
    metas: list[dict] = []
    for sample in selected:
        rendered_system, rendered_user, meta = build_prompt(
            sample=sample,
            user_template=user_template,
            system_prompt=system_prompt,
            profile=profile,
            raw_rules=raw_rules,
        )
        user_inputs.append(rendered_user)
        metas.append(meta)

    api_key = os.getenv("DF_API_KEY") or os.getenv("MINIMAX_API_KEY")
    if not api_key:
        raise ValueError("请先设置 DF_API_KEY 或 MINIMAX_API_KEY")
    os.environ.setdefault("DF_API_KEY", api_key)

    llm = APILLMServing_request(
        api_url=os.getenv("MINIMAX_API_URL", "https://api.minimax.chat/v1/chat/completions"),
        model_name=args.model_name or os.getenv("MINIMAX_MODEL", "MiniMax-M2.5-lightning"),
        key_name_of_api_key="DF_API_KEY",
        max_workers=max(1, int(args.max_workers)),
        temperature=float(args.temperature),
        read_timeout=float(os.getenv("MINIMAX_READ_TIMEOUT", "180")),
    )

    responses = llm.generate_from_input(
        user_inputs=user_inputs,
        system_prompt=system_prompt,
        json_schema=schema,
    )

    save_dir = Path(args.save_dir).expanduser() / args.book
    save_dir.mkdir(parents=True, exist_ok=True)
    raw_path = save_dir / f"raw_{start:04d}_{end - 1:04d}.jsonl"
    parsed_path = save_dir / f"parsed_{start:04d}_{end - 1:04d}.jsonl"

    parsed_rows: list[dict] = []
    success = 0
    with raw_path.open("w", encoding="utf-8") as raw_f:
        for offset, (sample, meta, user_prompt, resp) in enumerate(zip(selected, metas, user_inputs, responses)):
            idx = start + offset
            raw_record = {
                "index": idx,
                "meta": meta,
                "sample": sample,
                "user_prompt": user_prompt,
                "raw_response": resp,
            }
            raw_f.write(json.dumps(raw_record, ensure_ascii=False) + "\n")

            payload = filter_payload(parse_payload(resp or ""), sample)
            parsed_record = {
                "index": idx,
                "meta": meta,
                "sample": sample,
                "parsed_response": payload,
            }
            parsed_rows.append(parsed_record)
            if payload is not None:
                success += 1

    write_jsonl(parsed_path, parsed_rows)
    flat_rows = expand_parsed_rows(parsed_rows)
    out_vqa = book_dir / "vqa_dataset.jsonl"
    write_jsonl(out_vqa, flat_rows)

    print(f"[完成] 书名={args.book} 处理样本={len(selected)} 成功解析={success}/{len(selected)}")
    print(f"[输入] {split_path}")
    print(f"[输出] 原始响应: {raw_path}")
    print(f"[输出] 解析结果: {parsed_path}")
    print(f"[输出] QA数据集: {out_vqa}")
    print(f"[统计] QA pair 总数: {len(flat_rows)}")


if __name__ == "__main__":
    main()
