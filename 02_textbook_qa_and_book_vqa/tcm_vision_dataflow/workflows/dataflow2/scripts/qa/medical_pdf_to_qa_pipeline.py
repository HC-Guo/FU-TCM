from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pandas as pd

from dataflow.operators.core_text import FormatStrPromptedGenerator, PandasOperator
from dataflow.operators.knowledge_cleaning import (
    FileOrURLToMarkdownConverterLocal,
    KBCChunkGenerator,
)
from dataflow.prompts.core_text import FormatStrPrompt
from dataflow.serving import APILLMServing_request
from dataflow.utils.storage import FileStorage


MEDICAL_QA_PROMPT_TEMPLATE = """
你是一位跨学科的资深医学专家（精通中医临床、中药方剂、现代药理学）。请阅读以下来源于《{book_name}》的文本片段，提取用于训练医疗大模型的高质量问答对。

【智能抽取策略】（请根据书名和内容自动调整侧重点）：
1. 若内容偏向【中医基础与临床】（如内/外/妇/儿科、基础理论、诊断、中西医结合）：
   重点抽取病因病机、辨证论治、症状体征（望闻问切）、辨证分型要点、治则治法、代表方剂与加减。
2. 若内容偏向【中药与方剂】（如中药学、方剂学、中药炮制学、药典一部）：
   重点抽取药物基原、性味归经、功效主治、方剂组成（君臣佐使）、配伍意义、炮制方法及其对药性的改变。
3. 若内容偏向【现代医学与药典】（如药理学、药典二/三/四部）：
   重点抽取理化性质、药理作用机制（靶点/受体）、临床适应症、不良反应、禁忌症、药物相互作用、检验标准。

【通用红线规则】（违反即视为失败）：
- 严格过滤：若文本是版权页、前言、目录、纯人员名单或无实质医学知识，必须返回空数组，不生成任何问答。
- 绝对忠于原文：答案必须 100% 来自提供的片段，禁止用预训练知识脑补机制、副作用或数据。
- 指代明确：问答中严禁出现"本章"、"本书"、"本品"、"本方"、"该药"、"笔者"、"上文"等词，必须替换为具体的疾病名、中药名、化学药名或方剂名。
- 问答自包含：每条问答必须独立完整，不依赖上下文即可理解。
- 每段文本最多输出 6 组高信息密度的问答，宁缺毋滥。

【输出格式】：
严格输出 JSON，必须符合以下结构，不含任何多余说明：
{{
  "qa_pairs": [
    {{
      "question": "具体的问题（不含模糊指代）",
      "answer": "严格来自原文的答案",
      "evidence": "支持答案的原句摘录"
    }}
  ]
}}
若无可提取的医学知识，返回：{{"qa_pairs": []}}

【原文片段】：
{raw_chunk}
""".strip()


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
                    "evidence": {"type": "string"},
                },
                "required": ["question", "answer", "evidence"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["qa_pairs"],
    "additionalProperties": False,
}


def _extract_json_from_response(raw: str) -> dict:
    """从 LLM 响应中提取 JSON，兼容 <think> 标签和 ```json 代码块包裹。"""
    # 1. 去掉 <think>...</think> 推理过程
    raw = re.sub(r"<think>[\s\S]*?</think>", "", raw, flags=re.IGNORECASE).strip()
    # 2. 提取 ```json ... ``` 代码块
    code_block = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
    if code_block:
        raw = code_block.group(1).strip()
    # 3. 直接解析
    return json.loads(raw)


def parse_generated_qa_pairs(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []

    for _, row in df.iterrows():
        raw_payload = row.get("generated_content")
        if not raw_payload or not isinstance(raw_payload, str):
            continue

        try:
            payload = _extract_json_from_response(raw_payload)
        except (json.JSONDecodeError, Exception):
            continue

        qa_pairs = payload.get("qa_pairs", [])
        if not isinstance(qa_pairs, list):
            continue

        for qa in qa_pairs:
            if not isinstance(qa, dict):
                continue

            question = str(qa.get("question", "")).strip()
            answer = str(qa.get("answer", "")).strip()
            evidence = str(qa.get("evidence", "")).strip()

            if not question or not answer:
                continue

            rows.append(
                {
                    "source": row.get("book_name", ""),
                    "pdf_path": row.get("source", ""),
                    "chunk_id": row.get("chunk_id", 0),
                    "question": question,
                    "answer": answer,
                    "evidence": evidence,
                    "raw_chunk": row.get("raw_chunk", ""),
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "id",
                "source",
                "pdf_path",
                "chunk_id",
                "question",
                "answer",
                "evidence",
                "raw_chunk",
            ]
        )

    result = pd.DataFrame(rows)
    result = result.drop_duplicates(subset=["source", "question", "answer"]).reset_index(drop=True)
    result.insert(0, "id", range(1, len(result) + 1))
    return result


def add_chunk_index(df: pd.DataFrame) -> pd.DataFrame:
    indexed = df.copy()
    indexed["chunk_id"] = indexed.groupby("source").cumcount() + 1
    return indexed


# 匹配实质性正文章节标题的模式
# 优先级从高到低依次尝试，命中第一个即截断
import re

_BODY_START_PATTERNS = [
    # 「第X章」「第X节」（汉字数字或阿拉伯数字）
    re.compile(r"^#{1,3}\s*第[一二三四五六七八九十百千\d]+[章节篇]", re.MULTILINE),
    # 「一、」「（一）」作为顶级列表项开头的正文段落
    re.compile(r"^#{1,3}\s*[一二三四五六七八九十]+[、．.]", re.MULTILINE),
    # 纯数字编号标题「# 1.」「## 1 」
    re.compile(r"^#{1,3}\s*\d+[.、\s]", re.MULTILINE),
    # 「绪论」「总论」「前言」（正文起点）
    re.compile(r"^#{1,3}\s*(?:绪论|总论|前言|概论|概述|引言)", re.MULTILINE),
]


def strip_markdown_preamble(md_path: str) -> str:
    """
    读取 Markdown 文件，裁掉版权页/编委/目录等前置噪声，
    返回从第一个实质性章节标题开始的正文路径（原路径，内容已被覆写）。
    若未检测到任何章节标题，则原样返回不做处理。
    """
    path = Path(md_path)
    if not path.exists() or not md_path:
        return md_path

    text = path.read_text(encoding="utf-8", errors="replace")

    cut_pos = None
    for pattern in _BODY_START_PATTERNS:
        m = pattern.search(text)
        if m:
            if cut_pos is None or m.start() < cut_pos:
                cut_pos = m.start()

    if cut_pos is None or cut_pos == 0:
        # 未检测到噪声，或正文就在开头，不做处理
        return md_path

    stripped = text[cut_pos:]
    path.write_text(stripped, encoding="utf-8")
    return md_path


def apply_strip_preamble(df: pd.DataFrame) -> pd.DataFrame:
    """PandasOperator 包装：对每行的 text_path 执行前置噪声裁剪。"""
    result = df.copy()
    result["text_path"] = result["text_path"].apply(
        lambda p: strip_markdown_preamble(str(p)) if pd.notna(p) and str(p) else p
    )
    return result


def filter_noise_chunks(df: pd.DataFrame) -> pd.DataFrame:
    """过滤掉无实质内容的 chunk（纯标点、省略号、空白、目录行、过短文本等）。"""
    def is_valid(text) -> bool:
        if not isinstance(text, str):
            return False
        stripped = text.strip()
        # 过短
        if len(stripped) < 20:
            return False
        # 去掉所有标点、空白后，实际有意义字符数量要足够
        import unicodedata
        meaningful = [c for c in stripped if unicodedata.category(c) not in
                      ('Po', 'Ps', 'Pe', 'Pi', 'Pf', 'Pd', 'Zs', 'Cc', 'Cf') and not c.isspace()]
        if len(meaningful) < 10:
            return False
        # 目录格式检测：大量「汉字+空格+数字」模式（如 "麻黄汤 26 桂枝汤 28"）
        # 用正则统计「词 数字」对出现次数，占比过高则判定为目录
        toc_matches = re.findall(r'[\u4e00-\u9fa5]{1,20}\s+\d{1,4}', stripped)
        if len(toc_matches) >= 5 and len(toc_matches) * 8 > len(stripped):
            return False
        return True

    before = len(df)
    result = df[df["raw_chunk"].apply(is_valid)].reset_index(drop=True)
    after = len(result)
    if before - after > 0:
        from dataflow import get_logger
        get_logger().info(f"[filter_noise_chunks] 过滤掉 {before - after} 个噪声 chunk，剩余 {after} 个")
    return result


class MedicalBooksQAPipeline:
    def __init__(self):
        self.project_root = Path(__file__).resolve().parent
        self.bundle_root = self.project_root.parents[4]
        self.source_dir = self._resolve_source_dir()

        self.work_dir = self.project_root / "medical_books_qa_workdir"
        self.cache_dir = self.work_dir / "cache"
        self.markdown_dir = self.work_dir / "markdown"
        self.output_dir = self.project_root / "output"
        self.manifest_path = self.work_dir / "medical_books_input.jsonl"
        self.final_output_path = self.output_dir / "medical_books_qa_pairs.jsonl"

        for path in [self.work_dir, self.cache_dir, self.markdown_dir, self.output_dir]:
            path.mkdir(parents=True, exist_ok=True)

        self._create_manifest()
        api_url, model_name = self._load_minimax_config()

        self.storage = FileStorage(
            first_entry_file_name=str(self.manifest_path),
            cache_path=str(self.cache_dir),
            file_name_prefix="medical_books_qa",
            cache_type="jsonl",
        )

        self.pdf_to_markdown = FileOrURLToMarkdownConverterLocal(
            intermediate_dir=str(self.markdown_dir),
            mineru_backend=os.getenv("MEDICAL_MINERU_BACKEND", "pipeline"),
        )
        self.chunker = KBCChunkGenerator(
            chunk_size=int(os.getenv("MEDICAL_CHUNK_SIZE", "900")),
            chunk_overlap=int(os.getenv("MEDICAL_CHUNK_OVERLAP", "120")),
            split_method=os.getenv("MEDICAL_CHUNK_METHOD", "recursive"),
            tokenizer_name=os.getenv("MEDICAL_TOKENIZER", "bert-base-chinese"),
        )
        self.chunk_indexer = PandasOperator(process_fn=[add_chunk_index])
        self.preamble_stripper = PandasOperator(process_fn=[apply_strip_preamble])
        self.noise_filter = PandasOperator(process_fn=[filter_noise_chunks])

        self.llm_serving = APILLMServing_request(
            api_url=api_url,
            model_name=model_name,
            key_name_of_api_key="DF_API_KEY",
            max_workers=int(os.getenv("MEDICAL_QA_MAX_WORKERS", "8")),
            temperature=float(os.getenv("MEDICAL_QA_TEMPERATURE", "0.1")),
            read_timeout=float(os.getenv("MEDICAL_QA_READ_TIMEOUT", "180")),
        )
        self.qa_generator = FormatStrPromptedGenerator(
            llm_serving=self.llm_serving,
            system_prompt="你是严谨的医学教材问答抽取助手，只能依据给定文本输出结果。",
            prompt_template=FormatStrPrompt(MEDICAL_QA_PROMPT_TEMPLATE),
            json_schema=QA_JSON_SCHEMA,
        )
        self.qa_parser = PandasOperator(process_fn=[parse_generated_qa_pairs])

    def _resolve_source_dir(self) -> Path:
        configured = os.getenv("MEDICAL_BOOKS_DIR")
        if configured:
            source_dir = Path(configured)
        else:
            source_dir = self.bundle_root / "source_books"

        if not source_dir.exists():
            raise FileNotFoundError(
                f"未找到医学书籍目录：{source_dir}。请设置环境变量 MEDICAL_BOOKS_DIR 指向 PDF 所在目录。"
            )
        return source_dir

    def _load_minimax_config(self) -> tuple[str, str]:
        api_key = os.getenv("DF_API_KEY") or os.getenv("MINIMAX_API_KEY")
        if not api_key:
            raise ValueError("请设置 DF_API_KEY 或 MINIMAX_API_KEY。")
        os.environ.setdefault("DF_API_KEY", api_key)

        api_url = os.getenv("MINIMAX_API_URL")
        model_name = os.getenv("MINIMAX_MODEL")
        if not api_url or not model_name:
            raise ValueError("请设置 MINIMAX_API_URL 和 MINIMAX_MODEL。")
        return api_url, model_name

    def _create_manifest(self) -> None:
        pdf_files = sorted(self.source_dir.glob("*.pdf"))
        if not pdf_files:
            raise FileNotFoundError(f"目录中没有找到 PDF 文件：{self.source_dir}")

        with self.manifest_path.open("w", encoding="utf-8") as f:
            for pdf_path in pdf_files:
                row = {
                    "source": str(pdf_path),
                    "book_name": pdf_path.stem,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def forward(self) -> Path:
        from dataflow import get_logger
        logger = get_logger()

        def _skip_if_cached(step_num: int) -> bool:
            """如果该步骤的缓存文件已存在则跳过，返回 True 表示跳过。"""
            cache_file = self.cache_dir / f"medical_books_qa_step{step_num}.jsonl"
            if cache_file.exists():
                logger.info(f"[跳过 step{step_num}] 缓存已存在: {cache_file}")
                self.storage.operator_step = step_num
                return True
            return False

        # step1: PDF -> Markdown
        if not _skip_if_cached(1):
            self.pdf_to_markdown.run(
                storage=self.storage.step(),
                input_key="source",
                output_key="text_path",
            )

        # step2: 裁掉版权页/编委名单/目录等前置噪声
        if not _skip_if_cached(2):
            self.preamble_stripper.run(storage=self.storage.step())

        # step3: 过滤 text_path 不存在的行（如药典二/三/四部尚未转换完成）
        if not _skip_if_cached(3):
            def filter_missing_md(df: pd.DataFrame) -> pd.DataFrame:
                before = len(df)
                result = df[df["text_path"].apply(
                    lambda p: bool(p) and Path(str(p)).exists()
                )].reset_index(drop=True)
                skipped = before - len(result)
                if skipped:
                    get_logger().warning(f"[filter_missing_md] 跳过 {skipped} 本 Markdown 不存在的书")
                return result
            md_filter = PandasOperator(process_fn=[filter_missing_md])
            md_filter.run(storage=self.storage.step())

        # step4: Chunking
        if not _skip_if_cached(4):
            self.chunker.run(
                storage=self.storage.step(),
                input_key="text_path",
                output_key="raw_chunk",
            )

        # step5: 过滤纯噪声 chunk
        if not _skip_if_cached(5):
            self.noise_filter.run(storage=self.storage.step())

        # step6: chunk 编号
        if not _skip_if_cached(6):
            self.chunk_indexer.run(storage=self.storage.step())

        # step7: QA 生成（按书名分批，支持单书级断点续跑）
        step7_cache = self.cache_dir / "medical_books_qa_step7.jsonl"
        if step7_cache.exists():
            logger.info(f"[跳过 step7] 缓存已存在: {step7_cache}")
            self.storage.operator_step = 7
        else:
            # 读取 step6 的分块数据
            step6_path = self.cache_dir / "medical_books_qa_step6.jsonl"
            df_chunks = pd.read_json(str(step6_path), lines=True)
            book_names = df_chunks["book_name"].unique().tolist()
            logger.info(f"共 {len(book_names)} 本书待生成 QA，支持单书级断点续跑")

            all_gen_rows = []
            for book_name in book_names:
                book_cache = self.cache_dir / f"qa_gen_{book_name}.jsonl"
                if book_cache.exists():
                    logger.info(f"[跳过] {book_name} 已生成，读取缓存")
                    df_book = pd.read_json(str(book_cache), lines=True)
                    all_gen_rows.append(df_book)
                    continue

                logger.info(f"[生成] {book_name} 开始 QA 生成...")
                df_book_chunks = df_chunks[df_chunks["book_name"] == book_name].reset_index(drop=True)

                # 为这本书创建临时 storage
                book_tmp_path = self.cache_dir / f"qa_gen_{book_name}_input.jsonl"
                df_book_chunks.to_json(str(book_tmp_path), orient="records", lines=True, force_ascii=False)

                book_storage = FileStorage(
                    first_entry_file_name=str(book_tmp_path),
                    cache_path=str(self.cache_dir),
                    file_name_prefix=f"qa_gen_{book_name}_tmp",
                    cache_type="jsonl",
                )
                try:
                    self.qa_generator.run(
                        storage=book_storage.step(),
                        output_key="generated_content",
                        book_name="book_name",
                        raw_chunk="raw_chunk",
                    )
                    # 读取生成结果并保存单书缓存
                    gen_result_path = self.cache_dir / f"qa_gen_{book_name}_tmp_step1.jsonl"
                    df_gen = pd.read_json(str(gen_result_path), lines=True)
                    df_gen.to_json(str(book_cache), orient="records", lines=True, force_ascii=False)
                    all_gen_rows.append(df_gen)
                    logger.info(f"[完成] {book_name} QA 生成完毕，已缓存")
                except Exception as e:
                    logger.error(f"[失败] {book_name} QA 生成出错: {e}")
                    logger.error("已完成的书已缓存，重新运行可从此书开始续跑")
                    raise

            # 合并所有书的生成结果，写入 step7
            df_all_gen = pd.concat(all_gen_rows, ignore_index=True)
            df_all_gen.to_json(str(step7_cache), orient="records", lines=True, force_ascii=False)
            logger.info(f"[step7 完成] 所有书 QA 生成合并写入: {step7_cache}")
            self.storage.operator_step = 7

        # step8: 解析 QA 对
        if not _skip_if_cached(8):
            self.storage.operator_step = 7
            self.qa_parser.run(storage=self.storage.step())

        final_cache_path = self.storage._get_cache_file_path(8)
        shutil.copy2(final_cache_path, self.final_output_path)
        return self.final_output_path


if __name__ == "__main__":
    pipeline = MedicalBooksQAPipeline()
    output_path = pipeline.forward()
    print(f"QA 提取完成，输出文件：{output_path}")

