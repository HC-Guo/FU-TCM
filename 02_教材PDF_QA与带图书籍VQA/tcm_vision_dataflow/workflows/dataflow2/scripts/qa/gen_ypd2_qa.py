"""
单独为药典二部生成 QA 对，复用 medical_pdf_to_qa_pipeline.py 的 LLM 配置。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from medical_pdf_to_qa_pipeline import (
    MEDICAL_QA_PROMPT_TEMPLATE,
    QA_JSON_SCHEMA,
    parse_generated_qa_pairs,
    add_chunk_index,
    filter_noise_chunks,
)
from dataflow.operators.core_text import FormatStrPromptedGenerator, PandasOperator
from dataflow.operators.knowledge_cleaning import KBCChunkGenerator
from dataflow.prompts.core_text import FormatStrPrompt
from dataflow.serving import APILLMServing_request
from dataflow.utils.storage import FileStorage

BASE = Path(r"D:\Desktop\Dataflow\DataFlow-main")
CACHE_DIR = BASE / "medical_books_qa_workdir" / "cache"
MARKDOWN_DIR = BASE / "medical_books_qa_workdir" / "markdown"

TARGET_BOOKS = [
    {
        "book_name": "中华人民共和国药典2020年版二部",
        "text_path": str(MARKDOWN_DIR / "中华人民共和国药典2020年版二部" / "auto" / "中华人民共和国药典2020年版二部.md"),
        "source": "中华人民共和国药典2020年版二部",
    },
]


def main():
    from dataflow import get_logger
    logger = get_logger()

    api_key = os.getenv("DF_API_KEY") or os.getenv("MINIMAX_API_KEY")
    if not api_key:
        raise ValueError("请设置 DF_API_KEY 或 MINIMAX_API_KEY")
    os.environ.setdefault("DF_API_KEY", api_key)
    api_url = os.getenv("MINIMAX_API_URL")
    model_name = os.getenv("MINIMAX_MODEL")
    if not api_url or not model_name:
        raise ValueError("请设置 MINIMAX_API_URL 和 MINIMAX_MODEL")

    llm_serving = APILLMServing_request(
        api_url=api_url,
        model_name=model_name,
        key_name_of_api_key="DF_API_KEY",
        max_workers=int(os.getenv("MEDICAL_QA_MAX_WORKERS", "8")),
        temperature=float(os.getenv("MEDICAL_QA_TEMPERATURE", "0.1")),
        read_timeout=float(os.getenv("MEDICAL_QA_READ_TIMEOUT", "180")),
    )
    qa_generator = FormatStrPromptedGenerator(
        llm_serving=llm_serving,
        system_prompt="你是严谨的医学教材问答抽取助手，只能依据给定文本输出结果。",
        prompt_template=FormatStrPrompt(MEDICAL_QA_PROMPT_TEMPLATE),
        json_schema=QA_JSON_SCHEMA,
    )
    chunker = KBCChunkGenerator(
        chunk_size=int(os.getenv("MEDICAL_CHUNK_SIZE", "900")),
        chunk_overlap=int(os.getenv("MEDICAL_CHUNK_OVERLAP", "120")),
        split_method=os.getenv("MEDICAL_CHUNK_METHOD", "recursive"),
        tokenizer_name=os.getenv("MEDICAL_TOKENIZER", "bert-base-chinese"),
    )
    noise_filter = PandasOperator(process_fn=[filter_noise_chunks])
    chunk_indexer = PandasOperator(process_fn=[add_chunk_index])

    all_qa_rows = []

    for book in TARGET_BOOKS:
        book_name = book["book_name"]
        text_path = book["text_path"]
        final_qa_cache = CACHE_DIR / f"qa_final_{book_name}.jsonl"

        if final_qa_cache.exists():
            logger.info(f"[跳过] {book_name} QA 已完成，读取缓存")
            df_qa = pd.read_json(str(final_qa_cache), lines=True)
            all_qa_rows.append(df_qa)
            continue

        if not Path(text_path).exists():
            logger.error(f"[错误] {book_name} 的 markdown 不存在: {text_path}")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"[开始] {book_name}")

        # 构建输入 manifest
        input_path = CACHE_DIR / f"ypd2_input_{book_name}.jsonl"
        with input_path.open("w", encoding="utf-8") as f:
            f.write(json.dumps(
                {"source": book["source"], "book_name": book_name, "text_path": text_path},
                ensure_ascii=False
            ) + "\n")

        # 分块
        chunk_cache = CACHE_DIR / f"ypd2_chunks_{book_name}.jsonl"
        if not chunk_cache.exists():
            logger.info(f"[分块] {book_name} 开始分块...")
            storage_chunk = FileStorage(
                first_entry_file_name=str(input_path),
                cache_path=str(CACHE_DIR),
                file_name_prefix=f"ypd2_chunk_{book_name}",
                cache_type="jsonl",
            )
            chunker.run(storage=storage_chunk.step(), input_key="text_path", output_key="raw_chunk")
            noise_filter.run(storage=storage_chunk.step())
            chunk_indexer.run(storage=storage_chunk.step())
            chunk_result = CACHE_DIR / f"ypd2_chunk_{book_name}_step3.jsonl"
            df_chunks = pd.read_json(str(chunk_result), lines=True)
            df_chunks.to_json(str(chunk_cache), orient="records", lines=True, force_ascii=False)
            logger.info(f"[分块完成] {book_name}：{len(df_chunks)} 个有效 chunk")
        else:
            df_chunks = pd.read_json(str(chunk_cache), lines=True)
            logger.info(f"[跳过分块] {book_name}：读取缓存 {len(df_chunks)} 个 chunk")

        # QA 生成
        gen_cache = CACHE_DIR / f"qa_gen_{book_name}.jsonl"
        if not gen_cache.exists():
            logger.info(f"[生成] {book_name} 开始 QA 生成（{len(df_chunks)} 个 chunk）...")
            book_tmp_path = CACHE_DIR / f"qa_gen_{book_name}_input.jsonl"
            df_chunks.to_json(str(book_tmp_path), orient="records", lines=True, force_ascii=False)

            book_storage = FileStorage(
                first_entry_file_name=str(book_tmp_path),
                cache_path=str(CACHE_DIR),
                file_name_prefix=f"qa_gen_{book_name}_tmp",
                cache_type="jsonl",
            )
            qa_generator.run(
                storage=book_storage.step(),
                output_key="generated_content",
                book_name="book_name",
                raw_chunk="raw_chunk",
            )
            gen_result_path = CACHE_DIR / f"qa_gen_{book_name}_tmp_step1.jsonl"
            df_gen = pd.read_json(str(gen_result_path), lines=True)
            df_gen.to_json(str(gen_cache), orient="records", lines=True, force_ascii=False)
            logger.info(f"[完成] {book_name} QA 生成完毕，已缓存")
        else:
            df_gen = pd.read_json(str(gen_cache), lines=True)
            logger.info(f"[跳过生成] {book_name}：读取缓存")

        # 解析 QA 对
        logger.info(f"[解析] {book_name} 开始解析 QA 对...")
        df_qa = parse_generated_qa_pairs(df_gen)
        df_qa.to_json(str(final_qa_cache), orient="records", lines=True, force_ascii=False)
        logger.info(f"[完成] {book_name}：解析出 {len(df_qa)} 个 QA 对，已保存到 {final_qa_cache}")
        all_qa_rows.append(df_qa)

    if all_qa_rows:
        df_all = pd.concat(all_qa_rows, ignore_index=True)
        output_path = CACHE_DIR / "ypd2_qa_pairs.jsonl"
        df_all.to_json(str(output_path), orient="records", lines=True, force_ascii=False)
        print(f"\n=== 完成 ===")
        print(f"药典二部共生成 {len(df_all)} 个 QA 对")
        print(f"输出文件：{output_path}")
    else:
        print("没有生成任何 QA 对，请检查 markdown 文件是否存在")


if __name__ == "__main__":
    main()

