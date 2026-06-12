from dataflow.operators.knowledge_cleaning.generate.mineru_operators import FileOrURLToMarkdownConverterAPI

from dataflow.serving import APILLMServing_request
from dataflow.utils.storage import FileStorage
from dataflow.operators.pdf2vqa import MinerU2LLMInputOperator, LLMOutputParser, QA_Merger, PDF_Merger, VQAFormatter
from dataflow.operators.core_text import ChunkedPromptedGenerator

from dataflow.pipeline import PipelineABC
from dataflow.prompts.pdf2vqa import QAExtractPrompt

import os
    
class PDF_VQA_extract_optimized_pipeline(PipelineABC):
    """
    全 API 模式（无本地推理）需在环境中配置：
    - MINERU_API_KEY：MinerU 云端解析 PDF，见 https://mineru.net/apiManage/token
    - DF_API_KEY（或 DF_LLM_API_KEY_ENV 指向的变量）：兼容 OpenAI 的聊天 API，默认按 MiniMax 文档填写
    - 可选：DF_LLM_READ_TIMEOUT（秒，默认 600）、DF_LLM_CONNECT_TIMEOUT（默认 30）、DF_LLM_MAX_RETRIES（默认 5）
    - 可选：DF_LLM_MAX_TOKENS（默认 32768；MiniMax M2 等若仍 8192，推理易占满额度导致 content 无 <chapter>）。
      Prompt 现为「每图 3～5 题 + source_text」，输出量约为原先 3～5 倍，不足时可再提高该值或缩小 DF_PDF2VQA_MAX_JSON_ITEMS。
    - 可选：DF_PDF2VQA_MAX_CHUNK_LEN（默认 20000，单段「用户 JSON」的 token 上限；与 system prompt 分开计）
    - 可选：DF_PDF2VQA_MAX_JSON_ITEMS（默认 80，顶层为 JSON 数组时每段最多多少条元素；增大可减少 LLM 调用次数，但若触达 max_chunk_len 会在段内再切分。整书建议配合分段覆盖全量）
      若需恢复「整包单次」（不推荐）：设 DF_PDF2VQA_MAX_JSON_ITEMS=0。
      长 JSON 单次推理易超时或漏图，MiniMax 等建议 read 超时 ≥600。
    """

    def __init__(self):
        super().__init__()
        # -------- configurable via env vars (Windows/PowerShell friendly) --------
        # Input list
        input_jsonl = os.getenv(
            "DF_PDF2VQA_INPUT_JSONL",
            "./dataflow/example/PDF2VQAPipeline/vqa_extract_tongue_book.jsonl",
        )
        # Output/cache dirs
        cache_dir = os.getenv("DF_PDF2VQA_CACHE_DIR", "./cache")
        flash_intermediate_dir = os.getenv("DF_PDF2VQA_FLASH_DIR", os.path.join(cache_dir, "flash"))

        # LLM：OpenAI 兼容 HTTP API（默认 MiniMax，可改用其它网关）
        llm_api_url = os.getenv(
            "DF_LLM_API_URL",
            "https://api.minimax.io/v1/chat/completions",
        )
        llm_model_name = os.getenv("DF_LLM_MODEL", "MiniMax-M2.1")
        llm_max_workers = int(os.getenv("DF_LLM_MAX_WORKERS", "8"))
        llm_read_timeout = float(os.getenv("DF_LLM_READ_TIMEOUT", "600"))
        llm_connect_timeout = float(os.getenv("DF_LLM_CONNECT_TIMEOUT", "30"))
        llm_max_retries = int(os.getenv("DF_LLM_MAX_RETRIES", "5"))
        llm_max_tokens = int(os.getenv("DF_LLM_MAX_TOKENS", "32768"))
        # 单段用户 JSON 的 token 上限；整本 MinerU 数组建议配合 DF_PDF2VQA_MAX_JSON_ITEMS 分段
        llm_max_chunk_len = int(os.getenv("DF_PDF2VQA_MAX_CHUNK_LEN", "20000"))
        api_key_env_name = os.getenv("DF_LLM_API_KEY_ENV", "DF_API_KEY")

        mineru_backend = os.getenv("DF_MINERU_API_BACKEND", "vlm")

        if not (os.getenv("MINERU_API_KEY") or "").strip():
            raise RuntimeError(
                "全 API 模式需要 MinerU 云端密钥。请设置环境变量 MINERU_API_KEY，"
                "见 https://mineru.net/apiManage/token"
            )

        self.storage = FileStorage(
            first_entry_file_name=input_jsonl,
            cache_path=cache_dir,
            file_name_prefix="vqa",
            cache_type="jsonl",
        )
        
        self.llm_serving = APILLMServing_request(
            api_url=llm_api_url,
            key_name_of_api_key=api_key_env_name,
            model_name=llm_model_name,
            max_workers=llm_max_workers,
            max_retries=llm_max_retries,
            connect_timeout=llm_connect_timeout,
            read_timeout=llm_read_timeout,
            max_tokens=llm_max_tokens,
        )
        
        self.vqa_extract_prompt = QAExtractPrompt()
        
        self.pdf_merger = PDF_Merger(output_dir=cache_dir)

        self.mineru_executor = FileOrURLToMarkdownConverterAPI(
            intermediate_dir=flash_intermediate_dir,
            mineru_backend=mineru_backend,
        )

        self.input_formatter = MinerU2LLMInputOperator()
        self.vqa_extractor = ChunkedPromptedGenerator(
            llm_serving=self.llm_serving,
            system_prompt = self.vqa_extract_prompt.build_prompt(),
            max_chunk_len=llm_max_chunk_len,
        )
        self.llm_output_parser = LLMOutputParser(output_dir=cache_dir, intermediate_dir="intermediate")
        self.qa_merger = QA_Merger(output_dir=cache_dir, strict_title_match=True)

        self.vqa_format_converter = VQAFormatter(output_json_file="./.cache/data/qa.json",)

    def forward(self):
        self.pdf_merger.run(
            storage=self.storage.step(),
            input_pdf_list_key="input_pdf_paths",
            input_name_key="name",
            output_pdf_path_key="merged_pdf_path",
        )
        self.mineru_executor.run(
            storage=self.storage.step(),
            input_key="merged_pdf_path",
            output_key="vqa_markdown_path",
        )
        self.input_formatter.run(
            storage=self.storage.step(),
            input_markdown_path_key="vqa_markdown_path",
            output_converted_layout_key="converted_vqa_layout_path",
        )
        self.vqa_extractor.run(
            storage=self.storage.step(),
            input_path_key="converted_vqa_layout_path",
            output_path_key="extracted_llm_vqa_path",
        )
        self.llm_output_parser.run(
            storage=self.storage.step(),
            input_response_path_key="extracted_llm_vqa_path",
            input_converted_layout_path_key="converted_vqa_layout_path",
            input_name_key="name",
            output_qalist_path_key="extracted_vqa_path",
        )
        self.qa_merger.run(
            storage=self.storage.step(),
            input_qalist_path_key="extracted_vqa_path",
            input_name_key="name",
            output_merged_qalist_path_key="output_merged_vqalist_path",
            output_merged_md_path_key="output_merged_md_path",
            input_converted_layout_path_key="converted_vqa_layout_path",
            output_qa_item_key="vqa_pair",
        )
        self.vqa_format_converter.run(
            storage=self.storage.step(),
            input_qa_item_key="vqa_pair",
            output_messages_key="messages",
            output_images_key="images",
        )



if __name__ == "__main__":
    # jsonl中每一行包含input_pdf_path, name (math1, math2, physics1, chemistry1, ...)
    pipeline = PDF_VQA_extract_optimized_pipeline()
    pipeline.compile()
    pipeline.forward()