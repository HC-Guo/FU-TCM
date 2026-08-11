import os
import json
import pandas as pd
import tiktoken
import re
from datetime import datetime
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from pathlib import Path

from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC

@OPERATOR_REGISTRY.register()
class ChunkedPromptedGenerator(OperatorABC):
    """
    基于Prompt的生成算子，支持自动chunk输入。
    - 使用tiktoken或HuggingFace的AutoTokenizer计算token数量；
    - 若输入超过max_chunk_len，采用递归二分法切分；
    - 从指定输入文件路径读取内容，生成结果保存至指定输出文件路径；
    - 生成结果是以separator拼接的字符串。
    """

    def __init__(
        self,
        llm_serving: LLMServingABC,
        system_prompt: str = "You are a helpful agent.",
        json_schema: dict = None,
        max_chunk_len: int = 128000,
        enc = tiktoken.get_encoding("cl100k_base"), # 支持len(enc.encode(text))的tokenizer都可以，比如tiktoken或HuggingFace的AutoTokenizer
        separator: str = "\n",
    ):
        self.logger = get_logger()
        self.llm_serving = llm_serving
        self.system_prompt = system_prompt
        self.json_schema = json_schema
        self.max_chunk_len = max_chunk_len
        self.enc = enc
        self.separator = separator

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "基于提示词的生成算子，支持长文本自动分chunk。"
                "采用递归二分方式进行chunk切分，确保每段不超过max_chunk_len tokens。"
                "从给定的输入文件路径读取内容，生成结果保存至指定输出文件路径。"
                "多个生成结果以separator拼接成最终输出字符串。"
                "输入参数：\n"
                "- llm_serving：LLM服务对象，需实现LLMServingABC接口\n"
                "- system_prompt：系统提示词，定义模型行为，默认为'You are a helpful agent.'\n"
                "- max_chunk_len：单个chunk的最大token长度，默认为128000\n"
                "- input_path_key：输入文件路径字段名，默认为'input_path'\n"
                "- output_path_key：输出文件路径字段名，默认为'output_path'\n"
                "- json_schema：可选，生成结果的JSON Schema约束\n"
                "- enc：用于token计算的编码器，需要实现encode方法，默认为tiktoken的cl100k_base编码器，也可以使用HuggingFace 的 AutoTokenizer\n"
                "- separator：chunk结果拼接分隔符，默认为换行符\n"
            )
        else:
            return (
                "Prompt-based generator with recursive chunk splitting."
                "Splits long text inputs into chunks using recursive bisection to ensure each chunk does not exceed max_chunk_len tokens."
                "Reads content from specified input file paths and saves generated results to designated output file paths."
                "Multiple generated results are joined as a string using the specified separator."
                "Input Parameters:\n"
                "- llm_serving: LLM serving object implementing LLMServingABC interface\n"
                "- system_prompt: System prompt to define model behavior, default is 'You are a helpful agent.'\n"
                "- max_chunk_len: Maximum token length per chunk, default is 128000\n"
                "- input_path_key: Field name for input file path, default is 'input_path'\n"
                "- output_path_key: Field name for output file path, default is 'output_path'\n"
                "- json_schema: Optional JSON Schema constraint for generated results\n"
                "- enc: Encoder for token counting, default is tiktoken's cl100k_base encoder; can also use HuggingFace's AutoTokenizer\n"
                "- separator: Separator for chunk results, default is newline character\n"
            )

    # === token计算 ===
    def _count_tokens(self, text: str) -> int:
        return len(self.enc.encode(text))

    # === 递归二分分chunk ===
    def _pick_structural_split(self, text: str, mid: int) -> int:
        """
        优先在 JSON/段落边界切分，避免把结构体从中间劈开导致模型难以解析。
        返回 0 表示未找到合适边界（调用方回退到普通二分）。
        """
        # 常见边界：JSON 对象分隔、数组项分隔、换行段落分隔
        markers = [r"\},\s*\{", r"\}\s*\n\s*\{", r"\]\s*,\s*\[", r"\n\n+"]
        window = max(512, min(len(text) // 3, 20000))
        left = max(1, mid - window)
        right = min(len(text) - 1, mid + window)
        region = text[left:right]

        best_pos = 0
        best_dist = None
        for pat in markers:
            for m in re.finditer(pat, region):
                # 在分隔符末尾切，更自然
                pos = left + m.end()
                if pos <= 0 or pos >= len(text):
                    continue
                dist = abs(pos - mid)
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_pos = pos
        return best_pos

    def _split_recursive(self, text: str) -> list[str]:
        """递归地将文本拆分为不超过max_chunk_len的多个chunk"""
        token_len = self._count_tokens(text)
        if token_len <= self.max_chunk_len:
            return [text]
        else:
            mid = len(text) // 2
            split_pos = self._pick_structural_split(text, mid)
            if split_pos <= 0 or split_pos >= len(text):
                split_pos = mid
            left, right = text[:split_pos], text[split_pos:]
            return self._split_recursive(left) + self._split_recursive(right)

    def _json_slice_to_chunks(self, lo: int, hi: int, data: list) -> list[str]:
        """将 data[lo:hi] 序列化为 JSON 数组；若超长则按索引二分，保证每段不超过 max_chunk_len。"""
        if lo >= hi:
            return []
        chunk_json = json.dumps(data[lo:hi], ensure_ascii=False)
        if self._count_tokens(chunk_json) <= self.max_chunk_len or hi - lo <= 1:
            return [chunk_json]
        mid = (lo + hi) // 2
        return self._json_slice_to_chunks(lo, mid, data) + self._json_slice_to_chunks(mid, hi, data)

    def _split_json_array(self, data: list) -> list[str]:
        """
        顶层 JSON 数组按「元素条数 + token 上限」切块，每块仍是合法 JSON 数组字符串。
        环境变量：DF_PDF2VQA_MAX_JSON_ITEMS（默认 80；≤0 表示禁用本路径，走纯文本二分）。
        段内若仍超 max_chunk_len，会再按 token 二分，故过大项数+过小 max_chunk_len 时单次调用数可能不会线性下降。
        """
        max_items = int(os.getenv("DF_PDF2VQA_MAX_JSON_ITEMS", "80"))
        if max_items <= 0:
            return []
        out: list[str] = []
        n = len(data)
        i = 0
        while i < n:
            j = min(i + max_items, n)
            out.extend(self._json_slice_to_chunks(i, j, data))
            i = j
        total = len(out)
        return [
            f"【输入为 JSON 数组的第 {idx + 1}/{total} 段；须按系统提示覆盖本段内每一个有效 img_path（通常每图多条 QA），不可遗漏应生成的图】\n{c}"
            for idx, c in enumerate(out)
        ]

    def _split_into_chunks(self, raw_content: str) -> list[str]:
        """优先：合法 JSON 数组则按元素切块；否则按 token 递归二分原文。"""
        max_items = int(os.getenv("DF_PDF2VQA_MAX_JSON_ITEMS", "80"))
        s = raw_content.strip()
        if max_items > 0 and s.startswith("["):
            try:
                data = json.loads(s)
            except json.JSONDecodeError:
                data = None
            if isinstance(data, list) and len(data) > 0:
                chunks = self._split_json_array(data)
                if chunks:
                    self.logger.info(
                        f"JSON 数组切分：共 {len(data)} 条元素 -> {len(chunks)} 次 LLM 调用 "
                        f"(DF_PDF2VQA_MAX_JSON_ITEMS={max_items}, max_chunk_len={self.max_chunk_len})"
                    )
                    return chunks
        return self._split_recursive(raw_content)

    def run(
        self,
        storage: DataFlowStorage,
        input_path_key,
        output_path_key,
    ):
        self.logger.info("Running ChunkedPromptedGenerator...")
        dataframe = storage.read("dataframe")
        self.logger.info(f"Loaded DataFrame with {len(dataframe)} rows.")

        all_generated_results = []

        all_llm_inputs = []
        row_chunk_map = []  # 记录每个row对应的chunk数量

        # === 先收集所有chunk ===
        for i, row in dataframe.iterrows():
            raw_content = Path(row[input_path_key]).read_text(encoding='utf-8')

            chunks = self._split_into_chunks(raw_content)
            self.logger.info(f"Row {i}: split into {len(chunks)} chunks")

            system_prompt = self.system_prompt + "\n"
            llm_inputs = [system_prompt + chunk for chunk in chunks]
            all_llm_inputs.extend(llm_inputs)
            row_chunk_map.append(len(chunks))

        # === 一次性并发调用 ===
        self.logger.info(f"Total {len(all_llm_inputs)} chunks to generate")

        try:
            if self.json_schema:
                all_responses = self.llm_serving.generate_from_input(
                    all_llm_inputs, json_schema=self.json_schema
                )
            else:
                all_responses = self.llm_serving.generate_from_input(all_llm_inputs)
        except Exception as e:
            self.logger.error(f"Global generation failed: {e}")
            all_generated_results = [[] for _ in range(len(dataframe))]
        else:
            # === 按row重新划分responses ===
            all_generated_results = []
            idx = 0
            for num_chunks in row_chunk_map:
                if num_chunks == 0:
                    all_generated_results.append([])
                else:
                    all_generated_results.append(all_responses[idx:idx + num_chunks])
                    idx += num_chunks

            # 记录全局失败 chunk，便于定向补跑
            failed_chunks = [
                {"global_chunk_id": i}
                for i, resp in enumerate(all_responses)
                if resp is None
            ]
            if failed_chunks:
                failure_report_path = os.path.join(
                    storage.cache_path,
                    f"failed_llm_chunks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                )
                with open(failure_report_path, "w", encoding="utf-8") as fw:
                    json.dump(
                        {
                            "failed_count": len(failed_chunks),
                            "total_chunks": len(all_responses),
                            "failed_chunks": failed_chunks,
                        },
                        fw,
                        ensure_ascii=False,
                        indent=2,
                    )
                self.logger.warning(
                    f"Detected {len(failed_chunks)} failed chunks. Report saved to: {failure_report_path}"
                )

        for (i, row), gen_results in zip(dataframe.iterrows(), all_generated_results):
            output_path = row[input_path_key].split('.')[0] + '_llm_output.txt'
            safe_parts = []
            for ci, part in enumerate(gen_results):
                if part is None:
                    self.logger.warning(
                        f"Row {i} chunk {ci}: LLM returned None (timeout or API error); writing empty string for this chunk."
                    )
                    safe_parts.append("")
                else:
                    safe_parts.append(part if isinstance(part, str) else str(part))
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(self.separator.join(safe_parts))
            dataframe.at[i, output_path_key] = output_path
        
        output_file = storage.write(dataframe)
        self.logger.info(f"Generation complete. Output saved to {output_file}")
        return output_path_key