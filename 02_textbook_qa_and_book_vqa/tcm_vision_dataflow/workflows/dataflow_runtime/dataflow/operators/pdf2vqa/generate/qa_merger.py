import os
import json
from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.pdf2vqa.format_utils import merge_qa_pair, jsonl_to_md

import re

@OPERATOR_REGISTRY.register()
class QA_Merger(OperatorABC):
    def __init__(self, output_dir, strict_title_match=False):
        self.output_dir = output_dir
        self.strict_title_match = strict_title_match
        
    @staticmethod
    def get_desc(lang: str = "zh") -> str:
        if lang == 'zh':
            return (
                "QA对合并算子。"
                "将问题和答案的QA列表进行合并，生成最终的QA对文件，"
                "并转换为Markdown格式。"
            )
        else:
            return (
                "QA pair merging operator."
                "Merges question and answer QA lists to generate final QA pair files,"
                "and converts them to Markdown format."
            )
    
    def run(self, storage: DataFlowStorage,
            input_qalist_path_key,
            input_name_key,
            output_merged_qalist_path_key,
            output_merged_md_path_key,
            input_converted_layout_path_key=None,
            output_qa_item_key="qa_item"  # 新增：展开后的 QA 内容列名
            ):
        dataframe = storage.read("dataframe")
        
        # 为了能存储 list 对象，先初始化该列为 object 类型
        dataframe[output_qa_item_key] = None
        dataframe[output_qa_item_key] = dataframe[output_qa_item_key].astype(object)

        for idx, row in dataframe.iterrows():
            qa_list_path = row[input_qalist_path_key]
            name = row[input_name_key]
            
            output_merged_qalist_path = os.path.join(self.output_dir, name, "merged_qa_pairs.jsonl")
            merge_qa_pair(qa_list_path, output_merged_qalist_path, strict_title_match=self.strict_title_match)

            qa_pairs = []
            if os.path.exists(output_merged_qalist_path):
                with open(output_merged_qalist_path, 'r', encoding='utf-8') as f:
                    qa_pairs = [json.loads(line) for line in f]

            # 在合并阶段直接补充图号字段，避免后补脚本
            def _extract_figure_id(text: str) -> str:
                if not isinstance(text, str):
                    return ""
                m = re.search(r'图\s*([0-9]+)\s*-\s*([0-9]+)', text)
                if not m:
                    return ""
                return f"图{m.group(1)}-{m.group(2)}"

            def _extract_image_path(text: str) -> str:
                if not isinstance(text, str):
                    return ""
                m = re.search(r'!\[(.*?)\]\((.*?)\)', text)
                return m.group(2) if m else ""

            # 优先：从 converted layout 里构建 img_path -> figure_id 映射（比 source_text 更稳）
            layout_image_to_figure = {}
            if input_converted_layout_path_key and input_converted_layout_path_key in row:
                converted_path = row[input_converted_layout_path_key]
                if isinstance(converted_path, str) and os.path.exists(converted_path):
                    try:
                        with open(converted_path, 'r', encoding='utf-8') as cf:
                            layout_items = json.load(cf)
                        if isinstance(layout_items, list):
                            def _text_from_item(it):
                                texts = []
                                if isinstance(it.get("image_caption"), list):
                                    texts.extend([str(x) for x in it.get("image_caption") if x])
                                elif isinstance(it.get("image_caption"), str):
                                    texts.append(it.get("image_caption"))
                                if isinstance(it.get("text"), str):
                                    texts.append(it.get("text"))
                                return " ".join(texts)

                            for i_it, it in enumerate(layout_items):
                                if not isinstance(it, dict):
                                    continue
                                img_path = it.get("img_path", "")
                                if not isinstance(img_path, str) or not img_path:
                                    continue
                                # 候选文本：本项 caption/text + 前后邻接项 text/caption
                                cand_texts = [_text_from_item(it)]
                                if i_it - 1 >= 0 and isinstance(layout_items[i_it - 1], dict):
                                    cand_texts.append(_text_from_item(layout_items[i_it - 1]))
                                if i_it + 1 < len(layout_items) and isinstance(layout_items[i_it + 1], dict):
                                    cand_texts.append(_text_from_item(layout_items[i_it + 1]))
                                figure_id = ""
                                for t in cand_texts:
                                    figure_id = _extract_figure_id(t)
                                    if figure_id:
                                        break
                                if figure_id:
                                    layout_image_to_figure[os.path.basename(img_path)] = figure_id
                    except Exception:
                        pass

            # 先从 source_text 提取
            for qa in qa_pairs:
                if not isinstance(qa, dict):
                    continue
                if not qa.get("figure_id"):
                    qa["figure_id"] = _extract_figure_id(qa.get("source_text", ""))

            # 再按同图片路径回填
            image_to_figure = {}
            for qa in qa_pairs:
                if not isinstance(qa, dict):
                    continue
                img_path = _extract_image_path(qa.get("question", ""))
                fig_id = qa.get("figure_id", "")
                if img_path and fig_id and img_path not in image_to_figure:
                    image_to_figure[img_path] = fig_id

            for qa in qa_pairs:
                if not isinstance(qa, dict):
                    continue
                if qa.get("figure_id"):
                    continue
                img_path = _extract_image_path(qa.get("question", ""))
                img_base = os.path.basename(img_path)
                # 1) 先用 converted layout 的映射（推荐）
                if img_base and img_base in layout_image_to_figure:
                    qa["figure_id"] = layout_image_to_figure[img_base]
                # 2) 再回退到同批 qa 的图片路径映射
                elif img_path in image_to_figure:
                    qa["figure_id"] = image_to_figure[img_path]
                else:
                    qa["figure_id"] = ""

            # 将 figure_id 放在 question 之前，保证输出结构稳定
            reordered_pairs = []
            for qa in qa_pairs:
                if not isinstance(qa, dict):
                    reordered_pairs.append(qa)
                    continue
                ordered = {}
                preferred_order = [
                    "question_chapter_title",
                    "answer_chapter_title",
                    "label",
                    "figure_id",
                    "question",
                    "answer",
                    "solution",
                    "source_text",
                ]
                for key in preferred_order:
                    if key in qa:
                        ordered[key] = qa[key]
                for key, value in qa.items():
                    if key not in ordered:
                        ordered[key] = value
                reordered_pairs.append(ordered)
            qa_pairs = reordered_pairs

            # 回写 merged_qa_pairs.jsonl（带 figure_id）
            with open(output_merged_qalist_path, 'w', encoding='utf-8') as f:
                for qa in qa_pairs:
                    f.write(json.dumps(qa, ensure_ascii=False) + "\n")

            output_merged_md_path = os.path.join(self.output_dir, name, "merged_qa_pairs.md")
            jsonl_to_md(output_merged_qalist_path, output_merged_md_path)
            
            dataframe.at[idx, output_qa_item_key] = qa_pairs

            dataframe.loc[idx, output_merged_qalist_path_key] = output_merged_qalist_path
            dataframe.loc[idx, output_merged_md_path_key] = output_merged_md_path
            
        dataframe = dataframe.explode(output_qa_item_key).reset_index(drop=True)

        # 汇总jsonl中的图片路径需要将 ![alt](path) 中的 path 替换为 name/path
        def fix_image_paths(row):
            qa_item = row[output_qa_item_key]
            name_val = str(row[input_name_key])
            
            if isinstance(qa_item, dict):
                keys_to_check = ["question", "answer", "solution", "source_text"]
                for key in keys_to_check:
                    if key in qa_item and isinstance(qa_item[key], str):
                        qa_item[key] = re.sub(
                            r'!\[(.*?)\]\((.*?)\)',
                            lambda m: f"![{m.group(1)}]({os.path.join(name_val, m.group(2))})",
                            qa_item[key]
                        )
            return qa_item

        dataframe[output_qa_item_key] = dataframe.apply(fix_image_paths, axis=1)

        storage.write(dataframe)