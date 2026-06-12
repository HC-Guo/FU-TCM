import os
import json
import re
import shutil
from pathlib import Path
from typing import Literal
from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage
from dataflow import get_logger
from dataflow.utils.pdf2vqa.format_utils import normalize_qa_label

def _strip_reasoning_and_answer_wrapper(text: str) -> str:
    """
    MiniMax 等推理模型常在正文前输出思维链标签（例如 `<redacted_thinking>…</redacted_thinking>` 或 `<think>…</think>`），
    思维链里若含示例 `<chapter>`/`<qa_pair>` 会导致解析误匹配；先剥离再想定 XML。
    若外层包了一层 `<answer>…</answer>`（format_response 行为），只保留内部。
    """
    # 1) 剥离常见的“推理/思考”包装块（正常闭合时生效）
    text = re.sub(r"<redacted_thinking>.*?</redacted_thinking>\s*", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL | re.IGNORECASE)
    # 有些模型用 <thinking> 做相同用途
    text = re.sub(r"<thinking>.*?</thinking>\s*", "", text, flags=re.DOTALL | re.IGNORECASE)
    # 未闭合的推理块：截断输出时常只剩半截 thinking，去掉以免干扰；若其后无正文则整段清空
    if re.search(r"<redacted_thinking>", text, re.IGNORECASE) and not re.search(
        r"</redacted_thinking>", text, re.IGNORECASE
    ):
        text = re.sub(r"<redacted_thinking>.*", "", text, flags=re.DOTALL | re.IGNORECASE)
    if re.search(r"<thinking>", text, re.IGNORECASE) and not re.search(
        r"</thinking>", text, re.IGNORECASE
    ):
        text = re.sub(r"<thinking>.*", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = text.strip()

    # 2) 若外层包了一层 <answer>…</answer>，只保留内部
    m = re.match(r"\s*<answer>(.*)</answer>\s*\Z", text, flags=re.DOTALL)
    if m:
        text = m.group(1).strip()
    return text


def _sanitize_llm_xmlish_text(text: str) -> str:
    """
    对“看起来像 XML 但经常被模型输出破坏”的文本做轻量清洗，尽量提高正则解析成功率。
    目标：不追求还原为严格 XML，只为后续容错抽取 chapter/qa_pair。
    """
    if not text:
        return ""

    # 0) 优先提取“看起来像最终结果”的完整块，避免被思维链里的示例标签干扰。
    # 只接受含 <title> 的 chapter 块，尽量排除诸如 "<chapter>...</chapter>" 的说明文字。
    chapter_blocks = re.findall(
        r"<chapter>\s*<title>.*?</chapter>",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if chapter_blocks:
        return "\n".join(b.strip() for b in chapter_blocks if b.strip())

    empty_blocks = re.findall(r"<empty>\s*</empty>|<empty></empty>", text, flags=re.IGNORECASE)
    if empty_blocks:
        # empty 模式只需要一个即可
        return "<empty></empty>"

    # 1) 丢掉首个 <chapter>/<empty> 之前的所有杂质（常见：免责声明、推理残留、自然语言）
    m = re.search(r"(<chapter>|<empty>\s*</empty>|<empty></empty>)", text, flags=re.IGNORECASE)
    if m:
        text = text[m.start():]

    # 2) 修正常见的重复嵌套标签（模型经常输出 <source_text><source_text>...）
    text = re.sub(r"<source_text>\s*<source_text>", "<source_text>", text, flags=re.IGNORECASE)
    text = re.sub(r"</source_text>\s*</source_text>", "</source_text>", text, flags=re.IGNORECASE)

    return text.strip()


def _iter_tag_blocks(text: str, tag: str):
    """
    容错抽取形如 <tag>...</tag> 的块。
    - 若缺失 </tag>：则截断到下一个 <tag> 或文本末尾。
    - 返回的是“块内部内容”（不含外围 tag）。
    """
    if not text:
        return
    open_pat = re.compile(rf"<{tag}\b[^>]*>", flags=re.IGNORECASE)
    close_pat = re.compile(rf"</{tag}\s*>", flags=re.IGNORECASE)

    starts = [m.start() for m in open_pat.finditer(text)]
    if not starts:
        return

    for i, s in enumerate(starts):
        # 找到开标签结束位置
        m_open = open_pat.search(text, s)
        if not m_open:
            continue
        content_start = m_open.end()
        # 当前块的“搜索终点”：下一个开标签或末尾
        search_end = starts[i + 1] if i + 1 < len(starts) else len(text)
        # 在 [content_start, search_end) 内找闭标签；找不到就容错截断
        m_close = close_pat.search(text, content_start, search_end)
        content_end = m_close.start() if m_close else search_end
        yield text[content_start:content_end]


@OPERATOR_REGISTRY.register()
class LLMOutputParser(OperatorABC):
    def __init__(self, 
                 output_dir,
                 intermediate_dir: str = "intermediate",
                 ):
        self.logger = get_logger()
        self.output_dir = output_dir
        self.intermediate_dir = intermediate_dir
        
    @staticmethod
    def get_desc(lang: str = "zh") -> str:
        if lang == 'zh':
            return (
                "LLM输出解析算子。"
                "将LLM生成的包含题目和答案ID的响应文本，"
                "转换为结构化的QA列表，并复制相关图片到输出目录。"
            )
        else:
            return (
                "LLM output parsing operator."
                "Converts LLM-generated response text containing question and answer IDs"
                "into a structured QA list and copies related images to the output directory."
            )
    
    @staticmethod
    def _looks_like_id_list(raw: str) -> bool:
        """
        判断字符串是否形如 "12,13,14"（允许空格）。
        只有在这种情况下才会走“id->文本”映射；否则认为是自然语言内容直接返回。
        """
        return bool(re.fullmatch(r"\s*\d+(?:\s*,\s*\d+)*\s*", str(raw or "")))

    def _id_to_text(self, input_ids, input_json, image_prefix="images"):
        raw = str(input_ids or "")
        if not self._looks_like_id_list(raw):
            # 新版 Prompt 会直接在 <title>/<question>/<solution> 输出自然语言，
            # 此时不要再当 id 列表去映射。
            return raw.strip()

        texts = []
        id_list = raw.replace(' ', '').split(',')
        for id in id_list:
            try:
                int_id = int(id)
            except Exception:
                continue
            if int_id < len(input_json):
                try:
                    item = input_json[int_id]
                except Exception:
                    continue
                if 'text' in item:
                    texts.append(item['text'])
                elif 'table_body' in item:
                    texts.append(item['table_body'])
                elif 'img_path' in item:
                    try:
                        img_path = item.get('img_path', '')
                        img_name = os.path.basename(img_path)
                        new_path = f"{image_prefix}/{img_name}"
                        texts.append(f"![{' '.join(item.get('image_caption','image'))}]({new_path})")
                    except Exception:
                        pass
                elif item.get('type','') == 'list':
                    if item['sub_type'] == 'text':
                        try:
                            texts.append(input_json[int_id]['list_items'].pop(0))
                        except Exception:
                            pass
        return '\n'.join(texts)
    
    def _convert_response(self, input_response, input_json_path, image_prefix="images"):
        qa_list = []
        with open(input_json_path, 'r', encoding='utf-8') as infile:
            input_json = list(json.load(infile))
        input_response = _sanitize_llm_xmlish_text(input_response or "")
        # 提取title
        for chapter_block in _iter_tag_blocks(input_response, "chapter"):
            title = re.search(r'<title>(.*?)</title>', chapter_block, flags=re.DOTALL)
            if title:
                chapter_title = self._id_to_text(title.group(1).strip(), input_json, image_prefix)
            else:
                chapter_title = ""
            # 找出所有 qa_pair 块
            chapter_qa_counter = 0
            for pair in _iter_tag_blocks(chapter_block, "qa_pair"):
                # 提取 question 部分
                q_match = re.search(r'<question>(.*?)</question>', pair, flags=re.DOTALL)
                # 提取 answer 部分
                a_match = re.search(r'<answer>(.*?)</answer>', pair, flags=re.DOTALL)
                # 提取solution部分
                s_match = re.search(r'<solution>(.*?)</solution>', pair, flags=re.DOTALL)
                src_match = re.search(r'<source_text>(.*?)</source_text>', pair, flags=re.DOTALL)
                fig_match = re.search(r'<figure_id>(.*?)</figure_id>', pair, flags=re.DOTALL)
                # 提取label
                label_match = re.search(r'<label>(.*?)</label>', pair, flags=re.DOTALL)
                # 解析图片：<pic>img_path</pic>
                # 注意：VQAFormatter 只会从 question/solution 抽 markdown 图片，所以这里把图片追加到 question。
                pic_paths = re.findall(r'<pic>(.*?)</pic>', pair, flags=re.DOTALL)
                pic_paths = [p.strip() for p in pic_paths if str(p).strip()]

                # 兼容性：有些模型输出缺少 <label> 标签（只有 <question>/<answer>/<solution>）。
                # 为了让后续 QA_Merger/VQAFormatter 不至于直接跳过整条样本，
                # 这里给一个“正整数默认 label”，按章节递增生成。
                if q_match is None and a_match is None and s_match is None:
                    continue

                chapter_qa_counter += 1
                raw_label = label_match.group(1).strip() if label_match else None
                # 有些模型会输出 <label>○</label> 这类非数字标签；
                # merge_qa_pair 的 normalize_qa_label 无法解析它们，会导致整批被跳过。
                # 因此：若无法归一化为 int，则回退为 chapter 内递增的正整数 label。
                normalized = normalize_qa_label(raw_label)
                if normalized is None:
                    label = str(chapter_qa_counter)
                else:
                    label = str(normalized)

                def _strip_pic_tags(s: str) -> str:
                    # 如果模型把 <pic> 塞进 question/answer 内部，这里也一并去掉。
                    return re.sub(r"<pic>.*?</pic>", "", s or "", flags=re.DOTALL).strip()

                question_raw = _strip_pic_tags(q_match.group(1)) if q_match else ""
                answer_raw = _strip_pic_tags(a_match.group(1)) if a_match else ""
                solution_raw = _strip_pic_tags(s_match.group(1)) if s_match else ""
                source_raw = _strip_pic_tags(src_match.group(1)) if src_match else ""
                figure_id_raw = fig_match.group(1).strip() if fig_match else ""

                # 把 img_path 转为当前 pipeline 输出目录相对路径：vqa_images/<basename>
                pic_md = []
                for pic in pic_paths:
                    img_name = os.path.basename(pic)
                    if not img_name:
                        continue
                    pic_md.append(f"![image]({image_prefix}/{img_name})")
                pic_md_text = "\n".join(pic_md).strip()

                if pic_md_text:
                    if question_raw.strip():
                        question_out = f"{question_raw.strip()}\n{pic_md_text}"
                    else:
                        question_out = pic_md_text
                else:
                    question_out = question_raw.strip()

                qa_list.append({
                    'question': self._id_to_text(question_out, input_json, image_prefix) if question_out else "",
                    'answer': answer_raw.strip(),
                    'solution': self._id_to_text(solution_raw, input_json, image_prefix) if solution_raw else "",
                    'source_text': self._id_to_text(source_raw, input_json, image_prefix) if source_raw else "",
                    'figure_id': figure_id_raw,
                    'label': label,
                    'chapter_title': chapter_title
                })
        return qa_list
    
    def run(self, storage: DataFlowStorage,
            input_response_path_key,
            input_converted_layout_path_key,
            input_name_key,
            output_qalist_path_key,
            ):
        dataframe = storage.read("dataframe")
        
        # Response 转换
        for idx, row in dataframe.iterrows():
            converted_json_path = row[input_converted_layout_path_key]
            raw_response = Path(row[input_response_path_key]).read_text(encoding='utf-8')
            response = _strip_reasoning_and_answer_wrapper(raw_response)
            # 容错：
            # 1) 部分模型把最终 XML 也包在 <think> 内，剥离后会被清空；
            # 2) 也可能剥离后仅剩一个 <empty>，而原文其实含大量 <chapter>。
            stripped_chapter_cnt = response.count("<chapter>")
            raw_chapter_cnt = raw_response.count("<chapter>")
            stripped_has_xml = (stripped_chapter_cnt > 0) or ("<empty" in response)
            raw_has_xml = (raw_chapter_cnt > 0) or ("<empty" in raw_response)
            should_fallback_raw = ((not stripped_has_xml) and raw_has_xml) or (raw_chapter_cnt > stripped_chapter_cnt)
            if should_fallback_raw:
                self.logger.warning(
                    f"Reasoning strip lost XML richness (chapters stripped={stripped_chapter_cnt}, raw={raw_chapter_cnt}); fallback to raw response for parsing."
                )
                response = raw_response
            name = row[input_name_key]

            # 🚨 罪魁祸首在这里：它把 name（比如 math1）强行拼到了前缀里
            # image_prefix = os.path.join(name, f"vqa_images")
            # ✅ 修复 1：Markdown 的相对路径只需要文件夹名即可
            image_prefix = "vqa_images"
            # 这里把错误的带 math1/ 的前缀传给了内容解析器，写进了 JSON 和 MD 里
            qa_list = self._convert_response(response, converted_json_path, image_prefix)
            if not qa_list and "<empty>" not in (response or ""):
                self.logger.warning(
                    "extracted_vqa 为空：未从 LLM 输出中解析到 <chapter>。"
                    "若使用 MiniMax M2 等带 reasoning_content 的接口，请提高 DF_LLM_MAX_TOKENS，"
                    "并检查 *_llm_output.txt 是否在 </redacted_thinking> 之后仍有 <chapter>。"
                )
            output_qalist_path = os.path.join(self.output_dir, name, f"extracted_vqa.jsonl")
            os.makedirs(os.path.dirname(output_qalist_path), exist_ok=True)
            with open(output_qalist_path, 'w', encoding='utf-8') as outfile:
                for qa in qa_list:
                    json.dump(qa, outfile, ensure_ascii=False)
                    outfile.write('\n')
            
            # 复制图片
            src_dir = os.path.dirname(converted_json_path)
            src_images = os.path.join(src_dir, 'vlm', 'images')
            if not os.path.exists(src_images):
                src_images = os.path.join(src_dir, 'images')
            if not os.path.exists(src_images):
                self.logger.warning(f"Images directory {src_images} not found, skipping image copy (PDF may contain no images).")
            else:
                dst_images = os.path.join(self.output_dir, name, image_prefix)
                try:
                    shutil.copytree(src_images, dst_images)
                except Exception as e:
                    self.logger.warning(f"Failed to copy images from {src_images} to {dst_images}: {e}")
            
            dataframe.loc[idx, output_qalist_path_key] = output_qalist_path
            
        storage.write(dataframe)