from __future__ import annotations

import json
import re

# 合并 QA 时用于把书内题号/小节号转成 int（兼容 （一）、①、例1 等）
_CN_DIGIT = {
    "零": 0,
    "一": 1,
    "二": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}


def normalize_qa_label(raw) -> int | None:
    """
    将 LLM/书籍中的 label 规范为可排序的 int，供 merge_qa_pair 使用。
    无法解析时返回 None（该行在合并阶段跳过）。
    """
    if raw is None:
        return None
    if isinstance(raw, int):
        return raw
    s = str(raw).strip()
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        pass
    m = re.search(r"\d+", s)
    if m:
        return int(m.group())
    # （一）、(二)
    pm = re.search(r"[（(]\s*([一二三四五六七八九十]+)\s*[）)]", s)
    if pm:
        cn = pm.group(1)
        if len(cn) == 1 and cn in _CN_DIGIT:
            return _CN_DIGIT[cn]
        if cn == "十":
            return 10
        if len(cn) == 2 and cn[0] == "十" and cn[1] in _CN_DIGIT:
            return 10 + _CN_DIGIT[cn[1]]
        if len(cn) == 2 and cn[1] == "十" and cn[0] in _CN_DIGIT:
            return _CN_DIGIT[cn[0]] * 10
    # ① — ⑨（U+2460–U+2468）
    if len(s) == 1:
        o = ord(s)
        if 0x2460 <= o <= 0x2468:
            return o - 0x2460 + 1
    return None


def refine_title(title: str, strict_title_match=False):
    # TODO : 这里可能需要更复杂的title清洗逻辑
    # 删除title中的空格与换行符
    title = re.sub(r'\s+', '', title)
    if not strict_title_match:
        try:
            # 优先提取阿拉伯数字章节编号（如1.1，2等）
            new_title = re.search(r"\d+\.\d+|\d+", title).group()
        except:    
            try:
                # 其次提取中文数字章节编号（如六、二十四等）
                new_title = re.search(r'[一二三四五六七八九零十百]+', title).group()   
            except Exception:
                new_title = title
        title = new_title
    return title

def merge_qa_pair(vqa_jsonl, output_jsonl, strict_title_match=False):
    already_complete_count = 0
    question_list = []
    answer_list = []
    with open(vqa_jsonl, 'r', encoding='utf-8') as vqa_file:
        for line in vqa_file:
            data = json.loads(line)
            if data["question"] != "":
                question_list.append(data)
            else:
                # 用于支持题目在前面，答案在后面的pdf
                answer_list.append(data)

    with open(output_jsonl, 'w', encoding='utf-8') as out_file:
        chapter_id = 0
        chapter_title = ""
        label = float('inf')
        questions = {}
        answers = {}
        for data in question_list:
            nl = normalize_qa_label(data.get("label"))
            if nl is None:
                continue
            data["label"] = nl
            if data["chapter_title"] == "":
                data["chapter_title"] = chapter_title
            
            if data["chapter_title"] != "" and data["chapter_title"] != chapter_title:
                if data["label"] < label:
                    chapter_id += 1
                    chapter_title = data["chapter_title"]
                else:
                    # 如果题号增加，章节标题却发生变化，说明可能错误提取了子标题。因此继续使用之前的章节标题。
                    data["chapter_title"] = chapter_title
            label = data["label"]
            data["chapter_title"] = refine_title(data["chapter_title"], strict_title_match)
            if data['label'] > 0:
                # 已经完整的题目直接写入out_file
                if data["answer"] or data["solution"]:
                    already_complete_count += 1
                    qa_pair = {
                        "question_chapter_title": data["chapter_title"],
                        "answer_chapter_title": data["chapter_title"],
                        "label": data['label'],
                        "question": data["question"],
                        "answer": data["answer"],
                        "solution": data.get("solution", ""),
                        "source_text": data.get("source_text", ""),
                    }
                    out_file.write(json.dumps(qa_pair, ensure_ascii=False) + '\n')
                    
                else:
                    questions[(data["chapter_title"], data['label'])] = data
        
        chapter_id = 0
        chapter_title = ""
        label = float('inf')
        for data in answer_list:
            nl = normalize_qa_label(data.get("label"))
            if nl is None:
                continue
            data["label"] = nl
            if data["chapter_title"] == "":
                data["chapter_title"] = chapter_title
            
            if data["chapter_title"] != "" and data["chapter_title"] != chapter_title:
                if data["label"] < label:
                    chapter_id += 1
                    chapter_title = data["chapter_title"]
                else:
                    # 如果题号增加，章节标题却发生变化，说明可能错误提取了子标题。因此继续使用之前的章节标题。
                    data["chapter_title"] = chapter_title
            label = data["label"]
            data["chapter_title"] = refine_title(data["chapter_title"], strict_title_match)
            # 动态更新，防止错误的重复label覆盖掉之前的solution或answer
            if data['label'] > 0:
                if not answers.get((data["chapter_title"], data['label'])):
                    answers[(data["chapter_title"], data['label'])] = data
                else:
                    if not answers[(data["chapter_title"], data['label'])].get("solution") and data.get("solution"):
                        answers[(data["chapter_title"], data['label'])]["solution"] = data["solution"]
                    if not answers[(data["chapter_title"], data['label'])].get("answer") and data.get("answer"):
                        answers[(data["chapter_title"], data['label'])]["answer"] = data["answer"]
      
        for label in questions:
            if label in answers:
                qa_pair = {
                    "question_chapter_title": questions[label]["chapter_title"],
                    "answer_chapter_title": answers[label]["chapter_title"],
                    "label": label[1],
                    "question": questions[label]["question"],
                    "answer": answers[label]["answer"],
                    "solution": answers[label].get("solution", ""),
                    "source_text": questions[label].get("source_text", "")
                    or answers[label].get("source_text", ""),
                }
                out_file.write(json.dumps(qa_pair, ensure_ascii=False) + '\n')
        
        print(f"Merged QA pairs: {len(questions.keys() & answers.keys()) + already_complete_count}")
        
def jsonl_to_md(jsonl_file, md_file):
    with open(jsonl_file, 'r', encoding='utf-8') as in_file, open(md_file, 'w', encoding='utf-8') as out_file:
        for line in in_file:
            data = json.loads(line)
            out_file.write(f"### Question {data['label']}\n\n")
            out_file.write(f"{data['question']}\n\n")
            out_file.write(f"**Answer:** {data['answer']}\n\n")
            if data.get('solution'):
                out_file.write(f"**Solution:**\n\n{data['solution']}\n\n")
            if data.get('source_text'):
                out_file.write(f"**Source (校验):**\n\n{data['source_text']}\n\n")
            out_file.write("---\n\n")