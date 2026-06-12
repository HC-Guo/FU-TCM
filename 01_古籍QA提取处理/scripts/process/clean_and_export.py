#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
清洗项目根目录 qa_output 中有问题的 QA 对，并将删除的数据导出到 removed_qa_v1.json。

处理 8 类问题：
1. 思考泄露：LLM 的内部推理混入答案
2. 指代词：答案用了"以上药物"等代词
3. 空 q/a：问题或答案为空
4. 空 qa_pairs 的篇：整篇无有效 QA 对
5. 提示词残留："用户要求"、"基于原文"等 LLM 提示词混入
6. 对话式结尾：向用户索要原文的对话语气
7. 内容不完整：答案以省略号结尾
8. Markdown格式残留：**、--- 等格式符号
"""

import json
import glob
import re
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE = str(PROJECT_ROOT / 'qa_output')
OUTPUT_FILE = str(PROJECT_ROOT / 'removed_qa_v1.json')

# ==================== 检测规则 ====================

THINKING_PATTERNS = [
    '</think>', '<think>',
    '<analysis>', '</analysis>',
    '检查问题：', '检查是否', '确保每个问答', '确保问题', '确保基于原文',
    '让我分析一下', '让我仔细分析', '让我逐一思考', '让我思考', '用户要求针对', '用户要求我',
    '可以生成', '我可以生成', '我需要生成', '我将生成', '我会生成', '为了全面，', '合并或分别',
    '问答对应该', '问题4和问题5', '问题1和问题2', '问题2和问题3',
    '输出格式是', '格式要求',
    '现在，输出', '现在，整理', '现在整理所有','问答'
]

REFERENCE_PATTERNS = [
    '以上药物', '上述药物', '以上各药', '以上诸药',
    '以上方法', '上述方法', '以上材料', '以上配方', '上述配方',
    '上述成分', '以上成分', '以上内容', '上述内容',
    '以上症状', '上述症状',
]

# 5. 提示词残留
PROMPT_RESIDUE_PATTERNS = [
    '用户要求', '基于原文', '请提供', '请给出',
    '根据原文内容', '根据文本内容', '请回答', '请分析',
    '作为中医专家', '你是中医', '你是一个',
]

# 6. 对话式结尾（索要原文）
DIALOGUE_ENDING_PATTERNS = [
    '请提供原文', '请给出原文', '能否提供原文', '原文是什么',
    '需要查看原文', '需要原文', '请补充原文', '请上传原文',
]

# 7. 内容不完整（答案以省略号或冒号结尾）
INCOMPLETE_ENDINGS = ['...', '……', '···', '以此类推', '未完待续', '：', ':']

# 8. Markdown格式残留（**问答X： 或字面量 \n 或 * 格式）
MARKDOWN_PATTERNS = ['**问答', '\n', '*', '\\n']

# 9. 换行后接数字编号（格式混乱）
NUMBER_AFTER_NEWLINE = ['\n\n1.', '\n\n2.', '\n\n3.', '\n\n4.', '\n\n5.', '\n\n6.', '\n\n7.', '\n\n8.', '\n\n9.', '\n\n0.', '：\\n']

# 10. Q数字残留（\n\nQ, \n- Q 等）
Q_NUMBER_PATTERNS = ['\n\nQ', '\n\nq', '\n- Q', '\n- q']

# 11. 中英混杂（检测英文单词）
ENGLISH_PATTERN = re.compile(r'[a-zA-Z]{2,}')  # 匹配2个及以上英文字母

# 12. 问题中出现缺乏上下文的指代词
VAGUE_REFERENCE_PATTERNS = [
    # 药物指代
    '这味药', '这个药', '这种药', '该药', '此药', '本药', '此味中药',
    # 方剂指代
    '这个方', '该方', '此方', '本方', '上方', '前方',
    # 人物指代
    '此医', '该医', '此人',
    # 病症指代
    '此病', '该病', '本病', '这种病', '这个病',
    # 证候指代
    '此证', '该证', '本证', '此症', '该症', '本症',
    # 文献指代
    '此书', '该书', '本书', '此篇', '该篇', '本篇', '此文', '本文', '此卷', '本卷',
    # 章节指代
    '此节', '本节', '此章', '本章', '此条', '本条',
    # 医案指代
    '此案', '该案', '本案',
    # 其他指代
    '此法', '该法', '本法', '此穴', '该穴', '本穴', '此处', '该处',
    # 上下文引用
    '上述', '前述', '如上', '如前', '上文', '下文', '后文', '上一', '下一', '前一', '后一',
    # 属性指代
    '其功效', '其主治', '其用法', '其配伍', '其禁忌', '其性味', '其归经',
    # 代词开头
    '它的功效', '它的主治', '它的作用',
]

# 13. 问题或答案中出现"原文"（通常是提示词残留）
YUANWEN_PATTERN = '原文'


def detect_issue(pair: dict) -> tuple:
    """
    检测单个 QA 对的问题类型。
    返回 (issue_type, matched_keywords) 或 (None, None)
    """
    q = pair.get('q') or ''
    a = pair.get('a') or ''
    combined = q + '\x00' + a  # 用 null 字符连接，正常文本中不可能出现

    # 1. 空 q/a
    if not q.strip() or not a.strip():
        return 'empty_qa', []

    # 2. 思考泄露 (检测问题和答案)
    matched = [p for p in THINKING_PATTERNS if p in combined]
    if matched:
        return 'thinking_leak', matched

    # 3. 指代词 (检测答案)
    matched = [p for p in REFERENCE_PATTERNS if p in combined]
    if matched:
        return 'reference', matched

    # 5. 提示词残留 (检测答案)
    matched = [p for p in PROMPT_RESIDUE_PATTERNS if p in combined]
    if matched:
        return 'prompt_residue', matched

    # 6. 对话式结尾 (检测答案)
    matched = [p for p in DIALOGUE_ENDING_PATTERNS if p in combined]
    if matched:
        return 'dialogue_ending', matched

    # 7. 内容不完整（省略号结尾，检测答案）
    a_stripped = a.strip()
    for ending in INCOMPLETE_ENDINGS:
        if a_stripped.endswith(ending):
            return 'incomplete', [ending]

    # 8. Markdown格式残留 (检测问题和答案)
    matched = [p for p in MARKDOWN_PATTERNS if p in combined]
    if matched:
        return 'markdown', matched

    # 9. 换行后接数字编号 (检测问题和答案)
    for pattern in NUMBER_AFTER_NEWLINE:
        if pattern in combined:
            return 'number_after_newline', [pattern.strip()]

    # 10. Q数字残留 (检测问题和答案)
    for pattern in Q_NUMBER_PATTERNS:
        if pattern in combined:
            return 'q_number_residue', [pattern.strip()]

    # 11. 中英混杂 (检测问题和答案)
    english_matches = ENGLISH_PATTERN.findall(combined)
    if english_matches:
        return 'english_mixed', english_matches[:5]  # 最多返回前5个匹配

    # 12. 问题中出现"此方"等缺乏上下文的指代词 (仅检测问题)
    matched = [p for p in VAGUE_REFERENCE_PATTERNS if p in q]
    if matched:
        return 'vague_reference', matched

    # 13. 问题或答案中出现"原文"（通常是提示词残留）
    if YUANWEN_PATTERN in combined:
        return 'yuanwen_residue', ['原文']

    return None, None


def clean_file(filepath: str, rel_path: str) -> dict:
    """
    清洗单个文件。
    返回该文件被删除的数据信息。
    """
    with open(filepath, encoding='utf-8') as f:
        data = json.load(f)

    book_name = data.get('book_name', '')
    qa_data = data.get('qa_data', [])

    removed_items = []
    new_qa_data = []

    for pian in qa_data:
        pian_name = pian.get('pian_name', '')
        original_content = pian.get('original_content', '')
        pairs = pian.get('qa_pairs') or []

        # 空 qa_pairs 的篇
        if not pairs:
            removed_items.append({
                'pian_name': pian_name,
                'original_content': original_content,
                'issue_type': 'empty_pian',
                'matched_keywords': [],
                'q': '',
                'a': '',
                'full_response': (pian.get('full_response') or '')[:500],
            })
            continue

        clean_pairs = []
        for pair in pairs:
            issue_type, matched = detect_issue(pair)
            if issue_type:
                removed_items.append({
                    'pian_name': pian_name,
                    'original_content': original_content,
                    'issue_type': issue_type,
                    'matched_keywords': matched,
                    'q': pair.get('q', ''),
                    'a': pair.get('a', ''),
                })
            else:
                clean_pairs.append(pair)

        # 清洗后如果篇还有 QA 对，保留
        if clean_pairs:
            pian['qa_pairs'] = clean_pairs
            new_qa_data.append(pian)
        else:
            # 整篇的 QA 对全被删了，记录篇级删除
            removed_items.append({
                'pian_name': pian_name,
                'original_content': original_content,
                'issue_type': 'all_pairs_removed',
                'matched_keywords': [],
                'q': '',
                'a': '',
            })

    # 覆盖原文件
    if len(new_qa_data) != len(qa_data) or removed_items:
        data['qa_data'] = new_qa_data
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    return {
        'source_file': rel_path,
        'book_name': book_name,
        'items': removed_items,
    }


def main():
    files = sorted([
        f for f in glob.glob(BASE + '/**/*.json', recursive=True)
        if not f.endswith('.bak')
    ])

    print(f'共找到 {len(files)} 个 JSON 文件，开始清洗...\n')

    all_removed = []
    stats = defaultdict(int)
    files_modified = 0

    for filepath in files:
        rel_path = filepath.replace(BASE + '/', '')
        result = clean_file(filepath, rel_path)

        if result['items']:
            all_removed.append(result)
            files_modified += 1
            for item in result['items']:
                stats[item['issue_type']] += 1

    # 汇总
    total_removed = sum(len(r['items']) for r in all_removed)

    summary = {
        'total_files': len(files),
        'files_modified': files_modified,
        'total_removed': total_removed,
        'thinking_leak': stats['thinking_leak'],
        'reference': stats['reference'],
        'empty_qa': stats['empty_qa'],
        'empty_pian': stats['empty_pian'],
        'all_pairs_removed': stats['all_pairs_removed'],
        'prompt_residue': stats['prompt_residue'],
        'dialogue_ending': stats['dialogue_ending'],
        'incomplete': stats['incomplete'],
        'markdown': stats['markdown'],
        'number_after_newline': stats['number_after_newline'],
        'q_number_residue': stats['q_number_residue'],
        'english_mixed': stats['english_mixed'],
        'vague_reference': stats['vague_reference'],
        'yuanwen_residue': stats['yuanwen_residue'],
    }

    output = {
        'summary': summary,
        'removed_data': all_removed,
    }

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print('===== 清洗完成 =====')
    print(f'处理文件数:          {summary["total_files"]}')
    print(f'修改文件数:          {summary["files_modified"]}')
    print(f'删除数据总数:        {summary["total_removed"]}')
    print(f'  思考泄露:          {summary["thinking_leak"]}')
    print(f'  指代词:            {summary["reference"]}')
    print(f'  空q/a:             {summary["empty_qa"]}')
    print(f'  空qa_pairs篇:      {summary["empty_pian"]}')
    print(f'  整篇清空:          {summary["all_pairs_removed"]}')
    print(f'  提示词残留:        {summary["prompt_residue"]}')
    print(f'  对话式结尾:        {summary["dialogue_ending"]}')
    print(f'  内容不完整:        {summary["incomplete"]}')
    print(f'  Markdown残留:      {summary["markdown"]}')
    print(f'  换行后数字:        {summary["number_after_newline"]}')
    print(f'  Q数字残留:         {summary["q_number_residue"]}')
    print(f'  中英混杂:          {summary["english_mixed"]}')
    print(f'  问题指代模糊:      {summary["vague_reference"]}')
    print(f'  原文残留:          {summary["yuanwen_residue"]}')
    print(f'\n已保存删除数据: {OUTPUT_FILE}')


if __name__ == '__main__':
    main()
