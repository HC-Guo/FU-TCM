#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
《望诊之钥 十字面形诊治法》专用切分脚本

切分策略：
1. 所有章节的"一、臉形特徵"小节：提取图片+文字 → VQA
2. 所有章节的"六、疾病特质"里的病例：提取图片+文字 → VQA
3. 其他小节：纯文字 → QA
"""

import json
import re
import os

def extract_chunks_from_markdown(md_path):
    """从markdown中提取chunks"""
    with open(md_path, 'r', encoding='utf-8') as f:
        content = f.read()

    chunks = []

    # 找到所有章节标题位置
    chapter_pattern = r'^# 第([一二三四五六七八九十]+)章 (.+?)$'
    matches = list(re.finditer(chapter_pattern, content, re.MULTILINE))

    print(f"找到 {len(matches)} 个章节标题")

    # 只处理包含 '# 一、' 的章节（实际内容，非目录）
    valid_chapters = []
    for i, match in enumerate(matches):
        chapter_num = match.group(1)
        chapter_title = match.group(2).strip()
        start_pos = match.end()
        end_pos = matches[i+1].start() if i+1 < len(matches) else len(content)
        chapter_content = content[start_pos:end_pos]

        # 只保留有实际小节标题的章节（不是目录）
        if '# 一、' in chapter_content:
            valid_chapters.append((chapter_num, chapter_title, chapter_content))

    print(f"有效章节数: {len(valid_chapters)}")

    for chapter_num, chapter_title, chapter_content in valid_chapters:
        print(f"处理: 第{chapter_num}章 {chapter_title}")

        # 处理该章节的所有小节
        sub_chunks = process_chapter(chapter_content)

        for chunk in sub_chunks:
            chunk['chapter'] = f"第{chapter_num}章 {chapter_title}"
            chunks.append(chunk)

    return chunks

def process_chapter(content):
    """处理单个章节，提取所有小节"""
    chunks = []

    # 找到所有小节
    subsection_pattern = r'^# ([一二三四五六七]、.+?)$'
    subsection_matches = list(re.finditer(subsection_pattern, content, re.MULTILINE))

    for i, match in enumerate(subsection_matches):
        title = match.group(1).strip()
        start_pos = match.end()
        end_pos = subsection_matches[i+1].start() if i+1 < len(subsection_matches) else len(content)
        section_content = content[start_pos:end_pos]

        # 判断小节类型
        is_first_section = title.startswith('一、')
        is_disease_section = '六' in title and '疾病' in title

        if is_first_section:
            # "一、臉形特徵" → VQA（保留图片）
            images = extract_images(section_content)
            text = clean_content(section_content)
            if text.strip() or images:
                chunks.append({
                    'title': title,
                    'text': text,
                    'images': images,
                    'type': 'vqa' if images else 'qa'
                })

        elif is_disease_section:
            # "六、疾病特质" → 提取里面的病例作为VQA
            case_chunks = extract_cases_from_disease_section(section_content)
            chunks.extend(case_chunks)

        else:
            # 其他小节 → QA（纯文字，不保留图片）
            text = clean_content(section_content)
            if text.strip():
                chunks.append({
                    'title': title,
                    'text': text,
                    'images': [],
                    'type': 'qa'
                })

    return chunks

def extract_cases_from_disease_section(content):
    """从疾病特质小节中提取病例 - 按图片划分"""
    chunks = []

    # 找到所有图片的位置
    img_pattern = r'!\[.*?\]\((images/[^)]+)\)'
    img_matches = list(re.finditer(img_pattern, content))

    if not img_matches:
        print("  未找到图片")
        return chunks

    print(f"  找到 {len(img_matches)} 张图片，按图片划分病例")

    # 找到所有标题（用于获取病例上下文）
    header_pattern = r'^# (.+?)$'
    headers = list(re.finditer(header_pattern, content, re.MULTILINE))
    header_dict = {m.start(): m.group(1).strip() for m in headers}

    for i, img_match in enumerate(img_matches):
        img_start = img_match.start()

        # 确定这个病例的结束位置：下一张图片开始，或内容结束
        if i + 1 < len(img_matches):
            case_end = img_matches[i + 1].start()
        else:
            case_end = len(content)

        # 提取病例内容（从当前图片开始）
        case_content = content[img_start:case_end]

        # 查找这个病例的标题
        # 1. 优先查找最近的 "# 病例X" 标记
        case_title = None
        for pos in sorted(header_dict.keys(), reverse=True):
            if pos < img_start:
                if '病例' in header_dict[pos]:
                    case_title = header_dict[pos]
                    break
                # 2. 如果没有病例标记，找最近的疾病类别（如 "# 1.肝膽疾病"）
                elif re.match(r'^\d+\.', header_dict[pos]):
                    case_title = f"疾病类别: {header_dict[pos]}"
                    break

        if not case_title:
            case_title = f"病例 {i+1}"

        images = extract_images(case_content)
        text = clean_content(case_content)

        if not text.strip() and not images:
            continue

        chunks.append({
            'title': f"六、疾病特质 - {case_title}",
            'text': text,
            'images': images,
            'type': 'vqa' if images else 'qa'
        })

    return chunks

def extract_images(content):
    """提取markdown中的图片"""
    images = []
    pattern = r'!\[.*?\]\((images/[^)]+)\)'
    matches = re.findall(pattern, content)
    for img_path in matches:
        # 提取对应的caption（图片后的文字）
        caption_pattern = r'!\[.*?\]\(' + re.escape(img_path) + r'\)\s*\n([^\n#]+)'
        caption_match = re.search(caption_pattern, content)
        caption = caption_match.group(1).strip() if caption_match else ""
        images.append({
            'path': img_path,
            'caption': caption
        })
    return images

def clean_content(content):
    """清理内容，移除图片标记但保留caption"""
    # 移除图片标记，但保留caption
    content = re.sub(r'!\[.*?\]\(images/[^)]+\)\s*\n?([^\n#])', r'\1', content)
    # 移除多余的空行
    content = re.sub(r'\n{3,}', '\n\n', content)
    return content.strip()

def main():
    md_path = './.cache/mineru/linyuanquan/0/full.md'
    output_path = './cache/linyuanquan/custom_chunks.json'

    print("开始切分《望诊之钥 十字面形诊治法》...")
    chunks = extract_chunks_from_markdown(md_path)

    # 统计
    vqa_count = sum(1 for c in chunks if c['type'] == 'vqa')
    qa_count = sum(1 for c in chunks if c['type'] == 'qa')

    print(f"\n切分完成:")
    print(f"  总chunks: {len(chunks)}")
    print(f"  VQA类型(有图): {vqa_count}")
    print(f"  QA类型(无图): {qa_count}")

    # 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"\n保存到: {output_path}")

    # 显示VQA chunks
    print("\n=== VQA chunks ===")
    for chunk in chunks:
        if chunk['type'] == 'vqa':
            print(f"\n[{chunk['chapter']}] {chunk['title']}")
            print(f"  图片数: {len(chunk['images'])}")
            for img in chunk['images']:
                print(f"    - {img['caption']}")

if __name__ == '__main__':
    main()
