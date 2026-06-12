#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import re

input_file = "merged_qa_pairs(1).jsonl"
output_file = "中医望诊与舌诊彩色图解_题库.html"

records = []
with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))

# Extract image path from markdown: ![image](vqa_images/xxx.jpg)
def extract_image(text):
    m = re.search(r'!\[image\]\(([^)]+)\)', text)
    return m.group(1) if m else None

def clean_question(text):
    return re.sub(r'!\[image\]\([^)]+\)', '', text).strip()

# Group by chapter
chapters = {}
for r in records:
    ch = r.get("question_chapter_title", "未分类")
    if ch not in chapters:
        chapters[ch] = []
    chapters[ch].append(r)

html_parts = []
html_parts.append("<!DOCTYPE html>")
html_parts.append('<html lang="zh-CN">')
html_parts.append("<head>")
html_parts.append('<meta charset="UTF-8">')
html_parts.append('<meta name="viewport" content="width=device-width, initial-scale=1.0">')
html_parts.append("<title>中医望诊与舌诊彩色图解 - 题库</title>")
html_parts.append("<style>")
html_parts.append("""
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
    background: #f5f7fa;
    color: #333;
    line-height: 1.6;
  }
  .header {
    background: linear-gradient(135deg, #1a5276 0%, #2e86ab 100%);
    color: white;
    padding: 30px 20px;
    text-align: center;
    position: sticky;
    top: 0;
    z-index: 100;
    box-shadow: 0 2px 10px rgba(0,0,0,0.15);
  }
  .header h1 { font-size: 1.6rem; margin-bottom: 8px; }
  .header .stats { font-size: 0.9rem; opacity: 0.9; }
  .nav {
    background: #fff;
    padding: 12px 16px;
    border-bottom: 1px solid #e0e6ed;
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    justify-content: center;
    position: sticky;
    top: 88px;
    z-index: 99;
    box-shadow: 0 2px 6px rgba(0,0,0,0.05);
  }
  .nav a {
    text-decoration: none;
    color: #2e86ab;
    padding: 6px 14px;
    border-radius: 20px;
    border: 1px solid #d0e3f0;
    font-size: 0.85rem;
    transition: all 0.2s;
    white-space: nowrap;
  }
  .nav a:hover, .nav a.active {
    background: #2e86ab;
    color: white;
    border-color: #2e86ab;
  }
  .container { max-width: 1000px; margin: 0 auto; padding: 20px; }
  .chapter {
    margin-bottom: 40px;
    scroll-margin-top: 140px;
  }
  .chapter-title {
    font-size: 1.3rem;
    color: #1a5276;
    border-left: 5px solid #2e86ab;
    padding-left: 14px;
    margin-bottom: 20px;
    font-weight: 600;
  }
  .card {
    background: white;
    border-radius: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    margin-bottom: 20px;
    overflow: hidden;
    transition: box-shadow 0.2s;
  }
  .card:hover { box-shadow: 0 4px 16px rgba(0,0,0,0.1); }
  .card-header {
    background: #f8fafc;
    padding: 12px 18px;
    border-bottom: 1px solid #edf2f7;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .card-label {
    background: #2e86ab;
    color: white;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 0.75rem;
  }
  .figure-id {
    color: #718096;
    font-size: 0.8rem;
  }
  .card-body { padding: 18px; }
  .card-image {
    width: 100%;
    max-height: 400px;
    object-fit: contain;
    border-radius: 8px;
    margin-bottom: 14px;
    background: #f0f0f0;
  }
  .question {
    font-size: 1.05rem;
    color: #2d3748;
    margin-bottom: 14px;
    font-weight: 500;
  }
  .answer-box, .solution-box {
    border-radius: 8px;
    padding: 14px;
    margin-bottom: 10px;
  }
  .answer-box {
    background: #f0fff4;
    border-left: 4px solid #48bb78;
  }
  .solution-box {
    background: #fffaf0;
    border-left: 4px solid #ed8936;
  }
  .answer-label, .solution-label {
    font-weight: 600;
    font-size: 0.85rem;
    margin-bottom: 6px;
    display: block;
  }
  .answer-label { color: #276749; }
  .solution-label { color: #c05621; }
  .answer-text { color: #2d3748; }
  .source-text {
    font-size: 0.8rem;
    color: #a0aec0;
    margin-top: 10px;
    padding-top: 10px;
    border-top: 1px dashed #e2e8f0;
  }
  .back-to-top {
    position: fixed;
    bottom: 24px;
    right: 24px;
    background: #2e86ab;
    color: white;
    width: 44px;
    height: 44px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    text-decoration: none;
    font-size: 1.2rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    transition: transform 0.2s;
  }
  .back-to-top:hover { transform: translateY(-2px); }
  @media (max-width: 600px) {
    .header h1 { font-size: 1.2rem; }
    .nav { top: 76px; }
    .chapter { scroll-margin-top: 130px; }
    .card-body { padding: 12px; }
  }
""")
html_parts.append("</style>")
html_parts.append("</head>")
html_parts.append("<body>")
html_parts.append('<div class="header" id="top">')
html_parts.append("  <h1>中医望诊与舌诊彩色图解</h1>")
html_parts.append('  <div class="stats">共 ' + str(len(records)) + ' 道题目 · ' + str(len(chapters)) + ' 个章节</div>')
html_parts.append("</div>")
html_parts.append('<div class="nav">')

for i, ch in enumerate(chapters.keys()):
    anchor = "ch" + str(i)
    html_parts.append('  <a href="#' + anchor + '">' + ch + '</a>')

html_parts.append('  <a href="#top" style="background:#1a5276;color:white;border-color:#1a5276;">顶部</a>')
html_parts.append("</div>")
html_parts.append('<div class="container">')

for i, (ch, items) in enumerate(chapters.items()):
    anchor = "ch" + str(i)
    html_parts.append('<div class="chapter" id="' + anchor + '">')
    html_parts.append('  <div class="chapter-title">' + ch + '（' + str(len(items)) + '题）</div>')

    for idx, r in enumerate(items):
        img_path = extract_image(r.get("question", ""))
        question_text = clean_question(r.get("question", ""))
        answer = r.get("answer", "")
        solution = r.get("solution", "")
        figure_id = r.get("figure_id", "")
        label = r.get("label", idx + 1)
        source_text = r.get("source_text", "")

        html_parts.append('  <div class="card">')
        html_parts.append('    <div class="card-header">')
        html_parts.append('      <span class="card-label">第 ' + str(label) + ' 题</span>')
        html_parts.append('      <span class="figure-id">' + figure_id + '</span>')
        html_parts.append('    </div>')
        html_parts.append('    <div class="card-body">')
        if img_path:
            html_parts.append('      <img class="card-image" src="' + img_path + '" alt="' + figure_id + '" loading="lazy">')
        html_parts.append('      <div class="question">' + question_text.replace('\n', '<br>') + '</div>')
        html_parts.append('      <div class="answer-box">')
        html_parts.append('        <span class="answer-label">答案</span>')
        html_parts.append('        <div class="answer-text">' + answer.replace('\n', '<br>') + '</div>')
        html_parts.append('      </div>')
        if solution:
            html_parts.append('      <div class="solution-box">')
            html_parts.append('        <span class="solution-label">解析</span>')
            html_parts.append('        <div class="answer-text">' + solution.replace('\n', '<br>') + '</div>')
            html_parts.append('      </div>')
        if source_text:
            html_parts.append('      <div class="source-text">来源：' + source_text + '</div>')
        html_parts.append('    </div>')
        html_parts.append('  </div>')

    html_parts.append('</div>')

html_parts.append("</div>")
html_parts.append('<a href="#top" class="back-to-top" title="回到顶部">&#8679;</a>')
html_parts.append("</body>")
html_parts.append("</html>")

with open(output_file, "w", encoding="utf-8") as f:
    f.write("\n".join(html_parts))

print("已生成: " + output_file)
print("总题数: " + str(len(records)))
print("章节数: " + str(len(chapters)))
for ch, items in chapters.items():
    print("  - " + ch + ": " + str(len(items)) + " 题")
