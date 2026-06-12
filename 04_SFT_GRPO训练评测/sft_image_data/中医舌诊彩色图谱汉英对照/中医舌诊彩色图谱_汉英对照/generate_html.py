#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate a single-file HTML viewer for merged_qa_pairs.jsonl."""

import json
import re
import sys
import html
import os
from collections import OrderedDict

INPUT = sys.argv[1] if len(sys.argv) > 1 else 'merged_qa_pairs.jsonl'
OUTPUT = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(INPUT)[0].replace(' ', '_') + '_viewer.html'
# Fallback: if INPUT is the default, keep historical filename
if INPUT == 'merged_qa_pairs.jsonl' and len(sys.argv) <= 2:
    OUTPUT = 'qa_viewer.html'

IMG_RE = re.compile(r'!\[image\]\(([^)]+)\)')


def parse_question(q):
    """Split a question string into (text, [image_paths])."""
    images = IMG_RE.findall(q)
    text = IMG_RE.sub('', q).strip()
    return text, images


def main():
    chapters = OrderedDict()
    with open(INPUT, 'r', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            ch = d.get('question_chapter_title') or '未分类'
            chapters.setdefault(ch, []).append(d)

    total = sum(len(v) for v in chapters.values())

    # Build sidebar
    sidebar_items = []
    for idx, (ch, items) in enumerate(chapters.items()):
        anchor = f'chapter-{idx}'
        sidebar_items.append(
            f'<li><a href="#{anchor}" data-chapter="{html.escape(ch)}">'
            f'{html.escape(ch)} <span class="count">({len(items)})</span></a></li>'
        )
    sidebar_html = '\n'.join(sidebar_items)

    # Build main content
    sections = []
    for idx, (ch, items) in enumerate(chapters.items()):
        anchor = f'chapter-{idx}'
        cards = []
        for it in items:
            label = it.get('label', '')
            q_text, q_imgs = parse_question(it.get('question', ''))
            answer = it.get('answer', '') or ''
            solution = it.get('solution', '') or ''
            source = it.get('source_text', '') or ''
            figure_id = it.get('figure_id', '') or ''

            img_html = ''
            if q_imgs:
                img_tags = []
                for p in q_imgs:
                    img_tags.append(
                        f'<a href="{html.escape(p)}" target="_blank">'
                        f'<img src="{html.escape(p)}" alt="tongue image" loading="lazy"></a>'
                    )
                img_html = f'<div class="images">{"".join(img_tags)}</div>'

            meta_bits = [f'#{label}']
            if figure_id:
                meta_bits.append(f'figure: {html.escape(figure_id)}')
            meta = ' · '.join(meta_bits)

            solution_html = (
                f'<div class="row"><span class="lbl">解析</span>'
                f'<div class="val">{html.escape(solution)}</div></div>'
                if solution.strip() else ''
            )
            source_html = (
                f'<div class="row"><span class="lbl">原文</span>'
                f'<div class="val src">{html.escape(source)}</div></div>'
                if source.strip() else ''
            )

            cards.append(f'''
<article class="card" data-search="{html.escape((q_text + ' ' + answer + ' ' + solution + ' ' + source).lower())}">
  <header class="card-head">
    <span class="badge">{meta}</span>
    <span class="chapter-tag">{html.escape(ch)}</span>
  </header>
  {img_html}
  <div class="qa">
    <div class="row"><span class="lbl q">问</span><div class="val">{html.escape(q_text)}</div></div>
    <div class="row"><span class="lbl a">答</span><div class="val">{html.escape(answer)}</div></div>
    {solution_html}
    {source_html}
  </div>
</article>''')

        sections.append(f'''
<section id="{anchor}" class="chapter" data-chapter="{html.escape(ch)}">
  <h2 class="chapter-title">{html.escape(ch)} <span class="count">({len(items)})</span></h2>
  {''.join(cards)}
</section>''')

    main_html = '\n'.join(sections)

    page = f'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>温病舌诊图谱 QA 查看器 ({total} 条)</title>
<style>
  :root {{
    --bg: #f7f6f2;
    --panel: #ffffff;
    --ink: #1f2328;
    --muted: #6a737d;
    --accent: #b22222;
    --accent2: #1f6feb;
    --border: #e3e1da;
    --tag: #efece4;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    font-family: -apple-system, "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
    background: var(--bg);
    color: var(--ink);
    line-height: 1.6;
  }}
  .layout {{ display: flex; min-height: 100vh; }}
  aside {{
    width: 260px;
    flex-shrink: 0;
    border-right: 1px solid var(--border);
    background: var(--panel);
    position: sticky;
    top: 0;
    height: 100vh;
    overflow-y: auto;
    padding: 20px 16px;
  }}
  aside h1 {{
    font-size: 16px;
    margin: 0 0 4px;
    color: var(--accent);
  }}
  aside .sub {{
    font-size: 12px;
    color: var(--muted);
    margin-bottom: 16px;
  }}
  aside ul {{
    list-style: none;
    padding: 0;
    margin: 0;
  }}
  aside li {{ margin: 2px 0; }}
  aside a {{
    display: block;
    padding: 6px 10px;
    border-radius: 6px;
    color: var(--ink);
    text-decoration: none;
    font-size: 13px;
  }}
  aside a:hover {{ background: var(--tag); }}
  aside a.active {{ background: var(--accent); color: #fff; }}
  aside a.active .count {{ color: #fff; }}
  aside .count {{ color: var(--muted); font-size: 11px; }}

  main {{
    flex: 1;
    padding: 0 28px 80px;
    max-width: 1100px;
  }}
  .toolbar {{
    position: sticky;
    top: 0;
    background: var(--bg);
    padding: 16px 0 12px;
    border-bottom: 1px solid var(--border);
    z-index: 10;
    display: flex;
    gap: 12px;
    align-items: center;
    flex-wrap: wrap;
  }}
  .toolbar input {{
    flex: 1;
    min-width: 220px;
    padding: 8px 12px;
    border: 1px solid var(--border);
    border-radius: 6px;
    font-size: 14px;
    background: #fff;
  }}
  .toolbar .stat {{ font-size: 12px; color: var(--muted); }}

  .chapter-title {{
    font-size: 22px;
    margin: 28px 0 14px;
    padding-bottom: 8px;
    border-bottom: 2px solid var(--accent);
    color: var(--accent);
  }}
  .chapter-title .count {{ color: var(--muted); font-size: 14px; font-weight: normal; }}

  .card {{
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 16px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.03);
  }}
  .card.hidden {{ display: none; }}
  .card-head {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 10px;
  }}
  .badge {{
    font-size: 12px;
    color: var(--muted);
    font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
  }}
  .chapter-tag {{
    font-size: 11px;
    color: var(--muted);
    background: var(--tag);
    padding: 2px 8px;
    border-radius: 10px;
  }}
  .images {{
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    margin: 8px 0 14px;
  }}
  .images img {{
    max-width: 240px;
    max-height: 240px;
    border: 1px solid var(--border);
    border-radius: 6px;
    cursor: zoom-in;
    background: #fafafa;
  }}
  .qa .row {{
    display: flex;
    gap: 10px;
    align-items: flex-start;
    padding: 6px 0;
    border-top: 1px dashed var(--border);
  }}
  .qa .row:first-child {{ border-top: none; }}
  .lbl {{
    flex-shrink: 0;
    width: 36px;
    text-align: center;
    font-size: 12px;
    color: #fff;
    background: var(--muted);
    border-radius: 4px;
    padding: 2px 0;
    height: fit-content;
    line-height: 1.4;
    margin-top: 3px;
  }}
  .lbl.q {{ background: var(--accent2); }}
  .lbl.a {{ background: var(--accent); }}
  .val {{ flex: 1; white-space: pre-wrap; word-break: break-word; }}
  .val.src {{ color: var(--muted); font-size: 13px; }}

  /* Lightbox */
  .lightbox {{
    position: fixed;
    inset: 0;
    background: rgba(0,0,0,0.85);
    display: none;
    align-items: center;
    justify-content: center;
    z-index: 100;
    cursor: zoom-out;
  }}
  .lightbox.open {{ display: flex; }}
  .lightbox img {{ max-width: 92%; max-height: 92%; }}

  @media (max-width: 760px) {{
    aside {{ display: none; }}
    main {{ padding: 0 14px 60px; }}
    .images img {{ max-width: 45%; max-height: 200px; }}
  }}
</style>
</head>
<body>
<div class="layout">
  <aside>
    <h1>温病舌诊图谱</h1>
    <div class="sub">QA 校对查看器 · 共 {total} 条</div>
    <ul id="chapter-nav">
      {sidebar_html}
    </ul>
  </aside>
  <main>
    <div class="toolbar">
      <input id="search" type="search" placeholder="搜索题目 / 答案 / 解析 / 原文 …">
      <span class="stat" id="stat">显示 {total} / {total}</span>
    </div>
    {main_html}
  </main>
</div>

<div class="lightbox" id="lightbox"><img id="lightbox-img" alt=""></div>

<script>
  // Search filter
  const input = document.getElementById('search');
  const stat = document.getElementById('stat');
  const cards = Array.from(document.querySelectorAll('.card'));
  const chapters = Array.from(document.querySelectorAll('.chapter'));
  const total = cards.length;

  input.addEventListener('input', () => {{
    const q = input.value.trim().toLowerCase();
    let shown = 0;
    cards.forEach(c => {{
      const hay = c.dataset.search || '';
      const match = !q || hay.includes(q);
      c.classList.toggle('hidden', !match);
      if (match) shown++;
    }});
    chapters.forEach(ch => {{
      const visible = ch.querySelectorAll('.card:not(.hidden)').length;
      ch.style.display = visible ? '' : 'none';
    }});
    stat.textContent = `显示 ${{shown}} / ${{total}}`;
  }});

  // Lightbox
  const lb = document.getElementById('lightbox');
  const lbImg = document.getElementById('lightbox-img');
  document.querySelectorAll('.images a').forEach(a => {{
    a.addEventListener('click', e => {{
      e.preventDefault();
      lbImg.src = a.getAttribute('href');
      lb.classList.add('open');
    }});
  }});
  lb.addEventListener('click', () => lb.classList.remove('open'));

  // Sidebar active state on scroll
  const navLinks = Array.from(document.querySelectorAll('#chapter-nav a'));
  const obs = new IntersectionObserver(entries => {{
    entries.forEach(en => {{
      if (en.isIntersecting) {{
        const id = en.target.id;
        navLinks.forEach(l => l.classList.toggle('active', l.getAttribute('href') === '#' + id));
      }}
    }});
  }}, {{ rootMargin: '-40% 0px -55% 0px' }});
  chapters.forEach(c => obs.observe(c));
</script>
</body>
</html>
'''

    with open(OUTPUT, 'w', encoding='utf-8') as f:
        f.write(page)

    print(f'Wrote {OUTPUT} with {total} entries across {len(chapters)} chapters.')


if __name__ == '__main__':
    main()
