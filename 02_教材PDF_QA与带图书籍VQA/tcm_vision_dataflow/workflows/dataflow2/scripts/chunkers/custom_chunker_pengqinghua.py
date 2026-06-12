#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
《中医脉诊临床图解》专用切分脚本

规则：
1. 在正文中找到对应图号引用（优先取图片前最近一次引用）。
2. 上文回溯到最近的标题边界，标题保留。
3. 下文在遇到下一个同级标题时截止，不保留该标题。
4. 标题边界包含两类：
   - 一、二、三……
   - 1. 2. 3. / 1、2、
5. 输出写入当前代码目录，并同步拷贝图片。
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path


BOOK_NAME = "中医脉诊临床图解"
MD_ROOT = Path(
    r"E:\md\中医脉诊临床图解 (彭清华，谢梦洲主编) (Z-Library).pdf-88864f6a-f832-45c7-9825-eefaa2b4e2bb"
)
MD_PATH = MD_ROOT / "full.md"
OUTPUT_ROOT = Path(
    r"d:\Desktop\Dataflow\Dataflow\DataFlow-main\maizhen_vqa_workdir\output\manual_split"
) / BOOK_NAME
OUTPUT_PATH = OUTPUT_ROOT / "split_samples.jsonl"
IMAGE_OUT_DIR = OUTPUT_ROOT / "images"

IMG_RE = re.compile(r"!\[\]\((images/[^)]+)\)")
FIGURE_ID_RE = re.compile(r"图\s*([0-9]+(?:[-－—.][0-9]+)+)")
CAPTION_RE = re.compile(r"^图\s*([0-9]+(?:[-－—.][0-9]+)+)\s*(.*)$")
FIGURE_REF_RE = re.compile(r"图\s*([0-9]+(?:[-－—.][0-9]+)+)")
CN_TITLE_RE = re.compile(r"^#?\s*([一二三四五六七八九十]+)、\s*(.+)$")
NUM_TITLE_RE = re.compile(r"^#?\s*([0-9]+)[\.、]\s*(.+)$")


@dataclass
class FigureEntry:
    figure_id: str
    caption: str
    line_idx: int
    image_path: str


def normalize_figure_id(text: str) -> str:
    return re.sub(r"[－—.]", "-", text.replace(" ", ""))


def normalize_text(text: str) -> str:
    text = text.replace("\u3000", " ")
    text = text.replace("\r", "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def strip_hash(text: str) -> str:
    return re.sub(r"^#\s*", "", text.strip())


def get_title_kind(text: str) -> str | None:
    if CN_TITLE_RE.match(text):
        return "cn"
    if NUM_TITLE_RE.match(text):
        return "num"
    return None


def extract_figures(lines: list[str]) -> list[FigureEntry]:
    figures: list[FigureEntry] = []
    seen_caption_lines: set[int] = set()
    for idx, line in enumerate(lines):
        img = IMG_RE.search(line.strip())
        if not img:
            continue
        image_path = str((MD_ROOT / img.group(1)).resolve())
        for off in range(1, 4):
            nxt = idx + off
            if nxt >= len(lines):
                break
            text = lines[nxt].strip()
            if not text:
                continue
            if text.startswith("#"):
                break
            cap = CAPTION_RE.match(text)
            if not cap:
                continue
            if nxt in seen_caption_lines:
                break
            seen_caption_lines.add(nxt)
            figure_id = normalize_figure_id(cap.group(1))
            figures.append(
                FigureEntry(
                    figure_id=figure_id,
                    caption=f"图{figure_id} {cap.group(2)}".strip(),
                    line_idx=nxt,
                    image_path=image_path,
                )
            )
            break
    return figures


def find_mentions(lines: list[str], fig: FigureEntry) -> list[int]:
    mentions: list[int] = []
    for idx, line in enumerate(lines):
        if idx >= fig.line_idx:
            break
        text = line.strip()
        if not text or IMG_RE.search(text):
            continue
        for match in FIGURE_REF_RE.finditer(text):
            if normalize_figure_id(match.group(1)) == fig.figure_id:
                mentions.append(idx)
                break
    return mentions


def previous_boundary(lines: list[str], line_idx: int) -> tuple[int, str, str] | None:
    for idx in range(line_idx, -1, -1):
        text = lines[idx].strip()
        kind = get_title_kind(text)
        if kind:
            return idx, strip_hash(text), kind
    return None


def next_boundary(lines: list[str], line_idx: int, kind: str) -> tuple[int, str] | None:
    for idx in range(line_idx + 1, len(lines)):
        text = lines[idx].strip()
        if kind == "cn" and CN_TITLE_RE.match(text):
            return idx, strip_hash(text)
        if kind == "num" and NUM_TITLE_RE.match(text):
            return idx, strip_hash(text)
    return None


def build_context(lines: list[str], mention_idx: int) -> tuple[str, str]:
    prev = previous_boundary(lines, mention_idx)
    start = prev[0] if prev else 0
    section_title = prev[1] if prev else ""
    next_info = next_boundary(lines, mention_idx, prev[2]) if prev else None
    end = next_info[0] if next_info else len(lines)

    out: list[str] = []
    for idx in range(start, end):
        stripped = lines[idx].strip()
        if not stripped:
            out.append("")
            continue
        if IMG_RE.search(stripped):
            continue
        if CAPTION_RE.match(stripped):
            continue
        out.append(strip_hash(stripped))
    return section_title, normalize_text("\n".join(out))


def copy_images_and_rewrite_paths(samples: list[dict]) -> list[dict]:
    IMAGE_OUT_DIR.mkdir(parents=True, exist_ok=True)
    keep_names = {Path(item["image_path"]).name for item in samples if Path(item["image_path"]).exists()}
    for existing in IMAGE_OUT_DIR.iterdir():
        if existing.is_file() and existing.name not in keep_names:
            existing.unlink()

    copied: dict[str, str] = {}
    rewritten: list[dict] = []
    for item in samples:
        src = Path(item["image_path"])
        if not src.exists():
            rewritten.append(item)
            continue
        if item["image_path"] not in copied:
            dst = IMAGE_OUT_DIR / src.name
            if not dst.exists():
                shutil.copy2(src, dst)
            copied[item["image_path"]] = str(dst)
        row = dict(item)
        row["image_path"] = copied[item["image_path"]]
        rewritten.append(row)
    return rewritten


def extract_samples(md_path: Path) -> list[dict]:
    lines = md_path.read_text(encoding="utf-8").splitlines()
    figures = extract_figures(lines)

    samples: list[dict] = []
    seen_ids: set[str] = set()
    for fig in figures:
        if fig.figure_id in seen_ids:
            continue
        mentions = find_mentions(lines, fig)
        if not mentions:
            continue
        mention_idx = mentions[-1]
        section_title, context_text = build_context(lines, mention_idx)
        if not context_text:
            continue
        seen_ids.add(fig.figure_id)
        samples.append(
            {
                "book_name": BOOK_NAME,
                "anchor_type": "figure_ref_manual",
                "section_title": section_title,
                "image_path": fig.image_path,
                "image_caption": fig.caption,
                "figure_id": f"图{fig.figure_id}",
                "context_text": context_text,
                "mention_lines": [mention_idx],
            }
        )
    return samples


def main() -> None:
    if not MD_PATH.exists():
        raise FileNotFoundError(f"未找到 markdown 文件: {MD_PATH}")

    samples = extract_samples(MD_PATH)
    samples = copy_images_and_rewrite_paths(samples)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for row in samples:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"切分完成: {BOOK_NAME}")
    print(f"样本数: {len(samples)}")
    print(f"输出文件: {OUTPUT_PATH}")
    print("\n前 8 条样本预览:")
    for item in samples[:8]:
        print("-" * 60)
        print(f"figure_id: {item['figure_id']}")
        print(f"section_title: {item['section_title']}")
        print(f"image_caption: {item['image_caption']}")
        preview = item["context_text"][:220].replace("\n", " ")
        print(f"context: {preview}...")


if __name__ == "__main__":
    main()
