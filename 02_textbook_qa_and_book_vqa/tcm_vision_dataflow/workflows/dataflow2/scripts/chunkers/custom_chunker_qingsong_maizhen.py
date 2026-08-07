#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
《轻松学脉诊》专用切分脚本

规则：
1. 前五章：按图号在正文中的引用定位，提取对应正文段落。
2. 后续章节（当前主要是 6-* 图组）：总图号对应一个大标题，小图按顺序对应 1/2/3/4 子标题。
3. 输出仅写入当前代码目录。
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

BOOK_NAME = "轻松学脉诊"
MD_ROOT = Path(
    r"E:\md\轻松学脉诊 (古聖先賢, 周幸来, 陈新华, 中華傳統文化, 中医学习, 中医医案) (Z-Library).pdf-7d5707a9-460c-4f7f-87f0-0dec9c3273b4"
)
MD_PATH = MD_ROOT / "full.md"
OUTPUT_ROOT = Path(
    r"d:\Desktop\Dataflow\Dataflow\DataFlow-main\maizhen_vqa_workdir\output\manual_split"
) / BOOK_NAME
OUTPUT_PATH = OUTPUT_ROOT / "split_samples.jsonl"
IMAGE_OUT_DIR = OUTPUT_ROOT / "images"

IMG_RE = re.compile(r"!\[\]\((images/[^)]+)\)")
FIG_LINE_RE = re.compile(r"^图\s*(\d+-\d+)\s*(.*)$")
CHAPTER_RE = re.compile(r"^#\s*第([一二三四五六七八九十0-9]+)章")
HEADER_RE = re.compile(r"^#\s*(.+?)\s*$")
MAJOR_HEAD_RE = re.compile(r"^#\s*[（(]([一二三四五六七八九十])[）)]\s*(.+)$")
MINOR_HEAD_RE = re.compile(r"^(?:#\s*)?(\d+)\.\s*([^\s]+.*)$")
FIG_REF_RE = re.compile(r"图\s*(\d+-\d+)")

CN_NUM = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}


@dataclass
class FigureGroup:
    figure_id: str
    caption: str
    caption_line_idx: int
    image_paths: list[str]


@dataclass
class MinorBlock:
    index: int
    title: str
    start: int
    end: int
    text: str


@dataclass
class MajorBlock:
    title: str
    start: int
    end: int
    text: str
    minors: list[MinorBlock]


def normalize_text(text: str) -> str:
    text = text.replace("\u3000", " ")
    text = text.replace("\r", "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def chapter_num_from_line(text: str) -> int | None:
    m = CHAPTER_RE.match(text.strip())
    if not m:
        return None
    raw = m.group(1)
    if raw.isdigit():
        return int(raw)
    return CN_NUM.get(raw)


def figure_chapter_num(figure_id: str) -> int | None:
    m = re.match(r"^(\d+)-", figure_id)
    return int(m.group(1)) if m else None


def clean_block(lines: list[str], start: int, end: int, drop_figure_lines: bool = False) -> str:
    out: list[str] = []
    for idx in range(start, end):
        stripped = lines[idx].strip()
        if not stripped:
            out.append("")
            continue
        if IMG_RE.search(stripped):
            continue
        if stripped == "图":
            continue
        if drop_figure_lines and FIG_LINE_RE.match(stripped):
            continue
        out.append(stripped)
    return normalize_text("\n".join(out))


def strip_heading_marks(text: str) -> str:
    return re.sub(r"^#\s*", "", text.strip())


def extract_figure_groups(lines: list[str]) -> list[FigureGroup]:
    groups: list[FigureGroup] = []
    pending_images: list[str] = []
    for idx, line in enumerate(lines):
        stripped = line.strip()
        img_match = IMG_RE.search(stripped)
        if img_match:
            pending_images.append(str((MD_ROOT / img_match.group(1)).resolve()))
            continue

        if stripped == "图" and idx + 1 < len(lines):
            merged = f"图 {lines[idx + 1].strip()}"
            m = FIG_LINE_RE.match(merged)
            if m and pending_images:
                groups.append(
                    FigureGroup(
                        figure_id=m.group(1),
                        caption=f"图{m.group(1)} {m.group(2)}".strip(),
                        caption_line_idx=idx,
                        image_paths=pending_images[:],
                    )
                )
                pending_images.clear()
            continue

        m = FIG_LINE_RE.match(stripped)
        if m and pending_images:
            groups.append(
                FigureGroup(
                    figure_id=m.group(1),
                    caption=f"图{m.group(1)} {m.group(2)}".strip(),
                    caption_line_idx=idx,
                    image_paths=pending_images[:],
                )
            )
            pending_images.clear()
    return groups


def nearest_ref_paragraph(lines: list[str], figure_id: str) -> tuple[int, str] | None:
    pattern = re.compile(rf"图\s*{re.escape(figure_id)}")
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or IMG_RE.search(stripped):
            continue
        if FIG_LINE_RE.match(stripped):
            continue
        if pattern.search(stripped):
            start = idx
            end = idx + 1
            while start > 0 and lines[start - 1].strip() and not HEADER_RE.match(lines[start - 1].strip()):
                if IMG_RE.search(lines[start - 1].strip()):
                    break
                start -= 1
            while end < len(lines) and lines[end].strip() and not HEADER_RE.match(lines[end].strip()):
                if IMG_RE.search(lines[end].strip()):
                    break
                end += 1
            return idx, normalize_text("\n".join(lines[i].strip() for i in range(start, end)))
    return None


def find_surrounding_header(lines: list[str], ref_idx: int) -> str:
    for idx in range(ref_idx, -1, -1):
        m = HEADER_RE.match(lines[idx].strip())
        if m:
            return m.group(1).strip()
    return ""


def extract_major_blocks(lines: list[str], start_line: int) -> list[MajorBlock]:
    major_starts: list[tuple[int, str]] = []
    for idx in range(start_line, len(lines)):
        m = MAJOR_HEAD_RE.match(lines[idx].strip())
        if m:
            major_starts.append((idx, f"（{m.group(1)}）{m.group(2).strip()}"))

    blocks: list[MajorBlock] = []
    for i, (start, title) in enumerate(major_starts):
        end = major_starts[i + 1][0] if i + 1 < len(major_starts) else len(lines)
        major_text = clean_block(lines, start, end, drop_figure_lines=True)
        minors = extract_minor_blocks(lines, start + 1, end)
        blocks.append(MajorBlock(title=title, start=start, end=end, text=major_text, minors=minors))
    return blocks


def extract_minor_blocks(lines: list[str], start: int, end: int) -> list[MinorBlock]:
    starts: list[tuple[int, int, str]] = []
    for idx in range(start, end):
        m = MINOR_HEAD_RE.match(lines[idx].strip())
        if m:
            starts.append((idx, int(m.group(1)), m.group(2).strip()))

    blocks: list[MinorBlock] = []
    for i, (line_idx, num, title) in enumerate(starts):
        block_end = starts[i + 1][0] if i + 1 < len(starts) else end
        text = clean_block(lines, line_idx, block_end, drop_figure_lines=True)
        blocks.append(MinorBlock(index=num, title=title, start=line_idx, end=block_end, text=text))
    return blocks


def find_major_for_figure(major_blocks: list[MajorBlock], line_idx: int) -> MajorBlock | None:
    for block in major_blocks:
        if block.start <= line_idx < block.end:
            return block
    return None


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
    groups = extract_figure_groups(lines)
    chapter6_blocks = extract_major_blocks(lines, 1000)

    samples: list[dict] = []
    for group in groups:
        chap_num = figure_chapter_num(group.figure_id)
        if chap_num is None:
            continue

        if chap_num <= 5:
            ref = nearest_ref_paragraph(lines, group.figure_id)
            if not ref:
                continue
            ref_idx, context_text = ref
            section_title = find_surrounding_header(lines, ref_idx)
            image_path = group.image_paths[-1]
            samples.append(
                {
                    "book_name": BOOK_NAME,
                    "anchor_type": "figure_ref_manual",
                    "section_title": section_title,
                    "image_path": image_path,
                    "image_caption": group.caption,
                    "figure_id": group.figure_id,
                    "context_text": context_text,
                    "mention_lines": [ref_idx],
                }
            )
            continue

        major = find_major_for_figure(chapter6_blocks, group.caption_line_idx)
        if not major:
            continue
        minors = major.minors or []
        if not minors:
            continue

        for idx, image_path in enumerate(group.image_paths):
            minor = minors[idx] if idx < len(minors) else minors[-1]
            image_caption = f"{minor.title}示意图"
            context_text = normalize_text(f"{major.title}\n\n{strip_heading_marks(minor.text)}")
            samples.append(
                {
                    "book_name": BOOK_NAME,
                    "anchor_type": "figure_ref_subimage_manual",
                    "section_title": major.title,
                    "subsection_title": minor.title,
                    "image_path": image_path,
                    "image_caption": image_caption,
                    "figure_id": f"{group.figure_id}-{idx+1}" if len(group.image_paths) > 1 else group.figure_id,
                    "parent_figure_id": group.figure_id,
                    "context_text": context_text,
                    "subsection_text": minor.text,
                    "mention_lines": [group.caption_line_idx],
                }
            )

    deduped: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for item in samples:
        key = (item["image_path"], item["figure_id"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


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
        print(f"image_caption: {item['image_caption']}")
        print(f"section_title: {item['section_title']}")
        preview = item["context_text"][:220].replace("\n", " ")
        print(f"context: {preview}...")


if __name__ == "__main__":
    main()
