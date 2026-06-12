#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
《王光宇精准脉诊带教录 第2版》专用切分脚本

规则：
1. 普通图（图1、图2... / ★图1...）：在正文中找对应图号引用。
   上文回溯到最近的 "1. / 2. / 3." 这类大标题，保留该标题。
2. 附图（附图1、附图2... / ★附图1...）：在正文中找对应附图引用。
   上文回溯到最近的 "（1）/（2）" 这类小标题，保留该标题。
3. 下文均在遇到下一个同级标题前截止，不包含下一个标题。
4. 输出只写到当前代码目录。
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

BOOK_NAME = "王光宇精准脉诊带教录 第2版"
MD_ROOT = Path(
    r"E:\md\王光宇精准脉诊带教录 第2版 (王光宇) (Z-Library).pdf-c1e60d56-8152-462e-ac5f-e6121c66d927"
)
MD_PATH = MD_ROOT / "full.md"
OUTPUT_ROOT = Path(
    r"d:\Desktop\Dataflow\Dataflow\DataFlow-main\maizhen_vqa_workdir\output\manual_split"
) / BOOK_NAME
OUTPUT_PATH = OUTPUT_ROOT / "split_samples.jsonl"
IMAGE_OUT_DIR = OUTPUT_ROOT / "images"

IMG_RE = re.compile(r"!\[\]\((images/[^)]+)\)")
NORMAL_CAPTION_RE = re.compile(r"^[★*]?\s*图\s*(\d+)\s*(.*)$")
ATTACH_CAPTION_RE = re.compile(r"^[★*]?\s*附图\s*(\d+)\s*(.*)$")
NORMAL_REF_RE = re.compile(r"(?<!附)(?:★)?图\s*(\d+)")
ATTACH_REF_RE = re.compile(r"(?:附录三)?附图\s*(\d+)")
HEADER_RE = re.compile(r"^#\s*(.+?)\s*$")
BIG_TITLE_RE = re.compile(r"^#?\s*(\d+)\.\s*(.+)$")
SMALL_TITLE_RE = re.compile(r"^#?\s*[（(](\d+)[）)]\s*(.+)$")


@dataclass
class FigureEntry:
    kind: str  # normal / attach
    figure_id: str
    caption: str
    line_idx: int
    image_path: str


def normalize_text(text: str) -> str:
    text = text.replace("\u3000", " ")
    text = text.replace("\r", "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def strip_hash(text: str) -> str:
    return re.sub(r"^#\s*", "", text.strip())


def extract_figures(lines: list[str]) -> list[FigureEntry]:
    figures: list[FigureEntry] = []
    seen_caption_keys: set[tuple[str, int]] = set()
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
            m1 = NORMAL_CAPTION_RE.match(text)
            if m1:
                key = ("normal", nxt)
                if key in seen_caption_keys:
                    break
                seen_caption_keys.add(key)
                figures.append(
                    FigureEntry(
                        kind="normal",
                        figure_id=m1.group(1),
                        caption=f"图{m1.group(1)} {m1.group(2)}".strip(),
                        line_idx=nxt,
                        image_path=image_path,
                    )
                )
                break
            m2 = ATTACH_CAPTION_RE.match(text)
            if m2:
                key = ("attach", nxt)
                if key in seen_caption_keys:
                    break
                seen_caption_keys.add(key)
                figures.append(
                    FigureEntry(
                        kind="attach",
                        figure_id=m2.group(1),
                        caption=f"附图{m2.group(1)} {m2.group(2)}".strip(),
                        line_idx=nxt,
                        image_path=image_path,
                    )
                )
                break
            if text.startswith("#"):
                break
    return figures


def find_mentions(lines: list[str], fig: FigureEntry) -> list[int]:
    pattern = NORMAL_REF_RE if fig.kind == "normal" else ATTACH_REF_RE
    mentions: list[int] = []
    for idx, line in enumerate(lines):
        if idx >= fig.line_idx:
            break
        text = line.strip()
        if not text or IMG_RE.search(text):
            continue
        if fig.kind == "normal":
            if "附图" in text:
                continue
            for m in pattern.finditer(text):
                if m.group(1) == fig.figure_id:
                    mentions.append(idx)
                    break
        else:
            for m in pattern.finditer(text):
                if m.group(1) == fig.figure_id:
                    mentions.append(idx)
                    break
    return mentions


def previous_boundary(lines: list[str], line_idx: int, kind: str) -> tuple[int, str] | None:
    for idx in range(line_idx, -1, -1):
        text = lines[idx].strip()
        if kind == "normal":
            if BIG_TITLE_RE.match(text):
                return idx, strip_hash(text)
            if text.startswith("#") and not SMALL_TITLE_RE.match(text):
                return idx, strip_hash(text)
        else:
            if SMALL_TITLE_RE.match(text):
                return idx, strip_hash(text)
    return None


def next_boundary(lines: list[str], line_idx: int, kind: str) -> tuple[int, str] | None:
    matcher = BIG_TITLE_RE if kind == "normal" else SMALL_TITLE_RE
    for idx in range(line_idx + 1, len(lines)):
        text = lines[idx].strip()
        m = matcher.match(text)
        if m:
            return idx, strip_hash(text)
    return None


def build_context(lines: list[str], mention_idx: int, kind: str) -> tuple[str, str]:
    prev = previous_boundary(lines, mention_idx, kind)
    nxt = next_boundary(lines, mention_idx, kind)
    start = prev[0] if prev else 0
    end = nxt[0] if nxt else len(lines)
    section_title = prev[1] if prev else ""

    out: list[str] = []
    for idx in range(start, end):
        stripped = lines[idx].strip()
        if not stripped:
            out.append("")
            continue
        if IMG_RE.search(stripped):
            continue
        if NORMAL_CAPTION_RE.match(stripped) or ATTACH_CAPTION_RE.match(stripped):
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
    for fig in figures:
        mentions = find_mentions(lines, fig)
        if not mentions:
            continue
        mention_idx = mentions[-1]
        section_title, context_text = build_context(lines, mention_idx, fig.kind)
        if not context_text:
            continue
        samples.append(
            {
                "book_name": BOOK_NAME,
                "anchor_type": "figure_ref_manual" if fig.kind == "normal" else "appendix_figure_ref_manual",
                "section_title": section_title,
                "image_path": fig.image_path,
                "image_caption": fig.caption,
                "figure_id": (f"图{fig.figure_id}" if fig.kind == "normal" else f"附图{fig.figure_id}"),
                "figure_kind": fig.kind,
                "context_text": context_text,
                "mention_lines": [mention_idx],
            }
        )

    deduped: list[dict] = []
    seen: set[str] = set()
    for item in samples:
        key = item["figure_id"]
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
        print(f"section_title: {item['section_title']}")
        print(f"image_caption: {item['image_caption']}")
        preview = item["context_text"][:220].replace("\n", " ")
        print(f"context: {preview}...")


if __name__ == "__main__":
    main()
