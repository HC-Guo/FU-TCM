#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
《中医脉诊一点通（修订版）》专用切分脚本

规则：
1. 本书图片没有图号，按上下文标题结构切分。
2. 如果图片位于某个“脉象类标题”之下：
   - 上文从该脉象类标题开始保留
   - 下文遇到标题时，若该标题是“主病”或“专家提示”，则继续向下找
   - 再遇到下一个标题时截止，并舍去该标题
3. 如果图片落在“特效方”区域内，直接丢弃。
4. 如果图片落在“按摩方”区域内：
   - “按摩方”标题本身不要
   - 保留其下的按摩法标题（如“按揉风池穴”）
   - 下文遇到下一个标题即截止并舍去
5. 其他普通图片：按最近上级标题到下一个标题切分。
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from PIL import Image


BOOK_NAME = "中医脉诊一点通（修订版）"
MD_ROOT = Path(
    r"E:\md\中医脉诊一点通（修订版） (王桂茂) (Z-Library).pdf-69deb583-6b99-480d-9053-a49b58f49517"
)
MD_PATH = MD_ROOT / "full.md"
OUTPUT_ROOT = Path(
    r"d:\Desktop\Dataflow\Dataflow\DataFlow-main\maizhen_vqa_workdir\output\manual_split"
) / BOOK_NAME
OUTPUT_PATH = OUTPUT_ROOT / "split_samples.jsonl"
IMAGE_OUT_DIR = OUTPUT_ROOT / "images"
MIN_IMAGE_WIDTH = 120
MIN_IMAGE_HEIGHT = 120

IMG_RE = re.compile(r"!\[\]\((images/[^)]+)\)")
HEADING_RE = re.compile(r"^#+\s*(.+?)\s*$")

PULSE_NAMES = (
    "平脉", "浮脉", "沉脉", "迟脉", "数脉", "虚脉", "实脉", "洪脉", "细脉", "滑脉",
    "涩脉", "弦脉", "紧脉", "缓脉", "短脉", "长脉", "动脉", "促脉", "结脉", "代脉",
    "疾脉", "伏脉", "牢脉", "弱脉", "濡脉", "芤脉", "革脉", "散脉", "微脉",
)
PASS_THROUGH_HEADINGS = {"主病", "专家提示"}
PLAIN_HEADINGS = {"特效方", "按摩方", "主病", "专家提示", "脉象解析", "寸口三部脉象", "对应的健康问题", "延伸辨证及确诊"}
DROP_SECTION_TITLES = {"中医入门随手查通 【修订版】王桂茂主编", "你想看的书 我这里都有 □", "东东电子书"}
TITLE_TE_XIAO_FANG = "\u7279\u6548\u65b9"
TITLE_AN_MO_FANG = "\u6309\u6469\u65b9"


@dataclass
class Heading:
    line_idx: int
    text: str


def normalize_text(text: str) -> str:
    text = text.replace("\u3000", " ").replace("\r", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_hash(text: str) -> str:
    return re.sub(r"^#+\s*", "", text.strip())


def is_heading(text: str) -> bool:
    stripped = text.strip()
    return bool(HEADING_RE.match(stripped) or stripped in PLAIN_HEADINGS)


def heading_text(text: str) -> str:
    stripped = text.strip()
    m = HEADING_RE.match(stripped)
    if m:
        return m.group(1).strip()
    if stripped in PLAIN_HEADINGS:
        return stripped
    return ""


def is_pulse_heading(text: str) -> bool:
    clean = heading_text(text) if is_heading(text) else text.strip()
    return any(clean.startswith(name) for name in PULSE_NAMES)


def parse_headings(lines: list[str]) -> list[Heading]:
    out: list[Heading] = []
    for idx, line in enumerate(lines):
        if is_heading(line):
            text = heading_text(line)
            if text and text != "#":
                out.append(Heading(idx, text))
    return out


def previous_heading_idx(headings: list[Heading], idx: int) -> int | None:
    for i in range(len(headings) - 1, -1, -1):
        if headings[i].line_idx < idx:
            return i
    return None


def next_heading_line(headings: list[Heading], idx: int) -> int:
    for h in headings:
        if h.line_idx > idx:
            return h.line_idx
    return -1


def clean_range(lines: list[str], start: int, end: int) -> str:
    out: list[str] = []
    for i in range(start, end):
        stripped = lines[i].strip()
        if IMG_RE.search(stripped):
            continue
        if is_heading(stripped):
            out.append(strip_hash(stripped))
            continue
        out.append(stripped)
    return normalize_text("\n".join(out))


def find_last_pulse_heading(headings: list[Heading], prev_idx: int | None) -> int | None:
    if prev_idx is None:
        return None
    for i in range(prev_idx, -1, -1):
        if any(headings[j].text == TITLE_TE_XIAO_FANG for j in range(i + 1, prev_idx + 1)):
            break
        if any(headings[j].text == TITLE_AN_MO_FANG for j in range(i + 1, prev_idx + 1)):
            break
        if is_pulse_heading(headings[i].text):
            return i
    return None


def image_in_special_zone(headings: list[Heading], prev_idx: int | None, special: str) -> int | None:
    if prev_idx is None:
        return None
    for i in range(prev_idx, -1, -1):
        if headings[i].text == special:
            return i
        if is_pulse_heading(headings[i].text):
            return None
    return None


def massage_method_heading(headings: list[Heading], massage_idx: int, prev_idx: int) -> int | None:
    for i in range(prev_idx, massage_idx, -1):
        if headings[i].text != TITLE_AN_MO_FANG:
            return i
    return None


def pulse_mode_end(headings: list[Heading], image_idx: int, total_lines: int) -> int:
    seen_first = False
    for h in headings:
        if h.line_idx <= image_idx:
            continue
        if not seen_first:
            if h.text in PASS_THROUGH_HEADINGS:
                seen_first = True
                continue
            return h.line_idx
        return h.line_idx
    return total_lines


def general_mode_end(headings: list[Heading], image_idx: int, total_lines: int) -> int:
    nxt = next_heading_line(headings, image_idx)
    return nxt if nxt != -1 else total_lines


def looks_like_recipe_context(text: str) -> bool:
    compact = re.sub(r"\s+", "", text)
    recipe_terms = ("原料", "做法", "用法")
    hits = sum(1 for term in recipe_terms if term in compact)
    return hits >= 2


def image_size_ok(image_path: Path) -> bool:
    try:
        with Image.open(image_path) as img:
            return img.width >= MIN_IMAGE_WIDTH and img.height >= MIN_IMAGE_HEIGHT
    except OSError:
        return False


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
            continue
        row = dict(item)
        if item["image_path"] not in copied:
            dst = IMAGE_OUT_DIR / src.name
            if not dst.exists():
                shutil.copy2(src, dst)
            copied[item["image_path"]] = str(dst)
        row["image_path"] = copied[item["image_path"]]
        rewritten.append(row)
    return rewritten


def extract_samples(md_path: Path) -> list[dict]:
    lines = md_path.read_text(encoding="utf-8").splitlines()
    headings = parse_headings(lines)
    samples: list[dict] = []
    image_no = 0

    for idx, line in enumerate(lines):
        m = IMG_RE.search(line.strip())
        if not m:
            continue
        image_no += 1
        image_path = (MD_ROOT / m.group(1)).resolve()
        if not image_size_ok(image_path):
            continue
        prev_idx = previous_heading_idx(headings, idx)
        if prev_idx is None:
            continue

        ma_idx = image_in_special_zone(headings, prev_idx, TITLE_AN_MO_FANG)
        if ma_idx is not None:
            method_idx = massage_method_heading(headings, ma_idx, prev_idx)
            if method_idx is None:
                continue
            start = headings[method_idx].line_idx
            end = general_mode_end(headings, idx, len(lines))
            section_title = headings[method_idx].text
            anchor_type = "heading_context_manual"
        else:
            te_idx = image_in_special_zone(headings, prev_idx, TITLE_TE_XIAO_FANG)
            if te_idx is not None:
                continue
            pulse_idx = find_last_pulse_heading(headings, prev_idx)
            if pulse_idx is not None:
                start = headings[pulse_idx].line_idx
                end = pulse_mode_end(headings, idx, len(lines))
                section_title = headings[pulse_idx].text
                anchor_type = "heading_context_manual"
            else:
                start = headings[prev_idx].line_idx
                end = general_mode_end(headings, idx, len(lines))
                section_title = headings[prev_idx].text
                anchor_type = "heading_context_manual"

        context_text = clean_range(lines, start, end)
        if not context_text:
            continue
        if section_title in DROP_SECTION_TITLES:
            continue
        if looks_like_recipe_context(context_text):
            continue

        samples.append(
            {
                "book_name": BOOK_NAME,
                "anchor_type": anchor_type,
                "section_title": section_title,
                "image_path": str(image_path),
                "image_caption": "",
                "figure_id": f"img-{image_no:04d}",
                "context_text": context_text,
                "mention_lines": [idx],
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


if __name__ == "__main__":
    main()
