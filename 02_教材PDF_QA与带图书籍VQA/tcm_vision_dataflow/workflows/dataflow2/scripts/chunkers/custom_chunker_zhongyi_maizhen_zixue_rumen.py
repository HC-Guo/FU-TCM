#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
《中医脉诊自学入门》专用切分脚本

规则：
1. 前 1-5 章：在正文中找到与图号完全匹配的引用，只截取该引用所在段落。
2. 后续章节：一个总图包含多个子图。
   - section_title = 上一个大标题（如 1.XXX / 一、XXX）
   - context_text = 大标题 + 当前子图对应的小标题块
   - 小标题按顺序匹配 (1)/(2)/... 这类真正的“脉名标题”
   - 像“(1)主病”“(2)特征”这类 OCR 噪声不当作新的子图标题
3. 输出仅写到当前工作目录 output/manual_split/中医脉诊自学入门
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path


BOOK_NAME = "中医脉诊自学入门"
MD_ROOT = Path(
    r"E:\md\中医脉诊自学入门 (古聖先賢, 周幸来主编 周绩副主编, 周举, 姜衰芳[等]编著, 中華傳統文化, 中医医案 etc.) (Z-Library).pdf-1130d4b7-3c1b-4b8c-accc-f919e206ec7b"
)
MD_PATH = MD_ROOT / "full.md"
OUTPUT_ROOT = Path(
    r"d:\Desktop\Dataflow\Dataflow\DataFlow-main\maizhen_vqa_workdir\output\manual_split"
) / BOOK_NAME
OUTPUT_PATH = OUTPUT_ROOT / "split_samples.jsonl"
IMAGE_OUT_DIR = OUTPUT_ROOT / "images"

IMG_RE = re.compile(r"!\[\]\((images/[^)]+)\)")
FIG_LINE_RE = re.compile(r"^图\s*(\d+-\d+)\s*(.*)$")
BIG_HEAD_RE = re.compile(r"^(?:#\s*)?(\d+)\.\s*(.+)$")
CN_HEAD_RE = re.compile(r"^(?:#\s*)?([一二三四五六七八九十]+)、\s*(.+)$")
SUB_HEAD_RE = re.compile(r"^(?:#\s*)?[（(](\d+)[)）]\s*(.+)$")
ITEM_HEAD_RE = re.compile(r"^([①②③④⑤⑥⑦⑧⑨⑩])\s*(.+)$")
CHAPTER_RE = re.compile(r"^#\s*第([一二三四五六七八九十]+)章")

CN_NUM = {
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
BODY_PREFIXES = ("主病", "特征", "脉理", "鉴别", "兼脉", "按语", "说明")
PARENT_PREFIXES = ("脉理", "鉴别", "兼脉", "主病", "特征")
ITEM_ORDER = "①②③④⑤⑥⑦⑧⑨⑩"


def circled_from_num(num: int) -> str:
    return ITEM_ORDER[num - 1] if 1 <= num <= len(ITEM_ORDER) else str(num)


@dataclass
class FigureGroup:
    figure_id: str
    caption: str
    caption_line_idx: int
    image_paths: list[str]


@dataclass
class TextBlock:
    title: str
    start: int
    end: int
    text: str


def normalize_text(text: str) -> str:
    text = text.replace("\u3000", " ").replace("\r", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_hash(text: str) -> str:
    return re.sub(r"^#\s*", "", text.strip())


def chapter_num_from_line(text: str) -> int | None:
    m = CHAPTER_RE.match(text.strip())
    if not m:
        return None
    return CN_NUM.get(m.group(1))


def figure_chapter_num(figure_id: str) -> int | None:
    m = re.match(r"^(\d+)-", figure_id)
    return int(m.group(1)) if m else None


def figure_index_num(figure_id: str) -> int | None:
    m = re.match(r"^\d+-(\d+)$", figure_id)
    return int(m.group(1)) if m else None


def is_big_heading(text: str) -> bool:
    stripped = text.strip()
    return bool(BIG_HEAD_RE.match(stripped) or CN_HEAD_RE.match(stripped))


def clean_range(lines: list[str], start: int, end: int, drop_figure_lines: bool = True) -> str:
    out: list[str] = []
    for idx in range(start, end):
        stripped = lines[idx].strip()
        if not stripped:
            out.append("")
            continue
        if IMG_RE.search(stripped):
            continue
        if drop_figure_lines and FIG_LINE_RE.match(stripped):
            continue
        out.append(strip_hash(stripped))
    return normalize_text("\n".join(out))


def is_true_sub_heading(text: str) -> tuple[bool, str]:
    m = SUB_HEAD_RE.match(text.strip())
    if not m:
        return False, ""
    title = re.sub(r"\s+", "", m.group(2))
    if not title:
        return False, ""
    if title.startswith(BODY_PREFIXES):
        return False, ""
    if any(ch in title for ch in "。；：，、;:,."):
        return False, ""
    if len(title) > 12:
        return False, ""
    if "脉" not in title:
        return False, ""
    return True, title


def is_parent_sub_heading(text: str) -> tuple[bool, str]:
    m = SUB_HEAD_RE.match(text.strip())
    if not m:
        return False, ""
    title = re.sub(r"\s+", "", m.group(2))
    title = re.split(r"[：:]", title, maxsplit=1)[0]
    title = re.sub(r"[。；;，,]+$", "", title)
    if not title:
        return False, ""
    if not title.startswith(PARENT_PREFIXES):
        return False, ""
    if len(title) > 16:
        return False, ""
    return True, title


def collect_caption_image_block(lines: list[str], caption_idx: int) -> list[str]:
    image_paths: list[str] = []
    seen_image = False
    idx = caption_idx - 1
    while idx >= 0:
        stripped = lines[idx].strip()
        if not stripped:
            idx -= 1
            continue
        if stripped.startswith("#"):
            break
        if FIG_LINE_RE.match(stripped):
            break
        if seen_image and is_big_heading(stripped):
            break

        img_match = IMG_RE.search(stripped)
        if img_match:
            image_paths.append(str((MD_ROOT / img_match.group(1)).resolve()))
            seen_image = True
            idx -= 1
            continue

        if not seen_image:
            idx -= 1
            continue

        normalized = re.sub(r"\s+", "", stripped)
        is_short_label = (
            len(normalized) <= 24
            and not any(p in normalized for p in ("。", "；", "：", ";", ":"))
        )
        if not is_short_label:
            break
        idx -= 1

    image_paths.reverse()
    return image_paths


def extract_figure_groups(lines: list[str]) -> list[FigureGroup]:
    groups: list[FigureGroup] = []
    for idx, line in enumerate(lines):
        stripped = line.strip()

        if stripped == "图" and idx + 1 < len(lines):
            merged = f"图{lines[idx + 1].strip()}"
            m = FIG_LINE_RE.match(merged)
            if m:
                image_paths = collect_caption_image_block(lines, idx)
                if image_paths:
                    groups.append(
                        FigureGroup(
                            figure_id=m.group(1),
                            caption=f"图{m.group(1)} {m.group(2)}".strip(),
                            caption_line_idx=idx,
                            image_paths=image_paths,
                        )
                    )
            continue

        m = FIG_LINE_RE.match(stripped)
        if not m:
            continue
        image_paths = collect_caption_image_block(lines, idx)
        if not image_paths:
            continue
        groups.append(
            FigureGroup(
                figure_id=m.group(1),
                caption=f"图{m.group(1)} {m.group(2)}".strip(),
                caption_line_idx=idx,
                image_paths=image_paths,
            )
        )
    return groups


def nearest_ref_paragraph(lines: list[str], figure_id: str) -> tuple[int, str] | None:
    pattern = re.compile(rf"图\s*{re.escape(figure_id)}(?![-\d])")
    hits: list[int] = []
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if IMG_RE.search(stripped):
            continue
        if FIG_LINE_RE.match(stripped):
            continue
        if pattern.search(stripped):
            hits.append(idx)
    if not hits:
        return None

    ref_idx = hits[-1]
    start = ref_idx
    end = ref_idx + 1
    while start > 0 and lines[start - 1].strip() and not lines[start - 1].strip().startswith("#"):
        if IMG_RE.search(lines[start - 1].strip()):
            break
        start -= 1
    while end < len(lines) and lines[end].strip() and not lines[end].strip().startswith("#"):
        if IMG_RE.search(lines[end].strip()):
            break
        end += 1
    return ref_idx, normalize_text("\n".join(lines[i].strip() for i in range(start, end)))


def surrounding_header(lines: list[str], ref_idx: int) -> str:
    for idx in range(ref_idx, -1, -1):
        stripped = lines[idx].strip()
        if stripped.startswith("#"):
            return strip_hash(stripped)
    return ""


def previous_big_heading(lines: list[str], idx: int) -> tuple[int, str] | None:
    for i in range(idx, -1, -1):
        stripped = lines[i].strip()
        if is_big_heading(stripped):
            return i, strip_hash(stripped)
    return None


def next_big_heading(lines: list[str], idx: int) -> int:
    for i in range(idx + 1, len(lines)):
        if is_big_heading(lines[i].strip()):
            return i
    return len(lines)


def collect_sub_blocks(lines: list[str], start: int, end: int) -> list[TextBlock]:
    starts: list[tuple[int, str]] = []
    for idx in range(start, end):
        ok, title = is_true_sub_heading(lines[idx].strip())
        if ok:
            starts.append((idx, title))

    blocks: list[TextBlock] = []
    for i, (line_idx, title) in enumerate(starts):
        block_end = starts[i + 1][0] if i + 1 < len(starts) else end
        text = clean_range(lines, line_idx, block_end)
        blocks.append(TextBlock(title=title, start=line_idx, end=block_end, text=text))
    return blocks


def find_parent_sub_range(
    lines: list[str], section_start: int, caption_line: int, section_end: int, image_count: int
) -> tuple[int, int, str] | None:
    starts: list[tuple[int, str]] = []
    for idx in range(section_start, section_end):
        ok, title = is_parent_sub_heading(lines[idx].strip())
        if ok:
            starts.append((idx, title))
    if not starts:
        return None

    candidates: list[tuple[int, int, str, int, int]] = []
    for i, (start_idx, start_title) in enumerate(starts):
        end_idx = starts[i + 1][0] if i + 1 < len(starts) else section_end
        item_count = len(collect_item_blocks(lines, start_idx + 1, end_idx))
        if start_idx <= caption_line <= end_idx:
            distance = 0
        elif caption_line < start_idx:
            distance = start_idx - caption_line
        else:
            distance = caption_line - end_idx
        score = 0
        if item_count >= image_count:
            score += 100
        elif item_count > 0:
            score += 50 + item_count
        if start_idx <= caption_line <= end_idx:
            score += 20
        score -= distance // 20
        candidates.append((score, start_idx, end_idx, start_title, item_count))

    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[0], x[4], -x[1]), reverse=True)
    _, start_idx, end_idx, start_title, _ = candidates[0]
    return start_idx, end_idx, start_title


def collect_item_blocks(lines: list[str], start: int, end: int) -> list[TextBlock]:
    starts: list[tuple[int, str]] = []
    for idx in range(start, end):
        stripped = lines[idx].strip()
        title = ""
        m = re.match(r"^\$?\s*([①②③④⑤⑥⑦⑧⑨⑩])\s*\$?\s*(.+)$", stripped)
        if m:
            title = normalize_text(f"{m.group(1)}{m.group(2)}")
        else:
            m2 = re.match(r"^[（(]\s*(\d+)\s*[)）]\s*(.+)$", stripped)
            if m2:
                num = int(m2.group(1))
                body = normalize_text(m2.group(2))
                if 1 <= num <= 10 and body:
                    title = f"{circled_from_num(num)}{body}"
        if not title:
            continue
        starts.append((idx, title))

    blocks: list[TextBlock] = []
    for i, (line_idx, title) in enumerate(starts):
        block_end = starts[i + 1][0] if i + 1 < len(starts) else end
        text = clean_range(lines, line_idx, block_end)
        blocks.append(TextBlock(title=title, start=line_idx, end=block_end, text=text))
    return blocks


def extract_sub_blocks(
    lines: list[str],
    section_start: int,
    caption_line: int,
    section_end: int,
    image_count: int,
) -> list[TextBlock]:
    post_blocks = collect_sub_blocks(lines, caption_line + 1, section_end)
    if len(post_blocks) >= image_count:
        return post_blocks[:image_count]

    all_blocks = collect_sub_blocks(lines, section_start, section_end)
    nearby_blocks = [b for b in all_blocks if b.start > caption_line - 120]
    if len(nearby_blocks) >= image_count:
        return nearby_blocks[:image_count]

    return post_blocks or nearby_blocks


def extract_item_blocks_after_subheading(
    lines: list[str],
    section_start: int,
    caption_line: int,
    section_end: int,
    image_count: int,
) -> tuple[str, list[TextBlock]] | None:
    parent_range = find_parent_sub_range(lines, section_start, caption_line, section_end, image_count)
    if not parent_range:
        return None
    parent_start, parent_end, parent_title = parent_range
    item_blocks = collect_item_blocks(lines, parent_start + 1, parent_end)
    if not item_blocks:
        return None
    if len(item_blocks) > image_count:
        item_blocks = item_blocks[:image_count]
    return parent_title, item_blocks


def pre_caption_labels(lines: list[str], start: int, end: int) -> list[str]:
    labels: list[str] = []
    for idx in range(start, end):
        stripped = lines[idx].strip()
        if not stripped:
            continue
        if IMG_RE.search(stripped):
            continue
        if FIG_LINE_RE.match(stripped):
            continue
        if is_big_heading(stripped) or stripped.startswith("#"):
            continue
        normalized = re.sub(r"\s+", "", stripped)
        if len(normalized) <= 24 and not any(p in normalized for p in ("。", "；", "：", ";", ":")):
            labels.append(stripped)
    return labels


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
        row = dict(item)
        if src.exists():
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
    groups = extract_figure_groups(lines)

    samples: list[dict] = []
    for group in groups:
        chap_num = figure_chapter_num(group.figure_id)
        if chap_num is None:
            continue
        fig_num = figure_index_num(group.figure_id)

        if chap_num <= 5:
            ref = nearest_ref_paragraph(lines, group.figure_id)
            if not ref:
                continue
            ref_idx, context_text = ref
            samples.append(
                {
                    "book_name": BOOK_NAME,
                    "anchor_type": "figure_ref_manual",
                    "section_title": surrounding_header(lines, ref_idx),
                    "image_path": group.image_paths[-1],
                    "image_caption": group.caption,
                    "figure_id": group.figure_id,
                    "context_text": context_text,
                    "mention_lines": [ref_idx],
                }
            )
            continue

        prev_big = previous_big_heading(lines, group.caption_line_idx)
        if not prev_big:
            continue
        big_start, big_title = prev_big
        big_end = next_big_heading(lines, big_start)
        use_item_mode = chap_num == 6 and fig_num is not None and fig_num >= 7
        image_paths = list(group.image_paths)
        labels = pre_caption_labels(lines, big_start, group.caption_line_idx)

        parent_title = ""
        item_blocks: list[TextBlock] = []
        sub_blocks: list[TextBlock] = []
        if use_item_mode:
            item_mode = extract_item_blocks_after_subheading(
                lines, big_start, group.caption_line_idx, big_end, len(image_paths)
            )
            if item_mode:
                parent_title, item_blocks = item_mode
        if not item_blocks:
            sub_blocks = extract_sub_blocks(lines, big_start, group.caption_line_idx, big_end, len(image_paths))

        expected_count = len(item_blocks) or len(sub_blocks) or len(labels)
        if expected_count and len(image_paths) > expected_count:
            image_paths = image_paths[-expected_count:]

        for idx, image_path in enumerate(image_paths):
            block = item_blocks[idx] if idx < len(item_blocks) else (
                sub_blocks[idx] if idx < len(sub_blocks) else None
            )
            if block:
                subsection_title = block.title
                subsection_text = block.text
                if item_blocks and parent_title:
                    context_text = normalize_text(f"{big_title}\n\n({parent_title})\n\n{block.text}")
                else:
                    context_text = normalize_text(f"{big_title}\n\n{block.text}")
            else:
                subsection_title = labels[idx] if idx < len(labels) else f"子图{idx + 1}"
                subsection_text = subsection_title
                if parent_title:
                    context_text = normalize_text(f"{big_title}\n\n({parent_title})")
                else:
                    context_text = normalize_text(big_title)

            samples.append(
                {
                    "book_name": BOOK_NAME,
                    "anchor_type": "figure_ref_subimage_manual",
                    "section_title": big_title,
                    "subsection_title": subsection_title,
                    "image_path": image_path,
                    "image_caption": subsection_title,
                    "figure_id": f"{group.figure_id}-{idx + 1}" if len(image_paths) > 1 else group.figure_id,
                    "parent_figure_id": group.figure_id,
                    "context_text": context_text,
                    "subsection_text": subsection_text,
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


if __name__ == "__main__":
    main()
