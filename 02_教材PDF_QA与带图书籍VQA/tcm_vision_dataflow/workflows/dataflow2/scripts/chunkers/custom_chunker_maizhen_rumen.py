#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
《脉诊入门基础》专用手动切分脚本

规则：
1. 单图：按图号所在章节切分正文。
2. 多图共用一个总图号：拆成子图样本。
3. 子图语义优先从小图标题提取；若没有，则从正文枚举项中提取。
4. 子图上下文优先绑定该脉象自己的正文段落，再补充当前总图中的鉴别句。
5. 输出只写入当前代码目录的 manual_split。
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

BOOK_NAME = "脉诊入门基础"
MD_ROOT = Path(
    r"E:\md\脉诊入门基础 (古聖先賢, 周幸来主编, Xinglai Zhou, 中医医案, 中医学习) (Z-Library).pdf-7ef1a897-caf6-418c-8eff-8ed2a01eff52"
)
MD_PATH = MD_ROOT / "full.md"
OUTPUT_ROOT = Path(
    r"d:\Desktop\Dataflow\Dataflow\DataFlow-main\maizhen_vqa_workdir\output\manual_split"
) / BOOK_NAME
OUTPUT_PATH = OUTPUT_ROOT / "split_samples.jsonl"
IMAGE_OUT_DIR = OUTPUT_ROOT / "images"

IMG_RE = re.compile(r"!\[\]\((images/[^)]+)\)")
TOP_SECTION_RE = re.compile(r"^#\s*(\d+)\.\s*(.+?)\s*$")
HEADER_RE = re.compile(r"^#\s+(.+?)\s*$")
FIG_LINE_RE = re.compile(r"^图\s*(\d+-\d+)\s*(.*)$")
PULSE_LABEL_RE = re.compile(r"([一-龥]{1,8}?脉)")
ENUM_ITEM_RE = re.compile(r"^[①②③④⑤⑥⑦⑧⑨⑩⑪⑫]\s*(.+)$")
LABEL_ALIAS = {
    "花脉": "芤脉",
    "芯脉": "芤脉",
    "乳脉": "芤脉",
    "芋脉": "芤脉",
}


@dataclass
class TopSection:
    title: str
    pulse_label: str
    start: int
    end: int
    clean_text: str


@dataclass
class PendingImage:
    line_idx: int
    image_path: str
    local_caption: str


@dataclass
class FigureGroup:
    figure_id: str
    figure_caption: str
    caption_line_idx: int
    images: list[PendingImage]


def normalize_text(text: str) -> str:
    text = text.replace("\u3000", " ")
    text = text.replace("\r", "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def clean_section_text(lines: list[str], start: int, end: int) -> str:
    selected: list[str] = []
    for idx in range(start, end):
        stripped = lines[idx].strip()
        if not stripped:
            selected.append("")
            continue
        if IMG_RE.search(stripped):
            continue
        if stripped == "图":
            continue
        if FIG_LINE_RE.match(stripped):
            continue
        selected.append(stripped)
    return normalize_text("\n".join(selected))


def extract_first_pulse_label(text: str) -> str:
    if not text:
        return ""
    matches = PULSE_LABEL_RE.findall(text)
    if not matches:
        return ""
    return LABEL_ALIAS.get(matches[0], matches[0])


def extract_top_sections(lines: list[str]) -> list[TopSection]:
    starts: list[tuple[int, str, str]] = []
    for idx, line in enumerate(lines):
        m = TOP_SECTION_RE.match(line.strip())
        if not m:
            continue
        pulse_label = m.group(2).strip()
        starts.append((idx, line.strip(), pulse_label))

    sections: list[TopSection] = []
    for i, (start, title, pulse_label) in enumerate(starts):
        end = starts[i + 1][0] if i + 1 < len(starts) else len(lines)
        sections.append(
            TopSection(
                title=title.lstrip("# ").strip(),
                pulse_label=pulse_label,
                start=start,
                end=end,
                clean_text=clean_section_text(lines, start + 1, end),
            )
        )
    return sections


def extract_local_caption(lines: list[str], image_idx: int) -> str:
    captions: list[str] = []
    for idx in range(image_idx + 1, min(image_idx + 4, len(lines))):
        stripped = lines[idx].strip()
        if not stripped:
            if captions:
                break
            continue
        if IMG_RE.search(stripped) or stripped.startswith("#"):
            break
        if stripped == "图" or FIG_LINE_RE.match(stripped):
            break
        captions.append(stripped)
    return normalize_text(" ".join(captions))


def extract_figure_groups(lines: list[str]) -> list[FigureGroup]:
    groups: list[FigureGroup] = []
    pending: list[PendingImage] = []
    idx = 0
    while idx < len(lines):
        stripped = lines[idx].strip()
        if HEADER_RE.match(stripped) and pending:
            pending.clear()

        img_match = IMG_RE.search(stripped)
        if img_match:
            pending.append(
                PendingImage(
                    line_idx=idx,
                    image_path=str((MD_ROOT / img_match.group(1)).resolve()),
                    local_caption=extract_local_caption(lines, idx),
                )
            )
            idx += 1
            continue

        if stripped == "图":
            next_line = lines[idx + 1].strip() if idx + 1 < len(lines) else ""
            m = FIG_LINE_RE.match(f"图 {next_line}")
            if m and pending:
                groups.append(
                    FigureGroup(
                        figure_id=m.group(1),
                        figure_caption=f"图{m.group(1)} {m.group(2)}".strip(),
                        caption_line_idx=idx,
                        images=pending[:],
                    )
                )
                pending.clear()
                idx += 2
                continue

        m = FIG_LINE_RE.match(stripped)
        if m and pending:
            groups.append(
                FigureGroup(
                    figure_id=m.group(1),
                    figure_caption=f"图{m.group(1)} {m.group(2)}".strip(),
                    caption_line_idx=idx,
                    images=pending[:],
                )
            )
            pending.clear()
        idx += 1
    return groups


def find_section_for_line(sections: list[TopSection], line_idx: int) -> TopSection | None:
    for section in sections:
        if section.start <= line_idx < section.end:
            return section
    return None


def label_from_local_caption(text: str) -> str:
    if not text:
        return ""
    label = extract_first_pulse_label(text)
    if label:
        return label
    text = text.replace("示意图", "").strip()
    return LABEL_ALIAS.get(text, text)


def extract_enum_lines(lines: list[str], start: int, end: int) -> list[str]:
    out: list[str] = []
    for idx in range(start, end):
        stripped = lines[idx].strip()
        if not stripped:
            continue
        m = ENUM_ITEM_RE.match(stripped)
        if m:
            out.append(m.group(1).strip())
    return out


def infer_subfigure_labels(group: FigureGroup, section: TopSection | None, lines: list[str]) -> list[str]:
    labels = [label_from_local_caption(img.local_caption) for img in group.images]
    if all(labels):
        return labels

    search_start = section.start if section else max(0, group.images[0].line_idx - 60)
    enum_lines = extract_enum_lines(lines, search_start, group.caption_line_idx)
    enum_labels: list[str] = []
    for item in enum_lines:
        label = extract_first_pulse_label(item)
        if label:
            enum_labels.append(label)

    if len(enum_labels) >= len(group.images):
        return enum_labels[-len(group.images):]

    if section and section.pulse_label and len(group.images) == 1:
        return [section.pulse_label]

    main = ""
    m = re.search(r"图\s*\d+-\d+\s*([一-龥]{1,12}脉)", group.figure_caption)
    if m:
        main = m.group(1)
    fallback = [main] if main else []
    while len(fallback) < len(group.images):
        fallback.append(f"{main or '子图'}{len(fallback)+1}")
    return fallback


def collect_compare_lines(group_section: TopSection | None, lines: list[str], end_line: int) -> dict[str, str]:
    if not group_section:
        return {}
    enum_lines = extract_enum_lines(lines, group_section.start, end_line)
    mapping: dict[str, str] = {}
    for item in enum_lines:
        label = extract_first_pulse_label(item)
        if not label:
            continue
        mapping[label] = item
    return mapping


def build_context_for_label(
    label: str,
    label_sections: dict[str, TopSection],
    compare_line: str,
    group_section: TopSection | None,
) -> str:
    parts: list[str] = []
    label_section = label_sections.get(label)
    if label_section:
        parts.append(label_section.title)
        parts.append(label_section.clean_text)
    elif group_section:
        parts.append(group_section.title)
        parts.append(group_section.clean_text)
    if compare_line and compare_line not in "\n".join(parts):
        parts.append(compare_line)
    return normalize_text("\n\n".join(p for p in parts if p))


def copy_images_and_rewrite_paths(samples: list[dict]) -> list[dict]:
    IMAGE_OUT_DIR.mkdir(parents=True, exist_ok=True)
    keep_names = {
        Path(item["image_path"]).name for item in samples if Path(item["image_path"]).exists()
    }
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
        new_item = dict(item)
        new_item["image_path"] = copied[item["image_path"]]
        rewritten.append(new_item)
    return rewritten


def extract_samples(md_path: Path) -> list[dict]:
    lines = md_path.read_text(encoding="utf-8").splitlines()
    sections = extract_top_sections(lines)
    label_sections = {s.pulse_label: s for s in sections if s.pulse_label}
    groups = extract_figure_groups(lines)

    samples: list[dict] = []
    for group in groups:
        section = find_section_for_line(sections, group.caption_line_idx)
        labels = infer_subfigure_labels(group, section, lines)
        compare_lines = collect_compare_lines(section, lines, group.caption_line_idx)

        if len(group.images) == 1:
            context_text = section.clean_text if section else ""
            if not context_text:
                continue
            samples.append(
                {
                    "book_name": BOOK_NAME,
                    "anchor_type": "figure_ref_manual",
                    "section_title": section.title if section else "",
                    "image_path": group.images[0].image_path,
                    "image_caption": group.figure_caption,
                    "figure_id": group.figure_id,
                    "context_text": context_text,
                    "mention_lines": [group.caption_line_idx],
                }
            )
            continue

        for idx, image in enumerate(group.images):
            label = labels[idx] if idx < len(labels) else f"子图{idx+1}"
            compare_line = compare_lines.get(label, "")
            context_text = build_context_for_label(label, label_sections, compare_line, section)
            if not context_text:
                continue
            samples.append(
                {
                    "book_name": BOOK_NAME,
                    "anchor_type": "figure_ref_subimage_manual",
                    "section_title": label_sections.get(label).title if label in label_sections else (section.title if section else ""),
                    "image_path": image.image_path,
                    "image_caption": image.local_caption or f"{label}示意图",
                    "figure_id": f"{group.figure_id}-{idx+1}",
                    "parent_figure_id": group.figure_id,
                    "subfigure_label": label,
                    "context_text": context_text,
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
