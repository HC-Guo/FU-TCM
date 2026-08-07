#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
《当代中医脉诊精华手册》专用手动切分脚本

按用户确认的规则切分：
1. 不是所有图片都要
2. 只保留“带图号”的图片
3. 必须能在正文中找到对应图号引用
4. 同一图号在正文中可能出现多次，但通常彼此很近，应合并为一个上下文块
5. 向上/向下收集上下文时，遇到相邻标题边界则截断
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

BOOK_NAME = "当代中医脉诊精华手册"
MD_ROOT = Path(r"E:\md\当代中医脉诊精华手册 (里昂·汉默) (Z-Library).pdf-5fc34d1b-0f0e-4adf-ae39-5c0278bc5aa4")
MD_PATH = MD_ROOT / "full.md"
OUTPUT_ROOT = Path(r"E:\output_maizhen") / BOOK_NAME
OUTPUT_PATH = OUTPUT_ROOT / "split_samples.jsonl"
IMAGE_OUT_DIR = OUTPUT_ROOT / "images"
CODE_OUTPUT_ROOT = Path(r"d:\Desktop\Dataflow\Dataflow\DataFlow-main\maizhen_vqa_workdir\output\manual_split") / BOOK_NAME
CODE_OUTPUT_PATH = CODE_OUTPUT_ROOT / "split_samples.jsonl"
CODE_IMAGE_OUT_DIR = CODE_OUTPUT_ROOT / "images"

IMG_RE = re.compile(r"!\[\]\((images/[^)]+)\)")
HEADER_RE = re.compile(r"^(#)\s+(.+)$")
FIGURE_CAPTION_RE = re.compile(r"^(图\s*([0-9A-Za-z]+(?:[-－—\.][0-9A-Za-z]+)*).*)$")
FIGURE_REF_RE = re.compile(r"图\s*([0-9A-Za-z]+(?:[-－—\.][0-9A-Za-z]+)*)")
CHAPTER_RE = re.compile(r"^第\s*\d+\s*章")
CASE_RE = re.compile(r"^病例\s*\d+[:：]")

NOISE_TITLE_HINTS = [
    "图书在版编目",
    "协编",
    "译者",
    "审阅",
    "原作者中文版序",
    "陈序",
    "译者序",
    "如何学习沈一汉默氏脉诊",
    "导论",
    "目录",
    "后记",
    "注解",
    "名词解释",
    "参考书目",
    "附录",
    "索引",
]

DROP_FIGURE_IDS = {
    "6-4a",
    "6-4b",
    "6-5",
    "7-1a",
    "7-1b",
    "7-2",
}

PRESERVE_EXISTING_FIGURE_IDS = {
    "2-1",
    "2-2",
}


@dataclass
class FigureEntry:
    line_idx: int
    image_path: str
    caption: str
    figure_id: str
    section_title: str
    chapter_num: int | None


def normalize_text(text: str) -> str:
    text = text.replace("\u3000", " ")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def normalize_figure_id(raw: str) -> str:
    return raw.replace("－", "-").replace("—", "-").replace(".", "-").replace(" ", "").strip()


def extract_chapter_num_from_figure_id(figure_id: str) -> int | None:
    m = re.match(r"^(\d+)-", figure_id)
    if not m:
        return None
    return int(m.group(1))


def is_standalone_figure_caption_line(text: str) -> bool:
    return bool(FIGURE_CAPTION_RE.match(text.strip()))


def is_noise_title(title: str) -> bool:
    return any(h in (title or "") for h in NOISE_TITLE_HINTS)


def header_positions(lines: list[str]) -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []
    for idx, line in enumerate(lines):
        m = HEADER_RE.match(line.strip())
        if m:
            out.append((idx, m.group(2).strip()))
    return out


def choose_section_title(headers: list[tuple[int, str]], line_idx: int) -> str:
    candidates = [title for pos, title in headers if pos < line_idx and not is_noise_title(title)]
    return candidates[-1] if candidates else ""


def chapter_started(headers: list[tuple[int, str]], line_idx: int) -> bool:
    return any(CHAPTER_RE.search(title) for pos, title in headers if pos < line_idx)


def extract_figures(lines: list[str], headers: list[tuple[int, str]]) -> list[FigureEntry]:
    figures: list[FigureEntry] = []
    for idx, line in enumerate(lines):
        img_match = IMG_RE.search(line.strip())
        if not img_match:
            continue
        if not chapter_started(headers, idx):
            continue

        caption = ""
        figure_id = ""
        for offset in range(1, 4):
            nxt = idx + offset
            if nxt >= len(lines):
                break
            text = lines[nxt].strip()
            if not text:
                continue
            if text.startswith("# "):
                break
            cap_match = FIGURE_CAPTION_RE.match(text)
            if cap_match:
                caption = cap_match.group(1).strip()
                figure_id = normalize_figure_id(cap_match.group(2))
                break

        if not figure_id:
            continue

        section_title = choose_section_title(headers, idx)
        if not section_title or is_noise_title(section_title):
            continue

        figures.append(
            FigureEntry(
                line_idx=idx,
                image_path=str((MD_ROOT / img_match.group(1)).resolve()),
                caption=caption,
                figure_id=figure_id,
                section_title=section_title,
                chapter_num=extract_chapter_num_from_figure_id(figure_id),
            )
        )
    return figures


def collect_figure_mentions(lines: list[str], figure_id: str, strict_mode: bool) -> list[int]:
    mentions: list[int] = []
    pattern = re.compile(rf"图\s*{re.escape(figure_id).replace('\\-', '[-－—\\.]')}")
    for idx, line in enumerate(lines):
        text = line.strip()
        if not text or text.startswith("# "):
            continue
        if IMG_RE.search(text):
            continue
        if strict_mode and is_standalone_figure_caption_line(text):
            continue
        if pattern.search(text):
            mentions.append(idx)
    return mentions


def has_non_caption_figure_reference(lines: list[str], mention_lines: list[int]) -> bool:
    for idx in mention_lines:
        text = lines[idx].strip()
        if not is_standalone_figure_caption_line(text):
            return True
    return False


def nearest_figure_clusters(mentions: list[int], figure_line_idx: int, gap: int = 6) -> list[int]:
    if not mentions:
        return []
    mentions = sorted(mentions, key=lambda x: abs(x - figure_line_idx))
    seed = mentions[0]
    cluster = [seed]
    left = seed
    right = seed
    remaining = sorted(set(mentions[1:]))

    changed = True
    while changed:
        changed = False
        for m in remaining[:]:
            if abs(m - left) <= gap or abs(m - right) <= gap:
                cluster.append(m)
                left = min(left, m)
                right = max(right, m)
                remaining.remove(m)
                changed = True
    return sorted(cluster)


def previous_header(headers: list[tuple[int, str]], line_idx: int) -> tuple[int, str] | None:
    prev = [item for item in headers if item[0] < line_idx]
    return prev[-1] if prev else None


def next_header(headers: list[tuple[int, str]], line_idx: int) -> tuple[int, str] | None:
    nxt = [item for item in headers if item[0] > line_idx]
    return nxt[0] if nxt else None


def is_chapter4_special_case(figure_id: str) -> bool:
    m = re.match(r"^4-(\d+)", figure_id)
    return bool(m and int(m.group(1)) >= 5)


def previous_figure_caption_line(lines: list[str], line_idx: int) -> int | None:
    for idx in range(line_idx - 1, -1, -1):
        text = lines[idx].strip()
        if not text:
            continue
        if is_standalone_figure_caption_line(text):
            return idx
    return None


def next_figure_caption_line(lines: list[str], line_idx: int) -> int | None:
    passed_self = False
    for idx in range(line_idx + 1, len(lines)):
        text = lines[idx].strip()
        if not text:
            continue
        if is_standalone_figure_caption_line(text):
            if not passed_self:
                passed_self = True
                continue
            return idx
    return None


def build_context_chapter4_special(
    lines: list[str],
    headers: list[tuple[int, str]],
    fig: FigureEntry,
) -> str:
    prev_header = previous_header(headers, fig.line_idx)
    next_hdr = next_header(headers, fig.line_idx)
    prev_caption_idx = previous_figure_caption_line(lines, fig.line_idx)
    next_caption_idx = next_figure_caption_line(lines, fig.line_idx)

    start = prev_header[0] if prev_header else 0
    if prev_caption_idx is not None:
        start = max(start, prev_caption_idx + 1)
    end = next_hdr[0] if next_hdr else len(lines)
    if next_caption_idx is not None:
        end = min(end, next_caption_idx)

    selected: list[str] = []
    for idx in range(start, end):
        text = lines[idx].rstrip()
        stripped = text.strip()
        if not stripped:
            selected.append("")
            continue
        if IMG_RE.search(stripped):
            continue
        if is_standalone_figure_caption_line(stripped):
            continue
        selected.append(stripped)

    return normalize_text("\n".join(selected))


def build_context(
    lines: list[str],
    headers: list[tuple[int, str]],
    mention_lines: list[int],
    strict_mode: bool,
) -> str:
    if not mention_lines:
        return ""

    start_anchor = min(mention_lines)
    end_anchor = max(mention_lines)

    prev_header = previous_header(headers, start_anchor)
    next_hdr = next_header(headers, end_anchor)

    start = (prev_header[0] + 1) if prev_header else 0
    end = next_hdr[0] if next_hdr else len(lines)

    selected: list[str] = []
    for idx in range(start, end):
        text = lines[idx].rstrip()
        stripped = text.strip()
        if not stripped:
            selected.append("")
            continue
        if IMG_RE.search(stripped):
            continue
        if strict_mode and is_standalone_figure_caption_line(stripped):
            continue
        selected.append(stripped)

    return normalize_text("\n".join(selected))


def extract_samples(md_path: Path) -> list[dict]:
    lines = md_path.read_text(encoding="utf-8").splitlines()
    headers = header_positions(lines)
    figures = extract_figures(lines, headers)

    samples: list[dict] = []
    for fig in figures:
        if fig.figure_id in DROP_FIGURE_IDS:
            continue
        strict_mode = bool(fig.chapter_num and fig.chapter_num >= 5)
        mentions = collect_figure_mentions(lines, fig.figure_id, strict_mode=strict_mode)
        if not mentions:
            continue
        if strict_mode and not has_non_caption_figure_reference(lines, mentions):
            continue
        cluster = nearest_figure_clusters(mentions, fig.line_idx)
        if not cluster:
            continue

        if is_chapter4_special_case(fig.figure_id):
            context_text = build_context_chapter4_special(lines, headers, fig)
        else:
            context_text = build_context(lines, headers, cluster, strict_mode=strict_mode)
        if not context_text:
            continue
        if strict_mode and not any(
            line.strip() and not is_standalone_figure_caption_line(line.strip())
            for line in context_text.splitlines()
        ):
            continue

        samples.append(
            {
                "book_name": BOOK_NAME,
                "anchor_type": "figure_ref_manual",
                "section_title": fig.section_title,
                "image_path": fig.image_path,
                "image_caption": fig.caption,
                "figure_id": fig.figure_id,
                "strict_mode": strict_mode,
                "context_text": context_text,
                "mention_lines": cluster,
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


def copy_images_and_rewrite_paths(samples: list[dict]) -> list[dict]:
    IMAGE_OUT_DIR.mkdir(parents=True, exist_ok=True)
    keep_names = {
        Path(item["image_path"]).name
        for item in samples
        if Path(item["image_path"]).exists()
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


def copy_images_to_dir(samples: list[dict], image_out_dir: Path) -> list[dict]:
    image_out_dir.mkdir(parents=True, exist_ok=True)
    keep_names = {
        Path(item["image_path"]).name
        for item in samples
        if Path(item["image_path"]).exists()
    }
    for existing in image_out_dir.iterdir():
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
            dst = image_out_dir / src.name
            if not dst.exists():
                shutil.copy2(src, dst)
            copied[item["image_path"]] = str(dst)

        new_item = dict(item)
        new_item["image_path"] = copied[item["image_path"]]
        rewritten.append(new_item)

    return rewritten


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def is_chapter4_update_target(figure_id: str) -> bool:
    return is_chapter4_special_case(figure_id)


def merge_with_existing(existing: list[dict], regenerated: list[dict]) -> list[dict]:
    regen_map = {item["figure_id"]: item for item in regenerated if is_chapter4_update_target(item["figure_id"])}
    merged: list[dict] = []
    seen: set[str] = set()

    for item in existing:
        fid = item.get("figure_id", "")
        if fid in PRESERVE_EXISTING_FIGURE_IDS:
            merged.append(item)
            seen.add(fid)
        elif fid in regen_map:
            merged.append(regen_map[fid])
            seen.add(fid)
        else:
            merged.append(item)
            seen.add(fid)

    for item in regenerated:
        fid = item["figure_id"]
        if fid in seen:
            continue
        merged.append(item)
        seen.add(fid)

    return merged


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    if not MD_PATH.exists():
        raise FileNotFoundError(f"未找到 markdown 文件: {MD_PATH}")

    regenerated = extract_samples(MD_PATH)
    existing = load_jsonl(OUTPUT_PATH)
    samples = merge_with_existing(existing, regenerated) if existing else regenerated
    samples = copy_images_to_dir(samples, IMAGE_OUT_DIR)
    write_jsonl(OUTPUT_PATH, samples)

    code_samples = copy_images_to_dir(samples, CODE_IMAGE_OUT_DIR)
    write_jsonl(CODE_OUTPUT_PATH, code_samples)

    print(f"切分完成: {BOOK_NAME}")
    print(f"样本数: {len(samples)}")
    print(f"输出文件: {OUTPUT_PATH}")
    print(f"代码目录输出: {CODE_OUTPUT_PATH}")
    print("\n前 5 条样本预览:")
    for item in samples[:5]:
        print("-" * 60)
        print(f"section_title: {item['section_title']}")
        print(f"figure_id: {item['figure_id']}")
        print(f"image_caption: {item['image_caption']}")
        preview = item["context_text"][:220].replace("\n", " ")
        print(f"context: {preview}...")


if __name__ == "__main__":
    main()
