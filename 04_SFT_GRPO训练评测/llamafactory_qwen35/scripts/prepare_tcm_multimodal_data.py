#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path


IMAGE_TOKEN_RE = re.compile(r"<image>")


def load_records(path):
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
    else:
        with path.open("r", encoding="utf-8") as f:
            yield from json.load(f)


def resolve_image(image_path, image_root, input_dir):
    path = Path(image_path)
    if path.is_absolute():
        return path

    candidates = [
        image_root / path,
        input_dir / path,
        Path.cwd() / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (image_root / path).resolve()


def remove_image_tags(text):
    text = IMAGE_TOKEN_RE.sub("", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_image_tags(record, image_count):
    conversations = record.get("conversations") or []
    if not conversations:
        return record

    first_user = None
    image_tag_count = 0
    for message in conversations:
        value = message.get("value")
        if isinstance(value, str):
            image_tag_count += len(IMAGE_TOKEN_RE.findall(value))
        if first_user is None and message.get("from") in {"human", "user"}:
            first_user = message

    if image_count <= 0:
        for message in conversations:
            value = message.get("value")
            if isinstance(value, str):
                message["value"] = remove_image_tags(value)
        return record

    if image_tag_count == image_count:
        return record

    for message in conversations:
        value = message.get("value")
        if isinstance(value, str):
            message["value"] = remove_image_tags(value)

    if first_user is not None:
        prefix = "\n".join(["<image>"] * image_count)
        first_user["value"] = f"{prefix}\n{first_user.get('value', '')}".strip()

    return record


def write_dataset_info(output_dir, file_name, dataset_name):
    dataset_info = {
        dataset_name: {
            "file_name": file_name,
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
                "images": "images"
            },
            "tags": {
                "role_tag": "from",
                "content_tag": "value",
                "user_tag": "human",
                "assistant_tag": "gpt"
            }
        }
    }
    with (output_dir / "dataset_info.json").open("w", encoding="utf-8") as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)
        f.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="sft_merged/tcm_sft_merged.json")
    parser.add_argument("--output-dir", default="llamafactory_qwen35/data")
    parser.add_argument("--output-name", default="tcm_sft_mm.json")
    parser.add_argument("--dataset-name", default="tcm_sft_mm")
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--keep-missing-images", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.output_name
    image_root = Path(args.image_root).resolve() if args.image_root else input_path.parent.resolve()

    total = 0
    written = 0
    with_images = 0
    skipped_missing = 0

    with output_path.open("w", encoding="utf-8") as out:
        out.write("[\n")
        first = True

        for record in load_records(input_path):
            total += 1
            if args.max_samples and written >= args.max_samples:
                break

            images = record.get("images") or []
            if images:
                resolved_images = [resolve_image(img, image_root, input_path.parent) for img in images]
                missing = [str(img) for img in resolved_images if not img.exists()]
                if missing and not args.keep_missing_images:
                    skipped_missing += 1
                    continue
                record["images"] = [str(img) for img in resolved_images]
                record = normalize_image_tags(record, len(resolved_images))
                with_images += 1
            else:
                record = normalize_image_tags(record, 0)

            if not first:
                out.write(",\n")
            json.dump(record, out, ensure_ascii=False, separators=(",", ":"))
            first = False
            written += 1

        out.write("\n]\n")

    write_dataset_info(output_dir, args.output_name, args.dataset_name)

    print(f"input: {input_path}")
    print(f"output: {output_path}")
    print(f"dataset_info: {output_dir / 'dataset_info.json'}")
    print(f"total scanned: {total}")
    print(f"written: {written}")
    print(f"with images: {with_images}")
    print(f"skipped missing images: {skipped_missing}")


if __name__ == "__main__":
    main()
