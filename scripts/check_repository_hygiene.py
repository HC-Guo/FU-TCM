#!/usr/bin/env python3
"""Fail when tracked files contain datasets, credentials, or portability regressions."""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_EXTENSIONS = {
    ".csv", ".doc", ".docx", ".jpg", ".jpeg", ".jsonl", ".mp3",
    ".parquet", ".pdf", ".png", ".tar", ".tgz", ".tsv", ".wav",
    ".webp", ".xls", ".xlsx", ".zip",
}
FORBIDDEN_PATH_PARTS = {
    "benchmark", "checkpoints", "data_parquet", "grpodata", "source_books",
    "source_cases", "sft_data", "sft_image_data", "sft_merged", "verl_data",
}
FORBIDDEN_PREFIXES = (
    "02_textbook_qa_and_book_vqa/tcm_vision_dataflow/workflows/dataflow_runtime/",
    "02_textbook_qa_and_book_vqa/tcm_vision_dataflow/workflows/dataflow2/framework/DataFlow-main/",
)
SECRET_PATTERNS = {
    "OpenAI-style token": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "GitHub token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "Google API key": re.compile(r"\bAIza[0-9A-Za-z_-]{30,}\b"),
    "Hugging Face token": re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    "Slack token": re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
}
WINDOWS_ABSOLUTE_PATH = re.compile(r"[A-Za-z]:\\")
KEY_DERIVATION_LOG = re.compile(
    r"\b(?:key_prefix|key_suffix)\s*=|"
    r"\b(?:api_key|access_token|secret_key)\s*\[\s*:\s*\d+|"
    r"len\(\s*(?:api_key|access_token|secret_key)\s*\)|"
    r"print\(.*\b(?:key_prefix|key_suffix)\b",
    re.IGNORECASE,
)


def tracked_files() -> list[Path]:
    output = subprocess.check_output(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=ROOT
    ).decode("utf-8")
    return [ROOT / name for name in output.rstrip("\0").split("\0") if name]


def read_text(path: Path) -> str | None:
    try:
        if path.stat().st_size > 2_000_000:
            return None
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def main() -> int:
    errors: list[str] = []
    files = tracked_files()

    for path in files:
        relative = path.relative_to(ROOT).as_posix()
        if any(ord(char) > 127 for char in relative):
            errors.append(f"non-English tracked path: {relative}")
        if path.suffix.lower() in DATA_EXTENSIONS:
            errors.append(f"tracked dataset/binary source file: {relative}")
        if any(part in FORBIDDEN_PATH_PARTS for part in path.relative_to(ROOT).parts):
            errors.append(f"tracked local-data directory: {relative}")
        if relative.startswith(FORBIDDEN_PREFIXES):
            errors.append(f"vendored DataFlow tree must stay removed: {relative}")
        if path.name == "prompt_example.json":
            errors.append(f"tracked clinical example data: {relative}")
        if path.name == ".env" or (path.name.startswith(".env.") and path.name != ".env.example"):
            errors.append(f"tracked environment file: {relative}")

        text = read_text(path)
        if text is None:
            continue
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                errors.append(f"{label} candidate: {relative}")
        if path.suffix == ".py":
            try:
                # Validate against the oldest Python version supported by CI even
                # when this script is run from a newer local interpreter.
                ast.parse(text, filename=relative, feature_version=(3, 11))
            except SyntaxError as exc:
                errors.append(f"Python syntax error: {relative}:{exc.lineno}: {exc.msg}")
            for line_number, line in enumerate(text.splitlines(), start=1):
                if WINDOWS_ABSOLUTE_PATH.search(line):
                    errors.append(f"hardcoded Windows path: {relative}:{line_number}")
                if relative != "scripts/check_repository_hygiene.py" and KEY_DERIVATION_LOG.search(line):
                    errors.append(f"credential-derived logging: {relative}:{line_number}")

    if errors:
        print("Repository hygiene check failed:")
        for error in sorted(set(errors)):
            print(f"- {error}")
        return 1

    print(f"Repository hygiene check passed ({len(files)} tracked files).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
