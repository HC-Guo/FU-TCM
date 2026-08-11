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
    "textbook_qa_and_book_vqa/tcm_vision_dataflow/workflows/dataflow2/framework/DataFlow-main/",
    "textbook_qa_and_book_vqa/tcm_vision_dataflow/workflows/dataflow_runtime/dataflow/example/",
    "textbook_qa_and_book_vqa/tcm_vision_dataflow/workflows/dataflow_runtime/test/",
)
ALLOWED_BINARY_PREFIXES = (
    "textbook_qa_and_book_vqa/tcm_vision_dataflow/workflows/dataflow_runtime/static/logo/",
)
ALLOWED_BINARY_FILES = {
    "assets/fu-tcm-framework-overview.png",
}
REQUIRED_PIPELINE_FILES = (
    "classical_text_qa/scripts/process/convert_to_sft_format.py",
    "textbook_qa_and_book_vqa/tcm_vision_dataflow/workflows/dataflow_runtime/pyproject.toml",
    "textbook_qa_and_book_vqa/tcm_vision_dataflow/workflows/dataflow_runtime/dataflow/serving/api_llm_serving_request.py",
    "textbook_qa_and_book_vqa/tcm_vision_dataflow/workflows/dataflow_runtime/dataflow/operators/pdf2vqa/generate/llm_output_parser.py",
    "textbook_qa_and_book_vqa/tcm_vision_dataflow/workflows/dataflow_runtime/dataflow/statics/pipelines/api_pipelines/pdf_vqa_extract_pipeline_part1.py",
    "textbook_qa_and_book_vqa/tcm_vision_dataflow/workflows/dataflow2/scripts/vqa/gen_shezhen_vqa_final.py",
    "clinical_case_reasoning/mlzy_reasoning/scripts/prepare/split_dataset.py",
    "clinical_case_reasoning/mlzy_reasoning/scripts/prepare/prepare_data.py",
    "sft_grpo_training_evaluation/convert_bianzheng_to_verl_grpo.py",
    "sft_grpo_training_evaluation/reward_functions.py",
    "sft_grpo_training_evaluation/run_tcm_grpo_smoke.sh",
    "sft_grpo_training_evaluation/run_tcm_grpo_smoke_hf.sh",
    "sft_grpo_training_evaluation/verl_py_shims/sitecustomize.py",
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
WINDOWS_ABSOLUTE_PATH = re.compile(
    r"\b[A-Za-z]:\\(?:[^\\\r\n\"']+\\)+[^\\\r\n\"']*"
)
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


def indexed_relatives() -> set[str]:
    output = subprocess.check_output(
        ["git", "ls-files", "-z", "--cached"], cwd=ROOT
    ).decode("utf-8")
    return {name for name in output.rstrip("\0").split("\0") if name}


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
    tracked_relatives = indexed_relatives()

    for required in REQUIRED_PIPELINE_FILES:
        if required not in tracked_relatives:
            errors.append(f"required pipeline file is not tracked: {required}")

    for path in files:
        relative = path.relative_to(ROOT).as_posix()
        if any(ord(char) > 127 for char in relative):
            errors.append(f"non-English tracked path: {relative}")
        if (
            path.suffix.lower() in DATA_EXTENSIONS
            and relative not in ALLOWED_BINARY_FILES
            and not relative.startswith(ALLOWED_BINARY_PREFIXES)
        ):
            errors.append(f"tracked dataset/binary source file: {relative}")
        if any(part in FORBIDDEN_PATH_PARTS for part in path.relative_to(ROOT).parts):
            errors.append(f"tracked local-data directory: {relative}")
        if relative.startswith(FORBIDDEN_PREFIXES):
            errors.append(f"excluded DataFlow snapshot/example/test file: {relative}")
        if path.name == "prompt_example.json":
            errors.append(f"tracked clinical example data: {relative}")
        if path.name == ".env" or (path.name.startswith(".env.") and path.name != ".env.example"):
            errors.append(f"tracked environment file: {relative}")

        text = read_text(path)
        if text is None:
            continue
        if text.startswith("version https://git-lfs.github.com/spec/v1"):
            errors.append(f"unresolved Git LFS pointer: {relative}")
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                errors.append(f"{label} candidate: {relative}")
        is_unrendered_template = "{{" in relative or "{%" in relative
        if path.suffix == ".py" and not is_unrendered_template:
            try:
                # Validate against the oldest Python version supported by CI even
                # when this script is run from a newer local interpreter.
                ast.parse(text, filename=relative, feature_version=(3, 11))
            except SyntaxError as exc:
                errors.append(f"Python syntax error: {relative}:{exc.lineno}: {exc.msg}")
            for line_number, line in enumerate(text.splitlines(), start=1):
                if WINDOWS_ABSOLUTE_PATH.search(line):
                    errors.append(f"hardcoded Windows path: {relative}:{line_number}")
                if relative != "scripts/validate_repository.py" and KEY_DERIVATION_LOG.search(line):
                    errors.append(f"credential-derived logging: {relative}:{line_number}")

    if errors:
        print("Repository validation failed:")
        for error in sorted(set(errors)):
            print(f"- {error}")
        return 1

    print(f"Repository validation passed ({len(files)} tracked files).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
