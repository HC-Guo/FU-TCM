# FU-TCM QA and VQA Preparation Scripts

This directory contains project-specific scripts for preparing text QA and visual QA from locally held TCM books. It uses the bundled DataFlow runtime for document parsing and model calls.

Source books, images, generated QA/VQA records, and aggregate results are private local artifacts and are not included in this repository.

## Structure

```text
scripts/
  vqa/                  Illustrated-book VQA generators
  qa/                   Medical-textbook QA generators
  chunkers/             Book-specific text chunking
prompt_engineering/     Prompt definitions and routing notes
```

## Output schemas

Visual QA generators produce records with source metadata, one image reference, a question, an answer, and a task type. Text QA generators produce records with source metadata, a question, an answer, and supporting evidence. The exact source material remains local.

## Generation strategies

| Strategy | Intended use |
| --- | --- |
| Theory illustration | One figure paired with explanatory text |
| Region-based case expansion | One case image expanded into questions for each supported region |
| Section routing | Different prompt templates for structurally different sections of one book |
| Fixed question set | Stable task types for consistently formatted sources |
| Enriched open QA | Open-ended questions generated from an image-text pair |

## Environment

```bash
pip install -r ../../requirements.txt

export DF_API_KEY="<your-api-key>"
export MINIMAX_API_URL="<openai-compatible-chat-completions-url>"
export MINIMAX_MODEL="<model-name>"
```

## Main entry points

```text
scripts/vqa/gen_shezhen_vqa_final.py
scripts/vqa/gen_shezhen_vqa_from_split.py
scripts/vqa/gen_zhongyao_caotu_vqa.py
scripts/qa/medical_pdf_to_qa_pipeline.py
```

Input and output paths are script-specific. Keep them in Git-ignored local directories.
