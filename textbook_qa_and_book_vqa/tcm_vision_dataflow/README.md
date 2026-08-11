# TCM Textbook QA and Illustrated-Book VQA

This module contains the code and prompt definitions used to prepare text QA and visual QA for FU-TCM. It includes a project-specific DataFlow runtime because several preparation scripts depend on that framework; DataFlow is an implementation dependency, not the focus of the FU-TCM project.

## Directory layout

```text
source_books/                 Local PDF and EPUB sources
archives/                     Local source archives
data/                         Local intermediate QA/VQA records
results/                      Local generated outputs
workflows/dataflow2/          FU-TCM preparation scripts and prompts
workflows/dataflow_runtime/   Project-specific DataFlow 1.0.10 runtime
requirements.txt             Python dependencies
```

Books, images, generated records, and result files are local artifacts and are not included in this public code repository.

## Environment variables

No API key is stored in the code. Set the provider configuration required by the selected script:

```bash
export DF_API_KEY="<your-api-key>"
export MINIMAX_API_URL="<openai-compatible-chat-completions-url>"
export MINIMAX_MODEL="<model-name>"
```

## Installation

```bash
cd textbook_qa_and_book_vqa/tcm_vision_dataflow
pip install -r requirements.txt
```

## Main entry points

```text
workflows/dataflow2/scripts/vqa/
workflows/dataflow2/scripts/qa/
workflows/dataflow2/prompt_engineering/
workflows/dataflow_runtime/
```
