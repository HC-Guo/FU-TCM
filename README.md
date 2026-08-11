# FU-TCM

FU-TCM is a code-only repository for producing traditional Chinese medicine QA/VQA data, expanding clinical reasoning cases, and running SFT, GRPO, and evaluation workflows.

> Training data, source books, generated datasets, images, and evaluation samples are not included in the current branch. The expected local data directories are listed in `.gitignore` so they are not accidentally committed again.

## Repository structure

| Directory | Purpose |
| --- | --- |
| `01_classical_text_qa/` | Generate, clean, and convert QA pairs from classical TCM texts. |
| `02_textbook_qa_and_book_vqa/` | Generate text QA and image-grounded VQA from textbooks and illustrated books. |
| `03_clinical_case_reasoning/` | Convert and verify structured syndrome-differentiation reasoning data. |
| `04_sft_grpo_training_evaluation/` | SFT/GRPO preparation, training configuration, reward functions, and evaluation tools. |

Open `index.html` for the browsable project overview. The current upload inventory and confirmed missing files are listed in [`docs/repository_inventory.html`](docs/repository_inventory.html).

## Install code dependencies

Install only the module you intend to run. DataFlow is a pinned external dependency; its duplicated source and upstream example/test data are not vendored here.

```bash
pip install -r 01_classical_text_qa/requirements.txt
pip install -r 02_textbook_qa_and_book_vqa/tcm_vision_dataflow/requirements.txt
pip install -r 03_clinical_case_reasoning/mlzy_reasoning/requirements.txt
pip install -r 04_sft_grpo_training_evaluation/requirements.txt
```

## Local data layout

The scripts recreate their output directories as needed. Place private source data only in the ignored locations relevant to your workflow:

```text
01_classical_text_qa/source_texts/
01_classical_text_qa/qa_output/
01_classical_text_qa/sft_data/

02_textbook_qa_and_book_vqa/sft_data/
02_textbook_qa_and_book_vqa/sft_image_data/
02_textbook_qa_and_book_vqa/tcm_vision_dataflow/{source_books,data,results,archives}/

03_clinical_case_reasoning/{meta_reasoning,mlzy_reasoning}/data/
03_clinical_case_reasoning/mlzy_reasoning/source_cases/
03_clinical_case_reasoning/mlzy_reasoning/prompt_example.json

04_sft_grpo_training_evaluation/{sft_data,sft_image_data,sft_merged,grpodata,verl_data,benchmark}/
```

## Environment variables

Copy `.env.example` to a local environment file and provide only the credentials required by the selected workflow. Do not commit secrets.

```bash
export TCM_API_KEY="<your_api_key>"
export DF_API_KEY="<your_api_key>"
export OPENAI_API_KEY="<your_api_key>"
export OPENAI_BASE_URL="<your_api_base_url>"
export OPENAI_MODEL="<your_model>"
```

No key material is printed. Real `.env` files, private keys, and common credential files are ignored. See `SECURITY.md` for the repository policy.

## Quick entry points

```bash
# Classical-text QA generation
python 01_classical_text_qa/scripts/generate/generate_qa_01.py

# Clinical reasoning conversion
python 03_clinical_case_reasoning/meta_reasoning/scripts/convert/convert_bianzheng.py

# Local MLZY flow: extract -> convert -> verify -> split -> parquet
python 03_clinical_case_reasoning/mlzy_reasoning/scripts/extract/extract_minglaoyishi.py
python 03_clinical_case_reasoning/mlzy_reasoning/scripts/convert/convert_minimax.py
python 03_clinical_case_reasoning/mlzy_reasoning/scripts/verify/verify_mlzy.py
python 03_clinical_case_reasoning/mlzy_reasoning/scripts/prepare/split_dataset.py
python 03_clinical_case_reasoning/mlzy_reasoning/scripts/prepare/prepare_data.py

# GRPO data conversion (requires local input data)
cd 04_sft_grpo_training_evaluation
python convert_bianzheng_to_verl_grpo.py
```

## Repository check

```bash
python scripts/check_repository_hygiene.py
```

GitHub Actions runs the same check and rejects tracked datasets, credential candidates, non-English tracked paths, hardcoded Windows paths, and Python syntax errors.
