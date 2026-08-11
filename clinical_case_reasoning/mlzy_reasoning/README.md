# Senior TCM Practitioner Case Reasoning

This directory contains the code path for extracting case records, converting them to the structured *bianzheng* schema, reviewing four-examination evidence and intermediate fields, and preparing local training inputs. All paths are resolved relative to this module.

## Directory layout

```text
source_cases/                      Local source-case directory
configs/mapping_table.json         Syndrome-mapping rules
prompt_example.json                Optional local few-shot case
scripts/extract/                   Source extraction
scripts/convert/                   Structured case conversion
scripts/verify/                    Evidence and field review
scripts/prepare/split_dataset.py   Deterministic local split
scripts/prepare/prepare_data.py    Local verl/parquet preparation
data/processed/                    Local intermediate outputs
data_parquet/                      Default local parquet output
```

Source cases, generated records, train/test splits, and parquet files are local artifacts and are not included in this public code repository.

## Environment variables

No API key is stored in the code. Configure the model provider before running scripts that make API calls:

```bash
export OPENAI_API_KEY="<your-api-key>"
export OPENAI_BASE_URL="<openai-compatible-base-url>"
export OPENAI_MODEL="<model-name>"
```

The verification script also supports Anthropic-compatible configuration:

```bash
export ANTHROPIC_API_KEY="<your-api-key>"
export ANTHROPIC_BASE_URL="<anthropic-compatible-base-url>"
export ANTHROPIC_MODEL="<model-name>"
```

Set `TCM_PROMPT_EXAMPLE_FILE` to use an optional local few-shot example. If no example file exists, the conversion and preparation scripts use zero-shot mode.

## Execution order

```bash
python scripts/extract/extract_minglaoyishi.py
python scripts/convert/convert_minimax.py
python scripts/verify/verify_mlzy.py
python scripts/prepare/split_dataset.py
python scripts/prepare/prepare_data.py
```
