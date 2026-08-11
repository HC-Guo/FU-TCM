# Meta-Reasoning Case Conversion

This directory contains the conversion and verification code for mapping local meta-reasoning cases to the structured *bianzheng* schema. All paths are resolved relative to this module.

## Directory layout

```text
data/raw/                              Local source cases
data/processed/                        Local conversion and verification outputs
configs/mapping_table.json             Syndrome-mapping rules
scripts/convert/convert_bianzheng.py   Multi-model structured conversion
scripts/verify/verify_bianzheng.py     Intermediate-field verification
```

Raw cases, converted records, train/test splits, and verified outputs are local artifacts and are not included in this public code repository.

## Environment variables

No API key is stored in the code. Configure the selected providers before running conversion or verification:

```bash
export OPENAI_API_KEY="<your-api-key>"
export OPENAI_BASE_URL="<openai-compatible-base-url>"
export META_REASONING_MODELS="<comma-separated-model-names>"

export ANTHROPIC_API_KEY="<your-api-key>"
export ANTHROPIC_BASE_URL="<anthropic-compatible-base-url>"
export ANTHROPIC_MODEL="<model-name>"
```

## Execution order

```bash
python scripts/convert/convert_bianzheng.py
python scripts/verify/verify_bianzheng.py
```
