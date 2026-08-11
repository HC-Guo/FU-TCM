#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MODULE_DIR=$(cd "${SCRIPT_DIR}/../.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3}

"${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_tcm_multimodal_data.py" \
  --input "${MODULE_DIR}/sft_merged/tcm_sft_merged.json" \
  --output-dir "${MODULE_DIR}/llamafactory_qwen35/data" \
  --output-name tcm_sft_mm.json \
  --dataset-name tcm_sft_mm
