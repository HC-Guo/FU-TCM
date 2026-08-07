#!/usr/bin/env bash
set -euo pipefail

python llamafactory_qwen35/scripts/prepare_tcm_multimodal_data.py \
  --input sft_merged/tcm_sft_merged.json \
  --output-dir llamafactory_qwen35/data \
  --output-name tcm_sft_mm.json \
  --dataset-name tcm_sft_mm
