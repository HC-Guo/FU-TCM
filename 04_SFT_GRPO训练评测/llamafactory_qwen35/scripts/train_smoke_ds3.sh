#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}"
export FORCE_TORCHRUN=1
export DISABLE_VERSION_CHECK=1
export TOKENIZERS_PARALLELISM=false
export MASTER_PORT="${MASTER_PORT:-29500}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export WANDB_PROJECT="${WANDB_PROJECT:-tcm-qwen35-sft}"
export WANDB_DIR="${WANDB_DIR:-../wandb}"
export WANDB_MODE="${WANDB_MODE:-online}"

llamafactory-cli train qwen35_9b_full_sft_ds3.yaml \
  max_samples=64 \
  max_steps=5 \
  logging_steps=1 \
  save_steps=5 \
  output_dir=../saves/Qwen3.5-9B/full/smoke_ds3 \
  run_name=qwen35_9b_full_sft_tcm_mm_ds3_smoke \
  overwrite_output_dir=true
