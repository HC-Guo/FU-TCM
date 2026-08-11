#!/usr/bin/env bash
# Clean GRPO smoke run for TCM Qwen3.5-9B using HF rollout.
# Run from TCM_final after activating the tcm_grpo_hf conda env.

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
VERL_DIR="${VERL_DIR:-${ROOT_DIR}/verl}"
PYTHON_BIN="${PYTHON_BIN:-python}"

MODEL_PATH="${MODEL_PATH:-${ROOT_DIR}/saves/Qwen3.5-9B/coldstart_v1}"
FULL_TRAIN_FILE="${FULL_TRAIN_FILE:-${ROOT_DIR}/verl_data/bianzheng_grpo_train.parquet}"
FULL_TEST_FILE="${FULL_TEST_FILE:-${ROOT_DIR}/verl_data/bianzheng_grpo_test.parquet}"
TRAIN_FILE="${TRAIN_FILE:-${ROOT_DIR}/verl_data/bianzheng_grpo_train_smoke.parquet}"
TEST_FILE="${TEST_FILE:-${ROOT_DIR}/verl_data/bianzheng_grpo_test_smoke.parquet}"
REWARD_PATH="${REWARD_PATH:-${ROOT_DIR}/reward_functions.py}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}"
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
export WANDB_PROJECT="${WANDB_PROJECT:-tcm-grpo}"
export WANDB_MODE="${WANDB_MODE:-online}"

NNODES="${NNODES:-1}"
NDEVICES_PER_NODE="${NDEVICES_PER_NODE:-6}"
FSDP_SIZE="${FSDP_SIZE:-6}"
STRATEGY="${STRATEGY:-fsdp}"

SMOKE_TRAIN_SIZE="${SMOKE_TRAIN_SIZE:-12}"
SMOKE_TEST_SIZE="${SMOKE_TEST_SIZE:-4}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-6}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-6}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-512}"
ROLLOUT_N="${ROLLOUT_N:-1}"
ACTOR_LR="${ACTOR_LR:-5e-7}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.01}"
USE_KL_LOSS="${USE_KL_LOSS:-false}"
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"

TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
SAVE_FREQ="${SAVE_FREQ:--1}"
TEST_FREQ="${TEST_FREQ:-1}"
PROJECT_NAME="${PROJECT_NAME:-tcm-grpo}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen35_9b_tcm_grpo_hf_smoke}"
SAVE_DIR="${SAVE_DIR:-${ROOT_DIR}/saves/Qwen3.5-9B/grpo_hf_smoke}"

require_file() {
    if [ ! -f "$1" ]; then
        echo "Missing file: $1" >&2
        exit 1
    fi
}

require_dir() {
    if [ ! -d "$1" ]; then
        echo "Missing directory: $1" >&2
        exit 1
    fi
}

require_dir "${VERL_DIR}"
require_file "${VERL_DIR}/verl/trainer/main_ppo.py"
require_file "${MODEL_PATH}/config.json"
require_file "${FULL_TRAIN_FILE}"
require_file "${FULL_TEST_FILE}"
require_file "${REWARD_PATH}"

mkdir -p "${ROOT_DIR}/verl_data" "${SAVE_DIR}" "${ROOT_DIR}/logs"

echo "ROOT_DIR=${ROOT_DIR}"
echo "VERL_DIR=${VERL_DIR}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "TRAIN_FILE=${TRAIN_FILE}"
echo "TEST_FILE=${TEST_FILE}"
echo "REWARD_PATH=${REWARD_PATH}"
echo "rollout=hf strategy=${STRATEGY} rollout_n=${ROLLOUT_N}"
echo "train_batch_size=${TRAIN_BATCH_SIZE} ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} max_response_length=${MAX_RESPONSE_LENGTH} use_kl_loss=${USE_KL_LOSS}"
echo "lora_rank=${LORA_RANK} lora_alpha=${LORA_ALPHA}"

"${PYTHON_BIN}" - <<'PY'
import sys
import torch
import transformers

print("python:", sys.executable)
print("torch:", torch.__version__, "cuda:", torch.version.cuda, "cuda_available:", torch.cuda.is_available())
print("transformers:", transformers.__version__)
PY

"${PYTHON_BIN}" - <<PY
import pandas as pd

train = pd.read_parquet("${FULL_TRAIN_FILE}").head(int("${SMOKE_TRAIN_SIZE}"))
test = pd.read_parquet("${FULL_TEST_FILE}").head(int("${SMOKE_TEST_SIZE}"))
train.to_parquet("${TRAIN_FILE}", index=False)
test.to_parquet("${TEST_FILE}", index=False)
print("wrote smoke train:", "${TRAIN_FILE}", len(train))
print("wrote smoke test:", "${TEST_FILE}", len(test))
PY

cd "${VERL_DIR}"

DATA_ARGS=(
    data.train_files="${TRAIN_FILE}"
    data.val_files="${TEST_FILE}"
    data.train_batch_size="${TRAIN_BATCH_SIZE}"
    data.max_prompt_length="${MAX_PROMPT_LENGTH}"
    data.max_response_length="${MAX_RESPONSE_LENGTH}"
    data.filter_overlong_prompts=True
    data.truncation=error
    data.shuffle=False
)

ALGO_ARGS=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
)

REWARD_ARGS=(
    custom_reward_function.path="${REWARD_PATH}"
    custom_reward_function.name=compute_score
    reward_model.reward_manager=naive
)

MODEL_ARGS=(
    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.model.trust_remote_code=True
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.model.lora_rank="${LORA_RANK}"
    actor_rollout_ref.model.lora_alpha="${LORA_ALPHA}"
    +actor_rollout_ref.model.override_config._attn_implementation=eager
)

ACTOR_ARGS=(
    actor_rollout_ref.actor.strategy="${STRATEGY}"
    actor_rollout_ref.actor.optim.lr="${ACTOR_LR}"
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}"
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}"
    actor_rollout_ref.actor.use_kl_loss="${USE_KL_LOSS}"
    actor_rollout_ref.actor.kl_loss_coef="${KL_LOSS_COEF}"
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.use_dynamic_bsz=False
    actor_rollout_ref.actor.use_torch_compile=False
    actor_rollout_ref.actor.fsdp_config.fsdp_size="${FSDP_SIZE}"
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
    actor_rollout_ref.actor.fsdp_config.reshard_after_forward=True
)

REF_ARGS=(
    actor_rollout_ref.ref.strategy="${STRATEGY}"
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}"
    actor_rollout_ref.ref.use_torch_compile=False
    actor_rollout_ref.ref.fsdp_config.param_offload=True
    actor_rollout_ref.ref.fsdp_config.reshard_after_forward=True
)

ROLLOUT_ARGS=(
    actor_rollout_ref.rollout.name=hf
    actor_rollout_ref.rollout.mode=sync
    actor_rollout_ref.rollout.n="${ROLLOUT_N}"
    actor_rollout_ref.rollout.ignore_eos=False
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}"
)

TRAINER_ARGS=(
    trainer.project_name="${PROJECT_NAME}"
    trainer.experiment_name="${EXPERIMENT_NAME}"
    trainer.logger='["console","wandb"]'
    trainer.nnodes="${NNODES}"
    trainer.n_gpus_per_node="${NDEVICES_PER_NODE}"
    trainer.default_local_dir="${SAVE_DIR}"
    trainer.critic_warmup=0
    trainer.val_before_train=False
    trainer.save_freq="${SAVE_FREQ}"
    trainer.test_freq="${TEST_FREQ}"
    trainer.total_epochs="${TOTAL_EPOCHS}"
)

LOG_FILE="${ROOT_DIR}/logs/${EXPERIMENT_NAME}_$(date +%Y%m%d_%H%M%S).log"

"${PYTHON_BIN}" -m verl.trainer.main_ppo \
    "${DATA_ARGS[@]}" \
    "${ALGO_ARGS[@]}" \
    "${REWARD_ARGS[@]}" \
    "${MODEL_ARGS[@]}" \
    "${ACTOR_ARGS[@]}" \
    "${REF_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${TRAINER_ARGS[@]}" \
    "$@" 2>&1 | tee "${LOG_FILE}"
