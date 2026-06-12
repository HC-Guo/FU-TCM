#!/usr/bin/env bash
# GRPO smoke run for TCM bianzheng data.
# Based on verl examples/grpo_trainer/run_qwen3_5_27b_fsdp.sh.

set -xeuo pipefail

########################### paths ###########################
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PREFERRED_VERL_DIR="${PREFERRED_VERL_DIR:-/cpfs01/projects-HDD/cfff-afe2df89e32e_HDD/ghc_45160/slr/verl}"

if [ -n "${ROOT_DIR:-}" ]; then
    ROOT_DIR=$(cd "${ROOT_DIR}" && pwd)
elif [ -d "${SCRIPT_DIR}/verl_data" ]; then
    ROOT_DIR="${SCRIPT_DIR}"
else
    ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
fi

if [ -n "${VERL_DIR:-}" ]; then
    VERL_DIR=$(cd "${VERL_DIR}" && pwd)
elif [ -f "${PREFERRED_VERL_DIR}/verl/trainer/main_ppo.py" ]; then
    VERL_DIR="${PREFERRED_VERL_DIR}"
elif [ -d "${ROOT_DIR}/verl" ]; then
    VERL_DIR="${ROOT_DIR}/verl"
else
    VERL_DIR="${SCRIPT_DIR}"
fi

cd "${VERL_DIR}"
mkdir -p logs

########################### environment ###########################
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}"
export CUDA_HOME="${CUDA_HOME:-/cpfs01/projects-HDD/cfff-afe2df89e32e_HDD/ghc_45160/cuda-12.8}"
if [ -d "${CUDA_HOME}" ]; then
    export PATH="${CUDA_HOME}/bin:${PATH}"
    CUDA_LIB_PATH="${CUDA_HOME}/lib64"
    if [ -d "${CUDA_HOME}/compat" ]; then
        CUDA_LIB_PATH="${CUDA_HOME}/compat:${CUDA_LIB_PATH}"
    fi
    export LD_LIBRARY_PATH="${CUDA_LIB_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
else
    echo "WARNING: CUDA_HOME does not exist: ${CUDA_HOME}" >&2
fi
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT="${WANDB_PROJECT:-tcm-grpo}"
export WANDB_MODE="${WANDB_MODE:-online}"
if [ -d "${ROOT_DIR}/verl_py_shims" ]; then
    export PYTHONPATH="${ROOT_DIR}/verl_py_shims${PYTHONPATH:+:${PYTHONPATH}}"
fi

########################### user-adjustable ###########################
PYTHON_BIN=${PYTHON_BIN:-python3}
DEVICE=${DEVICE:-$("${PYTHON_BIN}" -c 'import torch_npu' 2>/dev/null && echo npu || echo gpu)}
INFER_BACKEND=${INFER_BACKEND:-vllm}
PROJECT_NAME=${PROJECT_NAME:-tcm-grpo}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen35_9b_tcm_grpo_smoke_$(date +%Y%m%d_%H%M)}

MODEL_PATH=${MODEL_PATH:-"${ROOT_DIR}/saves/Qwen3.5-9B/coldstart_v1"}
FULL_TRAIN_FILE=${FULL_TRAIN_FILE:-"${ROOT_DIR}/verl_data/bianzheng_grpo_train.parquet"}
FULL_TEST_FILE=${FULL_TEST_FILE:-"${ROOT_DIR}/verl_data/bianzheng_grpo_test.parquet"}
TRAIN_FILE=${TRAIN_FILE:-"${ROOT_DIR}/verl_data/bianzheng_grpo_train_smoke.parquet"}
TEST_FILE=${TEST_FILE:-"${ROOT_DIR}/verl_data/bianzheng_grpo_test_smoke.parquet"}
REWARD_PATH=${REWARD_PATH:-"${ROOT_DIR}/reward_functions.py"}

NNODES=${NNODES:-1}
NDEVICES_PER_NODE=${NDEVICES_PER_NODE:-6}
GEN_TP=${GEN_TP:-2}
SP_SIZE=${SP_SIZE:-1}
FSDP_SIZE=${FSDP_SIZE:-6}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.55}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-12}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-6}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}
ROLLOUT_N=${ROLLOUT_N:-4}
ACTOR_LR=${ACTOR_LR:-5e-7}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.01}

SAVE_FREQ=${SAVE_FREQ:--1}
TEST_FREQ=${TEST_FREQ:-1}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
########################### end user-adjustable ###########################

case "${DEVICE}" in
    gpu) ;;
    npu)
        export HCCL_CONNECT_TIMEOUT=1500
        export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
        export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050
        export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
        ;;
    *)
        echo "Unsupported DEVICE=${DEVICE}. Expected 'gpu' or 'npu'." >&2
        exit 1
        ;;
esac

test -f "${FULL_TRAIN_FILE}"
test -f "${FULL_TEST_FILE}"
test -f "${REWARD_PATH}"
test -f "${MODEL_PATH}/config.json"
test -f "${VERL_DIR}/verl/trainer/main_ppo.py"

echo "ROOT_DIR=${ROOT_DIR}"
echo "VERL_DIR=${VERL_DIR}"
echo "PYTHON_BIN=${PYTHON_BIN}"
echo "CUDA_HOME=${CUDA_HOME}"
command -v nvcc >/dev/null 2>&1 && nvcc -V || true
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true

"${PYTHON_BIN}" - <<'PY'
import sys

import torch

print("python:", sys.executable)
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
try:
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))
        print("capability:", torch.cuda.get_device_capability(0))
except Exception as exc:
    print("torch cuda check failed:", repr(exc))
    raise

try:
    import vllm
    print("vllm:", vllm.__version__)
except Exception as exc:
    print("vllm import failed:", repr(exc))
    raise

import transformers
print("transformers:", transformers.__version__)
print("has AutoModelForVision2Seq:", hasattr(transformers, "AutoModelForVision2Seq"))
PY

"${PYTHON_BIN}" - <<'PY'
from pathlib import Path
import site
import textwrap

module_code = textwrap.dedent(
    """
    try:
        import transformers
    except Exception:
        transformers = None

    if transformers is not None:
        if not hasattr(transformers, "AutoModelForVision2Seq") and hasattr(transformers, "AutoModelForImageTextToText"):
            transformers.AutoModelForVision2Seq = transformers.AutoModelForImageTextToText
    """
).lstrip()

site_dirs = [Path(p) for p in site.getsitepackages()]
site_dirs.extend(Path(p) for p in site.getusersitepackages().split(":") if p)

for site_dir in site_dirs:
    if site_dir.exists() and site_dir.is_dir():
        module_path = site_dir / "tcm_transformers_compat.py"
        pth_path = site_dir / "tcm_transformers_compat.pth"
        try:
            module_path.write_text(module_code, encoding="utf-8")
            pth_path.write_text("import tcm_transformers_compat\n", encoding="utf-8")
        except OSError as exc:
            print("skip unwritable site-packages:", site_dir, repr(exc))
            continue
        else:
            print("installed transformers compat shim:", module_path)
            break
else:
    raise RuntimeError(f"No writable site-packages found from: {site_dirs}")
PY

"${PYTHON_BIN}" - <<'PY'
from pathlib import Path

import transformers

init_path = Path(transformers.__file__)
marker = "# TCM compatibility alias for verl AutoModelForVision2Seq"
patch = """

{marker}
try:
    AutoModelForVision2Seq
except NameError:
    try:
        AutoModelForVision2Seq = AutoModelForImageTextToText
    except NameError:
        try:
            from transformers.models.auto.modeling_auto import AutoModelForImageTextToText as AutoModelForVision2Seq
        except Exception:
            pass
""".format(marker=marker)

text = init_path.read_text(encoding="utf-8")
if marker not in text:
    init_path.write_text(text.rstrip() + patch + "\n", encoding="utf-8")
    print("patched transformers __init__:", init_path)
else:
    print("transformers __init__ already patched:", init_path)

import transformers as _check
print("post-patch has AutoModelForVision2Seq:", hasattr(_check, "AutoModelForVision2Seq"))
PY

"${PYTHON_BIN}" - <<PY
from pathlib import Path

model_py = Path("${VERL_DIR}") / "verl" / "utils" / "model.py"
text = model_py.read_text(encoding="utf-8")
if "AutoModelForVision2Seq" in text:
    patched = text.replace("AutoModelForVision2Seq", "AutoModelForImageTextToText")
    model_py.write_text(patched, encoding="utf-8")
    print("patched verl transformers import:", model_py)
else:
    print("verl transformers import already compatible:", model_py)
PY

# Keep smoke short without relying on trainer.total_training_steps support.
if [ ! -f "${TRAIN_FILE}" ] || [ ! -f "${TEST_FILE}" ]; then
    "${PYTHON_BIN}" - <<PY
import pandas as pd
train = pd.read_parquet("${FULL_TRAIN_FILE}").head(24)
test = pd.read_parquet("${FULL_TEST_FILE}").head(8)
train.to_parquet("${TRAIN_FILE}", index=False)
test.to_parquet("${TEST_FILE}", index=False)
print("wrote", "${TRAIN_FILE}", len(train))
print("wrote", "${TEST_FILE}", len(test))
PY
fi

start_time=$(date +%Y%m%d)_$(date +%H%M%S)

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="${TRAIN_FILE}"
    data.val_files="${TEST_FILE}"
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.filter_overlong_prompts=True
    data.truncation='error'
    data.shuffle=False
)

REWARD=(
    custom_reward_function.path="${REWARD_PATH}"
    custom_reward_function.name=compute_score
    reward_model.reward_manager=naive
)

MODEL=(
    actor_rollout_ref.model.path=${MODEL_PATH}
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF}
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.use_torch_compile=False
    actor_rollout_ref.actor.strategy=fsdp2
    actor_rollout_ref.actor.use_dynamic_bsz=False
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${FSDP_SIZE}
    actor_rollout_ref.actor.fsdp_config.reshard_after_forward=True
    actor_rollout_ref.actor.fsdp_config.entropy_checkpointing=True
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=True
    actor_rollout_ref.actor.fsdp_config.offload_policy=True
    actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=${SP_SIZE}
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
)

REF=(
    actor_rollout_ref.ref.strategy=fsdp2
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.ref.fsdp_config.param_offload=True
    actor_rollout_ref.ref.fsdp_config.reshard_after_forward=True
    actor_rollout_ref.ref.entropy_from_logits_with_chunking=True
    actor_rollout_ref.ref.fsdp_config.ulysses_sequence_parallel_size=${SP_SIZE}
    actor_rollout_ref.ref.use_torch_compile=False
    actor_rollout_ref.ref.fsdp_config.offload_policy=True
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=${INFER_BACKEND}
    actor_rollout_ref.rollout.ignore_eos=False
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.tensor_model_parallel_size=${GEN_TP}
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.max_num_batched_tokens=8192
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.enforce_eager=False
    actor_rollout_ref.rollout.enable_prefix_caching=False
)

RAY_ENV=(
    +ray_kwargs.ray_init.runtime_env.env_vars.CUDA_HOME="${CUDA_HOME}"
    +ray_kwargs.ray_init.runtime_env.env_vars.LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
    +ray_kwargs.ray_init.runtime_env.env_vars.PATH="${PATH}"
    +ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH="${PYTHONPATH:-}"
)

TRAINER=(
    trainer.critic_warmup=0
    trainer.logger='["console","wandb"]'
    trainer.project_name="${PROJECT_NAME}"
    trainer.experiment_name="${EXPERIMENT_NAME}"
    trainer.n_gpus_per_node=${NDEVICES_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.balance_batch=False
    trainer.val_before_train=False
    trainer.save_freq=${SAVE_FREQ}
    trainer.test_freq=${TEST_FREQ}
    trainer.total_epochs=${TOTAL_EPOCHS}
)

"${PYTHON_BIN}" -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${REWARD[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${REF[@]}" \
    "${ROLLOUT[@]}" \
    "${RAY_ENV[@]}" \
    "${TRAINER[@]}" \
    "$@" 2>&1 | tee logs/tcm-qwen3_5-9b-smoke-${start_time}.log
