#!/usr/bin/env bash
set -euo pipefail

# Create a clean LLaMA-Factory environment for Qwen3.5 multimodal full SFT.
# Run from TCM_final on the server:
#   bash llamafactory_qwen35/scripts/setup_llamafactory_env.sh
#
# Optional:
#   ENV_NAME=qwen35_ft PYTHON_VERSION=3.10 bash llamafactory_qwen35/scripts/setup_llamafactory_env.sh

ENV_NAME="${ENV_NAME:-qwen35_ft}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
TORCH_CUDA="${TORCH_CUDA:-cu121}"
LLAMA_FACTORY_DIR="${LLAMA_FACTORY_DIR:-LLaMA-Factory}"

if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
  fi
  conda activate "${ENV_NAME}"
else
  VENV_DIR="${VENV_DIR:-.venv_qwen35_ft}"
  "${PYTHON_BIN:-python3}" -m venv "${VENV_DIR}"
  source "${VENV_DIR}/bin/activate"
fi

python -m pip install -U pip setuptools wheel packaging ninja
python -m pip install --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}" torch torchvision torchaudio
python -m pip install -U deepspeed datasets accelerate peft trl sentencepiece protobuf einops pillow qwen-vl-utils

if [[ ! -d "${LLAMA_FACTORY_DIR}" ]]; then
  git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git "${LLAMA_FACTORY_DIR}"
fi

cd "${LLAMA_FACTORY_DIR}"
git pull --ff-only || true
python -m pip install -e .

# Keep Transformers aligned with the current LLaMA-Factory constraint.
python -m pip install -U "transformers==5.6.0"

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("torch cuda:", torch.version.cuda)
if torch.cuda.is_available():
    print("gpu count:", torch.cuda.device_count())
PY

llamafactory-cli version || true

echo
echo "Environment ready."
echo "Activate it with: conda activate ${ENV_NAME}"
