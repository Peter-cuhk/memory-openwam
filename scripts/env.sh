#!/usr/bin/env bash
# Source this file in every training/deployment job using the shared base image.
OPENWAM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export OPENWAM_ROOT
export OPENWAM_CACHE_ROOT=/mnt/cpfs/feiyang/cache
export UV_CACHE_DIR="${OPENWAM_CACHE_ROOT}/uv"
export PIP_CACHE_DIR="${OPENWAM_CACHE_ROOT}/pip"
export XDG_CACHE_HOME="${OPENWAM_CACHE_ROOT}/xdg"
export HF_HOME="${OPENWAM_CACHE_ROOT}/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export MODELSCOPE_CACHE="${OPENWAM_CACHE_ROOT}/modelscope"
export TORCH_HOME="${OPENWAM_CACHE_ROOT}/torch"
export TORCH_EXTENSIONS_DIR="${OPENWAM_CACHE_ROOT}/openwam/torch_extensions"
export TORCHINDUCTOR_CACHE_DIR="${OPENWAM_CACHE_ROOT}/openwam/torchinductor"
export TRITON_CACHE_DIR="${OPENWAM_CACHE_ROOT}/openwam/triton"
export CUDA_CACHE_PATH="${OPENWAM_CACHE_ROOT}/cuda"
export NUMBA_CACHE_DIR="${OPENWAM_CACHE_ROOT}/numba"
export RUFF_CACHE_DIR="${OPENWAM_CACHE_ROOT}/ruff"
export PRE_COMMIT_HOME="${OPENWAM_CACHE_ROOT}/pre-commit"
export WANDB_CACHE_DIR="${OPENWAM_CACHE_ROOT}/wandb"
export WANDB_CONFIG_DIR="${WANDB_CACHE_DIR}/config"
export WANDB_DIR="${OPENWAM_ROOT}/outputs"
export WANDB_MODE="${WANDB_MODE:-offline}"
export UV_LINK_MODE=copy
export UV_PYTHON_DOWNLOADS=never
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PIP_REQUIRE_VIRTUALENV=true
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
if [[ -f "${OPENWAM_ROOT}/.venv/bin/activate" ]]; then
    source "${OPENWAM_ROOT}/.venv/bin/activate"
fi
