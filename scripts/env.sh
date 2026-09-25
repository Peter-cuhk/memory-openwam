#!/usr/bin/env bash
# Source this file in every training/deployment job using the shared base image.
OPENWAM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export OPENWAM_ROOT
export PATH="/mnt/cpfs/workspace/tools:${PATH}"

# PPU CUDA toolkit (HGML). Must run before CUDA_HOME is consumed by setup_env.sh.
if [[ -f /usr/local/PPU_SDK/envsetup.sh ]]; then
    _openwam_nounset=0
    case $- in *u*) _openwam_nounset=1; set +u ;; esac
    # shellcheck disable=SC1091
    source /usr/local/PPU_SDK/envsetup.sh cuda
    if [[ "${_openwam_nounset}" -eq 1 ]]; then set -u; fi
    unset _openwam_nounset
fi

export OPENWAM_CACHE_ROOT="${OPENWAM_CACHE_ROOT:-/mnt/cpfs/workspace/data/openwam-cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/mnt/cpfs/uv_cache}"
export PIP_CACHE_DIR="${OPENWAM_CACHE_ROOT}/pip"
export XDG_CACHE_HOME="${OPENWAM_CACHE_ROOT}/xdg"
export HF_HOME="${HF_HOME:-/mnt/cpfs/workspace/data/hf_cache}"
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
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.wandb.ai}"
export TMPDIR="${TMPDIR:-${OPENWAM_CACHE_ROOT}/tmp}"
# PPU DeepSpeed lives on pg1-pip; everything else comes from the extra index.
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://aiext-pypi.mirrors.aliyuncs.com/pg1-pip/ubuntu_cu128/simple/}"
export PIP_EXTRA_INDEX_URL="${PIP_EXTRA_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple/}"
export UV_EXTRA_INDEX_URL="${UV_EXTRA_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple/}"
if [[ -f "${OPENWAM_ROOT}/.env" ]]; then
    source "${OPENWAM_ROOT}/.env"
    if [[ -n "${WANDB_API_KEY:-}" ]]; then
        export WANDB_API_KEY
    fi
elif [[ -f /mnt/cpfs/workspace/env/wandb.env ]]; then
    # shellcheck disable=SC1091
    source /mnt/cpfs/workspace/env/wandb.env
fi
export UV_LINK_MODE=copy
export UV_PYTHON_DOWNLOADS=never
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PIP_REQUIRE_VIRTUALENV=true
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
if [[ -f "${OPENWAM_ROOT}/.venv/bin/activate" ]]; then
    source "${OPENWAM_ROOT}/.venv/bin/activate"
fi

