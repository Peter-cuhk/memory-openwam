#!/usr/bin/env bash
# Shared client paths; does not activate or modify the model-server environment.
LIBERO_BENCHMARK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIBERO_REPO_ROOT="$(cd "${LIBERO_BENCHMARK_DIR}/../.." && pwd)"
export LIBERO_ENV_PREFIX="${LIBERO_ENV_PREFIX:-/mnt/cpfs/feiyang/envs/libero}"
export LIBERO_PYTHON="${LIBERO_PYTHON:-${LIBERO_ENV_PREFIX}/bin/python}"
export LIBERO_PATH="${LIBERO_PATH:-${LIBERO_REPO_ROOT}/third_party/LIBERO}"
export LIBERO_CONFIG_ROOT="${LIBERO_CONFIG_ROOT:-${LIBERO_REPO_ROOT}/outputs/libero/runtime}"
export UV_CACHE_DIR=/mnt/cpfs/feiyang/cache/uv
export PIP_CACHE_DIR=/mnt/cpfs/feiyang/cache/pip
export XDG_CACHE_HOME=/mnt/cpfs/feiyang/cache/xdg
export HF_HOME=/mnt/cpfs/feiyang/cache/huggingface
export NUMBA_CACHE_DIR=/mnt/cpfs/feiyang/cache/numba
export UV_PYTHON_INSTALL_DIR="${LIBERO_ENV_PREFIX}/python"
export UV_PYTHON_DOWNLOADS=never
export UV_LINK_MODE=copy
export PYTHONNOUSERSITE=1
