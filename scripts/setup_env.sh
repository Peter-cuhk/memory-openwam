#!/usr/bin/env bash
# Reuse Python/CUDA/PyTorch from the base image; install additions on CPFS only.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "${OPENWAM_ROOT}"
BASE_PYTHON=/usr/local/bin/python3
"${BASE_PYTHON}" - <<'PY'
import sys
import torch
from torch.utils.cpp_extension import CUDA_HOME, get_compiler_abi_compatibility_and_version
print(f"Base Python: {sys.version}")
print(f"Base torch: {torch.__version__}; CUDA: {torch.version.cuda}; toolkit: {CUDA_HOME}")
print("C++ compiler:", get_compiler_abi_compatibility_and_version("g++"))
PY
"${CUDA_HOME}/bin/nvcc" --version
mkdir -p "${UV_CACHE_DIR}" "${PIP_CACHE_DIR}" "${TMPDIR}" "${WANDB_DIR}" \
    "${HF_HOME}" "${TORCH_HOME}" "${TORCH_EXTENSIONS_DIR}" \
    "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}" "${CUDA_CACHE_PATH}" \
    "${WANDB_CACHE_DIR}" "${WANDB_CONFIG_DIR}" "${XDG_CACHE_HOME}" \
    "${MODELSCOPE_CACHE}" "${NUMBA_CACHE_DIR}" "${RUFF_CACHE_DIR}" "${PRE_COMMIT_HOME}"
if [[ ! -f .venv/pyvenv.cfg ]]; then
    uv venv --python "${BASE_PYTHON}" --no-python-downloads .venv
fi
# Link package code read-only in practice; copy metadata so uv recognizes it
# without exposing unrelated system packages in this isolated environment.
"${BASE_PYTHON}" - <<'PY'
import importlib.metadata as md
from pathlib import Path
import shutil
import sys

venv = Path('.venv').resolve()
site = venv / f'lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages'
pins = []
for name in ('torch', 'torchvision', 'torchaudio', 'triton'):
    dist = md.distribution(name)
    pins.append(f'{name}==={dist.version}')
    for root in sorted({p.parts[0] for p in dist.files}):
        source = Path(dist.locate_file(root)).resolve()
        target = site / root
        if root.startswith('.') or not source.exists() or target.exists():
            continue
        if root.endswith('.dist-info'):
            shutil.copytree(source, target)
        else:
            target.symlink_to(source, target_is_directory=source.is_dir())
    for ep in dist.entry_points:
        if ep.group != 'console_scripts':
            continue
        module, func = ep.value.split(':')
        script = venv / 'bin' / ep.name
        script.write_text(f'#!{venv}/bin/python\nimport sys\nfrom {module} import {func}\n'
                          f'if __name__ == "__main__":\n    sys.exit({func}())\n')
        script.chmod(0o755)
(venv / 'base-constraints.txt').write_text('\n'.join(pins) + '\n')
PY
source scripts/env.sh
UV_INDEX_ARGS=(
    --index-url "${PIP_INDEX_URL}"
    --extra-index-url "${PIP_EXTRA_INDEX_URL}"
    --index-strategy unsafe-best-match
)
uv pip install --python .venv/bin/python "${UV_INDEX_ARGS[@]}" \
    pip setuptools wheel packaging ninja \
    filelock 'typing-extensions>=4.10.0' networkx jinja2 fsspec 'sympy==1.13.1' numpy pillow
install_args=(-e '.[dev]' -c .venv/base-constraints.txt)
if [[ -f requirements/native.txt ]]; then
    # native.txt was snapshotted against torch 2.9.0 + PPU DeepSpeed.
    # This image is torch 2.6.0 / cu128 / ubuntu2404 / cp312, which has no
    # matching PPU DeepSpeed binary. Use upstream DeepSpeed with DS_BUILD_OPS=0
    # and the sympy pin required by torch 2.6.0.
    grep -vE '^(sympy|SymPy|deepspeed|opencv-python)==' requirements/native.txt > .venv/native-adapted.txt
    {
        echo 'sympy==1.13.1'
        echo 'deepspeed==0.18.9'
        echo 'opencv-python-headless==4.14.0.94'
    } >> .venv/native-adapted.txt
    install_args+=(-r .venv/native-adapted.txt)
fi
DS_BUILD_OPS=0 uv pip install --python .venv/bin/python \
    --no-build-isolation \
    "${UV_INDEX_ARGS[@]}" \
    "${install_args[@]}"
# openwam metadata asks for opencv-python; this host has no libGL, so we ship headless.
uv pip check --python .venv/bin/python || true
python scripts/check_env.py
