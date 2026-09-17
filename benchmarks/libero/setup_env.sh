#!/usr/bin/env bash
# Create the isolated LIBERO client environment with uv on the mounted disk.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"
UV_BIN="${UV_BIN:-uv}"
ENV_PREFIX="${LIBERO_ENV_PREFIX}"
LIBERO_REMOTE="https://github.com/Lifelong-Robot-Learning/LIBERO.git"
LIBERO_COMMIT="8f1084e3132a39270c3a13ebe37270a43ece2a01"
REQUIREMENTS="${SCRIPT_DIR}/requirements.txt"
LIBERO_PATCH="${SCRIPT_DIR}/patches/libero-pytorch-load.patch"
UPSTREAM_REQUIREMENTS="${LIBERO_PATH}/requirements.txt"

# Never download a different Python implicitly. Install Python 3.10 on the
# mounted disk only after the repository's required user approval.
BASE_PYTHON="$("${UV_BIN}" --no-cache python find --no-project --system --no-python-downloads "${LIBERO_BASE_PYTHON:-3.10}")" || {
    echo "[ERROR] Python 3.10 is required by LIBERO's NumPy 1.22.4 pin." >&2
    echo "Install an approved Python 3.10 under ${UV_PYTHON_INSTALL_DIR}, or set LIBERO_BASE_PYTHON to an existing interpreter." >&2
    exit 1
}
"${BASE_PYTHON}" - <<'PY'
import ctypes.util
import sys

if sys.version_info[:2] != (3, 10):
    raise SystemExit(f"LIBERO requires Python 3.10, got {sys.version}")
print(f"Client Python: {sys.executable} ({sys.version.split()[0]})")
print("Client torch: 2.7.1+cpu; no CUDA toolkit or model-server torch required")
print(f"System render libraries: EGL={ctypes.util.find_library('EGL')}, OSMesa={ctypes.util.find_library('OSMesa')}")
PY

if [[ -f "${ENV_PREFIX}/pyvenv.cfg" ]]; then
    [[ -f "${ENV_PREFIX}/pyvenv.cfg" && -x "${ENV_PREFIX}/bin/python" ]] || {
        echo "[ERROR] Existing path is not a virtual environment: ${ENV_PREFIX}" >&2
        exit 1
    }
    "${ENV_PREFIX}/bin/python" -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 10) else "Existing LIBERO environment must use Python 3.10")'
else
    # Python itself lives under ENV_PREFIX/python. Preserve it when creating
    # the virtual environment in the same parent directory.
    "${UV_BIN}" venv --allow-existing --python "${BASE_PYTHON}" --no-python-downloads "${ENV_PREFIX}"
fi

if [[ ! -d "${LIBERO_PATH}/.git" ]]; then
    git clone "${LIBERO_REMOTE}" "${LIBERO_PATH}"
    git -C "${LIBERO_PATH}" checkout --detach "${LIBERO_COMMIT}"
fi

actual_commit="$(git -C "${LIBERO_PATH}" rev-parse HEAD)"
[[ "${actual_commit}" == "${LIBERO_COMMIT}" ]] || {
    echo "[ERROR] Expected LIBERO commit ${LIBERO_COMMIT}, found ${actual_commit}" >&2
    exit 1
}

if git -C "${LIBERO_PATH}" apply --reverse --check "${LIBERO_PATCH}" >/dev/null 2>&1; then
    echo "[setup] LIBERO PyTorch compatibility patch already applied"
elif git -C "${LIBERO_PATH}" apply --check "${LIBERO_PATCH}" >/dev/null 2>&1; then
    git -C "${LIBERO_PATH}" apply "${LIBERO_PATCH}"
else
    echo "[ERROR] LIBERO checkout has incompatible local changes" >&2
    exit 1
fi

# Match pip's previous selection across these two explicit public indexes.
# Resolve upstream and client requirements together so upstream installation
# cannot replace the CPU-only torch or the simulator pins afterwards.
"${UV_BIN}" --no-config pip install --python "${ENV_PREFIX}/bin/python" \
    --default-index https://pypi.org/simple \
    --index https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match \
    --requirement "${REQUIREMENTS}" --requirement "${UPSTREAM_REQUIREMENTS}"
"${UV_BIN}" --no-config pip install --python "${ENV_PREFIX}/bin/python" \
    --default-index https://pypi.org/simple --no-deps --editable "${LIBERO_PATH}"
# LIBERO's setup.py uses a namespace-style outer ``libero/`` directory that
# modern PEP 660 editable discovery leaves unmapped. Pin the checkout root on
# sys.path explicitly so ``import libero`` remains valid after a fresh install.
site_packages="$("${ENV_PREFIX}/bin/python" -c 'import site; print(site.getsitepackages()[0])')"
printf '%s\n' "${LIBERO_PATH}" > "${site_packages}/libero_source.pth"
"${UV_BIN}" pip check --python "${ENV_PREFIX}/bin/python"
"${ENV_PREFIX}/bin/python" - "${REQUIREMENTS}" <<'PY'
import importlib.metadata as metadata
import os
import sys
from pathlib import Path

import mujoco
from packaging.requirements import Requirement

for line in Path(sys.argv[1]).read_text().splitlines():
    if not line.strip() or line.lstrip().startswith("#"):
        continue
    requirement = Requirement(line)
    actual = metadata.version(requirement.name)
    if actual not in requirement.specifier:
        raise RuntimeError(f"{requirement.name}: expected {requirement.specifier}, found {actual}")

import libero

expected_package_root = (Path(os.environ["LIBERO_PATH"]).resolve() / "libero").resolve()
if not any(Path(path).resolve() == expected_package_root for path in libero.__path__):
    raise RuntimeError(f"Unexpected LIBERO import paths: {list(libero.__path__)}")
print("LIBERO evaluation environment ready with official requirements and MuJoCo 3.3.2")
PY

LIBERO_PYTHON="${ENV_PREFIX}/bin/python" bash "${SCRIPT_DIR}/run_smoke.sh" task
