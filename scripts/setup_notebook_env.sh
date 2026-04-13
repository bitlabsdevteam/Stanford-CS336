#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
PYTHON_BIN="${PYTHON_BIN:-/opt/homebrew/bin/python3.12}"
REQUIREMENTS_IN="${ROOT_DIR}/requirements-notebooks.in"
LOCK_FILE="${ROOT_DIR}/requirements-notebooks.lock.txt"
KERNEL_NAME="${KERNEL_NAME:-llm-assignments}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Expected Python interpreter not found at ${PYTHON_BIN}" >&2
  echo "Set PYTHON_BIN=/path/to/python3.12 and re-run this script." >&2
  exit 1
fi

if [[ ! -f "${REQUIREMENTS_IN}" ]]; then
  echo "Missing requirements file: ${REQUIREMENTS_IN}" >&2
  exit 1
fi

if [[ ! -d "${VENV_DIR}" ]]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "${REQUIREMENTS_IN}"
python -m pip freeze --exclude-editable | sort > "${LOCK_FILE}"
python -m ipykernel install --user --name "${KERNEL_NAME}" --display-name "Python (${KERNEL_NAME})"

echo
echo "Virtual environment ready: ${VENV_DIR}"
echo "Lockfile written to: ${LOCK_FILE}"
echo "Notebook kernel installed as: Python (${KERNEL_NAME})"
echo
echo "Activate with:"
echo "  source \"${VENV_DIR}/bin/activate\""
