#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"

if [[ ! -d "${VENV_DIR}" ]]; then
  python3 -m venv "${VENV_DIR}"
fi

# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip
python -m pip install -r "${ROOT_DIR}/requirements-dev.txt"

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

python -m compileall -q "${ROOT_DIR}/custom_components/ai_subscription_assist"
ruff check "${ROOT_DIR}/custom_components/ai_subscription_assist" "${ROOT_DIR}/tests"
pytest -q "${ROOT_DIR}/tests"
