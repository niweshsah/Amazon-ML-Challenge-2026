#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${1:-${SCRIPT_DIR}/indicxlit_model}"
ARCHIVE_PATH="${MODEL_DIR}/indicxlit-indic-en-v1.0.zip"
MODEL_URL="https://github.com/AI4Bharat/IndicXlit/releases/download/v1.0/indicxlit-indic-en-v1.0.zip"

mkdir -p "${MODEL_DIR}"
if [[ ! -f "${ARCHIVE_PATH}" ]]; then
  curl -fL "${MODEL_URL}" -o "${ARCHIVE_PATH}"
fi
unzip -o "${ARCHIVE_PATH}" -d "${MODEL_DIR}"

printf 'IndicXlit model extracted under %s\n' "${MODEL_DIR}"
printf 'Expected files: corpus-bin/, transformer/indicxlit.pt, lang_list.txt\n'
