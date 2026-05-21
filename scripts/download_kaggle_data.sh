#!/usr/bin/env bash
# Download dataset (requires Kaggle API: https://www.kaggle.com/docs/api)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="${ROOT}/data/raw"
mkdir -p "${DATA_DIR}"

if ! command -v kaggle &>/dev/null; then
  echo "Install Kaggle CLI: pip install kaggle"
  echo "Configure ~/.kaggle/kaggle.json with your API token"
  exit 1
fi

kaggle datasets download -d agungpambudi/network-malware-detection-connection-analysis -p "${DATA_DIR}" --unzip
echo "Dataset saved under ${DATA_DIR}"
