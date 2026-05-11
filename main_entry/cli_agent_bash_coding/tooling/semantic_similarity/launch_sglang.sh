#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/sglang_port_registry.sh"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

TP_SIZE=$(nvidia-smi -L | wc -l)
model_path="zai-org/GLM-5.1-FP8"

SGLANG_PORT=30033
REG_FILE="$(sglang_registry_file_for_port "$LOG_DIR" "$SGLANG_PORT")"
if [ -f "$REG_FILE" ]; then
  echo "ERROR: SGLang port ${SGLANG_PORT} is already registered (refusing to start). File: $REG_FILE"
  echo "If the old process is gone, remove this file and retry."
  echo "--- existing registry ---"
  cat "$REG_FILE"
  exit 1
fi

model_tag="$(basename "$model_path")"
{
  echo "model_path=$model_path"
  echo "model_tag=$model_tag"
  echo "port=$SGLANG_PORT"
  echo "registered_at=$(date -Iseconds 2>/dev/null || date)"
} > "$REG_FILE"

python -m sglang.launch_server \
  --model-path "$model_path" \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port "$SGLANG_PORT" \
  --mem-fraction-static 0.9 \
  --tp-size "$TP_SIZE" \
  --cuda-graph-max-bs 16
