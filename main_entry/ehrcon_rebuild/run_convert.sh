#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 -u "$SCRIPT_DIR/convert_ehrcon_to_agent_parquet.py" \
  --note-variant original \
  --db-mode hardlink \
  --max-per-hadm-note-type 3 \
  --max-per-note-type 250 \
  --overwrite \
  "$@"
