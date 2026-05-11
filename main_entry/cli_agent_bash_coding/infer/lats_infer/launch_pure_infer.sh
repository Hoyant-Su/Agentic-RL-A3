#!/usr/bin/env bash
# Thin wrapper: same as ../launch_pure_infer.sh but defaults to this directory's config.yaml (lats_infer.*).
set -euo pipefail
LATS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export INDEX_CONFIG_YAML="$LATS_DIR/config.yaml"
export INFER_RESULTS_DIR_PREFIX="${INFER_RESULTS_DIR_PREFIX:-lats}"
exec bash "$LATS_DIR/../launch_pure_infer.sh" "$@"
