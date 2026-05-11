#!/usr/bin/env bash
# Thin wrapper: same as ../launch_pure_infer.sh but defaults to this directory's config.yaml (rstar_infer.*).
set -euo pipefail
RS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export INDEX_CONFIG_YAML="$RS_DIR/config.yaml"
export INFER_RESULTS_DIR_PREFIX="${INFER_RESULTS_DIR_PREFIX:-rstar}"
exec bash "$RS_DIR/../launch_pure_infer.sh" "$@"
