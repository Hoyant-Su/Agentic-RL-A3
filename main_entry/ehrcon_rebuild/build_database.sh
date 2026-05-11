#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MIMIC_DIR="${MIMIC_DIR:?set MIMIC_DIR to MIMIC-III 1.4 directory (CSVs)}"
exec python3 -u "$SCRIPT_DIR/build_ehrcon_database.py" "$MIMIC_DIR"
