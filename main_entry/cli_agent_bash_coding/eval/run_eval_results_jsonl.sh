#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_DIR"

RESULTS_JSONL="${RESULTS_JSONL:-main_entry/cli_agent_bash_coding/results/gspo_agentbench_os_databench_shellops_ehrcon_curated_agentbench_dbbench_tablebench__Qwen3-4B/20260501_175203/shellops_pro_harness_vanilla_step_6/34.jsonl}"

PARQUET="${PARQUET:-}"

RESULTS_JSONL="${1:-$RESULTS_JSONL}"
PARQUET="${2:-$PARQUET}"
EVAL_WORKERS="${EVAL_WORKERS:-32}"

ARGS=(--results_jsonl "$RESULTS_JSONL")
[ -n "$PARQUET" ] && ARGS+=(--parquet "$PARQUET")
ARGS+=(--workers "$EVAL_WORKERS")

RESULT_DIR="$(dirname "$RESULTS_JSONL")"
python3 main_entry/cli_agent_bash_coding/eval/merge_results.py "$RESULT_DIR"
