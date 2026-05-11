#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV_NAME="${CONDA_ENV_NAME:-cli_agent}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLI_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

run_py() {
    conda run --no-capture-output -n "$CONDA_ENV_NAME" python "$@"
}

INDEX_CONFIG_YAML="${INDEX_CONFIG_YAML:-$SCRIPT_DIR/config.yaml}"
CONFIG_YAML="$INDEX_CONFIG_YAML"
CONFIG_VARS="$(run_py "$CLI_ROOT/train/load_config_yaml.py" "$CONFIG_YAML")"
eval "$CONFIG_VARS"
RESUME_FOR_INFERENCE="${RESUME_FOR_INFERENCE:-${resume_for_inference:-}}"
if [ -n "$RESUME_FOR_INFERENCE" ]; then
    CONFIG_YAML="$(dirname "$RESUME_FOR_INFERENCE")/config.yaml"
    CONFIG_VARS="$(run_py "$CLI_ROOT/train/load_config_yaml.py" "$CONFIG_YAML")"
    eval "$CONFIG_VARS"
fi

BACKEND="${BACKEND:-${backend:-local}}"
if [ "$BACKEND" != "local" ]; then
    echo "ERROR: Only local SGLang is supported (got BACKEND=$BACKEND). Use a local sglang server and backend: local in infer/config.yaml."
    exit 1
fi

# settings.port in infer config YAML (flattened to `port` by load_config_yaml.py). Override with LOCAL_SGLANG_URL.
SGLANG_HTTP_PORT="${port:-30001}"
export LOCAL_SGLANG_URL="${LOCAL_SGLANG_URL:-http://127.0.0.1:${SGLANG_HTTP_PORT}}"
export LOCAL_SGLANG_MODEL="${LOCAL_SGLANG_MODEL:-__auto__}"
export LOCAL_SGLANG_KEY_ENV="${LOCAL_SGLANG_KEY_ENV:-LOCAL_SGLANG_UNUSED_API_KEY}"

API_BASE_URL="$LOCAL_SGLANG_URL"
API_MODEL="$LOCAL_SGLANG_MODEL"
API_KEY_ENV="$LOCAL_SGLANG_KEY_ENV"
API_MESSAGE_STYLE="plain"

BENCH="${BENCH:-${bench:-}}"
MAX_SAMPLES="${MAX_SAMPLES:-${max_samples:-0}}"
BATCH_SIZE="${BATCH_SIZE:-${batch_size:-${val_batch_size:-8}}}"
ROLLOUT_N="${ROLLOUT_N:-${rollout_n:-${val_rollout_n:-1}}}"
DO_SAMPLE="${DO_SAMPLE:-${do_sample:-${val_do_sample:-0}}}"
MAX_STEPS="${MAX_STEPS:-${max_steps:-${env_max_steps:-6}}}"
HISTORY_LENGTH="${HISTORY_LENGTH:-${history_length:-${env_history_length:-$MAX_STEPS}}}"
EXEC_BACKEND="${EXEC_BACKEND:-${exec_backend:-sandbox}}"
EXECUTE_COMMANDS="${EXECUTE_COMMANDS:-${execute_commands:-1}}"
NO_PROGRESS_ON_ANSWER="${NO_PROGRESS_ON_ANSWER:-${no_progress_on_answer:-1}}"
ENABLE_REGEX_CONSTRAINT="${ENABLE_REGEX_CONSTRAINT:-${enable_regex_constraint:-1}}"
LLM_TIMEOUT="${LLM_TIMEOUT:-${llm_timeout:-120}}"
DISABLE_QWEN_THINKING="${DISABLE_QWEN_THINKING:-${disable_qwen_thinking:-0}}"
TEMPERATURE="${TEMPERATURE:-${temperature:-${val_temperature:-1.0}}}"
TOP_P="${TOP_P:-${top_p:-${val_top_p:-1.0}}}"
TIMEOUT="${TIMEOUT:-${timeout:-10}}"
SEED="${SEED:-${seed:-${env_seed:-0}}}"
REGEX_PLAN_LEN="${REGEX_PLAN_LEN:-${regex_plan_len:-256}}"
REGEX_CODE_LEN="${REGEX_CODE_LEN:-${regex_code_len:-192}}"
REGEX_ANSWER_LEN="${REGEX_ANSWER_LEN:-${regex_answer_len:-128}}"
REGEX_JSON_OVERHEAD=128
CODE_RESPONSE_LEN=$((REGEX_PLAN_LEN + REGEX_CODE_LEN + REGEX_JSON_OVERHEAD))
ANSWER_RESPONSE_LEN=$((REGEX_PLAN_LEN + REGEX_ANSWER_LEN + REGEX_JSON_OVERHEAD))
if [ "$CODE_RESPONSE_LEN" -ge "$ANSWER_RESPONSE_LEN" ]; then
    MAX_TOKENS="$CODE_RESPONSE_LEN"
else
    MAX_TOKENS="$ANSWER_RESPONSE_LEN"
fi

PARQUET="${PARQUET:-}"

export BASH_CODING_REGEX_PLAN_LEN="$REGEX_PLAN_LEN"
export BASH_CODING_REGEX_CODE_LEN="$REGEX_CODE_LEN"
export BASH_CODING_REGEX_ANSWER_LEN="$REGEX_ANSWER_LEN"

# shellcheck source=../tooling/semantic_similarity/sglang_port_registry.sh
source "$CLI_ROOT/tooling/semantic_similarity/sglang_port_registry.sh"
SGLANG_LOG_DIR="$CLI_ROOT/tooling/semantic_similarity/logs"
SGLANG_REG_FILE="$(sglang_registry_file_for_port "$SGLANG_LOG_DIR" "$SGLANG_HTTP_PORT")"
MODEL_TAG="$(sglang_read_model_tag_from_registry "$SGLANG_REG_FILE")"
if [ -z "$MODEL_TAG" ]; then
    SGLANG_LAUNCH_SCRIPT="$CLI_ROOT/tooling/semantic_similarity/launch_sglang.sh"
    MODEL_PATH=""
    if grep -qE '^model_path=' "$SGLANG_LAUNCH_SCRIPT" 2>/dev/null; then
        MODEL_PATH="$(grep -E '^model_path=' "$SGLANG_LAUNCH_SCRIPT" | head -n 1 | cut -d'=' -f2- | tr -d '\"')"
    fi
    if [ -n "$MODEL_PATH" ]; then
        MODEL_TAG="$(basename "$MODEL_PATH")"
    else
        MODEL_TAG="$API_MODEL"
    fi
fi
if [ -z "$MODEL_TAG" ]; then
    MODEL_TAG="$API_MODEL"
fi
MODEL_TAG="$(printf '%s' "$MODEL_TAG" | tr '/: ' '___' | tr -cd '[:alnum:]_.-')"
RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
BENCH_PARQUET_LINES="$(
    BENCH="$BENCH" PARQUET="$PARQUET" python - <<'PY'
import json
import os

bench_raw = os.environ["BENCH"]
parquet_raw = os.environ["PARQUET"]

if bench_raw.startswith("["):
    benches = [str(item) for item in json.loads(bench_raw)]
else:
    benches = [bench_raw]

if parquet_raw:
    if parquet_raw.startswith("["):
        parquets = [str(item) for item in json.loads(parquet_raw)]
        if len(parquets) != len(benches):
            raise ValueError("PARQUET list must align with BENCH list length.")
    else:
        if len(benches) != 1:
            raise ValueError("Multi-bench infer requires empty PARQUET or a PARQUET list.")
        parquets = [parquet_raw]
else:
    parquets = [f"main_entry/data/{bench}/test.parquet" for bench in benches]

for bench, parquet in zip(benches, parquets, strict=True):
    print(f"{bench}\t{parquet}")
PY
)"
mapfile -t BENCH_PARQUET_ARRAY <<< "$BENCH_PARQUET_LINES"
BENCH_TAG="$(printf '%s' "$BENCH" | tr '[]\", ' '_' | tr -s '_' | sed 's/^_//;s/_$//')"
RESULT_SUBDIR="${BENCH_TAG}_${MODEL_TAG}__${RUN_TIMESTAMP}"
if [ -n "${INFER_RESULTS_DIR_PREFIX:-}" ]; then
    RESULT_SUBDIR="${INFER_RESULTS_DIR_PREFIX}_${RESULT_SUBDIR}"
fi

if [ -n "$RESUME_FOR_INFERENCE" ]; then
    OUTPUT_JSONL="${OUTPUT_JSONL:-$RESUME_FOR_INFERENCE}"
elif [ -n "$output_jsonl" ]; then
    OUTPUT_JSONL="${OUTPUT_JSONL:-$output_jsonl}"
else
    OUTPUT_JSONL="${OUTPUT_JSONL:-main_entry/cli_agent_bash_coding/infer/results/${RESULT_SUBDIR}/${BENCH_TAG}_${MODEL_TAG}_${RUN_TIMESTAMP}.jsonl}"
fi
OUTPUT_DIR="$(dirname "$OUTPUT_JSONL")"
mkdir -p "$OUTPUT_DIR"
if [ -z "$RESUME_FOR_INFERENCE" ]; then
    cp "$CONFIG_YAML" "$OUTPUT_DIR/$(basename "$CONFIG_YAML")"
fi

for bench_parquet in "${BENCH_PARQUET_ARRAY[@]}"; do
    CURRENT_BENCH="${bench_parquet%%$'\t'*}"
    CURRENT_PARQUET="${bench_parquet#*$'\t'}"
    echo "Running pure_infer for bench=${CURRENT_BENCH} parquet=${CURRENT_PARQUET}"
    run_py -u \
        "$CLI_ROOT/infer/pure_infer.py" \
        --parquet "$CURRENT_PARQUET" \
        --output_jsonl "$OUTPUT_JSONL" \
        --config_yaml "$CONFIG_YAML" \
        --api_base_url "$API_BASE_URL" \
        --api_model "$API_MODEL" \
        --api_key_env "$API_KEY_ENV" \
        --api_message_style "$API_MESSAGE_STYLE" \
        --backend_name "$BACKEND" \
        --max_samples "$MAX_SAMPLES" \
        --batch_size "$BATCH_SIZE" \
        --rollout_n "$ROLLOUT_N" \
        --do_sample "$DO_SAMPLE" \
        --max_steps "$MAX_STEPS" \
        --history_length "$HISTORY_LENGTH" \
        --exec_backend "$EXEC_BACKEND" \
        --execute_commands "$EXECUTE_COMMANDS" \
        --no_progress_on_answer "$NO_PROGRESS_ON_ANSWER" \
        --enable_regex_constraint "$ENABLE_REGEX_CONSTRAINT" \
        --llm_timeout "$LLM_TIMEOUT" \
        --disable_qwen_thinking "$DISABLE_QWEN_THINKING" \
        --temperature "$TEMPERATURE" \
        --top_p "$TOP_P" \
        --max_tokens "$MAX_TOKENS" \
        --timeout "$TIMEOUT" \
        --seed "$SEED"
done
