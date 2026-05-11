#!/bin/bash

set -eu

export CUDA_VISIBLE_DEVICES="$(nvidia-smi -L | awk 'BEGIN{ORS=","} {print NR-1}' | sed 's/,$//')"

export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
unset RAY_ADDRESS
python3 -m ray stop --force || true
NUM_GPUS=$(nvidia-smi -L | wc -l)

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$PROJECT_DIR"
JUDGER_LAUNCH_SCRIPT="$PROJECT_DIR/main_entry/cli_agent_bash_coding/tooling/semantic_similarity/launch_llm_as_a_judger.sh"
JUDGER_LOG_DIR="$PROJECT_DIR/main_entry/cli_agent_bash_coding/tooling/semantic_similarity/logs"
JUDGER_LOG_FILE="$JUDGER_LOG_DIR/llm_judger_30003.log"

MASTER_PORT_DEFAULT="$(python3 - <<'PY'
import socket
s = socket.socket()
s.bind(('', 0))
print(s.getsockname()[1])
s.close()
PY
)"
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT="$MASTER_PORT_DEFAULT"
export BASH_CODING_SEMANTIC_SIMILARITY_URL="http://127.0.0.1:30003"

if ! python3 - <<'PY'
import socket
sock = socket.socket()
sock.settimeout(1.0)
try:
    sock.connect(("127.0.0.1", 30003))
except OSError:
    raise SystemExit(1)
finally:
    sock.close()
PY
then
    mkdir -p "$JUDGER_LOG_DIR"
    nohup bash "$JUDGER_LAUNCH_SCRIPT" >"$JUDGER_LOG_FILE" 2>&1 &
    echo "Started LLM judger on port 30003: $JUDGER_LAUNCH_SCRIPT"
fi

if [ -v PYTHONPATH ]; then
    export PYTHONPATH="$PROJECT_DIR/main_entry:$PYTHONPATH"
else
    export PYTHONPATH="$PROJECT_DIR/main_entry"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_YAML="${BASH_CODING_CONFIG_YAML:-$SCRIPT_DIR/config.yaml}"
eval "$(python3 "$PROJECT_DIR/main_entry/cli_agent_bash_coding/train/load_config_yaml.py" "$CONFIG_YAML")"

REQUESTED_CHECKPOINT_PAIR="$checkpoint_pair"
REQUESTED_RESUME_FOR_TRAINING="$resume_for_training"

resolve_abs_path() {
    local path="$1"
    if [[ "$path" = /* ]]; then
        printf '%s\n' "$path"
    else
        printf '%s\n' "$PROJECT_DIR/$path"
    fi
}

MERGE_SOURCE_CONFIG_YAML="$(resolve_abs_path "$CONFIG_YAML")"

find_timestamp_dir() {
    local path
    path="$(resolve_abs_path "$1")"
    while [ "$path" != "/" ]; do
        local base
        base="$(basename "$path")"
        if [[ "$base" =~ ^[0-9]{8}_[0-9]{6}$ ]]; then
            printf '%s\n' "$path"
            return 0
        fi
        path="$(dirname "$path")"
    done
    return 1
}

CHECKPOINT_SOURCE_PATH=""
if [ -n "$REQUESTED_RESUME_FOR_TRAINING" ]; then
    CHECKPOINT_SOURCE_PATH="$REQUESTED_RESUME_FOR_TRAINING"
elif [ -n "$REQUESTED_CHECKPOINT_PAIR" ]; then
    IFS=':' read -r REQUESTED_CHECKPOINT_PATH _REQUESTED_PRETRAINED_MODEL_PATH <<< "$REQUESTED_CHECKPOINT_PAIR"
    CHECKPOINT_SOURCE_PATH="$REQUESTED_CHECKPOINT_PATH"
fi

CHECKPOINT_TIMESTAMP_DIR=""
CHECKPOINT_EXP_NAME=""
CHECKPOINT_RESULTS_DIR=""
if [ -n "$CHECKPOINT_SOURCE_PATH" ]; then
    CHECKPOINT_TIMESTAMP_DIR="$(find_timestamp_dir "$CHECKPOINT_SOURCE_PATH")"
    CHECKPOINT_EXP_NAME="$(basename "$(dirname "$CHECKPOINT_TIMESTAMP_DIR")")"
    CHECKPOINT_RESULTS_DIR="$PROJECT_DIR/main_entry/cli_agent_bash_coding/results/$CHECKPOINT_EXP_NAME/$(basename "$CHECKPOINT_TIMESTAMP_DIR")"
    CONFIG_YAML="$CHECKPOINT_RESULTS_DIR/config.yaml"
    if [ ! -f "$CONFIG_YAML" ]; then
        echo "ERROR: Missing historical config for checkpoint: $CONFIG_YAML"
        exit 1
    fi
    eval "$(python3 "$PROJECT_DIR/main_entry/cli_agent_bash_coding/train/load_config_yaml.py" "$CONFIG_YAML")"
    checkpoint_pair="$REQUESTED_CHECKPOINT_PAIR"
    resume_for_training="$REQUESTED_RESUME_FOR_TRAINING"
fi

CHECKPOINT_INFER_TRAIN_OVERRIDES_PY="$PROJECT_DIR/main_entry/cli_agent_bash_coding/train/print_train_checkpoint_infer_overrides.py"
CHECKPOINT_INFER_SUBDIR_PY="$PROJECT_DIR/main_entry/cli_agent_bash_coding/train/compute_checkpoint_infer_results_subdir.py"
MERGE_CHECKPOINT_INFER_CONFIG_PY="$PROJECT_DIR/main_entry/cli_agent_bash_coding/train/merge_checkpoint_infer_config.py"
BASH_CODING_RESULTS_NESTED_SUBDIR=""
if [ -n "${REQUESTED_CHECKPOINT_PAIR:-}" ] && [ -f "$MERGE_SOURCE_CONFIG_YAML" ] && [ -f "${CONFIG_YAML:-}" ]; then
    eval "$(python3 "$CHECKPOINT_INFER_TRAIN_OVERRIDES_PY" "$MERGE_SOURCE_CONFIG_YAML")"
    BASH_CODING_RESULTS_NESTED_SUBDIR="$(python3 "$CHECKPOINT_INFER_SUBDIR_PY" "$MERGE_SOURCE_CONFIG_YAML" "$REQUESTED_CHECKPOINT_PAIR" "$PROJECT_DIR")"
    if [ -n "${BASH_CODING_RESULTS_NESTED_SUBDIR:-}" ]; then
        echo "A3: results subdir: ${BASH_CODING_RESULTS_NESTED_SUBDIR} (train: bench+harness+steps; save config = results timestamp + env from train)"
    fi
fi

ENABLE_ALGO_KEY="enable_${algo}"
eval "ENABLE_ALGO_VALUE=\${$ENABLE_ALGO_KEY:-0}"
ACTOR_ALGO_ENABLE_ARG="+actor_rollout_ref.actor.${ENABLE_ALGO_KEY}=${ENABLE_ALGO_VALUE}"
TRAINER_ALGO_ENABLE_ARG="+algorithm.${ENABLE_ALGO_KEY}=${ENABLE_ALGO_VALUE}"

export SGLANG_CI_SMALL_KV_SIZE="$sglang_ci_small_kv_size"
export SGLANG_IS_FLASHINFER_AVAILABLE="$sglang_is_flashinfer_available"

mapfile -t BENCH_LIST < <(BENCH_LITERAL="$bench" python3 - <<'PY'
import json
import os

raw = os.environ["BENCH_LITERAL"]
try:
    value = json.loads(raw)
except json.JSONDecodeError:
    value = raw

if isinstance(value, str):
    benches = [value]
elif isinstance(value, list):
    benches = value
else:
    raise TypeError(f"Unsupported bench type: {type(value)}")

for bench in benches:
    if not isinstance(bench, str) or not bench:
        raise ValueError(f"Invalid bench entry: {bench!r}")
    print(bench)
PY
)

if [ "${#BENCH_LIST[@]}" -eq 0 ]; then
  echo "ERROR: bench must contain at least one dataset name"
  exit 1
fi

if [ "${#BENCH_LIST[@]}" -eq 1 ]; then
  BENCH_TAG="${BENCH_LIST[0]}"
else
  BENCH_TAG=""
  for bench_name in "${BENCH_LIST[@]}"; do
    BENCH_TAG="${BENCH_TAG}${bench_name}_"
  done
fi

TRAIN_FILES=()
VAL_FILES=()
for bench_name in "${BENCH_LIST[@]}"; do
  data_dir="main_entry/data/$bench_name"
  if [ ! -f "$data_dir/train.parquet" ] || [ ! -f "$data_dir/test.parquet" ]; then
    echo "ERROR: Missing parquet files in $data_dir"
    exit 1
  fi
  TRAIN_FILES+=("$data_dir/train.parquet")
  VAL_FILES+=("$data_dir/test.parquet")
done

TRAIN_FILES_LITERAL="$(python3 - <<'PY' "${TRAIN_FILES[@]}"
import json
import sys
paths = sys.argv[1:]
print(paths[0] if len(paths) == 1 else json.dumps(paths, separators=(",", ":")))
PY
)"
VAL_FILES_LITERAL="$(python3 - <<'PY' "${VAL_FILES[@]}"
import json
import sys
paths = sys.argv[1:]
print(paths[0] if len(paths) == 1 else json.dumps(paths, separators=(",", ":")))
PY
)"
ENABLE_MID_VAL="${enable_mid_val:-0}"

env_name="${algo}_${BENCH_TAG}"

date_time=$(date +%Y%m%d_%H%M%S)
checkpoint_path=""
pretrained_model_path="$pretrained_model_path_default"
if [ -n "$checkpoint_pair" ]; then
    IFS=':' read -r checkpoint_path pretrained_model_path <<< "$checkpoint_pair"
fi

exp_name="${env_name}_$(basename "$pretrained_model_path")"
if [ -n "$CHECKPOINT_EXP_NAME" ]; then
    exp_name="$CHECKPOINT_EXP_NAME"
fi

if [ -n "$CHECKPOINT_TIMESTAMP_DIR" ]; then
    checkpoint_dir="$(dirname "$(resolve_abs_path "$CHECKPOINT_SOURCE_PATH")")"
    if [[ "$(basename "$checkpoint_dir")" != "$(basename "$CHECKPOINT_TIMESTAMP_DIR")" ]]; then
        checkpoint_dir="$CHECKPOINT_TIMESTAMP_DIR"
    fi
    save_dir="main_entry/cli_agent_bash_coding/results/$exp_name/$(basename "$CHECKPOINT_TIMESTAMP_DIR")"
    run_id="$(basename "$CHECKPOINT_TIMESTAMP_DIR")"
else
    checkpoint_dir="main_entry/cli_agent_bash_coding/checkpoints/$exp_name/${date_time}"
    save_dir="main_entry/cli_agent_bash_coding/results/$exp_name/${date_time}"
    run_id="${date_time}"
fi
if [ -n "${BASH_CODING_RESULTS_NESTED_SUBDIR:-}" ]; then
    save_dir="${save_dir}/${BASH_CODING_RESULTS_NESTED_SUBDIR}"
fi
save_count="$save_count"

save_dir_abs="$(resolve_abs_path "$save_dir")"
allow_existing_results_dir=0
if [ -n "$resume_for_training" ]; then
    allow_existing_results_dir=1
elif [ -n "${RESUME_INFERENCE_DIR:-}" ]; then
    allow_existing_results_dir=1
elif [ "$bash_coding_enable" = "1" ] && { [ "$enable_pretrained_eval_only" = "1" ] || [ -n "$checkpoint_path" ]; }; then
    allow_existing_results_dir=1
fi
if [ -d "$save_dir_abs" ] && [ "$allow_existing_results_dir" = "0" ]; then
    echo "ERROR: Results save directory already exists: $save_dir_abs"
    exit 1
fi

tensorboard_dir="main_entry/cli_agent_bash_coding/tensorboard/$exp_name/${run_id}"
if [ -n "${BASH_CODING_RESULTS_NESTED_SUBDIR:-}" ]; then
    tensorboard_dir="${tensorboard_dir}/${BASH_CODING_RESULTS_NESTED_SUBDIR}"
fi
mkdir -p "$tensorboard_dir"
export TENSORBOARD_DIR="$tensorboard_dir"
mkdir -p "$save_dir"
SAVE_CONFIG_YAML="$save_dir/config.yaml"
if [ -n "$CHECKPOINT_TIMESTAMP_DIR" ] && [ -f "$CHECKPOINT_RESULTS_DIR/config.yaml" ] && [ -f "$MERGE_SOURCE_CONFIG_YAML" ]; then
  python3 "$MERGE_CHECKPOINT_INFER_CONFIG_PY" "$CHECKPOINT_RESULTS_DIR/config.yaml" "$MERGE_SOURCE_CONFIG_YAML" "$SAVE_CONFIG_YAML"
elif [ "$(resolve_abs_path "$CONFIG_YAML")" != "$(resolve_abs_path "$SAVE_CONFIG_YAML")" ]; then
  cp "$CONFIG_YAML" "$SAVE_CONFIG_YAML"
fi
TENSORBOARD_CONFIG_YAML="$tensorboard_dir/config.yaml"
if [ "$(resolve_abs_path "$SAVE_CONFIG_YAML")" != "$(resolve_abs_path "$TENSORBOARD_CONFIG_YAML")" ]; then
  cp "$SAVE_CONFIG_YAML" "$TENSORBOARD_CONFIG_YAML"
fi

if [ -n "$resume_for_training" ]; then
    echo "Running in RESUME TRAIN mode from: $resume_for_training"
    TRAINER_VAL_ONLY=""
    IS_VAL_ONLY=0
    TRAINER_VAL_BEFORE_TRAIN="trainer.val_before_train=False"
    TRAINER_TEST_FREQ="trainer.test_freq=0"
    TRAINER_RESUME_MODE="trainer.resume_mode=resume_path"
    TRAINER_RESUME_PATH="trainer.resume_from_path=$resume_for_training"
    TRAINER_TOTAL_EPOCHS="trainer.total_epochs=$trainer_total_epochs"
    TRAINER_SAVE_FREQ="trainer.save_freq=5"
    TRAINER_EXPERIMENT_NAME="trainer.experiment_name=$exp_name"
    DEFAULT_DISABLE_CUDA_GRAPH=1
elif [ "$enable_pretrained_eval_only" = "1" ]; then
    echo "Running PRETRAINED EVAL-ONLY mode with model: $pretrained_model_path"
    TRAINER_VAL_ONLY="trainer.val_only=True"
    IS_VAL_ONLY=1
    TRAINER_VAL_BEFORE_TRAIN="trainer.val_before_train=True"
    TRAINER_TEST_FREQ="trainer.test_freq=1"
    TRAINER_RESUME_MODE=""  
    TRAINER_RESUME_PATH=""
    TRAINER_TOTAL_EPOCHS="trainer.total_epochs=1"
    TRAINER_SAVE_FREQ="trainer.save_freq=0"
    TRAINER_EXPERIMENT_NAME="trainer.experiment_name=${exp_name}_pretrained_eval"
    DEFAULT_DISABLE_CUDA_GRAPH=0
elif [ -n "$checkpoint_path" ]; then
    echo "Running in TEST mode with checkpoint: $checkpoint_path"
    TRAINER_VAL_ONLY="trainer.val_only=True"
    IS_VAL_ONLY=1
    TRAINER_VAL_BEFORE_TRAIN="trainer.val_before_train=True"
    TRAINER_TEST_FREQ="trainer.test_freq=1"
    TRAINER_RESUME_MODE="trainer.resume_mode=resume_path"
    TRAINER_RESUME_PATH="trainer.resume_from_path=$checkpoint_path"
    TRAINER_TOTAL_EPOCHS="trainer.total_epochs=1"
    TRAINER_SAVE_FREQ="trainer.save_freq=0"
    TRAINER_EXPERIMENT_NAME="trainer.experiment_name=${exp_name}_test"
    DEFAULT_DISABLE_CUDA_GRAPH=0
else
    echo "Running in TRAIN mode (from scratch)"
    TRAINER_VAL_ONLY=""
    IS_VAL_ONLY=0
    TRAINER_VAL_BEFORE_TRAIN="trainer.val_before_train=False"
    TRAINER_TEST_FREQ="trainer.test_freq=0"
    TRAINER_RESUME_MODE=""
    TRAINER_RESUME_PATH=""
    TRAINER_TOTAL_EPOCHS="trainer.total_epochs=$trainer_total_epochs"
    TRAINER_SAVE_FREQ="trainer.save_freq=5"
    TRAINER_EXPERIMENT_NAME="trainer.experiment_name=$exp_name"
    DEFAULT_DISABLE_CUDA_GRAPH=1
fi

DISABLE_CUDA_GRAPH="$disable_cuda_graph"
if [ "$DISABLE_CUDA_GRAPH" = "1" ]; then
    ROLLOUT_DISABLE_CUDA_GRAPH=True
else
    ROLLOUT_DISABLE_CUDA_GRAPH=False
fi
CUDA_GRAPH_MAX_BS="$cuda_graph_max_bs"

ENABLE_MEMORY_SAVER="$enable_memory_saver"
if [ "$ENABLE_MEMORY_SAVER" = "1" ]; then
    ROLLOUT_ENABLE_MEMORY_SAVER=True
else
    ROLLOUT_ENABLE_MEMORY_SAVER=False
fi

SGLANG_ATTN_BACKEND="$sglang_attn_backend"
ACTION_OUTPUT_REGEX="$action_output_regex"
ENABLE_REGEX_CONSTRAINT="$enable_regex_constraint"
ROLLOUT_REGEX_OVERRIDE="+actor_rollout_ref.rollout.regex='$ACTION_OUTPUT_REGEX'"
[ "$ENABLE_REGEX_CONSTRAINT" = "1" ] || ROLLOUT_REGEX_OVERRIDE=""
ROLLOUT_STOP_OVERRIDE=""

QWEN_THINKING_OVERRIDE=""
if [ "$disable_qwen_thinking" = "1" ]; then
    QWEN_THINKING_OVERRIDE="+data.apply_chat_template_kwargs.enable_thinking=False"
fi

ENV_MAX_STEPS="$env_max_steps"
ENV_HISTORY_LENGTH="$ENV_MAX_STEPS"
MAX_PROMPT_LEN_VALUE="$max_prompt_len_value"
VAL_ROLLOUT_N="$val_rollout_n"
VAL_TEMPERATURE="$val_temperature"
VAL_TOP_P="$val_top_p"
VAL_TOP_K="$val_top_k"
VAL_DO_SAMPLE="$val_do_sample"
[ "$VAL_DO_SAMPLE" = "1" ] && VAL_DO_SAMPLE=True || VAL_DO_SAMPLE=False
COMPUTE_MEAN_STD_CROSS_STEPS="${compute_mean_std_cross_steps:?}"
REGEX_PLAN_LEN="$regex_plan_len"
REGEX_CODE_LEN="$regex_code_Len"
REGEX_ANSWER_LEN="$regex_answer_len"
export BASH_CODING_REGEX_PLAN_LEN="$REGEX_PLAN_LEN"
export BASH_CODING_REGEX_CODE_LEN="$REGEX_CODE_LEN"
export BASH_CODING_REGEX_ANSWER_LEN="$REGEX_ANSWER_LEN"
REGEX_JSON_OVERHEAD=128
CODE_RESPONSE_LEN=$((REGEX_PLAN_LEN + REGEX_CODE_LEN + REGEX_JSON_OVERHEAD))
ANSWER_RESPONSE_LEN=$((REGEX_PLAN_LEN + REGEX_ANSWER_LEN + REGEX_JSON_OVERHEAD))
if [ "$CODE_RESPONSE_LEN" -ge "$ANSWER_RESPONSE_LEN" ]; then
    MAX_RESP_LEN_VALUE="$CODE_RESPONSE_LEN"
else
    MAX_RESP_LEN_VALUE="$ANSWER_RESPONSE_LEN"
fi

NO_PROGRESS_ON_ANSWER="$no_progress_on_answer"
[ "$NO_PROGRESS_ON_ANSWER" = "1" ] && NO_PROGRESS_ON_ANSWER=True || NO_PROGRESS_ON_ANSWER=False
ANSWER_REWARD="$answer_reward"
EXEC_ERROR_PENALTY="$exec_error_penalty"
USE_MODEL_EVIDENCE_GAIN="$use_model_evidence_gain"
PROGRESS_GAIN_COEF="$progress_gain_coef"
RSTAR_ENABLE="$rstar_enable"
RSTAR_REJECT_EQUAL_REWARD="$rstar_reject_equal_reward"
RSTAR_ROC_ERROR_RATIO="$rstar_roc_error_ratio"
RSTAR_ROC_ANSWER_FORMAT="$rstar_roc_answer_format"
RSTAR_MIN_ZERO_REWARD_TRACE_NUM="$rstar_min_zero_reward_trace_num"
RSTAR_MIN_NON_ZERO_REWARD_TRACE_NUM="$rstar_min_non_zero_reward_trace_num"
RSTAR_DOWNSAMPLE_TO_N="$rstar_downsample_to_n"
RETROAGENT_ENABLE="$retroagent_enable"
RETROAGENT_TOP_K="$retroagent_top_k"
RETROAGENT_STORE_THRESHOLD="$retroagent_store_threshold"
RETROAGENT_NUMERICAL_REWARD_COEF="$retroagent_numerical_reward_coef"
RETROAGENT_LANGUAGE_REWARD_COEF="$retroagent_language_reward_coef"
RETROAGENT_SIMILARITY_THRESHOLD="$retroagent_similarity_threshold"
RETROAGENT_MAX_MEMORY_PER_TASK="$retroagent_max_memory_per_task"
RETROAGENT_REFLECTION_MAX_TOKENS="$retroagent_reflection_max_tokens"
RETROAGENT_MEMORY_PATH="$(resolve_abs_path "$retroagent_memory_path")"
if [ "$RETROAGENT_ENABLE" = "1" ]; then
    RETROAGENT_MEMORY_PATH="$(dirname "$RETROAGENT_MEMORY_PATH")/memory_store_${run_id}.json"
fi
RETROAGENT_RETRIEVAL_TYPE="$retroagent_retrieval_type"
RETROAGENT_RETRIEVE_MODE="$retroagent_retrieve_mode"
RETROAGENT_ALPHA="$retroagent_alpha"
RETROAGENT_BETA="$retroagent_beta"
RETROAGENT_TEMPERATURE="$retroagent_temperature"
RETROAGENT_UCB_SCALE="$retroagent_ucb_scale"
RETROAGENT_EMBEDDING_MODEL_PATH="$retroagent_embedding_model_path"
RETROAGENT_REFLECTION_MAX_PROMPT_TOKENS="$retroagent_reflection_max_prompt_tokens"
RETROAGENT_REFLECTION_HARD_TRUNCATE_TOKENS="$retroagent_reflection_hard_truncate_tokens"
RETROAGENT_REFLECTION_KEEP_LAST_K_STEPS="$retroagent_reflection_keep_last_k_steps"
RETROAGENT_REFLECTION_MAX_OBS_CHARS_PER_STEP="$retroagent_reflection_max_obs_chars_per_step"
RETROAGENT_REFLECTION_MAX_FEEDBACK_CHARS_PER_STEP="$retroagent_reflection_max_feedback_chars_per_step"
RETROAGENT_REFLECTION_MAX_CHANGED_FILES="$retroagent_reflection_max_changed_files"
RETROAGENT_GROUP_RELATIVE_INTRINSIC_REWARDS="$retroagent_group_relative_intrinsic_rewards"
RETROAGENT_POTENTIAL_BASED_ON_BINARY_SUCCESS="$retroagent_potential_based_on_binary_success"
RETROAGENT_FULL_GROUP_MEMORY="$retroagent_full_group_memory"
RETROAGENT_SINGLE_REFLECTION_PER_GROUP="$retroagent_single_reflection_per_group"
RETROAGENT_GROUP_OUTPERFORMANCE="$retroagent_group_outperformance"
export BASH_CODING_RETROAGENT_ENABLE="$RETROAGENT_ENABLE"
export BASH_CODING_RETROAGENT_TOP_K="$RETROAGENT_TOP_K"
export BASH_CODING_RETROAGENT_STORE_THRESHOLD="$RETROAGENT_STORE_THRESHOLD"
export BASH_CODING_RETROAGENT_NUMERICAL_REWARD_COEF="$RETROAGENT_NUMERICAL_REWARD_COEF"
export BASH_CODING_RETROAGENT_LANGUAGE_REWARD_COEF="$RETROAGENT_LANGUAGE_REWARD_COEF"
export BASH_CODING_RETROAGENT_SIMILARITY_THRESHOLD="$RETROAGENT_SIMILARITY_THRESHOLD"
export BASH_CODING_RETROAGENT_MAX_MEMORY_PER_TASK="$RETROAGENT_MAX_MEMORY_PER_TASK"
export BASH_CODING_RETROAGENT_REFLECTION_MAX_TOKENS="$RETROAGENT_REFLECTION_MAX_TOKENS"
export BASH_CODING_RETROAGENT_MEMORY_PATH="$RETROAGENT_MEMORY_PATH"
ENABLE_FAST_INFER_ONLY="$enable_fast_infer_only"
USE_KL_IN_REWARD="$use_kl_in_reward"
KL_PENALTY="$kl_penalty"
KL_OSCILLATION_STOP_ENABLE="${kl_oscillation_stop_enable:?}"
KL_OSCILLATION_STOP_BASELINE_STEPS="${kl_oscillation_stop_baseline_steps:?}"
KL_OSCILLATION_STOP_WINDOW_SIZE="${kl_oscillation_stop_window_size:?}"
KL_OSCILLATION_STOP_RATIO="${kl_oscillation_stop_ratio:?}"
KL_OSCILLATION_STOP_PATIENCE="${kl_oscillation_stop_patience:?}"
KL_OSCILLATION_STOP_METRIC_KEY="${kl_oscillation_stop_metric_key:?}"
KL_OSCILLATION_STOP_MIN_STEPS="${kl_oscillation_stop_min_steps:?}"
KL_OSCILLATION_STOP_RANGE_ONLY="${kl_oscillation_stop_range_only:?}"
KL_CTRL_TYPE="$kl_ctrl_type"
KL_CTRL_KL_COEF="$kl_ctrl_kl_coef"
KL_CTRL_HORIZON="$kl_ctrl_horizon"
KL_CTRL_TARGET_KL="$kl_ctrl_target_kl"

ALGORITHM_USE_KL_IN_REWARD=False
if [ "$USE_KL_IN_REWARD" = "1" ]; then
    ALGORITHM_USE_KL_IN_REWARD=True
fi

if [ "$ALGORITHM_USE_KL_IN_REWARD" = "True" ]; then
    ACTOR_USE_KL_LOSS=False
elif python3 - <<PY
import sys
sys.exit(0 if float("$actor_kl_loss_coef") == 0.0 else 1)
PY
then
    ACTOR_USE_KL_LOSS=False
elif [ "$ENABLE_FAST_INFER_ONLY" = "1" ] && [ "$IS_VAL_ONLY" = "1" ]; then
    echo "Fast infer-only mode enabled: skip ref-policy initialization in val-only runs"
    ACTOR_USE_KL_LOSS=False
else
    ACTOR_USE_KL_LOSS=True
fi

MID_VAL_ARGS=("+trainer.enable_mid_val=False")
if [ "$ENABLE_MID_VAL" = "1" ] && [ "$IS_VAL_ONLY" = "0" ]; then
    MID_VAL_DIR="$save_dir/mid_train"
    mkdir -p "$MID_VAL_DIR"
    MID_VAL_ARGS=(
        "+trainer.enable_mid_val=True"
        "+trainer.mid_val_freq=5"
        "+trainer.mid_val_rollout_n=1"
        "+data.mid_val_files=$TRAIN_FILES_LITERAL"
        "+trainer.mid_val_data_dir=$MID_VAL_DIR"
    )
fi

EXTRA_ARGS=()
for arg in "$@"; do
    case "$arg" in
        *)
            EXTRA_ARGS+=("$arg")
            ;;
    esac
done

export BASH_CODING_ENABLE="$bash_coding_enable"
ENTRYPOINT="verl.trainer.main_ppo"
COUPLING_ARG=""
if [ "$bash_coding_harness" = "commit_if_better" ]; then
    export BASH_CODING_COMMIT_JUDGE_ENABLE="${BASH_CODING_COMMIT_JUDGE_ENABLE:-1}"
else
    export BASH_CODING_COMMIT_JUDGE_ENABLE="${BASH_CODING_COMMIT_JUDGE_ENABLE:-0}"
fi

python3 -m "$ENTRYPOINT" \
    algorithm.adv_estimator=grpo \
    $TRAINER_ALGO_ENABLE_ARG \
    "data.train_files=$TRAIN_FILES_LITERAL" \
    "data.val_files=$VAL_FILES_LITERAL" \
    data.train_batch_size=$train_batch_size \
    data.val_batch_size=$val_batch_size \
    data.max_prompt_length="$MAX_PROMPT_LEN_VALUE" \
    data.max_response_length="$MAX_RESP_LEN_VALUE" \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    data.return_raw_chat=True \
    $QWEN_THINKING_OVERRIDE \
    actor_rollout_ref.model.path=$pretrained_model_path \
    +actor_rollout_ref.model.override_config.attn_implementation=flash \
    actor_rollout_ref.actor.optim.lr=$actor_lr \
    $ACTOR_ALGO_ENABLE_ARG \
    +actor_rollout_ref.actor.algorithm_name=$algo \
    +actor_rollout_ref.actor.algorithm_turn_weight=$algo_turn_weight \
    +actor_rollout_ref.actor.algorithm_gamma=${algo_gamma:-$gamma} \
    +actor_rollout_ref.actor.algorithm_compute_mean_std_cross_steps=$COMPUTE_MEAN_STD_CROSS_STEPS \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$micro_batch_size \
    actor_rollout_ref.actor.use_kl_loss=$ACTOR_USE_KL_LOSS \
    actor_rollout_ref.actor.kl_loss_coef=$actor_kl_loss_coef \
    actor_rollout_ref.actor.kl_loss_type=$KL_PENALTY \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$micro_batch_size \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=$rollout_gpu_memory_utilization \
    actor_rollout_ref.rollout.val_kwargs.n=$VAL_ROLLOUT_N \
    actor_rollout_ref.rollout.val_kwargs.do_sample=$VAL_DO_SAMPLE \
    actor_rollout_ref.rollout.val_kwargs.temperature=$VAL_TEMPERATURE \
    actor_rollout_ref.rollout.val_kwargs.top_p=$VAL_TOP_P \
    actor_rollout_ref.rollout.val_kwargs.top_k=$VAL_TOP_K \
    +actor_rollout_ref.rollout.mem_fraction_static=$rollout_mem_fraction_static \
    +actor_rollout_ref.rollout.cuda_graph_max_bs=$CUDA_GRAPH_MAX_BS \
    +actor_rollout_ref.rollout.disable_cuda_graph=$ROLLOUT_DISABLE_CUDA_GRAPH \
    +actor_rollout_ref.rollout.enable_memory_saver=$ROLLOUT_ENABLE_MEMORY_SAVER \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=$SGLANG_ATTN_BACKEND \
    $ROLLOUT_STOP_OVERRIDE \
    $ROLLOUT_REGEX_OVERRIDE \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$micro_batch_size \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=False \
    algorithm.use_kl_in_reward=$ALGORITHM_USE_KL_IN_REWARD \
    algorithm.kl_penalty=$KL_PENALTY \
    algorithm.kl_ctrl.type=$KL_CTRL_TYPE \
    algorithm.kl_ctrl.kl_coef=$KL_CTRL_KL_COEF \
    algorithm.kl_ctrl.horizon=$KL_CTRL_HORIZON \
    algorithm.kl_ctrl.target_kl=$KL_CTRL_TARGET_KL \
    algorithm.gamma=$gamma \
    env.env_name=bash_coding/BashCodingMultiProcessEnv \
    env.seed=$env_seed \
    env.max_steps=$ENV_MAX_STEPS \
    env.history_length=$ENV_HISTORY_LENGTH \
    +env.answer_reward=$ANSWER_REWARD \
    +env.exec_error_penalty=$EXEC_ERROR_PENALTY \
    +env.use_model_evidence_gain=$USE_MODEL_EVIDENCE_GAIN \
    +env.progress_gain_coef=$PROGRESS_GAIN_COEF \
    ++reward.rstar_enable=$RSTAR_ENABLE \
    ++reward.rstar_reject_equal_reward=$RSTAR_REJECT_EQUAL_REWARD \
    ++reward.rstar_roc_error_ratio=$RSTAR_ROC_ERROR_RATIO \
    ++reward.rstar_roc_answer_format=$RSTAR_ROC_ANSWER_FORMAT \
    ++reward.rstar_min_zero_reward_trace_num=$RSTAR_MIN_ZERO_REWARD_TRACE_NUM \
    ++reward.rstar_min_non_zero_reward_trace_num=$RSTAR_MIN_NON_ZERO_REWARD_TRACE_NUM \
    ++reward.rstar_downsample_to_n=$RSTAR_DOWNSAMPLE_TO_N \
    ++reward.retroagent_enable=$RETROAGENT_ENABLE \
    ++reward.retroagent_top_k=$RETROAGENT_TOP_K \
    ++reward.retroagent_store_threshold=$RETROAGENT_STORE_THRESHOLD \
    ++reward.retroagent_numerical_reward_coef=$RETROAGENT_NUMERICAL_REWARD_COEF \
    ++reward.retroagent_language_reward_coef=$RETROAGENT_LANGUAGE_REWARD_COEF \
    ++reward.retroagent_similarity_threshold=$RETROAGENT_SIMILARITY_THRESHOLD \
    ++reward.retroagent_max_memory_per_task=$RETROAGENT_MAX_MEMORY_PER_TASK \
    ++reward.retroagent_reflection_max_tokens=$RETROAGENT_REFLECTION_MAX_TOKENS \
    "++reward.retroagent_memory_path=$RETROAGENT_MEMORY_PATH" \
    "++reward.retroagent_retrieval_type=$RETROAGENT_RETRIEVAL_TYPE" \
    "++reward.retroagent_retrieve_mode=$RETROAGENT_RETRIEVE_MODE" \
    ++reward.retroagent_alpha=$RETROAGENT_ALPHA \
    ++reward.retroagent_beta=$RETROAGENT_BETA \
    ++reward.retroagent_temperature=$RETROAGENT_TEMPERATURE \
    ++reward.retroagent_ucb_scale=$RETROAGENT_UCB_SCALE \
    "++reward.retroagent_embedding_model_path=$RETROAGENT_EMBEDDING_MODEL_PATH" \
    ++reward.retroagent_reflection_max_prompt_tokens=$RETROAGENT_REFLECTION_MAX_PROMPT_TOKENS \
    ++reward.retroagent_reflection_hard_truncate_tokens=$RETROAGENT_REFLECTION_HARD_TRUNCATE_TOKENS \
    ++reward.retroagent_reflection_keep_last_k_steps=$RETROAGENT_REFLECTION_KEEP_LAST_K_STEPS \
    ++reward.retroagent_reflection_max_obs_chars_per_step=$RETROAGENT_REFLECTION_MAX_OBS_CHARS_PER_STEP \
    ++reward.retroagent_reflection_max_feedback_chars_per_step=$RETROAGENT_REFLECTION_MAX_FEEDBACK_CHARS_PER_STEP \
    ++reward.retroagent_reflection_max_changed_files=$RETROAGENT_REFLECTION_MAX_CHANGED_FILES \
    ++reward.retroagent_group_relative_intrinsic_rewards=$RETROAGENT_GROUP_RELATIVE_INTRINSIC_REWARDS \
    ++reward.retroagent_potential_based_on_binary_success=$RETROAGENT_POTENTIAL_BASED_ON_BINARY_SUCCESS \
    ++reward.retroagent_full_group_memory=$RETROAGENT_FULL_GROUP_MEMORY \
    ++reward.retroagent_single_reflection_per_group=$RETROAGENT_SINGLE_REFLECTION_PER_GROUP \
    ++reward.retroagent_group_outperformance=$RETROAGENT_GROUP_OUTPERFORMANCE \
    +env.timeout=$timeout \
    +env.execute_commands=$execute_commands \
    +env.exec_backend=$exec_backend \
    +env.no_progress_on_answer=$NO_PROGRESS_ON_ANSWER \
    +env.bash_coding_harness=$bash_coding_harness \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=0.1 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name=$exp_name \
    $TRAINER_EXPERIMENT_NAME \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.nnodes=1 \
    $TRAINER_SAVE_FREQ \
    $TRAINER_TEST_FREQ \
    $TRAINER_TOTAL_EPOCHS \
    $TRAINER_VAL_BEFORE_TRAIN \
    trainer.validation_data_dir=$save_dir \
    "${MID_VAL_ARGS[@]}" \
    trainer.default_local_dir=$checkpoint_dir \
    trainer.max_actor_ckpt_to_keep=$save_count \
    trainer.max_critic_ckpt_to_keep=$save_count \
    +trainer.kl_oscillation_stop.enable=$KL_OSCILLATION_STOP_ENABLE \
    +trainer.kl_oscillation_stop.metric_key=$KL_OSCILLATION_STOP_METRIC_KEY \
    +trainer.kl_oscillation_stop.baseline_steps=$KL_OSCILLATION_STOP_BASELINE_STEPS \
    +trainer.kl_oscillation_stop.window_size=$KL_OSCILLATION_STOP_WINDOW_SIZE \
    +trainer.kl_oscillation_stop.ratio=$KL_OSCILLATION_STOP_RATIO \
    +trainer.kl_oscillation_stop.patience=$KL_OSCILLATION_STOP_PATIENCE \
    +trainer.kl_oscillation_stop.min_steps=$KL_OSCILLATION_STOP_MIN_STEPS \
    +trainer.kl_oscillation_stop.range_only=$KL_OSCILLATION_STOP_RANGE_ONLY \
    +ray_init.include_dashboard=false \
    $TRAINER_RESUME_MODE \
    $TRAINER_RESUME_PATH \
    $TRAINER_VAL_ONLY "${EXTRA_ARGS[@]}"
