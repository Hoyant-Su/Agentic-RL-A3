#!/bin/bash

set -eu

export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
unset RAY_ADDRESS
python3 -m ray stop --force || true

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

checkpoint_stamp="group"

CONFIG_YAML="${BASH_CODING_CONFIG_YAML:-$SCRIPT_DIR/$checkpoint_stamp/config_intent_scopes_12345_neg1.yaml}"

eval "$(python3 "$PROJECT_DIR/main_entry/cli_agent_bash_coding/train/load_config_yaml.py" "$CONFIG_YAML")"

exec env BASH_CODING_CONFIG_YAML="$CONFIG_YAML" \
    bash "$PROJECT_DIR/main_entry/cli_agent_bash_coding/train/A3/launch_bash_coding.sh" \
    +actor_rollout_ref.actor.algorithm_epsilon=$algo_epsilon \
    +actor_rollout_ref.actor.algorithm_adv_bound=$algo_adv_bound \
    +actor_rollout_ref.actor.algorithm_trie_min_count=$algo_trie_min_count \
    +actor_rollout_ref.actor.algorithm_trie_edge_max_tokens=$algo_trie_edge_max_tokens \
    +actor_rollout_ref.actor.algorithm_w_trie=$algo_w_trie \
    +actor_rollout_ref.actor.algorithm_support_prior=$algo_support_prior \
    +actor_rollout_ref.actor.algorithm_decision_gamma=$algo_decision_gamma \
    +actor_rollout_ref.actor.algorithm_w_action_cluster=$algo_w_action_cluster \
    +actor_rollout_ref.actor.algorithm_intent_scopes="$algo_intent_scopes" \
    +actor_rollout_ref.actor.algorithm_w_intent_scopes="$algo_w_intent_scopes" \
    +actor_rollout_ref.actor.algorithm_A3_algo_tree_intent_match_tau=$algo_A3_algo_tree_intent_match_tau \
    +actor_rollout_ref.actor.algorithm_A3_algo_tree_bucket_time_decay=$algo_A3_algo_tree_bucket_time_decay \
    "$@"
