
import math
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple
from main_entry.cli_agent_bash_coding.env_package.bash_coding.projection import (
    ANSWER_PREFIX,
    FORMAT_VIOLATION_PREFIX,
)

@dataclass
class _LatsArm:
    raw_response: str
    completion_tokens: int
    candidate: Dict[str, Any]
    visits: int = 0
    value_sum: float = 0.0

def _uct_score(arm: _LatsArm, parent_visits: int, exploration_weight: float) -> float:
    if arm.visits == 0:
        return float("inf")
    exploit = arm.value_sum / arm.visits
    explore = exploration_weight * math.sqrt(math.log(parent_visits + 1) / arm.visits)
    return exploit + explore

def _rollout_value(candidate: Dict[str, Any]) -> float:
    return float(candidate["reward"])

def _run_uct_on_arms(
    arms: List[_LatsArm],
    iterations: int,
    exploration_weight: float,
) -> _LatsArm:
    root_visits = 0
    for _ in range(iterations):
        picked = max(arms, key=lambda a: _uct_score(a, root_visits, exploration_weight))
        r = _rollout_value(picked.candidate)
        picked.visits += 1
        picked.value_sum += r
        root_visits += 1
    return max(arms, key=lambda a: (a.visits, a.value_sum))

def run_lats_batched_turn(
    episodes: List[Dict[str, Any]],
    args: Any,
    api_key: str,
    enable_commit: bool,
    action_regex: str,
    request_model_response: Callable[..., Tuple[str, int]],
    fork_worker_replay_and_step: Callable[
        ..., Tuple[Any, str, float, bool, Dict[str, Any], str]
    ],
    episode_max_tokens_fn: Callable[[Dict[str, Any], Any], int],
    per_branch_max_tokens_fn: Callable[[Dict[str, Any], Any, int], int],
) -> None:
    n_gen = int(args.lats_n_generate_sample)
    iterations = int(args.lats_iterations)
    exploration_weight = float(args.lats_exploration_weight)
    max_depth = int(args.lats_max_depth)
    if max_depth != 1:
        raise ValueError(
            "lats_infer: only lats_max_depth=1 is implemented for bash coding baseline."
        )
    for episode in episodes:
        cap_turn = episode_max_tokens_fn(episode, args)
        prompt = episode["prompt_state"].build_text_obs(
            str(episode["observation"]),
            init=not episode["history"],
        )
        branch_k = n_gen
        cap = per_branch_max_tokens_fn(episode, args, branch_k)
        env_kwargs = episode["env_kwargs"]
        sample_env_kwargs = episode["sample_env_kwargs"]
        seed_off = int(episode["seed_offset"])
        hist = episode["history"]
        if cap <= 0:
            raw_responses = [""] * branch_k
            completion_tokens = [0] * branch_k
        else:
            max_workers = min(branch_k, int(args.lats_expand_max_workers))

            def _one_sample() -> Tuple[str, int]:
                return request_model_response(
                    api_base_url=args.api_base_url,
                    api_key=api_key,
                    api_model=args.api_model,
                    api_message_style=args.api_message_style,
                    prompt=prompt,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=cap,
                    llm_timeout=args.llm_timeout,
                    disable_qwen_thinking=bool(args.disable_qwen_thinking),
                    do_sample=bool(args.do_sample),
                    enable_regex_constraint=bool(args.enable_regex_constraint),
                    backend_name=args.backend_name,
                    action_regex=action_regex,
                )

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(_one_sample) for _ in range(branch_k)]
                results = [f.result() for f in futures]
            raw_responses = [r[0] for r in results]
            completion_tokens = [r[1] for r in results]
        arms: List[_LatsArm] = []
        candidates: List[Dict[str, Any]] = []
        for raw in raw_responses:
            w, obs, rew, done, last_info, env_action = fork_worker_replay_and_step(
                args,
                env_kwargs,
                sample_env_kwargs,
                seed_off,
                hist,
                raw,
                enable_commit,
            )
            cand = {
                "worker": w,
                "raw_response": raw,
                "observation": obs,
                "reward": rew,
                "done": done,
                "last_info": last_info,
                "env_action": env_action,
            }
            candidates.append(cand)
            arms.append(_LatsArm(raw_response=raw, completion_tokens=0, candidate=cand))
        for arm, n_tok in zip(arms, completion_tokens, strict=True):
            arm.completion_tokens = int(n_tok)
        if not arms:
            continue
        uct_final_arm = _run_uct_on_arms(arms, iterations, exploration_weight)
        chosen = uct_final_arm.candidate
        prev_observation = str(episode["observation"])
        raw_response = chosen["raw_response"]
        n_tok = int(sum(a.completion_tokens for a in arms))
        env_action = chosen["env_action"]
        observation = str(chosen["observation"])
        reward = float(chosen["reward"])
        done = bool(chosen["done"])
        last_info = chosen["last_info"]
        old_w = episode["worker"]
        if old_w.work_dir and os.path.isdir(old_w.work_dir):
            old_w.close()
        episode["worker"] = chosen["worker"]
        for c in candidates:
            if c["worker"] is not chosen["worker"]:
                ow = c["worker"]
                if ow.work_dir and os.path.isdir(ow.work_dir):
                    ow.close()
        episode["prompt_state"].harness.apply_history_commit_decisions(
            episode["prompt_state"].memory,
            [last_info],
            next_text_obs=[observation],
        )
        episode["prompt_state"].store_transition(prev_observation, raw_response)
        episode["lats_gen_tokens_used"] = int(episode["lats_gen_tokens_used"]) + n_tok
        prefix_len = int(args.lats_history_raw_prefix_chars)
        episode["lats_history_actions"].append(
            {
                "raw_response_prefix": raw_response[:prefix_len],
                "completion_tokens": n_tok,
            }
        )
        if env_action.startswith(ANSWER_PREFIX):
            episode["final_answer"] = env_action[len(ANSWER_PREFIX) :].strip()
        elif env_action.startswith(FORMAT_VIOLATION_PREFIX):
            episode["final_answer"] = None
        episode["observation"] = observation
        episode["reward"] = reward
        episode["done"] = done
        if int(cap_turn) <= 0 and int(args.lats_stop_on_exhausted_budget):
            episode["done"] = True
        episode["last_info"] = last_info
        episode["lats_history_tool_calls"].append(
            {
                "name": "bash_coding",
                "model_output": raw_response,
                "observation": observation,
            }
        )
        arm_rewards = [float(c["reward"]) for c in candidates]
        arm_visits = [a.visits for a in arms]
        step_payload: Dict[str, Any] = {
            "prompt": prompt,
            "raw_response": raw_response,
            "action": raw_response,
            "observation": prev_observation,
            "lats_infer": 1,
            "lats_n_generate_sample": branch_k,
            "lats_iterations": iterations,
            "lats_exploration_weight": exploration_weight,
            "lats_arm_rewards": arm_rewards,
            "lats_arm_visits_final": arm_visits,
            "lats_chosen_arm_visits": uct_final_arm.visits,
            "lats_completion_tokens_expand_sum": n_tok,
            "lats_completion_tokens": n_tok,
            "lats_cumulative_gen_tokens": int(episode["lats_gen_tokens_used"]),
            "lats_max_new_tokens_cap": int(cap_turn),
            "lats_tool_history_steps_after": len(episode["lats_history_tool_calls"]),
        }
        episode["history"].append(step_payload)
