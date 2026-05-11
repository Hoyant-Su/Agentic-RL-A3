import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Set, Tuple
import uuid

from omegaconf import OmegaConf
import pandas as pd
import requests
from tqdm import tqdm

from main_entry.cli_agent_bash_coding.action_schema import get_bash_coding_action_regex
from main_entry.cli_agent_bash_coding.action_schema import uses_commit_action_schema
from main_entry.cli_agent_bash_coding.baseline_methods import BaselineMethodManager
from main_entry.cli_agent_bash_coding.baseline_methods.lats_infer import run_lats_batched_turn
from main_entry.cli_agent_bash_coding.env_package.bash_coding.envs import BashCodingWorker
from main_entry.cli_agent_bash_coding.env_package.bash_coding.projection import (
    ANSWER_PREFIX,
    FORMAT_VIOLATION_PREFIX,
    bash_coding_projection,
)
from main_entry.cli_agent_bash_coding.harness import build_bash_coding_harness
from main_entry.cli_agent_bash_coding.memory import SimpleMemory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--config_yaml", default="")
    parser.add_argument("--api_base_url", required=True)
    parser.add_argument("--api_model", required=True)
    parser.add_argument("--api_key_env", required=True)
    parser.add_argument("--api_message_style", choices=["plain", "content_blocks"], default="plain")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--rollout_n", type=int, default=1)
    parser.add_argument("--do_sample", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=8)
    parser.add_argument("--history_length", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=10)
    parser.add_argument("--llm_timeout", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--backend_name", type=str, default="")
    parser.add_argument("--exec_backend", type=str, default="sandbox")
    parser.add_argument("--execute_commands", type=int, default=1)
    parser.add_argument("--no_progress_on_answer", type=int, default=1)
    parser.add_argument("--enable_regex_constraint", type=int, default=0)
    parser.add_argument("--disable_qwen_thinking", type=int, default=0)
    parser.add_argument("--flush_every", type=int, default=100)
    parser.add_argument("--rstar_infer_enable", type=int, default=0)
    parser.add_argument("--rstar_response_budget_tokens", type=int, default=0)
    parser.add_argument("--rstar_inject_history_json_block", type=int, default=0)
    parser.add_argument("--rstar_history_action_max_chars", type=int, default=800)
    parser.add_argument("--rstar_history_obs_max_chars", type=int, default=1500)
    parser.add_argument("--rstar_stop_on_exhausted_budget", type=int, default=1)
    parser.add_argument("--rstar_infer_tree_search", type=int, default=0)
    parser.add_argument("--rstar_infer_branch_k", type=int, default=4)
    parser.add_argument("--rstar_infer_tree_budget_split", type=int, default=1)
    parser.add_argument("--lats_infer_enable", type=int, default=0)
    parser.add_argument("--lats_iterations", type=int, default=None)
    parser.add_argument("--lats_n_generate_sample", type=int, default=None)
    parser.add_argument("--lats_exploration_weight", type=float, default=None)
    parser.add_argument("--lats_max_depth", type=int, default=None)
    parser.add_argument("--lats_response_budget_tokens", type=int, default=None)
    parser.add_argument("--lats_stop_on_exhausted_budget", type=int, default=None)
    parser.add_argument("--lats_tree_budget_split", type=int, default=None)
    parser.add_argument("--lats_expand_max_workers", type=int, default=None)
    parser.add_argument("--lats_history_raw_prefix_chars", type=int, default=None)
    return parser.parse_args()


LATS_INFER_REQUIRED_FIELDS = (
    "lats_iterations",
    "lats_n_generate_sample",
    "lats_exploration_weight",
    "lats_max_depth",
    "lats_response_budget_tokens",
    "lats_stop_on_exhausted_budget",
    "lats_tree_budget_split",
    "lats_expand_max_workers",
    "lats_history_raw_prefix_chars",
)


def _ensure_lats_infer_args(args: argparse.Namespace) -> None:
    if not int(args.lats_infer_enable):
        return
    missing = [name for name in LATS_INFER_REQUIRED_FIELDS if getattr(args, name) is None]
    if missing:
        raise ValueError(
            "lats_infer.enable is set but the following keys are missing under lats_infer in config_yaml (or CLI): "
            + ", ".join(missing)
        )
    if int(args.lats_n_generate_sample) < 1:
        raise ValueError("lats_infer.n_generate_sample must be >= 1")
    if int(args.lats_iterations) < 1:
        raise ValueError("lats_infer.iterations must be >= 1")
    if int(args.lats_expand_max_workers) < 1:
        raise ValueError("lats_infer.expand_max_workers must be >= 1")
    if int(args.lats_history_raw_prefix_chars) < 0:
        raise ValueError("lats_infer.history_raw_prefix_chars must be >= 0")


def build_messages(prompt: str, api_message_style: str) -> List[Dict[str, Any]]:
    if api_message_style == "content_blocks":
        return [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    return [{"role": "user", "content": prompt}]


def _truncate_visible(s: str, n: int) -> str:
    s = str(s)
    if n <= 0:
        return ""
    if len(s) <= n:
        return s
    return s[: max(0, n - 20)] + "\n...(truncated)"


def build_rstar_history_prefix(tool_calls: List[Dict[str, Any]], args: argparse.Namespace) -> str:
    if not tool_calls:
        return ""
    ac = int(getattr(args, "rstar_history_action_max_chars", 800))
    oc = int(getattr(args, "rstar_history_obs_max_chars", 1500))
    payload = []
    for i, call in enumerate(tool_calls):
        payload.append(
            {
                "idx": i,
                "name": call.get("name", "bash_coding"),
                "model_output_excerpt": _truncate_visible(call.get("model_output", ""), ac),
                "environment_observation_excerpt": _truncate_visible(call.get("observation", ""), oc),
            }
        )
    header = (
        "[rStar2-aligned prior steps: analogous to history_tool_calls for tool execution; "
        "this bash harness still requires submit_code/submit_answer schema below.]\n"
    )
    return header + json.dumps(payload, ensure_ascii=False) + "\n\n"


def load_runtime_config(config_yaml: str) -> Any:
    if not config_yaml:
        return OmegaConf.create({"env": {}, "reward": {}})
    config = OmegaConf.load(config_yaml)
    if OmegaConf.select(config, "env") is None:
        config.env = {}
    if OmegaConf.select(config, "reward") is None:
        config.reward = {}
    return config


def build_runtime_env_kwargs(args: argparse.Namespace, config: Any) -> Dict[str, Any]:
    env_kwargs: Dict[str, Any] = {}
    env_cfg = OmegaConf.to_container(OmegaConf.select(config, "env", default={}), resolve=True)
    reward_cfg = OmegaConf.to_container(OmegaConf.select(config, "reward", default={}), resolve=True)
    env_kwargs.update(env_cfg)
    env_kwargs.update(reward_cfg)
    if "env_max_steps" in env_kwargs and "max_steps" not in env_kwargs:
        env_kwargs["max_steps"] = env_kwargs.pop("env_max_steps")
    if "env_history_length" in env_kwargs and "history_length" not in env_kwargs:
        env_kwargs["history_length"] = env_kwargs.pop("env_history_length")
    env_kwargs["max_steps"] = int(args.max_steps)
    env_kwargs["history_length"] = int(args.history_length)
    env_kwargs["timeout"] = int(args.timeout)
    env_kwargs["execute_commands"] = bool(args.execute_commands)
    env_kwargs["exec_backend"] = args.exec_backend
    env_kwargs["no_progress_on_answer"] = bool(args.no_progress_on_answer)
    if "bash_coding_harness" not in env_kwargs:
        raise KeyError("Missing env.bash_coding_harness in runtime config.")
    return env_kwargs


def apply_yaml_infer_and_rstar_overrides(args: argparse.Namespace, config: Any) -> None:
    root = OmegaConf.to_container(config, resolve=True)
    if isinstance(root, dict):
        if "rstar_infer_enable" in root:
            args.rstar_infer_enable = int(root["rstar_infer_enable"])
        if "rstar_response_budget_tokens" in root:
            args.rstar_response_budget_tokens = int(root["rstar_response_budget_tokens"])
        if "rstar_inject_history_json_block" in root:
            args.rstar_inject_history_json_block = int(root["rstar_inject_history_json_block"])
        if "rstar_history_action_max_chars" in root:
            args.rstar_history_action_max_chars = int(root["rstar_history_action_max_chars"])
        if "rstar_history_obs_max_chars" in root:
            args.rstar_history_obs_max_chars = int(root["rstar_history_obs_max_chars"])
        if "rstar_stop_on_exhausted_budget" in root:
            args.rstar_stop_on_exhausted_budget = int(root["rstar_stop_on_exhausted_budget"])
        if "rstar_infer_tree_search" in root:
            args.rstar_infer_tree_search = int(root["rstar_infer_tree_search"])
        if "rstar_infer_branch_k" in root:
            args.rstar_infer_branch_k = int(root["rstar_infer_branch_k"])
        if "rstar_infer_tree_budget_split" in root:
            args.rstar_infer_tree_budget_split = int(root["rstar_infer_tree_budget_split"])
        if "lats_infer_enable" in root:
            args.lats_infer_enable = int(root["lats_infer_enable"])
        if "lats_iterations" in root:
            args.lats_iterations = int(root["lats_iterations"])
        if "lats_n_generate_sample" in root:
            args.lats_n_generate_sample = int(root["lats_n_generate_sample"])
        if "lats_exploration_weight" in root:
            args.lats_exploration_weight = float(root["lats_exploration_weight"])
        if "lats_max_depth" in root:
            args.lats_max_depth = int(root["lats_max_depth"])
        if "lats_response_budget_tokens" in root:
            args.lats_response_budget_tokens = int(root["lats_response_budget_tokens"])
        if "lats_stop_on_exhausted_budget" in root:
            args.lats_stop_on_exhausted_budget = int(root["lats_stop_on_exhausted_budget"])
        if "lats_tree_budget_split" in root:
            args.lats_tree_budget_split = int(root["lats_tree_budget_split"])
        if "lats_expand_max_workers" in root:
            args.lats_expand_max_workers = int(root["lats_expand_max_workers"])
        if "lats_history_raw_prefix_chars" in root:
            args.lats_history_raw_prefix_chars = int(root["lats_history_raw_prefix_chars"])
    infer = OmegaConf.select(config, "infer", default=None)
    if infer is not None:
        ic = OmegaConf.to_container(infer, resolve=True)
        if isinstance(ic, dict):
            if "batch_size" in ic:
                args.batch_size = int(ic["batch_size"])
            if "rollout_n" in ic:
                args.rollout_n = int(ic["rollout_n"])
            if "do_sample" in ic:
                args.do_sample = int(ic["do_sample"])
            if "max_steps" in ic:
                args.max_steps = int(ic["max_steps"])
            if "history_length" in ic:
                args.history_length = int(ic["history_length"])
            if "llm_timeout" in ic:
                args.llm_timeout = int(ic["llm_timeout"])
            if "disable_qwen_thinking" in ic:
                args.disable_qwen_thinking = int(ic["disable_qwen_thinking"])
            if "temperature" in ic:
                args.temperature = float(ic["temperature"])
            if "top_p" in ic:
                args.top_p = float(ic["top_p"])
            if "seed" in ic:
                args.seed = int(ic["seed"])
            if "enable_regex_constraint" in ic:
                args.enable_regex_constraint = int(ic["enable_regex_constraint"])
            if "max_tokens" in ic:
                args.max_tokens = int(ic["max_tokens"])
    ri = OmegaConf.select(config, "rstar_infer", default=None)
    if ri is not None:
        ric = OmegaConf.to_container(ri, resolve=True)
        if isinstance(ric, dict):
            if "enable" in ric:
                args.rstar_infer_enable = int(ric["enable"])
            if "rstar_infer_enable" in ric:
                args.rstar_infer_enable = int(ric["rstar_infer_enable"])
            if "response_budget_tokens" in ric:
                args.rstar_response_budget_tokens = int(ric["response_budget_tokens"])
            if "inject_history_json_block" in ric:
                args.rstar_inject_history_json_block = int(ric["inject_history_json_block"])
            if "history_action_max_chars" in ric:
                args.rstar_history_action_max_chars = int(ric["history_action_max_chars"])
            if "history_obs_max_chars" in ric:
                args.rstar_history_obs_max_chars = int(ric["history_obs_max_chars"])
            if "stop_on_exhausted_budget" in ric:
                args.rstar_stop_on_exhausted_budget = int(ric["stop_on_exhausted_budget"])
            if "infer_tree_search" in ric:
                args.rstar_infer_tree_search = int(ric["infer_tree_search"])
            if "branch_k" in ric:
                args.rstar_infer_branch_k = int(ric["branch_k"])
            if "tree_budget_split" in ric:
                args.rstar_infer_tree_budget_split = int(ric["tree_budget_split"])
    li = OmegaConf.select(config, "lats_infer", default=None)
    if li is not None:
        lic = OmegaConf.to_container(li, resolve=True)
        if isinstance(lic, dict):
            if "enable" in lic:
                args.lats_infer_enable = int(lic["enable"])
            if "lats_infer_enable" in lic:
                args.lats_infer_enable = int(lic["lats_infer_enable"])
            if "iterations" in lic:
                args.lats_iterations = int(lic["iterations"])
            if "n_generate_sample" in lic:
                args.lats_n_generate_sample = int(lic["n_generate_sample"])
            if "exploration_weight" in lic:
                args.lats_exploration_weight = float(lic["exploration_weight"])
            if "max_depth" in lic:
                args.lats_max_depth = int(lic["max_depth"])
            if "response_budget_tokens" in lic:
                args.lats_response_budget_tokens = int(lic["response_budget_tokens"])
            if "stop_on_exhausted_budget" in lic:
                args.lats_stop_on_exhausted_budget = int(lic["stop_on_exhausted_budget"])
            if "tree_budget_split" in lic:
                args.lats_tree_budget_split = int(lic["tree_budget_split"])
            if "expand_max_workers" in lic:
                args.lats_expand_max_workers = int(lic["expand_max_workers"])
            if "history_raw_prefix_chars" in lic:
                args.lats_history_raw_prefix_chars = int(lic["history_raw_prefix_chars"])
    settings = OmegaConf.select(config, "settings", default=None)
    if settings is not None:
        sc = OmegaConf.to_container(settings, resolve=True)
        if isinstance(sc, dict) and sc.get("backend"):
            args.backend_name = str(sc["backend"])


def build_prompt_config(config: Any, env_kwargs: Dict[str, Any]) -> Any:
    prompt_config = OmegaConf.create(
        {
            "env": OmegaConf.to_container(OmegaConf.select(config, "env", default={}), resolve=True),
            "reward": OmegaConf.to_container(OmegaConf.select(config, "reward", default={}), resolve=True),
        }
    )
    prompt_config.env.max_steps = int(env_kwargs["max_steps"])
    prompt_config.env.history_length = int(env_kwargs["history_length"])
    prompt_config.env.bash_coding_harness = str(env_kwargs["bash_coding_harness"])
    return prompt_config


class PureInferPromptState:
    def __init__(self, config: Any, task: str) -> None:
        self.config = config
        self.tasks = [task]
        self.memory = SimpleMemory()
        self.memory.reset(batch_size=1)
        self.method_manager = BaselineMethodManager.from_config(config)
        self.harness = build_bash_coding_harness(OmegaConf.to_container(config.env, resolve=True))
        self.method_manager.prepare_batch(self.tasks, is_eval=True, group_n=1)

    def build_text_obs(self, observation: str, *, init: bool) -> str:
        return self.harness.build_text_obs(self, [observation], init=init)[0]

    def store_transition(self, observation: str, action: str) -> None:
        self.memory.store(
            {
                "observation": [observation],
                "observation_for_history": [observation],
                "action": [action],
            }
        )


def request_model_response(
    *,
    api_base_url: str,
    api_key: str,
    api_model: str,
    api_message_style: str,
    prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    llm_timeout: int,
    disable_qwen_thinking: bool,
    do_sample: bool,
    enable_regex_constraint: bool,
    backend_name: str,
    action_regex: str,
) -> Tuple[str, int]:
    if max_tokens <= 0:
        return "", 0
    url = api_base_url.rstrip("/")
    if url.endswith("/chat/completions"):
        pass
    elif url.endswith("/v1"):
        url = url + "/chat/completions"
    else:
        url = url + "/v1/chat/completions"

    payload = {
        "model": api_model,
        "messages": build_messages(prompt, api_message_style),
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "top_p": float(top_p),
    }
    if backend_name != "inf_minimax":
        payload["do_sample"] = bool(do_sample)
    if disable_qwen_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": False, "thinking": False}
        if backend_name in ("local", "inf_minimax", "inf_qwen"):
            payload["reasoning_effort"] = "none"
    regex_branch = bool(enable_regex_constraint) and backend_name == "local"
    if regex_branch:
        payload["regex"] = action_regex
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    response = requests.post(url, json=payload, headers=headers, timeout=llm_timeout)
    response.raise_for_status()
    data = response.json()
    choices = data.get("choices") or []
    if not choices:
        return "", 0
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        text = content.strip()
    elif isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block.get("text", ""))
        text = "\n".join(text_parts).strip()
    else:
        text = str(content).strip()
    usage = data.get("usage") or {}
    comp = usage.get("completion_tokens")
    if comp is None:
        ntok = max(1, len(text) // 4) if text else 0
    else:
        ntok = int(comp)
    return text, ntok


def resolve_api_model(api_base_url: str, api_key: str, api_model: str, llm_timeout: int) -> str:
    if api_model and api_model not in {"auto", "__auto__"}:
        return api_model

    url = api_base_url.rstrip("/")
    if url.endswith("/chat/completions"):
        url = url.rsplit("/", 2)[0]
    elif not url.endswith("/v1"):
        url = url + "/v1"
    url = url + "/models"

    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    response = requests.get(url, headers=headers, timeout=llm_timeout)
    response.raise_for_status()
    data = response.json()
    models = data.get("data") or []
    if not models:
        raise ValueError(f"No models found at {url}")
    model_id = models[0].get("id")
    if not model_id:
        raise ValueError(f"Invalid model metadata from {url}: {models[0]}")
    return str(model_id)


def create_worker(args: argparse.Namespace, seed_offset: int, env_kwargs: Dict[str, Any]) -> BashCodingWorker:
    return BashCodingWorker(
        seed=args.seed + seed_offset,
        env_kwargs=env_kwargs,
    )


def infer_dataset_name_from_parquet(parquet_path: str) -> str:
    return Path(parquet_path).resolve().parent.name


def resolve_dataset_name(row: pd.Series, parquet_path: str) -> str:
    dataset_name = row.get("dataset", None)
    if pd.isna(dataset_name):
        dataset_name = None
    if dataset_name is None or not str(dataset_name).strip():
        dataset_name = infer_dataset_name_from_parquet(parquet_path)
    return str(dataset_name)


def build_traj_uid(dataset_name: str, sample_id_base: str, rollout_idx: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{dataset_name}:{sample_id_base}:{rollout_idx}"))


def init_episode(
    row: pd.Series,
    args: argparse.Namespace,
    env_kwargs: Dict[str, Any],
    prompt_config: Any,
    seed_offset: int,
    rollout_idx: int,
) -> Dict[str, Any]:
    sample_env_kwargs = dict(row["env_kwargs"])
    worker = create_worker(args, seed_offset, env_kwargs)
    observation, info = worker.reset(sample_env_kwargs)
    dataset_name = resolve_dataset_name(row, args.parquet)
    sample_id_base = str(row["id"])
    task = str(info.get("task", row["query"]))
    return {
        "row": row,
        "dataset_name": dataset_name,
        "sample_id_base": sample_id_base,
        "traj_uid": build_traj_uid(dataset_name, sample_id_base, rollout_idx),
        "rollout_idx": rollout_idx,
        "worker": worker,
        "sample_env_kwargs": sample_env_kwargs,
        "observation": observation,
        "history": [],
        "done": False,
        "last_info": info,
        "prompt_state": PureInferPromptState(prompt_config, task),
        "final_answer": None,
        "reward": 0.0,
        "rstar_gen_tokens_used": 0,
        "rstar_history_actions": [],
        "rstar_history_tool_calls": [],
        "lats_gen_tokens_used": 0,
        "lats_history_actions": [],
        "lats_history_tool_calls": [],
        "seed_offset": seed_offset,
        "env_kwargs": dict(env_kwargs),
    }


def _episode_max_tokens(episode: Dict[str, Any], args: argparse.Namespace) -> int:
    if not int(getattr(args, "rstar_infer_enable", 0)):
        return int(args.max_tokens)
    budget = int(getattr(args, "rstar_response_budget_tokens", 0))
    if budget <= 0:
        budget = int(args.max_steps) * int(args.max_tokens)
    used = int(episode.get("rstar_gen_tokens_used", 0))
    rem = budget - used
    if rem <= 0:
        return 0
    return max(1, min(int(args.max_tokens), rem))


def _per_branch_max_tokens(episode: Dict[str, Any], args: argparse.Namespace, branch_k: int) -> int:
    base = _episode_max_tokens(episode, args)
    branch_k = max(1, branch_k)
    if int(getattr(args, "rstar_infer_tree_budget_split", 1)):
        return base // branch_k
    return base


def _episode_max_tokens_lats(episode: Dict[str, Any], args: argparse.Namespace) -> int:
    if not int(args.lats_infer_enable):
        return int(args.max_tokens)
    budget = int(args.lats_response_budget_tokens)
    if budget <= 0:
        budget = int(args.max_steps) * int(args.max_tokens)
    used = int(episode["lats_gen_tokens_used"])
    rem = budget - used
    if rem <= 0:
        return 0
    return max(1, min(int(args.max_tokens), rem))


def _per_branch_max_tokens_lats(episode: Dict[str, Any], args: argparse.Namespace, branch_k: int) -> int:
    base = _episode_max_tokens_lats(episode, args)
    branch_k = max(1, branch_k)
    if int(args.lats_tree_budget_split):
        return base // branch_k
    return base


def _fork_worker_replay_and_step(
    args: argparse.Namespace,
    env_kwargs: Dict[str, Any],
    sample_env_kwargs: Dict[str, Any],
    seed_offset: int,
    history: List[Dict[str, Any]],
    raw_response: str,
    enable_commit: bool,
) -> Tuple[Any, str, float, bool, Dict[str, Any], str]:
    worker = create_worker(args, seed_offset, env_kwargs)
    observation, last_info = worker.reset(sample_env_kwargs)
    if history:
        raws = [h["raw_response"] for h in history]
        past_actions, _ = bash_coding_projection(raws, enable_commit=enable_commit)
        for a in past_actions:
            observation, _reward, done, last_info = worker.step(a)
            if done:
                break
    proj, _ = bash_coding_projection([raw_response], enable_commit=enable_commit)
    env_action = proj[0]
    observation, reward, done, last_info = worker.step(env_action)
    return worker, observation, reward, done, last_info, env_action


def _tree_winner_tuple(c: Dict[str, Any]) -> Tuple[float, int, int, int]:
    r = float(c["reward"])
    done = bool(c["done"])
    won = 1 if c["last_info"].get("won") else 0
    ia = str(c["env_action"])
    prefer_terminal_answer = 2 if (done and ia.startswith(ANSWER_PREFIX)) else (1 if done else 0)
    return (r, prefer_terminal_answer, won, len(ia))


def run_batched_turn_tree_search(
    episodes: List[Dict[str, Any]],
    args: argparse.Namespace,
    api_key: str,
    enable_commit: bool,
    action_regex: str,
) -> None:
    branch_k = max(1, int(getattr(args, "rstar_infer_branch_k", 4)))
    for episode in episodes:
        cap_turn = _episode_max_tokens(episode, args)
        prompt = episode["prompt_state"].build_text_obs(
            str(episode["observation"]),
            init=not episode["history"],
        )
        if int(getattr(args, "rstar_inject_history_json_block", 0)):
            hist = episode.get("rstar_history_tool_calls", [])
            if hist:
                prompt = build_rstar_history_prefix(hist, args) + prompt
        cap = _per_branch_max_tokens(episode, args, branch_k)
        env_kwargs = episode["env_kwargs"]
        sample_env_kwargs = episode["sample_env_kwargs"]
        seed_off = int(episode["seed_offset"])
        hist = episode["history"]

        if cap <= 0:
            raw_responses = [""] * branch_k
            completion_tokens = [0] * branch_k
        else:
            max_workers = max(1, min(branch_k, 32))

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

        candidates: List[Dict[str, Any]] = []
        for raw in raw_responses:
            w, obs, rew, done, last_info, env_action = _fork_worker_replay_and_step(
                args,
                env_kwargs,
                sample_env_kwargs,
                seed_off,
                hist,
                raw,
                enable_commit,
            )
            candidates.append(
                {
                    "worker": w,
                    "raw_response": raw,
                    "observation": obs,
                    "reward": rew,
                    "done": done,
                    "last_info": last_info,
                    "env_action": env_action,
                }
            )

        winner = max(candidates, key=_tree_winner_tuple)
        prev_observation = str(episode["observation"])
        raw_response = winner["raw_response"]
        n_tok = int(sum(completion_tokens))
        env_action = winner["env_action"]
        observation = str(winner["observation"])
        reward = float(winner["reward"])
        done = bool(winner["done"])
        last_info = winner["last_info"]

        old_w = episode["worker"]
        if old_w.work_dir and os.path.isdir(old_w.work_dir):
            old_w.close()
        episode["worker"] = winner["worker"]

        for c in candidates:
            if c["worker"] is not winner["worker"]:
                ow = c["worker"]
                if ow.work_dir and os.path.isdir(ow.work_dir):
                    ow.close()

        episode["prompt_state"].harness.apply_history_commit_decisions(
            episode["prompt_state"].memory,
            [last_info],
            next_text_obs=[observation],
        )
        episode["prompt_state"].store_transition(prev_observation, raw_response)
        if int(getattr(args, "rstar_infer_enable", 0)):
            episode["rstar_gen_tokens_used"] = int(episode["rstar_gen_tokens_used"]) + n_tok
            episode["rstar_history_actions"].append(
                {"raw_response_prefix": raw_response[:512], "completion_tokens": n_tok}
            )
        if env_action.startswith(ANSWER_PREFIX):
            episode["final_answer"] = env_action[len(ANSWER_PREFIX):].strip()
        elif env_action.startswith(FORMAT_VIOLATION_PREFIX):
            episode["final_answer"] = None
        episode["observation"] = observation
        episode["reward"] = reward
        episode["done"] = done
        if int(getattr(args, "rstar_infer_enable", 0)) and int(cap_turn) <= 0 and int(
            getattr(args, "rstar_stop_on_exhausted_budget", 1)
        ):
            episode["done"] = True
        episode["last_info"] = last_info
        if int(getattr(args, "rstar_infer_enable", 0)):
            episode["rstar_history_tool_calls"].append(
                {
                    "name": "bash_coding",
                    "model_output": raw_response,
                    "observation": observation,
                }
            )
        branch_scores = [float(c["reward"]) for c in candidates]
        step_payload: Dict[str, Any] = {
            "prompt": prompt,
            "raw_response": raw_response,
            "action": raw_response,
            "observation": prev_observation,
            "rstar_infer_tree_search": 1,
            "rstar_infer_branch_k": branch_k,
            "rstar_tree_branch_scores": branch_scores,
            "rstar_tree_completion_tokens_sum": n_tok,
        }
        if int(getattr(args, "rstar_infer_enable", 0)):
            step_payload["rstar_completion_tokens"] = n_tok
            step_payload["rstar_cumulative_gen_tokens"] = int(episode["rstar_gen_tokens_used"])
            step_payload["rstar_max_new_tokens_cap"] = int(cap_turn)
            step_payload["rstar_tool_history_steps_after"] = len(episode.get("rstar_history_tool_calls", []))
        episode["history"].append(step_payload)


def run_batched_turn(
    episodes: List[Dict[str, Any]],
    args: argparse.Namespace,
    api_key: str,
) -> None:
    active_episodes = [episode for episode in episodes if not episode["done"] and len(episode["history"]) < args.max_steps]
    if not active_episodes:
        return
    enable_commit = uses_commit_action_schema(str(active_episodes[0]["prompt_state"].harness.name))
    action_regex = get_bash_coding_action_regex(enable_commit=enable_commit)

    if int(args.lats_infer_enable):
        run_lats_batched_turn(
            active_episodes,
            args,
            api_key,
            enable_commit,
            action_regex,
            request_model_response,
            _fork_worker_replay_and_step,
            _episode_max_tokens_lats,
            _per_branch_max_tokens_lats,
        )
        return

    if int(getattr(args, "rstar_infer_enable", 0)) and int(getattr(args, "rstar_infer_tree_search", 0)):
        run_batched_turn_tree_search(active_episodes, args, api_key, enable_commit, action_regex)
        return

    prompts: List[str] = []
    for episode in active_episodes:
        prompt = episode["prompt_state"].build_text_obs(
            str(episode["observation"]),
            init=not episode["history"],
        )
        if int(getattr(args, "rstar_infer_enable", 0)) and int(getattr(args, "rstar_inject_history_json_block", 0)):
            hist = episode.get("rstar_history_tool_calls", [])
            if hist:
                prompt = build_rstar_history_prefix(hist, args) + prompt
        prompts.append(prompt)

    turn_max_tokens = [_episode_max_tokens(ep, args) for ep in active_episodes]
    if all(t > 0 for t in turn_max_tokens):
        max_workers = max(1, min(args.batch_size, len(prompts)))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    request_model_response,
                    api_base_url=args.api_base_url,
                    api_key=api_key,
                    api_model=args.api_model,
                    api_message_style=args.api_message_style,
                    prompt=prompt,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=mt,
                    llm_timeout=args.llm_timeout,
                    disable_qwen_thinking=bool(args.disable_qwen_thinking),
                    do_sample=bool(args.do_sample),
                    enable_regex_constraint=bool(args.enable_regex_constraint),
                    backend_name=args.backend_name,
                    action_regex=action_regex,
                )
                for prompt, mt in zip(prompts, turn_max_tokens, strict=True)
            ]
            results = [f.result() for f in futures]
    else:
        results = []
        for prompt, mt in zip(prompts, turn_max_tokens, strict=True):
            if mt <= 0:
                results.append(("", 0))
            else:
                results.append(
                    request_model_response(
                        api_base_url=args.api_base_url,
                        api_key=api_key,
                        api_model=args.api_model,
                        api_message_style=args.api_message_style,
                        prompt=prompt,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        max_tokens=mt,
                        llm_timeout=args.llm_timeout,
                        disable_qwen_thinking=bool(args.disable_qwen_thinking),
                        do_sample=bool(args.do_sample),
                        enable_regex_constraint=bool(args.enable_regex_constraint),
                        backend_name=args.backend_name,
                        action_regex=action_regex,
                    )
                )
    raw_responses = [r[0] for r in results]
    completion_tokens = [r[1] for r in results]

    projected_actions, _ = bash_coding_projection(raw_responses, enable_commit=enable_commit)
    for episode, prompt, raw_response, n_tok, env_action, cap in zip(
        active_episodes,
        prompts,
        raw_responses,
        completion_tokens,
        projected_actions,
        turn_max_tokens,
        strict=True,
    ):
        prev_observation = str(episode["observation"])
        observation, reward, done, last_info = episode["worker"].step(env_action)
        episode["prompt_state"].harness.apply_history_commit_decisions(
            episode["prompt_state"].memory,
            [last_info],
            next_text_obs=[str(observation)],
        )
        episode["prompt_state"].store_transition(prev_observation, raw_response)
        if int(getattr(args, "rstar_infer_enable", 0)):
            episode["rstar_gen_tokens_used"] = int(episode["rstar_gen_tokens_used"]) + int(n_tok)
            episode["rstar_history_actions"].append(
                {"raw_response_prefix": raw_response[:512], "completion_tokens": int(n_tok)}
            )
        if env_action.startswith(ANSWER_PREFIX):
            episode["final_answer"] = env_action[len(ANSWER_PREFIX):].strip()
        elif env_action.startswith(FORMAT_VIOLATION_PREFIX):
            episode["final_answer"] = None
        episode["observation"] = observation
        episode["reward"] = reward
        episode["done"] = done
        if int(getattr(args, "rstar_infer_enable", 0)) and int(cap) <= 0 and int(
            getattr(args, "rstar_stop_on_exhausted_budget", 1)
        ):
            episode["done"] = True
        episode["last_info"] = last_info
        if int(getattr(args, "rstar_infer_enable", 0)):
            episode["rstar_history_tool_calls"].append(
                {
                    "name": "bash_coding",
                    "model_output": raw_response,
                    "observation": str(observation),
                }
            )
        step_payload: Dict[str, Any] = {
            "prompt": prompt,
            "raw_response": raw_response,
            "action": raw_response,
            "observation": prev_observation,
        }
        if int(getattr(args, "rstar_infer_enable", 0)):
            step_payload["rstar_completion_tokens"] = int(n_tok)
            step_payload["rstar_cumulative_gen_tokens"] = int(episode["rstar_gen_tokens_used"])
            step_payload["rstar_max_new_tokens_cap"] = int(cap)
            step_payload["rstar_tool_history_steps_after"] = len(episode.get("rstar_history_tool_calls", []))
        episode["history"].append(step_payload)

    return


def finalize_episode(episode: Dict[str, Any], args: argparse.Namespace) -> List[Dict[str, Any]]:
    sample_id_base = episode["sample_id_base"]
    dataset_name = episode["dataset_name"]
    traj_uid = episode["traj_uid"]
    records: List[Dict[str, Any]] = []
    for turn_idx, step_record in enumerate(episode["history"]):
        sample_id = f"{sample_id_base}_idx_{turn_idx}"
        row: Dict[str, Any] = {
            "dataset": dataset_name,
            "sample_id": sample_id,
            "sample_id_base": sample_id_base,
            "turn_idx": turn_idx,
            "traj_uid": traj_uid,
            "sample_turn_id": f"{sample_id}_rollout_{episode['rollout_idx']}_{turn_idx}",
            "input": step_record["prompt"],
            "output": step_record["raw_response"],
            "score": None,
            "step": turn_idx,
            "rollout_idx": episode["rollout_idx"],
        }
        if int(getattr(args, "rstar_infer_enable", 0)):
            row["rstar_completion_tokens"] = step_record.get("rstar_completion_tokens")
            row["rstar_cumulative_gen_tokens"] = step_record.get("rstar_cumulative_gen_tokens")
            row["rstar_max_new_tokens_cap"] = step_record.get("rstar_max_new_tokens_cap")
            row["rstar_tool_history_steps_after"] = step_record.get("rstar_tool_history_steps_after")
        if int(args.lats_infer_enable):
            row["lats_completion_tokens"] = step_record.get("lats_completion_tokens")
            row["lats_cumulative_gen_tokens"] = step_record.get("lats_cumulative_gen_tokens")
            row["lats_max_new_tokens_cap"] = step_record.get("lats_max_new_tokens_cap")
            row["lats_tool_history_steps_after"] = step_record.get("lats_tool_history_steps_after")
        if step_record.get("lats_infer"):
            row["lats_infer"] = 1
            row["lats_n_generate_sample"] = step_record.get("lats_n_generate_sample")
            row["lats_iterations"] = step_record.get("lats_iterations")
            row["lats_exploration_weight"] = step_record.get("lats_exploration_weight")
            row["lats_arm_rewards"] = step_record.get("lats_arm_rewards")
            row["lats_arm_visits_final"] = step_record.get("lats_arm_visits_final")
            row["lats_chosen_arm_visits"] = step_record.get("lats_chosen_arm_visits")
            row["lats_completion_tokens_expand_sum"] = step_record.get("lats_completion_tokens_expand_sum")
        if step_record.get("rstar_infer_tree_search"):
            row["rstar_infer_tree_search"] = 1
            row["rstar_infer_branch_k"] = step_record.get("rstar_infer_branch_k")
            row["rstar_tree_branch_scores"] = step_record.get("rstar_tree_branch_scores")
            row["rstar_tree_completion_tokens_sum"] = step_record.get("rstar_tree_completion_tokens_sum")
        records.append(row)
    return records


def extract_sample_id_base(raw_line: str) -> str:
    match = re.search(r'"sample_id_base"\s*:\s*"([^"]+)', raw_line)
    if match is None:
        return ""
    return match.group(1).strip()


def sanitize_incomplete_output(output_path: Path) -> None:
    if not output_path.exists():
        return

    lines = output_path.read_text(encoding="utf-8").splitlines()
    last_idx = -1
    for idx in range(len(lines) - 1, -1, -1):
        if lines[idx].strip():
            last_idx = idx
            break
    if last_idx < 0:
        return

    last_raw = lines[last_idx].strip()
    if last_raw.endswith("}"):
        return

    sample_id_base = extract_sample_id_base(last_raw)
    if not sample_id_base:
        for idx in range(last_idx - 1, -1, -1):
            raw = lines[idx].strip()
            if not raw:
                continue
            sample_id_base = str(json.loads(raw)["sample_id_base"]).strip()
            break
    if not sample_id_base:
        raise ValueError(f"Cannot recover sample_id_base from truncated output: {output_path}")

    kept_lines: List[str] = []
    removed_count = 0
    for idx, line in enumerate(lines):
        raw = line.strip()
        if not raw:
            continue
        if idx == last_idx:
            removed_count += 1
            continue
        record_sample_id_base = str(json.loads(raw)["sample_id_base"]).strip()
        if record_sample_id_base == sample_id_base:
            removed_count += 1
            continue
        kept_lines.append(raw)

    rewritten = "\n".join(kept_lines)
    if rewritten:
        rewritten += "\n"
    output_path.write_text(rewritten, encoding="utf-8")
    print(f"Removed {removed_count} trailing records for sample_id_base={sample_id_base} from {output_path}")


def load_completed_trajs(output_path: Path) -> Set[Tuple[str, str]]:
    completed: Set[Tuple[str, str]] = set()
    if not output_path.exists():
        return completed
    with output_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            obj = json.loads(raw)
            dataset_name = str(obj.get("dataset", "")).strip()
            traj_uid = str(obj.get("traj_uid", "")).strip()
            if dataset_name and traj_uid:
                completed.add((dataset_name, traj_uid))
            sample_id_base = str(obj.get("sample_id_base", "")).strip()
            rollout_idx = obj.get("rollout_idx", None)
            if dataset_name and sample_id_base and rollout_idx is not None:
                completed.add((dataset_name, build_traj_uid(dataset_name, sample_id_base, int(rollout_idx))))
    return completed


def append_records(output_path: Path, records: List[Dict[str, Any]]) -> None:
    if not records:
        return
    with output_path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    if args.rollout_n <= 0:
        raise ValueError("--rollout_n must be positive")
    if args.flush_every <= 0:
        raise ValueError("--flush_every must be positive")
    runtime_config = load_runtime_config(args.config_yaml)
    apply_yaml_infer_and_rstar_overrides(args, runtime_config)
    _ensure_lats_infer_args(args)
    env_kwargs = build_runtime_env_kwargs(args, runtime_config)
    prompt_config = build_prompt_config(runtime_config, env_kwargs)
    os.environ["BASH_CODING_ENV_CONFIG_JSON"] = json.dumps(env_kwargs)
    api_key = os.environ.get(args.api_key_env, "")
    args.api_model = resolve_api_model(args.api_base_url, api_key, args.api_model, args.llm_timeout)

    dataset_name = infer_dataset_name_from_parquet(args.parquet)
    frame = pd.read_parquet(args.parquet)
    if "dataset" not in frame.columns:
        frame["dataset"] = dataset_name
    else:
        frame["dataset"] = frame["dataset"].fillna("").astype(str)
        frame.loc[frame["dataset"].str.len() == 0, "dataset"] = dataset_name
    if args.max_samples > 0:
        frame = frame.iloc[: args.max_samples].copy()

    output_path = Path(args.output_jsonl).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sanitize_incomplete_output(output_path)
    completed_trajs = load_completed_trajs(output_path)

    rollout_specs = []
    for idx in range(len(frame)):
        row = frame.iloc[idx]
        dataset_name = resolve_dataset_name(row, args.parquet)
        sample_id_base = str(row["id"])
        for rollout_idx in range(args.rollout_n):
            traj_uid = build_traj_uid(dataset_name, sample_id_base, rollout_idx)
            if (dataset_name, traj_uid) in completed_trajs:
                continue
            rollout_specs.append((row, rollout_idx))
    skipped = len(frame) * args.rollout_n - len(rollout_specs)
    pending_records: List[Dict[str, Any]] = []
    processed_since_flush = 0
    with tqdm(total=len(rollout_specs), desc="pure_infer") as progress:
        for start in range(0, len(rollout_specs), args.batch_size):
            batch_specs = rollout_specs[start:start + args.batch_size]
            episodes = [
                init_episode(row, args, env_kwargs, prompt_config, start + offset, rollout_idx)
                for offset, (row, rollout_idx) in enumerate(batch_specs)
            ]
            while any(not episode["done"] and len(episode["history"]) < args.max_steps for episode in episodes):
                run_batched_turn(episodes, args, api_key)
            for episode in episodes:
                pending_records.extend(finalize_episode(episode, args))
                processed_since_flush += 1
                progress.update(1)
                worker = episode["worker"]
                if worker.work_dir and os.path.isdir(worker.work_dir):
                    worker.close()
            if processed_since_flush >= args.flush_every:
                append_records(output_path, pending_records)
                pending_records = []
                processed_since_flush = 0
    append_records(output_path, pending_records)

    print(f"Saved pure inference trajectories to {output_path}")
    print(f"Skipped completed rollouts: {skipped}")


if __name__ == "__main__":
    main()
