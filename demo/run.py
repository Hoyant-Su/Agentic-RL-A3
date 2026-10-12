"""Run the trained A3 policy on two fixed ShellOps test tasks.

Install demo/requirements.txt and the CLI tools in main_entry/pre_build/install_apt.sh.
On Linux with a CUDA GPU, run:
    bash demo/run.sh --model /path/to/downloaded/A3-weights --harness both

Both harnesses share the same policy, batch, seed, six-turn budget,
and greedy unconstrained decoding.
This small demonstration is not a reproduction of the full paper evaluation.
"""

import argparse
from contextlib import ExitStack
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import tempfile

from huggingface_hub import snapshot_download
from omegaconf import OmegaConf
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, set_seed

from main_entry.cli_agent_bash_coding.env_package.bash_coding.envs import BashCodingWorker
from main_entry.cli_agent_bash_coding.env_package.bash_coding.projection import bash_coding_projection
from main_entry.cli_agent_bash_coding.env_package.bash_coding.sandbox_exec import run_in_unshare_sandbox
from main_entry.cli_agent_bash_coding.infer.pure_infer import PureInferPromptState


DEMO_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True, help="Downloaded Hugging Face weight directory")
    parser.add_argument("--harness", choices=["vanilla", "sigma_reveal_rd", "both"], default="both")
    parser.add_argument("--output-dir", type=Path, default=DEMO_DIR.parent / "outputs" / "demo")
    parser.add_argument("--cache-dir", type=Path, help="Hugging Face dataset cache")
    return parser.parse_args()


def load_tasks(config, cache_dir, data_dir):
    dataset = config.dataset
    patterns = [dataset.parquet]
    patterns.extend(f"shellops/assets/{sample_id}/*" for sample_id in dataset.sample_ids)
    root = Path(snapshot_download(
        repo_id=dataset.repo_id,
        repo_type="dataset",
        revision=dataset.revision,
        allow_patterns=patterns,
        cache_dir=cache_dir,
    ))
    frame = pd.read_parquet(root / dataset.parquet).set_index("id", verify_integrity=True)
    tasks = []
    for sample_id in dataset.sample_ids:
        # Materialize Hub cache symlinks for the filesystem-based verifier.
        assets = Path(data_dir) / sample_id
        shutil.copytree(root / "shellops" / "assets" / sample_id, assets)
        task = deepcopy(frame.loc[sample_id, "env_kwargs"])
        task["init_dir"] = str(assets / "init")
        task["reward_spec"]["gold_dir"] = str(assets / "gold")
        tasks.append((sample_id, task))
    return tasks


def visible(text, worker):
    # A temporary host path is not task evidence; commands already run in this directory.
    return text.replace(worker.work_dir, ".")


def run_harness(name, tasks, config, model, tokenizer, generation, output_dir):
    set_seed(config.seed)
    prompt_config = OmegaConf.create({"env": config.env, "reward": config.reward})
    prompt_config.env.bash_coding_harness = name
    env_kwargs = OmegaConf.to_container(prompt_config.env, resolve=True)
    env_kwargs.update(OmegaConf.to_container(config.reward, resolve=True))
    summaries = []
    with ExitStack() as stack:
        stream = stack.enter_context((output_dir / f"{name}.jsonl").open("x"))
        episodes = []
        for sample_id, task in tasks:
            worker = BashCodingWorker(seed=config.seed, env_kwargs=env_kwargs)
            observation, info = worker.reset(task)
            stack.callback(shutil.rmtree, worker.work_dir)
            episodes.append({
                "id": sample_id,
                "worker": worker,
                "prompt": PureInferPromptState(prompt_config, info["task"]),
                "observation": visible(observation, worker),
                "done": False,
            })
        for turn in range(config.env.max_steps):
            active = [episode for episode in episodes if not episode["done"]]
            if not active:
                break
            prompts = [episode["prompt"].build_text_obs(episode["observation"], init=turn == 0) for episode in active]
            chats = [tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True,
                enable_thinking=config.model.enable_thinking,
            ) for prompt in prompts]
            inputs = tokenizer(chats, padding=True, add_special_tokens=False, return_tensors="pt").to(model.device)
            if inputs.input_ids.shape[1] > config.model.max_prompt_tokens:
                raise ValueError("Prompt exceeds the declared token budget; no prompt truncation is performed")
            with torch.inference_mode():
                generated = model.generate(**inputs, generation_config=generation)
            generated = generated[:, inputs.input_ids.shape[1]:]
            responses = tokenizer.batch_decode(generated, skip_special_tokens=True)
            actions, _ = bash_coding_projection(
                responses, enable_commit=active[0]["worker"].harness.uses_commit_action_schema(),
            )
            for episode, prompt, response, action in zip(active, prompts, responses, actions, strict=True):
                worker = episode["worker"]
                observation, reward, done, info = worker.step(action)
                if info.get("sandbox_error") or info.get("env_error"):
                    raise RuntimeError(info["error"])
                episode["prompt"].store_transition(episode["observation"], response)
                episode["observation"] = visible(observation, worker)
                episode["done"] = done
                record = {
                    "sample_id": episode["id"], "turn": turn + 1,
                    "prompt": prompt, "response": response,
                    "action": action,
                    "observation": episode["observation"],
                    "stdout": visible(info["stdout"], worker),
                    "stderr": visible(info["stderr"], worker),
                    "returncode": info["returncode"],
                    "reward": reward,
                    "done": done, "success": bool(info["won"]),
                }
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                print(f"[{name}] {episode['id']} turn {turn + 1}: {response}", flush=True)
                print(episode["observation"], flush=True)
                if done:
                    summaries.append({"sample_id": episode["id"], "turns": turn + 1, "success": bool(info["won"])})
        if len(summaries) != len(tasks):
            raise RuntimeError("An episode did not terminate within the declared turn budget")
    return summaries


def main():
    args = parse_args()
    config = OmegaConf.load(DEMO_DIR / "config.yaml")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    set_seed(config.seed)
    with tempfile.TemporaryDirectory() as work_dir:
        result = run_in_unshare_sandbox(work_dir=work_dir, command="printf sandbox-ok", timeout_s=config.env.timeout)
        if result.returncode != 0 or result.stdout != "sandbox-ok":
            raise RuntimeError(f"Sandbox preflight failed: {result.stderr}")
    with tempfile.TemporaryDirectory(prefix="shellops_") as data_dir:
        tasks = load_tasks(config, args.cache_dir, data_dir)
        print("Loading A3 weights", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left", local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=getattr(torch, config.model.dtype),
            attn_implementation=config.model.attention,
            device_map={"": "cuda:0"}, local_files_only=True,
        ).eval()
        generation = GenerationConfig(
            **OmegaConf.to_container(config.generation, resolve=True),
            eos_token_id=model.generation_config.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        names = ["vanilla", "sigma_reveal_rd"] if args.harness == "both" else [args.harness]
        summary = {name: run_harness(name, tasks, config, model, tokenizer, generation, args.output_dir) for name in names}
        with (args.output_dir / "summary.json").open("x") as stream:
            json.dump(summary, stream, indent=2)
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
