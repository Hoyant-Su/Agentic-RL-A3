import os
import datasets
import argparse
import json
import shutil
from typing import List, Dict, Any

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local_dir",
        default="main_entry/data/gsm8k",
        help="Directory to save processed parquet files",
    )
    parser.add_argument(
        "--local_dataset_path",
        default=None,
        help="Path to local parquet files directory (if already processed)",
    )
    parser.add_argument(
        "--data_source",
        default="openai/gsm8k",
        help="HuggingFace dataset name to load (e.g., openai/gsm8k)",
    )
    parser.add_argument(
        "--jsonl_path",
        default=None,
        help="Path to bash-coding jsonl dataset (overrides --data_source if set)",
    )
    parser.add_argument(
        "--train_data_size",
        default=0,
        type=int,
        help="Number of training samples to use. 0 means use all data.",
    )
    parser.add_argument(
        "--val_data_size",
        default=0,
        type=int,
        help="Number of validation samples to use. 0 means use all data.",
    )
    parser.add_argument(
        "--num_proc",
        default=1,
        type=int,
        help="datasets.map num_proc. Use 1 to avoid multiprocess issues on some Python versions.",
    )
    parser.add_argument(
        "--default_string_match",
        default="exact",
        choices=["exact", "fuzzy"],
        help="Default match mode for string expected.",
    )
    parser.add_argument(
        "--fuzzy_threshold",
        default=0.85,
        type=float,
        help="Threshold for fuzzy string match (SequenceMatcher ratio).",
    )
    parser.add_argument(
        "--clean_assets",
        default=0,
        type=int,
        help="If 1, remove {local_dir}/assets before materializing init/gold (prevents stale assets from previous runs).",
    )
    args = parser.parse_args()
    args.local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(args.local_dir, exist_ok=True)
    dataset_name = os.path.basename(os.path.normpath(args.local_dir))
    if int(args.clean_assets) == 1:
        shutil.rmtree(os.path.join(args.local_dir, "assets"), ignore_errors=True)
        shutil.rmtree(os.path.join(args.local_dir, "dest_assets"), ignore_errors=True)
    if args.local_dataset_path:
        local_dataset_path = os.path.expanduser(args.local_dataset_path)
        print(f"Loading dataset from local path: {local_dataset_path}")
        train_file = os.path.join(local_dataset_path, "train.parquet")
        test_file = os.path.join(local_dataset_path, "test.parquet")
        if os.path.exists(train_file) and os.path.exists(test_file):
            train_dataset = datasets.load_dataset(
                "parquet", data_files=train_file, split="train"
            )
            test_dataset = datasets.load_dataset(
                "parquet", data_files=test_file, split="train"
            )
        else:
            raise FileNotFoundError(f"Parquet files not found in {local_dataset_path}")
    elif args.jsonl_path:
        jsonl_path = os.path.expanduser(args.jsonl_path)
        if not os.path.isfile(jsonl_path):
            raise FileNotFoundError(f"jsonl_path not found: {jsonl_path}")
        print(f"Loading dataset from jsonl: {jsonl_path}")
        ds = datasets.load_dataset("json", data_files=jsonl_path, split="train")
        n = len(ds)
        n_train = max(1, int(n * 0.8))
        indices = list(range(n))
        import random

        random.seed(42)
        random.shuffle(indices)
        train_indices = sorted(indices[:n_train])
        test_indices = sorted(indices[n_train:])
        train_dataset = ds.select(train_indices)
        test_dataset = ds.select(test_indices)
        print(f"Split: {n_train} train, {n - n_train} test (80/20 split, seed=42)")
    else:
        print(f"Loading dataset from huggingface: {args.data_source}")
        dataset = datasets.load_dataset(args.data_source, "main")
        train_dataset = dataset["train"]
        test_dataset = dataset["test"]
    if args.train_data_size > 0:
        train_dataset = train_dataset.select(
            range(min(args.train_data_size, len(train_dataset)))
        )
    if args.val_data_size > 0:
        test_dataset = test_dataset.select(
            range(min(args.val_data_size, len(test_dataset)))
        )

    def _write_files(root: str, files: List[Dict[str, Any]]):
        os.makedirs(root, exist_ok=True)
        for f in files or []:
            rel = f["path"]
            if str(f.get("type", "")).strip().lower() == "dir":
                os.makedirs(os.path.join(root, rel), exist_ok=True)
                continue
            if str(f.get("type", "")).strip().lower() == "truncate":
                os.makedirs(os.path.join(root, os.path.dirname(rel)), exist_ok=True)
                size_bytes = int(f.get("size_bytes", 0))
                with open(os.path.join(root, rel), "wb") as wf:
                    wf.truncate(size_bytes)
                continue
            content = f.get("content", "")
            abs_path = os.path.join(root, rel)
            os.makedirs(os.path.dirname(abs_path), exist_ok=True)
            with open(abs_path, "w", encoding="utf-8") as wf:
                wf.write(content)

    def _materialize_assets(example: Dict[str, Any]) -> Dict[str, Any]:
        ex_id = example.get("id", "")
        pre_files = example.get("pre_files", None)
        post_files = example.get("post_files", None)
        task_type = str(example.get("task_type", "") or "").strip().lower()
        expected_text = (
            example.get("expected_text")
            or example.get("expected")
            or example.get("answer")
        )
        if expected_text is not None:
            expected_text = str(expected_text).strip() or None
        has_post_files = isinstance(post_files, list) and len(post_files) > 0
        has_expected = expected_text is not None
        assets_root = os.path.join(args.local_dir, "assets", ex_id)
        init_dir = os.path.join(assets_root, "init")
        gold_dir = os.path.join(args.local_dir, "dest_assets", ex_id)
        if pre_files is not None:
            _write_files(init_dir, pre_files)
        if post_files is not None:
            _write_files(gold_dir, post_files)
        out = {}
        if task_type == "hybrid" or (
            has_expected and has_post_files and task_type not in {"string", "files"}
        ):
            if not (has_expected and has_post_files):
                if pre_files is not None:
                    out["init_dir"] = init_dir
                return out
            spec = _infer_string_reward_spec(expected_text, example)
            spec["type"] = "hybrid"
            spec["gold_dir"] = gold_dir
            out["reward_spec"] = spec
            if pre_files is not None:
                out["init_dir"] = init_dir
            return out
        if task_type == "string" or (has_expected and not has_post_files):
            out["reward_spec"] = _infer_string_reward_spec(expected_text, example)
            if pre_files is not None:
                out["init_dir"] = init_dir
            return out
        if task_type == "files" or (has_post_files and not has_expected):
            out["reward_spec"] = {
                "type": "files",
                "gold_dir": gold_dir,
                "success_reward": 1.0,
            }
            out["init_dir"] = init_dir
            return out
        if pre_files is not None:
            return {"init_dir": init_dir}
        return {}

    def _infer_string_reward_spec(
        expected_text: str, example: Dict[str, Any]
    ) -> Dict[str, Any]:
        ex_spec = example.get("reward_spec", None)
        if isinstance(ex_spec, dict) and ex_spec.get("type") == "string":
            merged = dict(ex_spec)
            merged.setdefault("expected", expected_text)
            merged.setdefault("success_reward", 1.0)
            match = merged.get("match", "exact")
            if match not in {"exact", "fuzzy"}:
                raise ValueError(f"Unsupported match type: {match}")
            if merged.get("match") == "fuzzy":
                merged.setdefault("threshold", float(args.fuzzy_threshold))
                merged.setdefault("ignore_case", True)
            if merged.get("match") == "exact":
                merged.setdefault("ignore_case", True)
            return merged
        if args.default_string_match == "fuzzy":
            return {
                "type": "string",
                "match": "fuzzy",
                "expected": expected_text,
                "threshold": float(args.fuzzy_threshold),
                "ignore_case": True,
                "success_reward": 1.0,
            }
        return {
            "type": "string",
            "match": "exact",
            "expected": expected_text,
            "success_reward": 1.0,
        }

    def make_map_fn(split):
        def process_fn(example, idx):
            question = example.get("query", example.get("question", ""))
            prompt = ""
            extra_env_kwargs = {}
            has_expected = (
                "expected_text" in example
                or "expected" in example
                or "answer" in example
            )
            if "pre_files" in example or "post_files" in example or has_expected:
                extra_env_kwargs = _materialize_assets(example)
            data = {
                "data_source": "bash_coding",
                "dataset": dataset_name,
                "prompt": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                "ability": "agent",
                "env_kwargs": {
                    "task": question,
                    "index": idx,
                    **extra_env_kwargs,
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "id": example.get("id", str(idx)),
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(
        function=make_map_fn("train"),
        with_indices=True,
        num_proc=max(1, int(args.num_proc)),
    )
    test_dataset = test_dataset.map(
        function=make_map_fn("test"),
        with_indices=True,
        num_proc=max(1, int(args.num_proc)),
    )
    train_dataset.to_parquet(os.path.join(args.local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(args.local_dir, "test.parquet"))
    n_train_spec = sum(
        1
        for i in range(len(train_dataset))
        if (train_dataset[i].get("env_kwargs") or {}).get("reward_spec")
    )
    n_test_spec = sum(
        1
        for i in range(len(test_dataset))
        if (test_dataset[i].get("env_kwargs") or {}).get("reward_spec")
    )
    print(f"Data prepared and saved to: {args.local_dir}")
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(test_dataset)}")
    print(
        f"Train with reward_spec: {n_train_spec}, Test with reward_spec: {n_test_spec}"
    )
