#!/usr/bin/env python3
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib.util
import json
import os
import sqlite3
from pathlib import Path
from configparser import ConfigParser
from collections import Counter, defaultdict
import shutil
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple
import datasets
from diff_match_patch import diff_match_patch
import yaml
from main_entry.cli_agent_bash_coding.action_schema import AnswerAction
from main_entry.cli_agent_bash_coding.action_schema import CodeAction
from main_entry.cli_agent_bash_coding.action_schema import parse_bash_coding_action
from main_entry.cli_agent_bash_coding.tooling.semantic_similarity.semantic_similarity import (
    semantic_similarity_batch,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SANDBOX_EXEC_PATH = (
    PROJECT_ROOT
    / "main_entry/cli_agent_bash_coding/env_package/bash_coding/sandbox_exec.py"
)
_sandbox_spec = importlib.util.spec_from_file_location(
    "bash_coding_sandbox_exec", SANDBOX_EXEC_PATH
)
_sandbox_mod = importlib.util.module_from_spec(_sandbox_spec)
assert _sandbox_spec is not None and _sandbox_spec.loader is not None
sys.modules[_sandbox_spec.name] = _sandbox_mod
_sandbox_spec.loader.exec_module(_sandbox_mod)
SandboxError = _sandbox_mod.SandboxError
run_in_unshare_sandbox = _sandbox_mod.run_in_unshare_sandbox
ANSWER_PREFIX = "__ANSWER__:"
FORMAT_VIOLATION_PREFIX = "__FORMAT_VIOLATION__"

def extract_answer(text: str, enable_commit: bool) -> Optional[str]:
    parsed = parse_bash_coding_action(text, enable_commit=enable_commit)
    if not isinstance(parsed, AnswerAction):
        return None
    return parsed.answer

def normalize_text(s: str, ignore_case: bool = True) -> str:
    out = " ".join((s or "").strip().split())
    return out.lower() if ignore_case else out

def normalize_bool(s: str) -> Optional[bool]:
    token = normalize_text(s, ignore_case=True)
    if token in ConfigParser.BOOLEAN_STATES:
        return bool(ConfigParser.BOOLEAN_STATES[token])
    return None

def parse_expected_alternatives(expected: Any) -> Optional[List[Any]]:
    parsed = expected
    if isinstance(expected, str):
        try:
            parsed = json.loads(expected)
        except Exception:
            parsed = expected
    if isinstance(parsed, dict) and isinstance(parsed.get("any_of"), list):
        return list(parsed["any_of"])
    return None

def match_single_exact(pred: str, expected: Any, ignore_case: bool) -> bool:
    pb = normalize_bool(pred)
    gb = normalize_bool(str(expected))
    if pb is not None and gb is not None:
        return pb == gb
    p = normalize_text(pred, ignore_case=ignore_case)
    g = normalize_text(str(expected), ignore_case=ignore_case)
    return p == g

def match_string(pred: str, spec: Dict[str, Any]) -> bool:
    return string_score(pred, spec) == 1.0

def string_score(pred: str, spec: Dict[str, Any]) -> float:
    expected = spec.get("expected", None)
    if expected is None:
        return 0.0
    ignore_case = bool(spec.get("ignore_case", True))
    alternatives = parse_expected_alternatives(expected)
    if alternatives is not None:
        return float(
            any(match_single_exact(pred, item, ignore_case) for item in alternatives)
        )
    return float(match_single_exact(pred, expected, ignore_case))

def final_string_score(
    pred: str, spec: Dict[str, Any], llm_judge_score: Optional[float] = None
) -> float:
    exact_score = string_score(pred, spec)
    if exact_score == 1.0:
        return 1.0
    return 0.0 if llm_judge_score is None else float(llm_judge_score)

def _format_violation(reason: str) -> str:
    return f"{FORMAT_VIOLATION_PREFIX}:{reason}"

def project_action(text: str, enable_commit: bool) -> str:
    parsed = parse_bash_coding_action(text, enable_commit=enable_commit)
    if parsed is None:
        return _format_violation("invalid_json_action")
    if isinstance(parsed, AnswerAction):
        return f"{ANSWER_PREFIX}{parsed.answer}"
    if isinstance(parsed, CodeAction):
        return parsed.code
    return _format_violation("invalid_json_action")

def _extract_answer_payload(output_text: str) -> str:
    return output_text

def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _is_sqlite_file(path: str, raw: bytes) -> bool:
    return raw.startswith(b"SQLite format 3\x00") or os.path.splitext(path)[
        1
    ].lower() in {".db", ".sqlite", ".sqlite3"}

def _quote_sqlite_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'

def _normalize_sqlite_value(value: Any) -> Any:
    if isinstance(value, float):
        return format(value, ".17g")
    if isinstance(value, bytes):
        return {"hex": value.hex()}
    return value

def _canonicalize_sqlite_text(path: str) -> List[str]:
    conn = sqlite3.connect(path)
    try:
        lines: List[str] = []
        objects = conn.execute(
            """
            SELECT type, name, tbl_name, sql
            FROM sqlite_master
            WHERE name NOT LIKE 'sqlite_%'
            ORDER BY type, name
            """
        ).fetchall()
        for obj_type, name, tbl_name, sql in objects:
            lines.append(f"{obj_type}\t{name}\t{tbl_name}")
            lines.append(f"sql\t{sql or ''}")
            if obj_type != "table":
                continue
            table_name = _quote_sqlite_identifier(name)
            columns = [
                row[1]
                for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()
            ]
            lines.append("columns\t" + json.dumps(columns, ensure_ascii=False))
            rows = conn.execute(f"SELECT * FROM {table_name}").fetchall()
            row_texts = sorted(
                json.dumps(
                    [_normalize_sqlite_value(value) for value in row],
                    ensure_ascii=False,
                )
                for row in rows
            )
            lines.extend(f"row\t{row_text}" for row_text in row_texts)
        return lines
    finally:
        conn.close()

def _canonicalize_file_text(
    path: str, raw: bytes, max_bytes: int
) -> Tuple[List[str], bool]:
    if _is_sqlite_file(path, raw):
        return _canonicalize_sqlite_text(path), False
    truncated = len(raw) > max_bytes
    if truncated:
        raw = raw[:max_bytes]
    return raw.decode("utf-8", errors="replace").splitlines(), truncated

def _snapshot_dir(root: str) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in {".tmp", ".sandbox"}]
        for fn in filenames:
            if fn == ".DS_Store":
                continue
            p = os.path.join(dirpath, fn)
            if not os.path.isfile(p) or os.path.islink(p):
                continue
            rel = os.path.relpath(p, root)
            result[rel] = _sha256_file(p)
    return result

def _capture_fs_state(root: str, max_bytes: int = 200_000) -> Dict[str, Dict[str, Any]]:
    state: Dict[str, Dict[str, Any]] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in {".tmp", ".sandbox"}]
        for fn in filenames:
            if fn == ".DS_Store":
                continue
            p = os.path.join(dirpath, fn)
            if not os.path.isfile(p) or os.path.islink(p):
                continue
            rel = os.path.relpath(p, root)
            sha = _sha256_file(p)
            with open(p, "rb") as bf:
                raw = bf.read(max_bytes + 1)
            text, truncated = _canonicalize_file_text(p, raw, max_bytes)
            state[rel] = {"sha": sha, "text": text, "truncated": truncated}
    return state

def _line_mode_text(lines: List[str]) -> str:
    return "".join(f"{line}\n" for line in lines)

def _line_edit_multiset(
    before_lines: List[str] | None,
    after_lines: List[str] | None,
) -> Counter[Tuple[str, str]]:
    before = before_lines or []
    after = after_lines or []
    edits: Counter[Tuple[str, str]] = Counter()
    dmp = diff_match_patch()
    before_text = _line_mode_text(before)
    after_text = _line_mode_text(after)
    before_chars, after_chars, line_array = dmp.diff_linesToChars(
        before_text, after_text
    )
    diffs = dmp.diff_main(before_chars, after_chars, False)
    dmp.diff_charsToLines(diffs, line_array)
    dmp.diff_cleanupMerge(diffs)
    for op, text in diffs:
        if op == 0:
            continue
        for line in text.splitlines():
            if op < 0:
                edits[("subline", line)] += 1
            else:
                edits[("addline", line)] += 1
    return edits

def _fs_edit_multiset(
    before_state: Dict[str, Dict[str, Any]],
    after_state: Dict[str, Dict[str, Any]],
) -> Counter[Tuple[str, str]]:
    edits: Counter[Tuple[str, str]] = Counter()
    for path in sorted(set(before_state) | set(after_state)):
        before_entry = before_state.get(path)
        after_entry = after_state.get(path)
        before_sha = before_entry.get("sha") if before_entry is not None else None
        after_sha = after_entry.get("sha") if after_entry is not None else None
        if before_sha == after_sha:
            continue
        edits.update(
            _line_edit_multiset(
                before_entry["text"] if before_entry is not None else None,
                after_entry["text"] if after_entry is not None else None,
            )
        )
    return edits

def _delta_coverage_score(
    work_dir: str,
    init_fs_state: Dict[str, Dict[str, Any]],
    gold_fs_state: Dict[str, Dict[str, Any]],
) -> float:
    current_fs_state = _capture_fs_state(work_dir)
    gold_edits = _fs_edit_multiset(init_fs_state, gold_fs_state)
    if not gold_edits:
        return 1.0
    model_edits = _fs_edit_multiset(init_fs_state, current_fs_state)
    matched = sum((model_edits & gold_edits).values())
    total = sum(gold_edits.values())
    return matched / total

def _resolve_project_path(path: str) -> str:
    return path if os.path.isabs(path) else str(PROJECT_ROOT / path)

def _snapshot_delta(before: Dict[str, str], after: Dict[str, str]) -> Dict[str, Any]:
    before_keys = set(before)
    after_keys = set(after)
    created = {path: after[path] for path in sorted(after_keys - before_keys)}
    deleted = sorted(before_keys - after_keys)
    modified = {
        path: after[path]
        for path in sorted(before_keys & after_keys)
        if before[path] != after[path]
    }
    return {
        "created": created,
        "modified": modified,
        "deleted": deleted,
    }

def _is_correct_files(
    work_dir: str,
    gold_fs_state: Optional[Dict[str, Dict[str, Any]]],
) -> bool:
    if gold_fs_state is None:
        return False
    if not work_dir or not os.path.isdir(work_dir):
        return False
    current_fs_state = _capture_fs_state(work_dir)
    if set(current_fs_state) != set(gold_fs_state):
        return False
    for path, target_entry in gold_fs_state.items():
        current_entry = current_fs_state.get(path)
        if current_entry is None:
            return False
        if current_entry.get("text") != target_entry.get("text"):
            return False
    return True

def _check_task_completion(
    output_text: str,
    spec: Dict[str, Any],
    work_dir: str,
    gold_fs_state: Optional[Dict[str, Dict[str, Any]]],
    llm_judge_score: Optional[float] = None,
) -> bool:
    reward_type = spec.get("type", None)
    answer_out = _extract_answer_payload(output_text)
    if reward_type == "string":
        return (
            final_string_score(answer_out, spec, llm_judge_score=llm_judge_score) == 1.0
        )
    if reward_type == "files":
        return _is_correct_files(work_dir, gold_fs_state)
    if reward_type == "hybrid":
        return final_string_score(
            answer_out, spec, llm_judge_score=llm_judge_score
        ) == 1.0 and _is_correct_files(work_dir, gold_fs_state)
    return False

def _judge_string_candidates(candidates: List[Dict[str, Any]]) -> Dict[str, float]:
    pair_requests: List[Tuple[str, str]] = []
    pair_owners: List[str] = []
    for item in candidates:
        expected = item["spec"]["expected"]
        alternatives = parse_expected_alternatives(expected)
        expected_values = alternatives if alternatives is not None else [expected]
        for candidate in expected_values:
            pair_requests.append((item["answer"], str(candidate)))
            pair_owners.append(item["traj_uid"])
    if not pair_requests:
        return {}
    try:
        scores = semantic_similarity_batch(pair_requests, batch_size=len(pair_requests))
    except Exception as exc:
        print(
            f"WARNING: semantic similarity judge unavailable, skip llm-incorporated scoring: {exc}"
        )
        return {}
    best_scores: Dict[str, float] = {}
    for traj_uid, score in zip(pair_owners, scores):
        prev = best_scores.get(traj_uid, 0.0)
        if score > prev:
            best_scores[traj_uid] = float(score)
    return best_scores

def _infer_dataset_name_from_parquet(parquet_path: str) -> Optional[str]:
    parquet = Path(_resolve_project_path(parquet_path)).resolve()
    if parquet.parent.name:
        return parquet.parent.name
    return None

def _default_parquet_for_dataset(dataset_name: str) -> str:
    return str(PROJECT_ROOT / "main_entry" / "data" / dataset_name / "test.parquet")

def _resolve_enable_commit(results_jsonl: str) -> bool:
    config_path = Path(results_jsonl).resolve().parent / "config.yaml"
    if not config_path.is_file():
        raise SystemExit(f"Missing config.yaml beside results_jsonl: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    env_config = config.get("env", None) if isinstance(config, dict) else None
    if not isinstance(env_config, dict) or "bash_coding_harness" not in env_config:
        raise SystemExit(f"Missing env.bash_coding_harness in {config_path}")
    return str(env_config["bash_coding_harness"]).strip().lower() == "commit_if_better"

def _collect_result_datasets(results_jsonl: str) -> List[str]:
    dataset_names: List[str] = []
    seen = set()
    with open(results_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            dataset_name = str(obj.get("dataset", "") or "").strip()
            if not dataset_name or dataset_name in seen:
                continue
            seen.add(dataset_name)
            dataset_names.append(dataset_name)
    return dataset_names

def load_gt_map(
    parquet_path: str, dataset_name: Optional[str]
) -> Dict[Tuple[Optional[str], str], Dict[str, Any]]:
    ds = datasets.load_dataset("parquet", data_files=parquet_path, split="train")
    gt = {}
    for ex in ds:
        extra = ex.get("extra_info", None)
        if isinstance(extra, dict) and "id" in extra:
            sid = str(extra["id"])
        elif ex.get("id", None) is not None:
            sid = str(ex.get("id", ""))
        else:
            raise ValueError("Each row must contain id in extra_info.id or id.")
        env_kwargs = ex.get("env_kwargs", {}) or {}
        spec = env_kwargs.get("reward_spec", {}) or {}
        gt[sid] = {
            "env_kwargs": env_kwargs,
            "reward_spec": spec,
            "post_files": ex.get("post_files", None),
        }
        gt[(dataset_name, sid)] = gt.pop(sid)
    return gt

def build_gt_map(
    results_jsonl: str, parquet_path: str
) -> tuple[
    Dict[Tuple[Optional[str], str], Dict[str, Any]], Dict[str, str], Optional[str]
]:
    resolved_parquets: Dict[str, str] = {}
    gt_map: Dict[Tuple[Optional[str], str], Dict[str, Any]] = {}
    if parquet_path:
        dataset_name = _infer_dataset_name_from_parquet(parquet_path)
        resolved_path = _resolve_project_path(parquet_path)
        resolved_key = dataset_name or "__default__"
        resolved_parquets[resolved_key] = resolved_path
        gt_map.update(load_gt_map(resolved_path, dataset_name))
        return gt_map, resolved_parquets, dataset_name
    dataset_names = _collect_result_datasets(results_jsonl)
    if not dataset_names:
        raise SystemExit(
            "Missing --parquet and results_jsonl does not contain a dataset field."
        )
    for dataset_name in dataset_names:
        resolved_path = _default_parquet_for_dataset(dataset_name)
        if not os.path.isfile(resolved_path):
            raise SystemExit(
                f"Auto parquet resolution failed for dataset={dataset_name}: {resolved_path}"
            )
        resolved_parquets[dataset_name] = resolved_path
        gt_map.update(load_gt_map(resolved_path, dataset_name))
    default_dataset_name = dataset_names[0] if len(dataset_names) == 1 else None
    return gt_map, resolved_parquets, default_dataset_name

def safe_ratio(numerator: int, denominator: int) -> float:
    return (numerator / denominator) if denominator else 0.0

def pass_at_k(num_samples: int, num_correct: int, k: int) -> float:
    if k <= 0 or num_samples <= 0:
        return 0.0
    if num_samples - num_correct < k:
        return 1.0
    prod = 1.0
    for i in range(k):
        prod *= (num_samples - num_correct - i) / (num_samples - i)
    return 1.0 - prod

def _resolve_val_rollout_n(results_jsonl: str) -> Optional[int]:
    config_path = Path(results_jsonl).resolve().parent / "config.yaml"
    if not config_path.is_file():
        return None
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        return None
    hyperparam = config.get("hyperparam", None)
    if not isinstance(hyperparam, dict):
        return None
    rollout_n = hyperparam.get("val_rollout_n", None)
    if rollout_n is None:
        return None
    rollout_n = int(rollout_n)
    if rollout_n <= 0:
        return None
    return rollout_n

def _extract_sample_idx(sample_id: str) -> int:
    sid = str(sample_id or "")
    return int(sid.rsplit("_idx_", 1)[-1])

def _group_steps_by_traj(
    results_jsonl: str,
) -> Dict[str, list[Tuple[int, Dict[str, Any]]]]:
    by_traj: Dict[str, list[Tuple[int, Dict[str, Any]]]] = defaultdict(list)
    with open(results_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            traj = str(obj.get("traj_uid", ""))
            step_idx = _extract_sample_idx(str(obj.get("sample_id", "")))
            by_traj[traj].append((step_idx, obj))
    for traj in by_traj:
        by_traj[traj].sort(key=lambda x: x[0])
    return by_traj

def _replay_trajectory(
    traj_steps: list[Tuple[int, Dict[str, Any]]],
    sample_env_kwargs: Dict[str, Any],
    enable_commit: bool,
) -> tuple[Dict[str, Any], Optional[str]]:
    work_dir = tempfile.mkdtemp(prefix="bash_coding_eval_")
    try:
        init_dir = sample_env_kwargs.get("init_dir", None)
        if init_dir:
            init_dir = _resolve_project_path(str(init_dir))
            if not os.path.isdir(init_dir):
                return {
                    "correct": False,
                    "file_sha_match": None,
                    "delta_coverage": None,
                }, f"init_dir_missing:{init_dir}"
            shutil.copytree(init_dir, work_dir, dirs_exist_ok=True)
        init_snapshot = _snapshot_dir(work_dir)
        spec = (
            sample_env_kwargs.get("reward_spec", {})
            if isinstance(sample_env_kwargs.get("reward_spec", {}), dict)
            else {}
        )
        gold_snapshot: Optional[Dict[str, str]] = None
        init_fs_state: Optional[Dict[str, Dict[str, Any]]] = None
        gold_fs_state: Optional[Dict[str, Dict[str, Any]]] = None
        if spec.get("type") in {"files", "hybrid"}:
            gold_dir = spec.get("gold_dir", None)
            if not gold_dir:
                return {
                    "correct": False,
                    "file_sha_match": None,
                    "delta_coverage": None,
                }, "missing_gold_dir"
            gold_dir = _resolve_project_path(str(gold_dir))
            if not os.path.isdir(gold_dir):
                return {
                    "correct": False,
                    "file_sha_match": None,
                    "delta_coverage": None,
                }, f"gold_dir_missing:{gold_dir}"
            gold_snapshot = _snapshot_dir(gold_dir)
            init_fs_state = _capture_fs_state(work_dir)
            gold_fs_state = _capture_fs_state(gold_dir)
        final_output_text = ""
        for _step_idx, obj in traj_steps:
            raw_output = str(obj.get("output", "") or "")
            action = project_action(raw_output, enable_commit=enable_commit)
            if action.startswith(FORMAT_VIOLATION_PREFIX):
                final_output_text = action
                continue
            if action.startswith(ANSWER_PREFIX):
                final_output_text = action[len(ANSWER_PREFIX) :].strip()
                continue
            result = run_in_unshare_sandbox(
                work_dir=work_dir,
                command=action,
                timeout_s=10,
                extra_env={"PATH": "/usr/bin:/bin"},
            )
            final_output_text = (result.stdout or "") + (result.stderr or "")
        file_sha_match = None
        delta_coverage = None
        if spec.get("type") in {"files", "hybrid"}:
            file_sha_match = _is_correct_files(work_dir, gold_fs_state)
            if init_fs_state is None or gold_fs_state is None:
                raise RuntimeError(
                    "File-side evaluation requires initialized fs states."
                )
            delta_coverage = _delta_coverage_score(
                work_dir, init_fs_state, gold_fs_state
            )
        return {
            "correct": _check_task_completion(
                final_output_text, spec, work_dir, gold_fs_state
            ),
            "file_sha_match": file_sha_match,
            "delta_coverage": delta_coverage,
        }, None
    except SandboxError as e:
        return {
            "correct": False,
            "file_sha_match": None,
            "delta_coverage": None,
        }, f"SandboxError: {e}"
    finally:
        shutil.rmtree(work_dir)

def _resolve_eval_workers(requested_workers: int) -> int:
    if requested_workers > 0:
        return requested_workers
    cpu_count = os.cpu_count() or 1
    return min(32, max(1, cpu_count))

def _evaluate_single_trajectory(task: Dict[str, Any]) -> Dict[str, Any]:
    traj_steps = task["traj_steps"]
    sample_env_kwargs = task["sample_env_kwargs"]
    enable_commit = bool(task["enable_commit"])
    spec = task["spec"]
    spec_type = spec.get("type", None)
    traj_uid = task["traj_uid"]
    dataset_key = task["dataset_key"]
    sid = task["sid"]
    out = str(traj_steps[-1][1].get("output", "") or "")
    ans = extract_answer(out, enable_commit=enable_commit)
    replay_metrics = {"correct": False, "file_sha_match": None, "delta_coverage": None}
    try:
        replay_metrics, replay_error = _replay_trajectory(
            traj_steps,
            sample_env_kwargs,
            enable_commit=enable_commit,
        )
    except Exception as e:
        replay_metrics = {
            "correct": False,
            "file_sha_match": None,
            "delta_coverage": None,
        }
        replay_error = str(e)
    answer_out = _extract_answer_payload(ans) if ans is not None else None
    return {
        "traj_uid": traj_uid,
        "dataset": dataset_key,
        "sid": sid,
        "spec": spec,
        "spec_type": spec_type,
        "answer": answer_out,
        "replay_metrics": replay_metrics,
        "replay_error": replay_error,
        "file_sha_match": replay_metrics.get("file_sha_match", None),
        "delta_coverage": replay_metrics.get("delta_coverage", None),
    }

def _default_metrics_output_path(results_jsonl: str) -> str:
    return results_jsonl + ".with_metrics.jsonl"

def _default_summary_output_path(results_jsonl: str) -> str:
    return results_jsonl + ".eval_summary.json"

def _append_dataset_to_path(path_str: str, dataset_name: str) -> str:
    path = Path(path_str)
    return str(path.with_name(f"{path.stem}.{dataset_name}{path.suffix}"))

def _init_summary_stats() -> Dict[str, Any]:
    return {
        "total_trajectories": 0,
        "string_total": 0,
        "string_exact_correct": 0.0,
        "string_correct": 0.0,
        "files_total": 0,
        "files_correct": 0.0,
        "hybrid_total": 0,
        "hybrid_correct": 0.0,
        "files_sha_match": 0,
        "hybrid_sha_match": 0,
        "files_delta_coverage_sum": 0.0,
        "hybrid_delta_coverage_sum": 0.0,
        "missing_gt": 0,
        "replay_fail": 0,
        "per_match_exact": defaultdict(lambda: {"n": 0, "score_sum": 0.0}),
        "per_match": defaultdict(lambda: {"n": 0, "score_sum": 0.0}),
    }

def _build_pass_at_k_summary(
    by_sample_rollouts: Dict[Any, Dict[str, Any]],
    max_k: Optional[int] = None,
) -> tuple[
    Dict[str, Dict[str, float | int]], Dict[str, Dict[str, Dict[str, float | int]]]
]:
    pass_at_k_summary: Dict[str, Dict[str, float | int]] = {}
    pass_at_k_by_type: Dict[str, Dict[str, Dict[str, float | int]]] = defaultdict(dict)
    max_group_n = max((item["n"] for item in by_sample_rollouts.values()), default=0)
    if max_k is not None:
        max_group_n = min(max_group_n, int(max_k))
    for k in range(1, max_group_n + 1):
        eligible = [item for item in by_sample_rollouts.values() if item["n"] >= k]
        if not eligible:
            continue
        pass_at_k_summary[f"pass@{k}"] = {
            "value": sum(
                pass_at_k(item["n"], item["binary_correct"], k) for item in eligible
            )
            / len(eligible),
            "eval_count": len(eligible),
        }
        grouped = defaultdict(list)
        for item in eligible:
            grouped[str(item["type"])].append(item)
        for spec_type, items in grouped.items():
            pass_at_k_by_type[spec_type][f"pass@{k}"] = {
                "value": sum(
                    pass_at_k(item["n"], item["binary_correct"], k) for item in items
                )
                / len(items),
                "eval_count": len(items),
            }
    return pass_at_k_summary, {
        k: dict(v) for k, v in sorted(pass_at_k_by_type.items(), key=lambda x: x[0])
    }

def _build_answer_rate_summary(
    by_sample_rollouts: Dict[Any, Dict[str, Any]],
) -> tuple[Dict[str, float | int], Dict[str, Dict[str, float | int]]]:
    answer_rate_summary: Dict[str, float | int] = {"value": 0.0, "eval_count": 0}
    answer_rate_by_type: Dict[str, Dict[str, float | int]] = {}
    items = list(by_sample_rollouts.values())
    if not items:
        return answer_rate_summary, answer_rate_by_type
    answer_rate_summary = {
        "value": sum(
            item["answered_count"] / item["n"] for item in items if item["n"] > 0
        )
        / len(items),
        "eval_count": len(items),
    }
    grouped = defaultdict(list)
    for item in items:
        grouped[str(item["type"])].append(item)
    for spec_type, grouped_items in grouped.items():
        answer_rate_by_type[spec_type] = {
            "value": sum(
                item["answered_count"] / item["n"]
                for item in grouped_items
                if item["n"] > 0
            )
            / len(grouped_items),
            "eval_count": len(grouped_items),
        }
    return answer_rate_summary, dict(
        sorted(answer_rate_by_type.items(), key=lambda x: x[0])
    )

def _build_summary_payload(
    results_jsonl: str,
    parquet: str,
    resolved_parquets: Dict[str, str],
    dataset_name: str,
    stats: Dict[str, Any],
    pass_at_k_summary: Dict[str, Dict[str, float | int]],
    pass_at_k_by_type: Dict[str, Dict[str, Dict[str, float | int]]],
    answer_rate_summary: Dict[str, float | int],
    answer_rate_by_type: Dict[str, Dict[str, float | int]],
) -> Dict[str, Any]:
    return {
        "dataset": dataset_name,
        "results_jsonl": results_jsonl,
        "parquet": parquet,
        "resolved_parquets": resolved_parquets,
        "total_trajectories": stats["total_trajectories"],
        "string_exact_accuracy": safe_ratio(
            stats["string_exact_correct"], stats["string_total"]
        ),
        "string_exact_correct_count": stats["string_exact_correct"],
        "string_llm_incorporated_accuracy": safe_ratio(
            stats["string_correct"], stats["string_total"]
        ),
        "string_llm_incorporated_correct_count": stats["string_correct"],
        "string_accuracy": safe_ratio(stats["string_correct"], stats["string_total"]),
        "string_correct_count": stats["string_correct"],
        "string_total_count": stats["string_total"],
        "files_accuracy": safe_ratio(stats["files_correct"], stats["files_total"]),
        "files_correct_count": stats["files_correct"],
        "files_total_count": stats["files_total"],
        "files_sha_match_rate": safe_ratio(
            stats["files_sha_match"], stats["files_total"]
        ),
        "files_sha_match_count": stats["files_sha_match"],
        "files_delta_coverage": safe_ratio(
            stats["files_delta_coverage_sum"], stats["files_total"]
        ),
        "hybrid_accuracy": safe_ratio(stats["hybrid_correct"], stats["hybrid_total"]),
        "hybrid_correct_count": stats["hybrid_correct"],
        "hybrid_total_count": stats["hybrid_total"],
        "hybrid_sha_match_rate": safe_ratio(
            stats["hybrid_sha_match"], stats["hybrid_total"]
        ),
        "hybrid_sha_match_count": stats["hybrid_sha_match"],
        "hybrid_delta_coverage": safe_ratio(
            stats["hybrid_delta_coverage_sum"], stats["hybrid_total"]
        ),
        "missing_gt": stats["missing_gt"],
        "replay_fail": stats["replay_fail"],
        "pass_at_k": pass_at_k_summary,
        "pass_at_k_by_type": pass_at_k_by_type,
        "answered_rate": answer_rate_summary,
        "answered_rate_by_type": answer_rate_by_type,
        "per_match_exact": {
            m: {
                "accuracy": safe_ratio(d["score_sum"], d["n"]),
                "correct_count": d["score_sum"],
                "eval_count": d["n"],
            }
            for m, d in sorted(stats["per_match_exact"].items(), key=lambda x: x[0])
        },
        "per_match_llm_incorporated": {
            m: {
                "accuracy": safe_ratio(d["score_sum"], d["n"]),
                "correct_count": d["score_sum"],
                "eval_count": d["n"],
            }
            for m, d in sorted(stats["per_match"].items(), key=lambda x: x[0])
        },
        "per_match": {
            m: {
                "accuracy": safe_ratio(d["score_sum"], d["n"]),
                "correct_count": d["score_sum"],
                "eval_count": d["n"],
            }
            for m, d in sorted(stats["per_match"].items(), key=lambda x: x[0])
        },
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_jsonl", required=True)
    ap.add_argument("--parquet", default="")
    ap.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("BASH_CODING_EVAL_WORKERS", "32")),
    )
    ap.add_argument(
        "--output_jsonl",
        default="",
        help="Optional. If set, write a new jsonl with appended metrics field.",
    )
    ap.add_argument(
        "--output_summary_json",
        default="",
        help="Optional. If set, write the printed summary metrics as json.",
    )
    args = ap.parse_args()
    gt_map, resolved_parquets, default_dataset_name = build_gt_map(
        args.results_jsonl, args.parquet.strip()
    )
    configured_val_rollout_n = _resolve_val_rollout_n(args.results_jsonl)
    enable_commit = _resolve_enable_commit(args.results_jsonl)
    by_traj = _group_steps_by_traj(args.results_jsonl)
    if len(by_traj) == 0:
        raise SystemExit("No trajectories found in results_jsonl (missing traj_uid?)")
    overall_stats = _init_summary_stats()
    per_dataset_stats: Dict[str, Dict[str, Any]] = defaultdict(_init_summary_stats)
    metrics_by_traj: Dict[str, Dict[str, Any]] = {}
    by_sample_rollouts: Dict[Tuple[str, str], Dict[str, Any]] = {}
    by_dataset_sample_rollouts: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    eval_tasks: List[Dict[str, Any]] = []
    traj_records: List[Dict[str, Any]] = []
    judge_candidates: List[Dict[str, Any]] = []
    for _traj, traj_steps in by_traj.items():
        _t, obj = traj_steps[-1]
        sid_full = str(obj.get("sample_id", ""))
        sid = sid_full.split("_idx")[0] if "_idx" in sid_full else sid_full
        dataset_name = str(obj.get("dataset", "") or "").strip() or default_dataset_name
        dataset_key = dataset_name or "__unknown__"
        overall_stats["total_trajectories"] += 1
        per_dataset_stats[dataset_key]["total_trajectories"] += 1
        out = str(obj.get("output", "") or "")
        ans = extract_answer(out, enable_commit=enable_commit)
        has_answer = ans is not None
        gt_info = gt_map.get((dataset_name, sid), None)
        if gt_info is None and default_dataset_name is not None:
            gt_info = gt_map.get((default_dataset_name, sid), None)
        if gt_info is None:
            gt_info = gt_map.get((None, sid), None)
        if not isinstance(gt_info, dict):
            overall_stats["missing_gt"] += 1
            per_dataset_stats[dataset_key]["missing_gt"] += 1
            metrics_by_traj[str(obj.get("traj_uid", ""))] = {
                "dataset": dataset_key,
                "acc": None,
                "answered": has_answer,
                "evaluated": False,
                "type": None,
                "match": None,
                "missing_gt": True,
            }
            continue
        spec = (
            gt_info.get("reward_spec", {})
            if isinstance(gt_info.get("reward_spec", {}), dict)
            else {}
        )
        sample_env_kwargs = (
            gt_info.get("env_kwargs", {})
            if isinstance(gt_info.get("env_kwargs", {}), dict)
            else {}
        )
        spec_type = spec.get("type", None)
        sample_key = (dataset_key, sid)
        if sample_key not in by_sample_rollouts:
            by_sample_rollouts[sample_key] = {
                "type": spec_type,
                "n": 0,
                "score_sum": 0.0,
                "binary_correct": 0,
                "answered_count": 0,
            }
        by_sample_rollouts[sample_key]["n"] += 1
        by_sample_rollouts[sample_key]["answered_count"] += int(has_answer)
        if sid not in by_dataset_sample_rollouts[dataset_key]:
            by_dataset_sample_rollouts[dataset_key][sid] = {
                "type": spec_type,
                "n": 0,
                "score_sum": 0.0,
                "binary_correct": 0,
                "answered_count": 0,
            }
        by_dataset_sample_rollouts[dataset_key][sid]["n"] += 1
        by_dataset_sample_rollouts[dataset_key][sid]["answered_count"] += int(
            has_answer
        )
        if spec_type == "string":
            overall_stats["string_total"] += 1
            per_dataset_stats[dataset_key]["string_total"] += 1
            m = spec.get("match", "exact")
            overall_stats["per_match_exact"][m]["n"] += 1
            per_dataset_stats[dataset_key]["per_match_exact"][m]["n"] += 1
            overall_stats["per_match"][m]["n"] += 1
            per_dataset_stats[dataset_key]["per_match"][m]["n"] += 1
        elif spec_type == "files":
            overall_stats["files_total"] += 1
            per_dataset_stats[dataset_key]["files_total"] += 1
        elif spec_type == "hybrid":
            overall_stats["hybrid_total"] += 1
            per_dataset_stats[dataset_key]["hybrid_total"] += 1
        eval_tasks.append(
            {
                "traj_uid": str(obj.get("traj_uid", "")),
                "traj_steps": traj_steps,
                "sample_env_kwargs": sample_env_kwargs,
                "enable_commit": enable_commit,
                "spec": spec,
                "dataset_key": dataset_key,
                "sid": sid,
            }
        )
    eval_workers = _resolve_eval_workers(args.workers)
    task_count = len(eval_tasks)
    effective_workers = min(eval_workers, task_count) if task_count else 1
    if effective_workers > 1:
        chunksize = max(1, task_count // (effective_workers * 4))
        with ProcessPoolExecutor(max_workers=effective_workers) as executor:
            eval_results = list(
                executor.map(
                    _evaluate_single_trajectory, eval_tasks, chunksize=chunksize
                )
            )
    else:
        eval_results = [_evaluate_single_trajectory(task) for task in eval_tasks]
    for result in eval_results:
        traj_uid = result["traj_uid"]
        dataset_key = result["dataset"]
        sid = result["sid"]
        spec = result["spec"]
        spec_type = result["spec_type"]
        replay_metrics = result["replay_metrics"]
        replay_error = result["replay_error"]
        file_sha = result["file_sha_match"]
        delta_coverage = result["delta_coverage"]
        ans = result["answer"]
        if replay_error is not None:
            overall_stats["replay_fail"] += 1
            per_dataset_stats[dataset_key]["replay_fail"] += 1
        if spec_type == "files":
            overall_stats["files_sha_match"] += int(bool(file_sha))
            per_dataset_stats[dataset_key]["files_sha_match"] += int(bool(file_sha))
            if delta_coverage is not None:
                overall_stats["files_delta_coverage_sum"] += float(delta_coverage)
                per_dataset_stats[dataset_key]["files_delta_coverage_sum"] += float(
                    delta_coverage
                )
        elif spec_type == "hybrid":
            overall_stats["hybrid_sha_match"] += int(bool(file_sha))
            per_dataset_stats[dataset_key]["hybrid_sha_match"] += int(bool(file_sha))
            if delta_coverage is not None:
                overall_stats["hybrid_delta_coverage_sum"] += float(delta_coverage)
                per_dataset_stats[dataset_key]["hybrid_delta_coverage_sum"] += float(
                    delta_coverage
                )
        if ans is None:
            metrics_by_traj[traj_uid] = {
                "dataset": dataset_key,
                "acc": None,
                "answered": False,
                "evaluated": False,
                "type": spec_type,
                "match": spec.get("match", None),
                "missing_gt": False,
                "file_sha_match": file_sha,
                "delta_coverage": delta_coverage,
                "replay_error": replay_error,
                "llm_judge_used": False,
                "llm_judge_score": None,
            }
            continue
        exact_zero_needs_judge = (
            spec_type in {"string", "hybrid"} and string_score(ans, spec) == 0.0
        )
        traj_records.append(
            {
                "traj_uid": traj_uid,
                "dataset": dataset_key,
                "sid": sid,
                "spec": spec,
                "spec_type": spec_type,
                "answer": ans,
                "replay_metrics": replay_metrics,
                "replay_error": replay_error,
                "file_sha_match": file_sha,
                "delta_coverage": delta_coverage,
            }
        )
        if exact_zero_needs_judge:
            judge_candidates.append({"traj_uid": traj_uid, "answer": ans, "spec": spec})
    judge_scores = _judge_string_candidates(judge_candidates)
    for record in traj_records:
        traj_uid = record["traj_uid"]
        dataset_key = record["dataset"]
        sid = record["sid"]
        spec = record["spec"]
        spec_type = record["spec_type"]
        replay_metrics = record["replay_metrics"]
        replay_error = record["replay_error"]
        file_sha = record["file_sha_match"]
        delta_coverage = record["delta_coverage"]
        llm_judge_score = judge_scores.get(traj_uid, None)
        exact_string_acc = (
            string_score(record["answer"], spec)
            if spec_type in {"string", "hybrid"}
            else None
        )
        if spec_type == "string":
            acc = final_string_score(
                record["answer"], spec, llm_judge_score=llm_judge_score
            )
        elif spec_type == "files":
            acc = float(bool(replay_metrics.get("correct", False)))
        else:
            acc = final_string_score(
                record["answer"], spec, llm_judge_score=llm_judge_score
            ) * float(bool(file_sha))
        correct = acc == 1.0
        by_sample_rollouts[(dataset_key, sid)]["score_sum"] += float(acc)
        by_sample_rollouts[(dataset_key, sid)]["binary_correct"] += int(correct)
        by_dataset_sample_rollouts[dataset_key][sid]["score_sum"] += float(acc)
        by_dataset_sample_rollouts[dataset_key][sid]["binary_correct"] += int(correct)
        if spec_type == "string":
            m = spec.get("match", "exact")
            overall_stats["per_match_exact"][m]["score_sum"] += float(exact_string_acc)
            per_dataset_stats[dataset_key]["per_match_exact"][m]["score_sum"] += float(
                exact_string_acc
            )
            overall_stats["per_match"][m]["score_sum"] += float(acc)
            per_dataset_stats[dataset_key]["per_match"][m]["score_sum"] += float(acc)
            overall_stats["string_exact_correct"] += float(exact_string_acc)
            per_dataset_stats[dataset_key]["string_exact_correct"] += float(
                exact_string_acc
            )
            overall_stats["string_correct"] += float(acc)
            per_dataset_stats[dataset_key]["string_correct"] += float(acc)
        elif spec_type == "files":
            overall_stats["files_correct"] += float(acc)
            per_dataset_stats[dataset_key]["files_correct"] += float(acc)
        elif spec_type == "hybrid":
            overall_stats["hybrid_correct"] += float(acc)
            per_dataset_stats[dataset_key]["hybrid_correct"] += float(acc)
        metrics_by_traj[traj_uid] = {
            "dataset": dataset_key,
            "acc": float(acc),
            "string_exact_score": exact_string_acc,
            "string_llm_incorporated_score": float(acc)
            if spec_type == "string"
            else None,
            "answered": True,
            "evaluated": True,
            "type": spec_type,
            "match": spec.get("match", None),
            "missing_gt": False,
            "file_sha_match": file_sha,
            "delta_coverage": delta_coverage,
            "replay_error": replay_error,
            "llm_judge_used": llm_judge_score is not None
            and string_score(record["answer"], spec) == 0.0,
            "llm_judge_score": llm_judge_score,
        }
    pass_at_k_summary, pass_at_k_by_type = _build_pass_at_k_summary(
        by_sample_rollouts,
        max_k=configured_val_rollout_n,
    )
    answer_rate_summary, answer_rate_by_type = _build_answer_rate_summary(
        by_sample_rollouts
    )
    per_dataset_summary = {}
    for dataset_key in sorted(per_dataset_stats):
        dataset_resolved_parquets = {}
        if dataset_key in resolved_parquets:
            dataset_resolved_parquets[dataset_key] = resolved_parquets[dataset_key]
        dataset_pass_at_k_summary, dataset_pass_at_k_by_type = _build_pass_at_k_summary(
            by_dataset_sample_rollouts[dataset_key],
            max_k=configured_val_rollout_n,
        )
        dataset_answer_rate_summary, dataset_answer_rate_by_type = (
            _build_answer_rate_summary(by_dataset_sample_rollouts[dataset_key])
        )
        per_dataset_summary[dataset_key] = _build_summary_payload(
            results_jsonl=args.results_jsonl,
            parquet=dataset_resolved_parquets.get(dataset_key, ""),
            resolved_parquets=dataset_resolved_parquets,
            dataset_name=dataset_key,
            stats=per_dataset_stats[dataset_key],
            pass_at_k_summary=dataset_pass_at_k_summary,
            pass_at_k_by_type=dataset_pass_at_k_by_type,
            answer_rate_summary=dataset_answer_rate_summary,
            answer_rate_by_type=dataset_answer_rate_by_type,
        )
    summary = _build_summary_payload(
        results_jsonl=args.results_jsonl,
        parquet=args.parquet,
        resolved_parquets=resolved_parquets,
        dataset_name="__all__",
        stats=overall_stats,
        pass_at_k_summary=pass_at_k_summary,
        pass_at_k_by_type=pass_at_k_by_type,
        answer_rate_summary=answer_rate_summary,
        answer_rate_by_type=answer_rate_by_type,
    )
    summary["per_dataset"] = per_dataset_summary
    print("==== bash_coding offline eval ====")
    print(f"dataset:       {summary['dataset']}")
    print(f"results_jsonl: {summary['results_jsonl']}")
    print(f"eval_workers:  {effective_workers}")
    if summary["parquet"]:
        print(f"parquet:       {summary['parquet']}")
    else:
        print(f"resolved_parquets: {summary['resolved_parquets']}")
    print(f"total_trajectories: {summary['total_trajectories']}")
    print(
        f"string_exact_accuracy:       {summary['string_exact_accuracy']:.4f} "
        f"({summary['string_exact_correct_count']}/{summary['string_total_count']})"
    )
    print(
        f"string_llm_incorporated_accuracy: {summary['string_llm_incorporated_accuracy']:.4f} "
        f"({summary['string_llm_incorporated_correct_count']}/{summary['string_total_count']})"
    )
    print(
        f"string_accuracy:             {summary['string_accuracy']:.4f} "
        f"({summary['string_correct_count']}/{summary['string_total_count']})"
    )
    print(
        f"files_accuracy:              {summary['files_accuracy']:.4f} ({summary['files_correct_count']}/{summary['files_total_count']})"
    )
    print(
        f"files_sha_match_rate:        {summary['files_sha_match_rate']:.4f} "
        f"({summary['files_sha_match_count']}/{summary['files_total_count']})"
    )
    print(f"files_delta_coverage:        {summary['files_delta_coverage']:.4f}")
    print(
        f"hybrid_accuracy:             {summary['hybrid_accuracy']:.4f} "
        f"({summary['hybrid_correct_count']}/{summary['hybrid_total_count']})"
    )
    print(
        f"hybrid_sha_match_rate:       {summary['hybrid_sha_match_rate']:.4f} "
        f"({summary['hybrid_sha_match_count']}/{summary['hybrid_total_count']})"
    )
    print(
        f"answered_rate:               {summary['answered_rate']['value']:.4f} "
        f"(eval_count={summary['answered_rate']['eval_count']})"
    )
    print(f"hybrid_delta_coverage:       {summary['hybrid_delta_coverage']:.4f}")
    if summary["missing_gt"]:
        print(f"missing_gt: {summary['missing_gt']}")
    if summary["replay_fail"]:
        print(f"replay_fail: {summary['replay_fail']}")
    if pass_at_k_summary:
        print("\n-- pass@k --")
        for key, value in sorted(
            pass_at_k_summary.items(), key=lambda x: int(x[0].split("@", 1)[1])
        ):
            print(f"{key}: {value['value']:.4f} ({value['eval_count']})")
    print("\n-- per match exact (string tasks) --")
    for m, d in sorted(overall_stats["per_match_exact"].items(), key=lambda x: x[0]):
        n = d["n"]
        c = d["score_sum"]
        print(f"{m}: acc={safe_ratio(c, n):.4f} ({c}/{n})")
    print("\n-- per match llm incorporated (string tasks) --")
    for m, d in sorted(overall_stats["per_match"].items(), key=lambda x: x[0]):
        n = d["n"]
        c = d["score_sum"]
        print(f"{m}: acc={safe_ratio(c, n):.4f} ({c}/{n})")
    if per_dataset_summary:
        print("\n-- per dataset --")
        for dataset_key, dataset_metrics in per_dataset_summary.items():
            print(
                f"{dataset_key}: "
                f"string={dataset_metrics['string_accuracy']:.4f}, "
                f"files={dataset_metrics['files_accuracy']:.4f}, "
                f"hybrid={dataset_metrics['hybrid_accuracy']:.4f}, "
                f"missing_gt={dataset_metrics['missing_gt']}"
            )
    summary_path = args.output_summary_json.strip() or _default_summary_output_path(
        args.results_jsonl
    )
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"[written] eval_summary_json: {summary_path}")
    for dataset_key, dataset_metrics in per_dataset_summary.items():
        dataset_summary_path = _append_dataset_to_path(summary_path, dataset_key)
        with open(dataset_summary_path, "w", encoding="utf-8") as f:
            json.dump(dataset_metrics, f, ensure_ascii=False, indent=2)
            f.write("\n")
        print(
            f"[written] dataset_eval_summary_json[{dataset_key}]: {dataset_summary_path}"
        )
    if args.output_jsonl is not None:
        out_path = args.output_jsonl.strip() or _default_metrics_output_path(
            args.results_jsonl
        )
        final_step_idx = {traj: steps[-1][0] for traj, steps in by_traj.items()}
        dataset_output_paths = {
            dataset_key: _append_dataset_to_path(out_path, dataset_key)
            for dataset_key in per_dataset_summary
        }
        dataset_output_handles = {
            dataset_key: open(dataset_output_path, "w", encoding="utf-8")
            for dataset_key, dataset_output_path in dataset_output_paths.items()
        }
        with (
            open(args.results_jsonl, "r", encoding="utf-8") as f_in,
            open(out_path, "w", encoding="utf-8") as f_out,
        ):
            for line in f_in:
                if not line.strip():
                    continue
                obj = json.loads(line)
                traj = str(obj.get("traj_uid", ""))
                step_idx = _extract_sample_idx(str(obj.get("sample_id", "")))
                is_final = bool(step_idx == final_step_idx.get(traj, -1))
                dataset_key = (
                    str(obj.get("dataset", "") or "").strip()
                    or default_dataset_name
                    or "__unknown__"
                )
                if is_final:
                    metrics = metrics_by_traj.get(traj, None)
                    if metrics is not None:
                        obj["metrics"] = {
                            **metrics,
                            "final_step_idx": final_step_idx.get(traj, -1),
                            "is_final_step": True,
                        }
                else:
                    out = str(obj.get("output", "") or "")
                    ans = extract_answer(out, enable_commit=enable_commit)
                    obj["metrics"] = {
                        "dataset": dataset_key,
                        "acc": None,
                        "answered": ans is not None,
                        "evaluated": False,
                        "type": None,
                        "match": None,
                        "missing_gt": False,
                        "final_step_idx": final_step_idx.get(traj, -1),
                        "is_final_step": False,
                    }
                serialized = json.dumps(obj, ensure_ascii=False) + "\n"
                f_out.write(serialized)
                if dataset_key in dataset_output_handles:
                    dataset_output_handles[dataset_key].write(serialized)
        for dataset_key, handle in dataset_output_handles.items():
            handle.close()
        print(f"\n[written] results_jsonl_with_metrics: {out_path}")
        for dataset_key, dataset_output_path in dataset_output_paths.items():
            print(
                f"[written] dataset_results_jsonl_with_metrics[{dataset_key}]: {dataset_output_path}"
            )

if __name__ == "__main__":
    main()
