from typing import Any, Dict, List
from datasets import Features, Value

TARGET_BENCHES = [
    "agentbench_os",
    "databench",
    "shellops",
    "ehrcon_curated",
    "agentbench_dbbench",
]
FILE_ENTRY_FIELDS = ("content", "path", "size_bytes", "type")
PROMPT_ENTRY_FIELDS = ("content", "role")
REWARD_SPEC_FIELDS = (
    "expected",
    "gold_dir",
    "ignore_case",
    "match",
    "success_reward",
    "threshold",
    "type",
)
EXTRA_INFO_FIELDS = (
    "id",
    "index",
    "split",
    "claim",
    "entity",
    "hadm_id",
    "note_evidence",
    "note_type",
    "position",
    "raw_claim",
    "row_id",
    "answer_md5",
    "official_sql",
    "source",
    "table_name",
    "task_type",
)
CANONICAL_FEATURES = Features(
    {
        "id": Value("string"),
        "query": Value("string"),
        "gt_bash": Value("string"),
        "pre_files": [
            {
                "content": Value("string"),
                "path": Value("string"),
                "size_bytes": Value("float64"),
                "type": Value("string"),
            }
        ],
        "post_files": [
            {
                "content": Value("string"),
                "path": Value("string"),
                "size_bytes": Value("float64"),
                "type": Value("string"),
            }
        ],
        "expected_text": Value("string"),
        "task_type": Value("string"),
        "data_source": Value("string"),
        "prompt": [{"content": Value("string"), "role": Value("string")}],
        "ability": Value("string"),
        "env_kwargs": {
            "index": Value("int64"),
            "init_dir": Value("string"),
            "reward_spec": {
                "expected": Value("string"),
                "gold_dir": Value("string"),
                "ignore_case": Value("bool"),
                "match": Value("string"),
                "success_reward": Value("float64"),
                "threshold": Value("float64"),
                "type": Value("string"),
            },
            "task": Value("string"),
        },
        "extra_info": {
            "id": Value("string"),
            "index": Value("int64"),
            "split": Value("string"),
            "claim": Value("string"),
            "entity": Value("string"),
            "hadm_id": Value("int64"),
            "note_evidence": Value("string"),
            "note_type": Value("string"),
            "position": Value("int64"),
            "raw_claim": Value("string"),
            "row_id": Value("int64"),
            "answer_md5": Value("string"),
            "official_sql": Value("string"),
            "source": Value("string"),
            "table_name": Value("string"),
            "task_type": Value("string"),
        },
        "dataset": Value("string"),
    }
)

def _normalize_file_entries(entries: Any) -> List[Dict[str, Any]]:
    if not entries:
        return []
    normalized = []
    for entry in entries:
        if entry is None:
            continue
        normalized.append({field: entry.get(field) for field in FILE_ENTRY_FIELDS})
    return normalized

def _normalize_prompt_entries(entries: Any) -> List[Dict[str, Any]]:
    if not entries:
        return []
    normalized = []
    for entry in entries:
        if entry is None:
            continue
        normalized.append({field: entry.get(field) for field in PROMPT_ENTRY_FIELDS})
    return normalized

def _normalize_reward_spec(spec: Any) -> Dict[str, Any]:
    spec = spec or {}
    return {
        "expected": spec.get("expected"),
        "gold_dir": spec.get("gold_dir"),
        "ignore_case": spec["ignore_case"] if "ignore_case" in spec else True,
        "match": spec["match"] if "match" in spec else "exact",
        "success_reward": spec["success_reward"] if "success_reward" in spec else 1.0,
        "threshold": spec.get("threshold"),
        "type": spec.get("type"),
    }

def _normalize_env_kwargs(env_kwargs: Any) -> Dict[str, Any]:
    env_kwargs = env_kwargs or {}
    return {
        "index": env_kwargs.get("index"),
        "init_dir": env_kwargs.get("init_dir"),
        "reward_spec": _normalize_reward_spec(env_kwargs.get("reward_spec")),
        "task": env_kwargs.get("task"),
    }

def _normalize_extra_info(example: Dict[str, Any]) -> Dict[str, Any]:
    extra_info = example.get("extra_info") or {}
    env_kwargs = example.get("env_kwargs") or {}
    return {
        "id": extra_info.get("id", example.get("id")),
        "index": extra_info["index"]
        if "index" in extra_info
        else env_kwargs.get("index"),
        "split": extra_info.get("split"),
        "claim": extra_info.get("claim"),
        "entity": extra_info.get("entity"),
        "hadm_id": extra_info.get("hadm_id"),
        "note_evidence": extra_info.get("note_evidence"),
        "note_type": extra_info.get("note_type"),
        "position": extra_info.get("position"),
        "raw_claim": extra_info.get("raw_claim"),
        "row_id": extra_info.get("row_id"),
        "answer_md5": extra_info.get("answer_md5"),
        "official_sql": extra_info.get("official_sql"),
        "source": extra_info.get("source"),
        "table_name": extra_info.get("table_name"),
        "task_type": extra_info.get("task_type"),
    }

def normalize_example(example: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": example.get("id"),
        "query": example.get("query"),
        "gt_bash": example.get("gt_bash"),
        "pre_files": _normalize_file_entries(example.get("pre_files")),
        "post_files": _normalize_file_entries(example.get("post_files")),
        "expected_text": example.get("expected_text"),
        "task_type": example.get("task_type"),
        "data_source": example.get("data_source"),
        "prompt": _normalize_prompt_entries(example.get("prompt")),
        "ability": example.get("ability"),
        "env_kwargs": _normalize_env_kwargs(example.get("env_kwargs")),
        "extra_info": _normalize_extra_info(example),
        "dataset": example.get("dataset"),
    }
