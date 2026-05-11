import argparse
import csv
import json
from pathlib import Path

EMPTY_VALUE = "/"
EXACT_FIELDS = [
    ("string_exact_accuracy", "string_total_count"),
    ("files_sha_match_rate", "files_total_count"),
    ("hybrid_sha_match_rate", "hybrid_total_count"),
]
DELTA_FIELDS = [
    ("string_llm_incorporated_accuracy", "string_total_count"),
    ("files_delta_coverage", "files_total_count"),
    ("hybrid_delta_coverage", "hybrid_total_count"),
]
ANSWER_RATE_FIELDS = [
    ("string", "string"),
    ("files", "files"),
    ("hybrid", "hybrid"),
]

def format_value(value):
    if isinstance(value, float):
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value)

def format_metric(data, value_key, count_key):
    count = data[count_key]
    if count == 0:
        return EMPTY_VALUE
    return format_value(data[value_key])

def format_pass_at_k(data, pass_key):
    metric = data["pass_at_k"].get(pass_key, None)
    if metric is None:
        return EMPTY_VALUE
    if metric["eval_count"] == 0:
        return EMPTY_VALUE
    return format_value(metric["value"])

def format_answer_rate(data, spec_type):
    metric = data.get("answered_rate_by_type", {}).get(spec_type, None)
    if metric is None:
        return EMPTY_VALUE
    if metric["eval_count"] == 0:
        return EMPTY_VALUE
    return format_value(metric["value"])

def get_dataset_size(data):
    metric = data["pass_at_k"].get("pass@1", None)
    if metric is None:
        return 0
    return metric["eval_count"]

def collect_pass_keys(summaries):
    keys = set()
    for item in summaries:
        keys.update(item.get("pass_at_k", {}).keys())
    return sorted(keys, key=lambda x: int(x.split("@", 1)[1]))

def infer_model_name(result_dir, datasets):
    prefix = result_dir.name.split("__")[0]
    dataset_tokens = [
        tuple(dataset.split("_")) for dataset in datasets if dataset != "__all__"
    ]
    tokens = prefix.split("_")
    best_end = 0
    reachable = {0}
    while reachable:
        next_reachable = set()
        for start in reachable:
            for dataset_token in dataset_tokens:
                end = start + len(dataset_token)
                if tuple(tokens[start:end]) == dataset_token:
                    next_reachable.add(end)
                    best_end = max(best_end, end)
        reachable = next_reachable
    return "_".join(tokens[best_end:])

def build_row(data, model_name, result_dir, pass_keys):
    row = [
        data["dataset"],
        model_name,
        *(format_pass_at_k(data, key) for key in pass_keys),
        *(
            format_metric(data, value_key, count_key)
            for value_key, count_key in EXACT_FIELDS
        ),
        *(
            format_metric(data, value_key, count_key)
            for value_key, count_key in DELTA_FIELDS
        ),
        *(format_answer_rate(data, spec_type) for spec_type, _ in ANSWER_RATE_FIELDS),
        str(Path(data.get("results_jsonl", result_dir)).parent),
        get_dataset_size(data),
    ]
    return row

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path. Default: <result_dir>/merged_summary.csv",
    )
    args = parser.parse_args()
    result_dir = args.result_dir.resolve()
    output_path = (
        args.output.resolve() if args.output else result_dir / "merged_summary.csv"
    )
    summary_files = sorted(
        path for path in result_dir.glob("*.json") if ".eval_summary" in path.name
    )
    summaries = [json.loads(path.read_text()) for path in summary_files]
    summaries.sort(key=lambda item: (item["dataset"] != "__all__", item["dataset"]))
    pass_keys = collect_pass_keys(summaries)
    model_name = infer_model_name(result_dir, [item["dataset"] for item in summaries])
    header_group = [
        "Dataset",
        "Model",
        *(["Metrics"] + [""] * (len(pass_keys) - 1) if pass_keys else []),
        "Exact Match",
        "",
        "",
        "Delta",
        "",
        "",
        "Answered Rate",
        "",
        "",
        "Result_dir",
        "Dataset Size",
    ]
    header_name = [
        "",
        "",
        *[f"Pass@{key.split('@', 1)[1]}" for key in pass_keys],
        "String",
        "Files",
        "Hybrid",
        "String",
        "Files",
        "Hybrid",
        "String",
        "Files",
        "Hybrid",
        "",
        "",
    ]
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header_group)
        writer.writerow(header_name)
        for summary in summaries:
            writer.writerow(build_row(summary, model_name, result_dir, pass_keys))
    print(output_path)

if __name__ == "__main__":
    main()
