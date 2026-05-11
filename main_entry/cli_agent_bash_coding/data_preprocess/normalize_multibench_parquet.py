import argparse
import os
from pathlib import Path
from datasets import Dataset, concatenate_datasets
from main_entry.cli_agent_bash_coding.data_preprocess.multibench_schema import (
    CANONICAL_FEATURES,
    TARGET_BENCHES,
    normalize_example,
)

def normalize_parquet(parquet_path: str, dataset_name: str) -> int:
    dataset = Dataset.from_parquet(parquet_path)
    normalized_rows = []
    for idx in range(len(dataset)):
        row = normalize_example(dataset[idx])
        row["dataset"] = dataset_name
        normalized_rows.append(row)
    normalized_dataset = Dataset.from_list(normalized_rows, features=CANONICAL_FEATURES)
    tmp_path = f"{parquet_path}.tmp"
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    normalized_dataset.to_parquet(tmp_path)
    os.replace(tmp_path, parquet_path)
    return len(normalized_dataset)

def verify_concatenation(data_root: str, benches: list[str]) -> None:
    train_sets = [
        Dataset.from_parquet(os.path.join(data_root, bench, "train.parquet"))
        for bench in benches
    ]
    test_sets = [
        Dataset.from_parquet(os.path.join(data_root, bench, "test.parquet"))
        for bench in benches
    ]
    concatenate_datasets(train_sets)
    concatenate_datasets(test_sets)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "data"),
    )
    parser.add_argument("--benches", nargs="*", default=TARGET_BENCHES)
    parser.add_argument("--splits", nargs="*", default=["train", "test"])
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    for bench in args.benches:
        for split in args.splits:
            parquet_path = os.path.join(args.data_root, bench, f"{split}.parquet")
            count = normalize_parquet(parquet_path, dataset_name=bench)
            print(f"normalized {bench}/{split}: {count}")
    if "train" in args.splits:
        train_sets = [
            Dataset.from_parquet(os.path.join(args.data_root, bench, "train.parquet"))
            for bench in args.benches
        ]
        concatenate_datasets(train_sets)
    if "test" in args.splits:
        test_sets = [
            Dataset.from_parquet(os.path.join(args.data_root, bench, "test.parquet"))
            for bench in args.benches
        ]
        concatenate_datasets(test_sets)
    print("concatenation check passed")

if __name__ == "__main__":
    main()
