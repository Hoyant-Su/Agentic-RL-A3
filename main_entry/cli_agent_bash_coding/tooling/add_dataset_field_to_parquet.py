import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm

DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[2] / "data"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--overwrite", type=int, default=1)
    return parser.parse_args()

def parquet_files(data_root: Path) -> list[Path]:
    files: list[Path] = []
    for dataset_dir in sorted(
        path for path in data_root.iterdir() if path.is_dir() and path.name != "assets"
    ):
        files.extend(
            sorted(
                path
                for path in dataset_dir.iterdir()
                if path.is_file() and path.suffix == ".parquet"
            )
        )
    return files

def dataset_name_for(path: Path, data_root: Path) -> str:
    return path.relative_to(data_root).parts[0]

def update_parquet(path: Path, dataset_name: str, overwrite: bool) -> bool:
    df = pd.read_parquet(path)
    if "dataset" in df.columns and not overwrite:
        return False
    df["dataset"] = dataset_name
    df.to_parquet(path, index=False)
    return True

def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    files = parquet_files(data_root)
    updated = 0
    skipped = 0
    for path in tqdm(files, desc="Updating parquet dataset field"):
        dataset_name = dataset_name_for(path, data_root)
        changed = update_parquet(path, dataset_name, overwrite=bool(args.overwrite))
        if changed:
            updated += 1
            print(f"updated\t{path}\t{dataset_name}")
        else:
            skipped += 1
            print(f"skipped\t{path}\t{dataset_name}")
    print(f"done\tupdated={updated}\tskipped={skipped}")

if __name__ == "__main__":
    main()
