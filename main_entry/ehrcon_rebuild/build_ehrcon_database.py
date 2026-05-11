import os
import sqlite3
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm


EVENT_TABLES = {
    "CHARTEVENTS": "chartevents",
    "INPUTEVENTS_CV": "inputevents_cv",
    "INPUTEVENTS_MV": "inputevents_mv",
    "MICROBIOLOGYEVENTS": "microbiologyevents",
    "OUTPUTEVENTS": "outputevents",
    "LABEVENTS": "labevents",
    "PRESCRIPTIONS": "prescriptions",
    "PROCEDURES_ICD": "procedures",
    "DIAGNOSES_ICD": "diagnoses",
}

LOOKUP_TABLES = {
    "D_ITEMS": "d_items",
    "D_LABITEMS": "d_labitems",
    "D_ICD_PROCEDURES": "d_icd_procedures",
    "D_ICD_DIAGNOSES": "d_icd_diagnoses",
}

NOTE_FILE_NAMES = {
    "discharge_test.csv",
    "discharge_val.csv",
    "nursing_test.csv",
    "nursing_val.csv",
    "physician_test.csv",
    "physician_val.csv",
}

CHUNK_SIZE = 200000


def resolve_table_path(mimic_dir: Path, table_name: str) -> Path:
    csv_path = mimic_dir / f"{table_name}.csv"
    if csv_path.exists():
        return csv_path
    return mimic_dir / f"{table_name}.csv.gz"


def get_note_paths(dataset_dir: Path) -> list[Path]:
    return sorted(path for path in dataset_dir.rglob("*.csv") if path.name in NOTE_FILE_NAMES)


def collect_hadm_ids(note_paths: list[Path]) -> list[int]:
    hadm_ids: set[int] = set()
    for path in tqdm(note_paths, desc="Scanning note CSVs"):
        frame = pd.read_csv(path, usecols=["HADM_ID"], low_memory=False)
        values = pd.to_numeric(frame["HADM_ID"], errors="coerce").dropna().astype("int64")
        hadm_ids.update(values.tolist())
    return sorted(hadm_ids)


def read_event_table(path: Path, hadm_ids: set[int]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    empty_frame: pd.DataFrame | None = None
    for chunk in pd.read_csv(path, chunksize=CHUNK_SIZE, low_memory=False):
        if empty_frame is None:
            empty_frame = chunk.iloc[0:0].copy()
        hadm_series = pd.to_numeric(chunk["HADM_ID"], errors="coerce")
        filtered = chunk.loc[hadm_series.isin(hadm_ids)]
        if not filtered.empty:
            frames.append(filtered)
    if frames:
        return pd.concat(frames, ignore_index=True)
    return empty_frame


def read_lookup_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def normalize_table(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    for column in normalized.columns:
        name = column.lower()
        if "dose_val_rx" in name:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce").fillna(0)
            continue
        if "num" in name or "amount" in name or "rate" in name or "hadm_id" in name:
            continue
        normalized[column] = normalized[column].astype(str).str.lower()
    return normalized


def load_tables(mimic_dir: Path, hadm_ids: list[int]) -> dict[str, pd.DataFrame]:
    hadm_id_set = set(hadm_ids)
    tables: dict[str, pd.DataFrame] = {}
    for source_name, target_name in tqdm(EVENT_TABLES.items(), desc="Loading event tables"):
        tables[target_name] = read_event_table(resolve_table_path(mimic_dir, source_name), hadm_id_set)
    for source_name, target_name in tqdm(LOOKUP_TABLES.items(), desc="Loading lookup tables"):
        tables[target_name] = read_lookup_table(resolve_table_path(mimic_dir, source_name))
    return tables


def select_hadm_rows(frame: pd.DataFrame, hadm_id: int) -> pd.DataFrame:
    if "HADM_ID" not in frame.columns:
        return frame.reset_index(drop=True)
    hadm_series = pd.to_numeric(frame["HADM_ID"], errors="coerce")
    return frame.loc[hadm_series == hadm_id].reset_index(drop=True)


def write_database(output_dir: Path, hadm_id: int, tables: dict[str, pd.DataFrame]) -> None:
    database_path = output_dir / f"{hadm_id}.db"
    with sqlite3.connect(database_path) as connection:
        for table_name, frame in tables.items():
            admission_frame = select_hadm_rows(frame, hadm_id)
            normalize_table(admission_frame).to_sql(table_name, connection, if_exists="replace", index=False)


def resolve_ehrcon_root() -> Path:
    if len(sys.argv) > 2:
        return Path(sys.argv[2]).resolve()
    env = os.environ.get("EHRCON_ROOT")
    if env:
        return Path(env).resolve()
    auto = Path(__file__).resolve().parents[1]
    if (auto / "dataset").is_dir():
        return auto
    raise SystemExit(
        "EHRCon checkout: set EHRCON_ROOT, or pass it as argv[2] after MIMIC_DIR"
    )


def main() -> None:
    mimic_dir = Path(sys.argv[1]).resolve()
    repo_dir = resolve_ehrcon_root()
    dataset_dir = repo_dir / "dataset"
    output_dir = dataset_dir / "database"

    note_paths = get_note_paths(dataset_dir)
    hadm_ids = collect_hadm_ids(note_paths)
    label_paths = sorted(dataset_dir.rglob("*.pkl"))

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(note_paths)} note CSV files.")
    print(f"Found {len(label_paths)} label PKL files.")
    print(f"Building {len(hadm_ids)} admission databases into {output_dir}.")

    tables = load_tables(mimic_dir, hadm_ids)
    for hadm_id in tqdm(hadm_ids, desc="Building patient databases"):
        write_database(output_dir, hadm_id, tables)


if __name__ == "__main__":
    main()
