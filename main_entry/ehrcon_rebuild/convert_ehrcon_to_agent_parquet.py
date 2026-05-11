import argparse
import os
import pickle
import random
import re
import sqlite3
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from tqdm import tqdm


OUTPUT_COLUMNS = [
    "id",
    "query",
    "pre_files",
    "expected_text",
    "data_source",
    "prompt",
    "ability",
    "env_kwargs",
    "extra_info",
]

DATASET_NAME = "ehrcon_curated"
NOTE_TYPES = ["discharge", "nursing", "physician"]
TRAIN_RATIO = 0.8
SPLIT_SEED = 42
SUBSAMPLE_SEED = 42
SKIP_VALUE_COLUMNS = {
    "charttime",
    "storetime",
    "starttime",
    "endtime",
    "admittime",
    "dischtime",
    "dob",
    "hadm_id",
    "subject_id",
    "row_id",
}
QUERY_TEMPLATES = [
    "Check whether the following claim from a {note_type} note is consistent with the EHR database provided in `ehr.db`: \"{claim}\". Focus on the entity `{entity}`. Output only `consistent` or `inconsistent`.",
    "A {note_type} note contains the claim \"{claim}\" about `{entity}`. Verify it only against the provided `ehr.db` database and output only `consistent` or `inconsistent`.",
    "Use only the EHR database provided in `ehr.db` to determine whether this {note_type} note claim is supported: \"{claim}\". The target entity is `{entity}`. Output only `consistent` or `inconsistent`.",
    "Decide whether the provided `ehr.db` database agrees with this claim from a {note_type} note: \"{claim}\". Evaluate the entity `{entity}` and output only `consistent` or `inconsistent`.",
    "Verify the consistency of the `{entity}` claim \"{claim}\" from a {note_type} note using only the provided `ehr.db` database. Output only `consistent` or `inconsistent`.",
    "The provided `ehr.db` database and the {note_type} note should describe the same admission. Check whether the claim \"{claim}\" about `{entity}` is consistent. Output only `consistent` or `inconsistent`.",
    "Assess whether this statement from a {note_type} note matches the EHR data provided in `ehr.db`: \"{claim}\". Target entity: `{entity}`. Output only `consistent` or `inconsistent`.",
    "Determine if the EHR database provided in `ehr.db` supports the {note_type} note claim \"{claim}\" for `{entity}`. Output only `consistent` or `inconsistent`.",
    "For the entity `{entity}`, verify whether the following {note_type} note claim is consistent with the provided `ehr.db` database: \"{claim}\". Output only `consistent` or `inconsistent`.",
    "Read the claim \"{claim}\" from a {note_type} note and check it only against the provided `ehr.db` database. Focus on `{entity}`. Output only `consistent` or `inconsistent`.",
    "Judge whether this {note_type} note statement is consistent with the structured EHR record provided in `ehr.db`: \"{claim}\". Entity: `{entity}`. Output only `consistent` or `inconsistent`.",
    "Compare the provided `ehr.db` database with this {note_type} note claim: \"{claim}\". The claim is centered on `{entity}`. Output only `consistent` or `inconsistent`.",
    "Verify whether the structured EHR database provided in `ehr.db` agrees with the {note_type} note claim \"{claim}\" about `{entity}`. Output only `consistent` or `inconsistent`.",
    "The following claim appears in a {note_type} note: \"{claim}\". Check whether it is consistent with the provided `ehr.db` database for `{entity}`. Output only `consistent` or `inconsistent`.",
    "Decide if this claim from a {note_type} note is supported by the admission database provided in `ehr.db`: \"{claim}\". Target entity: `{entity}`. Output only `consistent` or `inconsistent`.",
    "Check the provided `ehr.db` database to see whether the {note_type} note claim \"{claim}\" is consistent. Focus on `{entity}`. Output only `consistent` or `inconsistent`.",
    "Does the EHR database provided in `ehr.db` support this {note_type} note claim: \"{claim}\"? Evaluate the entity `{entity}` and output only `consistent` or `inconsistent`.",
    "Using only the provided `ehr.db` database, verify the `{entity}` claim from a {note_type} note: \"{claim}\". Output only `consistent` or `inconsistent`.",
    "The claim \"{claim}\" is taken from a {note_type} note. Determine whether it is consistent with the provided `ehr.db` database for `{entity}`. Output only `consistent` or `inconsistent`.",
    "Test whether this {note_type} note statement matches the EHR database provided in `ehr.db`: \"{claim}\". Focus on `{entity}`. Output only `consistent` or `inconsistent`.",
    "Evaluate consistency between the provided `ehr.db` database and this {note_type} note claim: \"{claim}\". The target entity is `{entity}`. Output only `consistent` or `inconsistent`.",
    "For the same admission, compare the provided `ehr.db` database against this {note_type} note claim: \"{claim}\". Entity: `{entity}`. Output only `consistent` or `inconsistent`.",
    "Check if the following note claim is consistent with the database provided in `ehr.db`: \"{claim}\". It comes from a {note_type} note and concerns `{entity}`. Output only `consistent` or `inconsistent`.",
    "Verify this admission claim from a {note_type} note against the EHR database provided in `ehr.db`: \"{claim}\". Focus on `{entity}`. Output only `consistent` or `inconsistent`.",
    "Determine whether this structured-record check should pass using the provided `ehr.db` database for the {note_type} note claim \"{claim}\" about `{entity}`. Output only `consistent` or `inconsistent`.",
    "The EHR database provided in `ehr.db` is the reference. Decide whether the {note_type} note claim \"{claim}\" for `{entity}` is consistent. Output only `consistent` or `inconsistent`.",
    "Assess the claim \"{claim}\" from a {note_type} note by comparing it only with the database provided in `ehr.db`. Entity: `{entity}`. Output only `consistent` or `inconsistent`.",
    "Is the claim \"{claim}\" from the {note_type} note supported by the admission database provided in `ehr.db`? Focus on `{entity}`. Output only `consistent` or `inconsistent`.",
    "Use the database provided in `ehr.db` as ground truth and evaluate this {note_type} note claim: \"{claim}\". Target entity: `{entity}`. Output only `consistent` or `inconsistent`.",
    "For `{entity}`, decide whether the note claim \"{claim}\" is consistent with the EHR database provided in `ehr.db`. The claim comes from a {note_type} note. Output only `consistent` or `inconsistent`.",
    "Cross-check this {note_type} note claim with the provided `ehr.db` database: \"{claim}\". The relevant entity is `{entity}`. Output only `consistent` or `inconsistent`.",
    "The claim \"{claim}\" appears in a {note_type} note. Compare it with the database provided in `ehr.db` for the entity `{entity}` and output only `consistent` or `inconsistent`.",
    "Check consistency between the admission database provided in `ehr.db` and the {note_type} note claim \"{claim}\". Focus on `{entity}`. Output only `consistent` or `inconsistent`.",
    "Determine whether the database provided in `ehr.db` confirms this {note_type} note claim: \"{claim}\". The claim concerns `{entity}`. Output only `consistent` or `inconsistent`.",
    "Verify if the following claim is aligned with the EHR database provided in `ehr.db`: \"{claim}\". It is taken from a {note_type} note and targets `{entity}`. Output only `consistent` or `inconsistent`.",
    "This {note_type} note includes the claim \"{claim}\". Decide whether the database provided in `ehr.db` makes it consistent for `{entity}`. Output only `consistent` or `inconsistent`.",
    "Check whether the structured EHR data provided in `ehr.db` supports the claim \"{claim}\" from a {note_type} note. Entity: `{entity}`. Output only `consistent` or `inconsistent`.",
    "For the target entity `{entity}`, validate this claim from a {note_type} note against the database provided in `ehr.db`: \"{claim}\". Output only `consistent` or `inconsistent`.",
    "Use the admission database provided in `ehr.db` to judge the note claim \"{claim}\" from a {note_type} note. Focus on `{entity}`. Output only `consistent` or `inconsistent`.",
    "Determine if this {note_type} note claim remains consistent after checking the database provided in `ehr.db`: \"{claim}\". The target is `{entity}`. Output only `consistent` or `inconsistent`.",
    "The note says \"{claim}\" in a {note_type} note. Verify whether this is consistent with the database provided in `ehr.db` for `{entity}`. Output only `consistent` or `inconsistent`.",
    "Check whether the database provided in `ehr.db` and the {note_type} note agree on the claim \"{claim}\". Evaluate `{entity}` and output only `consistent` or `inconsistent`.",
    "Focus on `{entity}` and compare the {note_type} note claim \"{claim}\" with the EHR database provided in `ehr.db`. Output only `consistent` or `inconsistent`.",
    "The following claim should be validated using the admission database provided in `ehr.db`: \"{claim}\". It comes from a {note_type} note and concerns `{entity}`. Output only `consistent` or `inconsistent`.",
    "Decide whether the claim \"{claim}\" from the {note_type} note is consistent with the database provided in `ehr.db` for `{entity}`. Output only `consistent` or `inconsistent`.",
    "Validate this {note_type} note claim against the EHR database provided in `ehr.db`: \"{claim}\". The target entity is `{entity}`. Output only `consistent` or `inconsistent`.",
    "Use the EHR record provided in `ehr.db` to check whether the {note_type} note claim \"{claim}\" is consistent. Focus on `{entity}`. Output only `consistent` or `inconsistent`.",
    "Is this claim from a {note_type} note consistent with the database provided in `ehr.db`: \"{claim}\"? The target entity is `{entity}`. Output only `consistent` or `inconsistent`.",
    "Evaluate whether the database provided in `ehr.db` supports the claim \"{claim}\" extracted from a {note_type} note. Entity: `{entity}`. Output only `consistent` or `inconsistent`.",
    "Compare this note claim with the structured EHR data provided in `ehr.db`: \"{claim}\". It comes from a {note_type} note and targets `{entity}`. Output only `consistent` or `inconsistent`.",
]


def parse_args() -> argparse.Namespace:
    script_path = Path(__file__).resolve()
    parent = script_path.parents[1]
    if parent.name == "main_entry" and (parent / "data").is_dir():
        default_output_dir = parent / "data" / DATASET_NAME
    else:
        repo_root = script_path.parents[3]
        default_output_dir = repo_root / f"framework/verl-agent/shy_local/data/{DATASET_NAME}"

    parser = argparse.ArgumentParser()
    parser.add_argument("--ehrcon-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument("--note-variant", choices=["original", "processed"], default="original")
    parser.add_argument("--db-mode", choices=["hardlink", "copy"], default="hardlink")
    parser.add_argument("--context-window", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-per-hadm-note-type", type=int, default=0)
    parser.add_argument("--max-per-note-type", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.ehrcon_root is None:
        env = os.environ.get("EHRCON_ROOT")
        if env:
            args.ehrcon_root = Path(env).resolve()
        else:
            cand = script_path.parents[1]
            if (cand / "dataset").is_dir():
                args.ehrcon_root = cand
            else:
                parser.error("Set --ehrcon-root or EHRCON_ROOT (EHRCon root containing dataset/)")
    return args


def sanitize_entity(entity: str) -> str:
    token = re.sub(r"[^a-z0-9]+", "_", entity.lower()).strip("_")
    return token or "entity"


def resolve_note_csv(note_dir: Path, note_type: str, split_name: str) -> Path:
    suffix = "val" if split_name == "valid" else "test"
    return note_dir / f"{note_type}_{suffix}.csv"


def resolve_label_path(note_dir: Path, hadm_id: int) -> Path:
    matches = sorted(note_dir.rglob(f"EHRCon_{hadm_id}.0_data.pkl"))
    if len(matches) != 1:
        raise ValueError(f"Expected one PKL for HADM_ID={hadm_id}, found {len(matches)} under {note_dir}")
    return matches[0]


def normalize_expected_label(record: dict) -> str:
    label_text = str(record.get("label", "")).strip().lower()
    errors_text = str(record.get("errors", "")).strip().lower()
    if label_text == "consistency" or errors_text in {"nan", "0"}:
        return "consistent"
    return "inconsistent"


def resolve_payload_label(payload: dict) -> Optional[str]:
    data_records = payload.get("data", [])
    if not data_records:
        return None
    labels = {normalize_expected_label(record) for record in data_records}
    if len(labels) != 1:
        return None
    return next(iter(labels))


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_match(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def is_separator_line(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and all(char in "=#-.*_" for char in stripped)


def is_section_header(line: str) -> bool:
    stripped = line.strip()
    return stripped.endswith(":") or is_separator_line(stripped)


def extract_timestamp_prefix(line: str) -> Optional[str]:
    stripped = line.strip()
    match = re.match(r"^(\[\*\*[^\]]+\*\*\](?:\s+\d{1,2}:\d{2}(?:AM|PM)?)?)", stripped, flags=re.IGNORECASE)
    if not match:
        return None
    return normalize_space(match.group(1))


def contains_entity(line: str, entity: str) -> bool:
    return normalize_match(entity) in normalize_match(line)


def choose_anchor_line(lines: List[str], position: int, entity: str) -> int:
    target = max(0, min(len(lines) - 1, int(position) - 1))
    candidate_indices = list(range(max(0, target - 3), min(len(lines), target + 4)))
    scored = []
    for idx in candidate_indices:
        line = lines[idx]
        stripped = line.strip()
        score = -abs(idx - target)
        if contains_entity(line, entity):
            score += 100
        if stripped:
            score += 20
        if not is_section_header(stripped):
            score += 5
        if any(char.isdigit() for char in stripped):
            score += 3
        scored.append((score, idx))
    scored.sort(reverse=True)
    return scored[0][1]


def collect_target_values(payload: dict) -> List[str]:
    values = []
    seen = set()
    for record in payload.get("data", []):
        for table_name, table_payload in record.items():
            if table_name in {"label", "errors"} or not isinstance(table_payload, dict):
                continue
            for column_name, raw_value in table_payload.items():
                if column_name.lower() in SKIP_VALUE_COLUMNS:
                    continue
                value = normalize_space(str(raw_value))
                if not value or value.lower() in {"nan", "none", "chart"}:
                    continue
                normalized = normalize_match(value)
                if normalized and normalized not in seen:
                    seen.add(normalized)
                    values.append(value)
    return values


def normalize_label_phrase(label: str) -> str:
    parts = [part.strip() for part in label.split(",") if part.strip()]
    if len(parts) == 2 and len(parts[1].split()) <= 3:
        return f"{parts[1]} {parts[0]}"
    return label.strip()


def collect_reference_labels(payload: dict) -> List[str]:
    labels = []
    seen = set()
    for record in payload.get("data", []):
        for table_name, table_payload in record.items():
            if table_name in {"label", "errors"} or not isinstance(table_payload, dict):
                continue
            for column_name in ["label", "long_title", "short_title", "drug"]:
                if column_name not in table_payload:
                    continue
                label = normalize_label_phrase(normalize_space(str(table_payload[column_name])))
                normalized = normalize_match(label)
                if label and normalized and normalized not in seen:
                    seen.add(normalized)
                    labels.append(label)
    return labels


def get_primary_record(payload: dict) -> tuple[Optional[str], Optional[dict]]:
    for record in payload.get("data", []):
        for table_name, table_payload in record.items():
            if table_name in {"label", "errors"} or not isinstance(table_payload, dict):
                continue
            if not table_name.startswith("d_"):
                return table_name, table_payload
    return None, None


def get_numeric_or_text_value(table_payload: dict) -> Optional[str]:
    value_priority = [
        "valuenum",
        "value",
        "dose_val_rx",
        "amount",
        "rate",
        "org_name",
        "short_title",
        "long_title",
        "drug",
    ]
    for key in value_priority:
        if key not in table_payload:
            continue
        value = normalize_space(str(table_payload[key]))
        if value and value.lower() not in {"nan", "none", "chart", "n"}:
            return value
    return None


def get_unit_text(table_payload: dict) -> Optional[str]:
    unit_priority = ["valueuom", "dose_unit_rx", "amountuom", "rateuom", "form_unit_disp"]
    for key in unit_priority:
        if key not in table_payload:
            continue
        unit = normalize_space(str(table_payload[key]))
        if unit and unit.lower() not in {"nan", "none"}:
            return unit
    return None


def get_route_text(table_payload: dict) -> Optional[str]:
    for key in ["route", "originalroute"]:
        if key not in table_payload:
            continue
        route = normalize_space(str(table_payload[key]))
        if route and route.lower() not in {"nan", "none"}:
            return route
    return None


def extract_note_timestamp(note_evidence: str) -> Optional[str]:
    match = re.search(r"(\[\*\*[^\]]+\*\*\](?:\s+\d{1,2}:\d{2}(?:AM|PM)?)?)", note_evidence, flags=re.IGNORECASE)
    if match:
        timestamp = normalize_space(match.group(1))
        if re.search(r"\bNumeric Identifier\b", timestamp, flags=re.IGNORECASE):
            return None
        if not re.search(r"\[\*\*\d{4}-\d{1,2}-\d{1,2}\*\*\]", timestamp):
            return None
        return timestamp
    return None


def build_semantic_claim(entity: str, payload: dict, note_evidence: str, raw_claim: str, db_path: Path) -> Optional[str]:
    reference_labels = collect_reference_labels(payload)
    label = reference_labels[0] if reference_labels else entity.lower()
    primary_table, table_payload = get_primary_record(payload)
    if primary_table is None or table_payload is None:
        return raw_claim

    note_time = extract_note_timestamp(note_evidence)
    value = get_numeric_or_text_value(table_payload)
    unit = get_unit_text(table_payload)
    route = get_route_text(table_payload)
    candidate_count = count_matching_candidates(db_path, primary_table, label, value, unit, route)

    if primary_table in {"diagnoses", "procedures"}:
        prefix = "diagnosis" if primary_table == "diagnoses" else "procedure"
        return f"the {prefix} documented in the note is {label}"

    if primary_table == "microbiologyevents":
        spec_type = normalize_space(str(table_payload.get("spec_type_desc", "")))
        chartdate = normalize_space(str(table_payload.get("chartdate", "")))
        parts = [f"the microbiology finding is {value or label}"]
        if spec_type and spec_type.lower() not in {"nan", "none", "n"}:
            parts.append(f"from {spec_type.lower()}")
        if note_time:
            parts.append(f"at {note_time}")
        elif chartdate and chartdate.lower() not in {"nan", "none", "n"}:
            parts.append(f"on {chartdate}")
        return " ".join(parts)

    if primary_table == "prescriptions":
        if value is None:
            return None
        claim = f"the medication order for {label} is {value}"
        if unit:
            claim += f" {unit}"
        if route:
            claim += f" via {route}"
        if note_time:
            claim += f" at {note_time}"
        elif candidate_count > 1:
            return None
        return claim

    if primary_table in {"labevents", "chartevents", "inputevents_mv", "inputevents_cv", "outputevents"}:
        if value is None:
            return None
        claim = f"the note reports {label} = {value}"
        if unit:
            claim += f" {unit}"
        if note_time:
            claim += f" at {note_time}"
        elif candidate_count > 1:
            return None
        if route and primary_table.startswith("inputevents"):
            claim += f" via {route}"
        return claim

    if value:
        claim = f"the note reports {label} = {value}"
        if note_time:
            claim += f" at {note_time}"
        return claim
    return raw_claim


def normalize_sql_text(value: str) -> str:
    return normalize_space(value).lower()


def parse_numeric_text(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    text = normalize_space(str(value))
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    return float(match.group(0))


def count_matching_candidates(
    db_path: Path,
    primary_table: str,
    label: str,
    value: Optional[str],
    unit: Optional[str],
    route: Optional[str],
) -> int:
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    count = 0
    numeric_value = parse_numeric_text(value)

    if primary_table == "labevents":
        params = [normalize_sql_text(label)]
        query = """
            SELECT COUNT(*)
            FROM labevents AS l
            JOIN d_labitems AS d ON l.itemid = d.itemid
            WHERE lower(trim(d.label)) = ?
        """
        if value is not None:
            if numeric_value is not None:
                query += " AND l.valuenum = ?"
                params.append(numeric_value)
            else:
                query += " AND lower(trim(COALESCE(l.value, ''))) = ?"
                params.append(normalize_sql_text(value))
        cur.execute(query, params)
        count = cur.fetchone()[0]
    elif primary_table == "chartevents":
        params = [normalize_sql_text(label)]
        query = """
            SELECT COUNT(*)
            FROM chartevents AS c
            JOIN d_items AS d ON c.itemid = d.itemid
            WHERE lower(trim(d.label)) = ?
        """
        if value is not None:
            if numeric_value is not None:
                query += " AND CAST(c.value AS REAL) = ?"
                params.append(numeric_value)
            else:
                query += " AND lower(trim(COALESCE(c.value, ''))) = ?"
                params.append(normalize_sql_text(value))
        cur.execute(query, params)
        count = cur.fetchone()[0]
    elif primary_table == "prescriptions":
        params = [normalize_sql_text(label)]
        query = "SELECT COUNT(*) FROM prescriptions WHERE lower(trim(drug)) = ?"
        if value is not None:
            if numeric_value is not None:
                query += " AND dose_val_rx = ?"
                params.append(numeric_value)
            else:
                query += " AND lower(trim(CAST(dose_val_rx AS TEXT))) = ?"
                params.append(normalize_sql_text(value))
        if unit is not None:
            query += " AND lower(trim(dose_unit_rx)) = ?"
            params.append(normalize_sql_text(unit))
        if route is not None:
            query += " AND lower(trim(route)) = ?"
            params.append(normalize_sql_text(route))
        cur.execute(query, params)
        count = cur.fetchone()[0]
    elif primary_table == "diagnoses":
        cur.execute(
            """
            SELECT COUNT(*)
            FROM diagnoses AS x
            JOIN d_icd_diagnoses AS d ON x.icd9_code = d.icd9_code
            WHERE lower(trim(d.short_title)) = ? OR lower(trim(d.long_title)) = ?
            """,
            [normalize_sql_text(label), normalize_sql_text(label)],
        )
        count = cur.fetchone()[0]
    elif primary_table == "procedures":
        cur.execute(
            """
            SELECT COUNT(*)
            FROM procedures AS x
            JOIN d_icd_procedures AS d ON x.icd9_code = d.icd9_code
            WHERE lower(trim(d.short_title)) = ? OR lower(trim(d.long_title)) = ?
            """,
            [normalize_sql_text(label), normalize_sql_text(label)],
        )
        count = cur.fetchone()[0]
    else:
        count = 1

    conn.close()
    return int(count)


def has_specific_note_time(claim: str) -> bool:
    return bool(re.search(r"\bat\s+\[\*\*\d{4}-\d{1,2}-\d{1,2}\*\*\](?:\s+\d{1,2}:\d{2}(?:AM|PM)?)?", claim, flags=re.IGNORECASE))


def get_event_label_stats(cur: sqlite3.Cursor, label: str) -> tuple[int, int]:
    cur.execute(
        """
        SELECT COUNT(*), COUNT(DISTINCT COALESCE(CAST(l.valuenum AS TEXT), lower(trim(COALESCE(l.value, '')))))
        FROM labevents AS l
        JOIN d_labitems AS d ON l.itemid = d.itemid
        WHERE lower(trim(d.label)) = ?
        """,
        [label],
    )
    lab_total, lab_distinct = cur.fetchone()
    cur.execute(
        """
        SELECT COUNT(*), COUNT(DISTINCT lower(trim(COALESCE(c.value, ''))))
        FROM chartevents AS c
        JOIN d_items AS d ON c.itemid = d.itemid
        WHERE lower(trim(d.label)) = ?
        """,
        [label],
    )
    chart_total, chart_distinct = cur.fetchone()
    return int(lab_total + chart_total), int(lab_distinct + chart_distinct)


def is_semantically_ambiguous_claim(db_path: Path, claim: str) -> bool:
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    if claim.startswith("the medication order for ") and not has_specific_note_time(claim):
        match = re.match(
            r"the medication order for (.+?) is (.+?)(?: via (.+?))?(?: at .+)?$",
            claim,
            flags=re.IGNORECASE,
        )
        if not match:
            conn.close()
            return False
        drug = normalize_sql_text(match.group(1))
        cur.execute(
            """
            SELECT COUNT(*), COUNT(DISTINCT lower(trim(COALESCE(CAST(dose_val_rx AS TEXT), ''))) || '|' ||
                                       lower(trim(COALESCE(dose_unit_rx, ''))) || '|' ||
                                       lower(trim(COALESCE(route, ''))))
            FROM prescriptions
            WHERE lower(trim(drug)) = ?
            """,
            [drug],
        )
        total_count, distinct_count = cur.fetchone()
        conn.close()
        return total_count > 1 and distinct_count > 1

    if claim.startswith("the note reports ") and not has_specific_note_time(claim):
        match = re.match(r"the note reports (.+?) = (.+?)(?: at .+)?$", claim, flags=re.IGNORECASE)
        if not match:
            conn.close()
            return False
        label = normalize_sql_text(match.group(1))
        lab_total, lab_distinct = get_event_label_stats(cur, label)
        conn.close()
        return lab_total > 1 and lab_distinct > 1

    if claim.startswith("the note reports ") and has_specific_note_time(claim):
        match = re.match(r"the note reports (.+?) = (.+?)(?: at .+)?$", claim, flags=re.IGNORECASE)
        if not match:
            conn.close()
            return False
        label = normalize_sql_text(match.group(1))
        total_count, distinct_count = get_event_label_stats(cur, label)
        conn.close()
        return total_count >= 8 and distinct_count >= 3

    conn.close()
    return False


def parse_line_pairs(line: str) -> List[str]:
    tokens = line.strip().split()
    pairs = []
    idx = 0
    while idx < len(tokens):
        token = tokens[idx].strip(",;")
        if re.match(r"^\[\*\*.*\*\*\]$", token):
            idx += 1
            continue
        if re.match(r"^\d{1,2}:\d{2}(AM|PM)?$", token, flags=re.IGNORECASE):
            idx += 1
            continue
        if "-" in token and not token.startswith("-") and not token.endswith("-"):
            key, value = token.split("-", 1)
            if key and value:
                pairs.append(f"{key}-{value}")
                idx += 1
                continue
        if token.endswith(":") and len(token) > 1 and idx + 1 < len(tokens):
            value = tokens[idx + 1].strip(",;")
            if idx + 2 < len(tokens):
                value_next = tokens[idx + 2].strip(",;")
                if re.search(r"[\d%/]", value_next) and re.fullmatch(r"[A-Za-z]+", value):
                    value = f"{value} {value_next}"
                    idx += 1
            pairs.append(f"{token[:-1]}: {value}")
            idx += 2
            continue
        idx += 1
    return pairs


def match_score(candidate: str, entity: str, target_values: List[str]) -> int:
    score = 0
    normalized_candidate = normalize_match(candidate)
    normalized_entity = normalize_match(entity)
    if normalized_entity and normalized_entity in normalized_candidate:
        score += 10
    for value in target_values:
        normalized_value = normalize_match(value)
        if normalized_value and normalized_value in normalized_candidate:
            score += 8
    return score


def fallback_claim_from_value(line: str, target_values: List[str]) -> Optional[str]:
    best_claim = None
    best_length = None
    for value in target_values:
        match = re.search(re.escape(value), line, flags=re.IGNORECASE)
        if not match:
            continue
        start = max(line.rfind("  ", 0, match.start()), line.rfind("\t", 0, match.start()))
        start = 0 if start < 0 else start + 2
        prefix_match = re.search(r"([A-Za-z0-9\[\]\*\(\) /]+[-:]\s*)$", line[start:match.start()])
        if prefix_match:
            claim = normalize_space(prefix_match.group(1) + line[match.start():match.end()])
        else:
            claim = normalize_space(line[start:match.end()])
        if not claim:
            continue
        if not re.match(r"^[A-Za-z]", claim):
            continue
        claim_length = len(claim)
        if best_length is None or claim_length < best_length:
            best_claim = claim
            best_length = claim_length
    return best_claim


def extract_atomic_claim(note_text: str, entity: str, payload: dict, position: int, max_lines: int) -> Optional[str]:
    lines = note_text.splitlines()
    anchor = choose_anchor_line(lines, position, entity)
    target_values = collect_target_values(payload)
    best_claim = None
    best_score = 0
    start = max(0, anchor - max_lines)
    end = min(len(lines), anchor + max_lines + 1)

    for idx in range(start, end):
        line = lines[idx].strip()
        if not line or is_separator_line(line) or is_section_header(line):
            continue
        for candidate in parse_line_pairs(line):
            score = match_score(candidate, entity, target_values)
            if score > best_score:
                best_claim = candidate
                best_score = score
        if best_score >= 18:
            return best_claim

    for idx in range(start, end):
        line = lines[idx].strip()
        if not line or is_separator_line(line) or is_section_header(line):
            continue
        candidate = fallback_claim_from_value(line, target_values)
        if candidate is None:
            continue
        score = match_score(candidate, entity, target_values)
        if score > best_score:
            best_claim = candidate
            best_score = score

    if best_score > 0:
        return best_claim
    return None


def extract_note_evidence(note_text: str, entity: str, position: int, max_lines: int) -> str:
    lines = note_text.splitlines()
    anchor = choose_anchor_line(lines, position, entity)
    start = anchor
    while start > 0:
        prev = lines[start - 1].strip()
        curr = lines[start].strip()
        if not prev or is_separator_line(prev) or is_section_header(prev):
            break
        prev_prefix = extract_timestamp_prefix(prev)
        curr_prefix = extract_timestamp_prefix(curr)
        if prev_prefix and (curr_prefix == prev_prefix or curr_prefix is None):
            start -= 1
            continue
        if prev_prefix is None and curr_prefix is None and anchor - (start - 1) <= max_lines:
            start -= 1
            continue
        break

    end = anchor
    while end + 1 < len(lines):
        nxt = lines[end + 1].strip()
        curr = lines[end].strip()
        if not nxt or is_separator_line(nxt) or is_section_header(nxt):
            break
        nxt_prefix = extract_timestamp_prefix(nxt)
        curr_prefix = extract_timestamp_prefix(curr)
        if curr_prefix and (nxt_prefix == curr_prefix or nxt_prefix is None):
            end += 1
            continue
        if curr_prefix is None and nxt_prefix is None and (end + 1 - anchor) <= max_lines:
            end += 1
            continue
        break

    evidence_lines = [normalize_space(lines[idx]) for idx in range(start, end + 1) if lines[idx].strip()]
    return " ".join(evidence_lines)


def build_query(note_type: str, entity: str, semantic_claim: str, sample_id: str) -> str:
    template_idx = sum(ord(char) for char in sample_id) % len(QUERY_TEMPLATES)
    template = QUERY_TEMPLATES[template_idx]
    return template.format(note_type=note_type, entity=entity, claim=semantic_claim)


def write_db_file(source_db: Path, target_db: Path, db_mode: str) -> None:
    if db_mode == "hardlink":
        os.link(source_db, target_db)
        return
    shutil.copy2(source_db, target_db)


def materialize_init_dir(init_dir: Path, source_db: Path, db_mode: str) -> None:
    init_dir.mkdir(parents=True, exist_ok=True)
    write_db_file(source_db, init_dir / "ehr.db", db_mode)


def build_sample_row(
    *,
    sample_id: str,
    query: str,
    expected_text: str,
    init_dir: Path,
    index: int,
    extra_info: dict,
) -> dict:
    return {
        "id": sample_id,
        "query": query,
        "pre_files": [],
        "expected_text": expected_text,
        "data_source": "EHRCon",
        "prompt": [{"role": "user", "content": ""}],
        "ability": "database_retrieval",
        "env_kwargs": {
            "task": query,
            "index": index,
            "init_dir": str(init_dir.resolve()),
            "reward_spec": {
                "expected": expected_text,
                "ignore_case": True,
                "match": "exact",
                "threshold": None,
                "type": "string",
            },
        },
        "extra_info": extra_info,
    }


def convert_split(args: argparse.Namespace, split_name: str) -> List[dict]:
    ehrcon_root = args.ehrcon_root.resolve()
    dataset_root = ehrcon_root / "dataset" / args.note_variant / split_name
    database_root = ehrcon_root / "dataset" / "database"
    assets_root = args.output_dir.resolve() / "assets"

    rows = []
    sample_index = 0
    skipped_no_data = 0
    skipped_ambiguous = 0
    skipped_no_claim = 0
    skipped_non_unique = 0
    skipped_missing_annotation = 0

    for note_type in NOTE_TYPES:
        note_dir = dataset_root / note_type
        note_csv = resolve_note_csv(note_dir, note_type, split_name)
        frame = pd.read_csv(note_csv, low_memory=False)

        row_iterator = tqdm(
            frame.iterrows(),
            total=len(frame),
            desc=f"{split_name}:{note_type}",
        )
        for _, row in row_iterator:
            hadm_id = int(float(row["HADM_ID"]))
            row_id = int(row["ROW_ID"])
            note_text = str(row["TEXT"])
            label_path = resolve_label_path(note_dir, hadm_id)
            db_path = database_root / f"{hadm_id}.db"

            with label_path.open("rb") as handle:
                annotations = pickle.load(handle)

            if row_id not in annotations:
                skipped_missing_annotation += 1
                continue

            label_items = annotations[row_id]
            for entity_idx, item in enumerate(label_items):
                entity = next(iter(item))
                payload = item[entity]
                expected_text = resolve_payload_label(payload)
                if expected_text is None:
                    if payload.get("data"):
                        skipped_ambiguous += 1
                    else:
                        skipped_no_data += 1
                    continue

                position = int(payload["position"])
                entity_slug = sanitize_entity(entity)
                sample_id = f"{DATASET_NAME}_{split_name}_{note_type}_{hadm_id}_{row_id}_{entity_slug}_{entity_idx}"
                raw_claim = extract_atomic_claim(note_text, entity, payload, position, args.context_window)
                if raw_claim is None:
                    skipped_no_claim += 1
                    continue
                note_evidence = extract_note_evidence(note_text, entity, position, args.context_window)
                semantic_claim = build_semantic_claim(entity, payload, note_evidence, raw_claim, db_path)
                if semantic_claim is None:
                    skipped_non_unique += 1
                    continue
                if is_semantically_ambiguous_claim(db_path, semantic_claim):
                    skipped_non_unique += 1
                    continue
                query = build_query(note_type, entity, semantic_claim, sample_id)
                init_dir = assets_root / sample_id / "init"
                materialize_init_dir(init_dir, db_path, args.db_mode)
                extra_info = {
                    "id": sample_id,
                    "index": sample_index,
                    "split": split_name,
                    "hadm_id": hadm_id,
                    "row_id": row_id,
                    "note_type": note_type,
                    "entity": entity,
                    "position": position,
                    "claim": semantic_claim,
                    "raw_claim": raw_claim,
                    "note_evidence": note_evidence,
                }

                rows.append(
                    build_sample_row(
                        sample_id=sample_id,
                        query=query,
                        expected_text=expected_text,
                        init_dir=init_dir,
                        index=sample_index,
                        extra_info=extra_info,
                    )
                )
                sample_index += 1

                if args.limit and sample_index >= args.limit:
                    print(
                        f"[{split_name}] limit reached: kept={len(rows)} "
                        f"skipped_no_data={skipped_no_data} skipped_ambiguous={skipped_ambiguous} "
                        f"skipped_no_claim={skipped_no_claim} skipped_non_unique={skipped_non_unique} "
                        f"skipped_missing_annotation={skipped_missing_annotation}"
                    )
                    return rows

    print(
        f"[{split_name}] kept={len(rows)} "
        f"skipped_no_data={skipped_no_data} skipped_ambiguous={skipped_ambiguous} "
        f"skipped_no_claim={skipped_no_claim} skipped_non_unique={skipped_non_unique} "
        f"skipped_missing_annotation={skipped_missing_annotation}"
    )
    return rows


def random_split_rows(rows: List[dict]) -> Tuple[List[dict], List[dict]]:
    shuffled = list(rows)
    rng = random.Random(SPLIT_SEED)
    rng.shuffle(shuffled)
    train_size = int(len(shuffled) * TRAIN_RATIO)
    train_rows = shuffled[:train_size]
    test_rows = shuffled[train_size:]

    for idx, row in enumerate(train_rows):
        row["env_kwargs"]["index"] = idx
        row["extra_info"]["index"] = idx
        row["extra_info"]["split"] = "train"
    for idx, row in enumerate(test_rows):
        row["env_kwargs"]["index"] = idx
        row["extra_info"]["index"] = idx
        row["extra_info"]["split"] = "test"

    return train_rows, test_rows


def _group_key(row: dict) -> tuple[int, str]:
    extra = row["extra_info"]
    return int(extra["hadm_id"]), str(extra["note_type"])


def _row_priority(row: dict) -> tuple[str, str, str]:
    extra = row["extra_info"]
    return (
        str(extra.get("entity", "")),
        str(extra.get("claim", "")),
        str(row.get("id", "")),
    )


def subsample_rows(
    rows: List[dict],
    *,
    max_per_hadm_note_type: int,
    max_per_note_type: int,
) -> List[dict]:
    if max_per_hadm_note_type <= 0 and max_per_note_type <= 0:
        return rows

    rng = random.Random(SUBSAMPLE_SEED)
    grouped: dict[tuple[int, str], List[dict]] = {}
    for row in rows:
        grouped.setdefault(_group_key(row), []).append(row)

    selected: List[dict] = []
    for key in sorted(grouped):
        group_rows = list(grouped[key])
        rng.shuffle(group_rows)
        group_rows.sort(key=_row_priority)
        if max_per_hadm_note_type > 0:
            group_rows = group_rows[:max_per_hadm_note_type]
        selected.extend(group_rows)

    if max_per_note_type > 0:
        regrouped: dict[str, List[dict]] = {}
        for row in selected:
            note_type = str(row["extra_info"]["note_type"])
            regrouped.setdefault(note_type, []).append(row)
        final_rows: List[dict] = []
        for note_type in NOTE_TYPES:
            note_rows = list(regrouped.get(note_type, []))
            rng.shuffle(note_rows)
            note_rows.sort(key=lambda row: (int(row["extra_info"]["hadm_id"]),) + _row_priority(row))
            final_rows.extend(note_rows[:max_per_note_type])
        selected = final_rows

    selected.sort(key=lambda row: str(row["id"]))
    return selected


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()

    if output_dir.exists():
        if not args.overwrite:
            raise ValueError(f"Output directory already exists: {output_dir}")
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = convert_split(args, "valid") + convert_split(args, "test")
    all_rows = subsample_rows(
        all_rows,
        max_per_hadm_note_type=args.max_per_hadm_note_type,
        max_per_note_type=args.max_per_note_type,
    )
    train_rows, test_rows = random_split_rows(all_rows)

    train_frame = pd.DataFrame(train_rows, columns=OUTPUT_COLUMNS)
    test_frame = pd.DataFrame(test_rows, columns=OUTPUT_COLUMNS)

    train_frame.to_parquet(output_dir / "train.parquet", index=False)
    test_frame.to_parquet(output_dir / "test.parquet", index=False)

    print(f"Saved train split to {output_dir / 'train.parquet'}")
    print(f"Saved test split to {output_dir / 'test.parquet'}")
    print(f"Assets root: {output_dir / 'assets'}")
    print(f"Random split ratio: {TRAIN_RATIO:.1f}/{1 - TRAIN_RATIO:.1f}, seed={SPLIT_SEED}")
    print(
        "Subsample caps: "
        f"max_per_hadm_note_type={args.max_per_hadm_note_type}, "
        f"max_per_note_type={args.max_per_note_type}, seed={SUBSAMPLE_SEED}"
    )


if __name__ == "__main__":
    main()
