#!/usr/bin/env python3

from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EVAL_PY = Path(__file__).resolve().parent / "eval_results_jsonl.py"
MERGE_PY = Path(__file__).resolve().parent / "merge_results.py"
DATA_ROOT_DEFAULT = PROJECT_ROOT / "main_entry/data"
BENCHMARKS = [
    "agentbench_os",
    "databench",
    "shellops",
    "ehrcon_curated",
    "agentbench_dbbench",
    "tablebench",
]

def _extract_sample_idx(sample_id: str) -> int:
    sid = str(sample_id or "")
    return int(sid.rsplit("_idx_", 1)[-1])

def load_allowed_ids_by_dataset(data_root: Path) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for name in BENCHMARKS:
        path = data_root / name / "test.parquet"
        df = pd.read_parquet(path)
        out[name] = set(df["id"].astype(str))
    return out

def sid_from_sample_id(sample_id: str) -> str:
    sid_full = str(sample_id or "")
    return sid_full.split("_idx")[0] if "_idx" in sid_full else sid_full

def cleanup_eval_artifacts(result_dir: Path, base_jsonl_name: str) -> None:
    prefix = base_jsonl_name + "."
    for p in result_dir.iterdir():
        if not p.is_file():
            continue
        if not p.name.startswith(prefix):
            continue
        if ".eval_summary" in p.name or ".with_metrics" in p.name:
            p.unlink()

def filter_rollout_jsonl(
    jsonl_path: Path,
    allowed: dict[str, set[str]],
) -> tuple[int, int, int]:
    lines_in = jsonl_path.read_text(encoding="utf-8").splitlines()
    by_traj: dict[str, list[dict]] = {}
    for line in lines_in:
        raw = line.strip()
        if not raw:
            continue
        obj = json.loads(raw)
        traj = str(obj.get("traj_uid", ""))
        by_traj.setdefault(traj, []).append(obj)
    bad_trajs: set[str] = set()
    for traj, steps in by_traj.items():
        steps.sort(key=lambda o: _extract_sample_idx(str(o.get("sample_id", ""))))
        last = steps[-1]
        ds = str(last.get("dataset", "") or "").strip()
        sid = sid_from_sample_id(str(last.get("sample_id", "")))
        if ds in allowed and sid not in allowed[ds]:
            bad_trajs.add(traj)
    kept_lines = 0
    removed_lines = 0
    out_lines: list[str] = []
    for line in lines_in:
        raw = line.strip()
        if not raw:
            continue
        obj = json.loads(raw)
        traj = str(obj.get("traj_uid", ""))
        if traj in bad_trajs:
            removed_lines += 1
            continue
        out_lines.append(json.dumps(obj, ensure_ascii=False))
        kept_lines += 1
    text = "\n".join(out_lines) + ("\n" if out_lines else "")
    jsonl_path.write_text(text, encoding="utf-8")
    return kept_lines, removed_lines, len(bad_trajs)

def is_primary_rollout_jsonl(p: Path) -> bool:
    if p.suffix != ".jsonl":
        return False
    n = p.name
    if "with_metrics" in n:
        return False
    return True

def discover_rollout_jsonls(result_dir: Path) -> list[Path]:
    return sorted(p for p in result_dir.glob("*.jsonl") if is_primary_rollout_jsonl(p))

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Filter rollout JSONLs to ids in test.parquet, then eval + merge_results.",
    )
    ap.add_argument(
        "result_dir",
        type=Path,
        help="Directory containing rollout *.jsonl (e.g. checkpoint step folder).",
    )
    ap.add_argument(
        "--data-root",
        type=Path,
        default=DATA_ROOT_DEFAULT,
        help="Root containing <bench>/test.parquet.",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("EVAL_WORKERS", "32")),
    )
    ap.add_argument(
        "--jsonl",
        type=Path,
        default=None,
        help="Only process this rollout jsonl (basename under result_dir or absolute path).",
    )
    args = ap.parse_args()
    result_dir = args.result_dir.resolve()
    data_root = args.data_root.resolve()
    allowed = load_allowed_ids_by_dataset(data_root)
    if args.jsonl:
        jpath = args.jsonl if args.jsonl.is_absolute() else (result_dir / args.jsonl)
        jpath = jpath.resolve()
        to_run = [jpath]
    else:
        to_run = discover_rollout_jsonls(result_dir)
        if not to_run:
            raise SystemExit(f"No primary *.jsonl under {result_dir}")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    for jsonl_path in to_run:
        if not jsonl_path.is_file():
            raise SystemExit(f"Missing jsonl: {jsonl_path}")
        k, r, tr = filter_rollout_jsonl(jsonl_path, allowed)
        cleanup_eval_artifacts(result_dir, jsonl_path.name)
        print(
            f"{jsonl_path.name}: lines_kept={k} lines_removed={r} trajs_removed={tr}",
            flush=True,
        )
        subprocess.run(
            [
                sys.executable,
                str(EVAL_PY),
                "--results_jsonl",
                str(jsonl_path),
                "--workers",
                str(args.workers),
            ],
            cwd=str(PROJECT_ROOT),
            check=True,
            env=env,
        )
    subprocess.run(
        [sys.executable, str(MERGE_PY), str(result_dir)],
        cwd=str(PROJECT_ROOT),
        check=True,
        env=env,
    )

if __name__ == "__main__":
    main()
