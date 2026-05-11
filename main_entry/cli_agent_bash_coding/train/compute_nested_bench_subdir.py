#!/usr/bin/env python3

import os
import re
import sys
from typing import Optional
from omegaconf import OmegaConf

def find_timestamp_dir(path: str) -> Optional[str]:
    path = os.path.abspath(os.path.expanduser(path))
    while path not in ("/", ""):
        base = os.path.basename(path)
        if re.match(r"^[0-9]{8}_[0-9]{6}$", base):
            return path
        path = os.path.dirname(path)
    return None

def norm_bench(v):
    if v is None:
        return []
    if isinstance(v, str):
        s = v.strip()
        return [s] if s else []
    return [str(x).strip() for x in v if str(x).strip()]

def main() -> None:
    if len(sys.argv) != 4:
        print(
            "usage: compute_nested_bench_subdir.py <user_config.yaml> <checkpoint_pair> <project_dir>",
            file=sys.stderr,
        )
        sys.exit(2)
    user_path, pair_line, project = sys.argv[1], sys.argv[2], sys.argv[3]
    cfg = OmegaConf.load(user_path)
    user_bench = norm_bench(cfg.settings.get("bench"))
    ckpt = pair_line.split(":", 1)[0].strip()
    if not user_bench or not ckpt:
        return
    path = ckpt
    if not path.startswith("/"):
        path = os.path.join(project, path)
    ts = find_timestamp_dir(path)
    if not ts:
        return
    exp_name = os.path.basename(os.path.dirname(ts))
    ts_name = os.path.basename(ts)
    hist_path = os.path.join(
        project,
        "main_entry/cli_agent_bash_coding/results",
        exp_name,
        ts_name,
        "config.yaml",
    )
    if not os.path.isfile(hist_path):
        return
    hcfg = OmegaConf.load(hist_path)
    hist_bench = norm_bench(hcfg.settings.get("bench"))
    su, sh = set(user_bench), set(hist_bench)
    if su & sh:
        return
    bench_tag = "_".join(user_bench)
    harness = str(OmegaConf.select(cfg, "env.bash_coding_harness")).strip().lower()
    print(f"{bench_tag}_{harness}")

if __name__ == "__main__":
    main()
