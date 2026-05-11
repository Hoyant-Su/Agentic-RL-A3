#!/usr/bin/env python3

import os
import re
import sys
from typing import List, Optional
from omegaconf import OmegaConf

def find_timestamp_dir(path: str) -> Optional[str]:
    path = os.path.abspath(os.path.expanduser(path))
    while path not in ("/", ""):
        base = os.path.basename(path)
        if re.match(r"^[0-9]{8}_[0-9]{6}$", base):
            return path
        path = os.path.dirname(path)
    return None

def norm_bench(v) -> List[str]:
    if v is None:
        return []
    if isinstance(v, str):
        s = v.strip()
        return [s] if s else []
    return [str(x).strip() for x in v if str(x).strip()]

def _harness_token(raw: str) -> str:
    s = str(raw).strip().lower()
    s = re.sub(r"[^a-z0-9_.-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "vanilla"

def main() -> None:
    if len(sys.argv) != 4:
        print(
            "usage: compute_checkpoint_infer_results_subdir.py <train_config.yaml> <checkpoint_pair> <project_dir>",
            file=sys.stderr,
        )
        sys.exit(2)
    user_path, pair_line, project = sys.argv[1], sys.argv[2], sys.argv[3]
    ucfg = OmegaConf.load(user_path)
    user_bench = norm_bench(ucfg.settings.get("bench"))
    if not user_bench:
        return
    harness = str(OmegaConf.select(ucfg, "env.bash_coding_harness", default="vanilla"))
    step = int(OmegaConf.select(ucfg, "env.env_max_steps", default=6))
    htok = _harness_token(harness)
    ckpt = pair_line.split(":", 1)[0].strip()
    if not ckpt:
        return
    path = ckpt if ckpt.startswith("/") else os.path.join(project, ckpt)
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
    user_tag = "_".join(user_bench)
    if su == sh:
        name = f"infer_harness_{htok}_step_{step}"
    else:
        name = f"{user_tag}_harness_{htok}_step_{step}"
    print(name)

if __name__ == "__main__":
    main()
