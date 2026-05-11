#!/usr/bin/env python3

import os
import sys
from omegaconf import OmegaConf

def main() -> None:
    if len(sys.argv) != 4:
        print(
            "usage: merge_checkpoint_infer_config.py <results_timestamp_config.yaml> <train_config.yaml> <out.yaml>",
            file=sys.stderr,
        )
        sys.exit(2)
    hist_path, train_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    merged = OmegaConf.load(hist_path)
    train = OmegaConf.load(train_path)
    h = str(OmegaConf.select(train, "env.bash_coding_harness", default="vanilla"))
    s = int(OmegaConf.select(train, "env.env_max_steps", default=6))
    if "env" not in merged or merged.get("env") is None:
        merged.env = OmegaConf.create({})
    merged.env.bash_coding_harness = h
    merged.env.env_max_steps = s
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(OmegaConf.to_yaml(merged))

if __name__ == "__main__":
    main()
