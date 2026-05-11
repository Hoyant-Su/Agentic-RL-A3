#!/usr/bin/env python3

import shlex
import sys
from omegaconf import OmegaConf

def _quote(v) -> str:
    if v is None:
        return "''"
    if isinstance(v, bool):
        return "'1'" if v else "'0'"
    if isinstance(v, (int, float)):
        return shlex.quote(str(v))
    if isinstance(v, str):
        return shlex.quote(v)
    import json

    return shlex.quote(json.dumps(v, separators=(",", ":")))

def main() -> None:
    if len(sys.argv) != 2:
        print(
            "usage: print_train_checkpoint_infer_overrides.py <train_config.yaml>",
            file=sys.stderr,
        )
        sys.exit(2)
    cfg = OmegaConf.to_container(OmegaConf.load(sys.argv[1]), resolve=True)
    bench = cfg.get("settings", {}).get("bench")
    env = cfg.get("env", {})
    harness = env.get("bash_coding_harness", "vanilla")
    steps = env.get("env_max_steps", 6)
    print(f"bench={_quote(bench)}")
    print(f"bash_coding_harness={_quote(harness)}")
    print(f"env_max_steps={_quote(steps)}")

if __name__ == "__main__":
    main()
