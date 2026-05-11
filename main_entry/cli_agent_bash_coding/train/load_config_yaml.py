import sys
import json
import shlex
from omegaconf import OmegaConf

def _as_scalar(v):
    if v is None:
        return ""
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, (int, float, str)):
        return str(v)
    if isinstance(v, list):
        return json.dumps(v)
    raise TypeError(type(v))

def _iter_flat_items(cfg):
    for k, v in cfg.items():
        if not isinstance(k, str):
            raise TypeError(type(k))
        if k == "args":
            continue
        if isinstance(v, dict):
            for kk, vv in v.items():
                if not isinstance(kk, str):
                    raise TypeError(type(kk))
                yield kk, vv
        else:
            yield k, v

def main() -> None:
    path = sys.argv[1]
    cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    flat = {}
    for k, v in _iter_flat_items(cfg):
        if k in flat:
            raise ValueError(f"Duplicate key in config: {k}")
        flat[k] = v
    for k in sorted(flat):
        print(f"{k}={shlex.quote(_as_scalar(flat[k]))}")

if __name__ == "__main__":
    main()
