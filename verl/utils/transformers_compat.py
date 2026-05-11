
import importlib.metadata
from functools import lru_cache
from typing import Optional
from packaging import version

try:
    from transformers.modeling_flash_attention_utils import (
        flash_attn_supports_top_left_mask,
    )
except ImportError:

    def flash_attn_supports_top_left_mask():
        return False

@lru_cache
def is_transformers_version_in_range(
    min_version: Optional[str] = None, max_version: Optional[str] = None
) -> bool:
    try:
        transformers_version_str = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError as e:
        raise ModuleNotFoundError("The `transformers` package is not installed.") from e
    transformers_version = version.parse(transformers_version_str)
    lower_bound_check = True
    if min_version is not None:
        lower_bound_check = version.parse(min_version) <= transformers_version
    upper_bound_check = True
    if max_version is not None:
        upper_bound_check = transformers_version <= version.parse(max_version)
    return lower_bound_check and upper_bound_check
