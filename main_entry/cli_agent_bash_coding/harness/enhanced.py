from typing import Any
from .base import BashCodingHarnessBase

class EnhancedBashCodingHarness(BashCodingHarnessBase):
    name = "enhanced"

    def build_step_obs(
        self,
        worker,
        *,
        output_text_truncated: str,
        diff_text_truncated: str,
        trunc_meta: dict[str, Any],
    ) -> str:
        _ = worker
        _ = trunc_meta
        return output_text_truncated + "\n\n[FILE_CHANGES]\n" + diff_text_truncated
