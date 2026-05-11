from main_entry.cli_agent_bash_coding.harness.enhanced import EnhancedBashCodingHarness
from main_entry.cli_agent_bash_coding.harness.sigma_reveal_rd import SigmaRevealRDBashCodingHarness
from main_entry.cli_agent_bash_coding.harness.vanilla import VanillaBashCodingHarness

HARNESS_REGISTRY = {
    "vanilla": VanillaBashCodingHarness,
    "enhanced": EnhancedBashCodingHarness,
    "sigma_reveal_rd": SigmaRevealRDBashCodingHarness,
}
# Legacy key "enhanced" still resolves; prefer "vanilla" (paper baseline naming).


def build_bash_coding_harness(env_kwargs) -> object:
    harness_name = str(env_kwargs["bash_coding_harness"]).strip().lower()
    if harness_name not in HARNESS_REGISTRY:
        supported = ", ".join(sorted(HARNESS_REGISTRY))
        raise ValueError(
            f"Unsupported bash_coding_harness: {harness_name}. Supported: {supported}"
        )
    return HARNESS_REGISTRY[harness_name]()
