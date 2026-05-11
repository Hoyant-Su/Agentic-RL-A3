from main_entry.cli_agent_bash_coding.harness.enhanced import EnhancedBashCodingHarness


class VanillaBashCodingHarness(EnhancedBashCodingHarness):
    """Paper vanilla: same step observation as legacy ``enhanced`` (FILE_CHANGES appended)."""

    name = "vanilla"
