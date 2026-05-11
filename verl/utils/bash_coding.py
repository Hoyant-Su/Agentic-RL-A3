import os

BASH_CODING_TAG_TOKENS = ["<STEP>", "<OBS>", "[FILE_CHANGES]"]

def is_bash_coding_enabled() -> bool:
    return str(os.environ.get("BASH_CODING_ENABLE", "0")).strip() == "1"
