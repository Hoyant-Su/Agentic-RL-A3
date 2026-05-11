ANSWER_PREFIX = "__ANSWER__:"
CIB_CODE_PREFIX = "__CIB_CODE__:"

def encode_commit_if_better_action(code: str, commit: str) -> str:
    return f"{CIB_CODE_PREFIX}{commit}\n{code}"

def decode_commit_if_better_action(action: str) -> tuple[str, str] | None:
    if not action.startswith(CIB_CODE_PREFIX):
        return None
    body = action[len(CIB_CODE_PREFIX) :]
    split_idx = body.find("\n")
    if split_idx < 0:
        return None
    return body[:split_idx].strip(), body[split_idx + 1 :]
