from typing import List, Tuple
from main_entry.cli_agent_bash_coding.action_schema import AnswerAction
from main_entry.cli_agent_bash_coding.action_schema import CodeAction
from main_entry.cli_agent_bash_coding.action_schema import parse_bash_coding_action
from main_entry.cli_agent_bash_coding.cib_action_encoding import (
    ANSWER_PREFIX,
    encode_commit_if_better_action,
)

FORMAT_VIOLATION_PREFIX = "__FORMAT_VIOLATION__"

def bash_coding_projection(
    actions: List[str], enable_commit: bool
) -> Tuple[List[str], List[int]]:
    valids: List[int] = []
    processed: List[str] = []
    for action in actions:
        parsed = parse_bash_coding_action(action, enable_commit=enable_commit)
        if parsed is None:
            processed.append(FORMAT_VIOLATION_PREFIX)
            valids.append(0)
            continue
        if isinstance(parsed, AnswerAction):
            processed.append(f"{ANSWER_PREFIX}{parsed.answer}")
            valids.append(1)
            continue
        if isinstance(parsed, CodeAction):
            if enable_commit:
                if parsed.commit is None:
                    processed.append(FORMAT_VIOLATION_PREFIX)
                    valids.append(0)
                    continue
                processed.append(
                    encode_commit_if_better_action(parsed.code, parsed.commit)
                )
            else:
                processed.append(parsed.code)
            valids.append(1)
            continue
        processed.append(FORMAT_VIOLATION_PREFIX)
        valids.append(0)
    return processed, valids
