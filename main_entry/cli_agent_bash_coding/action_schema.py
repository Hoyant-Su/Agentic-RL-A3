import os
import re
from dataclasses import dataclass

@dataclass(frozen=True)
class CodeAction:
    plan: str
    code: str
    commit: str | None = None

@dataclass(frozen=True)
class AnswerAction:
    plan: str
    answer: str

@dataclass(frozen=True)
class ActionSpans:
    plan: tuple[int, int]
    payload: tuple[int, int]
    commit: tuple[int, int] | None = None

def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name, "").strip()
    if value == "":
        return default
    return int(value)

REGEX_PLAN_LEN = _env_int("BASH_CODING_REGEX_PLAN_LEN", 512)
REGEX_CODE_LEN = _env_int("BASH_CODING_REGEX_CODE_LEN", 256)
REGEX_ANSWER_LEN = _env_int("BASH_CODING_REGEX_ANSWER_LEN", 128)
COMMIT_LABELS = ("keep", "rollback")
COMMIT_ACTION_REGEX = r"(?:keep|rollback)"
HARMFUL_JUDGE_LABELS = ("OK", "HARMFUL")
HARMFUL_JUDGE_ACTION_REGEX = r"(?:OK|HARMFUL)"
_CODE_ACTION_REGEX = rf"<name>submit_code</name>\s*<plan>[\s\S]{{1,{REGEX_PLAN_LEN}}}</plan>\s*<code>[\s\S]{{1,{REGEX_CODE_LEN}}}</code>"
_CODE_ACTION_WITH_COMMIT_REGEX = (
    rf"<name>submit_code</name>\s*<commit>{COMMIT_ACTION_REGEX}</commit>\s*<plan>[\s\S]{{1,{REGEX_PLAN_LEN}}}</plan>"
    rf"\s*<code>[\s\S]{{1,{REGEX_CODE_LEN}}}</code>"
)
_ANSWER_ACTION_REGEX = rf"<name>submit_answer</name>\s*<plan>[\s\S]{{1,{REGEX_PLAN_LEN}}}</plan>\s*<answer>[\s\S]{{1,{REGEX_ANSWER_LEN}}}</answer>"
BASH_CODING_ACTION_REGEX = rf"(?:{_CODE_ACTION_REGEX}|{_ANSWER_ACTION_REGEX})"
BASH_CODING_ACTION_WITH_COMMIT_REGEX = (
    rf"(?:{_CODE_ACTION_WITH_COMMIT_REGEX}|{_ANSWER_ACTION_REGEX})"
)
_CODE_ACTION_PATTERN = re.compile(
    rf"^\s*<name>submit_code</name>\s*<plan>([\s\S]{{1,{REGEX_PLAN_LEN}}}?)</plan>\s*<code>([\s\S]{{1,{REGEX_CODE_LEN}}}?)</code>\s*$"
)
_CODE_ACTION_WITH_COMMIT_PATTERN = re.compile(
    rf"^\s*<name>submit_code</name>\s*<commit>({COMMIT_ACTION_REGEX})</commit>\s*<plan>([\s\S]{{1,{REGEX_PLAN_LEN}}}?)</plan>\s*<code>([\s\S]{{1,{REGEX_CODE_LEN}}}?)</code>\s*$"
)
_ANSWER_ACTION_PATTERN = re.compile(
    rf"^\s*<name>submit_answer</name>\s*<plan>([\s\S]{{1,{REGEX_PLAN_LEN}}}?)</plan>\s*<answer>([\s\S]{{1,{REGEX_ANSWER_LEN}}}?)</answer>\s*$"
)
_HARMFUL_JUDGE_PATTERN = re.compile(rf"^\s*({HARMFUL_JUDGE_ACTION_REGEX})\s*$")

def uses_commit_action_schema(harness_name: str) -> bool:
    return str(harness_name).strip().lower() == "commit_if_better"

def get_bash_coding_action_regex(enable_commit: bool) -> str:
    if enable_commit:
        return BASH_CODING_ACTION_WITH_COMMIT_REGEX
    return BASH_CODING_ACTION_REGEX

def parse_bash_coding_action(
    text: str, enable_commit: bool
) -> CodeAction | AnswerAction | None:
    if enable_commit:
        code_match = _CODE_ACTION_WITH_COMMIT_PATTERN.fullmatch(text)
        if code_match is not None:
            return CodeAction(
                plan=code_match.group(2),
                code=code_match.group(3),
                commit=code_match.group(1),
            )
    else:
        code_match = _CODE_ACTION_PATTERN.fullmatch(text)
        if code_match is not None:
            return CodeAction(plan=code_match.group(1), code=code_match.group(2))
    answer_match = _ANSWER_ACTION_PATTERN.fullmatch(text)
    if answer_match is not None:
        return AnswerAction(plan=answer_match.group(1), answer=answer_match.group(2))
    return None

def get_bash_coding_action_spans(text: str, enable_commit: bool) -> ActionSpans | None:
    if enable_commit:
        code_match = _CODE_ACTION_WITH_COMMIT_PATTERN.fullmatch(text)
        if code_match is not None:
            return ActionSpans(
                plan=code_match.span(2),
                payload=code_match.span(3),
                commit=code_match.span(1),
            )
    else:
        code_match = _CODE_ACTION_PATTERN.fullmatch(text)
        if code_match is not None:
            return ActionSpans(
                plan=code_match.span(1),
                payload=code_match.span(2),
            )
    answer_match = _ANSWER_ACTION_PATTERN.fullmatch(text)
    if answer_match is not None:
        return ActionSpans(
            plan=answer_match.span(1),
            payload=answer_match.span(2),
        )
    return None

def extract_plan_from_bash_coding_action(text: str, enable_commit: bool) -> str | None:
    parsed = parse_bash_coding_action(text, enable_commit=enable_commit)
    return None if parsed is None else parsed.plan

def extract_answer_from_bash_coding_action(
    text: str, enable_commit: bool
) -> str | None:
    parsed = parse_bash_coding_action(text, enable_commit=enable_commit)
    return parsed.answer if isinstance(parsed, AnswerAction) else None

def parse_harmful_judge_action(text: str) -> str | None:
    match = _HARMFUL_JUDGE_PATTERN.fullmatch(text)
    if match is None:
        return None
    return match.group(1)
