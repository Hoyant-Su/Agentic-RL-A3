
from __future__ import annotations
import os
import re
from typing import Any
from main_entry.cli_agent_bash_coding.harness.enhanced import EnhancedBashCodingHarness

_BUDGET_CHARS = 2400
_BUCKET_CHARS = 16
_MAX_DEPTH = 4
_MAX_HEAD_BYTES = 160
_LAMBDA_CITE = 4.0
_LAMBDA_DEPTH = 1.0
_LAMBDA_EXT = 0.8
_ALPHA_DEPTH = 0.7
_DIR_EXT_BASE = 0.3
_DATA_EXTS = {".csv", ".tsv", ".json", ".jsonl", ".txt", ".yaml", ".yml", ".log"}
_CODE_EXTS = {
    ".py",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".sh",
    ".js",
    ".ts",
    ".go",
    ".rs",
    ".rb",
}
_DATA_CUES = ("csv", "tsv", "json", "yaml", "dataset", "data", "table", "row", "column")
_CODE_CUES = (
    "script",
    "function",
    "class",
    "module",
    "python",
    ".py",
    "compile",
    "run",
)
_TOKEN_RE = re.compile(r"[A-Za-z0-9_./-]+")

def _is_probably_text(data: bytes) -> bool:
    if b"\x00" in data:
        return False
    printable = sum(1 for b in data if 9 <= b <= 13 or 32 <= b <= 126)
    return printable / max(len(data), 1) >= 0.85

def _o0_token_set(text: str) -> set[str]:
    out: set[str] = set()
    for tok in _TOKEN_RE.findall(text):
        t = tok.strip(".")
        if not t:
            continue
        out.add(t)
        out.add(t.lower())
        stem = os.path.splitext(t)[0]
        if stem and stem != t:
            out.add(stem)
            out.add(stem.lower())
    return out

def _task_type_weights(text: str) -> tuple[float, float]:
    low = text.lower()
    data_hits = sum(1 for c in _DATA_CUES if c in low)
    code_hits = sum(1 for c in _CODE_CUES if c in low)
    total = data_hits + code_hits
    if total == 0:
        return 0.5, 0.5
    return data_hits / total, code_hits / total

def _render_file_line(
    name: str, size: int, is_text: bool, head: bytes, indent: str
) -> str:
    if not is_text:
        return f"{indent}{name} ({size} B, binary)"
    first_line = head.split(b"\n", 1)[0].decode("utf-8", "replace").strip()
    if not first_line:
        return f"{indent}{name} ({size} B, empty)"
    if len(first_line) > 80:
        first_line = first_line[:77] + "..."
    return f"{indent}{name} ({size} B) -> {first_line!r}"

def _cite_score(name: str, o0_tokens: set[str]) -> float:
    stem = os.path.splitext(name)[0]
    if name in o0_tokens or name.lower() in o0_tokens:
        return 1.0
    if stem and (stem in o0_tokens or stem.lower() in o0_tokens):
        return 1.0
    return 0.0

def _ext_score(name: str, is_dir: bool, w_data: float, w_code: float) -> float:
    if is_dir:
        return _DIR_EXT_BASE
    ext = os.path.splitext(name)[1].lower()
    if ext in _DATA_EXTS:
        return w_data
    if ext in _CODE_EXTS:
        return w_code
    return 0.0

def _mu_hat(
    name: str,
    is_dir: bool,
    depth: int,
    o0_tokens: set[str],
    w_data: float,
    w_code: float,
) -> float:
    cite = _cite_score(name, o0_tokens)
    depth_prior = _ALPHA_DEPTH**depth
    ext = _ext_score(name, is_dir, w_data, w_code)
    return _LAMBDA_CITE * cite + _LAMBDA_DEPTH * depth_prior + _LAMBDA_EXT * ext

class _Node:
    __slots__ = ("idx", "name", "depth", "is_dir", "line", "cost_b", "mu", "children")

    def __init__(
        self,
        idx: int,
        name: str,
        depth: int,
        is_dir: bool,
        line: str,
        cost_b: int,
        mu: float,
    ) -> None:
        self.idx = idx
        self.name = name
        self.depth = depth
        self.is_dir = is_dir
        self.line = line
        self.cost_b = cost_b
        self.mu = mu
        self.children: list[_Node] = []

def _build_tree(
    root_path: str,
    o0_tokens: set[str],
    w_data: float,
    w_code: float,
) -> _Node:
    counter = [0]

    def make_node(name: str, depth: int, is_dir: bool, line: str) -> _Node:
        cost_b = max(1, (len(line) + 1 + _BUCKET_CHARS - 1) // _BUCKET_CHARS)
        mu = _mu_hat(name, is_dir, depth, o0_tokens, w_data, w_code)
        n = _Node(counter[0], name, depth, is_dir, line, cost_b, mu)
        counter[0] += 1
        return n

    root = make_node(".", 0, True, ".")

    def walk(path: str, parent: _Node, depth: int) -> None:
        if depth > _MAX_DEPTH:
            return
        for child_name in sorted(os.listdir(path)):
            child_path = os.path.join(path, child_name)
            if os.path.isdir(child_path):
                indent = "  " * depth
                line = f"{indent}{child_name}/"
                node = make_node(child_name, depth, True, line)
                parent.children.append(node)
                walk(child_path, node, depth + 1)
            else:
                size = os.path.getsize(child_path)
                with open(child_path, "rb") as fh:
                    head = fh.read(_MAX_HEAD_BYTES)
                is_text = _is_probably_text(head)
                indent = "  " * depth
                line = _render_file_line(child_name, size, is_text, head, indent)
                parent.children.append(make_node(child_name, depth, False, line))

    walk(root_path, root, 1)
    return root

def _subtree_knapsack(root: _Node, budget_b: int) -> set[int]:
    NEG = float("-inf")

    def dfs(v: _Node) -> tuple[list[float], list[set[int]]]:
        dp = [NEG] * (budget_b + 1)
        pick: list[set[int]] = [set() for _ in range(budget_b + 1)]
        if v.cost_b <= budget_b:
            dp[v.cost_b] = v.mu
            pick[v.cost_b] = {v.idx}
        for child in v.children:
            c_dp, c_pick = dfs(child)
            new_dp = dp[:]
            new_pick = [s.copy() for s in pick]
            for bv in range(budget_b + 1):
                if dp[bv] == NEG:
                    continue
                remain = budget_b - bv
                for bu in range(remain + 1):
                    if c_dp[bu] == NEG:
                        continue
                    val = dp[bv] + c_dp[bu]
                    tot = bv + bu
                    if val > new_dp[tot]:
                        new_dp[tot] = val
                        new_pick[tot] = pick[bv] | c_pick[bu]
            dp = new_dp
            pick = new_pick
        return dp, pick

    dp, pick = dfs(root)
    best_b = 0
    best_v = dp[0] if dp[0] != NEG else NEG
    for b in range(budget_b + 1):
        if dp[b] > best_v:
            best_v = dp[b]
            best_b = b
    return pick[best_b]

def _render_selected(root: _Node, selected: set[int]) -> str:
    lines: list[str] = []

    def emit(node: _Node) -> None:
        if node.idx not in selected:
            return
        lines.append(node.line)
        for child in node.children:
            emit(child)

    emit(root)
    return "\n".join(lines)

def _render_reveal(root_path: str, o0_text: str) -> str:
    o0_tokens = _o0_token_set(o0_text)
    w_data, w_code = _task_type_weights(o0_text)
    tree = _build_tree(root_path, o0_tokens, w_data, w_code)
    budget_b = max(1, _BUDGET_CHARS // _BUCKET_CHARS)
    selected = _subtree_knapsack(tree, budget_b)
    body = _render_selected(tree, selected)
    n_total = 0

    def count(n: _Node) -> None:
        nonlocal n_total
        n_total += 1
        for c in n.children:
            count(c)

    count(tree)
    header = (
        f"[sigma_reveal_rd  B={_BUDGET_CHARS}ch  |T*|={len(selected)}/{n_total}"
        f"  w_data={w_data:.2f}  w_code={w_code:.2f}]"
    )
    return header + "\n" + body

_SR_HEAD = (
    "Static initial-workspace layout (one-time side-information for this "
    "episode; the shell state evolves and is reflected in subsequent "
    "observations, but this layout is not re-rendered):"
)
_SR_FOOT = "[end of static initial-workspace layout]"

class SigmaRevealRDBashCodingHarness(EnhancedBashCodingHarness):

    name = "sigma_reveal_rd"

    def build_reset_obs(self, worker, base_obs: str, info: dict[str, Any]) -> str:
        task_text = (worker.current_task or "").rstrip()
        block = _render_reveal(worker.work_dir, task_text + "\n" + base_obs)
        info["task"] = f"{task_text}\n\n{_SR_HEAD}\n{block}\n{_SR_FOOT}"
        return super().build_reset_obs(worker, base_obs, info)
