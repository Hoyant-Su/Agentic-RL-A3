
from __future__ import annotations
from tree_sitter import Language, Parser
import tree_sitter_bash as tsb

class BashIntentActionSpace:
    _LITERAL_TYPES = frozenset({"word", "string_content", "variable_name", "number"})
    _STRUCT_KINDS = frozenset(
        {
            "pipeline",
            "subshell",
            "compound_statement",
            "list",
            "function_definition",
            "do_group",
            "negated_command",
            "case_item",
            "elif_clause",
            "else_clause",
        }
    )
    _NOOP_VERBS = frozenset({"true", "false", ":"})
    __slots__ = ("_parser", "_control_types")

    def __init__(self, parser: Parser | None = None) -> None:
        lang = Language(tsb.language())
        self._parser = parser if parser is not None else Parser(lang)
        self._control_types = self._control_types_from_grammar(lang)

    @staticmethod
    def _named_kinds(lang: Language) -> frozenset[str]:
        return frozenset(
            lang.node_kind_for_id(i)
            for i in range(lang.node_kind_count)
            if lang.node_kind_is_named(i)
        )

    @classmethod
    def _control_types_from_grammar(cls, lang: Language) -> frozenset[str]:
        named = cls._named_kinds(lang)
        stmt = frozenset(k for k in named if k.endswith("_statement"))
        return stmt | (cls._STRUCT_KINDS & named)

    @staticmethod
    def levenshtein(a: list[str], b: list[str]) -> int:
        la, lb = len(a), len(b)
        if la == 0:
            return lb
        if lb == 0:
            return la
        prev = list(range(lb + 1))
        for i in range(1, la + 1):
            cur = [i]
            ai = a[i - 1]
            for j in range(1, lb + 1):
                cost = 0 if ai == b[j - 1] else 1
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
            prev = cur
        return prev[lb]

    @staticmethod
    def _norm_dist(sig_a: list[str], sig_b: list[str]) -> float:
        return BashIntentActionSpace.levenshtein(sig_a, sig_b) / max(
            len(sig_a), len(sig_b), 1
        )

    @staticmethod
    def _node_text(node, source: bytes) -> str:
        return source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")

    @staticmethod
    def _normalize_surface(s: str) -> str:
        s = s.strip().strip("'\"")
        return s.lower()[:96]

    @classmethod
    def _canon_verb(cls, v: str) -> str:
        v = v.strip()
        if v in cls._NOOP_VERBS or v == "":
            return ":noop"
        return v

    @classmethod
    def _verb_from_command(cls, cmd, source: bytes) -> str:
        for ch in cmd.named_children:
            if ch.type == "command_name" and ch.named_child_count > 0:
                w = ch.named_children[0]
                if w.type in ("word", "string_content"):
                    return cls._canon_verb(
                        cls._normalize_surface(cls._node_text(w, source))
                    )
        return "?"

    @classmethod
    def _literals_shallow(cls, cmd, source: bytes) -> list[str]:
        verb_word = None
        for ch in cmd.named_children:
            if ch.type == "command_name" and ch.named_child_count > 0:
                w = ch.named_children[0]
                if w.type in cls._LITERAL_TYPES:
                    verb_word = w
                break
        out: list[str] = []

        def inner(node, inside_arith: bool) -> None:
            if node.type == "command" and node is not cmd:
                return
            if node.type == "arithmetic_expansion":
                for c in node.named_children:
                    inner(c, True)
                return
            if node.type in cls._LITERAL_TYPES:
                if not inside_arith:
                    if (
                        verb_word is None
                        or node.start_byte != verb_word.start_byte
                        or node.end_byte != verb_word.end_byte
                    ):
                        out.append(cls._normalize_surface(cls._node_text(node, source)))
                return
            for ch in node.named_children:
                inner(ch, inside_arith)

        inner(cmd, False)
        return out

    def _intent_tokens(self, root, source: bytes) -> list[str]:
        out: list[str] = []
        ct = self._control_types

        def walk(node):
            t = node.type
            if t in ct:
                out.append(f"K:{t}")
            if t == "command":
                out.append(f"V:{self._verb_from_command(node, source)}")
                for lit in self._literals_shallow(node, source):
                    out.append(f"W:{lit}")
            for ch in node.named_children:
                walk(ch)

        walk(root)
        return out

    def intent_signature(self, shell: str) -> list[str]:
        b = shell.encode("utf-8")
        t = self._parser.parse(b)
        return self._intent_tokens(t.root_node, b)

    def distance(self, shell_a: str, shell_b: str) -> float:
        return self._norm_dist(
            self.intent_signature(shell_a), self.intent_signature(shell_b)
        )

    def distance_matrix(self, shells: list[str]) -> list[list[float]]:
        n = len(shells)
        sigs = [self.intent_signature(s) for s in shells]
        mat = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                d = self._norm_dist(sigs[i], sigs[j])
                mat[i][j] = mat[j][i] = d
        return mat

if __name__ == "__main__":
    space = BashIntentActionSpace()
    x = "ls | wc -l"
    y = "ls *.txt | wc -l && echo 'done'"
    z = "echo $HOME"
    print("d(x,y)=", space.distance(x, y))
    print("d(x,z)=", space.distance(x, z))
    print("sig(x)=", space.intent_signature(x))
    shells = [x, y, z]
    labels = ["x", "y", "z"]
    m = space.distance_matrix(shells)
    w = max(len(l) for l in labels)
    cell = 8
    print(" " * (w + 1) + "".join(f"{lab:>{cell}s}" for lab in labels))
    for i, li in enumerate(labels):
        row = f"{li:>{w}s}"
        for j in range(len(labels)):
            row += f"{m[i][j]:>{cell}.4f}"
        print(row)
