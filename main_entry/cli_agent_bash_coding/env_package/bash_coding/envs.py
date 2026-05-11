import ray
import gym
import numpy as np
import tempfile
import os
import shutil
import hashlib
import difflib
import json
import ast
import sqlite3
import io
import tarfile
from pathlib import Path
from collections import Counter
from configparser import ConfigParser
from typing import List, Dict, Any, Tuple
from diff_match_patch import diff_match_patch
from main_entry.cli_agent_bash_coding.env_package.bash_coding.sandbox_exec import (
    SandboxError,
    ExecResult,
    run_in_unshare_sandbox,
)
from main_entry.cli_agent_bash_coding.harness import build_bash_coding_harness
from main_entry.cli_agent_bash_coding.env_package.bash_coding.projection import (
    ANSWER_PREFIX,
    FORMAT_VIOLATION_PREFIX,
)
from main_entry.cli_agent_bash_coding.reward_provenance import (
    add_reward_provenance,
    zero_reward_provenance,
)
from main_entry.cli_agent_bash_coding.tooling.semantic_similarity.semantic_similarity import (
    semantic_similarity,
)

class BashCodingWorker:

    def __init__(self, seed, env_kwargs):
        if not isinstance(env_kwargs, dict):
            env_kwargs = {} if env_kwargs is None else dict(env_kwargs)
        cfg = json.loads(os.environ.get("BASH_CODING_ENV_CONFIG_JSON", "") or "{}")
        if isinstance(cfg, dict):
            for k, v in cfg.items():
                env_kwargs.setdefault(k, v)
        self.seed = seed
        self.work_dir = None
        self.current_task = None
        self.step_count = 0
        self.max_steps = env_kwargs.get("max_steps", 10)
        self.timeout = env_kwargs.get("timeout", 10)
        self.execute_commands = env_kwargs.get("execute_commands", False)
        self.max_output_chars = int(env_kwargs.get("max_output_chars", 2000))
        self.max_diff_chars = int(env_kwargs.get("max_diff_chars", 1500))
        self.max_fs_diff_lines = int(env_kwargs.get("max_fs_diff_lines", 50))
        self.max_created_preview_lines = int(
            env_kwargs.get("max_created_preview_lines", 20)
        )
        self.unified_diff_context_lines = int(
            env_kwargs.get("unified_diff_context_lines", 2)
        )
        self.exec_backend = env_kwargs.get("exec_backend", "sandbox")
        self.harness = build_bash_coding_harness(env_kwargs)
        self.sample_kwargs: Dict[str, Any] | None = None
        self.exec_error_penalty = float(env_kwargs.get("exec_error_penalty", 0.2))
        self.answer_reward = float(env_kwargs.get("answer_reward", 1.0))
        self.progress_gain_coef = float(env_kwargs.get("progress_gain_coef", 0.3))
        self.use_model_evidence_gain = bool(
            int(env_kwargs.get("use_model_evidence_gain", 0))
        )
        self.no_progress_on_answer = bool(env_kwargs.get("no_progress_on_answer", True))
        self.require_answer_for_done = bool(
            env_kwargs.get("require_answer_for_done", True)
        )
        self.code_action_count = 0
        self.prev_progress = 0.0
        self.episode_progress_reward = 0.0
        self.init_snapshot: Dict[str, str] | None = None
        self.gold_snapshot: Dict[str, str] | None = None
        self.target_delta: Dict[str, Any] | None = None
        self.init_fs_state: Dict[str, Dict[str, Any]] | None = None
        self.gold_fs_state: Dict[str, Dict[str, Any]] | None = None
        np.random.seed(seed)

    def _truncate_text(
        self, text: str, max_chars: int, *, keep_tail: bool, label: str
    ) -> Tuple[str, Dict[str, Any]]:
        if max_chars <= 0:
            raise ValueError(f"{label} max_chars must be > 0, got {max_chars}")
        raw_chars = len(text)
        if raw_chars <= max_chars:
            return text, {
                "raw_chars": raw_chars,
                "shown_chars": raw_chars,
                "truncated": False,
            }
        clipped = text[-max_chars:] if keep_tail else text[:max_chars]
        side = "last" if keep_tail else "first"
        prefix = f"[TRUNCATED {label}: raw={raw_chars}, showing_{side}={max_chars}]\n"
        shown = prefix + clipped
        return shown, {
            "raw_chars": raw_chars,
            "shown_chars": len(shown),
            "truncated": True,
        }

    def _safe_workdir_files(self) -> List[str]:
        return sorted(os.listdir(self.work_dir))

    def _execute_action(
        self, action: str
    ) -> Tuple[str, str, ExecResult, str | None, List[str], str, Dict[str, Any]]:
        if self.work_dir is None:
            raise RuntimeError("Environment not initialized")
        track_fs = (
            self._should_track_fs_diff() or self.harness.uses_commit_action_schema()
        )
        before = self._capture_fs_state(self.work_dir) if track_fs else {}
        result = run_in_unshare_sandbox(
            work_dir=self.work_dir,
            command=action,
            timeout_s=int(self.timeout),
            extra_env={"PATH": "/usr/bin:/bin"},
        )
        if track_fs:
            after = self._capture_fs_state(self.work_dir)
            changed_files, diff_text = self._diff_fs_state(before, after)
        else:
            changed_files, diff_text = [], ""
        output_text = (result.stdout or "") + (result.stderr or "")
        exec_error = (
            None if result.returncode == 0 else f"Exit code: {result.returncode}"
        )
        output_text_truncated, output_meta = self._truncate_text(
            output_text, self.max_output_chars, keep_tail=True, label="stdout_stderr"
        )
        diff_text_truncated, diff_meta = self._truncate_text(
            diff_text, self.max_diff_chars, keep_tail=False, label="file_diff"
        )
        trunc_meta = {
            "output": output_meta,
            "diff": diff_meta,
            "caps": {
                "max_output_chars": self.max_output_chars,
                "max_diff_chars": self.max_diff_chars,
            },
            "returncode": int(result.returncode)
            if result.returncode is not None
            else None,
            "stdout_chars": len(result.stdout or ""),
            "stderr_chars": len(result.stderr or ""),
        }
        obs_for_agent = self.harness.build_step_obs(
            self,
            output_text_truncated=output_text_truncated,
            diff_text_truncated=diff_text_truncated,
            trunc_meta=trunc_meta,
        )
        return (
            obs_for_agent,
            output_text,
            result,
            exec_error,
            changed_files,
            diff_text,
            trunc_meta,
        )

    def _reward_type(self) -> str | None:
        spec = (
            (self.sample_kwargs or {}).get("reward_spec", {})
            if self.sample_kwargs
            else {}
        )
        return spec.get("type", None)

    def _should_track_fs_diff(self) -> bool:
        return self._reward_type() in {"files", "hybrid"}

    def step(self, action: str) -> Tuple[str, float, bool, Dict]:
        if self.work_dir is None:
            raise RuntimeError("Environment not initialized")
        action = self.harness.prepare_projected_action(self, action)
        if not action or not action.strip():
            self.step_count += 1
            provenance = zero_reward_provenance()
            reward = self._reward_total(provenance)
            done = self.step_count >= self.max_steps
            if self.require_answer_for_done and done:
                add_reward_provenance(
                    provenance,
                    self._terminal_submission_provenance("", submitted=False),
                )
                self._zero_failed_terminal_progress(provenance)
                reward = self._reward_total(provenance)
            info = {
                "won": False,
                "error": "Empty command",
                "returncode": None,
                "stdout": "",
                "stderr": "",
                "work_dir": self.work_dir,
                "changed_files": [],
                "diff": "",
                "step": self.step_count,
                "env_warnings": [],
                "reward_provenance": provenance,
                "reward_total": reward,
            }
            info.update(self.harness.build_step_info(self))
            return "Error: Empty command is not allowed.", reward, done, info
        if action.strip().startswith(FORMAT_VIOLATION_PREFIX):
            self.step_count += 1
            provenance = zero_reward_provenance()
            reward = self._reward_total(provenance)
            done = self.step_count >= self.max_steps
            if self.require_answer_for_done and done:
                add_reward_provenance(
                    provenance,
                    self._terminal_submission_provenance("", submitted=False),
                )
                self._zero_failed_terminal_progress(provenance)
                reward = self._reward_total(provenance)
            info = {
                "won": False,
                "error": "Format violation",
                "returncode": None,
                "stdout": "",
                "stderr": "",
                "work_dir": self.work_dir,
                "changed_files": [],
                "diff": "",
                "step": self.step_count,
                "env_warnings": [],
                "reward_provenance": provenance,
                "reward_total": reward,
            }
            info.update(self.harness.build_step_info(self))
            return "Error: Format violation.", reward, done, info
        if action.startswith(ANSWER_PREFIX):
            answer_text = action[len(ANSWER_PREFIX) :].strip()
            self.step_count += 1
            reward, provenance = self._calculate_reward(action, answer_text, None, [])
            done = True
            is_correct = self._check_task_completion(answer_text)
            if not is_correct:
                self._zero_failed_terminal_progress(provenance)
            reward = self._reward_total(provenance)
            obs = f"Answer provided: {answer_text}"
            info = {
                "won": is_correct,
                "error": None,
                "returncode": 0,
                "stdout": answer_text,
                "stderr": "",
                "work_dir": self.work_dir,
                "changed_files": [],
                "diff": "",
                "step": self.step_count,
                "direct_answer": True,
                "code_action_count": self.code_action_count,
                "env_warnings": [],
                "reward_provenance": provenance,
                "reward_total": reward,
            }
            info.update(self.harness.build_step_info(self))
            return obs, reward, done, info
        action = action.strip()
        self.step_count += 1
        self.code_action_count += 1
        try:
            (
                obs,
                output_text,
                exec_result,
                error,
                changed_files,
                diff_text,
                trunc_meta,
            ) = self._execute_action(action)
            reward, provenance = self._calculate_reward(
                action, output_text, error, changed_files
            )
            done = (
                self.step_count >= self.max_steps
                if self.require_answer_for_done
                else (
                    self._check_task_completion(output_text)
                    or self.step_count >= self.max_steps
                )
            )
            if (
                self.require_answer_for_done
                and self.step_count >= self.max_steps
                and not action.startswith(ANSWER_PREFIX)
            ):
                add_reward_provenance(
                    provenance,
                    self._terminal_submission_provenance("", submitted=False),
                )
                self._zero_failed_terminal_progress(provenance)
                reward = self._reward_total(provenance)
            info = {
                "won": done and self._check_task_completion(output_text)
                if not self.require_answer_for_done
                else False,
                "error": error,
                "returncode": exec_result.returncode,
                "stdout": exec_result.stdout or "",
                "stderr": exec_result.stderr or "",
                "work_dir": self.work_dir,
                "changed_files": changed_files,
                "diff": diff_text,
                "step": self.step_count,
                "obs_truncation": trunc_meta,
                "env_warnings": [],
                "reward_provenance": provenance,
                "reward_total": reward,
            }
            info.update(self.harness.build_step_info(self))
            return obs, reward, done, info
        except SandboxError as e:
            error_msg = f"SandboxError: {e}"
            obs = error_msg
            provenance = zero_reward_provenance()
            provenance["exec_error_penalty"] = -self.exec_error_penalty
            reward = self._reward_total(provenance)
            done = False
            info = {
                "won": False,
                "error": error_msg,
                "returncode": None,
                "stdout": "",
                "stderr": "",
                "work_dir": self.work_dir,
                "changed_files": [],
                "diff": "",
                "step": self.step_count,
                "sandbox_error": True,
                "env_warnings": [],
                "reward_provenance": provenance,
                "reward_total": reward,
            }
            info.update(self.harness.build_step_info(self))
            return obs, reward, done, info
        except Exception as e:
            error_msg = f"EnvError: {type(e).__name__}: {e}"
            obs = error_msg
            provenance = zero_reward_provenance()
            provenance["exec_error_penalty"] = -self.exec_error_penalty
            done = True
            if self.require_answer_for_done:
                add_reward_provenance(
                    provenance,
                    self._terminal_submission_provenance("", submitted=False),
                )
                self._zero_failed_terminal_progress(provenance)
            reward = self._reward_total(provenance)
            info = {
                "won": False,
                "error": error_msg,
                "returncode": None,
                "stdout": "",
                "stderr": "",
                "work_dir": self.work_dir,
                "changed_files": [],
                "diff": "",
                "step": self.step_count,
                "env_error": True,
                "env_warnings": [],
                "reward_provenance": provenance,
                "reward_total": reward,
            }
            info.update(self.harness.build_step_info(self))
            return obs, reward, done, info

    def reset(self, kwargs: Dict = None) -> Tuple[str, Dict]:
        self.sample_kwargs = kwargs
        if self.work_dir and os.path.exists(self.work_dir):
            shutil.rmtree(self.work_dir)
        self.work_dir = tempfile.mkdtemp(prefix="bash_coding_")
        self.current_task = kwargs["task"]
        self.step_count = 0
        self.code_action_count = 0
        self.prev_progress = 0.0
        self.episode_progress_reward = 0.0
        self.init_snapshot = None
        self.gold_snapshot = None
        self.target_delta = None
        self.init_fs_state = None
        self.gold_fs_state = None
        init_dir = (kwargs or {}).get("init_dir", None)
        if init_dir:
            init_dir = self._resolve_project_path(init_dir)
            if not os.path.isdir(init_dir):
                raise ValueError(f"init_dir must be a directory, got: {init_dir}")
            shutil.copytree(init_dir, self.work_dir, dirs_exist_ok=True)
        self.init_snapshot = self._snapshot_dir(self.work_dir)
        self.init_fs_state = self._capture_fs_state(self.work_dir)
        self._prepare_reward_cache()
        self.prev_progress = self._progress_score("")
        base_obs = (
            f"Working directory: {self.work_dir}\n"
            f"You can execute bash commands to explore the file system and complete this task."
        )
        info = {
            "won": False,
            "task": self.current_task,
            "work_dir": self.work_dir,
            "initial_files": [],
            "env_warnings": [],
        }
        obs = self.harness.build_reset_obs(self, base_obs, info)
        return obs, info

    def _resolve_project_path(self, path: str) -> str:
        path = str(path).strip()
        if not path:
            return path
        data_root = os.environ.get("BASH_CODING_DATA_ROOT", "").strip()
        if data_root and not os.path.isabs(path):
            rel = path
            if rel.startswith("main_entry/data/"):
                rel = rel[len("main_entry/data/") :]
            return str(Path(data_root).expanduser().resolve() / rel)
        if os.path.isabs(path):
            return path
        repo_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
        )
        if path.startswith("main_entry/data/"):
            return os.path.join(repo_root, path)
        return os.path.join(repo_root, "main_entry", "data", path)

    def _prepare_reward_cache(self) -> None:
        spec = (
            (self.sample_kwargs or {}).get("reward_spec", {})
            if self.sample_kwargs
            else {}
        )
        if spec.get("type") not in {"files", "hybrid"}:
            return
        gold_dir = spec.get("gold_dir", None)
        if not gold_dir:
            raise ValueError("reward_spec.type=files/hybrid requires gold_dir")
        gold_dir = self._resolve_project_path(gold_dir)
        if not os.path.isdir(gold_dir):
            raise ValueError(f"gold_dir must be a directory, got: {gold_dir}")
        self.gold_snapshot = self._snapshot_dir(gold_dir)
        if self.init_snapshot is None:
            raise RuntimeError("init_snapshot must be initialized before reward cache")
        self.gold_fs_state = self._capture_fs_state(gold_dir)
        self.target_delta = self._snapshot_delta(self.init_snapshot, self.gold_snapshot)

    def _sha256_file(self, path: str) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    def _snapshot_dir(self, root: str) -> Dict[str, str]:
        result: Dict[str, str] = {}
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in {".tmp", ".sandbox"}]
            for fn in filenames:
                if fn == ".DS_Store":
                    continue
                p = os.path.join(dirpath, fn)
                if not os.path.isfile(p) or os.path.islink(p):
                    continue
                rel = os.path.relpath(p, root)
                result[rel] = self._sha256_file(p)
        return result

    def _snapshot_delta(
        self, before: Dict[str, str], after: Dict[str, str]
    ) -> Dict[str, Any]:
        before_keys = set(before)
        after_keys = set(after)
        created = {path: after[path] for path in sorted(after_keys - before_keys)}
        deleted = sorted(before_keys - after_keys)
        modified = {
            path: after[path]
            for path in sorted(before_keys & after_keys)
            if before[path] != after[path]
        }
        return {
            "created": created,
            "modified": modified,
            "deleted": deleted,
        }

    @staticmethod
    def _reward_total(provenance: Dict[str, float]) -> float:
        return float(sum(provenance.values()))

    @staticmethod
    def _line_mode_text(lines: List[str]) -> str:
        return "".join(f"{line}\n" for line in lines)

    @staticmethod
    def _line_edit_multiset(
        before_lines: List[str] | None,
        after_lines: List[str] | None,
    ) -> Counter[Tuple[str, str]]:
        before = before_lines or []
        after = after_lines or []
        edits: Counter[Tuple[str, str]] = Counter()
        dmp = diff_match_patch()
        before_text = BashCodingWorker._line_mode_text(before)
        after_text = BashCodingWorker._line_mode_text(after)
        before_chars, after_chars, line_array = dmp.diff_linesToChars(
            before_text, after_text
        )
        diffs = dmp.diff_main(before_chars, after_chars, False)
        dmp.diff_charsToLines(diffs, line_array)
        dmp.diff_cleanupMerge(diffs)
        for op, text in diffs:
            if op == 0:
                continue
            for line in text.splitlines():
                if op < 0:
                    edits[("subline", line)] += 1
                else:
                    edits[("addline", line)] += 1
        return edits

    def _fs_edit_multiset(
        self,
        before_state: Dict[str, Dict[str, Any]],
        after_state: Dict[str, Dict[str, Any]],
    ) -> Counter[Tuple[str, str]]:
        edits: Counter[Tuple[str, str]] = Counter()
        all_paths = sorted(set(before_state) | set(after_state))
        for path in all_paths:
            before_entry = before_state.get(path)
            after_entry = after_state.get(path)
            before_sha = before_entry.get("sha") if before_entry is not None else None
            after_sha = after_entry.get("sha") if after_entry is not None else None
            if before_sha == after_sha:
                continue
            edits.update(
                self._line_edit_multiset(
                    before_entry["text"] if before_entry is not None else None,
                    after_entry["text"] if after_entry is not None else None,
                )
            )
        return edits

    @staticmethod
    def _is_sqlite_file(path: str, raw: bytes) -> bool:
        return raw.startswith(b"SQLite format 3\x00") or os.path.splitext(path)[
            1
        ].lower() in {".db", ".sqlite", ".sqlite3"}

    @staticmethod
    def _quote_sqlite_identifier(name: str) -> str:
        return '"' + name.replace('"', '""') + '"'

    @staticmethod
    def _normalize_sqlite_value(value: Any) -> Any:
        if isinstance(value, float):
            return format(value, ".17g")
        if isinstance(value, bytes):
            return {"hex": value.hex()}
        return value

    def _canonicalize_sqlite_text(self, path: str) -> List[str]:
        conn = sqlite3.connect(path)
        lines: List[str] = []
        objects = conn.execute(
            """
            SELECT type, name, tbl_name, sql
            FROM sqlite_master
            WHERE name NOT LIKE 'sqlite_%'
            ORDER BY type, name
            """
        ).fetchall()
        for obj_type, name, tbl_name, sql in objects:
            lines.append(f"{obj_type}\t{name}\t{tbl_name}")
            lines.append(f"sql\t{sql or ''}")
            if obj_type != "table":
                continue
            table_name = self._quote_sqlite_identifier(name)
            columns = [
                row[1]
                for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()
            ]
            lines.append("columns\t" + json.dumps(columns, ensure_ascii=False))
            rows = conn.execute(f"SELECT * FROM {table_name}").fetchall()
            row_texts = sorted(
                json.dumps(
                    [self._normalize_sqlite_value(value) for value in row],
                    ensure_ascii=False,
                )
                for row in rows
            )
            lines.extend(f"row\t{row_text}" for row_text in row_texts)
        conn.close()
        return lines

    def _canonicalize_file_text(
        self, path: str, raw: bytes, max_bytes: int
    ) -> Tuple[List[str], bool]:
        if self._is_sqlite_file(path, raw):
            return self._canonicalize_sqlite_text(path), False
        truncated = len(raw) > max_bytes
        if truncated:
            raw = raw[:max_bytes]
        return raw.decode("utf-8", errors="replace").splitlines(), truncated

    def _capture_fs_state(
        self, root: str, max_bytes: int = 200_000
    ) -> Dict[str, Dict[str, Any]]:
        state: Dict[str, Dict[str, Any]] = {}
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in {".tmp", ".sandbox"}]
            for fn in filenames:
                if fn == ".DS_Store":
                    continue
                p = os.path.join(dirpath, fn)
                if not os.path.isfile(p) or os.path.islink(p):
                    continue
                rel = os.path.relpath(p, root)
                sha = self._sha256_file(p)
                with open(p, "rb") as bf:
                    raw = bf.read(max_bytes + 1)
                text, truncated = self._canonicalize_file_text(p, raw, max_bytes)
                state[rel] = {"sha": sha, "text": text, "truncated": truncated}
        return state

    def _diff_fs_state(
        self, before: Dict[str, Dict[str, Any]], after: Dict[str, Dict[str, Any]]
    ) -> Tuple[List[str], str]:
        before_keys = set(before.keys())
        after_keys = set(after.keys())
        created = sorted(after_keys - before_keys)
        deleted = sorted(before_keys - after_keys)
        common = sorted(before_keys & after_keys)
        modified = [k for k in common if before[k].get("sha") != after[k].get("sha")]
        changed_files = created + deleted + modified
        if not changed_files:
            return [], "(no file changes detected)"
        lines_out: List[str] = []
        if created:
            lines_out.append("Created files:")
            for p in created:
                lines_out.append(f"- {p}")
        if deleted:
            lines_out.append("Deleted files:")
            for p in deleted:
                lines_out.append(f"- {p}")
        if modified:
            lines_out.append("Modified files:")
            for p in modified:
                lines_out.append(f"- {p}")
        diff_lines_budget = self.max_fs_diff_lines
        for p in modified:
            if diff_lines_budget <= 0:
                break
            btxt = before[p].get("text")
            atxt = after[p].get("text")
            if btxt is None or atxt is None:
                continue
            ud = list(
                difflib.unified_diff(
                    btxt,
                    atxt,
                    fromfile=f"a/{p}",
                    tofile=f"b/{p}",
                    lineterm="",
                    n=self.unified_diff_context_lines,
                )
            )
            if not ud:
                continue
            lines_out.append("")
            lines_out.append(f"Diff for {p}:")
            take = min(len(ud), diff_lines_budget)
            lines_out.extend(ud[:take])
            diff_lines_budget -= take
        for p in created:
            if diff_lines_budget <= 0:
                break
            atxt = after[p].get("text")
            if atxt is None:
                continue
            lines_out.append("")
            lines_out.append(f"Preview of {p}:")
            preview = atxt[
                : min(self.max_created_preview_lines, len(atxt), diff_lines_budget)
            ]
            lines_out.extend(preview)
            diff_lines_budget -= len(preview)
        return changed_files, "\n".join(lines_out)

    def _normalize_text(self, s: str) -> str:
        return " ".join((s or "").strip().split())

    def _normalize_text_match(self, s: str, *, ignore_case: bool = True) -> str:
        out = self._normalize_text(s)
        if ignore_case:
            out = out.lower()
        return out

    def _normalize_bool_match(self, s: str) -> bool | None:
        token = self._normalize_text_match(s, ignore_case=True)
        if token in ConfigParser.BOOLEAN_STATES:
            return bool(ConfigParser.BOOLEAN_STATES[token])
        return None

    def _parse_expected_alternatives(self, expected: Any) -> List[Any] | None:
        parsed = expected
        if isinstance(expected, str):
            try:
                parsed = json.loads(expected)
            except Exception:
                parsed = expected
        if isinstance(parsed, dict) and isinstance(parsed.get("any_of"), list):
            return list(parsed["any_of"])
        return None

    def _match_single_exact_expected(
        self, stdout: str, expected: Any, *, ignore_case: bool
    ) -> bool:
        pred_bool = self._normalize_bool_match(stdout)
        gt_bool = self._normalize_bool_match(str(expected))
        if pred_bool is not None and gt_bool is not None:
            return pred_bool == gt_bool
        pred = self._normalize_text_match(stdout, ignore_case=ignore_case)
        gt = self._normalize_text_match(str(expected), ignore_case=ignore_case)
        return pred == gt

    def _string_terminal_score(self, stdout: str) -> float:
        spec = (
            (self.sample_kwargs or {}).get("reward_spec", {})
            if self.sample_kwargs
            else {}
        )
        expected = spec.get("expected", None)
        if expected is None:
            return 0.0
        ignore_case = bool(spec.get("ignore_case", True))
        alternatives = self._parse_expected_alternatives(expected)
        if alternatives is not None:
            exact_score = float(
                any(
                    self._match_single_exact_expected(
                        stdout, item, ignore_case=ignore_case
                    )
                    for item in alternatives
                )
            )
            if exact_score == 1.0:
                return 1.0
            return max(semantic_similarity(stdout, str(item)) for item in alternatives)
        exact_score = float(
            self._match_single_exact_expected(stdout, expected, ignore_case=ignore_case)
        )
        if exact_score == 1.0:
            return 1.0
        return semantic_similarity(stdout, str(expected))

    def _is_correct_string(self, stdout: str) -> bool:
        score = self._string_terminal_score(stdout)
        return score == 1.0

    def _is_correct_files(self) -> bool:
        if self.init_snapshot is None or self.target_delta is None:
            return False
        if not self.work_dir or not os.path.isdir(self.work_dir):
            return False
        final_snapshot = self._snapshot_dir(self.work_dir)
        for path, target_sha in self.target_delta["created"].items():
            if final_snapshot.get(path) != target_sha:
                return False
        for path, target_sha in self.target_delta["modified"].items():
            if final_snapshot.get(path) != target_sha:
                return False
        for path in self.target_delta["deleted"]:
            if path in final_snapshot:
                return False
        return True

    def _files_progress_score(self) -> float:
        if self.init_snapshot is None or self.target_delta is None:
            return 0.0
        if self.init_fs_state is None or self.gold_fs_state is None:
            return 0.0
        if not self.work_dir or not os.path.isdir(self.work_dir):
            return 0.0
        current_fs_state = self._capture_fs_state(self.work_dir)
        gold_edits = self._fs_edit_multiset(self.init_fs_state, self.gold_fs_state)
        if not gold_edits:
            return 1.0
        model_edits = self._fs_edit_multiset(self.init_fs_state, current_fs_state)
        matched = sum((model_edits & gold_edits).values())
        total = sum(gold_edits.values())
        return float(matched) / float(total)

    def _progress_score(self, output_text: str) -> float:
        spec = (
            (self.sample_kwargs or {}).get("reward_spec", {})
            if self.sample_kwargs
            else {}
        )
        reward_type = spec.get("type", None)
        if reward_type == "files":
            return self._files_progress_score()
        if reward_type == "hybrid":
            return self._files_progress_score()
        return 0.0

    def _terminal_submission_score(self, output_text: str, submitted: bool) -> float:
        if not submitted:
            return 0.0
        spec = (
            (self.sample_kwargs or {}).get("reward_spec", {})
            if self.sample_kwargs
            else {}
        )
        reward_type = spec.get("type", None)
        if reward_type == "string":
            return self._string_terminal_score(output_text)
        if reward_type == "files":
            return float(self._is_correct_files())
        if reward_type == "hybrid":
            answer_score = self._string_terminal_score(output_text)
            file_score = float(self._is_correct_files())
            return 0.5 * (answer_score + file_score)
        return 0.0

    def _terminal_submission_provenance(
        self, output_text: str, submitted: bool
    ) -> Dict[str, float]:
        provenance = zero_reward_provenance()
        if not submitted:
            return provenance
        score = self._terminal_submission_score(output_text, submitted=True)
        provenance["answer_reward"] = self.answer_reward * score
        return provenance

    def _zero_failed_terminal_progress(self, provenance: Dict[str, float]) -> None:
        if self.episode_progress_reward == 0.0:
            return
        provenance["progress_gain_coef"] -= self.episode_progress_reward
        self.episode_progress_reward = 0.0

    def _calculate_reward(
        self,
        action: str,
        output_text: str,
        error: Any,
        changed_files: List[str],
    ) -> Tuple[float, Dict[str, float]]:
        provenance = zero_reward_provenance()
        is_answer = action.startswith(ANSWER_PREFIX)
        if error is not None:
            provenance["exec_error_penalty"] = -self.exec_error_penalty
            return self._reward_total(provenance), provenance
        if (not self.use_model_evidence_gain) and (
            not (is_answer and self.no_progress_on_answer)
        ):
            progress = self._progress_score(output_text)
            progress_delta = self.progress_gain_coef * (progress - self.prev_progress)
            provenance["progress_gain_coef"] += progress_delta
            self.episode_progress_reward += progress_delta
            self.prev_progress = progress
        if is_answer:
            add_reward_provenance(
                provenance,
                self._terminal_submission_provenance(output_text, submitted=True),
            )
        return self._reward_total(provenance), provenance

    def _pack_work_dir(self) -> bytes:
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            tar.add(self.work_dir, arcname=".")
        return buf.getvalue()

    def _restore_work_dir(self, payload: bytes) -> None:
        if self.work_dir and os.path.exists(self.work_dir):
            shutil.rmtree(self.work_dir)
        self.work_dir = tempfile.mkdtemp(prefix="bash_coding_")
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as tar:
            tar.extractall(self.work_dir)

    def export_state(self) -> Dict[str, Any]:
        return {
            "sample_kwargs": self.sample_kwargs,
            "current_task": self.current_task,
            "step_count": self.step_count,
            "code_action_count": self.code_action_count,
            "prev_progress": self.prev_progress,
            "episode_progress_reward": self.episode_progress_reward,
            "init_snapshot": self.init_snapshot,
            "gold_snapshot": self.gold_snapshot,
            "target_delta": self.target_delta,
            "init_fs_state": self.init_fs_state,
            "gold_fs_state": self.gold_fs_state,
            "work_dir_payload": self._pack_work_dir(),
        }

    def import_state(self, state: Dict[str, Any]) -> None:
        self.sample_kwargs = state["sample_kwargs"]
        self.current_task = state["current_task"]
        self.step_count = int(state["step_count"])
        self.code_action_count = int(state["code_action_count"])
        self.prev_progress = float(state["prev_progress"])
        self.episode_progress_reward = float(state.get("episode_progress_reward", 0.0))
        self.init_snapshot = state["init_snapshot"]
        self.gold_snapshot = state["gold_snapshot"]
        self.target_delta = state["target_delta"]
        self.init_fs_state = state["init_fs_state"]
        self.gold_fs_state = state["gold_fs_state"]
        self._restore_work_dir(state["work_dir_payload"])

    def _state_fingerprint(self) -> str:
        fs_state = self._capture_fs_state(self.work_dir)
        payload = {
            "fs_state": fs_state,
            "step_count": self.step_count,
            "code_action_count": self.code_action_count,
            "prev_progress": self.prev_progress,
            "episode_progress_reward": self.episode_progress_reward,
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def branch_step(
        self, state: Dict[str, Any], action: str
    ) -> Tuple[str, float, bool, Dict]:
        self.import_state(state)
        obs, reward, done, info = self.step(action)
        info["state_fingerprint"] = self._state_fingerprint()
        return obs, reward, done, info

    def _check_task_completion(self, output_text: str) -> bool:
        spec = (
            (self.sample_kwargs or {}).get("reward_spec", {})
            if self.sample_kwargs
            else {}
        )
        reward_type = spec.get("type", None)
        if reward_type == "string":
            return self._is_correct_string(output_text)
        if reward_type == "files":
            return self._is_correct_files()
        if reward_type == "hybrid":
            return self._is_correct_string(output_text) and self._is_correct_files()
        return False

    def close(self):
        if self.work_dir and os.path.exists(self.work_dir):
            shutil.rmtree(self.work_dir)

class BashCodingMultiProcessEnv(gym.Env):

    def __init__(
        self,
        seed: int,
        env_num: int,
        group_n: int,
        resources_per_worker: dict,
        is_train: bool = True,
        **env_kwargs,
    ):
        super().__init__()
        if not ray.is_initialized():
            ray.init()
        self.group_n = group_n
        self.env_num = env_num
        self.num_processes = env_num * group_n
        self.is_train = is_train
        if not is_train:
            assert group_n == 1
        self._rng = np.random.RandomState(seed)
        env_worker = ray.remote(**resources_per_worker)(BashCodingWorker)
        self._workers = []
        for i in range(self.num_processes):
            worker = env_worker.remote(seed + (i // self.group_n), env_kwargs)
            self._workers.append(worker)

    def reset(self, kwargs=None):
        if kwargs is None:
            raise ValueError("kwargs must be provided to reset()")
        if isinstance(kwargs, np.ndarray):
            kwargs_list = kwargs.tolist()
        elif isinstance(kwargs, (list, tuple)):
            kwargs_list = list(kwargs)
        else:
            raise TypeError(
                f"kwargs must be a list/tuple or numpy array, got {type(kwargs)}"
            )
        if len(kwargs_list) <= 0:
            raise ValueError("kwargs must not be empty")
        if len(kwargs_list) == self.env_num and self.group_n > 1:
            expanded_kwargs = []
            for kwarg in kwargs_list:
                for _ in range(self.group_n):
                    expanded_kwargs.append(kwarg)
            kwargs_list = expanded_kwargs
        if len(kwargs_list) > self.num_processes:
            raise ValueError(
                f"Too many kwargs: {len(kwargs_list)} > num_processes={self.num_processes}"
            )
        self._active_num_processes = len(kwargs_list)
        futures = []
        for worker, kwarg in zip(
            self._workers[: self._active_num_processes], kwargs_list
        ):
            future = worker.reset.remote(kwargs=kwarg)
            futures.append(future)
        results = ray.get(futures)
        obs_list, info_list = [], []
        for obs, info in results:
            obs_list.append(obs)
            info_list.append(info)
        return obs_list, info_list

    def step(self, actions: List[str]):
        active_n = getattr(self, "_active_num_processes", self.num_processes)
        assert len(actions) == active_n
        futures = []
        for worker, action in zip(self._workers[:active_n], actions):
            future = worker.step.remote(action)
            futures.append(future)
        results = ray.get(futures)
        obs_list, reward_list, done_list, info_list = [], [], [], []
        for obs, reward, done, info in results:
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)
        return obs_list, reward_list, done_list, info_list

    def export_states(self, indices: List[int]) -> List[Dict[str, Any]]:
        futures = [self._workers[i].export_state.remote() for i in indices]
        return ray.get(futures)

    def import_states(self, indices: List[int], states: List[Dict[str, Any]]) -> None:
        futures = [
            self._workers[i].import_state.remote(state)
            for i, state in zip(indices, states)
        ]
        ray.get(futures)

    def branch_step(
        self, indices: List[int], states: List[Dict[str, Any]], actions: List[str]
    ):
        futures = [
            self._workers[i].branch_step.remote(state, action)
            for i, state, action in zip(indices, states, actions)
        ]
        results = ray.get(futures)
        obs_list, reward_list, done_list, info_list = [], [], [], []
        for obs, reward, done, info in results:
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)
        return obs_list, reward_list, done_list, info_list

    def close(self):
        if getattr(self, "_closed", False):
            return
        close_futures = []
        for worker in self._workers:
            future = worker.close.remote()
            close_futures.append(future)
        ray.get(close_futures)
        self._closed = True

def build_bash_coding_envs(
    seed: int,
    env_num: int,
    group_n: int,
    resources_per_worker: dict,
    is_train: bool = True,
    **env_kwargs,
):
    return BashCodingMultiProcessEnv(
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        resources_per_worker=resources_per_worker,
        is_train=is_train,
        **env_kwargs,
    )
