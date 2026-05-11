from __future__ import annotations
from typing import Any, Dict, List, Tuple

class SimpleMemory:
    def __init__(self):
        self._data: List[List[Dict[str, Any]]] | None = None
        self.keys: List[str] | None = None
        self.batch_size = 0

    def __len__(self):
        return len(self._data or [])

    def __getitem__(self, idx):
        if self._data is None:
            raise RuntimeError("Memory not initialized")
        return self._data[idx]

    def reset(self, batch_size: int):
        self._data = [[] for _ in range(batch_size)]
        self.batch_size = batch_size
        self.keys = None

    def store(self, record: Dict[str, List[Any]]):
        if self._data is None:
            raise RuntimeError("Memory not initialized")
        if self.keys is None:
            self.keys = list(record.keys())
        assert self.keys == list(record.keys())
        for env_idx in range(self.batch_size):
            self._data[env_idx].append({k: record[k][env_idx] for k in self.keys})

    def update_last(self, updates: Dict[str, List[Any]]):
        if self._data is None:
            raise RuntimeError("Memory not initialized")
        for env_idx in range(self.batch_size):
            if not self._data[env_idx]:
                continue
            for key, values in updates.items():
                self._data[env_idx][-1][key] = values[env_idx]

    def fetch(
        self,
        history_length: int,
        obs_key: str = "text_obs",
        action_key: str = "action",
    ) -> Tuple[List[str], List[int]]:
        if self._data is None:
            raise RuntimeError("Memory not initialized")
        memory_contexts: List[str] = []
        valid_lengths: List[int] = []
        for env_idx in range(self.batch_size):
            recent = self._data[env_idx][-history_length:]
            valid_len = len(recent)
            start_idx = len(self._data[env_idx]) - valid_len
            blocks: List[str] = []
            for j, rec in enumerate(recent):
                step_num = start_idx + j + 1
                act = rec[action_key]
                obs = rec[obs_key]
                blocks.append(f"<STEP>{step_num}\n{act}\n<OBS>\n{obs}")
            memory_contexts.append("\n\n".join(blocks))
            valid_lengths.append(valid_len)
        return memory_contexts, valid_lengths
