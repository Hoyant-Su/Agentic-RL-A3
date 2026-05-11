import json
from dataclasses import asdict
from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path

@dataclass(frozen=True)
class MemoryEntry:
    entry_id: str
    task_description: str
    reflection: str
    trajectory: str
    utility_score: float
    count: int
    attempt_type: str
    created_at_progress: float
    successes: int

class RetroAgentMemory:
    def __init__(
        self, path: str, *, max_entries_per_task: int, utility_beta: float
    ) -> None:
        self.path = Path(path)
        self.max_entries_per_task = int(max_entries_per_task)
        self.utility_beta = float(utility_beta)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._entries = self._load()

    @property
    def entries(self) -> list[MemoryEntry]:
        return list(self._entries)

    def _load(self) -> list[MemoryEntry]:
        if not self.path.exists():
            return []
        raw_text = self.path.read_text().strip()
        if not raw_text:
            return []
        raw = json.loads(raw_text)
        if isinstance(raw, dict) and "entries" in raw:
            payload = raw["entries"]
        elif isinstance(raw, dict):
            payload = []
            for task_key, values in raw.items():
                for value in values:
                    payload.append(
                        {
                            "entry_id": sha1(
                                f"{task_key}\n{value.get('reflection', '')}".encode(
                                    "utf-8"
                                )
                            ).hexdigest(),
                            "task_description": str(task_key),
                            "reflection": str(value.get("reflection", "")),
                            "trajectory": str(value.get("response", "")),
                            "utility_score": float(
                                value.get("utility", value.get("score", 0.0))
                            ),
                            "count": int(value.get("visits", 1)),
                            "attempt_type": "success",
                            "created_at_progress": 0.0,
                            "successes": int(value.get("successes", 0)),
                        }
                    )
        else:
            payload = raw
        return [MemoryEntry(**item) for item in payload]

    def save(self) -> None:
        payload = {"entries": [asdict(entry) for entry in self._entries]}
        tmp_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
        tmp_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2))
        tmp_path.replace(self.path)

    def _prune_for_task(self, task_description: str) -> None:
        same_task = [
            entry
            for entry in self._entries
            if entry.task_description == task_description
        ]
        if len(same_task) <= self.max_entries_per_task:
            return
        same_task.sort(
            key=lambda item: (item.utility_score, item.successes, item.count),
            reverse=True,
        )
        keep_ids = {entry.entry_id for entry in same_task[: self.max_entries_per_task]}
        self._entries = [
            entry
            for entry in self._entries
            if entry.task_description != task_description or entry.entry_id in keep_ids
        ]

    def add(
        self,
        *,
        task_description: str,
        reflection_text: str,
        trajectory: str,
        initial_score: float,
        attempt_type: str,
        current_progress_ratio: float,
    ) -> None:
        entry_id = sha1(
            f"{task_description}\n{attempt_type}\n{reflection_text}".encode("utf-8")
        ).hexdigest()
        for idx, entry in enumerate(self._entries):
            if entry.entry_id != entry_id:
                continue
            self._entries[idx] = MemoryEntry(
                entry_id=entry.entry_id,
                task_description=entry.task_description,
                reflection=entry.reflection,
                trajectory=trajectory,
                utility_score=(1.0 - self.utility_beta) * entry.utility_score
                + self.utility_beta * float(initial_score),
                count=entry.count + 1,
                attempt_type=entry.attempt_type,
                created_at_progress=entry.created_at_progress,
                successes=entry.successes + int(initial_score > 0.0),
            )
            self.save()
            return
        self._entries.append(
            MemoryEntry(
                entry_id=entry_id,
                task_description=task_description,
                reflection=reflection_text,
                trajectory=trajectory,
                utility_score=float(initial_score),
                count=1,
                attempt_type=attempt_type,
                created_at_progress=float(current_progress_ratio),
                successes=int(initial_score > 0.0),
            )
        )
        self._prune_for_task(task_description)
        self.save()

    def update_utility(self, entry_id: str, score: float) -> None:
        for idx, entry in enumerate(self._entries):
            if entry.entry_id != entry_id:
                continue
            self._entries[idx] = MemoryEntry(
                entry_id=entry.entry_id,
                task_description=entry.task_description,
                reflection=entry.reflection,
                trajectory=entry.trajectory,
                utility_score=(1.0 - self.utility_beta) * entry.utility_score
                + self.utility_beta * float(score),
                count=entry.count + 1,
                attempt_type=entry.attempt_type,
                created_at_progress=entry.created_at_progress,
                successes=entry.successes + int(score > 0.0),
            )
            self.save()
            return
