"""
Git-friendly snapshot storage: "known good" outputs stored as JSON, so a
code reviewer can see exactly what changed in a pull request diff, the same
way they'd review a code change.
"""
from __future__ import annotations
import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone


@dataclass
class Snapshot:
    case_name: str
    prompt: str
    output: str
    recorded_at: str


class SnapshotStore:
    def __init__(self, path: str):
        self.path = path
        self._data: dict[str, dict] = {}
        if os.path.exists(path):
            with open(path) as f:
                self._data = json.load(f)

    def get(self, case_name: str) -> Snapshot | None:
        raw = self._data.get(case_name)
        return Snapshot(**raw) if raw else None

    def save(self, case_name: str, prompt: str, output: str) -> Snapshot:
        snap = Snapshot(
            case_name=case_name, prompt=prompt, output=output,
            recorded_at=datetime.now(timezone.utc).isoformat(),
        )
        self._data[case_name] = asdict(snap)
        self._flush()
        return snap

    def _flush(self) -> None:
        with open(self.path, "w") as f:
            json.dump(self._data, f, indent=2, sort_keys=True)

    def all_case_names(self) -> list[str]:
        return list(self._data.keys())
