"""Disk-based checkpoint zoo for LLM adapter checkpoints.

Copied from REDKWEEN (red/zoo.py). Stores adapter paths on disk rather
than in-memory state dicts (LLMs don't fit). One CheckpointZoo instance
manages either victim or adversary checkpoints.
"""

from __future__ import annotations

import os
import random
import re
from dataclasses import dataclass


@dataclass
class _Entry:
    round_num: int
    adapter_path: str


class CheckpointZoo:
    """Manages a zoo of adapter checkpoints on disk."""

    def __init__(self, max_size: int = 50):
        self.max_size = max_size
        self._entries: list[_Entry] = []

    def add(self, round_num: int, adapter_path: str) -> None:
        """Add a checkpoint to the zoo. FIFO eviction when full."""
        self._entries.append(_Entry(round_num=round_num, adapter_path=adapter_path))
        if len(self._entries) > self.max_size:
            self._entries.pop(0)

    def sample(self) -> str | None:
        """Sample a random adapter path (uniform). None if empty."""
        if not self._entries:
            return None
        return random.choice(self._entries).adapter_path

    @classmethod
    def from_checkpoints_dir(cls, path: str, role: str = "victim",
                             max_size: int = 50) -> CheckpointZoo:
        """Rebuild zoo from an existing checkpoints directory."""
        zoo = cls(max_size=max_size)
        if not os.path.isdir(path):
            return zoo

        round_dirs = []
        for entry in os.listdir(path):
            match = re.match(r"round_(\d+)$", entry)
            if match:
                round_num = int(match.group(1))
                adapter_dir = os.path.join(path, entry, role)
                adapter_file = os.path.join(adapter_dir, "adapter_model.safetensors")
                if os.path.exists(adapter_file):
                    round_dirs.append((round_num, adapter_dir))

        for round_num, adapter_dir in sorted(round_dirs):
            zoo.add(round_num, adapter_dir)

        return zoo

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        rounds = [e.round_num for e in self._entries]
        return f"CheckpointZoo(size={len(self)}, rounds={rounds})"
