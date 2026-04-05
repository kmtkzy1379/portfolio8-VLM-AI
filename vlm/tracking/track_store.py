"""Persistent track state storage for historical lookups."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Optional

from vlm.common.datatypes import TrackedEntity


class TrackStore:
    """Stores historical track states per entity.

    Keeps a sliding window of past states for each track_id,
    useful for motion computation and re-identification.

    Args:
        max_history: Maximum number of historical states per entity.
    """

    def __init__(self, max_history: int = 100):
        self._max_history = max_history
        self._history: dict[int, deque[TrackedEntity]] = defaultdict(
            lambda: deque(maxlen=max_history)
        )

    def store(self, entity: TrackedEntity) -> None:
        self._history[entity.track_id].append(entity)

    def get_latest(self, track_id: int) -> Optional[TrackedEntity]:
        history = self._history.get(track_id)
        if history:
            return history[-1]
        return None

    def get_history(
        self, track_id: int, n: int | None = None
    ) -> list[TrackedEntity]:
        history = self._history.get(track_id)
        if not history:
            return []
        if n is None:
            return list(history)
        return list(history)[-n:]

    def get_all_ids(self) -> list[int]:
        return list(self._history.keys())

    def clear(self) -> None:
        self._history.clear()

    def remove(self, track_id: int) -> None:
        self._history.pop(track_id, None)
