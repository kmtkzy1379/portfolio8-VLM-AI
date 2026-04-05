"""Per-ID temporal feature storage."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Optional

from vlm.common.datatypes import EntityFeatures


class FeatureStore:
    """Stores historical EntityFeatures per track_id.

    Args:
        max_history: Maximum frames of history per entity.
    """

    def __init__(self, max_history: int = 100):
        self._store: dict[int, deque[EntityFeatures]] = defaultdict(
            lambda: deque(maxlen=max_history)
        )

    def store(self, features: EntityFeatures) -> None:
        self._store[features.track_id].append(features)

    def get_latest(self, track_id: int) -> Optional[EntityFeatures]:
        history = self._store.get(track_id)
        if history:
            return history[-1]
        return None

    def get_history(self, track_id: int, n: int | None = None) -> list[EntityFeatures]:
        history = self._store.get(track_id)
        if not history:
            return []
        if n is None:
            return list(history)
        return list(history)[-n:]

    def clear(self) -> None:
        self._store.clear()

    def remove(self, track_id: int) -> None:
        self._store.pop(track_id, None)
