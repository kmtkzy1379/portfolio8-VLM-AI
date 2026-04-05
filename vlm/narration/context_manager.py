"""Running narrative context window management."""

from __future__ import annotations

from collections import deque


class ContextManager:
    """Maintains a sliding window of past narrations.

    Provides context to the LLM so it can maintain narrative continuity
    across multiple calls.

    Args:
        max_entries: Maximum number of past narrations to keep.
    """

    def __init__(self, max_entries: int = 3):
        self._history: deque[str] = deque(maxlen=max_entries)

    def append(self, narration: str) -> None:
        self._history.append(narration)

    def get_context(self) -> list[str]:
        return list(self._history)

    def get_context_text(self) -> str:
        if not self._history:
            return ""
        return "\n---\n".join(self._history)

    def clear(self) -> None:
        self._history.clear()
