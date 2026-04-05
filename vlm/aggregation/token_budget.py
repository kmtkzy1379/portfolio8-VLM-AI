"""Token counting and budget management."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class TokenCounter:
    """Estimates token count for text strings.

    Uses tiktoken for accurate counting when available,
    falls back to word-based approximation.
    """

    def __init__(self, encoding_name: str = "cl100k_base"):
        self._encoder = None
        try:
            import tiktoken
            self._encoder = tiktoken.get_encoding(encoding_name)
        except (ImportError, Exception):
            logger.warning("tiktoken not available, using word-based approximation")

    def count(self, text: str) -> int:
        if self._encoder is not None:
            return len(self._encoder.encode(text))
        # Rough approximation: ~1.3 tokens per word
        return int(len(text.split()) * 1.3)


class TokenBudgetManager:
    """Manages token budget for LLM input.

    Ensures the total output stays within budget by truncating
    lower-priority content.

    Args:
        max_tokens: Total token budget.
        scene_budget: Tokens reserved for scene context.
        history_budget: Tokens for narrative history.
        entity_budget: Tokens for entity deltas.
    """

    def __init__(
        self,
        max_tokens: int = 4000,
        scene_budget: int = 500,
        history_budget: int = 1000,
        entity_budget: int = 2500,
    ):
        self._max = max_tokens
        self._scene_budget = scene_budget
        self._history_budget = history_budget
        self._entity_budget = entity_budget
        self._counter = TokenCounter()

    def fits(self, text: str) -> bool:
        return self._counter.count(text) <= self._max

    def count(self, text: str) -> int:
        return self._counter.count(text)

    def truncate_to_budget(
        self, scene: str, entities: str, history: str
    ) -> tuple[str, str, str]:
        """Truncate components to fit within budget.

        Priority: scene (highest) > entities > history (lowest).
        """
        scene_tokens = self._counter.count(scene)
        entity_tokens = self._counter.count(entities)
        history_tokens = self._counter.count(history)

        total = scene_tokens + entity_tokens + history_tokens
        if total <= self._max:
            return scene, entities, history

        # Truncate history first
        if history_tokens > self._history_budget:
            history = self._truncate_text(history, self._history_budget)

        # Then entities if still over
        remaining = self._max - self._counter.count(scene) - self._counter.count(history)
        if self._counter.count(entities) > remaining:
            entities = self._truncate_text(entities, max(remaining, 200))

        return scene, entities, history

    def _truncate_text(self, text: str, target_tokens: int) -> str:
        """Truncate text to approximately target_tokens."""
        lines = text.split("\n")
        result = []
        current_tokens = 0
        for line in lines:
            line_tokens = self._counter.count(line)
            if current_tokens + line_tokens > target_tokens:
                result.append("... (truncated)")
                break
            result.append(line)
            current_tokens += line_tokens
        return "\n".join(result)
