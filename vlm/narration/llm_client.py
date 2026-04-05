"""LLM API client using litellm for multi-provider support."""

from __future__ import annotations

import logging
import time

import numpy as np

from vlm.aggregation.delta_encoder import DeltaEncoder
from vlm.common.datatypes import FrameDelta
from vlm.narration.context_manager import ContextManager
from vlm.narration.prompt_builder import PromptBuilder

logger = logging.getLogger(__name__)


class NarrationEngine:
    """Feeds aggregated features to LLM for understanding.

    Uses litellm for multi-provider support (Claude, GPT, Gemini, local).

    Args:
        model: Model identifier (e.g. "claude-sonnet-4-20250514", "gpt-4o").
        fallback_model: Fallback model if primary fails.
        delta_encoder: DeltaEncoder for formatting.
        min_interval: Minimum seconds between LLM calls.
        max_context_entries: Number of past narrations to include.
        max_crops: Maximum crop images to include.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        fallback_model: str | None = "gpt-4o",
        delta_encoder: DeltaEncoder | None = None,
        min_interval: float = 5.0,
        max_context_entries: int = 3,
        max_crops: int = 2,
        jpeg_quality: int = 70,
        screenshot_max_dim: int = 960,
        screenshot_jpeg_quality: int = 50,
    ):
        self._model = model
        self._fallback = fallback_model
        self._min_interval = min_interval
        self._max_crops = max_crops
        self._last_call_time = 0.0
        self._screenshot_max_dim = screenshot_max_dim
        self._screenshot_jpeg_quality = screenshot_jpeg_quality

        self._context = ContextManager(max_entries=max_context_entries)
        self._prompt_builder = PromptBuilder(
            delta_encoder=delta_encoder,
            jpeg_quality=jpeg_quality,
            screenshot_max_dim=screenshot_max_dim,
            screenshot_jpeg_quality=screenshot_jpeg_quality,
        ) if delta_encoder else None

    def set_delta_encoder(self, encoder: DeltaEncoder) -> None:
        self._prompt_builder = PromptBuilder(
            delta_encoder=encoder,
            screenshot_max_dim=self._screenshot_max_dim,
            screenshot_jpeg_quality=self._screenshot_jpeg_quality,
        )

    def narrate(
        self,
        delta: FrameDelta | list[FrameDelta],
        key_crops: list[tuple[int, np.ndarray]] | None = None,
        relations_text: str = "",
        memory_text: str = "",
        screenshot: np.ndarray | None = None,
    ) -> str | None:
        """Generate narration for frame delta(s).

        Respects rate limiting. Returns None if called too soon.

        Args:
            delta: Current frame delta, or list of accumulated deltas.
            key_crops: Selected entity crop images.
            relations_text: Scene graph spatial relation text.
            memory_text: Working memory episodic text.
            screenshot: Full screen image for LLM vision.
        """
        now = time.monotonic()
        if now - self._last_call_time < self._min_interval:
            logger.debug("Rate limited, skipping LLM call")
            return None

        if self._prompt_builder is None:
            logger.warning("No delta encoder set, cannot build prompt")
            return None

        # Build prompt
        context_text = self._context.get_context_text()
        crops = key_crops[:self._max_crops] if key_crops else None
        messages = self._prompt_builder.build(
            delta, context_text, crops,
            relations_text=relations_text,
            memory_text=memory_text,
            screenshot=screenshot,
        )

        # Call LLM
        narration = self._call_llm(messages)
        if narration:
            self._context.append(narration)
            self._last_call_time = now

        return narration

    def _call_llm(self, messages: list[dict]) -> str | None:
        """Call LLM via litellm with fallback."""
        try:
            import litellm

            response = litellm.completion(
                model=self._model,
                messages=messages,
                max_tokens=1000,
                temperature=0.3,
            )
            content = response.choices[0].message.content
            logger.info("LLM narration generated (%d chars)", len(content))
            return content

        except Exception as e:
            logger.warning("Primary LLM failed (%s): %s", self._model, e)

            if self._fallback:
                try:
                    import litellm

                    response = litellm.completion(
                        model=self._fallback,
                        messages=messages,
                        max_tokens=1000,
                        temperature=0.3,
                    )
                    content = response.choices[0].message.content
                    logger.info("Fallback LLM narration (%d chars)", len(content))
                    return content
                except Exception as e2:
                    logger.error("Fallback LLM also failed: %s", e2)

        return None

    def clear_context(self) -> None:
        self._context.clear()
