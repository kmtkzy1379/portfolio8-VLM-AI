"""Working Memory - inspired by the brain's hippocampus.

The hippocampus maintains episodic memory and object permanence:
- Objects that disappear temporarily are still "remembered"
- When a new object appears, it's compared against remembered objects
- Important events are logged as episodes for narrative continuity

This module provides:
1. Object Permanence: Remember lost entities' appearance for Re-ID
2. Re-Identification: Match new entities against remembered ones
3. Episodic Memory: Log significant events for LLM context
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from vlm.common.datatypes import TrackedEntity

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class EntityMemory:
    """Remembered information about a lost entity."""
    track_id: int
    class_name: str
    last_bbox_center: tuple[float, float]
    appearance_hist: np.ndarray   # Color histogram for Re-ID
    lost_at_frame: int
    lost_at_time: float           # monotonic seconds
    frames_alive: int


@dataclass(slots=True)
class EpisodeEvent:
    """A significant event in the episodic memory."""
    frame_id: int
    timestamp: float
    event_type: str      # "appear", "disappear", "reidentified", "scene_cut", "interaction"
    entity_ids: list[int]
    description: str


@dataclass(slots=True)
class ReIDMatch:
    """Result of re-identification matching."""
    new_track_id: int
    old_track_id: int
    similarity: float    # 0.0-1.0


class WorkingMemory:
    """Maintains object permanence and episodic memory.

    Args:
        memory_duration: Max seconds to remember a lost entity.
        max_remembered: Maximum number of lost entities to remember.
        reid_threshold: Minimum similarity for re-identification.
        max_episodes: Maximum episodic memory entries.
        hist_bins: Color histogram bins per channel for appearance.
    """

    def __init__(
        self,
        memory_duration: float = 30.0,
        max_remembered: int = 50,
        reid_threshold: float = 0.6,
        max_episodes: int = 100,
        hist_bins: int = 32,
    ):
        self._duration = memory_duration
        self._max_remembered = max_remembered
        self._reid_thresh = reid_threshold
        self._hist_bins = hist_bins

        # Object permanence: lost entities keyed by old track_id
        self._remembered: dict[int, EntityMemory] = {}

        # Episodic memory
        self._episodes: deque[EpisodeEvent] = deque(maxlen=max_episodes)

        # Re-ID mapping: new_id -> old_id
        self._reid_map: dict[int, int] = {}

    def on_entity_lost(
        self, entity: TrackedEntity, frame_id: int
    ) -> None:
        """Called when an entity is lost. Stores its appearance for Re-ID."""
        if entity.crop is None or entity.crop.size == 0:
            return

        hist = self._compute_histogram(entity.crop)
        memory = EntityMemory(
            track_id=entity.track_id,
            class_name=entity.class_name,
            last_bbox_center=entity.bbox.center,
            appearance_hist=hist,
            lost_at_frame=frame_id,
            lost_at_time=time.monotonic(),
            frames_alive=entity.frames_alive,
        )
        self._remembered[entity.track_id] = memory

        # Prune if over limit (remove oldest)
        if len(self._remembered) > self._max_remembered:
            oldest_id = min(
                self._remembered, key=lambda k: self._remembered[k].lost_at_time
            )
            del self._remembered[oldest_id]

        self._add_episode(
            frame_id, "disappear", [entity.track_id],
            f"E{entity.track_id}[{entity.class_name}] disappeared",
        )
        logger.debug("Remembered E%d for Re-ID", entity.track_id)

    def on_entity_new(
        self, entity: TrackedEntity, frame_id: int
    ) -> Optional[ReIDMatch]:
        """Called when a new entity appears. Attempts Re-ID against memory.

        Returns:
            ReIDMatch if a match is found, None otherwise.
        """
        if entity.crop is None or entity.crop.size == 0:
            self._add_episode(
                frame_id, "appear", [entity.track_id],
                f"E{entity.track_id}[{entity.class_name}] appeared",
            )
            return None

        # Expire old memories
        self._expire_memories()

        hist = self._compute_histogram(entity.crop)
        best_match: Optional[ReIDMatch] = None
        best_sim = 0.0

        for mem_id, mem in self._remembered.items():
            # Only match same class
            if mem.class_name != entity.class_name:
                continue

            sim = self._compare_histograms(hist, mem.appearance_hist)
            if sim > self._reid_thresh and sim > best_sim:
                best_sim = sim
                best_match = ReIDMatch(
                    new_track_id=entity.track_id,
                    old_track_id=mem_id,
                    similarity=round(sim, 3),
                )

        if best_match:
            self._reid_map[best_match.new_track_id] = best_match.old_track_id
            del self._remembered[best_match.old_track_id]
            self._add_episode(
                frame_id, "reidentified",
                [best_match.new_track_id, best_match.old_track_id],
                f"E{best_match.new_track_id} is likely former "
                f"E{best_match.old_track_id} (sim={best_match.similarity})",
            )
            logger.info(
                "Re-ID: E%d = former E%d (sim=%.3f)",
                best_match.new_track_id,
                best_match.old_track_id,
                best_match.similarity,
            )
        else:
            self._add_episode(
                frame_id, "appear", [entity.track_id],
                f"E{entity.track_id}[{entity.class_name}] appeared (new)",
            )

        return best_match

    def get_reid_mapping(self, track_id: int) -> Optional[int]:
        """Get old track_id for a re-identified entity."""
        return self._reid_map.get(track_id)

    def get_recent_episodes(self, n: int = 5) -> list[EpisodeEvent]:
        """Get the N most recent episodic memory entries."""
        return list(self._episodes)[-n:]

    def get_episodes_text(self, n: int = 5) -> str:
        """Format recent episodes as compact text for LLM."""
        episodes = self.get_recent_episodes(n)
        if not episodes:
            return ""
        lines = [f"  f{ep.frame_id}: {ep.description}" for ep in episodes]
        return "MEMORY:\n" + "\n".join(lines)

    def add_custom_episode(
        self, frame_id: int, event_type: str,
        entity_ids: list[int], description: str,
    ) -> None:
        """Add a custom event to episodic memory."""
        self._add_episode(frame_id, event_type, entity_ids, description)

    def reset(self) -> None:
        """Clear all memory (e.g. on scene cut)."""
        self._remembered.clear()
        self._reid_map.clear()
        # Keep episodes - they're narrative history
        self._add_episode(0, "scene_cut", [], "Scene cut - memory reset")

    @property
    def remembered_count(self) -> int:
        return len(self._remembered)

    @property
    def episode_count(self) -> int:
        return len(self._episodes)

    def _compute_histogram(self, crop: np.ndarray) -> np.ndarray:
        """Compute normalized color histogram for appearance matching."""
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        # Hue and Saturation histogram (ignore Value for lighting robustness)
        hist = cv2.calcHist(
            [hsv], [0, 1], None,
            [self._hist_bins, self._hist_bins],
            [0, 180, 0, 256],
        )
        cv2.normalize(hist, hist)
        return hist.flatten()

    @staticmethod
    def _compare_histograms(hist_a: np.ndarray, hist_b: np.ndarray) -> float:
        """Compare two histograms using correlation (Pearson)."""
        result = cv2.compareHist(
            hist_a.astype(np.float32).reshape(-1, 1),
            hist_b.astype(np.float32).reshape(-1, 1),
            cv2.HISTCMP_CORREL,
        )
        # Correlation ranges from -1 to 1; normalize to 0-1
        return max(0.0, (result + 1.0) / 2.0)

    def _expire_memories(self) -> None:
        """Remove memories older than duration."""
        now = time.monotonic()
        expired = [
            mid for mid, mem in self._remembered.items()
            if now - mem.lost_at_time > self._duration
        ]
        for mid in expired:
            del self._remembered[mid]

    def _add_episode(
        self, frame_id: int, event_type: str,
        entity_ids: list[int], description: str,
    ) -> None:
        self._episodes.append(EpisodeEvent(
            frame_id=frame_id,
            timestamp=time.monotonic(),
            event_type=event_type,
            entity_ids=entity_ids,
            description=description,
        ))
